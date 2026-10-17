"""Deterministic backpressure evidence with synthetic, in-memory WebSockets."""

import asyncio
import threading

import pytest
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from companion.bodylink.transport import Endpoint, NotConnectedError, Session
from companion.contracts.body import (
    AudioFrame,
    BodyHeartbeat,
    ButtonPressed,
    Hello,
    encode_audio_frame,
    encode_message,
    parse_body_message,
)


class GatedWebSocket:
    def __init__(self, session_id: str, *, blocked: bool = True) -> None:
        self.session_id = session_id
        self.gate = asyncio.Event()
        if not blocked:
            self.gate.set()
        self.entered = asyncio.Event()
        self.thread_entered = threading.Event()
        self.delivered: list[str | bytes] = []
        self.cancelled = 0
        self.received = 0
        self.peer_closed = False
        self.close_codes: list[int] = []

    async def send(self, data: str | bytes) -> None:
        self.entered.set()
        self.thread_entered.set()
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.delivered.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        await asyncio.sleep(0.005)
        if self.peer_closed:
            raise StopAsyncIteration
        self.received += 1
        return encode_message(
            BodyHeartbeat(protocol_version=1, seq=self.received, session_id=self.session_id)
        )

    async def close(self, code: int) -> None:
        self.close_codes.append(code)


class TestEndpoint(Endpoint):
    __test__ = False
    _thread_name = "bodylink-backpressure-test"
    _incoming_direction = "uplink"
    _parse = staticmethod(parse_body_message)
    _heartbeat_type = BodyHeartbeat
    _handshake_type = Hello

    def __init__(self) -> None:
        super().__init__(
            on_session_ended=self.record_end,
            on_message=lambda message: None,
            on_frame=lambda frame: None,
        )
        self.ended: list[tuple[str, str]] = []
        self.ready = threading.Event()

    def record_end(self, session_id: str, reason: str) -> None:
        assert not self.session.open
        self.ended.append((session_id, reason))

    def begin(
        self, ws: GatedWebSocket, *, timeout_s: float = 0.08, interval_s: float = 10
    ) -> asyncio.Task[None]:
        self._stop = asyncio.Event()
        self._seq = 0
        self.session = Session(
            ws=ws, session_id=ws.session_id, interval_s=interval_s,
            timeout_s=timeout_s, last_rx=asyncio.get_running_loop().time(),
        )
        self._session = self.session
        self.running = asyncio.create_task(self._run_session(self.session))
        return self.running

    async def finish(self) -> None:
        self._stop.set()
        await asyncio.wait_for(self.running, 1)

    async def send_path(self, path: str, session_id: str | None = None) -> None:
        session_id = self.session.session_id if session_id is None else session_id
        if path == "control":
            await self._send_message(control(session_id))
        else:
            await self._send_frame(session_id, audio())

    def start(self, ws: GatedWebSocket) -> None:
        self.thread_ws = ws
        self._start_thread()
        assert self.ready.wait(1)

    async def _main(self) -> None:
        running = self.begin(self.thread_ws, timeout_s=10)
        self.ready.set()
        await running

    def release_gate(self) -> None:
        self._loop.call_soon_threadsafe(self.thread_ws.gate.set)

    def stop_now(self) -> None:
        self._stop.set()


def control(session_id: str) -> ButtonPressed:
    return ButtonPressed(protocol_version=1, seq=0, session_id=session_id, observed_at_ms=7)


def audio() -> AudioFrame:
    return AudioFrame("uplink", 3, 0, True, b"\x01\x00" * 16)


def assert_timeout(error: NotConnectedError) -> None:
    assert str(error) == "session send timed out"
    assert error.__cause__ is None and error.__suppress_context__


def test_blocked_send_times_out_with_incoming_heartbeats() -> None:
    async def scenario() -> None:
        before = asyncio.all_tasks()
        endpoint = TestEndpoint()
        ws = GatedWebSocket("A")
        running = endpoint.begin(ws)
        try:
            with pytest.raises(NotConnectedError) as info:
                await asyncio.wait_for(endpoint.send_path("control"), 1)
            assert_timeout(info.value)
            await asyncio.wait_for(running, 1)
            assert ws.received >= 3
            assert endpoint.ended == [("A", "send_timeout")]
            assert endpoint.protocol_errors == 0
            assert not endpoint.session.open
            assert not endpoint.session.pending_sends
            assert ws.cancelled == 1 and ws.delivered == []
            assert ws.close_codes == [1000]
            assert asyncio.all_tasks() == before
        finally:
            await endpoint.finish()

    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["control", "audio", "heartbeat"])
@pytest.mark.parametrize("timeout_s", [0.08, 0.18])
def test_control_audio_and_heartbeat_share_send_deadline(path: str, timeout_s: float) -> None:
    async def scenario() -> None:
        endpoint = TestEndpoint()
        ws = GatedWebSocket("deadline")
        running = endpoint.begin(
            ws, timeout_s=timeout_s, interval_s=0.005 if path == "heartbeat" else 10
        )
        send = None if path == "heartbeat" else asyncio.create_task(endpoint.send_path(path))
        try:
            await asyncio.wait_for(ws.entered.wait(), 1)
            started = asyncio.get_running_loop().time()
            if send is not None:
                with pytest.raises(NotConnectedError) as info:
                    await asyncio.wait_for(send, 1)
                assert_timeout(info.value)
            await asyncio.wait_for(running, 1)
            elapsed = asyncio.get_running_loop().time() - started
            assert timeout_s * 0.8 <= elapsed <= timeout_s + 0.25
            assert endpoint.ended == [("deadline", "send_timeout")]
            assert endpoint.session.send_timed_out.is_set()
            assert not endpoint.session.pending_sends
            assert ws.cancelled == 1
        finally:
            await endpoint.finish()

    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["control", "audio"])
def test_timed_out_send_cannot_reach_next_session(path: str) -> None:
    async def scenario() -> None:
        endpoint = TestEndpoint()
        first_ws = GatedWebSocket("A")
        first_run = endpoint.begin(first_ws)
        first = endpoint.session
        try:
            with pytest.raises(NotConnectedError) as info:
                await asyncio.wait_for(endpoint.send_path(path), 1)
            assert_timeout(info.value)
            await asyncio.wait_for(first_run, 1)
        finally:
            await endpoint.finish()

        second_ws = GatedWebSocket("B", blocked=False)
        endpoint.begin(second_ws)
        try:
            first_ws.gate.set()
            await endpoint.send_path(path)
            await asyncio.sleep(0)
            expected = (
                encode_message(control("B")) if path == "control" else encode_audio_frame(audio())
            )
            assert second_ws.delivered == [expected]
            assert first_ws.delivered == []
            assert first_ws.cancelled == 1 and not first.pending_sends
            with pytest.raises(NotConnectedError, match="no live session"):
                await endpoint.send_path(path, "A")
            assert len(second_ws.delivered) == 1
        finally:
            await endpoint.finish()

    asyncio.run(scenario())


def test_close_during_blocked_send_leaves_no_task() -> None:
    before = set(threading.enumerate())
    endpoint = TestEndpoint()
    ws = GatedWebSocket("closing")
    errors: list[Exception] = []

    def send() -> None:
        try:
            endpoint.send_message(control("closing"))
        except Exception as exc:
            errors.append(exc)

    endpoint.start(ws)
    sender = threading.Thread(target=send, name="blocked-sync-sender")
    sender.start()
    try:
        assert ws.thread_entered.wait(1)
        endpoint.close()
        sender.join(1)
        assert not sender.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], NotConnectedError)
        assert str(errors[0]) == "session ended while sending"
        assert errors[0].__cause__ is None and errors[0].__suppress_context__
        assert not endpoint.session.pending_sends
        assert ws.cancelled == 1
        assert endpoint.ended == [("closing", "closed")]
        assert set(threading.enumerate()) <= before
    finally:
        if sender.is_alive():
            endpoint.release_gate()
        endpoint.close()
        sender.join(1)


@pytest.mark.parametrize("stop", [False, True])
def test_session_end_notifies_once_when_timeout_and_peer_close_race(monkeypatch, stop) -> None:
    async def scenario() -> None:
        endpoint = TestEndpoint()
        ws = GatedWebSocket("race")
        original_wait = asyncio.wait

        async def wait_with_late_events(*args, **kwargs):
            result = await original_wait(*args, **kwargs)
            # The receiver has finished, but the event waiters haven't resumed yet.
            endpoint.session.send_timed_out.set()
            if stop:
                endpoint.stop_now()
            assert all(not task.done() for task in result[1])
            return result

        monkeypatch.setattr(asyncio, "wait", wait_with_late_events)
        running = endpoint.begin(ws)
        ws.peer_closed = True
        await asyncio.wait_for(running, 1)
        assert endpoint.ended == [("race", "closed" if stop else "send_timeout")]
        assert not endpoint.session.pending_sends
        assert ws.close_codes == [1000]

    asyncio.run(scenario())


def test_connection_closed_send_suppresses_underlying_exception() -> None:
    class ClosedWebSocket(GatedWebSocket):
        async def send(self, data: str | bytes) -> None:
            raise ConnectionClosedOK(Close(1000, "synthetic"), Close(1000, "synthetic"), True)

    async def scenario() -> None:
        endpoint = TestEndpoint()
        endpoint.begin(ClosedWebSocket("closed"))
        try:
            with pytest.raises(NotConnectedError, match="session ended while sending") as info:
                await endpoint.send_path("control")
            assert info.value.__cause__ is None and info.value.__suppress_context__
            assert not endpoint.session.pending_sends
        finally:
            await endpoint.finish()

    asyncio.run(scenario())


def test_cancelled_send_in_open_session_propagates_cancellation() -> None:
    async def scenario() -> None:
        endpoint = TestEndpoint()
        ws = GatedWebSocket("cancelled")
        endpoint.begin(ws)
        send = asyncio.create_task(endpoint.send_path("control"))
        try:
            await asyncio.wait_for(ws.entered.wait(), 1)
            send.cancel()
            with pytest.raises(asyncio.CancelledError):
                await send
            assert endpoint.session.open
            assert not endpoint.session.send_timed_out.is_set()
            assert not endpoint.session.pending_sends
            assert ws.cancelled == 1
        finally:
            await endpoint.finish()

    asyncio.run(scenario())
