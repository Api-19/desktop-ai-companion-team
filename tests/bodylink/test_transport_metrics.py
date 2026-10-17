"""Transport observations using synthetic in-memory peers and loopback WSS."""

import asyncio
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest
from websockets.asyncio.connection import Connection
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from companion.bodylink import brain_ports, transport
from companion.bodylink.body import BodyClient
from companion.bodylink.brain import BrainServer
from companion.contracts.body import (
    AudioFrame,
    BodyHeartbeat,
    BrainHeartbeat,
    Fault,
    ShowState,
    encode_audio_frame,
    encode_message,
)
from companion.contracts.robot import RobotEvent

from .conftest import BODY_ID, Recorder
from .test_transport import make_body, make_brain
from .test_transport_backpressure import GatedWebSocket, TestEndpoint
from .test_transport_failures import HandshakeSocket, control, handshake, heartbeat


def unstarted(side, rec=None, **kwargs):
    rec = rec or Recorder()
    common = dict(
        host="127.0.0.1", port=1, body_id=BODY_ID,
        ssl_context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT), **rec.callbacks(), **kwargs,
    )
    if side == "body":
        return BodyClient(output_sample_rates=(44100,), **common)
    return BrainServer(**common)


class QueueSocket:
    def __init__(self, *data):
        self.incoming = asyncio.Queue()
        for item in data:
            self.incoming.put_nowait(item)
        self.sent = []
        self.closed = []
        self.sent_event = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        data = await self.incoming.get()
        if data is None:
            raise StopAsyncIteration
        return data

    async def send(self, data):
        self.sent.append(data)
        self.sent_event.set()

    async def close(self, code):
        self.closed.append(code)


def session(ws, sid="synthetic-session", **kwargs):
    return transport.Session(
        ws, sid, interval_s=10, timeout_s=1,
        last_rx=asyncio.get_running_loop().time(), **kwargs,
    )


def test_stats_snapshot_has_exact_frozen_fields() -> None:
    expected = {
        "sessions_started", "sessions_ended_liveness_timeout", "sessions_ended_protocol_error",
        "sessions_ended_peer_closed", "sessions_ended_closed", "sessions_ended_send_timeout",
        "protocol_errors", "tx_control_messages", "rx_control_messages", "tx_audio_frames",
        "rx_audio_frames", "tx_application_bytes", "rx_application_bytes", "connection_attempts",
        "connection_failures", "last_valid_rx_monotonic_s", "last_session_started_monotonic_s",
        "last_session_ended_monotonic_s", "last_ended_session_valid_rx_monotonic_s",
    }
    snapshot = TestEndpoint().stats_snapshot()
    assert type(snapshot) is transport.TransportStats
    assert {item.name for item in fields(snapshot)} == expected
    assert not hasattr(snapshot, "__dict__")
    for name in expected:
        unavailable = name.startswith("last_") or name.startswith("connection_")
        assert getattr(snapshot, name) == (transport.UNAVAILABLE if unavailable else 0)
    with pytest.raises(FrozenInstanceError):
        snapshot.sessions_started = 1


def loopback_exchange(pairing, closer):
    brain_rec, body_rec = Recorder(), Recorder()
    brain = make_brain(
        pairing, brain_rec, closer, heartbeat_interval_ms=30000, liveness_timeout_ms=60000
    )
    body = make_body(pairing, body_rec, closer, brain.port)
    brain_rec.wait_for(lambda: brain_rec.started)
    body_rec.wait_for(lambda: body_rec.started)
    # Barrier after the started callback and _run_session initialization on each loop.
    for endpoint in (brain, body):
        endpoint._call(asyncio.sleep(0))
        stats = endpoint.stats_snapshot()
        assert stats.sessions_started == 1
        assert stats.tx_application_bytes == stats.rx_application_bytes == 0
    sid = brain_rec.started[0][0]
    up_control = Fault(protocol_version=1, seq=1, session_id=sid,
                       kind="internal_error", detail="合成 café")
    down_control = ShowState(protocol_version=1, seq=1, session_id=sid,
                             event=RobotEvent(state="idle", detail="合成声音"))
    up_beat = BodyHeartbeat(protocol_version=1, seq=2, session_id=sid)
    down_beat = BrainHeartbeat(protocol_version=1, seq=2, session_id=sid)
    up = AudioFrame("uplink", 1, 0, True, b"\x01\x00")
    down = AudioFrame("downlink", 2, 0, True, b"")
    for endpoint, message, beat, frame in (
        (body, up_control, up_beat, up), (brain, down_control, down_beat, down)
    ):
        endpoint.send_message(message)
        endpoint.send_message(beat)
        endpoint.send_frame(sid, frame)
    brain_rec.wait_for(lambda: brain_rec.frames)
    body_rec.wait_for(lambda: body_rec.frames)
    return brain, body, brain_rec, body_rec, (
        [encode_message(up_control), encode_message(up_beat), encode_audio_frame(up)],
        [encode_message(down_control), encode_message(down_beat), encode_audio_frame(down)],
    )


def test_stats_count_successful_control_audio_and_utf8_bytes(pairing, closer):
    brain, body, brain_rec, body_rec, payloads = loopback_exchange(pairing, closer)
    snapshots = brain.stats_snapshot(), body.stats_snapshot()
    sizes = [sum(len(x.encode("utf-8")) if isinstance(x, str) else len(x) for x in data)
             for data in payloads]
    assert any(len(x.encode("utf-8")) > len(x) for x in payloads[0] if isinstance(x, str))
    for stats, tx_bytes, rx_bytes in (
        (snapshots[0], sizes[1], sizes[0]), (snapshots[1], sizes[0], sizes[1])
    ):
        assert stats.tx_control_messages == stats.rx_control_messages == 2
        assert stats.tx_audio_frames == stats.rx_audio_frames == 1
        assert stats.tx_application_bytes == tx_bytes
        assert stats.rx_application_bytes == rx_bytes
        assert stats.last_valid_rx_monotonic_s >= stats.last_session_started_monotonic_s
    assert len(brain_rec.messages) == len(body_rec.messages) == 1  # heartbeat is filtered


def test_metrics_do_not_add_wire_messages(pairing, closer, monkeypatch):
    sent, received = {"brain": [], "body": []}, {"brain": [], "body": []}
    lock = threading.Lock()
    original_send, original_recv = Connection.send, Connection.recv

    async def send(ws, data, *args, **kwargs):
        await original_send(ws, data, *args, **kwargs)
        with lock:
            sent["brain" if isinstance(ws, ServerConnection) else "body"].append(data)

    async def recv(ws, *args, **kwargs):
        data = await original_recv(ws, *args, **kwargs)
        with lock:
            received["brain" if isinstance(ws, ServerConnection) else "body"].append(data)
        return data

    monkeypatch.setattr(Connection, "send", send)
    monkeypatch.setattr(Connection, "recv", recv)
    brain, body, _, _, payloads = loopback_exchange(pairing, closer)
    for _ in range(10):
        brain.stats_snapshot()
        body.stats_snapshot()
    body.close()
    brain.close()
    for endpoint in (brain, body):
        stats = endpoint.stats_snapshot()
        assert stats.sessions_started == 1
        assert stats.tx_control_messages == stats.rx_control_messages == 2
        assert stats.tx_audio_frames == stats.rx_audio_frames == 1
        assert sum(getattr(stats, f.name) for f in fields(stats)
                   if f.name.startswith("sessions_ended_")) == 1
    assert sent["body"][1:] == received["brain"][1:] == payloads[0]
    assert sent["brain"][1:] == received["body"][1:] == payloads[1]
    assert len(sent["body"]) == len(sent["brain"]) == 4  # handshake + three payloads
    assert '"type":"hello"' in sent["body"][0]
    assert '"type":"welcome"' in sent["brain"][0]


@pytest.mark.parametrize("side", ["brain", "body"])
@pytest.mark.parametrize("kind", [
    "parse", "handshake", "session", "seq", "audio_decode", "audio_direction",
])
def test_failed_send_and_invalid_receive_are_not_counted_as_success(side, kind):
    async def scenario():
        endpoint = unstarted(side)
        ws = GatedWebSocket("synthetic-session")
        current = session(ws)
        current.timeout_s = 0.01
        with pytest.raises(transport.NotConnectedError, match="timed out"):
            await endpoint._send_payload(current, "not delivered")
        assert endpoint.stats_snapshot().tx_application_bytes == 0
        assert endpoint.stats_snapshot().tx_control_messages == 0
        bad = {
            "parse": "{", "handshake": encode_message(handshake(side)),
            "session": encode_message(control(side, "wrong", 1)),
            "seq": encode_message(control(side, current.session_id, 2)),
            "audio_decode": b"bad audio",
            "audio_direction": encode_audio_frame(AudioFrame(
                "downlink" if side == "brain" else "uplink", 1, 0, True, b""
            )),
        }[kind]
        current.ws = QueueSocket(bad)
        assert await endpoint._receive(current) == "protocol_error"
        stats = endpoint.stats_snapshot()
        assert stats.rx_control_messages == stats.rx_audio_frames == stats.rx_application_bytes == 0
        assert stats.last_valid_rx_monotonic_s == transport.UNAVAILABLE
        assert current.last_valid_rx is None

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["closed", "cancelled"])
def test_disconnected_and_cancelled_sends_are_not_successes(fault):
    class ClosedSocket(QueueSocket):
        async def send(self, data):
            raise ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)

    async def scenario():
        endpoint = unstarted("body")
        ws = ClosedSocket() if fault == "closed" else GatedWebSocket("synthetic-session")
        current = session(ws)
        task = asyncio.create_task(endpoint._send_payload(current, b"undelivered"))
        if fault == "cancelled":
            await ws.entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(transport.NotConnectedError):
                await task
        stats = endpoint.stats_snapshot()
        assert stats.tx_audio_frames == stats.tx_application_bytes == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", [
    "liveness_timeout", "protocol_error", "peer_closed", "closed", "send_timeout",
])
def test_stats_count_each_session_end_once(reason, monkeypatch):
    async def scenario():
        rec = Recorder()
        endpoint = unstarted("brain", rec)
        endpoint._stop = asyncio.Event()
        ws = QueueSocket()
        current = session(ws)
        if reason == "liveness_timeout":
            current.last_rx -= current.timeout_s
        elif reason == "protocol_error":
            ws.incoming.put_nowait("{")
        elif reason == "peer_closed":
            ws.incoming.put_nowait(None)
        else:
            # Make receiver completion race with timeout and local close, as in production.
            ws.incoming.put_nowait(None)
            original_wait = asyncio.wait

            async def racing_wait(*args, **kwargs):
                result = await original_wait(*args, **kwargs)
                current.send_timed_out.set()
                if reason == "closed":
                    endpoint._stop.set()
                return result

            monkeypatch.setattr(asyncio, "wait", racing_wait)

        initial_valid_rx = current.last_rx
        ended_snapshots = []

        def on_end(sid, actual):
            ended_snapshots.append(endpoint.stats_snapshot())  # lock released before callback
            rec.on_session_ended(sid, actual)

        endpoint._on_session_ended = on_end
        await asyncio.wait_for(endpoint._run_session(current), 1)
        endpoint.close()
        endpoint.close()
        stats = endpoint.stats_snapshot()
        assert stats.sessions_started == 1
        counts = {f.name: getattr(stats, f.name) for f in fields(stats)
                  if f.name.startswith("sessions_ended_")}
        assert sum(counts.values()) == 1
        assert counts[f"sessions_ended_{reason}"] == 1
        assert stats.protocol_errors == endpoint.protocol_errors == (reason == "protocol_error")
        assert rec.ended[0][:2] == (current.session_id, reason)
        assert ended_snapshots == [stats]
        assert stats.last_session_ended_monotonic_s >= stats.last_session_started_monotonic_s
        assert stats.last_ended_session_valid_rx_monotonic_s == initial_valid_rx
        assert not current.pending_sends

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", [
    "connect", "invalid", "timeout", "protocol_close", "cancelled", "started",
])
def test_client_attempts_and_failures_exclude_started_sessions(fault, monkeypatch):
    class AttemptSocket(HandshakeSocket):
        async def recv(self):
            if fault == "invalid":
                return "{"
            if fault == "timeout":
                await asyncio.Event().wait()
            if fault == "protocol_close":
                from websockets.exceptions import ConnectionClosedError

                raise ConnectionClosedError(Close(1002, ""), Close(1002, ""), True)
            if fault == "cancelled":
                raise asyncio.CancelledError
            return await super().recv()

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def scenario():
        rec = Recorder()
        endpoint = unstarted("body", rec, handshake_timeout_ms=10)
        assert endpoint.stats_snapshot().connection_attempts == 0
        assert endpoint.stats_snapshot().connection_failures == 0
        endpoint._stop = asyncio.Event()

        @asynccontextmanager
        async def connect(*args, **kwargs):
            if fault == "connect":
                raise OSError("synthetic connect failure")
            yield AttemptSocket("body")

        monkeypatch.setattr("companion.bodylink.body.connect", connect)
        for attempt in range(1, 3):
            if fault == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await endpoint._connect_once()
            else:
                assert await endpoint._connect_once() is (fault == "started")
            stats = endpoint.stats_snapshot()
            assert stats.connection_attempts == attempt
            assert stats.connection_failures == (0 if fault == "started" else attempt)
            assert stats.sessions_started == (attempt if fault == "started" else 0)
            assert stats.sessions_ended_peer_closed == stats.sessions_started
            assert stats.protocol_errors == endpoint.protocol_errors == (
                attempt if fault in {"invalid", "protocol_close"} else 0
            )
            assert stats.tx_application_bytes == stats.rx_application_bytes == 0
            assert endpoint._session is None
        assert len(rec.started) == len(rec.ended) == (2 if fault == "started" else 0)

    asyncio.run(scenario())


def test_server_connection_counts_are_unavailable(pairing, closer):
    brain = unstarted("brain")
    assert brain.stats_snapshot().connection_attempts == transport.UNAVAILABLE
    assert brain.stats_snapshot().connection_failures == transport.UNAVAILABLE
    brain, body, _, _, _ = loopback_exchange(pairing, closer)
    body.close()
    brain.close()
    assert brain.stats_snapshot().connection_attempts == transport.UNAVAILABLE
    assert brain.stats_snapshot().connection_failures == transport.UNAVAILABLE


@pytest.mark.parametrize("side", ["brain", "body"])
def test_handshake_protocol_errors_share_snapshot_counter(side, monkeypatch):
    async def scenario():
        endpoint = unstarted(side)
        ws = QueueSocket()

        async def recv():
            return "{"

        ws.recv = recv

        @asynccontextmanager
        async def connect(*args, **kwargs):
            yield ws

        monkeypatch.setattr("companion.bodylink.body.connect", connect)
        for count in (1, 2):
            if side == "brain":
                await endpoint._serve_body(ws)
            else:
                assert await endpoint._connect_once() is False
            stats = endpoint.stats_snapshot()
            assert stats.protocol_errors == endpoint.protocol_errors == count
            assert stats.sessions_started == 0
            assert sum(getattr(stats, f.name) for f in fields(stats)
                       if f.name.startswith("sessions_ended_")) == 0

    asyncio.run(scenario())


def test_stats_are_cumulative_and_snapshot_is_thread_safe():
    endpoint = unstarted("brain")
    snapshots = []
    stop = threading.Event()

    def read():
        local = []
        while not stop.wait(0.0001):
            stats = endpoint.stats_snapshot()
            assert stats.tx_application_bytes == stats.tx_audio_frames * 22
            assert 0 <= stats.sessions_started - stats.sessions_ended_closed <= 1
            local.append(stats)
        return local

    async def write():
        for index in range(3):
            ws = QueueSocket()
            endpoint._stop = asyncio.Event()
            current = session(ws, sid=f"session-{index}")
            running = asyncio.create_task(endpoint._run_session(current))
            await asyncio.sleep(0)
            for _ in range(40):
                await endpoint._send_payload(
                    current, encode_audio_frame(AudioFrame("uplink", 1, 0, True, b"\0\0"))
                )
            endpoint._stop.set()
            await running

    with ThreadPoolExecutor(max_workers=3) as pool:
        readers = [pool.submit(read) for _ in range(3)]
        try:
            asyncio.run(write())
        finally:
            stop.set()
        for future in readers:
            snapshots.extend(future.result())
    assert snapshots
    stats = endpoint.stats_snapshot()
    assert stats.sessions_started == stats.sessions_ended_closed == 3
    assert stats.tx_audio_frames == 120 and stats.tx_application_bytes == 2640
    assert snapshots[0].tx_audio_frames <= stats.tx_audio_frames
    endpoint.close()
    assert endpoint.stats_snapshot() == stats


def test_brain_link_exposes_transport_snapshot(monkeypatch, tmp_path):
    snapshot = transport.TransportStats(sessions_started=7)
    calls = []

    def stats_snapshot():
        calls.append(True)
        return snapshot

    monkeypatch.setattr(brain_ports, "BrainServer", lambda **kw: SimpleNamespace(
        stats_snapshot=stats_snapshot, start=lambda: None, close=lambda: None,
    ))
    link = brain_ports.BrainBodyLink(
        host="127.0.0.1", port=1, body_id=BODY_ID,
        ssl_context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT), run_root=tmp_path,
    )
    assert link.stats_snapshot() is snapshot
    link.start()
    assert link.stats_snapshot() is snapshot
    link.close()
    assert link.stats_snapshot() is snapshot
    assert len(calls) == 3


def test_last_end_preserves_source_valid_receive_time():
    async def scenario():
        endpoint = unstarted("brain")
        endpoint._stop = asyncio.Event()
        first = session(QueueSocket())
        first.ws.incoming.put_nowait(encode_message(heartbeat("brain", first.session_id, 1)))
        first.ws.incoming.put_nowait("{")
        await endpoint._run_session(first)
        ended = endpoint.stats_snapshot()
        assert ended.last_valid_rx_monotonic_s == first.last_valid_rx
        assert first.last_valid_rx < first.last_rx  # invalid input only moves liveness clock
        assert ended.last_ended_session_valid_rx_monotonic_s == first.last_valid_rx
        second = session(QueueSocket(None), sid="second-session")
        running = asyncio.create_task(endpoint._run_session(second))
        await asyncio.sleep(0)
        active = endpoint.stats_snapshot()
        assert active.last_valid_rx_monotonic_s == second.last_rx
        assert active.last_ended_session_valid_rx_monotonic_s == first.last_valid_rx
        await running
        final = endpoint.stats_snapshot()
        assert final.sessions_started == 2
        assert final.sessions_ended_protocol_error == final.sessions_ended_peer_closed == 1
        assert final.rx_control_messages == 1
        assert final.last_ended_session_valid_rx_monotonic_s == second.last_rx
        assert ended.last_ended_session_valid_rx_monotonic_s == first.last_valid_rx

    asyncio.run(scenario())


@pytest.mark.parametrize("side", ["brain", "body"])
def test_receive_stats_precede_callbacks_and_automatic_heartbeats_count(side):
    async def scenario():
        endpoint = unstarted(side)
        ws = QueueSocket()
        current = session(ws)
        current.interval_s = 0.001
        endpoint._seq = 1
        pumping = asyncio.create_task(endpoint._heartbeat(current))
        await ws.sent_event.wait()
        current.interval_s = 10
        while current.pending_sends:
            await asyncio.sleep(0)
        pumping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pumping
        assert len(ws.sent) == endpoint.stats_snapshot().tx_control_messages == 1
        assert endpoint.stats_snapshot().tx_application_bytes == len(ws.sent[0].encode("utf-8"))
        seen = []
        endpoint._on_message = lambda message: seen.append(endpoint.stats_snapshot())
        endpoint._on_frame = lambda frame: seen.append(endpoint.stats_snapshot())
        direction = "uplink" if side == "brain" else "downlink"
        current.ws = QueueSocket(
            encode_message(control(side, current.session_id, 1)),
            encode_audio_frame(AudioFrame(direction, 1, 0, True, b"")), None,
        )
        assert await endpoint._receive(current) == "peer_closed"
        assert seen[0].rx_control_messages == 1
        assert seen[1].rx_audio_frames == 1
        assert seen[1].last_valid_rx_monotonic_s == current.last_valid_rx

    asyncio.run(scenario())
