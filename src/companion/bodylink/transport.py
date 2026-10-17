"""Shared session machinery for the brain server and the body client.

Each endpoint runs one asyncio event loop in one background thread. Synchronous callers use
``send_message`` / ``send_frame`` / ``close`` from any thread except the transport thread
itself; callbacks run on the transport thread and must return quickly without raising.
Nothing here logs message bodies or audio.
"""

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeVar

from websockets.asyncio.connection import Connection
from websockets.exceptions import ConnectionClosed

from companion.contracts.body import (
    AUDIO_FRAME_HEADER_BYTES,
    MAX_AUDIO_FRAME_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    UINT32_MAX,
    AudioFrame,
    decode_audio_frame,
    encode_audio_frame,
    encode_message,
)

EndReason = Literal["liveness_timeout", "protocol_error", "peer_closed", "closed", "send_timeout"]

# Largest websocket message either side accepts: one full audio frame. Control messages are
# additionally capped at MAX_CONTROL_MESSAGE_BYTES by the contract parsers.
MAX_WS_MESSAGE_BYTES = AUDIO_FRAME_HEADER_BYTES + MAX_AUDIO_FRAME_PAYLOAD_BYTES
CLOSE_TIMEOUT_S = 1.0
CLOSE_NORMAL = 1000
CLOSE_PROTOCOL_ERROR = 1002
CLOSE_BUSY = 1013
# Close codes meaning "the other side violated the protocol" (incl. websockets' own size check).
_PROTOCOL_CLOSE_CODES = frozenset({1002, 1007, 1009})

# Discard all logs from this dedicated wire logger, including DEBUG frame contents and
# ERROR tracebacks that may contain control messages, request headers, or audio.
WIRE_LOGGER = logging.getLogger("companion.bodylink.wire")
WIRE_LOGGER.handlers[:] = [logging.NullHandler()]
WIRE_LOGGER.propagate = False
WIRE_LOGGER.setLevel(logging.INFO)

T = TypeVar("T")


class NotConnectedError(RuntimeError):
    """No live session, or the message targets a session that has ended."""


UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class TransportStats:
    sessions_started: int = 0
    sessions_ended_liveness_timeout: int = 0
    sessions_ended_protocol_error: int = 0
    sessions_ended_peer_closed: int = 0
    sessions_ended_closed: int = 0
    sessions_ended_send_timeout: int = 0
    protocol_errors: int = 0
    tx_control_messages: int = 0
    rx_control_messages: int = 0
    tx_audio_frames: int = 0
    rx_audio_frames: int = 0
    tx_application_bytes: int = 0
    rx_application_bytes: int = 0
    connection_attempts: int | Literal["unavailable"] = UNAVAILABLE
    connection_failures: int | Literal["unavailable"] = UNAVAILABLE
    last_valid_rx_monotonic_s: float | Literal["unavailable"] = UNAVAILABLE
    last_session_started_monotonic_s: float | Literal["unavailable"] = UNAVAILABLE
    last_session_ended_monotonic_s: float | Literal["unavailable"] = UNAVAILABLE
    last_ended_session_valid_rx_monotonic_s: float | Literal["unavailable"] = UNAVAILABLE


@dataclass
class Session:
    ws: Connection
    session_id: str
    interval_s: float
    timeout_s: float
    last_rx: float
    open: bool = True
    send_timed_out: asyncio.Event = field(default_factory=asyncio.Event)
    pending_sends: set[asyncio.Task[None]] = field(default_factory=set)
    next_rx_seq: int = 1
    last_valid_rx: float | None = None


class Endpoint:
    """Base for BrainServer / BodyClient. Subclasses set the class attributes and ``_main``."""

    _thread_name: str
    _incoming_direction: Literal["uplink", "downlink"]
    _parse: Callable[[str], Any]
    _heartbeat_type: type
    _handshake_type: type

    def __init__(
        self,
        *,
        on_session_ended: Callable[[str, EndReason], None],
        on_message: Callable[[Any], None],
        on_frame: Callable[[AudioFrame], None],
    ) -> None:
        self._on_session_ended = on_session_ended
        self._on_message = on_message
        self._on_frame = on_frame
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._session: Session | None = None
        self._seq = 0
        self.protocol_errors = 0
        self._stats_lock = threading.Lock()
        self._stats = asdict(TransportStats())
        # Keep the existing public counter as the single source of protocol errors.
        del self._stats["protocol_errors"]

    # ---- synchronous API ----

    def stats_snapshot(self) -> TransportStats:
        """Body-free, immutable observations accumulated over this endpoint's lifetime."""
        with self._stats_lock:
            return TransportStats(**self._stats, protocol_errors=self.protocol_errors)

    def send_message(self, message: Any) -> None:
        """Send a control message of the current session. ``seq`` is assigned here."""
        self._call(self._send_message(message))

    def send_frame(self, session_id: str, frame: AudioFrame) -> None:
        self._call(self._send_frame(session_id, frame))

    def close(self) -> None:
        """Stop the session and the background thread. Idempotent."""
        if self._thread is None:
            return
        assert self._loop is not None and self._stop is not None
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join()
        self._thread = None

    # ---- thread / loop plumbing ----

    def _start_thread(self) -> None:
        ready = threading.Event()

        async def lifetime() -> None:
            self._loop = asyncio.get_running_loop()
            self._stop = asyncio.Event()
            ready.set()
            await self._main()

        self._thread = threading.Thread(
            target=asyncio.run, args=(lifetime(),), name=self._thread_name, daemon=True
        )
        self._thread.start()
        ready.wait()

    def _call(self, coro: Coroutine[Any, Any, T]) -> T:
        if self._thread is None:
            coro.close()
            raise NotConnectedError("transport is not running")
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _main(self) -> None:
        raise NotImplementedError

    # ---- session machinery (loop thread) ----

    def _record_protocol_error(self) -> None:
        with self._stats_lock:
            self.protocol_errors += 1

    def _record_received(self, session: Session, data: str | bytes) -> None:
        received_at = asyncio.get_running_loop().time()
        size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)
        counter = "rx_control_messages" if isinstance(data, str) else "rx_audio_frames"
        with self._stats_lock:
            session.last_valid_rx = received_at
            self._stats[counter] += 1
            self._stats["rx_application_bytes"] += size
            self._stats["last_valid_rx_monotonic_s"] = received_at

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq = (seq + 1) % (UINT32_MAX + 1)
        return seq

    def _live_session(self, session_id: str) -> Session:
        session = self._session
        if session is None or not session.open or session.session_id != session_id:
            raise NotConnectedError("no live session with this session_id")
        return session

    async def _send_payload(self, session: Session, data: str | bytes) -> None:
        task = asyncio.create_task(session.ws.send(data))
        session.pending_sends.add(task)
        try:
            await asyncio.wait_for(task, timeout=session.timeout_s)
        except TimeoutError:
            session.open = False
            session.send_timed_out.set()
            raise NotConnectedError("session send timed out") from None
        except ConnectionClosed:
            raise NotConnectedError("session ended while sending") from None
        except asyncio.CancelledError:
            if not session.open:
                raise NotConnectedError("session ended while sending") from None
            raise
        else:
            size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)
            counter = "tx_control_messages" if isinstance(data, str) else "tx_audio_frames"
            with self._stats_lock:
                self._stats[counter] += 1
                self._stats["tx_application_bytes"] += size
        finally:
            session.pending_sends.remove(task)

    async def _send_message(self, message: Any) -> None:
        session = self._live_session(message.session_id)
        stamped = message.model_copy(update={"seq": self._next_seq()})
        await self._send_payload(session, encode_message(stamped))

    async def _send_frame(self, session_id: str, frame: AudioFrame) -> None:
        session = self._live_session(session_id)
        await self._send_payload(session, encode_audio_frame(frame))

    async def _handshake_message(self, ws: Connection) -> Any:
        """Receive and validate one seq=0 handshake; the caller owns its deadline."""
        data = await ws.recv()
        if not isinstance(data, str):
            raise ValueError("invalid handshake")
        message = self._parse(data)
        if not isinstance(message, self._handshake_type) or message.seq != 0:
            raise ValueError("invalid handshake")
        return message

    async def _run_session(self, session: Session) -> None:
        """Pump the session until it ends, then notify ``on_session_ended`` and close."""
        assert self._stop is not None
        started_at = asyncio.get_running_loop().time()
        with self._stats_lock:
            session.last_valid_rx = session.last_rx
            self._stats["sessions_started"] += 1
            self._stats["last_session_started_monotonic_s"] = started_at
            self._stats["last_valid_rx_monotonic_s"] = session.last_valid_rx
        receive = asyncio.create_task(self._receive(session))
        watchdog = asyncio.create_task(self._watchdog(session))
        stopper = asyncio.create_task(self._stop.wait())
        heartbeat = asyncio.create_task(self._heartbeat(session))
        send_timeout = asyncio.create_task(session.send_timed_out.wait())
        await asyncio.wait(
            {receive, watchdog, stopper, send_timeout}, return_when=asyncio.FIRST_COMPLETED
        )
        reason: EndReason
        if self._stop.is_set():
            reason = "closed"
        elif session.send_timed_out.is_set():
            reason = "send_timeout"
        elif receive.done():
            reason = receive.result()
        else:
            reason = "liveness_timeout"
        session.open = False
        tasks = (*session.pending_sends, receive, watchdog, stopper, heartbeat, send_timeout)
        others = [
            task for task in tasks if not task.done()
        ]
        for task in others:
            task.cancel()
        await asyncio.gather(*others, return_exceptions=True)
        ended_at = asyncio.get_running_loop().time()
        with self._stats_lock:
            self._stats[f"sessions_ended_{reason}"] += 1
            self._stats["last_session_ended_monotonic_s"] = ended_at
            self._stats["last_ended_session_valid_rx_monotonic_s"] = session.last_valid_rx
            if reason == "protocol_error":
                self.protocol_errors += 1
        self._on_session_ended(session.session_id, reason)
        await session.ws.close(CLOSE_PROTOCOL_ERROR if reason == "protocol_error" else CLOSE_NORMAL)

    async def _receive(self, session: Session) -> EndReason:
        loop = asyncio.get_running_loop()
        try:
            async for data in session.ws:
                session.last_rx = loop.time()
                if isinstance(data, str):
                    try:
                        message = self._parse(data)
                    except ValueError:
                        return "protocol_error"
                    if (
                        isinstance(message, self._handshake_type)
                        or message.session_id != session.session_id
                    ):
                        return "protocol_error"
                    if message.seq != session.next_rx_seq:
                        return "protocol_error"
                    session.next_rx_seq = (message.seq + 1) % (UINT32_MAX + 1)
                    self._record_received(session, data)
                    if not isinstance(message, self._heartbeat_type):
                        self._on_message(message)
                else:
                    try:
                        frame = decode_audio_frame(data)
                    except ValueError:
                        return "protocol_error"
                    if frame.direction != self._incoming_direction:
                        return "protocol_error"
                    self._record_received(session, data)
                    self._on_frame(frame)
        except ConnectionClosed as exc:
            return "protocol_error" if _is_protocol_close(exc) else "peer_closed"
        return "peer_closed"

    async def _watchdog(self, session: Session) -> None:
        loop = asyncio.get_running_loop()
        while (remaining := session.last_rx + session.timeout_s - loop.time()) > 0:
            await asyncio.sleep(remaining)

    async def _heartbeat(self, session: Session) -> None:
        try:
            while True:
                await asyncio.sleep(session.interval_s)
                beat = self._heartbeat_type(
                    protocol_version=PROTOCOL_VERSION,
                    seq=self._next_seq(),
                    session_id=session.session_id,
                )
                await self._send_payload(session, encode_message(beat))
        except NotConnectedError:
            return  # _run_session reports why the session ended


def _is_protocol_close(exc: ConnectionClosed) -> bool:
    return exc.sent is not None and exc.sent.code in _PROTOCOL_CLOSE_CODES
