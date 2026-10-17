"""Body (Raspberry Pi) side: connects to the brain, keeps one session, reconnects with backoff."""

import asyncio
import ssl
from collections.abc import Callable
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from companion.bodylink.pairing import BRAIN_SERVER_NAME
from companion.bodylink.transport import (
    CLOSE_NORMAL,
    CLOSE_PROTOCOL_ERROR,
    CLOSE_TIMEOUT_S,
    MAX_WS_MESSAGE_BYTES,
    WIRE_LOGGER,
    Endpoint,
    EndReason,
    Session,
    _is_protocol_close,
)
from companion.contracts.body import (
    INPUT_SAMPLE_RATE_HZ,
    LIVENESS_TIMEOUT_MS,
    PROTOCOL_VERSION,
    AudioFrame,
    BrainHeartbeat,
    Hello,
    Welcome,
    encode_message,
    parse_brain_message,
)


class BodyClient(Endpoint):
    """Connects to ``wss://host:port/``, verifying the brain as ``BRAIN_SERVER_NAME`` against the
    pairing CA. Heartbeat interval and liveness timeout come from the brain's Welcome.

    After a failed attempt or an ended session it waits ``reconnect_initial_s`` and doubles the
    wait after each further failure up to ``reconnect_max_s``; every connection is a new session.
    ``on_session_started(session_id)`` / ``on_session_ended(session_id, reason)``."""

    _thread_name = "bodylink-body"
    _incoming_direction = "downlink"
    _parse = staticmethod(parse_brain_message)
    _heartbeat_type = BrainHeartbeat
    _handshake_type = Welcome

    def __init__(
        self,
        *,
        host: str,
        port: int,
        ssl_context: ssl.SSLContext,
        body_id: str,
        output_sample_rates: tuple[int, ...],
        on_session_started: Callable[[str], None],
        on_session_ended: Callable[[str, EndReason], None],
        on_message: Callable[[Any], None],
        on_frame: Callable[[AudioFrame], None],
        reconnect_initial_s: float = 1.0,
        reconnect_max_s: float = 10.0,
        handshake_timeout_ms: int = LIVENESS_TIMEOUT_MS,
    ) -> None:
        super().__init__(
            on_session_ended=on_session_ended, on_message=on_message, on_frame=on_frame
        )
        self._uri = f"wss://{host}:{port}/"
        self._ssl_context = ssl_context
        self._body_id = body_id
        self._output_sample_rates = output_sample_rates
        self._on_session_started = on_session_started
        self._reconnect_initial_s = reconnect_initial_s
        self._reconnect_max_s = reconnect_max_s
        self._handshake_timeout_s = handshake_timeout_ms / 1000
        with self._stats_lock:
            self._stats["connection_attempts"] = 0
            self._stats["connection_failures"] = 0

    def start(self) -> None:
        self._start_thread()

    async def _main(self) -> None:
        assert self._stop is not None
        delay = self._reconnect_initial_s
        while not self._stop.is_set():
            if await self._connect_once():
                delay = self._reconnect_initial_s
            if self._stop.is_set():
                break
            try:
                async with asyncio.timeout(delay):
                    await self._stop.wait()
            except TimeoutError:
                pass
            delay = min(delay * 2, self._reconnect_max_s)

    async def _connect_once(self) -> bool:
        """One connection attempt; True if a session was established."""
        with self._stats_lock:
            self._stats["connection_attempts"] += 1
        published = False
        try:
            async with connect(
                self._uri,
                ssl=self._ssl_context,
                server_hostname=BRAIN_SERVER_NAME,
                proxy=None,
                compression=None,
                ping_interval=None,
                max_size=MAX_WS_MESSAGE_BYTES,
                open_timeout=self._handshake_timeout_s,
                close_timeout=CLOSE_TIMEOUT_S,
                logger=WIRE_LOGGER,
                user_agent_header=None,
            ) as ws:
                self._seq = 0
                try:
                    async with asyncio.timeout(self._handshake_timeout_s):
                        hello = Hello(
                            protocol_version=PROTOCOL_VERSION,
                            seq=self._next_seq(),
                            body_id=self._body_id,
                            input_sample_rate=INPUT_SAMPLE_RATE_HZ,
                            output_sample_rates=self._output_sample_rates,
                        )
                        await ws.send(encode_message(hello))
                        welcome = await self._handshake_message(ws)
                except ValueError:
                    self._record_protocol_error()
                    await ws.close(CLOSE_PROTOCOL_ERROR)
                    return False
                except TimeoutError:
                    await ws.close(CLOSE_NORMAL)
                    return False
                except ConnectionClosed as exc:
                    if _is_protocol_close(exc):
                        self._record_protocol_error()
                    return False
                session = Session(
                    ws=ws,
                    session_id=welcome.session_id,
                    interval_s=welcome.heartbeat_interval_ms / 1000,
                    timeout_s=welcome.liveness_timeout_ms / 1000,
                    last_rx=asyncio.get_running_loop().time(),
                )
                self._session = session
                published = True
                self._on_session_started(welcome.session_id)
                try:
                    await self._run_session(session)
                finally:
                    self._session = None
                return True
        except (OSError, TimeoutError, InvalidHandshake, ConnectionClosed):
            return False
        finally:
            if not published:
                with self._stats_lock:
                    self._stats["connection_failures"] += 1
