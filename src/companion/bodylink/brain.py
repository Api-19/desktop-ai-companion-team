"""Brain (Mac) side: WSS server that accepts exactly one paired body at a time."""

import asyncio
import secrets
import ssl
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from companion.bodylink.transport import (
    CLOSE_BUSY,
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
    HEARTBEAT_INTERVAL_MS,
    LIVENESS_TIMEOUT_MS,
    PROTOCOL_VERSION,
    AudioFrame,
    BodyHeartbeat,
    Hello,
    Welcome,
    encode_message,
    parse_body_message,
)


class BrainServer(Endpoint):
    """Listens on ``host:port``; only a client certificate whose CN is ``body_id`` (signed by the
    pairing CA in ``ssl_context``) may connect. A second body is closed with 1013 while the first
    is connected.

    ``on_session_started(session_id, hello)`` fires after Welcome is sent;
    ``on_session_ended(session_id, reason)`` fires once per started session."""

    _thread_name = "bodylink-brain"
    _incoming_direction = "uplink"
    _parse = staticmethod(parse_body_message)
    _heartbeat_type = BodyHeartbeat
    _handshake_type = Hello

    def __init__(
        self,
        *,
        host: str,
        port: int,
        ssl_context: ssl.SSLContext,
        body_id: str,
        on_session_started: Callable[[str, Hello], None],
        on_session_ended: Callable[[str, EndReason], None],
        on_message: Callable[[Any], None],
        on_frame: Callable[[AudioFrame], None],
        heartbeat_interval_ms: int = HEARTBEAT_INTERVAL_MS,
        liveness_timeout_ms: int = LIVENESS_TIMEOUT_MS,
    ) -> None:
        super().__init__(
            on_session_ended=on_session_ended, on_message=on_message, on_frame=on_frame
        )
        self._host = host
        self._requested_port = port
        self._ssl_context = ssl_context
        self._body_id = body_id
        self._on_session_started = on_session_started
        self._interval_ms = heartbeat_interval_ms
        self._timeout_ms = liveness_timeout_ms
        self._server: Server | None = None
        self._busy = False
        self.port: int | None = None

    def start(self) -> None:
        """Bind and start listening; raises (and leaves no thread) if binding fails."""
        self._start_thread()
        try:
            self._call(self._open())
        except Exception:
            self.close()
            raise

    async def _open(self) -> None:
        self._server = await serve(
            self._handle,
            self._host,
            self._requested_port,
            ssl=self._ssl_context,
            process_request=self._check_client_name,
            compression=None,
            ping_interval=None,
            max_size=MAX_WS_MESSAGE_BYTES,
            open_timeout=self._timeout_ms / 1000,
            close_timeout=CLOSE_TIMEOUT_S,
            logger=WIRE_LOGGER,
            server_header=None,
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def _main(self) -> None:
        assert self._stop is not None
        await self._stop.wait()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def _check_client_name(self, connection: ServerConnection, request: Request) -> Response | None:
        cert = connection.transport.get_extra_info("peercert")
        names = [
            value for rdn in cert.get("subject", ()) for key, value in rdn if key == "commonName"
        ]
        if names != [self._body_id]:
            return connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        if self._busy:
            await ws.close(CLOSE_BUSY, "busy")
            return
        self._busy = True
        try:
            await self._serve_body(ws)
        finally:
            self._busy = False

    async def _serve_body(self, ws: ServerConnection) -> None:
        self._seq = 0
        try:
            async with asyncio.timeout(self._timeout_ms / 1000):
                hello = await self._handshake_message(ws)
                if hello.body_id != self._body_id:
                    raise ValueError("unpaired body")
                session_id = secrets.token_urlsafe(12)
                welcome = Welcome(
                    protocol_version=PROTOCOL_VERSION,
                    seq=self._next_seq(),
                    session_id=session_id,
                    heartbeat_interval_ms=self._interval_ms,
                    liveness_timeout_ms=self._timeout_ms,
                )
                await ws.send(encode_message(welcome))
        except ValueError:
            self._record_protocol_error()
            await ws.close(CLOSE_PROTOCOL_ERROR)
            return
        except TimeoutError:
            await ws.close(CLOSE_NORMAL)
            return
        except ConnectionClosed as exc:
            if _is_protocol_close(exc):
                self._record_protocol_error()
            return
        session = Session(
            ws=ws,
            session_id=session_id,
            interval_s=self._interval_ms / 1000,
            timeout_s=self._timeout_ms / 1000,
            last_rx=asyncio.get_running_loop().time(),
        )
        self._session = session
        self._on_session_started(session_id, hello)
        try:
            await self._run_session(session)
        finally:
            self._session = None
