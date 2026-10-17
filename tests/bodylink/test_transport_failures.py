"""Synthetic peers, bounded handshake faults, and verified loopback TLS failures."""

import asyncio
import shutil
import ssl
import subprocess
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from queue import Queue

import pytest
from websockets.asyncio.client import connect as async_connect
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidHandshake,
)
from websockets.frames import Close
from websockets.sync.server import serve

from companion.bodylink.body import BodyClient
from companion.bodylink.brain import BrainServer
from companion.bodylink.pairing import (
    BRAIN_SERVER_NAME,
    PairingPaths,
    body_ssl_context,
    brain_ssl_context,
)
from companion.bodylink.transport import MAX_WS_MESSAGE_BYTES, WIRE_LOGGER
from companion.contracts.body import (
    UINT32_MAX,
    AudioFrame,
    BodyHeartbeat,
    BrainHeartbeat,
    ButtonPressed,
    Hello,
    StartRecording,
    Welcome,
    encode_audio_frame,
    encode_message,
    parse_body_message,
    parse_brain_message,
)

from .conftest import BODY_ID, Recorder
from .test_transport import make_body, make_brain, raw_connect, raw_hello, wait_closed_code

SID = "synthetic-session"
SIDES = ["brain", "body"]


def hello(*, seq=0, body_id=BODY_ID):
    return Hello(
        protocol_version=1, seq=seq, body_id=body_id, input_sample_rate=16000,
        output_sample_rates=(44100,),
    )  # fmt: skip


def welcome(*, seq=0, session_id=SID):
    return Welcome(
        protocol_version=1, seq=seq, session_id=session_id,
        heartbeat_interval_ms=1000, liveness_timeout_ms=3000,
    )  # fmt: skip


def handshake(side, *, seq=0):
    return hello(seq=seq) if side == "brain" else welcome(seq=seq)


def control(side, sid, seq):
    if side == "brain":
        return ButtonPressed(protocol_version=1, seq=seq, session_id=sid, observed_at_ms=7)
    return StartRecording(protocol_version=1, seq=seq, session_id=sid, turn_id=3, max_seconds=5)


def heartbeat(side, sid, seq):
    cls = BodyHeartbeat if side == "brain" else BrainHeartbeat
    return cls(protocol_version=1, seq=seq, session_id=sid)


@contextmanager
def raw_server(context: ssl.SSLContext, *, port=0) -> Iterator[tuple[int, Queue]]:
    peers = Queue()
    release = threading.Event()

    def handle(ws):
        peers.put(ws)
        release.wait(10)

    with serve(
        handle, "127.0.0.1", port, ssl=context, ping_interval=None,
        close_timeout=1, logger=WIRE_LOGGER,
    ) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield server.socket.getsockname()[1], peers
        finally:
            release.set()
            server.shutdown()
            thread.join(3)
            assert not thread.is_alive()


@contextmanager
def raw_peer(pairing, closer, side, *, timeout_ms=3000):
    """Expose a verified peer before its application handshake."""
    rec = Recorder()
    if side == "brain":
        endpoint = make_brain(pairing, rec, closer, liveness_timeout_ms=timeout_ms,
                              heartbeat_interval_ms=100 if timeout_ms < 2000 else 1000)
        with raw_connect(pairing, endpoint.port) as ws:
            yield endpoint, rec, ws
        endpoint.close()
    else:
        with raw_server(brain_ssl_context(pairing.brain_dir)) as (port, peers):
            endpoint = make_body(
                pairing, rec, closer, port, handshake_timeout_ms=timeout_ms,
                reconnect_initial_s=60,
            )
            ws = peers.get(timeout=3)
            received = parse_body_message(ws.recv(timeout=3))
            assert isinstance(received, Hello) and received.seq == 0
            try:
                yield endpoint, rec, ws
            finally:
                ws.close()
                endpoint.close()


@contextmanager
def raw_session(pairing, closer, side):
    with raw_peer(pairing, closer, side) as (endpoint, rec, ws):
        if side == "brain":
            sid = raw_hello(ws).session_id
        else:
            sid = SID
            ws.send(encode_message(welcome()))
        rec.wait_for(lambda: rec.started)
        yield endpoint, rec, ws, sid


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("message_kind", ["control", "heartbeat"])
@pytest.mark.parametrize("violation,seq", [("duplicate", 2), ("backward", 1), ("gap", 4)])
def test_sequence_violation_closes_each_endpoint(
    pairing, closer, side, message_kind, violation, seq,
):
    with raw_session(pairing, closer, side) as (endpoint, rec, ws, sid):
        ws.send(encode_message(heartbeat(side, sid, 1)))
        ws.send(encode_message(control(side, sid, 2)))
        rec.wait_for(lambda: len(rec.messages) == 1)
        message = control if message_kind == "control" else heartbeat
        ws.send(encode_message(message(side, sid, seq)))
        assert wait_closed_code(ws) == 1002
        rec.wait_for(lambda: rec.ended)
        assert len(rec.messages) == 1
        assert [end[:2] for end in rec.ended] == [(sid, "protocol_error")]
        assert endpoint.protocol_errors == 1


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("seq", [0, 2])
def test_first_session_control_sequence_must_be_one(pairing, closer, side, seq):
    with raw_session(pairing, closer, side) as (endpoint, rec, ws, sid):
        ws.send(encode_message(heartbeat(side, sid, seq)))
        assert wait_closed_code(ws) == 1002
        rec.wait_for(lambda: rec.ended)
        assert rec.messages == []
        assert [end[:2] for end in rec.ended] == [(sid, "protocol_error")]
        assert endpoint.protocol_errors == 1


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("seq", [1, UINT32_MAX])
def test_handshake_requires_zero_sequence_each_endpoint(pairing, closer, side, seq):
    with raw_peer(pairing, closer, side) as (endpoint, rec, ws):
        ws.send(encode_message(handshake(side, seq=seq)))
        assert wait_closed_code(ws) == 1002
        assert rec.started == rec.ended == []
        assert endpoint.protocol_errors == 1


@pytest.mark.parametrize("side", SIDES)
def test_sequence_wrap_is_accepted(pairing, closer, side):
    with raw_session(pairing, closer, side) as (endpoint, rec, ws, sid):
        async def at_wrap():
            endpoint._session.next_rx_seq = UINT32_MAX
            endpoint._seq = UINT32_MAX

        endpoint._call(at_wrap())
        for seq in (UINT32_MAX, 0, 1):
            ws.send(encode_message(control(side, sid, seq)))
        rec.wait_for(lambda: len(rec.messages) == 3)
        assert [message.seq for message in rec.messages] == [UINT32_MAX, 0, 1]
        parser = parse_brain_message if side == "brain" else parse_body_message
        # Public sends still accept the caller's placeholder seq=0.
        for _ in range(3):
            endpoint.send_message(control("body" if side == "brain" else "brain", sid, 0))
        messages = [parser(ws.recv(timeout=3)) for _ in range(3)]
        assert [message.seq for message in messages] == [UINT32_MAX, 0, 1]
        assert rec.ended == [] and endpoint.protocol_errors == 0


def test_sequence_restarts_after_reconnect(pairing, closer):
    brain_rec, body_rec = Recorder(), Recorder()
    brain = make_brain(pairing, brain_rec, closer)
    port = brain.port
    body = make_body(pairing, body_rec, closer, port,
                     reconnect_initial_s=0.01, reconnect_max_s=0.02)
    body_rec.wait_for(lambda: body_rec.started)
    sid = body_rec.started[0][0]
    for _ in range(2):
        body.send_message(control("brain", sid, 0))
        brain.send_message(control("body", sid, 0))
    brain_rec.wait_for(lambda: len(brain_rec.messages) == 2)
    body_rec.wait_for(lambda: len(body_rec.messages) == 2)
    assert [m.seq for m in brain_rec.messages] == [1, 2]
    assert [m.seq for m in body_rec.messages] == [1, 2]
    brain.close()
    second_rec = Recorder()
    second = make_brain(pairing, second_rec, closer, port=port)
    body_rec.wait_for(lambda: len(body_rec.started) == 2)
    sid2 = body_rec.started[1][0]
    assert sid2 != sid
    body.send_message(control("brain", sid2, 0))
    second.send_message(control("body", sid2, 0))
    second_rec.wait_for(lambda: second_rec.messages)
    body_rec.wait_for(lambda: len(body_rec.messages) == 3)
    assert second_rec.messages[0].seq == body_rec.messages[2].seq == 1
    assert brain_rec.hellos[0].seq == second_rec.hellos[0].seq == 0
    assert body.protocol_errors == second.protocol_errors == 0


@pytest.mark.parametrize("side", SIDES)
def test_audio_frames_do_not_consume_control_sequence(pairing, closer, side):
    with raw_session(pairing, closer, side) as (endpoint, rec, ws, sid):
        direction = "uplink" if side == "brain" else "downlink"
        frames = [AudioFrame(direction, 3, index, index == 1, b"\x01\x00")
                  for index in range(2)]
        ws.send(encode_message(heartbeat(side, sid, 1)))
        ws.send(encode_audio_frame(frames[0]))
        ws.send(encode_message(control(side, sid, 2)))
        ws.send(encode_audio_frame(frames[1]))
        ws.send(encode_message(control(side, sid, 3)))
        rec.wait_for(lambda: len(rec.frames) == len(rec.messages) == 2)
        assert rec.frames == frames
        assert [m.seq for m in rec.messages] == [2, 3]
        other = "body" if side == "brain" else "brain"
        outbound = AudioFrame("downlink" if side == "brain" else "uplink", 3, 0, True, b"")
        endpoint.send_message(control(other, sid, 0))
        endpoint.send_frame(sid, outbound)
        endpoint.send_message(control(other, sid, 0))
        parser = parse_brain_message if side == "brain" else parse_body_message
        assert parser(ws.recv(timeout=3)).seq == 1
        assert ws.recv(timeout=3) == encode_audio_frame(outbound)
        assert parser(ws.recv(timeout=3)).seq == 2
        assert rec.ended == [] and endpoint.protocol_errors == 0


BAD_KINDS = [
    "oversized_control", "unparsable_control", "wrong_direction_control", "wrong_session",
    "repeated_handshake", "corrupt_audio", "wrong_direction_audio", "oversized_websocket",
]


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("kind", BAD_KINDS)
def test_bad_input_closes_each_endpoint(pairing, closer, side, kind):
    with raw_session(pairing, closer, side) as (endpoint, rec, ws, sid):
        other = "body" if side == "brain" else "brain"
        payloads = {
            "oversized_control": " " * 4097,
            "unparsable_control": "{not json",
            "wrong_direction_control": encode_message(control(other, sid, 2)),
            "wrong_session": encode_message(control(side, "wrong-session", 2)),
            "repeated_handshake": encode_message(handshake(side)),
            "corrupt_audio": b"XXXX" + bytes(16),
            "wrong_direction_audio": encode_audio_frame(
                AudioFrame("downlink" if side == "brain" else "uplink", 3, 0, True, b"")
            ),
            "oversized_websocket": bytes(MAX_WS_MESSAGE_BYTES + 1),
        }
        ws.send(encode_message(heartbeat(side, sid, 1)))
        ws.send(payloads[kind])
        assert wait_closed_code(ws) == (1009 if kind == "oversized_websocket" else 1002)
        rec.wait_for(lambda: rec.ended)
        assert [end[:2] for end in rec.ended] == [(sid, "protocol_error")]
        assert rec.messages == rec.frames == []
        assert endpoint.protocol_errors == 1


@pytest.mark.parametrize("side", SIDES)
def test_missing_handshake_times_out_each_endpoint(pairing, closer, side):
    with raw_peer(pairing, closer, side, timeout_ms=200) as (endpoint, rec, ws):
        assert wait_closed_code(ws) == 1000
        assert rec.started == rec.ended == []
        assert endpoint.protocol_errors == 0


def async_raw_connect(pairing, port):
    """Single-threaded peer for 200 ms budgets. The threaded sync client can strand its first
    write in OpenSSL when it races the recv thread reading the TLS 1.3 session ticket."""
    return async_connect(
        f"wss://127.0.0.1:{port}/", ssl=body_ssl_context(pairing.body_dir),
        server_hostname=BRAIN_SERVER_NAME, proxy=None, open_timeout=3, close_timeout=1,
    )  # fmt: skip


async def async_closed_code(ws):
    with pytest.raises(ConnectionClosed) as info:
        while True:
            await asyncio.wait_for(ws.recv(), 3)
    return info.value.rcvd.code


@pytest.mark.parametrize("fault", ["bad_json", "binary", "wrong_type", "unpaired", "timeout"])
def test_handshake_failure_releases_brain_busy(pairing, closer, fault):
    rec = Recorder()
    brain = make_brain(pairing, rec, closer, heartbeat_interval_ms=100, liveness_timeout_ms=200)

    async def scenario():
        async with async_raw_connect(pairing, brain.port) as ws:
            payloads = {
                "bad_json": "{", "binary": b"not control",
                "wrong_type": encode_message(heartbeat("brain", SID, 0)),
                "unpaired": encode_message(hello(body_id="unpaired")),
            }
            if fault != "timeout":
                await ws.send(payloads[fault])
            assert await async_closed_code(ws) == (1000 if fault == "timeout" else 1002)
        assert rec.started == rec.ended == []
        async with async_raw_connect(pairing, brain.port) as ws:
            await ws.send(encode_message(hello()))
            welcome = parse_brain_message(await asyncio.wait_for(ws.recv(), 3))
            assert isinstance(welcome, Welcome) and welcome.seq == 0
            # Send inside the 200 ms liveness window before waiting on callbacks.
            await ws.send(encode_message(control("brain", welcome.session_id, 1)))
            await asyncio.to_thread(rec.wait_for, lambda: rec.messages)
            assert rec.messages[0].seq == 1

    asyncio.run(scenario())
    brain.close()
    assert brain.protocol_errors == (0 if fault == "timeout" else 1)
    assert len(rec.started) == len(rec.ended) == 1


class HandshakeSocket:
    """Application I/O gates, without changing TLS/HTTP open_timeout."""

    def __init__(self, side, *, send_delay=0, receive_delay=0, block_send=False,
                 error_at=None, error=None):
        self.side = side
        self.send_delay = send_delay
        self.receive_delay = receive_delay
        self.block_send = block_send
        self.error_at = error_at
        self.error = error
        self.closed = []
        self.cancelled = False
        self.sent = []
        self.received = 0

    async def send(self, data):
        if self.error_at == "send":
            raise self.error
        if self.block_send:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        await asyncio.sleep(self.send_delay)
        self.sent.append(data)

    async def recv(self):
        self.received += 1
        if self.error_at == "receive":
            raise self.error
        await asyncio.sleep(self.receive_delay)
        return encode_message(handshake(self.side))

    async def close(self, code):
        self.closed.append(code)
        # Cleanup is outside the application handshake budget.
        await asyncio.sleep(0.02)


async def exercise_handshake(pairing, monkeypatch, side, ws, rec, *, timeout_s=0.12):
    if side == "brain":
        endpoint = BrainServer(
            host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
            body_id=BODY_ID, heartbeat_interval_ms=100,
            liveness_timeout_ms=int(timeout_s * 1000), **rec.callbacks(),
        )
    else:
        endpoint = BodyClient(
            host="127.0.0.1", port=1, ssl_context=body_ssl_context(pairing.body_dir),
            body_id=BODY_ID, output_sample_rates=(44100,),
            handshake_timeout_ms=int(timeout_s * 1000), **rec.callbacks(),
        )

        @asynccontextmanager
        async def connection(*args, **kwargs):
            assert kwargs["open_timeout"] == timeout_s
            assert kwargs["server_hostname"] == BRAIN_SERVER_NAME
            yield ws

        monkeypatch.setattr("companion.bodylink.body.connect", connection)

    async def session_started(session):
        # Successful handshakes can be inspected without starting the session pump.
        pass

    monkeypatch.setattr(endpoint, "_run_session", session_started)
    endpoint._stop = asyncio.Event()
    result = await (endpoint._handle(ws) if side == "brain" else endpoint._connect_once())
    return endpoint, result


@pytest.mark.parametrize("side", SIDES)
def test_blocked_handshake_send_times_out_each_endpoint(pairing, monkeypatch, side):
    async def scenario():
        rec = Recorder()
        ws = HandshakeSocket(side, block_send=True)
        before = asyncio.all_tasks()
        endpoint, result = await asyncio.wait_for(
            exercise_handshake(pairing, monkeypatch, side, ws, rec), 0.6
        )
        assert result is (None if side == "brain" else False)
        assert ws.cancelled and ws.closed == [1000]
        assert rec.started == rec.ended == []
        assert endpoint._session is None and endpoint.protocol_errors == 0
        if side == "brain":
            assert not endpoint._busy
        assert asyncio.all_tasks() == before

    asyncio.run(scenario())


@pytest.mark.parametrize("side", SIDES)
def test_handshake_send_and_receive_share_one_deadline(pairing, monkeypatch, side):
    async def scenario():
        rec = Recorder()
        ws = HandshakeSocket(side, send_delay=0.08, receive_delay=0.08)
        started = asyncio.get_running_loop().time()
        endpoint, result = await exercise_handshake(pairing, monkeypatch, side, ws, rec)
        elapsed = asyncio.get_running_loop().time() - started
        assert 0.12 <= elapsed < 0.18
        assert result is (None if side == "brain" else False)
        assert ws.closed == [1000]
        assert rec.started == rec.ended == []
        assert endpoint._session is None and endpoint.protocol_errors == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("stage", ["send", "receive"])
@pytest.mark.parametrize("code", [1000, 1002, 1009])
def test_handshake_connection_close_is_counted_once(pairing, monkeypatch, side, stage, code):
    async def scenario():
        rec = Recorder()
        cls = ConnectionClosedOK if code == 1000 else ConnectionClosedError
        ws = HandshakeSocket(side, error_at=stage,
                             error=cls(Close(code, ""), Close(code, ""), True))
        endpoint, result = await exercise_handshake(pairing, monkeypatch, side, ws, rec)
        assert result is (None if side == "brain" else False)
        assert rec.started == rec.ended == []
        assert endpoint.protocol_errors == (0 if code == 1000 else 1)
        assert endpoint._session is None
        if side == "brain":
            assert not endpoint._busy

    asyncio.run(scenario())


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("kind", ["binary", "wrong_type", "parse_error"])
def test_invalid_handshake_closes_each_endpoint(pairing, monkeypatch, side, kind):
    class InvalidSocket(HandshakeSocket):
        async def recv(self):
            return {
                "binary": b"not text", "wrong_type": encode_message(heartbeat(side, SID, 0)),
                "parse_error": "{",
            }[kind]

    async def scenario():
        rec = Recorder()
        ws = InvalidSocket(side)
        endpoint, result = await exercise_handshake(pairing, monkeypatch, side, ws, rec)
        assert result is (None if side == "brain" else False)
        assert ws.closed == [1002]
        assert rec.started == rec.ended == []
        assert endpoint._session is None and endpoint.protocol_errors == 1
        if side == "brain":
            assert not endpoint._busy

    asyncio.run(scenario())


def test_backoff_caps_and_resets_after_session(pairing, monkeypatch):
    async def scenario():
        rec = Recorder()
        body = BodyClient(
            host="127.0.0.1", port=1, ssl_context=body_ssl_context(pairing.body_dir),
            body_id=BODY_ID, output_sample_rates=(44100,), **rec.callbacks(),
        )
        body._stop = asyncio.Event()
        outcomes = iter([False, False, False, False, False, True, False])
        waits = []

        async def attempt():
            return next(outcomes)

        class Timeout:
            def __init__(self, delay):
                waits.append(delay)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                if len(waits) == 7:
                    body._stop.set()
                assert exc_type is TimeoutError
                return False

        async def wait():
            raise TimeoutError

        monkeypatch.setattr(body, "_connect_once", attempt)
        monkeypatch.setattr(body._stop, "wait", wait)
        monkeypatch.setattr("companion.bodylink.body.asyncio.timeout", Timeout)
        await body._main()
        assert waits == [1, 2, 4, 8, 10, 1, 2]

    asyncio.run(scenario())


@pytest.fixture(scope="session")
def tls_materials(tmp_path_factory):
    """Independent synthetic CA with valid, wrong-SAN, and fixed-expiry leaves."""
    root = tmp_path_factory.mktemp("transport-failure-tls")
    (root / "newcerts").mkdir()
    (root / "index").write_text("", encoding="ascii")
    (root / "serial").write_text("01\n", encoding="ascii")
    config = root / "openssl.cnf"
    config.write_text(
        f"""[req]
distinguished_name = dn
[dn]
[ca]
default_ca = local
[local]
database = {root / 'index'}
serial = {root / 'serial'}
new_certs_dir = {root / 'newcerts'}
certificate = {root / 'ca.crt'}
private_key = {root / 'ca.key'}
default_md = sha256
default_days = 3650
policy = policy
unique_subject = no
[policy]
commonName = supplied
[ca_ext]
basicConstraints = critical,CA:TRUE,pathlen:0
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
[brain]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = serverAuth
subjectAltName = DNS:{BRAIN_SERVER_NAME}
[wrong_san]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = serverAuth
subjectAltName = DNS:wrong-brain
[body]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = clientAuth
""", encoding="ascii",
    )

    def run(*args):
        return subprocess.run(["openssl", *map(str, args)], check=True,
                              capture_output=True, text=True)

    def key(path):
        run("ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", path)

    key(root / "ca.key")
    run("req", "-new", "-x509", "-sha256", "-config", config, "-key", root / "ca.key",
        "-subj", "/CN=synthetic-failure-ca", "-days", "3650", "-extensions", "ca_ext",
        "-out", root / "ca.crt")
    dirs = {}
    for variant, name, extension, expired in [
        ("valid_brain", "brain", "brain", False),
        ("wrong_san", "brain", "wrong_san", False),
        ("expired_brain", "brain", "brain", True),
        ("valid_body", "body", "body", False),
        ("expired_body", "body", "body", True),
    ]:
        target = root / variant
        target.mkdir()
        dirs[variant] = target
        key(target / f"{name}.key")
        cn = BRAIN_SERVER_NAME if name == "brain" else BODY_ID
        run("req", "-new", "-sha256", "-config", config, "-key", target / f"{name}.key",
            "-subj", f"/CN={cn}", "-out", target / f"{name}.csr")
        dates = ["-startdate", "20000101000000Z", "-enddate", "20000102000000Z"] if expired else []
        run("ca", "-batch", "-notext", "-config", config, "-extensions", extension,
            "-in", target / f"{name}.csr", "-out", target / f"{name}.crt", *dates)
        shutil.copyfile(root / "ca.crt", target / "ca.crt")
        if expired:
            result = run("x509", "-in", target / f"{name}.crt", "-noout", "-dates")
            assert result.stdout.splitlines() == [
                "notBefore=Jan  1 00:00:00 2000 GMT", "notAfter=Jan  2 00:00:00 2000 GMT",
            ]
    return dirs


def unstarted_body(context, port, rec):
    return BodyClient(
        host="127.0.0.1", port=port, ssl_context=context, body_id=BODY_ID,
        output_sample_rates=(44100,), **rec.callbacks(),
    )


def test_handshake_message_preserves_parser_error_without_side_effects(pairing, monkeypatch):
    async def scenario():
        rec = Recorder()
        body = unstarted_body(body_ssl_context(pairing.body_dir), 1, rec)
        error = ValueError("synthetic parser error")

        def parse(data):
            raise error

        monkeypatch.setattr(body, "_parse", parse)
        ws = HandshakeSocket("body")
        with pytest.raises(ValueError) as info:
            await body._handshake_message(ws)
        assert info.value is error
        assert ws.received == 1 and ws.sent == ws.closed == []
        assert rec.started == rec.ended == [] and body.protocol_errors == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["foreign_ca", "wrong_san", "expired_brain"])
def test_body_rejects_wrong_server_identity(tls_materials, foreign_pairing, fault):
    server_dir = (foreign_pairing.brain_dir if fault == "foreign_ca" else tls_materials[fault])
    context = body_ssl_context(tls_materials["valid_body"])
    server_context = brain_ssl_context(server_dir)
    # Isolate rejection of the server: even the foreign server trusts this synthetic body.
    server_context.load_verify_locations(cafile=str(tls_materials["valid_body"] / "ca.crt"))
    with raw_server(server_context) as (port, peers):
        rec = Recorder()
        body = unstarted_body(context, port, rec)
        assert asyncio.run(body._connect_once()) is False
        assert peers.empty()  # No HTTP upgrade or application handshake.
        assert rec.started == rec.ended == []
        assert body.protocol_errors == 0
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert context.minimum_version == ssl.TLSVersion.TLSv1_3


def test_brain_rejects_expired_client(tls_materials, closer):
    paths = PairingPaths(tls_materials["valid_brain"], tls_materials["expired_body"])
    rec = Recorder()
    brain = make_brain(paths, rec, closer)
    with pytest.raises((InvalidHandshake, OSError)), raw_connect(paths, brain.port) as ws:
        raw_hello(ws)
    assert rec.started == rec.ended == []
    assert brain.protocol_errors == 0
    valid = PairingPaths(paths.brain_dir, tls_materials["valid_body"])
    with raw_connect(valid, brain.port) as ws:
        raw_hello(ws)
        rec.wait_for(lambda: rec.started)
    brain.close()
    assert len(rec.started) == len(rec.ended) == 1


def test_tls_failures_recover_without_disabling_verification(tls_materials):
    async def scenario():
        context = body_ssl_context(tls_materials["valid_body"])
        rec = Recorder()
        body = None
        port = 0
        for fault in ["wrong_san", "expired_brain", "wrong_san", "expired_brain", "valid_brain"]:
            with raw_server(brain_ssl_context(tls_materials[fault]), port=port) as (port, peers):
                if body is None:
                    body = unstarted_body(context, port, rec)
                    body._stop = asyncio.Event()
                if fault != "valid_brain":
                    assert await body._connect_once() is False
                    assert peers.empty()
                    assert rec.started == rec.ended == []
                else:
                    attempt = asyncio.create_task(body._connect_once())
                    try:
                        ws = await asyncio.to_thread(peers.get, timeout=3)
                        received = parse_body_message(await asyncio.to_thread(ws.recv, timeout=3))
                        assert received == hello()
                        ws.send(encode_message(welcome()))
                        await asyncio.to_thread(rec.wait_for, lambda: rec.started)
                        await asyncio.to_thread(ws.close)
                        assert await asyncio.wait_for(attempt, 3) is True
                    finally:
                        if not attempt.done():
                            attempt.cancel()
                        await asyncio.gather(attempt, return_exceptions=True)
                assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
                assert context.minimum_version == ssl.TLSVersion.TLSv1_3
        assert body.protocol_errors == 0
        assert [item[:2] for item in rec.ended] == [(SID, "peer_closed")]
        assert len(rec.started) == 1

    asyncio.run(scenario())
