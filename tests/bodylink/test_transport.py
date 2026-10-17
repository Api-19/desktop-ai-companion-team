import logging
import threading
import time
from collections.abc import Callable, Iterator

import pytest
from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.sync.client import ClientConnection, connect
from websockets.sync.server import serve

from companion.bodylink.body import BodyClient
from companion.bodylink.brain import BrainServer
from companion.bodylink.pairing import (
    BRAIN_SERVER_NAME,
    PairingPaths,
    body_ssl_context,
    brain_ssl_context,
)
from companion.bodylink.transport import NotConnectedError
from companion.contracts.body import (
    AudioFrame,
    BodyHeartbeat,
    ButtonPressed,
    Hello,
    PlayAudio,
    StartRecording,
    Welcome,
    encode_audio_frame,
    encode_message,
    parse_body_message,
    parse_brain_message,
)

from .conftest import BODY_ID, Recorder

FAST = {"heartbeat_interval_ms": 100, "liveness_timeout_ms": 300}
FAST_RECONNECT = {"reconnect_initial_s": 0.05, "reconnect_max_s": 0.2}


def make_brain(pairing: PairingPaths, rec: Recorder, closer, *, body_id=BODY_ID, port=0, **kw):
    brain = BrainServer(
        host="127.0.0.1",
        port=port,
        ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=body_id,
        **rec.callbacks(),
        **kw,
    )
    closer(brain)
    brain.start()
    return brain


def make_body(pairing: PairingPaths, rec: Recorder, closer, port: int, **kw):
    body = BodyClient(
        host="127.0.0.1",
        port=port,
        ssl_context=body_ssl_context(pairing.body_dir),
        body_id=BODY_ID,
        output_sample_rates=(44100,),
        **rec.callbacks(),
        **kw,
    )
    closer(body)
    body.start()
    return body


def raw_connect(pairing: PairingPaths, port: int) -> ClientConnection:
    return connect(
        f"wss://127.0.0.1:{port}/",
        ssl=body_ssl_context(pairing.body_dir),
        server_hostname=BRAIN_SERVER_NAME,
        proxy=None,
        open_timeout=3,
        close_timeout=1,
    )


def raw_hello(ws: ClientConnection) -> Welcome:
    hello = Hello(
        protocol_version=1, seq=0, body_id=BODY_ID, input_sample_rate=16000,
        output_sample_rates=(44100,),
    )  # fmt: skip
    ws.send(encode_message(hello))
    welcome = parse_brain_message(ws.recv(timeout=3))
    assert isinstance(welcome, Welcome) and welcome.seq == 0
    return welcome


def wait_closed_code(ws: ClientConnection, timeout: float = 3.0) -> int:
    deadline = time.monotonic() + timeout
    with pytest.raises(ConnectionClosed) as info:
        while True:
            ws.recv(timeout=max(0.01, deadline - time.monotonic()))
    return info.value.rcvd.code


@pytest.fixture
def linked(pairing, closer) -> Iterator[Callable]:
    def build(**brain_kw):
        brain_rec, body_rec = Recorder(), Recorder()
        brain = make_brain(pairing, brain_rec, closer, **brain_kw)
        body = make_body(pairing, body_rec, closer, brain.port, **FAST_RECONNECT)
        brain_rec.wait_for(lambda: brain_rec.started)
        body_rec.wait_for(lambda: body_rec.started)
        return brain, brain_rec, body, body_rec

    yield build


def test_handshake_assigns_matching_session_id(linked) -> None:
    _, brain_rec, _, body_rec = linked()
    assert brain_rec.started[0][0] == body_rec.started[0][0]
    hello = brain_rec.hellos[0]
    assert hello.body_id == BODY_ID and hello.output_sample_rates == (44100,)


def test_control_messages_round_trip(linked) -> None:
    brain, brain_rec, body, body_rec = linked()
    sid = brain_rec.started[0][0]
    body.send_message(ButtonPressed(protocol_version=1, seq=0, session_id=sid, observed_at_ms=7))
    brain.send_message(
        StartRecording(protocol_version=1, seq=0, session_id=sid, turn_id=3, max_seconds=60)
    )
    brain_rec.wait_for(lambda: brain_rec.messages)
    body_rec.wait_for(lambda: body_rec.messages)
    assert isinstance(brain_rec.messages[0], ButtonPressed)
    assert brain_rec.messages[0].observed_at_ms == 7
    assert isinstance(body_rec.messages[0], StartRecording)
    assert body_rec.messages[0].turn_id == 3


def test_audio_frames_round_trip_with_last_frame(linked) -> None:
    brain, brain_rec, body, body_rec = linked()
    sid = brain_rec.started[0][0]
    up = [
        AudioFrame("uplink", 3, 0, False, b"\x01\x00" * 16000),
        AudioFrame("uplink", 3, 1, True, b""),
    ]
    down = [
        AudioFrame("downlink", 9, 0, False, b"\x02\x00" * 100),
        AudioFrame("downlink", 9, 1, True, b"\x03\x00"),
    ]
    for frame in up:
        body.send_frame(sid, frame)
    brain.send_message(
        PlayAudio(protocol_version=1, seq=0, session_id=sid, playback_token=9, sample_rate=44100,
                  channels=1, sample_format="pcm_s16le", total_bytes=202)
    )  # fmt: skip
    for frame in down:
        brain.send_frame(sid, frame)
    brain_rec.wait_for(lambda: len(brain_rec.frames) == 2)
    body_rec.wait_for(lambda: len(body_rec.frames) == 2)
    assert brain_rec.frames == up
    assert body_rec.frames == down
    assert isinstance(body_rec.messages[0], PlayAudio)


def test_sending_requires_current_session(linked) -> None:
    brain, _, body, _ = linked()
    with pytest.raises(NotConnectedError):
        brain.send_message(
            StartRecording(protocol_version=1, seq=0, session_id="stale", turn_id=1, max_seconds=5)
        )
    with pytest.raises(NotConnectedError):
        body.send_frame("stale", AudioFrame("uplink", 1, 0, True, b""))


def test_heartbeats_keep_quiet_link_alive(linked) -> None:
    _, brain_rec, _, body_rec = linked(**FAST)
    time.sleep(1.5)  # 5 liveness windows at 100/300 ms
    assert brain_rec.ended == [] and body_rec.ended == []
    assert len(brain_rec.started) == 1 and len(body_rec.started) == 1


def test_brain_detects_silent_body_within_three_seconds(pairing, closer) -> None:
    rec = Recorder()
    brain = make_brain(pairing, rec, closer)  # default 1000 / 3000 ms
    with raw_connect(pairing, brain.port) as ws:
        welcome = raw_hello(ws)
        assert (welcome.heartbeat_interval_ms, welcome.liveness_timeout_ms) == (1000, 3000)
        sent_at = time.monotonic()
        rec.wait_for(lambda: rec.ended, timeout=5)
        session_id, reason, ended_at = rec.ended[0]
        assert session_id == welcome.session_id and reason == "liveness_timeout"
        assert 2.5 <= ended_at - sent_at <= 3.3


def test_body_detects_silent_brain_and_reconnects(pairing, closer) -> None:
    connections: list[str] = []

    def silent_brain(ws) -> None:
        parse_body_message(ws.recv())
        session_id = f"s{len(connections)}"
        connections.append(session_id)
        ws.send(encode_message(Welcome(protocol_version=1, seq=0, session_id=session_id, **FAST)))
        with pytest.raises(ConnectionClosed):
            while True:
                ws.recv()

    server = serve(
        silent_brain, "127.0.0.1", 0, ssl=brain_ssl_context(pairing.brain_dir), close_timeout=1
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        rec = Recorder()
        port = server.socket.getsockname()[1]
        body = make_body(pairing, rec, closer, port, **FAST_RECONNECT)
        rec.wait_for(lambda: len(rec.started) >= 2, timeout=5)
        body.close()
        first_started = rec.started[0][1]
        session_id, reason, ended_at = rec.ended[0]
        assert session_id == "s0" and reason == "liveness_timeout"
        assert 0.25 <= ended_at - first_started <= 1.0
        assert rec.started[1][0] == "s1"
    finally:
        server.shutdown()
        thread.join()


def test_client_from_foreign_ca_is_rejected(pairing, foreign_pairing, closer) -> None:
    rec = Recorder()
    brain = make_brain(pairing, rec, closer)
    with pytest.raises((InvalidHandshake, OSError)), raw_connect(foreign_pairing, brain.port) as ws:
        raw_hello(ws)
    assert rec.started == []


def test_client_with_wrong_common_name_is_rejected(pairing, closer) -> None:
    rec = Recorder()
    brain = make_brain(pairing, rec, closer, body_id="another-body")
    with pytest.raises(InvalidHandshake), raw_connect(pairing, brain.port):
        pass
    assert rec.started == []


def test_second_body_is_rejected_and_first_kept(linked, pairing) -> None:
    brain, brain_rec, body, _ = linked()
    with raw_connect(pairing, brain.port) as intruder:
        assert wait_closed_code(intruder) == 1013
    sid = brain_rec.started[0][0]
    body.send_message(ButtonPressed(protocol_version=1, seq=0, session_id=sid, observed_at_ms=1))
    brain_rec.wait_for(lambda: brain_rec.messages)
    assert brain_rec.ended == [] and len(brain_rec.started) == 1


BAD_INPUTS = {
    "oversized_control": "x" * 5000,
    "unparsable_control": "{not json",
    "wrong_direction_control": None,  # filled in test: a brain message sent by the body
    "oversized_frame": b"",  # filled in test
    "corrupt_frame": b"XXXX" + bytes(16),
    "wrong_direction_frame": encode_audio_frame(AudioFrame("downlink", 1, 0, True, b"")),
}


@pytest.mark.parametrize("kind", sorted(BAD_INPUTS))
def test_bad_input_disconnects_with_protocol_error(pairing, closer, kind) -> None:
    rec = Recorder()
    brain = make_brain(pairing, rec, closer)
    with raw_connect(pairing, brain.port) as ws:
        welcome = raw_hello(ws)
        sid = welcome.session_id
        payload = BAD_INPUTS[kind]
        if kind == "wrong_direction_control":
            payload = encode_message(
                StartRecording(protocol_version=1, seq=2, session_id=sid, turn_id=1, max_seconds=5)
            )
        elif kind == "oversized_frame":
            header = encode_audio_frame(AudioFrame("uplink", 1, 0, False, b""))[:16]
            payload = header + (32002).to_bytes(4, "little") + bytes(32002)
        ws.send(encode_message(BodyHeartbeat(protocol_version=1, seq=1, session_id=sid)))
        ws.send(payload)
        assert wait_closed_code(ws) == (1009 if kind == "oversized_frame" else 1002)
    rec.wait_for(lambda: rec.ended)
    assert rec.ended[0][:2] == (sid, "protocol_error")
    assert brain.protocol_errors == 1


def test_body_reconnects_with_new_session_after_brain_restart(pairing, closer) -> None:
    brain_rec, body_rec = Recorder(), Recorder()
    brain = make_brain(pairing, brain_rec, closer)
    port = brain.port
    body = make_body(pairing, body_rec, closer, port, **FAST_RECONNECT)
    body_rec.wait_for(lambda: body_rec.started)
    first = body_rec.started[0][0]
    brain.close()
    body_rec.wait_for(lambda: body_rec.ended)
    assert body_rec.ended[0][:2] == (first, "peer_closed")
    with pytest.raises(NotConnectedError):
        body.send_message(
            ButtonPressed(protocol_version=1, seq=0, session_id=first, observed_at_ms=1)
        )

    second_rec = Recorder()
    make_brain(pairing, second_rec, closer, port=port)
    body_rec.wait_for(lambda: len(body_rec.started) == 2, timeout=5)
    second = body_rec.started[1][0]
    assert second != first
    assert second_rec.started[0][0] == second


def test_close_leaves_no_threads(pairing) -> None:
    before = set(threading.enumerate())
    brain_rec, body_rec = Recorder(), Recorder()
    brain = BrainServer(
        host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=BODY_ID, **brain_rec.callbacks(),
    )  # fmt: skip
    brain.start()
    body = BodyClient(
        host="127.0.0.1", port=brain.port, ssl_context=body_ssl_context(pairing.body_dir),
        body_id=BODY_ID, output_sample_rates=(44100,), **body_rec.callbacks(),
    )  # fmt: skip
    body.start()
    body_rec.wait_for(lambda: body_rec.started)
    body.close()
    brain.close()
    body.close()  # idempotent
    assert set(threading.enumerate()) <= before
    assert body_rec.ended[0][1] == "closed"
    brain_rec.wait_for(lambda: brain_rec.ended)


def test_message_bodies_and_audio_never_reach_logs(linked, caplog, capfd) -> None:
    caplog.set_level(logging.DEBUG)
    brain, brain_rec, body, _ = linked()
    try:
        sid = brain_rec.started[0][0]
        body.send_message(
            ButtonPressed(protocol_version=1, seq=0, session_id=sid, observed_at_ms=424242)
        )
        body.send_frame(sid, AudioFrame("uplink", 1, 0, True, b"\x7a\x7a" * 8))
        brain_rec.wait_for(lambda: brain_rec.messages and brain_rec.frames)
    finally:
        body.close()
        brain.close()
    assert "424242" not in caplog.text
    assert "button_pressed" not in caplog.text
    assert "zz" not in caplog.text and "7a 7a" not in caplog.text
    stderr = capfd.readouterr().err
    assert "424242" not in stderr
    assert "button_pressed" not in stderr
    assert "zz" not in stderr and "7a 7a" not in stderr
    assert stderr == ""


def test_hello_with_unpaired_body_id_is_a_protocol_error(pairing, closer) -> None:
    rec = Recorder()
    brain = make_brain(pairing, rec, closer)
    with raw_connect(pairing, brain.port) as ws:
        hello = Hello(
            protocol_version=1, seq=0, body_id="someone-else", input_sample_rate=16000,
            output_sample_rates=(44100,),
        )  # fmt: skip
        ws.send(encode_message(hello))
        assert wait_closed_code(ws) == 1002
    assert rec.started == [] and brain.protocol_errors == 1
