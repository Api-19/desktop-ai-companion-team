import logging

import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from companion.bodylink.brain import BrainServer
from companion.bodylink.pairing import BRAIN_SERVER_NAME, body_ssl_context
from companion.bodylink.transport import WIRE_LOGGER
from companion.contracts.body import AudioFrame, ButtonPressed, StartRecording

from .conftest import Recorder
from .test_transport import make_body, make_brain, wait_closed_code


def assert_wire_silent(caplog, capfd, *canaries: str) -> None:
    captured = capfd.readouterr()
    for canary in canaries:
        assert canary not in caplog.text
        assert canary not in captured.err
        assert canary not in captured.out
    assert "Traceback" not in caplog.text
    assert captured.err == ""
    assert not [record for record in caplog.records if record.name == WIRE_LOGGER.name]


def test_wire_logger_discards_only_its_own_logs(caplog) -> None:
    assert WIRE_LOGGER.name == "companion.bodylink.wire"
    assert WIRE_LOGGER.level == logging.INFO
    assert len(WIRE_LOGGER.handlers) == 1
    assert type(WIRE_LOGGER.handlers[0]) is logging.NullHandler
    assert WIRE_LOGGER.propagate is False

    caplog.set_level(logging.DEBUG)
    logging.getLogger().warning("synthetic root diagnostic still visible")
    assert "synthetic root diagnostic still visible" in caplog.text


def test_wire_error_traceback_contains_no_canary(caplog, capfd) -> None:
    caplog.set_level(logging.DEBUG)
    try:
        try:
            raise ValueError("wire-inner-canary")
        except ValueError as cause:
            raise RuntimeError("wire-outer-canary") from cause
    except RuntimeError:
        WIRE_LOGGER.error("wire-error-text-canary", exc_info=True)

    assert_wire_silent(
        caplog, capfd, "wire-error-text-canary", "wire-inner-canary", "wire-outer-canary"
    )


def test_wire_handshake_failure_contains_no_header_canary(
    pairing, closer, monkeypatch, caplog, capfd
) -> None:
    caplog.set_level(logging.DEBUG)
    canary = "wire-request-header-canary"
    seen_headers = []

    def fail_request(self, connection, request):
        header = request.headers["X-Synthetic-Canary"]
        seen_headers.append(header)
        raise RuntimeError(header)

    monkeypatch.setattr(BrainServer, "_check_client_name", fail_request)
    brain = make_brain(pairing, Recorder(), closer)
    try:
        with pytest.raises(InvalidStatus) as info, connect(
            f"wss://127.0.0.1:{brain.port}/",
            ssl=body_ssl_context(pairing.body_dir),
            server_hostname=BRAIN_SERVER_NAME,
            proxy=None,
            additional_headers={"X-Synthetic-Canary": canary},
            logger=WIRE_LOGGER,
            open_timeout=3,
            close_timeout=1,
        ):
            pytest.fail("handshake unexpectedly succeeded")
        assert info.value.response.status_code == 500
        assert seen_headers == [canary]
    finally:
        brain.close()

    assert_wire_silent(caplog, capfd, canary)


def test_wire_handler_failure_contains_no_payload_canary(
    pairing, closer, monkeypatch, caplog, capfd
) -> None:
    caplog.set_level(logging.DEBUG)
    canary = "wire-handler-payload-canary"
    seen_payloads = []

    async def fail_handler(self, ws):
        payload = await ws.recv()
        seen_payloads.append(payload)
        raise RuntimeError(payload)

    monkeypatch.setattr(BrainServer, "_serve_body", fail_handler)
    brain = make_brain(pairing, Recorder(), closer)
    try:
        with connect(
            f"wss://127.0.0.1:{brain.port}/",
            ssl=body_ssl_context(pairing.body_dir),
            server_hostname=BRAIN_SERVER_NAME,
            proxy=None,
            logger=WIRE_LOGGER,
            open_timeout=3,
            close_timeout=1,
        ) as ws:
            ws.send(canary)
            assert wait_closed_code(ws) == 1011
        assert seen_payloads == [canary]
    finally:
        brain.close()

    assert_wire_silent(caplog, capfd, canary)


def test_wire_debug_frames_remain_silent(pairing, closer, caplog, capfd) -> None:
    caplog.set_level(logging.DEBUG)
    with caplog.at_level(logging.DEBUG, logger=WIRE_LOGGER.name):
        brain_rec, body_rec = Recorder(), Recorder()
        brain = make_brain(pairing, brain_rec, closer)
        body = make_body(pairing, body_rec, closer, brain.port)
        try:
            brain_rec.wait_for(lambda: brain_rec.started)
            body_rec.wait_for(lambda: body_rec.started)
            sid = brain_rec.started[0][0]
            body.send_message(
                ButtonPressed(protocol_version=1, seq=0, session_id=sid, observed_at_ms=8675309)
            )
            brain.send_message(
                StartRecording(protocol_version=1, seq=0, session_id=sid, turn_id=42, max_seconds=5)
            )
            up = AudioFrame("uplink", 1, 0, True, b"\x7a\x7a" * 8)
            down = AudioFrame("downlink", 42, 0, True, b"\x6b\x6b" * 8)
            body.send_frame(sid, up)
            brain.send_frame(sid, down)
            brain_rec.wait_for(lambda: brain_rec.messages and brain_rec.frames)
            body_rec.wait_for(lambda: body_rec.messages and body_rec.frames)
            assert brain_rec.messages[0].observed_at_ms == 8675309
            assert body_rec.messages[0].turn_id == 42
            assert brain_rec.frames == [up]
            assert body_rec.frames == [down]
        finally:
            body.close()
            brain.close()

        assert_wire_silent(
            caplog, capfd, sid, "8675309", "button_pressed", "start_recording",
            "zz", "7a 7a", "kk", "6b 6b",
        )
