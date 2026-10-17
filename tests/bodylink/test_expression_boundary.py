import ssl
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.bodylink import brain_ports
from companion.contracts.body import Hello, ShowState, encode_message
from companion.contracts.robot import RobotEvent

STATES = ("idle", "listening", "thinking", "speaking", "muted", "error")
CANARY = "synthetic-expression-detail-canary"


@pytest.fixture
def expression_link(monkeypatch):
    server = SimpleNamespace(messages=[])

    def make_server(**kwargs):
        server.callbacks = kwargs
        server.send_message = server.messages.append
        return server

    monkeypatch.setattr(brain_ports, "BrainServer", make_server)
    link = brain_ports.BrainBodyLink(
        host="127.0.0.1",
        port=0,
        ssl_context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        body_id="synthetic-body",
        run_root=Path("unused-expression-run"),
    )
    return link, server


def start_session(server, session_id="session-1"):
    server.callbacks["on_session_started"](
        session_id,
        Hello(
            protocol_version=1,
            seq=0,
            body_id="synthetic-body",
            input_sample_rate=16000,
            output_sample_rates=(44100,),
        ),
    )


@pytest.mark.parametrize("state", STATES)
def test_show_state_strips_detail_canary(expression_link, state) -> None:
    link, server = expression_link
    start_session(server)
    event = RobotEvent(state=state, detail=CANARY)

    link.show_state(event)

    assert len(server.messages) == 1
    message = server.messages[0]
    assert isinstance(message, ShowState)
    assert message.event == RobotEvent(state=state)
    assert message.event.detail is None
    assert CANARY not in encode_message(message)
    assert '"detail":null' in encode_message(message)
    assert event.detail == CANARY


def test_show_state_deduplicates_by_state_only(expression_link) -> None:
    link, server = expression_link
    start_session(server)

    for state in STATES:
        link.show_state(RobotEvent(state=state, detail=CANARY))
        link.show_state(RobotEvent(state=state, detail=f"{CANARY}-changed"))

    assert [message.event.state for message in server.messages] == list(STATES)
    assert all(message.event.detail is None for message in server.messages)


def test_show_state_session_end_discards_expression(expression_link) -> None:
    link, server = expression_link
    start_session(server)
    link.show_state(RobotEvent(state="thinking", detail=CANARY))

    server.callbacks["on_session_ended"]("session-1", "peer_closed")
    link.show_state(RobotEvent(state="listening", detail=f"{CANARY}-disconnected"))
    assert len(server.messages) == 1

    start_session(server, "session-2")
    assert len(server.messages) == 1
    link.show_state(RobotEvent(state="thinking", detail=f"{CANARY}-reconnected"))
    assert [message.session_id for message in server.messages] == ["session-1", "session-2"]
    assert [message.event.state for message in server.messages] == ["thinking", "thinking"]
    assert all(message.event.detail is None for message in server.messages)
