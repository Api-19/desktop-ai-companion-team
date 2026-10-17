# ruff: noqa: RUF001

from __future__ import annotations

from unittest.mock import Mock

import pytest

from companion.contracts.robot import RobotEvent
from companion.expression import ExpressionController, Face, HeadMotion
from companion.runtime import speech_chat
from companion.runtime.speech_expression import (
    BodyDisplayPort,
    NullHeadServo,
    SpeechExpressionDriver,
    TerminalFaceDisplay,
    robot_event_for,
)
from companion.speech.controller_models import SpeechControllerIssue, SpeechState

SEQUENCE = (
    SpeechState.IDLE,
    SpeechState.RECORDING,
    SpeechState.RECOGNIZING,
    SpeechState.THINKING,
    SpeechState.SYNTHESIZING,
    SpeechState.PLAYING,
    SpeechState.IDLE,
)


class RecordingDisplay:
    def __init__(self, *, fail: bool = False) -> None:
        self.faces: list[Face] = []
        self.fail = fail

    def show(self, face: Face) -> None:
        self.faces.append(face)
        if self.fail:
            raise RuntimeError("synthetic display failure")


class Gate:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def microphone_enabled(self) -> bool:
        return self.enabled


class SteppingController:
    """Moves to the next scripted state on every tick."""

    def __init__(self, states: tuple[SpeechState, ...]) -> None:
        self._states = list(states)
        self.state = self._states.pop(0)
        self.last_issue: SpeechControllerIssue | None = None
        self.turn_failure = None
        self.tick_calls = 0

    def tick(self) -> None:
        self.tick_calls += 1
        if self._states:
            self.state = self._states.pop(0)

    def button_pressed(self) -> None:
        raise AssertionError("not used")


class Assembly:
    def __init__(self, controller: object) -> None:
        self.controller = controller
        self.drain_playback_events = Mock()
        self.close = Mock()


def _run(controller: SteppingController, observe: object) -> list[str]:
    statuses: list[str] = []
    keys = iter([None] * (len(SEQUENCE) - 1) + ["q"])
    speech_chat._run_console_loop(
        assembly=Assembly(controller),
        read_key=lambda: next(keys),
        write_status=statuses.append,
        sleep=lambda _: None,
        observe=observe,  # type: ignore[arg-type]
    )
    return statuses


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (SpeechState.IDLE, "idle"),
        (SpeechState.RECORDING, "listening"),
        (SpeechState.RECOGNIZING, "thinking"),
        (SpeechState.THINKING, "thinking"),
        (SpeechState.SYNTHESIZING, "thinking"),
        (SpeechState.PLAYING, "speaking"),
    ],
)
def test_speech_states_map_to_robot_events(state: SpeechState, expected: str) -> None:
    assert robot_event_for(state, None, microphone_enabled=True) == RobotEvent(state=expected)


def test_idle_with_issue_is_error_and_closed_microphone_is_muted() -> None:
    issue = SpeechControllerIssue.RECOGNITION_FAILED
    assert robot_event_for(SpeechState.IDLE, issue, microphone_enabled=True).state == "error"
    assert robot_event_for(SpeechState.PLAYING, issue, microphone_enabled=True).state == "speaking"
    for state in SEQUENCE:
        assert robot_event_for(state, issue, microphone_enabled=False).state == "muted"


def test_state_sequence_drives_faces_without_repeats() -> None:
    display = RecordingDisplay()
    driver = SpeechExpressionDriver(ExpressionController(display, NullHeadServo()), Gate())
    controller = SteppingController(SEQUENCE)

    statuses = _run(controller, driver.observe)

    assert display.faces == [
        Face.NEUTRAL,
        Face.ATTENTIVE,
        Face.THINKING,
        Face.TALKING,
        Face.NEUTRAL,
    ]
    assert statuses[-1] == "speech-chat-stopped"


def test_issue_and_mute_drive_faces() -> None:
    display = RecordingDisplay()
    gate = Gate()
    driver = SpeechExpressionDriver(ExpressionController(display, NullHeadServo()), gate)
    controller = SteppingController((SpeechState.IDLE,))

    driver.observe(controller)
    controller.last_issue = SpeechControllerIssue.DIALOGUE_FAILED
    driver.observe(controller)
    gate.enabled = False
    driver.observe(controller)

    assert display.faces == [Face.NEUTRAL, Face.TROUBLED, Face.MUTED]


def test_processing_shows_thinking_face() -> None:
    display = RecordingDisplay()
    driver = SpeechExpressionDriver(ExpressionController(display, NullHeadServo()), Gate())

    driver.processing()

    assert display.faces == [Face.THINKING]


def test_processing_respects_closed_microphone() -> None:
    body = Mock()
    gate = Gate()
    driver = SpeechExpressionDriver(BodyDisplayPort(body), gate)

    driver.processing()
    gate.enabled = False
    driver.processing()
    gate.enabled = True
    driver.processing()

    events = [call.args[0] for call in body.show_state.call_args_list]
    assert [event.state for event in events] == ["thinking", "muted", "thinking"]
    assert [event.detail for event in events] == [None, None, None]


def test_processing_gate_failure_is_contained(capsys: pytest.CaptureFixture[str]) -> None:
    gate = Mock()
    gate.microphone_enabled.side_effect = RuntimeError("synthetic gate failure")
    sink = Mock()
    driver = SpeechExpressionDriver(sink, gate)

    assert driver.processing() is None
    sink.handle.assert_not_called()
    captured = capsys.readouterr()
    assert "synthetic gate failure" not in captured.out
    assert "synthetic gate failure" not in captured.err


def test_body_display_port_sends_robot_events_to_the_body() -> None:
    body = Mock()
    driver = SpeechExpressionDriver(BodyDisplayPort(body), Gate())
    controller = SteppingController((SpeechState.RECORDING,))

    driver.observe(controller)
    driver.processing()

    assert [c.args[0].state for c in body.show_state.call_args_list] == ["listening", "thinking"]


def test_display_failure_does_not_change_loop_or_speech() -> None:
    plain = SteppingController(SEQUENCE)
    expected = _run(plain, None)
    display = RecordingDisplay(fail=True)
    driver = SpeechExpressionDriver(ExpressionController(display, NullHeadServo()), Gate())
    controller = SteppingController(SEQUENCE)

    assert _run(controller, driver.observe) == expected
    assert controller.tick_calls == plain.tick_calls == len(SEQUENCE)
    assert display.faces  # The failing display was actually exercised.


def test_expression_controller_failure_is_contained() -> None:
    broken = Mock(spec=ExpressionController)
    broken.handle.side_effect = RuntimeError("synthetic expression failure")
    driver = SpeechExpressionDriver(broken, Gate())
    plain = SteppingController(SEQUENCE)
    controller = SteppingController(SEQUENCE)

    assert _run(controller, driver.observe) == _run(plain, None)
    driver.processing()
    # One initial observation, one per loop iteration, one processing cue.
    assert broken.handle.call_count == 1 + len(SEQUENCE) + 1


def test_terminal_display_prints_one_line_per_face(capsys: pytest.CaptureFixture[str]) -> None:
    display = TerminalFaceDisplay()
    for face in Face:
        display.show(face)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(Face)
    assert all(line.startswith("表情：") for line in lines)
    assert len(set(lines)) == len(Face)


def test_null_servo_does_nothing() -> None:
    servo = NullHeadServo()
    assert servo.perform(HeadMotion.NOD) is None
    assert servo.disable() is None
