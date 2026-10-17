import pytest
from pydantic import ValidationError

from companion.contracts.robot import RobotEvent
from companion.expression import (
    ExpressionController,
    ExpressionFrame,
    Face,
    FaceDisplay,
    HeadMotion,
    HeadServo,
)
from companion.expression.controller import STATE_FRAMES


class FakeDisplay:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls
        self.error: Exception | None = None

    def show(self, face: Face) -> None:
        self.calls.append(("show", face))
        if self.error is not None:
            raise self.error


class FakeServo:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls
        self.perform_error: Exception | None = None
        self.disable_error: Exception | None = None

    def perform(self, motion: HeadMotion) -> None:
        self.calls.append(("perform", motion))
        if self.perform_error is not None:
            raise self.perform_error

    def disable(self) -> None:
        self.calls.append(("disable", None))
        if self.disable_error is not None:
            raise self.disable_error


@pytest.fixture
def ports():
    calls: list[tuple[str, object]] = []
    display = FakeDisplay(calls)
    servo = FakeServo(calls)
    return ExpressionController(display, servo), display, servo, calls


STATES = [
    ("idle", Face.NEUTRAL, HeadMotion.NONE),
    ("listening", Face.ATTENTIVE, HeadMotion.TILT),
    ("thinking", Face.THINKING, HeadMotion.NONE),
    ("speaking", Face.TALKING, HeadMotion.NOD),
    ("muted", Face.MUTED, HeadMotion.NONE),
    ("error", Face.TROUBLED, HeadMotion.NONE),
]


@pytest.mark.parametrize(("previous_state", "previous_face", "previous_motion"), STATES)
@pytest.mark.parametrize(("state", "face", "motion"), STATES)
def test_all_state_pairs_project_latest_state(
    previous_state, previous_face, previous_motion, state, face, motion
):
    calls: list[tuple[str, object]] = []
    controller = ExpressionController(FakeDisplay(calls), FakeServo(calls))

    controller.handle(RobotEvent(state=previous_state, detail="synthetic previous detail"))
    previous_calls = calls.copy()
    frame = controller.handle(RobotEvent(state=state, detail="synthetic latest detail"))

    assert frame == ExpressionFrame(face=face, motion=motion)
    if state == previous_state:
        assert calls == previous_calls
    else:
        expected = [*previous_calls, ("show", face)]
        if motion != HeadMotion.NONE:
            expected.append(("perform", motion))
        assert calls == expected


def test_robot_event_requires_state():
    with pytest.raises(ValidationError) as exc_info:
        RobotEvent()

    assert exc_info.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize(("state", "face", "motion"), STATES)
def test_state_mapping_and_display_before_motion(ports, state, face, motion):
    controller, _, _, calls = ports

    frame = controller.handle(RobotEvent(state=state, detail="ignored"))

    assert frame == ExpressionFrame(face=face, motion=motion)
    expected = [("show", face)]
    if motion != HeadMotion.NONE:
        expected.append(("perform", motion))
    assert calls == expected
    assert controller.servo_disabled is False


@pytest.mark.parametrize(("state", "face", "motion"), STATES)
def test_repeated_state_ignores_detail_and_does_not_repeat_ports(ports, state, face, motion):
    controller, _, _, calls = ports
    first = controller.handle(RobotEvent(state=state, detail="first"))
    original_calls = calls.copy()

    second = controller.handle(RobotEvent(state=state, detail="different"))

    assert second == first == ExpressionFrame(face=face, motion=motion)
    assert calls == original_calls


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, Exception])
def test_servo_failure_disables_once_and_keeps_display_running(ports, error_type):
    controller, _, servo, calls = ports
    servo.perform_error = error_type("servo failed")

    assert controller.handle(RobotEvent(state="speaking")) == ExpressionFrame(
        face=Face.TALKING, motion=HeadMotion.NOD
    )
    assert controller.servo_disabled is True
    controller.handle(RobotEvent(state="listening"))
    controller.handle(RobotEvent(state="speaking"))

    assert calls == [
        ("show", Face.TALKING),
        ("perform", HeadMotion.NOD),
        ("disable", None),
        ("show", Face.ATTENTIVE),
        ("show", Face.TALKING),
    ]


def test_disable_failure_is_swallowed_and_servo_stays_disabled(ports):
    controller, _, servo, calls = ports
    servo.perform_error = RuntimeError("servo failed")
    servo.disable_error = RuntimeError("disable failed")

    assert controller.handle(RobotEvent(state="listening")) == ExpressionFrame(
        face=Face.ATTENTIVE, motion=HeadMotion.TILT
    )
    assert controller.servo_disabled is True
    controller.handle(RobotEvent(state="speaking"))
    assert calls == [
        ("show", Face.ATTENTIVE),
        ("perform", HeadMotion.TILT),
        ("disable", None),
        ("show", Face.TALKING),
    ]


def test_display_failure_skips_servo_and_retries_on_state_change(ports):
    controller, display, _, calls = ports
    display.error = RuntimeError("display failed")

    frame = controller.handle(RobotEvent(state="listening"))

    assert frame == ExpressionFrame(face=Face.ATTENTIVE, motion=HeadMotion.TILT)
    assert controller.servo_disabled is False
    assert calls == [("show", Face.ATTENTIVE)]
    display.error = None
    assert controller.handle(RobotEvent(state="listening")) == frame
    assert calls == [("show", Face.ATTENTIVE)]
    controller.handle(RobotEvent(state="speaking"))
    assert calls == [
        ("show", Face.ATTENTIVE),
        ("show", Face.TALKING),
        ("perform", HeadMotion.NOD),
    ]


def test_frame_and_mapping_are_immutable():
    frame = ExpressionFrame(face=Face.NEUTRAL, motion=HeadMotion.NONE)
    with pytest.raises(ValidationError, match="frozen"):
        frame.face = Face.TALKING
    with pytest.raises(TypeError):
        STATE_FRAMES["idle"] = ExpressionFrame(face=Face.TALKING, motion=HeadMotion.NOD)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExpressionFrame(face=Face.NEUTRAL, motion=HeadMotion.NONE, extra=True)


def test_servo_disabled_is_read_only_and_ports_are_exported(ports):
    controller, _, _, _ = ports
    with pytest.raises(AttributeError):
        controller.servo_disabled = True
    assert FaceDisplay.__name__ == "FaceDisplay"
    assert HeadServo.__name__ == "HeadServo"
