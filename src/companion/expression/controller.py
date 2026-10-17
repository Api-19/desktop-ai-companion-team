from contextlib import suppress
from types import MappingProxyType
from typing import Protocol

from companion.contracts.robot import RobotEvent
from companion.expression.models import ExpressionFrame, Face, HeadMotion


class FaceDisplay(Protocol):
    def show(self, face: Face) -> None: ...


class HeadServo(Protocol):
    def perform(self, motion: HeadMotion) -> None: ...

    def disable(self) -> None: ...


STATE_FRAMES = MappingProxyType(
    {
        "idle": ExpressionFrame(face=Face.NEUTRAL, motion=HeadMotion.NONE),
        "listening": ExpressionFrame(face=Face.ATTENTIVE, motion=HeadMotion.TILT),
        "thinking": ExpressionFrame(face=Face.THINKING, motion=HeadMotion.NONE),
        "speaking": ExpressionFrame(face=Face.TALKING, motion=HeadMotion.NOD),
        "muted": ExpressionFrame(face=Face.MUTED, motion=HeadMotion.NONE),
        "error": ExpressionFrame(face=Face.TROUBLED, motion=HeadMotion.NONE),
    }
)


class ExpressionController:
    def __init__(self, display: FaceDisplay, servo: HeadServo) -> None:
        self._display = display
        self._servo = servo
        self._last_state: str | None = None
        self._servo_disabled = False

    @property
    def servo_disabled(self) -> bool:
        return self._servo_disabled

    def handle(self, event: RobotEvent) -> ExpressionFrame:
        frame = STATE_FRAMES[event.state]
        if event.state == self._last_state:
            return frame
        self._last_state = event.state

        try:
            self._display.show(frame.face)
        except Exception:
            return frame

        if frame.motion != HeadMotion.NONE and not self._servo_disabled:
            try:
                self._servo.perform(frame.motion)
            except Exception:
                self._servo_disabled = True
                with suppress(Exception):
                    self._servo.disable()
        return frame
