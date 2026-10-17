# ruff: noqa: RUF001

from __future__ import annotations

from contextlib import suppress
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from companion.contracts.robot import RobotEvent
from companion.expression import Face, HeadMotion
from companion.speech.controller_models import SpeechControllerIssue, SpeechState

if TYPE_CHECKING:
    from companion.speech.controller import SpeechController

_STATE_EVENTS = MappingProxyType(
    {
        SpeechState.IDLE: "idle",
        SpeechState.RECORDING: "listening",
        SpeechState.RECOGNIZING: "thinking",
        SpeechState.THINKING: "thinking",
        SpeechState.SYNTHESIZING: "thinking",
        SpeechState.PLAYING: "speaking",
    }
)

FACE_TEXT = MappingProxyType(
    {
        Face.NEUTRAL: "表情：平静 (-_-)",
        Face.ATTENTIVE: "表情：在听 (o_o)",
        Face.THINKING: "表情：思考 (._.)?",
        Face.TALKING: "表情：说话 (^o^)",
        Face.MUTED: "表情：静音 (-x-)",
        Face.TROUBLED: "表情：为难 (>_<)",
    }
)


class _MuteGate(Protocol):
    def microphone_enabled(self) -> bool: ...


class _EventSink(Protocol):
    def handle(self, event: RobotEvent) -> object: ...


class _ShowStatePort(Protocol):
    def show_state(self, event: RobotEvent) -> None: ...


def robot_event_for(state: object, issue: object, *, microphone_enabled: bool) -> RobotEvent:
    if not microphone_enabled:
        return RobotEvent(state="muted")
    if state is SpeechState.IDLE and type(issue) is SpeechControllerIssue:
        return RobotEvent(state="error")
    return RobotEvent(state=_STATE_EVENTS[state])  # type: ignore[index]


class TerminalFaceDisplay:
    def show(self, face: Face) -> None:
        print(FACE_TEXT[face], flush=True)


class NullHeadServo:
    def perform(self, motion: HeadMotion) -> None:
        pass

    def disable(self) -> None:
        pass


class BodyDisplayPort:
    """Event sink that shows the expression on the robot body: the body runs its own
    ExpressionController, the brain only sends the RobotEvent (ShowState)."""

    def __init__(self, body: _ShowStatePort) -> None:
        self._body = body

    def handle(self, event: RobotEvent) -> None:
        self._body.show_state(event)


class SpeechExpressionDriver:
    """Feeds speech controller state into the expression state machine (a local
    ExpressionController or a BodyDisplayPort); never raises."""

    def __init__(self, expression: _EventSink, mute_gate: _MuteGate) -> None:
        self._expression = expression
        self._mute_gate = mute_gate

    def observe(self, controller: SpeechController) -> None:
        with suppress(Exception):
            event = robot_event_for(
                controller.state,
                controller.last_issue,
                microphone_enabled=self._mute_gate.microphone_enabled(),
            )
            self._expression.handle(event)

    def processing(self) -> None:
        # Project processing state before recognition starts; mute takes priority.
        with suppress(Exception):
            enabled = self._mute_gate.microphone_enabled()
            self._expression.handle(RobotEvent(state="thinking" if enabled else "muted"))
