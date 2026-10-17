import logging

import pytest

from companion.expression import ExpressionController, Face, HeadMotion
from companion.runtime.speech_expression import SpeechExpressionDriver
from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechState
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    DialogueFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)

_FAULT_CANARY = "synthetic-servo-fault-canary"


class FakeDisplay:
    def __init__(self) -> None:
        self.faces: list[Face] = []

    def show(self, face: Face) -> None:
        self.faces.append(face)


class FakeServo:
    def __init__(self, error_type: type[Exception], *, disable_fails: bool) -> None:
        self.error_type = error_type
        self.disable_fails = disable_fails
        self.motions: list[HeadMotion] = []
        self.perform_calls = 0
        self.disable_calls = 0

    def perform(self, motion: HeadMotion) -> None:
        self.motions.append(motion)
        self.perform_calls += 1
        if self.perform_calls == 1:
            raise self.error_type(f"{_FAULT_CANARY}: perform")

    def disable(self) -> None:
        self.disable_calls += 1
        if self.disable_fails:
            raise self.error_type(f"{_FAULT_CANARY}: disable")


def _assert_resources_released(
    recorder: AudioRecorderFake, synthesizer: SpeechSynthesizerFake
) -> None:
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0


def _assert_two_speech_turns(
    servo: FakeServo,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    recorder = AudioRecorderFake()
    recognizer = SpeechRecognizerFake()
    dialogue = DialogueFake()
    synthesizer = SpeechSynthesizerFake()
    player = AudioPlayerFake()
    mute_gate = MuteGateFake()
    clock = ClockFake()
    display = FakeDisplay()
    expression = ExpressionController(display, servo)
    driver = SpeechExpressionDriver(expression, mute_gate)
    controller = SpeechController(
        trusted_user_id="synthetic-user",
        recorder=recorder,
        recognizer=recognizer,
        dialogue=dialogue,
        synthesizer=synthesizer,
        player=player,
        mute_gate=mute_gate,
        clock=clock,
        processing_observer=driver.processing,
    )
    driver.observe(controller)
    assert display.faces == [Face.NEUTRAL]

    for turn in (1, 2):
        face_start = len(display.faces)
        controller.button_pressed()
        driver.observe(controller)
        assert controller.state is SpeechState.RECORDING
        assert display.faces[face_start:] == [Face.ATTENTIVE]
        assert recorder.live_session_count == 1
        assert servo.motions == [HeadMotion.TILT]
        assert servo.perform_calls == 1
        assert servo.disable_calls == 1
        assert expression.servo_disabled is True

        clock.now += 0.5
        controller.tick()
        driver.observe(controller)
        assert controller.state is SpeechState.RECORDING

        controller.button_pressed()
        driver.observe(controller)
        assert controller.state is SpeechState.PLAYING
        assert controller.last_issue is None
        assert recognizer.calls == turn
        assert dialogue.calls == [("synthetic-user", "synthetic transcript")] * turn
        assert synthesizer.inputs == ["synthetic safe response"] * turn
        assert len(player.tokens) == turn
        assert recorder.live_session_count == 0
        assert recorder.live_recorded_audio_count == 0
        assert synthesizer.live_audio_count == 1

        controller.playback_finished(player.tokens[-1])
        driver.observe(controller)
        assert controller.state is SpeechState.IDLE
        assert controller.last_issue is None
        _assert_resources_released(recorder, synthesizer)
        assert recorder.session_close_calls == turn
        assert recorder.recorded_audio_close_calls == turn
        assert synthesizer.audio_close_calls == turn
        assert display.faces[face_start:] == [
            Face.ATTENTIVE,
            Face.THINKING,
            Face.TALKING,
            Face.NEUTRAL,
        ]
        assert servo.perform_calls == 1
        assert servo.disable_calls == 1
        assert expression.servo_disabled is True

    assert player.tokens[0] < player.tokens[1]
    controller.shutdown()
    _assert_resources_released(recorder, synthesizer)
    assert servo.perform_calls == 1
    assert servo.disable_calls == 1
    assert expression.servo_disabled is True
    captured = capsys.readouterr()
    assert _FAULT_CANARY not in captured.out
    assert _FAULT_CANARY not in captured.err
    assert _FAULT_CANARY not in caplog.text


@pytest.mark.parametrize("error_type", [TimeoutError, OSError, RuntimeError])
def test_servo_fault_does_not_stop_two_speech_turns(
    error_type: type[Exception],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _assert_two_speech_turns(FakeServo(error_type, disable_fails=False), capsys, caplog)


@pytest.mark.parametrize("error_type", [TimeoutError, OSError, RuntimeError])
def test_disable_fault_does_not_stop_playback(
    error_type: type[Exception],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _assert_two_speech_turns(FakeServo(error_type, disable_fails=True), capsys, caplog)
