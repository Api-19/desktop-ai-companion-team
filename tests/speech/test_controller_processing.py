from unittest.mock import Mock, call

import pytest

from companion.speech.controller_models import SpeechState
from tests.speech.controller_fakes import ClockFake
from tests.speech.test_controller_events import make_controller


def test_manual_stop_notifies_before_recognition() -> None:
    states: list[SpeechState] = []

    def observer() -> None:
        recorder.calls.append("observer")
        states.append(controller.state)

    controller, recorder, recognizer, _, _, _, _, _ = make_controller(
        processing_observer=observer
    )
    recognizer.configure_callback(
        "recognizer.recognize", lambda: recorder.calls.append("recognize")
    )

    controller.button_pressed()
    controller.button_pressed()

    assert recorder.calls == [
        "start", "stop", "session.close", "observer", "recognize", "audio.close"
    ]
    assert states == [SpeechState.RECOGNIZING]
    assert controller.state is SpeechState.PLAYING


def test_recording_limit_notifies_before_recognition() -> None:
    states: list[SpeechState] = []

    def observer() -> None:
        recorder.calls.append("observer")
        states.append(controller.state)

    clock = ClockFake(now=100.0)
    controller, recorder, recognizer, _, _, _, _, _ = make_controller(
        processing_observer=observer, clock=clock
    )
    recognizer.configure_callback(
        "recognizer.recognize", lambda: recorder.calls.append("recognize")
    )

    controller.button_pressed()
    clock.now = 130.0
    controller.tick()

    assert recorder.calls == [
        "start", "stop", "session.close", "observer", "recognize", "audio.close"
    ]
    assert states == [SpeechState.RECOGNIZING]
    assert controller.state is SpeechState.PLAYING


@pytest.mark.parametrize(
    "scenario",
    ["muted", "stop_error", "invalid_audio", "close_error", "duplicate_finish", "idle"],
)
def test_aborted_failed_or_duplicate_finish_does_not_notify(scenario: str) -> None:
    observer = Mock()
    controller, recorder, recognizer, _, _, player, mute_gate, _ = make_controller(
        processing_observer=observer
    )
    if scenario == "idle":
        controller._finish_recording()
    elif scenario == "duplicate_finish":
        controller.button_pressed()
        controller.button_pressed()
        observer.assert_called_once_with()
        controller.playback_finished(player.tokens[-1])
        observer.reset_mock()
        controller._finish_recording()
        controller._finish_recording()
    else:
        controller.button_pressed()
        if scenario == "muted":
            mute_gate.enabled = False
            controller.mute_changed()
        else:
            if scenario == "stop_error":
                recorder.configure_failure("session.stop", OSError("synthetic stop failure"))
            elif scenario == "invalid_audio":
                recorder.configure_result("session.stop", object())
            else:
                recorder.configure_failure("session.close", OSError("synthetic close failure"))
            controller.button_pressed()

    observer.assert_not_called()
    assert recognizer.calls == int(scenario == "duplicate_finish")
    assert controller.state is SpeechState.IDLE


@pytest.mark.parametrize("error_type", [TimeoutError, OSError, RuntimeError])
def test_processing_observer_failure_keeps_playback(error_type: type[Exception]) -> None:
    observer = Mock(side_effect=error_type("synthetic observer failure"))
    controller, recorder, recognizer, dialogue, synthesizer, player, _, _ = make_controller(
        processing_observer=observer
    )

    for turn in range(1, 3):
        controller.button_pressed()
        controller.button_pressed()
        assert controller.state is SpeechState.PLAYING
        assert controller.last_issue is None
        assert player.tokens == list(range(1, turn + 1))
        controller.playback_finished(player.tokens[-1])
        assert controller.state is SpeechState.IDLE
        assert recorder.live_session_count == 0
        assert recorder.live_recorded_audio_count == 0
        assert synthesizer.live_audio_count == 0

    assert observer.call_args_list == [call(), call()]
    assert recognizer.calls == 2
    assert len(dialogue.calls) == 2
    assert synthesizer.inputs == ["synthetic safe response"] * 2
    assert recorder.session_close_calls == 2
    assert recorder.recorded_audio_close_calls == 2
    assert synthesizer.audio_close_calls == 2
    controller.shutdown()


def test_processing_observer_receives_no_payload() -> None:
    observer = Mock()
    controller, *_ = make_controller(processing_observer=observer)

    controller.button_pressed()
    controller.button_pressed()

    assert observer.mock_calls == [call()]


def test_processing_observer_exception_is_not_logged(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    body = "SYNTHETIC_PROCESSING_EXCEPTION_BODY"
    observer = Mock(side_effect=RuntimeError(body))
    controller, *_ = make_controller(processing_observer=observer)

    controller.button_pressed()
    controller.button_pressed()

    observer.assert_called_once_with()
    assert controller.state is SpeechState.PLAYING
    captured = capsys.readouterr()
    assert body not in captured.out
    assert body not in captured.err
    assert body not in caplog.text


def test_recognition_failure_does_not_repeat_processing_notification() -> None:
    observer = Mock()
    controller, recorder, recognizer, _, synthesizer, player, _, _ = make_controller(
        processing_observer=observer
    )
    recognizer.failure = RuntimeError("synthetic recognition failure")

    controller.button_pressed()
    controller.button_pressed()

    observer.assert_called_once_with()
    assert recognizer.calls == 1
    assert controller.state is SpeechState.PLAYING
    controller.playback_finished(player.tokens[-1])
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0
