from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechControllerIssue, SpeechState
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    DialogueFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)


def make_turn(**overrides: object) -> SimpleNamespace:
    ports = dict(
        recorder=AudioRecorderFake(),
        recognizer=SpeechRecognizerFake(),
        dialogue=DialogueFake(),
        synthesizer=SpeechSynthesizerFake(),
        player=AudioPlayerFake(),
        mute_gate=MuteGateFake(),
        clock=ClockFake(),
    )
    source = SimpleNamespace(session_id="A")
    starter = Mock(return_value=None)
    rows = []
    controller = SpeechController(
        trusted_user_id="synthetic-owner",
        **ports,
        **{
            "session_id_supplier": lambda: source.session_id,
            "session_playback_starter": starter,
            "timing_observer": rows.append,
            **overrides,
        },
    )
    return SimpleNamespace(
        **ports, controller=controller, source=source, starter=starter, rows=rows
    )


def finish(turn: SimpleNamespace) -> None:
    turn.controller.button_pressed()
    turn.controller.button_pressed()


def invalidate_after(turn: SimpleNamespace, stage: str) -> None:
    adapter, method = {
        "start": (turn.recorder, "recorder.start"),
        "stop": (turn.recorder, "session.stop"),
        "recognize": (turn.recognizer, "recognizer.recognize"),
        "dialogue": (turn.dialogue, "dialogue.reply"),
        "synthesize": (turn.synthesizer, "synthesizer.synthesize"),
    }[stage]
    adapter.configure_callback(method, lambda: setattr(turn.source, "session_id", None))


def assert_discarded(turn: SimpleNamespace) -> None:
    assert turn.controller.state is SpeechState.IDLE
    assert turn.controller.last_issue is None
    assert turn.player.tokens == []
    turn.starter.assert_not_called()
    assert turn.recorder.live_session_count == 0
    assert turn.recorder.live_recorded_audio_count == 0
    assert turn.synthesizer.live_audio_count == 0


@pytest.mark.parametrize("stage", ["stop", "recognize", "dialogue", "synthesize"])
def test_disconnect_after_each_stage_discards_turn(stage: str) -> None:
    turn = make_turn()
    invalidate_after(turn, stage)

    finish(turn)

    assert_discarded(turn)
    counts = [
        turn.recorder.calls.count("stop"), turn.recognizer.calls,
        len(turn.dialogue.calls), len(turn.synthesizer.inputs), turn.starter.call_count,
    ]
    index = ["stop", "recognize", "dialogue", "synthesize"].index(stage)
    assert counts == [1] * (index + 1) + [0] * (4 - index)
    assert len(turn.rows) == 1
    row = turn.rows[0].model_dump()
    assert row["played"] is False
    assert row["issue"] is None
    for later in ["stop", "recognize", "dialogue", "synthesize", "playback_start"][index + 1:]:
        assert row[f"{later}_ms"] is None
    assert set(row) == {
        "played", "issue", "stop_ms", "recognize_ms", "dialogue_ms",
        "synthesize_ms", "playback_start_ms", "total_ms",
    }


def test_reconnect_before_tts_returns_never_plays_old_audio() -> None:
    turn = make_turn()
    transitions = ["A"]

    def reconnect() -> None:
        for session_id in (None, "B"):
            turn.source.session_id = session_id
            transitions.append(session_id)

    turn.synthesizer.configure_callback("synthesizer.synthesize", reconnect)

    finish(turn)

    assert transitions == ["A", None, "B"]
    assert_discarded(turn)
    assert turn.synthesizer.audios[0].close_calls == 1
    assert turn.rows[0].played is False
    assert turn.rows[0].playback_start_ms is None


@pytest.mark.parametrize("stage", ["recording", "stop", "recognize", "synthesize"])
def test_invalidated_turn_closes_resources_once(stage: str) -> None:
    turn = make_turn()
    if stage == "recording":
        turn.controller.button_pressed()
        turn.source.session_id = None
        turn.controller.button_pressed()
    else:
        invalidate_after(turn, stage)
        finish(turn)

    assert_discarded(turn)
    assert turn.recorder.calls.count("start") == 1
    assert turn.recorder.calls.count("stop") == int(stage != "recording")
    assert turn.recorder.calls.count("abort") == int(stage == "recording")
    assert all(session.close_calls == 1 for session in turn.recorder.sessions)
    assert all(audio.close_calls == 1 for audio in turn.recorder.recorded_audios)
    assert all(audio.close_calls == 1 for audio in turn.synthesizer.audios)

    turn.controller.tick()
    turn.controller.playback_finished(1)
    turn.controller.shutdown()
    assert turn.recorder.session_close_calls == 1
    assert turn.recorder.recorded_audio_close_calls == int(stage != "recording")
    assert turn.synthesizer.audio_close_calls == int(stage == "synthesize")


@pytest.mark.parametrize("stage", ["recognize", "dialogue", "synthesize"])
def test_invalidated_stage_exception_never_speaks_fallback(stage: str) -> None:
    turn = make_turn()
    invalidate_after(turn, stage)
    adapter = {
        "recognize": turn.recognizer, "dialogue": turn.dialogue,
        "synthesize": turn.synthesizer,
    }[stage]
    adapter.failure = RuntimeError("synthetic stage failure")

    finish(turn)

    assert_discarded(turn)
    assert turn.synthesizer.inputs == (["synthetic safe response"] if stage == "synthesize" else [])
    assert turn.rows[0].played is False
    assert turn.rows[0].issue is None


def test_session_playback_starter_receives_captured_session() -> None:
    turn = make_turn()

    finish(turn)

    assert turn.controller.state is SpeechState.PLAYING
    turn.starter.assert_called_once_with(turn.synthesizer.audios[0], 1, "A")
    assert turn.player.tokens == []
    assert turn.synthesizer.audio_close_calls == 0
    turn.controller.playback_finished(1)
    assert turn.synthesizer.audio_close_calls == 1
    assert turn.controller.state is SpeechState.IDLE


@pytest.mark.parametrize("missing", ["session_id_supplier", "session_playback_starter"])
def test_session_callbacks_must_be_paired(missing: str) -> None:
    ports = {name: Mock() for name in (
        "recorder", "recognizer", "dialogue", "synthesizer", "player", "mute_gate", "clock"
    )}
    callbacks = dict(session_id_supplier=Mock(), session_playback_starter=Mock())
    callbacks[missing] = None

    with pytest.raises(ValueError, match=r"^session callbacks must be supplied together$"):
        SpeechController(trusted_user_id="synthetic-owner", **ports, **callbacks)

    for port in ports.values():
        assert port.mock_calls == []
    for callback in callbacks.values():
        if callback is not None:
            callback.assert_not_called()


def test_no_live_session_never_starts_recording() -> None:
    turn = make_turn()
    turn.source.session_id = None

    turn.controller.button_pressed()

    assert_discarded(turn)
    assert turn.recorder.calls == []
    assert turn.synthesizer.inputs == []
    assert turn.rows == []


@pytest.mark.parametrize("state", ["recording", "playing"])
@pytest.mark.parametrize("replacement", [None, "B"])
def test_tick_invalidates_recording_and_playback(state: str, replacement: str | None) -> None:
    turn = make_turn()
    turn.controller.button_pressed()
    if state == "playing":
        turn.controller.button_pressed()
        turn.starter.assert_called_once()
        turn.starter.reset_mock()
    turn.source.session_id = replacement
    turn.clock.now = 31.0

    turn.controller.tick()
    turn.controller.tick()

    assert_discarded(turn)
    assert turn.recorder.calls.count("start") == 1
    assert turn.recorder.calls.count("stop") == int(state == "playing")
    assert turn.recorder.calls.count("abort") == int(state == "recording")
    assert turn.player.stop_calls == int(state == "playing")
    assert turn.synthesizer.audio_close_calls == int(state == "playing")
    assert turn.recorder.session_close_calls == 1


@pytest.mark.parametrize("fails", [False, True])
def test_disconnect_during_recorder_start_aborts_acquired_session(fails: bool) -> None:
    turn = make_turn()
    invalidate_after(turn, "start")
    if fails:
        turn.recorder.configure_failure("recorder.start", RuntimeError("synthetic start failure"))

    turn.controller.button_pressed()

    assert_discarded(turn)
    assert turn.recorder.calls == (["start"] if fails else ["start", "abort", "session.close"])


def test_disconnect_before_recorder_start_does_not_open_microphone() -> None:
    turn = make_turn()

    def now() -> float:
        turn.source.session_id = None
        return 0.0

    turn.clock.monotonic = now
    turn.controller.button_pressed()

    assert_discarded(turn)
    assert turn.recorder.calls == []


def test_invalidated_stop_exception_closes_session_without_retry() -> None:
    turn = make_turn()
    invalidate_after(turn, "stop")
    turn.recorder.configure_failure("session.stop", RuntimeError("synthetic stop failure"))

    finish(turn)

    assert_discarded(turn)
    assert turn.recorder.calls == ["start", "stop", "session.close"]
    assert turn.recognizer.calls == 0


@pytest.mark.parametrize("boundary", ["session.close", "processing", "audio.close"])
def test_invalidation_before_next_stage_prevents_call(boundary: str) -> None:
    turn = make_turn()

    def disconnect() -> None:
        turn.source.session_id = None

    if boundary == "processing":
        turn = make_turn(processing_observer=disconnect)
    else:
        turn.recorder.configure_callback(boundary, disconnect)

    finish(turn)

    assert_discarded(turn)
    assert turn.recognizer.calls == int(boundary == "audio.close")
    assert turn.dialogue.calls == []
    assert turn.synthesizer.inputs == []
    assert turn.recorder.recorded_audio_close_calls == 1


@pytest.mark.parametrize("fails", [False, True])
def test_disconnect_during_playback_start_stops_and_closes(fails: bool) -> None:
    turn = make_turn()

    def start(audio: object, token: int, session_id: str) -> None:
        assert session_id == "A"
        turn.source.session_id = "B"
        if fails:
            raise RuntimeError("synthetic playback failure")

    turn.starter.side_effect = start
    finish(turn)

    assert turn.starter.call_count == 1
    assert turn.player.tokens == []
    assert turn.player.stop_calls == 1
    assert turn.synthesizer.audio_close_calls == 1
    assert turn.controller.state is SpeechState.IDLE
    assert turn.controller.last_issue is None
    assert turn.rows[0].played is False


@pytest.mark.parametrize(
    "failure", ["abort", "session-close", "audio-close", "tts-close", "player-stop"]
)
def test_invalidated_cleanup_failure_uses_existing_issue(failure: str) -> None:
    turn = make_turn()
    error = RuntimeError("synthetic cleanup failure")
    if failure in {"abort", "session-close"}:
        turn.recorder.configure_failure(
            "session.abort" if failure == "abort" else "session.close", error
        )
        turn.controller.button_pressed()
    elif failure == "audio-close":
        turn.recorder.configure_failure("audio.close", error)
        invalidate_after(turn, "recognize")
        finish(turn)
    elif failure == "tts-close":
        turn.synthesizer.audio_close_failure = error
        invalidate_after(turn, "synthesize")
        finish(turn)
    else:
        turn.player.stop_failure = error
        finish(turn)
    turn.source.session_id = None
    turn.controller.tick()

    assert turn.controller.state is SpeechState.IDLE
    assert turn.controller.last_issue is SpeechControllerIssue.CLEANUP_FAILED
    assert turn.recorder.session_close_calls == 1
    if failure == "audio-close":
        assert turn.recorder.recorded_audio_close_calls == 1
    if failure in {"tts-close", "player-stop"}:
        assert turn.synthesizer.audio_close_calls == 1


def test_discard_allows_a_fresh_turn_and_shutdown_releases_callbacks() -> None:
    turn = make_turn()
    invalidate_after(turn, "stop")
    finish(turn)
    assert_discarded(turn)
    turn.recorder.configure_callback("session.stop", lambda: None)
    turn.source.session_id = "B"

    finish(turn)

    assert turn.controller.state is SpeechState.PLAYING
    turn.starter.assert_called_once_with(turn.synthesizer.audios[0], 1, "B")
    turn.controller.shutdown()
    assert turn.controller._session_id_supplier is None
    assert turn.controller._session_playback_starter is None
    assert turn.controller._source_session_id is None
    assert turn.synthesizer.audio_close_calls == 1
