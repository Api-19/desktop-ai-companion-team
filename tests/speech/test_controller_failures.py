from collections.abc import Callable
from dataclasses import replace

import pytest

from companion.contracts import DialogueResponse, TranscriptConfidenceStatus
from companion.speech.controller import SpeechController
from companion.speech.controller_models import (
    CONNECTION_RETRY_PROMPT,
    EMPTY_REPLY_PROMPT,
    RECOGNITION_RETRY_PROMPT,
    RecognitionInput,
    SpeechControllerIssue,
    SpeechState,
    SpeechTurnFailure,
)
from companion.speech.recognition_gate import AudioQualityEvidence
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    DialogueFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)


def valid_audio() -> AudioQualityEvidence:
    return AudioQualityEvidence(
        sample_rate_hz=16_000,
        channels=1,
        sample_width_bytes=2,
        frame_count=4_800,
        rms=0.1,
        peak_abs=0.2,
        clipping_fraction=0.0,
        estimated_snr_db=20.0,
    )


def recognition_input(**changes: object) -> RecognitionInput:
    baseline = RecognitionInput(
        audio=valid_audio(),
        vad_passed=True,
        text="synthetic accepted transcript",
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
        confidence_threshold=None,
    )
    return replace(baseline, **changes)


def make_controller(
    *,
    recognizer: SpeechRecognizerFake | None = None,
    dialogue: DialogueFake | None = None,
    mute_gate: MuteGateFake | None = None,
    session_id_supplier: Callable[[], str | None] | None = None,
) -> tuple[
    SpeechController,
    AudioRecorderFake,
    DialogueFake,
    SpeechSynthesizerFake,
    AudioPlayerFake,
]:
    recorder = AudioRecorderFake()
    dialogue = dialogue or DialogueFake()
    synthesizer = SpeechSynthesizerFake()
    player = AudioPlayerFake()
    controller = SpeechController(
        trusted_user_id="trusted-owner",
        recorder=recorder,
        recognizer=recognizer or SpeechRecognizerFake(),
        dialogue=dialogue,
        synthesizer=synthesizer,
        player=player,
        mute_gate=mute_gate or MuteGateFake(),
        clock=ClockFake(),
        session_id_supplier=session_id_supplier,
        session_playback_starter=(
            (lambda audio, token, session_id: player.start(audio, token))
            if session_id_supplier is not None else None
        ),
    )
    return controller, recorder, dialogue, synthesizer, player


@pytest.mark.parametrize("reason", [
    "provider_timeout", "provider_unavailable", "invalid_provider_response",
])
@pytest.mark.parametrize("speech_failure", ["none", "tts", "start", "async", "cleanup"])
def test_provider_failure_survives_speech_issue(reason: str, speech_failure: str) -> None:
    response = DialogueResponse(
        text="synthetic cloud fallback", disclosed_ai_identity=False,
        billable=False, blocked_reason=reason,
    )
    controller, _, _, synthesizer, player = make_controller(
        dialogue=DialogueFake(result=response)
    )
    if speech_failure == "tts":
        synthesizer.failure = RuntimeError("synthetic synthesis failure")
    elif speech_failure == "start":
        player.start_failure = RuntimeError("synthetic playback start failure")
    elif speech_failure == "cleanup":
        synthesizer.audio_close_failure = RuntimeError("synthetic close failure")

    controller.button_pressed()
    controller.button_pressed()

    assert controller.turn_failure is SpeechTurnFailure(reason.replace("_", "-"))
    assert synthesizer.inputs == [response.text]
    if speech_failure in {"none", "async", "cleanup"}:
        assert controller.state is SpeechState.PLAYING
        assert controller.last_issue is None
        controller.tick()
        if speech_failure == "async":
            controller.playback_failed(player.tokens[-1])
        else:
            controller.playback_finished(player.tokens[-1])
    expected_issue = {
        "none": None,
        "tts": SpeechControllerIssue.SYNTHESIS_FAILED,
        "start": SpeechControllerIssue.PLAYBACK_FAILED,
        "async": SpeechControllerIssue.PLAYBACK_FAILED,
        "cleanup": SpeechControllerIssue.CLEANUP_FAILED,
    }[speech_failure]
    controller.tick()
    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is expected_issue
    assert controller.turn_failure is SpeechTurnFailure(reason.replace("_", "-"))


@pytest.mark.parametrize("reason", ["budget_exceeded", "memory_disabled", "unsafe_output", None])
def test_local_blocks_do_not_set_cloud_turn_failure(reason: str | None) -> None:
    controller, _, _, _, _ = make_controller(dialogue=DialogueFake(result=DialogueResponse(
        text="synthetic local fallback", disclosed_ai_identity=False,
        billable=False, blocked_reason=reason,
    )))
    controller.button_pressed()
    controller.button_pressed()
    assert controller.turn_failure is None
    assert controller.last_issue is None


@pytest.mark.parametrize("playing", [False, True])
def test_new_recording_clears_cloud_turn_failure(playing: bool) -> None:
    controller, _, _, _, player = make_controller(dialogue=DialogueFake(result=DialogueResponse(
        text="synthetic fallback", disclosed_ai_identity=False,
        billable=False, blocked_reason="provider_timeout",
    )))
    controller.button_pressed()
    controller.button_pressed()
    assert controller.turn_failure is SpeechTurnFailure.PROVIDER_TIMEOUT
    if not playing:
        controller.playback_finished(player.tokens[-1])
    controller.button_pressed()
    assert controller.state is SpeechState.RECORDING
    assert controller.turn_failure is None


@pytest.mark.parametrize("blocked_start", ["muted", "no-session"])
def test_muted_button_does_not_clear_previous_turn_failure(blocked_start: str) -> None:
    gate = MuteGateFake()
    session_id: str | None = "synthetic-session"
    controller, _, _, _, player = make_controller(
        dialogue=DialogueFake(result=DialogueResponse(
            text="synthetic fallback", disclosed_ai_identity=False,
            billable=False, blocked_reason="provider_timeout",
        )),
        mute_gate=gate,
        session_id_supplier=lambda: session_id,
    )
    controller.button_pressed()
    controller.button_pressed()
    controller.playback_finished(player.tokens[-1])
    if blocked_start == "muted":
        gate.enabled = False
    else:
        session_id = None
    controller.button_pressed()
    assert controller.state is SpeechState.IDLE
    assert controller.turn_failure is SpeechTurnFailure.PROVIDER_TIMEOUT


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_shutdown_clears_cloud_turn_failure(cleanup_fails: bool) -> None:
    controller, _, _, synthesizer, _ = make_controller(
        dialogue=DialogueFake(result=DialogueResponse(
            text="synthetic fallback", disclosed_ai_identity=False,
            billable=False, blocked_reason="provider_timeout",
        ))
    )
    if cleanup_fails:
        synthesizer.audio_close_failure = RuntimeError("synthetic close failure")
    controller.button_pressed()
    controller.button_pressed()
    assert controller.turn_failure is SpeechTurnFailure.PROVIDER_TIMEOUT
    controller.shutdown()
    assert controller.turn_failure is None
    assert controller.last_issue is (
        SpeechControllerIssue.CLEANUP_FAILED if cleanup_fails else None
    )


@pytest.mark.parametrize(
    "result",
    [
        recognition_input(vad_passed=False),
        recognition_input(audio=replace(valid_audio(), sample_rate_hz=8_000)),
        recognition_input(audio=replace(valid_audio(), rms=0.001)),
        recognition_input(text="   "),
        recognition_input(
            confidence=0.49,
            confidence_status=TranscriptConfidenceStatus.AVAILABLE,
            confidence_threshold=0.5,
        ),
        recognition_input(
            text=None,
            confidence_status=None,
            recognizer_failed=True,
        ),
    ],
    ids=[
        "silence-or-noise",
        "invalid-audio",
        "audio-quality",
        "blank-text",
        "below-threshold",
        "recognizer-failed",
    ],
)
def test_recognition_rejections_use_only_local_retry_without_dialogue(
    result: RecognitionInput,
) -> None:
    controller, recorder, dialogue, synthesizer, _ = make_controller(
        recognizer=SpeechRecognizerFake(result=result)
    )

    controller.button_pressed()
    controller.button_pressed()

    assert dialogue.calls == []
    assert synthesizer.inputs == [RECOGNITION_RETRY_PROMPT]
    assert controller.last_issue in {
        SpeechControllerIssue.RECOGNITION_REJECTED,
        SpeechControllerIssue.RECOGNITION_FAILED,
    }
    assert recorder.recorded_audios[0].live_count == 0
    assert controller.state is SpeechState.PLAYING


@pytest.mark.parametrize("failure", ["exception", "invalid-audio"])
def test_empty_reply_notice_synthesis_failure_does_not_loop(failure: str) -> None:
    response = DialogueResponse(
        text="[猪头]", disclosed_ai_identity=False, billable=False, blocked_reason=None
    )
    controller, _, dialogue, synthesizer, player = make_controller(
        dialogue=DialogueFake(result=response)
    )
    if failure == "exception":
        synthesizer.failure = RuntimeError("synthetic synthesis failure")
    else:
        synthesizer.result = None

    controller.button_pressed()
    controller.button_pressed()

    assert synthesizer.inputs == [EMPTY_REPLY_PROMPT]
    assert player.tokens == []
    assert len(dialogue.calls) == 1
    assert response.text == "[猪头]"
    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.SYNTHESIS_FAILED


@pytest.mark.parametrize("failure", ["start", "async"])
def test_empty_reply_notice_playback_failure_releases_audio(failure: str) -> None:
    response = DialogueResponse(
        text="[猪头]", disclosed_ai_identity=False, billable=False, blocked_reason=None
    )
    controller, _, dialogue, synthesizer, player = make_controller(
        dialogue=DialogueFake(result=response)
    )
    if failure == "start":
        player.start_failure = RuntimeError("synthetic playback failure")

    controller.button_pressed()
    controller.button_pressed()
    if failure == "async":
        assert controller.state is SpeechState.PLAYING
        controller.playback_failed(1)
    controller.playback_failed(1)
    controller.playback_finished(1)

    assert synthesizer.inputs == [EMPTY_REPLY_PROMPT]
    assert player.tokens == [1]
    assert len(dialogue.calls) == 1
    assert response.text == "[猪头]"
    assert synthesizer.audios[0].close_calls == 1
    assert synthesizer.live_audio_count == 0
    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.PLAYBACK_FAILED


def test_available_confidence_exactly_at_threshold_reaches_dialogue() -> None:
    result = recognition_input(
        confidence=0.5,
        confidence_status=TranscriptConfidenceStatus.AVAILABLE,
        confidence_threshold=0.5,
    )
    controller, _, dialogue, synthesizer, _ = make_controller(
        recognizer=SpeechRecognizerFake(result=result)
    )

    controller.button_pressed()
    controller.button_pressed()

    assert dialogue.calls == [("trusted-owner", "synthetic accepted transcript")]
    assert synthesizer.inputs == ["synthetic safe response"]
    assert controller.last_issue is None


@pytest.mark.parametrize(
    "changes",
    [
        {"confidence": 0.5},
        {"confidence_threshold": 0.5},
    ],
    ids=["score", "threshold"],
)
def test_unavailable_confidence_with_value_cannot_reach_dialogue(
    changes: dict[str, object],
) -> None:
    controller, _, dialogue, synthesizer, _ = make_controller(
        recognizer=SpeechRecognizerFake(result=recognition_input(**changes))
    )

    controller.button_pressed()
    controller.button_pressed()

    assert dialogue.calls == []
    assert synthesizer.inputs == [RECOGNITION_RETRY_PROMPT]
    assert controller.last_issue is SpeechControllerIssue.RECOGNITION_FAILED


def test_new_recording_clears_retained_local_fallback_issue() -> None:
    controller, _, _, _, _ = make_controller(
        recognizer=SpeechRecognizerFake(result=recognition_input(vad_passed=False))
    )
    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.PLAYING
    assert controller.last_issue is SpeechControllerIssue.RECOGNITION_REJECTED

    controller.button_pressed()

    assert controller.state is SpeechState.RECORDING
    assert controller.last_issue is None


def test_exact_dialogue_response_is_deep_validated_and_synthesized() -> None:
    response = DialogueResponse(
        text="synthetic Dialogue-owned safe fallback",
        disclosed_ai_identity=True,
        billable=False,
        blocked_reason="synthetic-blocked-reason",
    )
    dialogue = DialogueFake(result=response)
    controller, _, _, synthesizer, _ = make_controller(dialogue=dialogue)

    controller.button_pressed()
    controller.button_pressed()

    assert dialogue.calls == [("trusted-owner", "synthetic transcript")]
    assert synthesizer.inputs == ["synthetic Dialogue-owned safe fallback"]
    assert controller.last_issue is None
    assert controller.state is SpeechState.PLAYING


class ExplosiveRepr:
    def __repr__(self) -> str:
        raise AssertionError("repr must not be called")


class DialogueResponseSubclass(DialogueResponse):
    pass


@pytest.mark.parametrize(
    "bad_response",
    [
        DialogueResponseSubclass(
            text="synthetic subclass text",
            disclosed_ai_identity=False,
            billable=False,
            blocked_reason=None,
        ),
        ExplosiveRepr(),
        DialogueResponse.model_construct(
            text=ExplosiveRepr(),
            disclosed_ai_identity=False,
            billable=False,
            blocked_reason=None,
        ),
        DialogueResponse.model_construct(
            text="synthetic invalid identity",
            disclosed_ai_identity="false",
            billable=False,
            blocked_reason=None,
        ),
        DialogueResponse.model_construct(
            text="synthetic invalid billing",
            disclosed_ai_identity=False,
            billable=0,
            blocked_reason="provider_timeout",
        ),
        DialogueResponse.model_construct(
            text="synthetic invalid block reason",
            disclosed_ai_identity=False,
            billable=False,
            blocked_reason=1,
        ),
    ],
    ids=[
        "subclass",
        "arbitrary-object",
        "invalid-text",
        "invalid-identity",
        "invalid-billing",
        "invalid-blocked-reason",
    ],
)
def test_invalid_dialogue_response_uses_one_fixed_local_fallback(
    bad_response: object,
) -> None:
    dialogue = DialogueFake(result=bad_response)
    controller, _, _, synthesizer, _ = make_controller(dialogue=dialogue)

    controller.button_pressed()
    controller.button_pressed()

    assert dialogue.calls == [("trusted-owner", "synthetic transcript")]
    assert synthesizer.inputs == [CONNECTION_RETRY_PROMPT]
    assert controller.last_issue is SpeechControllerIssue.DIALOGUE_FAILED
    assert controller.turn_failure is None
    assert controller.state is SpeechState.PLAYING


@pytest.mark.parametrize("method_name", ["model_dump", "model_validate"])
def test_dialogue_validation_exception_uses_fixed_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    def fail_validation(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic validation failure")

    response = DialogueResponse(
        text="synthetic response hidden by validation failure",
        disclosed_ai_identity=False,
        billable=False,
        blocked_reason=None,
    )
    monkeypatch.setattr(DialogueResponse, method_name, fail_validation)
    dialogue = DialogueFake(result=response)
    controller, _, _, synthesizer, _ = make_controller(dialogue=dialogue)

    controller.button_pressed()
    controller.button_pressed()

    assert dialogue.calls == [("trusted-owner", "synthetic transcript")]
    assert synthesizer.inputs == [CONNECTION_RETRY_PROMPT]
    assert controller.last_issue is SpeechControllerIssue.DIALOGUE_FAILED


def make_failure_controller(
    *,
    recorder: AudioRecorderFake | None = None,
    recognizer: SpeechRecognizerFake | None = None,
    dialogue: DialogueFake | None = None,
    synthesizer: SpeechSynthesizerFake | None = None,
    player: AudioPlayerFake | None = None,
    mute_gate: MuteGateFake | None = None,
) -> tuple[
    SpeechController,
    AudioRecorderFake,
    SpeechRecognizerFake,
    DialogueFake,
    SpeechSynthesizerFake,
    AudioPlayerFake,
    MuteGateFake,
]:
    recorder = recorder or AudioRecorderFake()
    recognizer = recognizer or SpeechRecognizerFake()
    dialogue = dialogue or DialogueFake()
    synthesizer = synthesizer or SpeechSynthesizerFake()
    player = player or AudioPlayerFake()
    mute_gate = mute_gate or MuteGateFake()
    controller = SpeechController(
        trusted_user_id="trusted-owner",
        recorder=recorder,
        recognizer=recognizer,
        dialogue=dialogue,
        synthesizer=synthesizer,
        player=player,
        mute_gate=mute_gate,
        clock=ClockFake(),
    )
    return (
        controller,
        recorder,
        recognizer,
        dialogue,
        synthesizer,
        player,
        mute_gate,
    )


def controller_owned_reference_count(controller: SpeechController) -> int:
    return sum(
        getattr(controller, name) is not None
        for name in ("_recording_session", "_synthesized_audio", "_playback_token")
    )


def assert_no_live_resources(
    recorder: AudioRecorderFake,
    synthesizer: SpeechSynthesizerFake,
) -> None:
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0


def test_recorder_start_failure_is_fixed_and_creates_no_resources() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_failure("recorder.start", RuntimeError("synthetic start failure"))
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )

    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert recorder.calls == ["start"]
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


def test_session_stop_failure_closes_session_and_stops_downstream() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_failure("session.stop", RuntimeError("synthetic stop failure"))
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert recorder.calls == ["start", "stop", "session.close"]
    assert recorder.session_close_calls == 1
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


def test_session_abort_failure_still_closes_session_and_stops_downstream() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_failure("session.abort", RuntimeError("synthetic abort failure"))
    mute_gate = MuteGateFake(enabled=True)
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder, mute_gate=mute_gate
    )
    controller.button_pressed()
    mute_gate.enabled = False

    controller.mute_changed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert recorder.calls == ["start", "abort", "session.close"]
    assert recorder.session_close_calls == 1
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


@pytest.mark.parametrize(
    "failure_point",
    ["session.abort", "session.close", "player.stop", "synthesized_audio.close"],
)
def test_shutdown_cleanup_failure_clears_references_and_stops_downstream(
    failure_point: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = f"SHUTDOWN_PRIVATE_{failure_point}"
    recorder = AudioRecorderFake()
    synthesizer = SpeechSynthesizerFake(
        audio_close_failure=RuntimeError(marker)
        if failure_point == "synthesized_audio.close"
        else None
    )
    player = AudioPlayerFake(
        stop_failure=RuntimeError(marker) if failure_point == "player.stop" else None
    )
    if failure_point.startswith("session."):
        recorder.configure_failure(failure_point, RuntimeError(marker))
    controller, _, recognizer, dialogue, _, _, _ = make_failure_controller(
        recorder=recorder,
        synthesizer=synthesizer,
        player=player,
    )
    controller.button_pressed()
    if not failure_point.startswith("session."):
        controller.button_pressed()

    controller.shutdown()

    captured = capsys.readouterr()
    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.CLEANUP_FAILED
    assert controller._trusted_user_id is None
    assert controller._recorder is None
    assert controller._recognizer is None
    assert controller._dialogue is None
    assert controller._synthesizer is None
    assert controller._player is None
    assert controller._mute_gate is None
    assert controller._clock is None
    assert controller._recording_session is None
    assert controller._synthesized_audio is None
    assert controller._playback_token is None
    assert recognizer.calls == (1 if not failure_point.startswith("session.") else 0)
    assert len(dialogue.calls) == (1 if not failure_point.startswith("session.") else 0)
    assert marker not in caplog.text
    assert marker not in captured.out
    assert marker not in captured.err


def test_recognizer_exception_uses_local_retry_and_releases_resources() -> None:
    recognizer = SpeechRecognizerFake(failure=RuntimeError("synthetic recognize failure"))
    controller, recorder, _, dialogue, synthesizer, player, _ = make_failure_controller(
        recognizer=recognizer
    )

    controller.button_pressed()
    controller.button_pressed()
    controller.playback_finished(1)

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECOGNITION_FAILED
    assert dialogue.calls == []
    assert synthesizer.inputs == [RECOGNITION_RETRY_PROMPT]
    assert player.tokens == [1]
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


def test_invalid_gate_input_uses_local_retry_and_releases_resources() -> None:
    invalid_input = recognition_input()
    object.__setattr__(invalid_input, "vad_passed", 1)
    controller, recorder, _, dialogue, synthesizer, player, _ = make_failure_controller(
        recognizer=SpeechRecognizerFake(result=invalid_input)
    )

    controller.button_pressed()
    controller.button_pressed()
    controller.playback_finished(1)

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECOGNITION_FAILED
    assert dialogue.calls == []
    assert synthesizer.inputs == [RECOGNITION_RETRY_PROMPT]
    assert player.tokens == [1]
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


def test_dialogue_exception_uses_local_fallback_once_and_releases_resources() -> None:
    dialogue = DialogueFake(failure=RuntimeError("synthetic dialogue failure"))
    controller, recorder, _, _, synthesizer, player, _ = make_failure_controller(dialogue=dialogue)

    controller.button_pressed()
    controller.button_pressed()
    controller.playback_finished(1)

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.DIALOGUE_FAILED
    assert controller.turn_failure is None
    assert len(dialogue.calls) == 1
    assert synthesizer.inputs == [CONNECTION_RETRY_PROMPT]
    assert player.tokens == [1]
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


def test_synthesizer_failure_stops_before_player_and_releases_resources() -> None:
    synthesizer = SpeechSynthesizerFake(failure=RuntimeError("synthetic synthesis failure"))
    controller, recorder, _, dialogue, _, player, _ = make_failure_controller(
        synthesizer=synthesizer
    )

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.SYNTHESIS_FAILED
    assert len(dialogue.calls) == 1
    assert player.tokens == []
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


def test_player_start_failure_stops_player_and_closes_synthesized_audio() -> None:
    player = AudioPlayerFake(start_failure=RuntimeError("synthetic player start failure"))
    controller, recorder, _, _, synthesizer, _, _ = make_failure_controller(player=player)

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.PLAYBACK_FAILED
    assert player.tokens == [1]
    assert player.stop_calls == 1
    assert synthesizer.audio_close_calls == 1
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


def test_player_stop_failure_closes_audio_and_does_not_start_new_recording() -> None:
    player = AudioPlayerFake(stop_failure=RuntimeError("synthetic player stop failure"))
    controller, recorder, _, _, synthesizer, _, _ = make_failure_controller(player=player)
    controller.button_pressed()
    controller.button_pressed()

    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.PLAYBACK_FAILED
    assert player.stop_calls == 1
    assert recorder.calls.count("start") == 1
    assert synthesizer.audio_close_calls == 1
    assert controller_owned_reference_count(controller) == 0
    assert_no_live_resources(recorder, synthesizer)


def test_session_close_failure_closes_returned_audio_and_prevents_recognition() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_failure("session.close", RuntimeError("synthetic session close failure"))
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.CLEANUP_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []
    assert recorder.session_close_calls == 1
    assert recorder.recorded_audio_close_calls == 1
    assert recorder.live_recorded_audio_count == 0


def test_recorded_audio_close_failure_prevents_dialogue_and_is_not_retried() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_failure("audio.close", RuntimeError("synthetic audio close failure"))
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.CLEANUP_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recognizer.calls == 1
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []
    assert recorder.recorded_audio_close_calls == 1
    assert recorder.live_session_count == 0


def test_synthesized_audio_close_failure_is_not_retried_or_retained() -> None:
    synthesizer = SpeechSynthesizerFake(
        audio_close_failure=RuntimeError("synthetic synthesized close failure")
    )
    controller, recorder, _, _, _, player, _ = make_failure_controller(synthesizer=synthesizer)
    controller.button_pressed()
    controller.button_pressed()

    controller.playback_finished(1)

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.CLEANUP_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert player.tokens == [1]
    assert synthesizer.audio_close_calls == 1
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0


@pytest.mark.parametrize("bad_session", [None, ExplosiveRepr()])
def test_malformed_adapter_return_from_recorder_start_fails_closed(
    bad_session: object,
) -> None:
    recorder = AudioRecorderFake()
    recorder.configure_result("recorder.start", bad_session)
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )

    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


class NonCallableRecordingSession:
    stop = 1
    abort = 1
    close = 1


class NonCallableAudio:
    close = 1


def test_structural_session_with_non_callable_methods_fails_before_ownership() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_result("recorder.start", NonCallableRecordingSession())
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )

    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


@pytest.mark.parametrize("bad_audio", [None, ExplosiveRepr()])
def test_malformed_adapter_return_from_session_stop_fails_closed(
    bad_audio: object,
) -> None:
    recorder = AudioRecorderFake()
    recorder.configure_result("session.stop", bad_audio)
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )
    controller.button_pressed()

    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recorder.session_close_calls == 1
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


def test_structural_recorded_audio_with_non_callable_close_fails_before_recognition() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_result("session.stop", NonCallableAudio())
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )
    controller.button_pressed()

    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recorder.session_close_calls == 1
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


@pytest.mark.parametrize("bad_audio", [None, ExplosiveRepr()])
def test_malformed_adapter_return_from_synthesizer_fails_closed(
    bad_audio: object,
) -> None:
    synthesizer = SpeechSynthesizerFake(result=bad_audio)
    controller, recorder, _, dialogue, _, player, _ = make_failure_controller(
        synthesizer=synthesizer
    )

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.SYNTHESIS_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert len(dialogue.calls) == 1
    assert player.tokens == []


def test_structural_synthesized_audio_with_non_callable_close_fails_before_player() -> None:
    synthesizer = SpeechSynthesizerFake(result=NonCallableAudio())
    controller, recorder, _, dialogue, _, player, _ = make_failure_controller(
        synthesizer=synthesizer
    )

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.SYNTHESIS_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert len(dialogue.calls) == 1
    assert player.tokens == []


def test_non_none_completion_from_session_abort_is_recording_failure() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_result("session.abort", False)
    mute_gate = MuteGateFake(enabled=True)
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder,
        mute_gate=mute_gate,
    )
    controller.button_pressed()
    mute_gate.enabled = False

    controller.mute_changed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recorder.calls == ["start", "abort", "session.close"]
    assert recorder.session_close_calls == 1
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


def test_non_none_completion_from_session_close_overrides_and_stops_downstream() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_result("session.close", False)
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )
    controller.button_pressed()

    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.CLEANUP_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recorder.session_close_calls == 1
    assert recorder.recorded_audio_close_calls == 1
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


def test_non_none_completion_from_recorded_audio_close_stops_downstream() -> None:
    recorder = AudioRecorderFake()
    recorder.configure_result("audio.close", False)
    controller, _, recognizer, dialogue, synthesizer, player, _ = make_failure_controller(
        recorder=recorder
    )

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.CLEANUP_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert recorder.recorded_audio_close_calls == 1
    assert recognizer.calls == 1
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


def test_non_none_completion_from_synthesized_audio_close_is_cleanup_failure() -> None:
    synthesizer = SpeechSynthesizerFake(audio_close_result=False)
    controller, recorder, _, _, _, player, _ = make_failure_controller(synthesizer=synthesizer)
    controller.button_pressed()
    controller.button_pressed()

    controller.playback_finished(1)

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.CLEANUP_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert synthesizer.audio_close_calls == 1
    assert player.tokens == [1]
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0


def test_current_playback_failure_releases_audio_and_records_failure() -> None:
    controller, recorder, _, _, synthesizer, player, _ = make_failure_controller()
    controller.button_pressed()
    controller.button_pressed()

    controller.playback_failed(1)

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.PLAYBACK_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert synthesizer.audio_close_calls == 1
    assert player.stop_calls == 0
    assert_no_live_resources(recorder, synthesizer)


def test_stale_playback_failure_does_not_close_current_playback() -> None:
    controller, _, _, _, synthesizer, player, _ = make_failure_controller()
    controller.button_pressed()
    controller.button_pressed()

    controller.playback_failed(2)

    assert controller.state is SpeechState.PLAYING
    assert controller.last_issue is None
    assert synthesizer.audio_close_calls == 0
    assert player.stop_calls == 0


@pytest.mark.parametrize("invalid_token", [True, 1.0, "1", None])
def test_non_exact_integer_playback_failure_is_ignored(invalid_token: object) -> None:
    controller, _, _, _, synthesizer, player, _ = make_failure_controller()
    controller.button_pressed()
    controller.button_pressed()

    controller.playback_failed(invalid_token)

    assert controller.state is SpeechState.PLAYING
    assert controller.last_issue is None
    assert synthesizer.audio_close_calls == 0
    assert player.stop_calls == 0


def test_repeated_playback_failure_is_ignored_after_first_cleanup() -> None:
    controller, _, _, _, synthesizer, player, _ = make_failure_controller()
    controller.button_pressed()
    controller.button_pressed()

    controller.playback_failed(1)
    controller.playback_failed(1)

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.PLAYBACK_FAILED
    assert synthesizer.audio_close_calls == 1
    assert player.stop_calls == 0


def test_non_none_completion_from_player_start_closes_audio_once() -> None:
    player = AudioPlayerFake(start_result=False)
    controller, recorder, _, _, synthesizer, _, _ = make_failure_controller(player=player)

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.PLAYBACK_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert player.tokens == [1]
    assert player.stop_calls == 1
    assert synthesizer.audio_close_calls == 1
    assert_no_live_resources(recorder, synthesizer)


def test_non_none_completion_from_player_stop_closes_audio_without_new_recording() -> None:
    player = AudioPlayerFake(stop_result=False)
    controller, recorder, _, _, synthesizer, _, _ = make_failure_controller(player=player)
    controller.button_pressed()
    controller.button_pressed()

    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.PLAYBACK_FAILED
    assert controller_owned_reference_count(controller) == 0
    assert player.stop_calls == 1
    assert recorder.calls.count("start") == 1
    assert synthesizer.audio_close_calls == 1
    assert_no_live_resources(recorder, synthesizer)
