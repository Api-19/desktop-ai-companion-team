from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from companion.contracts import DialogueResponse, TranscriptConfidenceStatus
from companion.speech.controller_models import (
    CONNECTION_RETRY_PROMPT,
    RECOGNITION_RETRY_PROMPT,
    RecognitionInput,
    SpeechControllerIssue,
    SpeechState,
)
from companion.speech.controller_ports import (
    AudioPlayer,
    AudioRecorder,
    Clock,
    DialogueCallable,
    MuteGate,
    RecordedAudio,
    RecordingSession,
    SpeechRecognizer,
    SpeechSynthesizer,
    SynthesizedAudio,
)
from companion.speech.recognition_gate import AudioQualityEvidence

ROOT = Path(__file__).resolve().parents[2]


class _RecordedAudioFake:
    def close(self) -> None:
        pass


class _RecordingSessionFake:
    def stop(self) -> RecordedAudio:
        return _RecordedAudioFake()

    def abort(self) -> None:
        pass

    def close(self) -> None:
        pass


class _AudioRecorderFake:
    def start(self) -> RecordingSession:
        return _RecordingSessionFake()


class _SpeechRecognizerFake:
    def recognize(self, audio: RecordedAudio) -> RecognitionInput:
        raise NotImplementedError


class _DialogueCallableFake:
    def reply(self, *, user_id: str, text: str) -> DialogueResponse:
        raise NotImplementedError


class _SynthesizedAudioFake:
    def close(self) -> None:
        pass


class _SpeechSynthesizerFake:
    def synthesize(self, text: str) -> SynthesizedAudio:
        return _SynthesizedAudioFake()

    def close(self) -> None:
        pass


class _AudioPlayerFake:
    def start(self, audio: SynthesizedAudio, playback_token: int) -> None:
        pass

    def stop(self) -> None:
        pass


class _MuteGateFake:
    def microphone_enabled(self) -> bool:
        return True


class _ClockFake:
    def monotonic(self) -> float:
        return 0.0


def _audio_evidence() -> AudioQualityEvidence:
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


def test_controller_constants_are_fixed_and_package_exports_nothing() -> None:
    assert {state.value for state in SpeechState} == {
        "idle",
        "recording",
        "recognizing",
        "thinking",
        "synthesizing",
        "playing",
    }
    assert {issue.value for issue in SpeechControllerIssue} == {
        "recording-failed",
        "recognition-rejected",
        "recognition-failed",
        "dialogue-failed",
        "synthesis-failed",
        "playback-failed",
        "cleanup-failed",
    }
    assert RECOGNITION_RETRY_PROMPT == "我没听清，再说一次吧"  # noqa: RUF001
    assert CONNECTION_RETRY_PROMPT == "现在暂时连不上，请稍后再试"  # noqa: RUF001
    assert (ROOT / "src/companion/speech/__init__.py").read_bytes() == b""


def test_recognition_input_is_frozen_and_defaults_to_no_recognizer_failure() -> None:
    recognition_input = RecognitionInput(
        audio=_audio_evidence(),
        vad_passed=True,
        text="测试",
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
        confidence_threshold=None,
    )

    assert recognition_input.recognizer_failed is False
    with pytest.raises(FrozenInstanceError):
        recognition_input.text = "changed"  # type: ignore[misc]


def test_controller_ports_accept_complete_adapter_shapes() -> None:
    assert isinstance(_RecordedAudioFake(), RecordedAudio)
    assert isinstance(_RecordingSessionFake(), RecordingSession)
    assert isinstance(_AudioRecorderFake(), AudioRecorder)
    assert isinstance(_SpeechRecognizerFake(), SpeechRecognizer)
    assert isinstance(_DialogueCallableFake(), DialogueCallable)
    assert isinstance(_SynthesizedAudioFake(), SynthesizedAudio)
    assert isinstance(_SpeechSynthesizerFake(), SpeechSynthesizer)
    assert isinstance(_AudioPlayerFake(), AudioPlayer)
    assert isinstance(_MuteGateFake(), MuteGate)
    assert isinstance(_ClockFake(), Clock)
