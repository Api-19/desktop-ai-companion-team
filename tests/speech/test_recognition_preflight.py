import math

import pytest

from companion.contracts import TranscriptConfidenceStatus
from companion.speech.recognition_gate import (
    AudioQualityEvidence,
    RecognitionRejectionReason,
    gate_recognition,
    preflight_recognition_audio,
)


def valid_audio(**overrides: object) -> AudioQualityEvidence:
    values: dict[str, object] = {
        "sample_rate_hz": 16_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "frame_count": 16_000,
        "rms": 0.1,
        "peak_abs": 0.5,
        "clipping_fraction": 0.0,
        "estimated_snr_db": 20.0,
    }
    values.update(overrides)
    return AudioQualityEvidence(**values)


PRE_ASR_CASES = [
    (valid_audio(sample_rate_hz=8_000), False, RecognitionRejectionReason.INVALID_AUDIO),
    (valid_audio(channels=2), True, RecognitionRejectionReason.INVALID_AUDIO),
    (valid_audio(sample_width_bytes=1), True, RecognitionRejectionReason.INVALID_AUDIO),
    (valid_audio(frame_count=4_799), True, RecognitionRejectionReason.INVALID_AUDIO),
    (valid_audio(frame_count=480_001), True, RecognitionRejectionReason.INVALID_AUDIO),
    (valid_audio(rms=math.nan), True, RecognitionRejectionReason.INVALID_AUDIO),
    (valid_audio(peak_abs=math.inf), True, RecognitionRejectionReason.INVALID_AUDIO),
    (valid_audio(clipping_fraction=-0.01), True, RecognitionRejectionReason.INVALID_AUDIO),
    (valid_audio(), False, RecognitionRejectionReason.SILENCE_OR_NOISE),
    (valid_audio(rms=0.0029), True, RecognitionRejectionReason.AUDIO_QUALITY_REJECTED),
    (valid_audio(peak_abs=0.0099), True, RecognitionRejectionReason.AUDIO_QUALITY_REJECTED),
    (
        valid_audio(clipping_fraction=0.0101),
        True,
        RecognitionRejectionReason.AUDIO_QUALITY_REJECTED,
    ),
    (valid_audio(estimated_snr_db=None), True, RecognitionRejectionReason.AUDIO_QUALITY_REJECTED),
    (
        valid_audio(estimated_snr_db=math.nan),
        True,
        RecognitionRejectionReason.AUDIO_QUALITY_REJECTED,
    ),
    (valid_audio(estimated_snr_db=9.99), True, RecognitionRejectionReason.AUDIO_QUALITY_REJECTED),
    (valid_audio(estimated_snr_db=10.0), True, None),
]


@pytest.mark.parametrize(("audio", "vad_passed", "expected"), PRE_ASR_CASES)
def test_preflight_matches_full_gate_before_text(
    audio: AudioQualityEvidence,
    vad_passed: bool,
    expected: RecognitionRejectionReason | None,
) -> None:
    assert preflight_recognition_audio(audio=audio, vad_passed=vad_passed) is expected

    result = gate_recognition(
        audio=audio,
        vad_passed=vad_passed,
        text="synthetic accepted text",
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
        confidence_threshold=None,
    )

    assert result.rejection_reason is expected


@pytest.mark.parametrize("audio", [None, object()])
def test_preflight_and_full_gate_reject_non_evidence_audio(audio: object) -> None:
    with pytest.raises(TypeError, match="audio must be AudioQualityEvidence"):
        preflight_recognition_audio(audio=audio, vad_passed=True)
    with pytest.raises(TypeError, match="audio must be AudioQualityEvidence"):
        gate_recognition(
            audio=audio,
            vad_passed=True,
            text="synthetic accepted text",
            confidence=None,
            confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
            confidence_threshold=None,
        )


@pytest.mark.parametrize("vad_passed", [None, 1, "true"])
def test_preflight_and_full_gate_reject_non_boolean_vad(vad_passed: object) -> None:
    with pytest.raises(TypeError, match="vad_passed must be a bool"):
        preflight_recognition_audio(audio=valid_audio(), vad_passed=vad_passed)
    with pytest.raises(TypeError, match="vad_passed must be a bool"):
        gate_recognition(
            audio=valid_audio(),
            vad_passed=vad_passed,
            text="synthetic accepted text",
            confidence=None,
            confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
            confidence_threshold=None,
        )
