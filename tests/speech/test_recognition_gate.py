import hashlib
import json
import math
from pathlib import Path

import pytest

from companion.contracts import DialogueRequest, Transcript, TranscriptConfidenceStatus
from companion.speech.recognition_gate import (
    AudioQualityEvidence,
    RecognitionGateResult,
    RecognitionRejectionReason,
    gate_recognition,
)

ROOT = Path(__file__).resolve().parents[2]
QUALITY_CONTRACT = (
    ROOT / "docs" / "superpowers" / "research" / "2026-08-12-task1h-quality-gate-contract.json"
)
QUALITY_CONTRACT_SHA256 = "0f02603c0d13ebffc47dc0447cb9bce398ca966fb88df02bfb3a9a8c08bc735c"


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


def invoke_gate(**overrides: object) -> RecognitionGateResult:
    values: dict[str, object] = {
        "audio": valid_audio(),
        "vad_passed": True,
        "text": "synthetic transcript",
        "confidence": None,
        "confidence_status": TranscriptConfidenceStatus.UNAVAILABLE,
        "confidence_threshold": None,
        "recognizer_failed": False,
    }
    values.update(overrides)
    return gate_recognition(**values)


def test_quality_contract_digest_and_shape_are_frozen() -> None:
    payload = QUALITY_CONTRACT.read_bytes()
    contract = json.loads(payload)
    canonical_payload = payload.replace(b"\r\n", b"\n")

    assert hashlib.sha256(canonical_payload).hexdigest() == QUALITY_CONTRACT_SHA256
    assert contract["schema_version"] == 1
    assert contract["clean_sample_count"] == 20
    assert contract["quality_thresholds"] == {
        "channels": 1,
        "maximum_clipping_fraction": "0.01",
        "maximum_duration_ms": 30_000,
        "minimum_duration_ms": 300,
        "minimum_estimated_snr_db": "10.0",
        "minimum_peak_abs": "0.01",
        "minimum_rms": "0.003",
        "sample_rate_hz": 16_000,
        "sample_width_bytes": 2,
    }
    assert [(item["id"], item["expected"]) for item in contract["perturbations"]] == [
        ("too-short-200ms", "invalid-audio"),
        ("unsupported-8khz", "invalid-audio"),
        ("unsupported-stereo", "invalid-audio"),
        ("attenuation-gain-0.001", "audio-quality-rejected"),
        ("hard-clip-gain-20", "audio-quality-rejected"),
        ("additive-noise-snr-5db", "audio-quality-rejected"),
    ]


def test_result_requires_exactly_one_transcript_or_rejection() -> None:
    transcript = Transcript(
        text="synthetic",
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
    )

    with pytest.raises(ValueError):
        RecognitionGateResult(transcript=None, rejection_reason=None)
    with pytest.raises(ValueError):
        RecognitionGateResult(
            transcript=transcript,
            rejection_reason=RecognitionRejectionReason.LOW_CONFIDENCE,
        )


def test_private_result_and_audio_boundaries_reject_subclasses() -> None:
    class TranscriptSubclass(Transcript):
        pass

    class AudioEvidenceSubclass(AudioQualityEvidence):
        pass

    transcript = TranscriptSubclass(
        text="synthetic",
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
    )

    with pytest.raises(TypeError):
        RecognitionGateResult(transcript=transcript, rejection_reason=None)
    with pytest.raises(TypeError):
        invoke_gate(audio=AudioEvidenceSubclass(**valid_audio().__dict__))


def test_accepted_result_repr_does_not_expose_transcript_text() -> None:
    result = invoke_gate()

    assert "synthetic transcript" not in repr(result)
    assert result.rejection_reason is None
    assert result.transcript is not None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"recognizer_failed": True}, RecognitionRejectionReason.RECOGNIZER_FAILED),
        (
            {"audio": valid_audio(sample_rate_hz=8_000)},
            RecognitionRejectionReason.INVALID_AUDIO,
        ),
        ({"vad_passed": False}, RecognitionRejectionReason.SILENCE_OR_NOISE),
        (
            {"audio": valid_audio(rms=0.0029)},
            RecognitionRejectionReason.AUDIO_QUALITY_REJECTED,
        ),
        ({"text": "  "}, RecognitionRejectionReason.BLANK_TRANSCRIPT),
        (
            {
                "confidence": 0.49,
                "confidence_status": TranscriptConfidenceStatus.AVAILABLE,
                "confidence_threshold": 0.5,
            },
            RecognitionRejectionReason.LOW_CONFIDENCE,
        ),
    ],
)
def test_rejection_order_returns_no_transcript(
    overrides: dict[str, object], expected: RecognitionRejectionReason
) -> None:
    result = invoke_gate(**overrides)

    assert result == RecognitionGateResult(transcript=None, rejection_reason=expected)


def test_recognizer_failure_wins_before_other_rejections() -> None:
    result = invoke_gate(
        recognizer_failed=True,
        audio=valid_audio(sample_rate_hz=8_000),
        vad_passed=False,
        text="",
    )

    assert result.rejection_reason is RecognitionRejectionReason.RECOGNIZER_FAILED


@pytest.mark.parametrize(
    ("overrides", "expected_reason", "expected_transcript"),
    [
        (
            {"recognizer_failed": True, "text": ""},
            RecognitionRejectionReason.RECOGNIZER_FAILED,
            None,
        ),
        ({"text": ""}, RecognitionRejectionReason.BLANK_TRANSCRIPT, None),
        (
            {
                "confidence": 0.49,
                "confidence_status": TranscriptConfidenceStatus.AVAILABLE,
                "confidence_threshold": 0.5,
            },
            RecognitionRejectionReason.LOW_CONFIDENCE,
            None,
        ),
        (
            {
                "confidence": 0.5,
                "confidence_status": TranscriptConfidenceStatus.AVAILABLE,
                "confidence_threshold": 0.5,
            },
            None,
            {
                "text": "synthetic transcript",
                "confidence": 0.5,
                "confidence_status": "available",
            },
        ),
        (
            {},
            None,
            {
                "text": "synthetic transcript",
                "confidence": None,
                "confidence_status": "unavailable",
            },
        ),
    ],
)
def test_full_gate_result_states_and_transcript_model_dumps_remain_frozen(
    overrides: dict[str, object],
    expected_reason: RecognitionRejectionReason | None,
    expected_transcript: dict[str, object] | None,
) -> None:
    result = invoke_gate(**overrides)

    assert result.rejection_reason is expected_reason
    if expected_transcript is None:
        assert result.transcript is None
    else:
        assert result.transcript is not None
        assert result.transcript.model_dump(mode="json") == expected_transcript


@pytest.mark.parametrize(
    "audio",
    [
        valid_audio(sample_rate_hz=8_000),
        valid_audio(channels=2),
        valid_audio(sample_width_bytes=1),
        valid_audio(frame_count=4_799),
        valid_audio(frame_count=480_001),
        valid_audio(rms=float("nan")),
        valid_audio(peak_abs=float("inf")),
        valid_audio(clipping_fraction=-0.01),
    ],
)
def test_invalid_audio_format_duration_or_metrics_fail_closed(
    audio: AudioQualityEvidence,
) -> None:
    assert invoke_gate(audio=audio).rejection_reason is RecognitionRejectionReason.INVALID_AUDIO


@pytest.mark.parametrize(
    "audio",
    [
        valid_audio(rms=0.0029),
        valid_audio(peak_abs=0.0099),
        valid_audio(clipping_fraction=0.0101),
        valid_audio(estimated_snr_db=9.99),
        valid_audio(estimated_snr_db=None),
        valid_audio(estimated_snr_db=float("nan")),
    ],
)
def test_signal_quality_rejections_are_not_asr_confidence(
    audio: AudioQualityEvidence,
) -> None:
    result = invoke_gate(audio=audio)

    assert result.rejection_reason is RecognitionRejectionReason.AUDIO_QUALITY_REJECTED
    assert result.transcript is None


@pytest.mark.parametrize(
    "audio",
    [
        valid_audio(frame_count=4_800),
        valid_audio(frame_count=480_000),
        valid_audio(rms=0.003),
        valid_audio(peak_abs=0.01),
        valid_audio(clipping_fraction=0.01),
        valid_audio(estimated_snr_db=10.0),
    ],
)
def test_exact_quality_boundaries_are_accepted(audio: AudioQualityEvidence) -> None:
    result = invoke_gate(audio=audio)

    assert result.rejection_reason is None
    assert result.transcript is not None
    assert result.transcript.confidence is None
    assert result.transcript.confidence_status is TranscriptConfidenceStatus.UNAVAILABLE


def test_available_confidence_keeps_explicit_runtime_threshold() -> None:
    accepted = invoke_gate(
        confidence=0.5,
        confidence_status=TranscriptConfidenceStatus.AVAILABLE,
        confidence_threshold=0.5,
    )

    assert accepted.transcript is not None
    assert accepted.transcript.confidence == 0.5
    assert accepted.transcript.confidence_status is TranscriptConfidenceStatus.AVAILABLE


@pytest.mark.parametrize("threshold", [None, True, -0.01, 1.01, math.nan, math.inf])
def test_available_confidence_requires_explicit_finite_unit_threshold(
    threshold: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        invoke_gate(
            confidence=0.8,
            confidence_status=TranscriptConfidenceStatus.AVAILABLE,
            confidence_threshold=threshold,
        )


@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_unavailable_confidence_rejects_threshold_without_fabricating_score(
    threshold: float,
) -> None:
    with pytest.raises(ValueError):
        invoke_gate(confidence_threshold=threshold)


@pytest.mark.parametrize("field", ["sample_rate_hz", "channels", "frame_count", "rms"])
def test_audio_evidence_rejects_boolean_numeric_values(field: str) -> None:
    with pytest.raises(TypeError):
        valid_audio(**{field: True})


def test_rejection_reasons_are_fixed_safe_values() -> None:
    assert {reason.value for reason in RecognitionRejectionReason} == {
        "silence-or-noise",
        "invalid-audio",
        "audio-quality-rejected",
        "blank-transcript",
        "low-confidence",
        "recognizer-failed",
    }


def test_dialogue_callable_stays_zero_for_every_rejection_and_uses_only_text() -> None:
    calls: list[DialogueRequest] = []

    def call_dialogue_after_acceptance(result: RecognitionGateResult) -> None:
        if result.transcript is None:
            return
        calls.append(
            DialogueRequest(
                user_id="synthetic-user",
                text=result.transcript.text,
                recent_context=[],
            )
        )

    rejected_results = [
        invoke_gate(recognizer_failed=True),
        invoke_gate(audio=valid_audio(sample_rate_hz=8_000)),
        invoke_gate(vad_passed=False),
        invoke_gate(audio=valid_audio(rms=0.0029)),
        invoke_gate(text=""),
        invoke_gate(
            confidence=0.49,
            confidence_status=TranscriptConfidenceStatus.AVAILABLE,
            confidence_threshold=0.5,
        ),
    ]
    for result in rejected_results:
        call_dialogue_after_acceptance(result)

    assert calls == []

    call_dialogue_after_acceptance(invoke_gate())

    assert len(calls) == 1
    assert calls[0].model_dump(mode="json") == {
        "user_id": "synthetic-user",
        "text": "synthetic transcript",
        "recent_context": [],
    }
