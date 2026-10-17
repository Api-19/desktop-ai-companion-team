import math
from dataclasses import dataclass, field
from enum import StrEnum

from companion.contracts import Transcript, TranscriptConfidenceStatus

SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
MINIMUM_DURATION_MS = 300
MAXIMUM_DURATION_MS = 30_000
MINIMUM_RMS = 0.003
MINIMUM_PEAK_ABS = 0.01
MAXIMUM_CLIPPING_FRACTION = 0.01
MINIMUM_ESTIMATED_SNR_DB = 10.0


class RecognitionRejectionReason(StrEnum):
    SILENCE_OR_NOISE = "silence-or-noise"
    INVALID_AUDIO = "invalid-audio"
    AUDIO_QUALITY_REJECTED = "audio-quality-rejected"
    BLANK_TRANSCRIPT = "blank-transcript"
    LOW_CONFIDENCE = "low-confidence"
    RECOGNIZER_FAILED = "recognizer-failed"


def _strict_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    return value


def _strict_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a float or int")
    return float(value)


@dataclass(frozen=True)
class AudioQualityEvidence:
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    rms: float
    peak_abs: float
    clipping_fraction: float
    estimated_snr_db: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sample_rate_hz", _strict_int("sample_rate_hz", self.sample_rate_hz)
        )
        object.__setattr__(self, "channels", _strict_int("channels", self.channels))
        object.__setattr__(
            self,
            "sample_width_bytes",
            _strict_int("sample_width_bytes", self.sample_width_bytes),
        )
        object.__setattr__(self, "frame_count", _strict_int("frame_count", self.frame_count))
        object.__setattr__(self, "rms", _strict_float("rms", self.rms))
        object.__setattr__(self, "peak_abs", _strict_float("peak_abs", self.peak_abs))
        object.__setattr__(
            self,
            "clipping_fraction",
            _strict_float("clipping_fraction", self.clipping_fraction),
        )
        if self.estimated_snr_db is not None:
            object.__setattr__(
                self,
                "estimated_snr_db",
                _strict_float("estimated_snr_db", self.estimated_snr_db),
            )


@dataclass(frozen=True)
class RecognitionGateResult:
    transcript: Transcript | None = field(repr=False)
    rejection_reason: RecognitionRejectionReason | None

    def __post_init__(self) -> None:
        if (self.transcript is None) == (self.rejection_reason is None):
            raise ValueError("exactly one result state is required")
        if self.transcript is not None and type(self.transcript) is not Transcript:
            raise TypeError("transcript must be a Transcript")
        if (
            self.rejection_reason is not None
            and type(self.rejection_reason) is not RecognitionRejectionReason
        ):
            raise TypeError("rejection_reason must be a RecognitionRejectionReason")


def _rejected(reason: RecognitionRejectionReason) -> RecognitionGateResult:
    return RecognitionGateResult(transcript=None, rejection_reason=reason)


def _invalid_audio(audio: AudioQualityEvidence) -> bool:
    if (
        audio.sample_rate_hz != SAMPLE_RATE_HZ
        or audio.channels != CHANNELS
        or audio.sample_width_bytes != SAMPLE_WIDTH_BYTES
    ):
        return True
    minimum_frames = SAMPLE_RATE_HZ * MINIMUM_DURATION_MS // 1_000
    maximum_frames = SAMPLE_RATE_HZ * MAXIMUM_DURATION_MS // 1_000
    if not minimum_frames <= audio.frame_count <= maximum_frames:
        return True
    if not all(
        math.isfinite(value) for value in (audio.rms, audio.peak_abs, audio.clipping_fraction)
    ):
        return True
    return not (
        0.0 <= audio.rms <= 1.0
        and 0.0 <= audio.peak_abs <= 1.0
        and 0.0 <= audio.clipping_fraction <= 1.0
    )


def _quality_rejected(audio: AudioQualityEvidence) -> bool:
    snr = audio.estimated_snr_db
    return (
        audio.rms < MINIMUM_RMS
        or audio.peak_abs < MINIMUM_PEAK_ABS
        or audio.clipping_fraction > MAXIMUM_CLIPPING_FRACTION
        or snr is None
        or not math.isfinite(snr)
        or snr < MINIMUM_ESTIMATED_SNR_DB
    )


def _validate_boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _validate_threshold(value: object) -> float:
    threshold = _strict_float("confidence_threshold", value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be finite and between 0 and 1")
    return threshold


def preflight_recognition_audio(
    *,
    audio: AudioQualityEvidence,
    vad_passed: bool,
) -> RecognitionRejectionReason | None:
    vad_passed = _validate_boolean("vad_passed", vad_passed)
    if type(audio) is not AudioQualityEvidence:
        raise TypeError("audio must be AudioQualityEvidence")
    if _invalid_audio(audio):
        return RecognitionRejectionReason.INVALID_AUDIO
    if not vad_passed:
        return RecognitionRejectionReason.SILENCE_OR_NOISE
    if _quality_rejected(audio):
        return RecognitionRejectionReason.AUDIO_QUALITY_REJECTED
    return None


def gate_recognition(
    *,
    audio: AudioQualityEvidence,
    vad_passed: bool,
    text: str | None = None,
    confidence: float | None = None,
    confidence_status: TranscriptConfidenceStatus | None = None,
    confidence_threshold: float | None = None,
    recognizer_failed: bool = False,
) -> RecognitionGateResult:
    recognizer_failed = _validate_boolean("recognizer_failed", recognizer_failed)
    vad_passed = _validate_boolean("vad_passed", vad_passed)
    if type(audio) is not AudioQualityEvidence:
        raise TypeError("audio must be AudioQualityEvidence")

    if recognizer_failed:
        return _rejected(RecognitionRejectionReason.RECOGNIZER_FAILED)
    preflight_rejection = preflight_recognition_audio(audio=audio, vad_passed=vad_passed)
    if preflight_rejection is not None:
        return _rejected(preflight_rejection)
    if not isinstance(text, str):
        return _rejected(RecognitionRejectionReason.RECOGNIZER_FAILED)
    if not text.strip():
        return _rejected(RecognitionRejectionReason.BLANK_TRANSCRIPT)
    if not isinstance(confidence_status, TranscriptConfidenceStatus):
        raise TypeError("confidence_status must be explicit")

    if confidence_status is TranscriptConfidenceStatus.AVAILABLE:
        threshold = _validate_threshold(confidence_threshold)
        transcript = Transcript(
            text=text,
            confidence=confidence,
            confidence_status=confidence_status,
        )
        if transcript.confidence is None:
            raise ValueError("available confidence requires a value")
        if transcript.confidence < threshold:
            return _rejected(RecognitionRejectionReason.LOW_CONFIDENCE)
    else:
        if confidence is not None or confidence_threshold is not None:
            raise ValueError("unavailable confidence cannot use a value or threshold")
        transcript = Transcript(
            text=text,
            confidence=None,
            confidence_status=confidence_status,
        )

    return RecognitionGateResult(transcript=transcript, rejection_reason=None)
