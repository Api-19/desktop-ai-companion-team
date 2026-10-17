from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from companion.contracts import TranscriptConfidenceStatus
from companion.speech.recognition_gate import AudioQualityEvidence


class SpeechState(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    RECOGNIZING = "recognizing"
    THINKING = "thinking"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"


class SpeechControllerIssue(StrEnum):
    RECORDING_FAILED = "recording-failed"
    RECOGNITION_REJECTED = "recognition-rejected"
    RECOGNITION_FAILED = "recognition-failed"
    DIALOGUE_FAILED = "dialogue-failed"
    SYNTHESIS_FAILED = "synthesis-failed"
    PLAYBACK_FAILED = "playback-failed"
    CLEANUP_FAILED = "cleanup-failed"


class SpeechTurnFailure(StrEnum):
    PROVIDER_TIMEOUT = "provider-timeout"
    PROVIDER_UNAVAILABLE = "provider-unavailable"
    INVALID_PROVIDER_RESPONSE = "invalid-provider-response"


class TurnTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    played: bool
    issue: str | None
    stop_ms: int = Field(ge=0)
    recognize_ms: int | None = Field(ge=0)
    dialogue_ms: int | None = Field(ge=0)
    synthesize_ms: int | None = Field(ge=0)
    playback_start_ms: int | None = Field(ge=0)
    total_ms: int = Field(ge=0)


RECOGNITION_RETRY_PROMPT = "我没听清，再说一次吧"  # noqa: RUF001
CONNECTION_RETRY_PROMPT = "现在暂时连不上，请稍后再试"  # noqa: RUF001
EMPTY_REPLY_PROMPT = "这次没有可朗读的回复，换个说法再试吧。"  # noqa: RUF001


@dataclass(frozen=True)
class RecognitionInput:
    audio: AudioQualityEvidence
    vad_passed: bool
    text: str | None
    confidence: float | None
    confidence_status: TranscriptConfidenceStatus | None
    confidence_threshold: float | None
    recognizer_failed: bool = False
