import math
from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from companion.contracts.base import ContractModel


class TranscriptConfidenceStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class Transcript(ContractModel):
    text: str
    confidence: float | None = Field(ge=0, le=1)
    confidence_status: TranscriptConfidenceStatus

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence_source_type(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("confidence must be a real float or int")
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return float(value)

    @model_validator(mode="after")
    def validate_confidence_state(self) -> Self:
        if self.confidence_status is TranscriptConfidenceStatus.AVAILABLE:
            if self.confidence is None:
                raise ValueError("available confidence requires a value")
        elif self.confidence is not None:
            raise ValueError("unavailable confidence requires None")
        return self
