from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from companion.contracts.base import ContractModel, TimezoneRequiredModel
from companion.contracts.chat import ChatRecord
from companion.contracts.privacy import RedactedRecord


class RiskCategory(StrEnum):
    INTIMATE = "intimate"
    SEXUAL = "sexual"
    HEALTH = "health"
    FAMILY = "family"
    FINANCIAL = "financial"
    CREDENTIAL = "credential"
    CONFLICT = "conflict"
    LOCATION = "location"
    PRIVATE_EVENT = "private_event"


class ReviewDecisionValue(StrEnum):
    APPROVE = "approve"
    EXCLUDE = "exclude"


class ReviewSegment(ContractModel):
    segment_id: str = Field(min_length=1)
    source_segment_id: str = Field(min_length=1)
    records: list[ChatRecord]
    risk_categories: list[RiskCategory]
    reason_labels: list[str]
    recommended_decision: Literal[ReviewDecisionValue.EXCLUDE]


class PrivacyReviewPackage(TimezoneRequiredModel):
    dataset_fingerprint: str = Field(min_length=64, max_length=64)
    scanned_segment_count: int = Field(ge=0)
    risk_segments: list[ReviewSegment]
    created_at: datetime


class ReviewDecision(ContractModel):
    segment_id: str = Field(min_length=1)
    decision: ReviewDecisionValue


class PrivacyAuditEvent(TimezoneRequiredModel):
    event_type: Literal["privacy_batch_finalized"]
    processed_at: datetime
    input_count: int = Field(ge=0)
    scanned_segment_count: int = Field(ge=0)
    risk_segment_count: int = Field(ge=0)
    approved_risk_segment_count: int = Field(ge=0)
    excluded_risk_segment_count: int = Field(ge=0)
    auto_cleared_segment_count: int = Field(ge=0)
    owner_output_count: int = Field(ge=0)
    dropped_other_count: int = Field(ge=0)
    redaction_counts: dict[str, int]
    succeeded: Literal[True]


class PrivacyBatchResult(ContractModel):
    records: list[RedactedRecord]
    audit: PrivacyAuditEvent
