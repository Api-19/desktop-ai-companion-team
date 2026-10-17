from collections import Counter
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from companion.contracts.base import ContractModel, TimezoneRequiredModel


class CandidateCategory(StrEnum):
    GREETING = "greeting"
    COURTESY = "courtesy"
    MEALS = "meals"
    SLEEP = "sleep"
    WEATHER = "weather"
    HOUSEHOLD = "household"
    SHOPPING = "shopping"
    WORK_STUDY = "work_study"
    GENERAL_EMOTION = "general_emotion"
    GENTLE_AFFECTION = "gentle_affection"
    SHORT_REPLY = "short_reply"
    EMOJI = "emoji"
    CALIBRATED_DAILY = "calibrated_daily"


class CandidateDecision(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"
    UNKNOWN = "unknown"


class CandidateRejectReason(StrEnum):
    NON_ORIGINAL_OR_UNSUITABLE = "non_original_or_unsuitable"
    PERSONAL_BELIEF_OR_COMMITMENT = "personal_belief_or_commitment"
    GENDERED_OR_ROLE_LABEL = "gendered_or_role_label"
    MISCONDUCT = "misconduct"
    APPEARANCE_OR_BODY_PRIVATE = "appearance_or_body_private"
    SEXUAL_OR_SUGGESTIVE = "sexual_or_suggestive"
    VIOLENT_OR_DEGRADING = "violent_or_degrading"
    BODILY_PRIVATE = "bodily_private"
    RELATIONSHIP_CONFLICT = "relationship_conflict"
    HEALTH_PRIVATE = "health_private"
    FAMILY_OR_THIRD_PARTY_PRIVATE = "family_or_third_party_private"
    FINANCIAL_OR_CREDENTIAL = "financial_or_credential"
    IDENTITY_OR_CONTACT = "identity_or_contact"
    PRECISE_LOCATION_OR_SCHEDULE = "precise_location_or_schedule"
    PRIVATE_EVENT = "private_event"
    NON_TEXT = "non_text"
    SENSITIVE_CONTEXT = "sensitive_context"


class SafeStyleCandidate(ContractModel):
    text: str = Field(min_length=1)
    category: CandidateCategory
    scene_tags: list[str]

    @model_validator(mode="after")
    def require_category_scene_tag(self) -> Self:
        if self.scene_tags != [self.category.value]:
            raise ValueError("candidate scene tag must match its category")
        return self


class CandidateSelectionAudit(TimezoneRequiredModel):
    processed_at: datetime
    input_count: int = Field(ge=0)
    owner_input_count: int = Field(ge=0)
    dropped_other_count: int = Field(ge=0)
    allowed_category_counts: dict[str, int]
    rejection_reason_counts: dict[str, int]
    unknown_count: int = Field(ge=0)
    deduplicated_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    succeeded: Literal[True]

    @model_validator(mode="after")
    def require_consistent_aggregates(self) -> Self:
        approved_categories = {category.value for category in CandidateCategory}
        approved_reasons = {reason.value for reason in CandidateRejectReason}
        if (
            not set(self.allowed_category_counts).issubset(approved_categories)
            or not set(self.rejection_reason_counts).issubset(approved_reasons)
            or any(count <= 0 for count in self.allowed_category_counts.values())
            or any(count <= 0 for count in self.rejection_reason_counts.values())
        ):
            raise ValueError("candidate audit aggregates are invalid")
        if self.input_count != self.owner_input_count + self.dropped_other_count:
            raise ValueError("candidate audit input partition is invalid")
        if sum(self.allowed_category_counts.values()) != self.output_count:
            raise ValueError("candidate audit allowed counts are invalid")
        owner_outcome_count = (
            self.output_count
            + sum(self.rejection_reason_counts.values())
            + self.unknown_count
            + self.deduplicated_count
        )
        if self.owner_input_count != owner_outcome_count:
            raise ValueError("candidate audit owner outcome partition is invalid")
        return self


class SafeCandidatePackage(TimezoneRequiredModel):
    created_at: datetime
    candidates: list[SafeStyleCandidate]
    audit: CandidateSelectionAudit

    @model_validator(mode="after")
    def require_consistent_package(self) -> Self:
        category_counts = Counter(candidate.category.value for candidate in self.candidates)
        if len(self.candidates) != self.audit.output_count or dict(category_counts) != dict(
            self.audit.allowed_category_counts
        ):
            raise ValueError("candidate package counts are invalid")
        if self.created_at != self.audit.processed_at:
            raise ValueError("candidate package timestamps are inconsistent")
        return self
