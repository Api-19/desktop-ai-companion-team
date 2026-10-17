from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import Field, field_validator, model_validator

from companion.contracts.base import ContractModel, TimezoneRequiredModel
from companion.privacy.candidate_io import (
    _sanitized_candidate_failure,
    _validated_runtime_terms,
)
from companion.privacy.candidate_models import CandidateCategory
from companion.privacy.errors import PrivacyValidationError


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibration_submission_id(*, scenario_id: str, category: CandidateCategory, intent: str) -> str:
    return _canonical_hash(
        {
            "category": category.value,
            "intent": intent.strip(),
            "scenario_id": scenario_id.strip(),
        }
    )


def calibration_candidate_id(*, category: CandidateCategory, intent: str, text: str) -> str:
    return _canonical_hash(
        {
            "category": category.value,
            "intent": intent.strip(),
            "text": text.strip(),
        }
    )


def calibration_review_fingerprint(items: Sequence[ContractModel]) -> str:
    return _canonical_hash([item.model_dump(mode="json") for item in items])


def calibration_review_package_fingerprint(
    items: Sequence[ContractModel], *, submitted_at: datetime
) -> str:
    return _canonical_hash(
        {
            "content_fingerprint": calibration_review_fingerprint(items),
            "submitted_at": submitted_at.isoformat(),
        }
    )


def calibration_identity_config_fingerprint(
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> str:
    names: tuple[str, ...] | None = None
    nicknames: tuple[str, ...] | None = None
    try:
        names, nicknames = _validated_runtime_terms(
            true_name_terms,
            approved_nicknames,
        )
        return _canonical_hash(
            {
                "approved_nicknames": sorted({term.strip() for term in nicknames}),
                "true_name_terms": sorted({term.strip() for term in names}),
            }
        )
    except Exception as error:
        failure = _sanitized_candidate_failure(
            error,
            fallback_code="invalid_candidate_terms",
        )
    del true_name_terms, approved_nicknames, names, nicknames
    raise PrivacyValidationError(failure[0], failure[1]) from None


class CalibrationReviewSubmission(ContractModel):
    submission_id: str = Field(min_length=64, max_length=64)
    scenario_id: str = Field(min_length=1)
    category: CandidateCategory
    intent: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=160)
    source: Literal["explicit_owner_ideal_response"]

    @field_validator("scenario_id", "intent", "text", mode="before")
    @classmethod
    def strip_outer_whitespace(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_stable_submission_id(self) -> Self:
        expected = calibration_submission_id(
            scenario_id=self.scenario_id,
            category=self.category,
            intent=self.intent,
        )
        if self.submission_id != expected:
            raise ValueError("calibration submission ID is invalid")
        return self

    @property
    def content_fingerprint(self) -> str:
        return calibration_review_fingerprint([self])


PRECISE_INTENT_CALIBRATION_PURPOSE = "precise_intent_owner_calibration"
PRECISE_INTENT_CALIBRATION_MAPPING = (
    ("greeting-1", CandidateCategory.GREETING, "morning_greeting"),
    ("meals-1", CandidateCategory.MEALS, "missed_meal_care"),
    (
        "household-2",
        CandidateCategory.HOUSEHOLD,
        "unfinished_household_reassurance",
    ),
    ("shopping-1", CandidateCategory.SHOPPING, "shopping_comparison_request"),
    ("work_study-2", CandidateCategory.WORK_STUDY, "work_fatigue_care"),
    (
        "general_emotion-1",
        CandidateCategory.GENERAL_EMOTION,
        "positive_emotion_share",
    ),
    (
        "general_emotion-2",
        CandidateCategory.GENERAL_EMOTION,
        "negative_emotion_check_in",
    ),
    (
        "gentle_affection-1",
        CandidateCategory.GENTLE_AFFECTION,
        "missed_contact_affection",
    ),
)

FIVE_FIXED_INTENT_CALIBRATION_PURPOSE = "five_fixed_intent_owner_calibration"
FIVE_FIXED_INTENT_CALIBRATION_MAPPING = (
    ("meals-2", CandidateCategory.MEALS, "not_eaten_yet_care"),
    ("sleep-2", CandidateCategory.SLEEP, "sleep_reminder"),
    ("weather-2", CandidateCategory.WEATHER, "rain_reminder"),
    (
        "household-1",
        CandidateCategory.HOUSEHOLD,
        "household_completed_acknowledgement",
    ),
    (
        "gentle_affection-2",
        CandidateCategory.GENTLE_AFFECTION,
        "going_out_care",
    ),
)

SLEEP1_INCREMENTAL_CALIBRATION_PURPOSE = "sleep1_incremental_owner_calibration"
SLEEP1_INCREMENTAL_CALIBRATION_MAPPING = (("sleep-1", CandidateCategory.SLEEP, "sleep_farewell"),)


class PreciseIntentCalibrationHandoff(TimezoneRequiredModel):
    expected_mapping: ClassVar[set[tuple[str, CandidateCategory, str]]] = set(
        PRECISE_INTENT_CALIBRATION_MAPPING
    )

    purpose: Literal["precise_intent_owner_calibration"]
    version: Literal[1]
    created_at: datetime
    item_count: Literal[8]
    submissions: list[CalibrationReviewSubmission] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def require_exact_mapping(self) -> Self:
        actual = {
            (submission.scenario_id, submission.category, submission.intent)
            for submission in self.submissions
        }
        submission_ids = [submission.submission_id for submission in self.submissions]
        if (
            actual != self.expected_mapping
            or len(actual) != 8
            or len(submission_ids) != len(set(submission_ids))
        ):
            raise ValueError("precise intent calibration handoff is invalid")
        return self


class FiveFixedIntentCalibrationHandoff(TimezoneRequiredModel):
    expected_mapping: ClassVar[set[tuple[str, CandidateCategory, str]]] = set(
        FIVE_FIXED_INTENT_CALIBRATION_MAPPING
    )

    purpose: Literal["five_fixed_intent_owner_calibration"]
    version: Literal[1]
    created_at: datetime
    item_count: Literal[5]
    submissions: list[CalibrationReviewSubmission] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def require_exact_mapping(self) -> Self:
        actual = {
            (submission.scenario_id, submission.category, submission.intent)
            for submission in self.submissions
        }
        submission_ids = [submission.submission_id for submission in self.submissions]
        if (
            actual != self.expected_mapping
            or len(actual) != 5
            or len(submission_ids) != len(set(submission_ids))
        ):
            raise ValueError("five fixed intent calibration handoff is invalid")
        return self


class Sleep1IncrementalCalibrationHandoff(TimezoneRequiredModel):
    expected_mapping: ClassVar[set[tuple[str, CandidateCategory, str]]] = set(
        SLEEP1_INCREMENTAL_CALIBRATION_MAPPING
    )

    purpose: Literal["sleep1_incremental_owner_calibration"]
    version: Literal[1]
    created_at: datetime
    item_count: Literal[1]
    submissions: list[CalibrationReviewSubmission] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def require_exact_mapping(self) -> Self:
        actual = {
            (submission.scenario_id, submission.category, submission.intent)
            for submission in self.submissions
        }
        if actual != self.expected_mapping or len(actual) != 1:
            raise ValueError("sleep-1 incremental calibration handoff is invalid")
        return self


class CalibrationReviewPackage(TimezoneRequiredModel):
    submitted_at: datetime
    content_fingerprint: str = Field(min_length=64, max_length=64)
    submission_count: int = Field(ge=1)
    submissions: list[CalibrationReviewSubmission] = Field(min_length=1)

    @model_validator(mode="after")
    def require_consistent_review_package(self) -> Self:
        submission_ids = [submission.submission_id for submission in self.submissions]
        if (
            self.submission_count != len(self.submissions)
            or len(submission_ids) != len(set(submission_ids))
            or self.content_fingerprint != calibration_review_fingerprint(self.submissions)
        ):
            raise ValueError("calibration review package is invalid")
        return self


class CalibrationReviewDecisionValue(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class CalibrationReviewDecision(TimezoneRequiredModel):
    submission_id: str = Field(min_length=64, max_length=64)
    submission_fingerprint: str = Field(min_length=64, max_length=64)
    review_package_fingerprint: str = Field(min_length=64, max_length=64)
    decided_at: datetime
    decision: CalibrationReviewDecisionValue


class CalibrationCandidate(ContractModel):
    candidate_id: str = Field(min_length=64, max_length=64)
    category: CandidateCategory
    intent: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=160)

    @field_validator("intent", "text", mode="before")
    @classmethod
    def strip_outer_whitespace(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_stable_candidate_id(self) -> Self:
        expected = calibration_candidate_id(
            category=self.category,
            intent=self.intent,
            text=self.text,
        )
        if self.candidate_id != expected:
            raise ValueError("calibration candidate ID is invalid")
        return self


class CalibrationCandidateAudit(TimezoneRequiredModel):
    reviewed_at: datetime
    submission_count: int = Field(ge=0)
    approved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    privacy_rejected_count: int = Field(ge=0)
    revoked_count: int = Field(default=0, ge=0)
    output_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_audit(self) -> Self:
        if (
            self.submission_count
            != self.approved_count
            + self.rejected_count
            + self.privacy_rejected_count
            + self.revoked_count
            or self.output_count != self.approved_count
        ):
            raise ValueError("calibration candidate audit is invalid")
        return self


class CalibrationCandidatePackage(TimezoneRequiredModel):
    created_at: datetime
    content_fingerprint: str = Field(min_length=64, max_length=64)
    identity_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=0)
    candidates: list[CalibrationCandidate]
    audit: CalibrationCandidateAudit

    @model_validator(mode="after")
    def require_consistent_candidate_package(self) -> Self:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if (
            self.candidate_count != len(self.candidates)
            or self.candidate_count != self.audit.output_count
            or len(candidate_ids) != len(set(candidate_ids))
            or self.created_at != self.audit.reviewed_at
            or self.content_fingerprint != calibration_review_fingerprint(self.candidates)
        ):
            raise ValueError("calibration candidate package is invalid")
        return self


class FiveFixedIntentCalibrationPublishResult(ContractModel):
    package: CalibrationCandidatePackage
    added_count: Literal[5]
    reused_existing: Literal[1]


class Sleep1IncrementalCalibrationPublishResult(ContractModel):
    package: CalibrationCandidatePackage
    added_count: Literal[1]


class CalibrationRuntimeHandoff(TimezoneRequiredModel):
    purpose: Literal["calibration_runtime"]
    version: Literal[1]
    created_at: datetime
    expected_calibration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    true_name_terms: list[str] = Field(min_length=1)
    approved_nicknames: list[str]

    @model_validator(mode="after")
    def require_matching_identity_configuration(self) -> Self:
        expected = calibration_identity_config_fingerprint(
            self.true_name_terms,
            self.approved_nicknames,
        )
        if self.identity_config_fingerprint != expected:
            raise ValueError("calibration runtime handoff is invalid")
        return self
