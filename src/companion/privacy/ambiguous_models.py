from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import Field, model_validator

from companion.contracts.base import ContractModel, TimezoneRequiredModel
from companion.contracts.chat import SpeakerRole
from companion.privacy.candidate_models import CandidateCategory


class AmbiguousDecisionValue(StrEnum):
    KEEP = "keep"
    DISCARD = "discard"
    UNCERTAIN = "uncertain"


class ReviewRound(StrEnum):
    FIRST = "first"
    SECOND = "second"


class ReviewContextItem(ContractModel):
    role: Literal[SpeakerRole.OWNER, SpeakerRole.OTHER]
    text: str = Field(min_length=1)


class AmbiguousReviewItem(ContractModel):
    review_item_id: str = Field(min_length=16)
    text: str = Field(min_length=1)
    context_before: list[ReviewContextItem] = Field(max_length=2)
    context_after: list[ReviewContextItem] = Field(max_length=2)
    similarity_group: str = Field(min_length=1)
    occurrence_count: int = Field(ge=1)


class AmbiguousReviewPackage(TimezoneRequiredModel):
    package_token: str = Field(min_length=32)
    review_round: ReviewRound
    created_at: datetime
    content_fingerprint: str = Field(min_length=64, max_length=64)
    requested_count: int
    item_count: int = Field(ge=0)
    items: list[AmbiguousReviewItem]

    @model_validator(mode="after")
    def require_consistent_review(self) -> Self:
        expected = 100 if self.review_round is ReviewRound.FIRST else 50
        item_ids = [item.review_item_id for item in self.items]
        if self.requested_count != expected or self.item_count != len(self.items):
            raise ValueError("review counts are invalid")
        if len(self.items) > expected or len(item_ids) != len(set(item_ids)):
            raise ValueError("review items are invalid")
        return self


class AmbiguousDecisionItem(ContractModel):
    review_item_id: str = Field(min_length=16)
    decision: AmbiguousDecisionValue


class AmbiguousDecisionPackage(ContractModel):
    package_token: str = Field(min_length=32)
    content_fingerprint: str = Field(min_length=64, max_length=64)
    item_count: int = Field(ge=0)
    decisions: list[AmbiguousDecisionItem]

    @model_validator(mode="after")
    def require_complete_unique_decisions(self) -> Self:
        item_ids = [decision.review_item_id for decision in self.decisions]
        if self.item_count != len(self.decisions) or len(item_ids) != len(set(item_ids)):
            raise ValueError("review decisions are invalid")
        return self


class CalibrationRule(ContractModel):
    rule_id: str = Field(min_length=1)
    category: Literal[CandidateCategory.CALIBRATED_DAILY]
    similarity_group: str = Field(min_length=1)
    normalized_template: str = Field(min_length=1)
    similarity_threshold: float = Field(ge=0, le=1)
    first_round_approved: int = Field(ge=0)
    second_round_approved: int = Field(ge=0)
    second_round_discarded: int = Field(ge=0)
    second_round_uncertain: int = Field(ge=0)
    enabled: bool

    @model_validator(mode="after")
    def require_enabled_evidence(self) -> Self:
        if self.enabled and not (
            self.first_round_approved >= 2
            and self.first_round_approved + self.second_round_approved >= 3
            and self.second_round_approved >= 1
            and self.second_round_discarded == 0
            and self.second_round_uncertain == 0
        ):
            raise ValueError("enabled calibration rule lacks evidence")
        return self


class CalibrationProposal(TimezoneRequiredModel):
    proposal_id: str = Field(min_length=1)
    first_package_token: str = Field(min_length=32)
    first_content_fingerprint: str = Field(min_length=64, max_length=64)
    created_at: datetime
    manual_approved_texts: list[str]
    rules: list[CalibrationRule]
    first_round_keep_count: int = Field(ge=0)
    first_round_discard_count: int = Field(ge=0)
    first_round_uncertain_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_proposal(self) -> Self:
        rule_ids = [rule.rule_id for rule in self.rules]
        if (
            len(self.manual_approved_texts) != len(set(self.manual_approved_texts))
            or any(not text for text in self.manual_approved_texts)
            or len(rule_ids) != len(set(rule_ids))
            or any(rule.enabled for rule in self.rules)
            or len(self.manual_approved_texts) > self.first_round_keep_count
            or any(rule.first_round_approved > self.first_round_keep_count for rule in self.rules)
        ):
            raise ValueError("calibration proposal is invalid")
        return self


class SecondRoundRuleLink(ContractModel):
    review_item_id: str = Field(min_length=16)
    rule_id: str = Field(min_length=1)
    is_match: bool


class SecondRoundState(ContractModel):
    first_package_token: str = Field(min_length=32)
    first_content_fingerprint: str = Field(min_length=64, max_length=64)
    second_package_token: str = Field(min_length=32)
    second_content_fingerprint: str = Field(min_length=64, max_length=64)
    item_count: int = Field(ge=0)
    links: list[SecondRoundRuleLink]

    @model_validator(mode="after")
    def require_complete_unique_links(self) -> Self:
        item_ids = [link.review_item_id for link in self.links]
        if self.item_count != len(self.links) or len(item_ids) != len(set(item_ids)):
            raise ValueError("second round links are invalid")
        return self


class CalibrationProfile(TimezoneRequiredModel):
    profile_id: str = Field(min_length=1)
    activated_at: datetime
    manual_approved_texts: list[str]
    rules: list[CalibrationRule]
    first_round_keep_count: int = Field(ge=0)
    first_round_discard_count: int = Field(ge=0)
    first_round_uncertain_count: int = Field(ge=0)
    second_round_keep_count: int = Field(ge=0)
    second_round_discard_count: int = Field(ge=0)
    second_round_uncertain_count: int = Field(ge=0)
    user_confirmed: Literal[True]
    enabled: bool

    @model_validator(mode="after")
    def require_consistent_profile(self) -> Self:
        rule_ids = [rule.rule_id for rule in self.rules]
        if (
            len(self.manual_approved_texts) != len(set(self.manual_approved_texts))
            or any(not text for text in self.manual_approved_texts)
            or len(rule_ids) != len(set(rule_ids))
            or len(self.manual_approved_texts)
            > self.first_round_keep_count + self.second_round_keep_count
            or any(
                rule.first_round_approved > self.first_round_keep_count
                or rule.second_round_approved > self.second_round_keep_count
                or rule.second_round_discarded > self.second_round_discard_count
                or rule.second_round_uncertain > self.second_round_uncertain_count
                for rule in self.rules
            )
            or (self.enabled and any(not rule.enabled for rule in self.rules))
        ):
            raise ValueError("calibration profile is invalid")
        return self


class CalibrationActivationManifest(TimezoneRequiredModel):
    activation_id: str = Field(min_length=1)
    candidate_path: str = Field(min_length=1)
    profile_path: str | None = None
    mode: Literal["full", "manual_only", "base"]
    enabled: bool
    activated_at: datetime

    @model_validator(mode="after")
    def require_safe_activation_paths(self) -> Self:
        paths = [self.candidate_path]
        if self.profile_path is not None:
            paths.append(self.profile_path)
        if any(not _is_safe_relative_path(path) for path in paths):
            raise ValueError("activation paths must be relative")
        if (
            (self.mode == "full" and (not self.enabled or self.profile_path is None))
            or (self.mode == "manual_only" and (self.enabled or self.profile_path is None))
            or (self.mode == "base" and (self.enabled or self.profile_path is not None))
        ):
            raise ValueError("activation mode is invalid")
        return self


class CalibrationActivationResult(TimezoneRequiredModel):
    activation_id: str = Field(min_length=1)
    active_manifest_path: Path
    candidate_path: Path
    profile_path: Path
    activated_at: datetime
    enabled: Literal[True]


def _is_safe_relative_path(path: str) -> bool:
    candidates = (PurePosixPath(path), PureWindowsPath(path))
    return all(
        not candidate.is_absolute()
        and not candidate.drive
        and not candidate.root
        and ".." not in candidate.parts
        for candidate in candidates
    )
