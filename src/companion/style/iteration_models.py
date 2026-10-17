from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from companion.contracts.base import ContractModel, TimezoneRequiredModel
from companion.contracts.style import StyleIntent, StylePreference
from companion.privacy import CandidateCategory
from companion.style.models import CoverageStatus

SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"


class ContextWindowReference(ContractModel):
    window_token: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    group_token: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)


class ContextHoldout(ContractModel):
    dataset_fingerprint: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    learning_window_tokens: list[str]
    holdout_window_tokens: list[str]
    target_holdout_ratio: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def require_unique_disjoint_window_tokens(self) -> Self:
        learning = set(self.learning_window_tokens)
        holdout = set(self.holdout_window_tokens)
        if any(
            len(token) != 64 or any(character not in "0123456789abcdef" for character in token)
            for token in learning | holdout
        ):
            raise ValueError("holdout window tokens must be sha256 hex")
        if len(learning) != len(self.learning_window_tokens):
            raise ValueError("learning window tokens must be unique")
        if len(holdout) != len(self.holdout_window_tokens):
            raise ValueError("holdout window tokens must be unique")
        if learning & holdout:
            raise ValueError("learning and holdout window tokens must be disjoint")
        return self


class LocalContextSummary(ContractModel):
    window_token: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    category: CandidateCategory
    prompt: str = Field(min_length=1, max_length=160)
    privacy_approved: Literal[True]
    raw_text_retained: Literal[False]
    partner_text_retained: Literal[False]


class StyleAdjustmentEvidence(ContractModel):
    category: CandidateCategory
    expression: str = Field(min_length=1, max_length=32)
    trigger_tags: list[str] = Field(min_length=1, max_length=8)
    intent_tags: list[StyleIntent] = Field(min_length=1, max_length=4)
    aggregate_count: int = Field(ge=1)
    source: Literal["explicit_owner_feedback"]

    @model_validator(mode="after")
    def require_unique_intent_tags(self) -> Self:
        if len(set(self.intent_tags)) != len(self.intent_tags):
            raise ValueError("style adjustment evidence intent tags must be unique")
        return self


class ContextualResponsePattern(ContractModel):
    pattern_id: str = Field(min_length=1, max_length=80)
    category: CandidateCategory
    intent: StyleIntent
    min_length: int = Field(ge=0, le=160)
    max_length: int = Field(ge=1, le=160)
    tone_tags: list[Literal["brief"]] = Field(max_length=8)
    punctuation_tags: list[Literal["plain"]] = Field(max_length=8)
    required_expressions: list[str] = Field(max_length=4)
    prohibited_expressions: list[str] = Field(max_length=8)
    aggregate_count: int = Field(ge=1)
    source: Literal["explicit_owner_feedback"]
    raw_text_retained: Literal[False]
    partner_text_retained: Literal[False]

    @model_validator(mode="after")
    def require_bounded_distinct_pattern(self) -> Self:
        required = self.required_expressions
        prohibited = self.prohibited_expressions
        if self.min_length > self.max_length:
            raise ValueError("contextual pattern length range is invalid")
        if len(required) != len(set(required)) or len(prohibited) != len(set(prohibited)):
            raise ValueError("contextual pattern expressions must be unique")
        if len(self.tone_tags) != len(set(self.tone_tags)) or len(self.punctuation_tags) != len(
            set(self.punctuation_tags)
        ):
            raise ValueError("contextual pattern tags must be unique")
        if set(required) & set(prohibited):
            raise ValueError("required and prohibited expressions must be disjoint")
        if any(
            not expression.strip() or len(expression) > 32 for expression in required + prohibited
        ):
            raise ValueError("contextual pattern expression is invalid")
        return self


class ContextualCorrectionEvidence(ContractModel):
    scenario_id: str = Field(min_length=1, max_length=80)
    category: CandidateCategory
    intent: StyleIntent
    correction_type: Literal["required_expression", "prohibited_expression"]
    expression: str = Field(min_length=1, max_length=32)
    aggregate_count: int = Field(ge=1)
    source: Literal["explicit_owner_feedback"]


class AdjustmentDecision(ContractModel):
    rule_id: str
    decision: Literal["approve", "reject"]


class ContextSummarySkipReason(StrEnum):
    SPEAKER_MAPPING_UNCONFIRMED = "speaker_mapping_unconfirmed"
    PRIVACY_NOT_APPROVED = "privacy_not_approved"
    SENSITIVE_CONTEXT = "sensitive_context"
    IDENTITY_PLACEHOLDER = "identity_placeholder"
    UNSUPPORTED_CATEGORY = "unsupported_category"


class AdjustmentDecisionPackage(ContractModel):
    proposal_id: str = Field(min_length=1, max_length=80)
    decisions: list[AdjustmentDecision] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def require_unique_rule_ids(self) -> Self:
        rule_ids = [decision.rule_id for decision in self.decisions]
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("adjustment decision rule ids must be unique")
        return self


class ContextSummaryResult(ContractModel):
    window_token: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    summary: LocalContextSummary | None
    skipped_reason: ContextSummarySkipReason | None

    @model_validator(mode="after")
    def require_ready_or_skipped_result(self) -> Self:
        if (self.summary is None) == (self.skipped_reason is None):
            raise ValueError("context summary must be ready or skipped")
        if self.summary is not None and self.summary.window_token != self.window_token:
            raise ValueError("context summary window token must match result")
        return self


class StyleAdjustmentProposal(ContractModel):
    proposal_id: str = Field(min_length=1, max_length=80)
    dataset_fingerprint: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    base_style_version: str = Field(min_length=1)
    rules: list[StylePreference] = Field(min_length=1, max_length=5)
    aggregate_counts: dict[str, int]
    status: Literal["pending"]

    @model_validator(mode="after")
    def require_unique_rules_and_aggregate_counts(self) -> Self:
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("proposal rule ids must be unique")
        if set(self.aggregate_counts) != set(rule_ids) or any(
            count <= 0 for count in self.aggregate_counts.values()
        ):
            raise ValueError("proposal aggregate counts must exactly match rules")
        return self


class ApprovedAdjustmentSet(ContractModel):
    proposal_id: str
    dataset_fingerprint: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    base_style_version: str
    rules: list[StylePreference]
    rejected_rule_ids: list[str]

    @model_validator(mode="after")
    def require_unique_disjoint_rule_ids(self) -> Self:
        approved_rule_ids = [rule.rule_id for rule in self.rules]
        if len(set(approved_rule_ids)) != len(approved_rule_ids):
            raise ValueError("approved rule ids must be unique")
        if len(set(self.rejected_rule_ids)) != len(self.rejected_rule_ids):
            raise ValueError("rejected rule ids must be unique")
        if set(approved_rule_ids) & set(self.rejected_rule_ids):
            raise ValueError("approved and rejected rule ids must be disjoint")
        return self


class HoldoutEvaluationPlanItem(ContractModel):
    window_token: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    category: CandidateCategory
    baseline_prompt: str = Field(min_length=1)
    styled_prompt: str = Field(min_length=1)
    styled_examples: list[str] = Field(max_length=2)
    coverage_status: CoverageStatus


class HoldoutEvaluationPlan(ContractModel):
    base_style_version: str
    items: list[HoldoutEvaluationPlanItem] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_window_tokens(self) -> Self:
        window_tokens = [item.window_token for item in self.items]
        if len(set(window_tokens)) != len(window_tokens):
            raise ValueError("holdout evaluation window tokens must be unique")
        return self


class StyleIterationManifest(TimezoneRequiredModel):
    dataset_fingerprint: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    base_style_version: str
    candidate_style_version: str
    candidate_digest: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    approved_rule_ids: list[str]
    base_preference_rule_ids: list[str]
    iteration_round: int = Field(ge=1, le=3)
    previous_manifest_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=SHA256_HEX_PATTERN,
    )
    lineage_id: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_PATTERN)
    status: Literal["candidate", "active", "invalidated"]
    created_at: datetime
    activated_at: datetime | None
    invalidated_at: datetime | None

    @model_validator(mode="after")
    def require_status_matched_timestamps_and_unique_rules(self) -> Self:
        if len(set(self.approved_rule_ids)) != len(self.approved_rule_ids):
            raise ValueError("manifest approved rule ids must be unique")
        if len(set(self.base_preference_rule_ids)) != len(self.base_preference_rule_ids):
            raise ValueError("manifest base preference rule ids must be unique")
        if self.iteration_round == 1 and self.previous_manifest_digest is not None:
            raise ValueError("first iteration cannot have a previous manifest")
        if self.iteration_round > 1 and self.previous_manifest_digest is None:
            raise ValueError("later iterations require a previous manifest")
        if self.status == "candidate" and (
            self.activated_at is not None or self.invalidated_at is not None
        ):
            raise ValueError("candidate manifest cannot have activation timestamps")
        if self.status == "active" and (
            self.activated_at is None or self.invalidated_at is not None
        ):
            raise ValueError("active manifest requires activation only")
        if self.status == "invalidated" and self.invalidated_at is None:
            raise ValueError("invalidated manifest requires invalidation")
        return self


class StyleIterationSummary(ContractModel):
    approved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    candidate_preference_count: int = Field(ge=0)
