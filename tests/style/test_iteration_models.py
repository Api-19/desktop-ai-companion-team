from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from companion.contracts import StylePack
from companion.contracts.style import StyleIntent, StylePreference
from companion.privacy import CandidateCategory
from companion.style.iteration_models import (
    ApprovedAdjustmentSet,
    ContextHoldout,
    ContextSummaryResult,
    ContextWindowReference,
    HoldoutEvaluationPlan,
    HoldoutEvaluationPlanItem,
    LocalContextSummary,
    StyleAdjustmentEvidence,
    StyleAdjustmentProposal,
    StyleIterationManifest,
)
from companion.style.models import CoverageStatus

TOKEN_A = "a" * 64
TOKEN_B = "b" * 64
TOKEN_C = "c" * 64


def synthetic_proposal(rule_count: int = 1) -> StyleAdjustmentProposal:
    rules = [
        StylePreference(
            rule_id=f"synthetic-rule-{index}",
            category="sleep",
            expression=f"synthetic-expression-{index}",
            trigger_tags=["sleep"],
            max_consecutive_uses=1,
        )
        for index in range(rule_count)
    ]
    return StyleAdjustmentProposal(
        proposal_id="synthetic-proposal",
        dataset_fingerprint=TOKEN_A,
        base_style_version="synthetic-v1",
        rules=rules,
        aggregate_counts={rule.rule_id: 1 for rule in rules},
        status="pending",
    )


def test_preference_intents_are_optional_for_legacy_style_packs() -> None:
    legacy_pack = StylePack.model_validate(
        {
            "version": "synthetic-v1",
            "statistics": {},
            "scene_tags": [],
            "forbidden_traits": [],
            "examples": [],
            "preferences": [
                {
                    "rule_id": "synthetic-rule",
                    "category": "sleep",
                    "expression": "synthetic-expression",
                    "trigger_tags": ["sleep"],
                    "max_consecutive_uses": 1,
                }
            ],
        }
    )

    assert legacy_pack.preferences[0].intent_tags == []


@pytest.mark.parametrize(
    "intent_tags",
    [["unknown_intent"], [StyleIntent.SLEEP_FAREWELL, StyleIntent.SLEEP_FAREWELL]],
)
def test_adjustment_evidence_rejects_unknown_or_duplicate_intents(
    intent_tags: list[StyleIntent | str],
) -> None:
    with pytest.raises(ValidationError):
        StyleAdjustmentEvidence(
            category=CandidateCategory.SLEEP,
            expression="synthetic-expression",
            trigger_tags=["sleep"],
            intent_tags=intent_tags,
            aggregate_count=1,
            source="explicit_owner_feedback",
        )


def test_holdout_rejects_overlap() -> None:
    with pytest.raises(ValidationError):
        ContextHoldout(
            dataset_fingerprint=TOKEN_A,
            learning_window_tokens=[TOKEN_B],
            holdout_window_tokens=[TOKEN_B],
            target_holdout_ratio=0.2,
        )


def test_opaque_tokens_and_fingerprints_require_sha256_hex() -> None:
    with pytest.raises(ValidationError):
        ContextWindowReference(window_token="z" * 64, group_token="b" * 64)
    with pytest.raises(ValidationError):
        ContextHoldout(
            dataset_fingerprint="z" * 64,
            learning_window_tokens=["a" * 64],
            holdout_window_tokens=["b" * 64],
            target_holdout_ratio=0.2,
        )


def test_proposal_rejects_raw_evidence_fields() -> None:
    payload = synthetic_proposal().model_dump(mode="json")
    payload["raw_context"] = "synthetic forbidden field"

    with pytest.raises(ValidationError):
        StyleAdjustmentProposal.model_validate(payload)


def test_proposal_requires_unique_rules_and_exact_positive_aggregate_counts() -> None:
    proposal = synthetic_proposal(2)
    duplicate_rules = [proposal.rules[0], proposal.rules[0]]

    with pytest.raises(ValidationError):
        StyleAdjustmentProposal(
            proposal_id=proposal.proposal_id,
            dataset_fingerprint=proposal.dataset_fingerprint,
            base_style_version=proposal.base_style_version,
            rules=duplicate_rules,
            aggregate_counts={proposal.rules[0].rule_id: 1},
            status="pending",
        )
    with pytest.raises(ValidationError):
        StyleAdjustmentProposal(
            proposal_id=proposal.proposal_id,
            dataset_fingerprint=proposal.dataset_fingerprint,
            base_style_version=proposal.base_style_version,
            rules=proposal.rules,
            aggregate_counts={rule.rule_id: 0 for rule in proposal.rules},
            status="pending",
        )


def test_context_summary_requires_exactly_one_ready_or_skipped_result() -> None:
    summary = LocalContextSummary(
        window_token=TOKEN_A,
        category=CandidateCategory.SLEEP,
        prompt="synthetic safe context",
        privacy_approved=True,
        raw_text_retained=False,
        partner_text_retained=False,
    )

    with pytest.raises(ValidationError):
        ContextSummaryResult(
            window_token=TOKEN_A,
            summary=summary,
            skipped_reason="sensitive_context",
        )
    with pytest.raises(ValidationError):
        ContextSummaryResult(window_token=TOKEN_A, summary=None, skipped_reason=None)


def test_context_summary_rejects_mismatched_window_token() -> None:
    summary = LocalContextSummary(
        window_token=TOKEN_A,
        category=CandidateCategory.SLEEP,
        prompt="synthetic safe context",
        privacy_approved=True,
        raw_text_retained=False,
        partner_text_retained=False,
    )

    with pytest.raises(ValidationError):
        ContextSummaryResult(window_token=TOKEN_B, summary=summary, skipped_reason=None)


def test_approved_set_rejects_overlapping_rule_ids() -> None:
    rule = synthetic_proposal().rules[0]

    with pytest.raises(ValidationError):
        ApprovedAdjustmentSet(
            proposal_id="synthetic-proposal",
            dataset_fingerprint=TOKEN_A,
            base_style_version="synthetic-v1",
            rules=[rule],
            rejected_rule_ids=[rule.rule_id],
        )


def test_holdout_plan_rejects_duplicate_window_tokens() -> None:
    item = HoldoutEvaluationPlanItem(
        window_token=TOKEN_A,
        category=CandidateCategory.SLEEP,
        baseline_prompt="synthetic baseline",
        styled_prompt="synthetic styled",
        styled_examples=[],
        coverage_status=CoverageStatus.TESTABLE,
    )

    with pytest.raises(ValidationError):
        HoldoutEvaluationPlan(base_style_version="synthetic-v1", items=[item, item])


@pytest.mark.parametrize(
    ("status", "activated_at", "invalidated_at"),
    [
        ("candidate", datetime(2026, 7, 19, tzinfo=UTC), None),
        ("active", None, None),
        ("invalidated", None, None),
    ],
)
def test_manifest_requires_status_matched_timestamps(
    status: str,
    activated_at: datetime | None,
    invalidated_at: datetime | None,
) -> None:
    with pytest.raises(ValidationError):
        StyleIterationManifest(
            dataset_fingerprint=TOKEN_A,
            base_style_version="synthetic-v1",
            candidate_style_version="synthetic-v2",
            candidate_digest=TOKEN_B,
            approved_rule_ids=["synthetic-rule"],
            base_preference_rule_ids=[],
            iteration_round=1,
            previous_manifest_digest=None,
            lineage_id=TOKEN_C,
            status=status,
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
            activated_at=activated_at,
            invalidated_at=invalidated_at,
        )
