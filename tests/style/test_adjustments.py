import traceback
import warnings
from collections import Counter
from datetime import UTC, datetime

import pytest

from companion.contracts.style import StyleIntent, StylePreference
from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.style.adjustments import (
    propose_style_adjustments,
    review_style_adjustments,
)
from companion.style.errors import StyleValidationError
from companion.style.iteration_models import (
    AdjustmentDecision,
    StyleAdjustmentEvidence,
    StyleAdjustmentProposal,
)
from companion.style.models import EvaluationResponses, ScenarioResponses


def synthetic_evidence(
    category: CandidateCategory,
    expression: str,
    *,
    count: int,
    trigger_tags: list[str] | None = None,
    intent_tags: list[StyleIntent] | None = None,
) -> StyleAdjustmentEvidence:
    return StyleAdjustmentEvidence(
        category=category,
        expression=expression,
        trigger_tags=trigger_tags or [category.value],
        intent_tags=intent_tags or [StyleIntent.CATEGORY_DEFAULT],
        aggregate_count=count,
        source="explicit_owner_feedback",
    )


def candidate_package_for(evidence: list[StyleAdjustmentEvidence]) -> SafeCandidatePackage:
    candidates = [
        SafeStyleCandidate(
            text=item.expression,
            category=item.category,
            scene_tags=[item.category.value],
        )
        for item in evidence
    ]
    counts = Counter(item.category.value for item in candidates)
    timestamp = datetime(2026, 7, 19, tzinfo=UTC)
    return SafeCandidatePackage(
        created_at=timestamp,
        candidates=candidates,
        audit=CandidateSelectionAudit(
            processed_at=timestamp,
            input_count=len(candidates),
            owner_input_count=len(candidates),
            dropped_other_count=0,
            allowed_category_counts=dict(counts),
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=len(candidates),
            succeeded=True,
        ),
    )


def synthetic_proposal(rule_count: int):
    evidence = [
        synthetic_evidence(
            CandidateCategory.SLEEP,
            f"synthetic-expression-{index}",
            count=1,
        )
        for index in range(rule_count)
    ]
    return propose_style_adjustments(
        evidence,
        candidate_package=candidate_package_for(evidence),
        proposal_id="synthetic-proposal",
        base_style_version="synthetic-v1",
    )


def traceback_retains(error: BaseException, needle: str) -> bool:
    for frame, _ in traceback.walk_tb(error.__traceback__):
        filename = frame.f_code.co_filename.replace("/", "\\")
        if "companion\\style" in filename and needle in repr(frame.f_locals):
            return True
    return False


def test_proposal_deduplicates_owner_feedback_caps_rules_and_is_deterministic() -> None:
    evidence = [
        synthetic_evidence(
            CandidateCategory.SLEEP,
            "synthetic-short-form",
            count=3,
            trigger_tags=["sleep"],
        ),
        synthetic_evidence(
            CandidateCategory.SLEEP,
            "synthetic-short-form",
            count=2,
            trigger_tags=["sleep"],
        ),
        *[
            synthetic_evidence(CandidateCategory.MEALS, f"synthetic-{index}", count=1)
            for index in range(8)
        ],
    ]

    proposal = propose_style_adjustments(
        evidence,
        candidate_package=candidate_package_for(evidence),
        proposal_id="synthetic-proposal",
        base_style_version="synthetic-v1",
    )
    reversed_proposal = propose_style_adjustments(
        list(reversed(evidence)),
        candidate_package=candidate_package_for(evidence),
        proposal_id="synthetic-proposal",
        base_style_version="synthetic-v1",
    )

    assert len(proposal.rules) == 5
    assert proposal.model_dump(mode="json") == reversed_proposal.model_dump(mode="json")
    assert proposal.rules[0].expression == "synthetic-short-form"
    assert proposal.rules[0].trigger_tags == ["sleep"]
    assert proposal.aggregate_counts[proposal.rules[0].rule_id] == 5
    assert proposal.status == "pending"


def test_proposal_preserves_exact_privacy_approved_expression() -> None:
    evidence = synthetic_evidence(
        CandidateCategory.SLEEP,
        " synthetic-exact-form ",
        count=1,
        trigger_tags=["sleep"],
    )

    proposal = propose_style_adjustments(
        [evidence],
        candidate_package=candidate_package_for([evidence]),
        proposal_id="synthetic-proposal",
        base_style_version="synthetic-v1",
    )

    assert proposal.rules[0].expression == evidence.expression
    assert proposal.rules[0].trigger_tags == evidence.trigger_tags


def test_proposal_keeps_exact_candidate_binding_and_separate_intent() -> None:
    evidence = StyleAdjustmentEvidence(
        category=CandidateCategory.SLEEP,
        expression="synthetic-anan",
        trigger_tags=["sleep"],
        intent_tags=[StyleIntent.SLEEP_FAREWELL],
        aggregate_count=1,
        source="explicit_owner_feedback",
    )

    proposal = propose_style_adjustments(
        [evidence],
        candidate_package=candidate_package_for([evidence]),
        proposal_id="synthetic-proposal",
        base_style_version="synthetic-v1",
    )

    assert proposal.rules[0].trigger_tags == ["sleep"]
    assert proposal.rules[0].intent_tags == [StyleIntent.SLEEP_FAREWELL]


def test_proposal_rule_ids_distinguish_identical_bindings_with_different_intents() -> None:
    farewell = StyleAdjustmentEvidence(
        category=CandidateCategory.SLEEP,
        expression="synthetic-anan",
        trigger_tags=["sleep"],
        intent_tags=[StyleIntent.SLEEP_FAREWELL],
        aggregate_count=1,
        source="explicit_owner_feedback",
    )
    reminder = farewell.model_copy(update={"intent_tags": [StyleIntent.SLEEP_REMINDER]})

    proposal = propose_style_adjustments(
        [farewell, reminder],
        candidate_package=candidate_package_for([farewell]),
        proposal_id="synthetic-proposal",
        base_style_version="synthetic-v1",
    )

    assert len(proposal.rules) == 2
    assert len({rule.rule_id for rule in proposal.rules}) == 2


def test_proposal_rejects_forged_evidence_without_warning_or_stderr_retention(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "synthetic-forged-evidence-sentinel"
    unsafe_evidence = StyleAdjustmentEvidence.model_construct(
        category=sentinel,
        expression="synthetic-expression",
        trigger_tags=["sleep"],
        intent_tags=[StyleIntent.CATEGORY_DEFAULT],
        aggregate_count=1,
        source="explicit_owner_feedback",
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(StyleValidationError) as caught:
            propose_style_adjustments(
                [unsafe_evidence],
                candidate_package=candidate_package_for([]),
                proposal_id="synthetic-proposal",
                base_style_version="synthetic-v1",
            )

    assert caught.value.code == "invalid_adjustment_evidence"
    assert sentinel not in str(caught.value)
    assert not traceback_retains(caught.value, sentinel)
    assert sentinel not in "\n".join(str(item.message) for item in caught_warnings)
    assert sentinel not in capsys.readouterr().err


def test_proposal_rejects_generated_answer_text_input_without_retaining_it() -> None:
    sentinel = "synthetic-generated-answer-text-sentinel"

    with pytest.raises(StyleValidationError) as caught:
        propose_style_adjustments(
            [sentinel],  # type: ignore[list-item]
            candidate_package=candidate_package_for([]),
            proposal_id="synthetic-proposal",
            base_style_version="synthetic-v1",
        )

    assert caught.value.code == "invalid_adjustment_evidence"
    assert sentinel not in str(caught.value)
    assert not traceback_retains(caught.value, sentinel)


def test_proposal_rejects_evaluation_responses_as_evidence_without_retaining_text() -> None:
    sentinel = "synthetic-evaluation-response-sentinel"
    responses = EvaluationResponses(
        items=[
            ScenarioResponses(
                scenario_id="synthetic-scenario",
                baseline="synthetic-baseline",
                styled=sentinel,
            )
        ]
    )

    with pytest.raises(StyleValidationError) as caught:
        propose_style_adjustments(
            [responses],  # type: ignore[list-item]
            candidate_package=candidate_package_for([]),
            proposal_id="synthetic-proposal",
            base_style_version="synthetic-v1",
        )

    assert caught.value.code == "invalid_adjustment_evidence"
    assert sentinel not in str(caught.value)
    assert not traceback_retains(caught.value, sentinel)


def test_proposal_scrubs_model_dump_failure_without_retaining_sentinel() -> None:
    sentinel = "synthetic-model-dump-failure-sentinel"

    class FailingDumpEvidence(StyleAdjustmentEvidence):
        def model_dump(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            raise RuntimeError(sentinel)

    evidence = FailingDumpEvidence(
        category=CandidateCategory.SLEEP,
        expression="synthetic-expression",
        trigger_tags=["sleep"],
        intent_tags=[StyleIntent.CATEGORY_DEFAULT],
        aggregate_count=1,
        source="explicit_owner_feedback",
    )

    with pytest.raises(StyleValidationError) as caught:
        propose_style_adjustments(
            [evidence],
            candidate_package=candidate_package_for([]),
            proposal_id="synthetic-proposal",
            base_style_version="synthetic-v1",
        )

    assert caught.value.code == "invalid_adjustment_evidence"
    assert sentinel not in str(caught.value)
    assert not traceback_retains(caught.value, sentinel)


def test_review_requires_one_exact_unique_decision_per_rule() -> None:
    proposal = synthetic_proposal(rule_count=2)

    for decisions in (
        [AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="approve")],
        [
            AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="approve"),
            AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="reject"),
        ],
        [
            AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="approve"),
            AdjustmentDecision(rule_id="synthetic-extra-rule", decision="reject"),
        ],
    ):
        with pytest.raises(StyleValidationError, match="adjustment_decision_set_mismatch"):
            review_style_adjustments(proposal, decisions)


def test_review_partitions_approved_and_rejected_rules_in_proposal_order() -> None:
    proposal = synthetic_proposal(rule_count=3)

    approved = review_style_adjustments(
        proposal,
        [
            AdjustmentDecision(rule_id=proposal.rules[2].rule_id, decision="approve"),
            AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="reject"),
            AdjustmentDecision(rule_id=proposal.rules[1].rule_id, decision="approve"),
        ],
    )

    assert [rule.rule_id for rule in approved.rules] == [
        proposal.rules[1].rule_id,
        proposal.rules[2].rule_id,
    ]
    assert approved.rejected_rule_ids == [proposal.rules[0].rule_id]
    assert proposal.status == "pending"


def test_review_rejects_forged_nested_preference_without_warning_or_stderr_retention(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "synthetic-forged-preference-expression-sentinel"
    proposal = StyleAdjustmentProposal.model_construct(
        proposal_id="synthetic-proposal",
        dataset_fingerprint="a" * 64,
        base_style_version="synthetic-v1",
        rules=[
            StylePreference.model_construct(
                rule_id="synthetic-rule",
                category="sleep",
                expression=[sentinel],
                trigger_tags=["sleep"],
                max_consecutive_uses=1,
            )
        ],
        aggregate_counts={"synthetic-rule": 1},
        status="pending",
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(StyleValidationError) as caught:
            review_style_adjustments(
                proposal,
                [AdjustmentDecision(rule_id="synthetic-rule", decision="approve")],
            )

    assert caught.value.code == "invalid_adjustment_decisions"
    assert sentinel not in str(caught.value)
    assert not traceback_retains(caught.value, sentinel)
    assert sentinel not in "\n".join(str(item.message) for item in caught_warnings)
    assert sentinel not in capsys.readouterr().err


def test_review_rejects_forged_decision_without_warning_or_stderr_retention(
    capsys: pytest.CaptureFixture[str],
) -> None:
    proposal = synthetic_proposal(rule_count=1)
    sentinel = "synthetic-forged-decision-sentinel"
    decision = AdjustmentDecision.model_construct(
        rule_id=[sentinel],
        decision="approve",
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(StyleValidationError) as caught:
            review_style_adjustments(proposal, [decision])

    assert caught.value.code == "invalid_adjustment_decisions"
    assert sentinel not in str(caught.value)
    assert not traceback_retains(caught.value, sentinel)
    assert sentinel not in "\n".join(str(item.message) for item in caught_warnings)
    assert sentinel not in capsys.readouterr().err
