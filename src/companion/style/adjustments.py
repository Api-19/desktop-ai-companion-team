import hashlib
import json
from collections import Counter

from companion.contracts.style import StylePreference
from companion.privacy import SafeCandidatePackage
from companion.style.errors import StyleValidationError
from companion.style.iteration_models import (
    AdjustmentDecision,
    ApprovedAdjustmentSet,
    StyleAdjustmentEvidence,
    StyleAdjustmentProposal,
)

_OWNER_FEEDBACK_SOURCE = "explicit_owner_feedback"
_MAX_PROPOSED_RULES = 5


def candidate_package_fingerprint(package: SafeCandidatePackage) -> str:
    payload = json.dumps(
        package.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _revalidate_evidence(evidence: object) -> list[StyleAdjustmentEvidence]:
    if not isinstance(evidence, list):
        raise StyleValidationError(
            "invalid_adjustment_evidence",
            "style adjustment evidence is invalid",
        )
    validated: list[StyleAdjustmentEvidence] = []
    for item in evidence:
        if not isinstance(item, StyleAdjustmentEvidence):
            raise StyleValidationError(
                "invalid_adjustment_evidence",
                "style adjustment evidence is invalid",
            )
        validated.append(
            StyleAdjustmentEvidence.model_validate(item.model_dump(mode="json", warnings=False))
        )
    return validated


def _revalidate_candidate_package(package: object) -> SafeCandidatePackage:
    if not isinstance(package, SafeCandidatePackage):
        raise StyleValidationError("invalid_adjustment_evidence", "style evidence is invalid")
    return SafeCandidatePackage.model_validate(package.model_dump(mode="json", warnings=False))


def _revalidate_review_inputs(
    proposal: object,
    decisions: object,
) -> tuple[StyleAdjustmentProposal, list[AdjustmentDecision]]:
    if not isinstance(proposal, StyleAdjustmentProposal) or not isinstance(decisions, list):
        raise StyleValidationError(
            "invalid_adjustment_decisions",
            "style adjustment decisions are invalid",
        )
    validated_decisions: list[AdjustmentDecision] = []
    for decision in decisions:
        if not isinstance(decision, AdjustmentDecision):
            raise StyleValidationError(
                "invalid_adjustment_decisions",
                "style adjustment decisions are invalid",
            )
        validated_decisions.append(
            AdjustmentDecision.model_validate(decision.model_dump(mode="json", warnings=False))
        )
    return (
        StyleAdjustmentProposal.model_validate(proposal.model_dump(mode="json", warnings=False)),
        validated_decisions,
    )


def _propose_style_adjustments(
    evidence: list[StyleAdjustmentEvidence],
    *,
    candidate_package: SafeCandidatePackage,
    proposal_id: str,
    base_style_version: str,
) -> StyleAdjustmentProposal:
    if not evidence:
        raise StyleValidationError("empty_adjustment_evidence", "style evidence is empty")

    candidates = {
        (candidate.category, candidate.text, tuple(candidate.scene_tags))
        for candidate in candidate_package.candidates
    }
    grouped: Counter[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = Counter()
    for item in evidence:
        if not isinstance(item, StyleAdjustmentEvidence) or item.source != _OWNER_FEEDBACK_SOURCE:
            raise StyleValidationError(
                "invalid_adjustment_evidence",
                "style evidence must be explicit owner feedback",
            )
        if (item.category, item.expression, tuple(item.trigger_tags)) not in candidates:
            raise StyleValidationError(
                "unapproved_adjustment_evidence",
                "style evidence is not an approved candidate",
            )
        key = (
            item.category.value,
            item.expression,
            tuple(item.trigger_tags),
            tuple(item.intent_tags),
        )
        grouped[key] += item.aggregate_count

    ranked = sorted(grouped.items(), key=lambda entry: (-entry[1], entry[0]))[:_MAX_PROPOSED_RULES]
    rules: list[StylePreference] = []
    aggregate_counts: dict[str, int] = {}
    for (category, expression, trigger_tags, intent_tags), count in ranked:
        material = json.dumps(
            [category, expression, trigger_tags, intent_tags],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        rule_id = f"style-rule-{hashlib.sha256(material.encode()).hexdigest()[:16]}"
        rules.append(
            StylePreference(
                rule_id=rule_id,
                category=category,
                expression=expression,
                trigger_tags=list(trigger_tags),
                intent_tags=list(intent_tags),
                max_consecutive_uses=1,
            )
        )
        aggregate_counts[rule_id] = count
    return StyleAdjustmentProposal(
        proposal_id=proposal_id,
        dataset_fingerprint=candidate_package_fingerprint(candidate_package),
        base_style_version=base_style_version,
        rules=rules,
        aggregate_counts=aggregate_counts,
        status="pending",
    )


def propose_style_adjustments(
    evidence: list[StyleAdjustmentEvidence],
    *,
    candidate_package: SafeCandidatePackage,
    proposal_id: str,
    base_style_version: str,
) -> StyleAdjustmentProposal:
    try:
        validated_evidence = _revalidate_evidence(evidence)
        validated_package = _revalidate_candidate_package(candidate_package)
        return _propose_style_adjustments(
            validated_evidence,
            candidate_package=validated_package,
            proposal_id=proposal_id,
            base_style_version=base_style_version,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = (
            "invalid_adjustment_evidence",
            "style adjustment evidence is invalid",
        )
    finally:
        evidence = None
        proposal_id = None
        candidate_package = None
        base_style_version = None
        validated_evidence = None
        validated_package = None
    raise StyleValidationError(*failure) from None


def _review_style_adjustments(
    proposal: StyleAdjustmentProposal,
    decisions: list[AdjustmentDecision],
) -> ApprovedAdjustmentSet:
    if not isinstance(proposal, StyleAdjustmentProposal) or any(
        not isinstance(decision, AdjustmentDecision) for decision in decisions
    ):
        raise StyleValidationError(
            "invalid_adjustment_decisions",
            "style adjustment decisions are invalid",
        )
    supplied_rule_ids = [decision.rule_id for decision in decisions]
    expected_rule_ids = [rule.rule_id for rule in proposal.rules]
    if len(supplied_rule_ids) != len(set(supplied_rule_ids)) or set(supplied_rule_ids) != set(
        expected_rule_ids
    ):
        raise StyleValidationError(
            "adjustment_decision_set_mismatch",
            "adjustment decisions do not match proposal rules",
        )

    decisions_by_rule = {decision.rule_id: decision.decision for decision in decisions}
    if any(decision not in {"approve", "reject"} for decision in decisions_by_rule.values()):
        raise StyleValidationError(
            "invalid_adjustment_decisions",
            "style adjustment decisions are invalid",
        )
    approved_rules = [
        rule for rule in proposal.rules if decisions_by_rule[rule.rule_id] == "approve"
    ]
    rejected_rule_ids = [
        rule.rule_id for rule in proposal.rules if decisions_by_rule[rule.rule_id] == "reject"
    ]
    return ApprovedAdjustmentSet(
        proposal_id=proposal.proposal_id,
        dataset_fingerprint=proposal.dataset_fingerprint,
        base_style_version=proposal.base_style_version,
        rules=approved_rules,
        rejected_rule_ids=rejected_rule_ids,
    )


def review_style_adjustments(
    proposal: StyleAdjustmentProposal,
    decisions: list[AdjustmentDecision],
) -> ApprovedAdjustmentSet:
    try:
        validated_proposal, validated_decisions = _revalidate_review_inputs(proposal, decisions)
        return _review_style_adjustments(validated_proposal, validated_decisions)
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = (
            "invalid_adjustment_decisions",
            "style adjustment decisions are invalid",
        )
    finally:
        proposal = None
        decisions = None
        validated_proposal = None
        validated_decisions = None
    raise StyleValidationError(*failure) from None
