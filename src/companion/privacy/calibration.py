from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from companion.privacy.ambiguous_models import (
    AmbiguousDecisionPackage,
    AmbiguousDecisionValue,
    AmbiguousReviewPackage,
    CalibrationProfile,
    CalibrationProposal,
    CalibrationRule,
    ReviewRound,
    SecondRoundState,
)
from companion.privacy.ambiguous_sampling import text_similarity
from companion.privacy.candidate_models import CandidateCategory
from companion.privacy.candidate_rules import normalize_text
from companion.privacy.errors import PrivacyValidationError, _PrivacyFailure

_MIN_COMPONENT_SIMILARITY = 0.85


@dataclass(frozen=True)
class _DecisionEvidence:
    text: str
    similarity_group: str


@dataclass(frozen=True)
class _DecisionPartition:
    keep: list[_DecisionEvidence]
    discard: list[_DecisionEvidence]
    uncertain: list[_DecisionEvidence]


def _validate_decisions(
    package: AmbiguousReviewPackage, decisions: AmbiguousDecisionPackage
) -> None:
    if (
        decisions.package_token != package.package_token
        or decisions.content_fingerprint != package.content_fingerprint
    ):
        raise _PrivacyFailure(
            "stale_review_package",
            "review decisions do not match the review package",
        )
    expected_ids = {item.review_item_id for item in package.items}
    actual_ids = [decision.review_item_id for decision in decisions.decisions]
    if len(actual_ids) != len(set(actual_ids)):
        raise _PrivacyFailure("invalid_review_package", "review decisions are invalid")
    if set(actual_ids) != expected_ids or decisions.item_count != package.item_count:
        raise _PrivacyFailure(
            "incomplete_review_decisions",
            "review decisions must cover every item exactly once",
        )


def _partition_decisions(
    package: AmbiguousReviewPackage, decisions: AmbiguousDecisionPackage
) -> _DecisionPartition:
    _validate_decisions(package, decisions)
    items_by_id = {item.review_item_id: item for item in package.items}
    keep: list[_DecisionEvidence] = []
    discard: list[_DecisionEvidence] = []
    uncertain: list[_DecisionEvidence] = []
    for decision in decisions.decisions:
        item = items_by_id[decision.review_item_id]
        evidence = _DecisionEvidence(
            text=item.text,
            similarity_group=item.similarity_group,
        )
        if decision.decision is AmbiguousDecisionValue.KEEP:
            keep.append(evidence)
        elif decision.decision is AmbiguousDecisionValue.DISCARD:
            discard.append(evidence)
        else:
            uncertain.append(evidence)
    return _DecisionPartition(keep=keep, discard=discard, uncertain=uncertain)


def _dedupe_literals(texts: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for text in texts:
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(text)
    return values


def _components(texts: Sequence[str]) -> list[list[str]]:
    remaining = list(texts)
    components: list[list[str]] = []
    while remaining:
        seed = remaining.pop(0)
        component = [seed]
        changed = True
        while changed:
            changed = False
            next_remaining: list[str] = []
            for text in remaining:
                if any(
                    text_similarity(text, approved) >= _MIN_COMPONENT_SIMILARITY
                    for approved in component
                ):
                    component.append(text)
                    changed = True
                else:
                    next_remaining.append(text)
            remaining = next_remaining
        components.append(component)
    return components


def _medoid(component: Sequence[str]) -> str:
    return max(
        component,
        key=lambda text: (
            sum(text_similarity(text, other) for other in component),
            normalize_text(text),
        ),
    )


def _rule_for_component(
    *,
    proposal_id: str,
    index: int,
    group: str,
    component: Sequence[str],
    negative_group: Sequence[str],
) -> CalibrationRule | None:
    if len(component) < 2:
        return None
    medoid = _medoid(component)
    minimum_positive = min(text_similarity(medoid, text) for text in component)
    maximum_negative = max(
        (text_similarity(medoid, text) for text in negative_group),
        default=0.0,
    )
    if maximum_negative >= minimum_positive:
        return None
    threshold = max(
        _MIN_COMPONENT_SIMILARITY,
        (minimum_positive + maximum_negative) / 2,
    )
    if threshold > minimum_positive:
        return None
    return CalibrationRule(
        rule_id=f"{proposal_id}-rule-{index:04d}",
        category=CandidateCategory.CALIBRATED_DAILY,
        similarity_group=group,
        normalized_template=normalize_text(medoid),
        similarity_threshold=threshold,
        first_round_approved=len(component),
        second_round_approved=0,
        second_round_discarded=0,
        second_round_uncertain=0,
        enabled=False,
    )


def _build_rules(
    approved_texts: Sequence[_DecisionEvidence],
    negative_texts: Sequence[_DecisionEvidence],
    proposal_id: str,
) -> list[CalibrationRule]:
    approved_by_group: dict[str, list[str]] = {}
    negative_by_group: dict[str, list[str]] = {}
    seen_approved: set[tuple[str, str]] = set()
    for evidence in approved_texts:
        key = (evidence.similarity_group, normalize_text(evidence.text))
        if key in seen_approved:
            continue
        seen_approved.add(key)
        approved_by_group.setdefault(evidence.similarity_group, []).append(evidence.text)
    for evidence in negative_texts:
        negative_by_group.setdefault(evidence.similarity_group, []).append(evidence.text)

    rules: list[CalibrationRule] = []
    for group in sorted(approved_by_group):
        for component in _components(approved_by_group[group]):
            rule = _rule_for_component(
                proposal_id=proposal_id,
                index=len(rules) + 1,
                group=group,
                component=component,
                negative_group=negative_by_group.get(group, ()),
            )
            if rule is not None:
                rules.append(rule)
    return rules


def build_calibration_proposal(
    first_package: AmbiguousReviewPackage,
    first_decisions: AmbiguousDecisionPackage,
    *,
    created_at: datetime,
    proposal_id: str,
) -> CalibrationProposal:
    try:
        if first_package.review_round is not ReviewRound.FIRST:
            raise _PrivacyFailure(
                "invalid_review_package",
                "calibration proposal requires a first-round package",
            )
        partition = _partition_decisions(first_package, first_decisions)
        manual_approved = _dedupe_literals([evidence.text for evidence in partition.keep])
        rules = _build_rules(
            partition.keep,
            [*partition.discard, *partition.uncertain],
            proposal_id,
        )
        return CalibrationProposal(
            proposal_id=proposal_id,
            first_package_token=first_package.package_token,
            first_content_fingerprint=first_package.content_fingerprint,
            created_at=created_at,
            manual_approved_texts=manual_approved,
            rules=rules,
            first_round_keep_count=len(partition.keep),
            first_round_discard_count=len(partition.discard),
            first_round_uncertain_count=len(partition.uncertain),
        )
    except _PrivacyFailure as error:
        failure = (error.code, error.detail)
    del first_package, first_decisions, created_at, proposal_id
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _validate_second_state(
    proposal: CalibrationProposal,
    second_package: AmbiguousReviewPackage,
    second_state: SecondRoundState,
) -> None:
    if second_package.review_round is not ReviewRound.SECOND:
        raise _PrivacyFailure(
            "invalid_review_package",
            "second-round evaluation requires a second-round package",
        )
    if (
        second_state.first_package_token != proposal.first_package_token
        or second_state.first_content_fingerprint != proposal.first_content_fingerprint
        or second_state.second_package_token != second_package.package_token
        or second_state.second_content_fingerprint != second_package.content_fingerprint
        or second_state.item_count != second_package.item_count
    ):
        raise _PrivacyFailure(
            "stale_review_package",
            "second-round state does not match review packages",
        )
    expected_ids = {item.review_item_id for item in second_package.items}
    link_ids = [link.review_item_id for link in second_state.links]
    if len(link_ids) != len(set(link_ids)) or set(link_ids) != expected_ids:
        raise _PrivacyFailure("invalid_review_package", "second-round state is invalid")


def _decision_count_by_rule(
    proposal: CalibrationProposal,
    second_state: SecondRoundState,
    second_decisions: AmbiguousDecisionPackage,
) -> dict[str, Counter[AmbiguousDecisionValue]]:
    decision_by_id = {
        decision.review_item_id: decision.decision for decision in second_decisions.decisions
    }
    rule_ids = {rule.rule_id for rule in proposal.rules}
    counts = {rule_id: Counter() for rule_id in rule_ids}
    for link in second_state.links:
        if not link.is_match:
            continue
        if link.rule_id not in counts:
            raise _PrivacyFailure("invalid_review_package", "second-round state is invalid")
        counts[link.rule_id][decision_by_id[link.review_item_id]] += 1
    return counts


def evaluate_second_round(
    proposal: CalibrationProposal,
    second_package: AmbiguousReviewPackage,
    second_state: SecondRoundState,
    second_decisions: AmbiguousDecisionPackage,
    *,
    activated_at: datetime,
    user_confirmed: bool,
) -> CalibrationProfile:
    try:
        if user_confirmed is not True:
            raise _PrivacyFailure(
                "calibration_not_confirmed",
                "calibration requires explicit user confirmation",
            )
        _validate_second_state(proposal, second_package, second_state)
        partition = _partition_decisions(second_package, second_decisions)
        rule_counts = _decision_count_by_rule(proposal, second_state, second_decisions)
        enabled_rules: list[CalibrationRule] = []
        for rule in proposal.rules:
            counts = rule_counts[rule.rule_id]
            second_approved = counts[AmbiguousDecisionValue.KEEP]
            second_discarded = counts[AmbiguousDecisionValue.DISCARD]
            second_uncertain = counts[AmbiguousDecisionValue.UNCERTAIN]
            enabled = (
                rule.first_round_approved >= 2
                and rule.first_round_approved + second_approved >= 3
                and second_approved >= 1
                and second_discarded == 0
                and second_uncertain == 0
            )
            if enabled:
                enabled_rules.append(
                    rule.model_copy(
                        update={
                            "second_round_approved": second_approved,
                            "second_round_discarded": second_discarded,
                            "second_round_uncertain": second_uncertain,
                            "enabled": True,
                        }
                    )
                )
        manual_approved = _dedupe_literals(
            [
                *proposal.manual_approved_texts,
                *(evidence.text for evidence in partition.keep),
            ]
        )
        return CalibrationProfile(
            profile_id=f"{proposal.proposal_id}-profile",
            activated_at=activated_at,
            manual_approved_texts=manual_approved,
            rules=enabled_rules,
            first_round_keep_count=proposal.first_round_keep_count,
            first_round_discard_count=proposal.first_round_discard_count,
            first_round_uncertain_count=proposal.first_round_uncertain_count,
            second_round_keep_count=len(partition.keep),
            second_round_discard_count=len(partition.discard),
            second_round_uncertain_count=len(partition.uncertain),
            user_confirmed=True,
            enabled=bool(enabled_rules),
        )
    except _PrivacyFailure as error:
        failure = (error.code, error.detail)
    del proposal, second_package, second_state, second_decisions, activated_at, user_confirmed
    raise PrivacyValidationError(failure[0], failure[1]) from None
