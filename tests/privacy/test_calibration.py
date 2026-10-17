from datetime import UTC, datetime

import pytest

from companion.privacy.ambiguous_models import (
    AmbiguousDecisionItem,
    AmbiguousDecisionPackage,
    AmbiguousDecisionValue,
    AmbiguousReviewItem,
    AmbiguousReviewPackage,
    ReviewRound,
    SecondRoundRuleLink,
    SecondRoundState,
)
from companion.privacy.calibration import (
    build_calibration_proposal,
    evaluate_second_round,
)
from companion.privacy.errors import PrivacyValidationError

CREATED_AT = datetime(2026, 7, 18, 11, 0, tzinfo=UTC)
ACTIVATED_AT = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def review_package(
    texts: list[str],
    *,
    token: str,
    fingerprint: str,
    review_round: ReviewRound = ReviewRound.FIRST,
) -> AmbiguousReviewPackage:
    requested_count = 100 if review_round is ReviewRound.FIRST else 50
    return AmbiguousReviewPackage(
        package_token=token,
        review_round=review_round,
        created_at=CREATED_AT,
        content_fingerprint=fingerprint,
        requested_count=requested_count,
        item_count=len(texts),
        items=[
            AmbiguousReviewItem(
                review_item_id=f"{review_round.value}-synthetic-item-{index:04d}",
                text=text,
                context_before=[],
                context_after=[],
                similarity_group="m:plain:text",
                occurrence_count=1,
            )
            for index, text in enumerate(texts, start=1)
        ],
    )


def decisions_for(
    package: AmbiguousReviewPackage, values: list[AmbiguousDecisionValue]
) -> AmbiguousDecisionPackage:
    return AmbiguousDecisionPackage(
        package_token=package.package_token,
        content_fingerprint=package.content_fingerprint,
        item_count=len(values),
        decisions=[
            AmbiguousDecisionItem(review_item_id=item.review_item_id, decision=value)
            for item, value in zip(package.items, values, strict=True)
        ],
    )


def synthetic_negative_texts() -> tuple[str, ...]:
    return (
        "synthetic-discard-private",
        "synthetic-uncertain-private",
        "synthetic-hard-negative",
    )


def first_round_fixture() -> tuple[AmbiguousReviewPackage, AmbiguousDecisionPackage]:
    texts = [
        "daily alpha one",
        "daily alpha one!",
        "daily alpha two",
        "daily beta one",
        "daily beta one!",
        "solo keep literal",
        *synthetic_negative_texts(),
    ]
    package = review_package(texts, token="f" * 32, fingerprint="a" * 64)
    decisions = decisions_for(
        package,
        [
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.DISCARD,
            AmbiguousDecisionValue.UNCERTAIN,
            AmbiguousDecisionValue.DISCARD,
        ],
    )
    return package, decisions


def test_first_round_keeps_literals_but_never_persists_negative_text() -> None:
    package, decisions = first_round_fixture()

    proposal = build_calibration_proposal(
        package,
        decisions,
        created_at=CREATED_AT,
        proposal_id="proposal-synthetic-001",
    )

    serialized = proposal.model_dump_json()
    assert len(proposal.manual_approved_texts) == 6
    assert proposal.first_round_keep_count == 6
    assert proposal.first_round_discard_count == 2
    assert proposal.first_round_uncertain_count == 1
    assert proposal.rules
    assert all(text not in serialized for text in synthetic_negative_texts())


def test_first_round_rejects_stale_or_incomplete_decisions() -> None:
    package, decisions = first_round_fixture()
    stale = decisions.model_copy(update={"package_token": "s" * 32})

    with pytest.raises(PrivacyValidationError) as exc_info:
        build_calibration_proposal(
            package,
            stale,
            created_at=CREATED_AT,
            proposal_id="proposal-synthetic-002",
        )

    assert exc_info.value.code == "stale_review_package"


def test_duplicate_text_in_different_groups_does_not_overwrite_group_evidence() -> None:
    package = review_package(
        [
            "shared duplicate",
            "shared duplicate!",
            "shared duplicate",
        ],
        token="g" * 32,
        fingerprint="2" * 64,
    )
    package.items[0].similarity_group = "m:plain:text"
    package.items[1].similarity_group = "m:plain:text"
    package.items[2].similarity_group = "m:question:text"
    decisions = decisions_for(
        package,
        [
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.DISCARD,
        ],
    )

    proposal = build_calibration_proposal(
        package,
        decisions,
        created_at=CREATED_AT,
        proposal_id="proposal-synthetic-duplicate-groups",
    )

    assert [rule.similarity_group for rule in proposal.rules] == ["m:plain:text"]
    assert proposal.first_round_discard_count == 1


def test_negative_evidence_blocks_too_broad_component_without_persisting_text() -> None:
    negative_text = "daily alpha one!"
    package = review_package(
        ["daily alpha one", "daily alpha one!", negative_text],
        token="h" * 32,
        fingerprint="3" * 64,
    )
    decisions = decisions_for(
        package,
        [
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.DISCARD,
        ],
    )

    proposal = build_calibration_proposal(
        package,
        decisions,
        created_at=CREATED_AT,
        proposal_id="proposal-synthetic-negative-threshold",
    )

    assert proposal.rules == []


@pytest.mark.parametrize(
    ("second_keep", "second_discard", "second_uncertain", "enabled"),
    [(1, 0, 0, True), (0, 0, 0, False), (1, 1, 0, False), (1, 0, 1, False)],
)
def test_rule_eligibility_is_zero_false_accept(
    second_keep: int,
    second_discard: int,
    second_uncertain: int,
    enabled: bool,
) -> None:
    profile = evaluate_synthetic_rule(
        second_keep=second_keep,
        second_discard=second_discard,
        second_uncertain=second_uncertain,
    )

    assert bool(profile.rules) is enabled
    if enabled:
        assert profile.rules[0].enabled is True


def evaluate_synthetic_rule(
    *,
    second_keep: int,
    second_discard: int,
    second_uncertain: int,
):
    first_package = review_package(
        ["daily alpha one", "daily alpha one!"],
        token="p" * 32,
        fingerprint="b" * 64,
    )
    first_decisions = decisions_for(
        first_package,
        [AmbiguousDecisionValue.KEEP, AmbiguousDecisionValue.KEEP],
    )
    proposal = build_calibration_proposal(
        first_package,
        first_decisions,
        created_at=CREATED_AT,
        proposal_id="proposal-synthetic-rule",
    )
    values = [
        *([AmbiguousDecisionValue.KEEP] * second_keep),
        *([AmbiguousDecisionValue.DISCARD] * second_discard),
        *([AmbiguousDecisionValue.UNCERTAIN] * second_uncertain),
    ]
    if not values:
        values = [AmbiguousDecisionValue.DISCARD]
        link_is_match = False
    else:
        link_is_match = True
    second_package = review_package(
        [f"daily alpha follow {index}" for index in range(len(values))],
        token="q" * 32,
        fingerprint="c" * 64,
        review_round=ReviewRound.SECOND,
    )
    second_decisions = decisions_for(second_package, values)
    second_state = SecondRoundState(
        first_package_token=first_package.package_token,
        first_content_fingerprint=first_package.content_fingerprint,
        second_package_token=second_package.package_token,
        second_content_fingerprint=second_package.content_fingerprint,
        item_count=second_package.item_count,
        links=[
            SecondRoundRuleLink(
                review_item_id=item.review_item_id,
                rule_id=proposal.rules[0].rule_id,
                is_match=link_is_match,
            )
            for item in second_package.items
        ],
    )
    return evaluate_second_round(
        proposal,
        second_package,
        second_state,
        second_decisions,
        activated_at=ACTIVATED_AT,
        user_confirmed=True,
    )


def test_second_round_requires_explicit_user_confirmation() -> None:
    first_package = review_package(
        ["daily alpha one", "daily alpha one!"],
        token="r" * 32,
        fingerprint="d" * 64,
    )
    first_decisions = decisions_for(
        first_package,
        [AmbiguousDecisionValue.KEEP, AmbiguousDecisionValue.KEEP],
    )
    proposal = build_calibration_proposal(
        first_package,
        first_decisions,
        created_at=CREATED_AT,
        proposal_id="proposal-synthetic-confirm",
    )
    second_package = review_package(
        ["daily alpha follow"],
        token="u" * 32,
        fingerprint="e" * 64,
        review_round=ReviewRound.SECOND,
    )
    second_decisions = decisions_for(second_package, [AmbiguousDecisionValue.KEEP])
    second_state = SecondRoundState(
        first_package_token=first_package.package_token,
        first_content_fingerprint=first_package.content_fingerprint,
        second_package_token=second_package.package_token,
        second_content_fingerprint=second_package.content_fingerprint,
        item_count=1,
        links=[
            SecondRoundRuleLink(
                review_item_id=second_package.items[0].review_item_id,
                rule_id=proposal.rules[0].rule_id,
                is_match=True,
            )
        ],
    )

    with pytest.raises(PrivacyValidationError) as exc_info:
        evaluate_second_round(
            proposal,
            second_package,
            second_state,
            second_decisions,
            activated_at=ACTIVATED_AT,
            user_confirmed=False,
        )

    assert exc_info.value.code == "calibration_not_confirmed"


def test_second_round_rejects_stale_decisions_and_state() -> None:
    first_package = review_package(
        ["daily alpha one", "daily alpha one!"],
        token="2" * 32,
        fingerprint="4" * 64,
    )
    first_decisions = decisions_for(
        first_package,
        [AmbiguousDecisionValue.KEEP, AmbiguousDecisionValue.KEEP],
    )
    proposal = build_calibration_proposal(
        first_package,
        first_decisions,
        created_at=CREATED_AT,
        proposal_id="proposal-synthetic-stale-second",
    )
    second_package = review_package(
        ["daily alpha follow"],
        token="3" * 32,
        fingerprint="5" * 64,
        review_round=ReviewRound.SECOND,
    )
    second_decisions = decisions_for(second_package, [AmbiguousDecisionValue.KEEP])
    stale_decisions = second_decisions.model_copy(update={"package_token": "4" * 32})
    second_state = SecondRoundState(
        first_package_token=first_package.package_token,
        first_content_fingerprint=first_package.content_fingerprint,
        second_package_token=second_package.package_token,
        second_content_fingerprint=second_package.content_fingerprint,
        item_count=1,
        links=[
            SecondRoundRuleLink(
                review_item_id=second_package.items[0].review_item_id,
                rule_id=proposal.rules[0].rule_id,
                is_match=True,
            )
        ],
    )

    with pytest.raises(PrivacyValidationError) as decision_error:
        evaluate_second_round(
            proposal,
            second_package,
            second_state,
            stale_decisions,
            activated_at=ACTIVATED_AT,
            user_confirmed=True,
        )
    assert decision_error.value.code == "stale_review_package"

    stale_state = second_state.model_copy(update={"second_package_token": "5" * 32})
    with pytest.raises(PrivacyValidationError) as state_error:
        evaluate_second_round(
            proposal,
            second_package,
            stale_state,
            second_decisions,
            activated_at=ACTIVATED_AT,
            user_confirmed=True,
        )
    assert state_error.value.code == "stale_review_package"


def test_boundary_keep_becomes_manual_literal_not_rule_validation() -> None:
    first_package = review_package(
        ["daily alpha one", "daily alpha one!"],
        token="v" * 32,
        fingerprint="f" * 64,
    )
    first_decisions = decisions_for(
        first_package,
        [AmbiguousDecisionValue.KEEP, AmbiguousDecisionValue.KEEP],
    )
    proposal = build_calibration_proposal(
        first_package,
        first_decisions,
        created_at=CREATED_AT,
        proposal_id="proposal-synthetic-boundary",
    )
    second_package = review_package(
        ["boundary kept literal"],
        token="w" * 32,
        fingerprint="1" * 64,
        review_round=ReviewRound.SECOND,
    )
    second_decisions = decisions_for(second_package, [AmbiguousDecisionValue.KEEP])
    second_state = SecondRoundState(
        first_package_token=first_package.package_token,
        first_content_fingerprint=first_package.content_fingerprint,
        second_package_token=second_package.package_token,
        second_content_fingerprint=second_package.content_fingerprint,
        item_count=1,
        links=[
            SecondRoundRuleLink(
                review_item_id=second_package.items[0].review_item_id,
                rule_id=proposal.rules[0].rule_id,
                is_match=False,
            )
        ],
    )

    profile = evaluate_second_round(
        proposal,
        second_package,
        second_state,
        second_decisions,
        activated_at=ACTIVATED_AT,
        user_confirmed=True,
    )

    assert "boundary kept literal" in profile.manual_approved_texts
    assert profile.rules == []
    assert profile.first_round_keep_count == 2
    assert profile.first_round_discard_count == 0
    assert profile.first_round_uncertain_count == 0
    assert profile.second_round_keep_count == 1
    assert profile.second_round_discard_count == 0
    assert profile.second_round_uncertain_count == 0


def test_profile_aggregate_counts_include_mixed_second_round_decisions() -> None:
    first_package = review_package(
        ["daily alpha one", "daily alpha one!"],
        token="6" * 32,
        fingerprint="6" * 64,
    )
    first_decisions = decisions_for(
        first_package,
        [AmbiguousDecisionValue.KEEP, AmbiguousDecisionValue.KEEP],
    )
    proposal = build_calibration_proposal(
        first_package,
        first_decisions,
        created_at=CREATED_AT,
        proposal_id="proposal-synthetic-mixed-counts",
    )
    second_package = review_package(
        [
            "manual keep boundary",
            "discarded matched sample",
            "uncertain matched sample",
        ],
        token="7" * 32,
        fingerprint="7" * 64,
        review_round=ReviewRound.SECOND,
    )
    second_decisions = decisions_for(
        second_package,
        [
            AmbiguousDecisionValue.KEEP,
            AmbiguousDecisionValue.DISCARD,
            AmbiguousDecisionValue.UNCERTAIN,
        ],
    )
    second_state = SecondRoundState(
        first_package_token=first_package.package_token,
        first_content_fingerprint=first_package.content_fingerprint,
        second_package_token=second_package.package_token,
        second_content_fingerprint=second_package.content_fingerprint,
        item_count=3,
        links=[
            SecondRoundRuleLink(
                review_item_id=second_package.items[0].review_item_id,
                rule_id=proposal.rules[0].rule_id,
                is_match=False,
            ),
            SecondRoundRuleLink(
                review_item_id=second_package.items[1].review_item_id,
                rule_id=proposal.rules[0].rule_id,
                is_match=True,
            ),
            SecondRoundRuleLink(
                review_item_id=second_package.items[2].review_item_id,
                rule_id=proposal.rules[0].rule_id,
                is_match=True,
            ),
        ],
    )

    profile = evaluate_second_round(
        proposal,
        second_package,
        second_state,
        second_decisions,
        activated_at=ACTIVATED_AT,
        user_confirmed=True,
    )

    assert profile.rules == []
    assert profile.first_round_keep_count == 2
    assert profile.first_round_discard_count == 0
    assert profile.first_round_uncertain_count == 0
    assert profile.second_round_keep_count == 1
    assert profile.second_round_discard_count == 1
    assert profile.second_round_uncertain_count == 1
