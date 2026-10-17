from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.contracts.chat import SpeakerRole
from companion.privacy.ambiguous_models import (
    AmbiguousDecisionItem,
    AmbiguousDecisionPackage,
    AmbiguousDecisionValue,
    AmbiguousReviewItem,
    AmbiguousReviewPackage,
    CalibrationActivationManifest,
    CalibrationActivationResult,
    CalibrationProfile,
    CalibrationProposal,
    CalibrationRule,
    ReviewContextItem,
    ReviewRound,
    SecondRoundRuleLink,
    SecondRoundState,
)

CREATED_AT = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def synthetic_review_item(index: int) -> AmbiguousReviewItem:
    return AmbiguousReviewItem(
        review_item_id=f"synthetic-review-{index:04d}",
        text=f"synthetic ambiguous text {index}",
        context_before=[ReviewContextItem(role=SpeakerRole.OWNER, text="synthetic before")],
        context_after=[ReviewContextItem(role=SpeakerRole.OTHER, text="synthetic after")],
        similarity_group="synthetic-group",
        occurrence_count=1,
    )


def synthetic_review_package(
    *,
    item_count: int = 2,
    review_round: ReviewRound = ReviewRound.FIRST,
    requested_count: int | None = None,
) -> AmbiguousReviewPackage:
    expected = 100 if review_round is ReviewRound.FIRST else 50
    return AmbiguousReviewPackage(
        package_token="p" * 32,
        review_round=review_round,
        created_at=CREATED_AT,
        content_fingerprint="f" * 64,
        requested_count=expected if requested_count is None else requested_count,
        item_count=item_count,
        items=[synthetic_review_item(index) for index in range(item_count)],
    )


def synthetic_rule(**updates: object) -> CalibrationRule:
    values: dict[str, object] = {
        "rule_id": "synthetic-rule-0001",
        "category": "calibrated_daily",
        "similarity_group": "synthetic-group",
        "normalized_template": "synthetic template",
        "similarity_threshold": 0.9,
        "first_round_approved": 2,
        "second_round_approved": 1,
        "second_round_discarded": 0,
        "second_round_uncertain": 0,
        "enabled": True,
    }
    values.update(updates)
    return CalibrationRule(**values)


def synthetic_profile(**updates: object) -> CalibrationProfile:
    values: dict[str, object] = {
        "profile_id": "synthetic-profile-0001",
        "activated_at": CREATED_AT,
        "manual_approved_texts": ["synthetic approved"],
        "rules": [synthetic_rule()],
        "first_round_keep_count": 2,
        "first_round_discard_count": 0,
        "first_round_uncertain_count": 0,
        "second_round_keep_count": 1,
        "second_round_discard_count": 0,
        "second_round_uncertain_count": 0,
        "user_confirmed": True,
        "enabled": True,
    }
    values.update(updates)
    return CalibrationProfile(**values)


def test_review_package_contains_only_ephemeral_minimized_fields() -> None:
    package = synthetic_review_package(item_count=2, review_round=ReviewRound.FIRST)

    payload = package.model_dump(mode="json")

    assert set(payload) == {
        "package_token",
        "review_round",
        "created_at",
        "content_fingerprint",
        "requested_count",
        "item_count",
        "items",
    }
    assert set(payload["items"][0]) == {
        "review_item_id",
        "text",
        "context_before",
        "context_after",
        "similarity_group",
        "occurrence_count",
    }
    serialized = package.model_dump_json()
    assert all(key not in serialized for key in ("message_id", "conversation_id", "timestamp"))


@pytest.mark.parametrize(
    ("review_round", "requested_count", "item_count"),
    [
        (ReviewRound.FIRST, 50, 2),
        (ReviewRound.SECOND, 100, 2),
        (ReviewRound.FIRST, 100, 101),
    ],
)
def test_review_package_enforces_round_limits(
    review_round: ReviewRound, requested_count: int, item_count: int
) -> None:
    with pytest.raises(ValidationError):
        synthetic_review_package(
            item_count=item_count,
            review_round=review_round,
            requested_count=requested_count,
        )


def test_decision_package_requires_a_complete_unique_set() -> None:
    package = synthetic_review_package(item_count=2)

    with pytest.raises(ValidationError):
        AmbiguousDecisionPackage(
            package_token=package.package_token,
            content_fingerprint=package.content_fingerprint,
            item_count=2,
            decisions=[
                AmbiguousDecisionItem(
                    review_item_id=package.items[0].review_item_id,
                    decision=AmbiguousDecisionValue.KEEP,
                ),
                AmbiguousDecisionItem(
                    review_item_id=package.items[0].review_item_id,
                    decision=AmbiguousDecisionValue.DISCARD,
                ),
            ],
        )


def test_enabled_rule_requires_zero_second_round_negative_evidence() -> None:
    with pytest.raises(ValidationError):
        synthetic_rule(second_round_discarded=1)


def test_profile_requires_unique_rules_and_complete_round_counts() -> None:
    rule = synthetic_rule()

    with pytest.raises(ValidationError):
        CalibrationProfile(
            profile_id="synthetic-profile-0001",
            activated_at=CREATED_AT,
            manual_approved_texts=["synthetic approved"],
            rules=[rule, rule],
            first_round_keep_count=2,
            first_round_discard_count=0,
            first_round_uncertain_count=0,
            second_round_keep_count=1,
            second_round_discard_count=0,
            second_round_uncertain_count=0,
            user_confirmed=True,
            enabled=True,
        )


def test_proposal_rejects_rule_evidence_above_first_round_keeps() -> None:
    with pytest.raises(ValidationError):
        CalibrationProposal(
            proposal_id="synthetic-proposal-0001",
            first_package_token="p" * 32,
            first_content_fingerprint="f" * 64,
            created_at=CREATED_AT,
            manual_approved_texts=["synthetic approved"],
            rules=[synthetic_rule(enabled=False, first_round_approved=2)],
            first_round_keep_count=1,
            first_round_discard_count=0,
            first_round_uncertain_count=0,
        )


@pytest.mark.parametrize(
    ("rule", "updates"),
    [
        (synthetic_rule(enabled=False, first_round_approved=1), {"first_round_keep_count": 0}),
        (
            synthetic_rule(enabled=False, second_round_approved=1),
            {"second_round_keep_count": 0},
        ),
        (
            synthetic_rule(enabled=False, second_round_discarded=1),
            {"second_round_discard_count": 0},
        ),
        (
            synthetic_rule(enabled=False, second_round_uncertain=1),
            {"second_round_uncertain_count": 0},
        ),
    ],
)
def test_profile_rejects_rule_evidence_above_its_round_aggregate(
    rule: CalibrationRule, updates: dict[str, int]
) -> None:
    with pytest.raises(ValidationError):
        synthetic_profile(rules=[rule], enabled=False, **updates)


def test_profile_allows_overlapping_rule_evidence_without_summing_rules() -> None:
    profile = synthetic_profile(
        rules=[synthetic_rule(), synthetic_rule(rule_id="synthetic-rule-0002")]
    )

    assert len(profile.rules) == 2


def test_profile_rejects_more_manual_literals_than_total_keeps() -> None:
    with pytest.raises(ValidationError):
        synthetic_profile(
            manual_approved_texts=["synthetic approved one", "synthetic approved two"],
            first_round_keep_count=1,
            second_round_keep_count=0,
        )


def test_second_round_state_requires_a_unique_complete_link_partition() -> None:
    with pytest.raises(ValidationError):
        SecondRoundState(
            first_package_token="p" * 32,
            first_content_fingerprint="f" * 64,
            second_package_token="s" * 32,
            second_content_fingerprint="g" * 64,
            item_count=2,
            links=[
                SecondRoundRuleLink(
                    review_item_id="synthetic-review-0001",
                    rule_id="synthetic-rule-0001",
                    is_match=True,
                ),
                SecondRoundRuleLink(
                    review_item_id="synthetic-review-0001",
                    rule_id="synthetic-rule-0002",
                    is_match=False,
                ),
            ],
        )


def test_second_round_state_exposes_only_ephemeral_link_partition_fields() -> None:
    state = SecondRoundState(
        first_package_token="p" * 32,
        first_content_fingerprint="f" * 64,
        second_package_token="s" * 32,
        second_content_fingerprint="g" * 64,
        item_count=1,
        links=[
            SecondRoundRuleLink(
                review_item_id="synthetic-review-0001",
                rule_id="synthetic-rule-0001",
                is_match=True,
            )
        ],
    )

    payload = state.model_dump(mode="json")

    assert set(payload) == {
        "first_package_token",
        "first_content_fingerprint",
        "second_package_token",
        "second_content_fingerprint",
        "item_count",
        "links",
    }
    assert set(payload["links"][0]) == {"review_item_id", "rule_id", "is_match"}


def test_activation_result_rejects_disabled_final_state() -> None:
    with pytest.raises(ValidationError):
        CalibrationActivationResult(
            activation_id="synthetic-activation-0001",
            active_manifest_path=Path("synthetic/active.json"),
            candidate_path=Path("synthetic/candidate.json"),
            profile_path=Path("synthetic/profile.json"),
            activated_at=CREATED_AT,
            enabled=False,
        )


def test_activation_result_rejects_naive_activation_time() -> None:
    with pytest.raises(ValidationError):
        CalibrationActivationResult(
            activation_id="synthetic-activation-0001",
            active_manifest_path=Path("synthetic/active.json"),
            candidate_path=Path("synthetic/candidate.json"),
            profile_path=Path("synthetic/profile.json"),
            activated_at=datetime(2026, 7, 17, 12, 0),
            enabled=True,
        )


def test_activation_result_contains_only_final_activation_fields() -> None:
    result = CalibrationActivationResult(
        activation_id="synthetic-activation-0001",
        active_manifest_path=Path("synthetic/active.json"),
        candidate_path=Path("synthetic/candidate.json"),
        profile_path=Path("synthetic/profile.json"),
        activated_at=CREATED_AT,
        enabled=True,
    )

    assert set(result.model_dump(mode="json")) == {
        "activation_id",
        "active_manifest_path",
        "candidate_path",
        "profile_path",
        "activated_at",
        "enabled",
    }


@pytest.mark.parametrize(
    ("mode", "enabled", "profile_path"),
    [("full", False, "versions/synthetic/profile.json"), ("base", False, "../base.json")],
)
def test_activation_manifest_rejects_inconsistent_or_nonrelative_paths(
    mode: str, enabled: bool, profile_path: str
) -> None:
    with pytest.raises(ValidationError):
        CalibrationActivationManifest(
            activation_id="synthetic-activation-0001",
            candidate_path="versions/synthetic/candidate.json",
            profile_path=profile_path,
            mode=mode,
            enabled=enabled,
            activated_at=CREATED_AT,
        )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/synthetic/absolute.json",
        r"\synthetic\rooted.json",
        r"C:\synthetic\drive.json",
        r"\\synthetic-server\share\candidate.json",
        "../synthetic/traversal.json",
        r"..\synthetic\traversal.json",
    ],
)
def test_activation_manifest_rejects_posix_and_windows_unsafe_paths(unsafe_path: str) -> None:
    with pytest.raises(ValidationError):
        CalibrationActivationManifest(
            activation_id="synthetic-activation-0001",
            candidate_path=unsafe_path,
            profile_path="versions/synthetic/profile.json",
            mode="full",
            enabled=True,
            activated_at=CREATED_AT,
        )
