from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from companion.privacy.candidate_models import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)


def test_candidate_package_contains_no_source_identity_fields() -> None:
    audit = CandidateSelectionAudit(
        processed_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
        input_count=2,
        owner_input_count=1,
        dropped_other_count=1,
        allowed_category_counts={"greeting": 1},
        rejection_reason_counts={},
        unknown_count=0,
        deduplicated_count=0,
        output_count=1,
        succeeded=True,
    )
    package = SafeCandidatePackage(
        created_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
        candidates=[
            SafeStyleCandidate(
                text="合成问候呀",
                category=CandidateCategory.GREETING,
                scene_tags=["greeting"],
            )
        ],
        audit=audit,
    )

    payload = package.model_dump()
    candidate = payload["candidates"][0]
    assert set(candidate) == {"text", "category", "scene_tags"}
    assert all(
        forbidden not in package.model_dump_json()
        for forbidden in ("conversation_id", "message_id", "speaker_role", "timestamp")
    )


def test_candidate_package_requires_timezone() -> None:
    with pytest.raises(ValidationError):
        SafeCandidatePackage(
            created_at=datetime(2026, 7, 17, 12, 0),
            candidates=[],
            audit=CandidateSelectionAudit(
                processed_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
                input_count=0,
                owner_input_count=0,
                dropped_other_count=0,
                allowed_category_counts={},
                rejection_reason_counts={},
                unknown_count=0,
                deduplicated_count=0,
                output_count=0,
                succeeded=True,
            ),
        )


def _valid_audit_values(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "processed_at": datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
        "input_count": 2,
        "owner_input_count": 1,
        "dropped_other_count": 1,
        "allowed_category_counts": {CandidateCategory.GREETING.value: 1},
        "rejection_reason_counts": {},
        "unknown_count": 0,
        "deduplicated_count": 0,
        "output_count": 1,
        "succeeded": True,
    }
    values.update(updates)
    return values


def _valid_audit(**updates: object) -> CandidateSelectionAudit:
    return CandidateSelectionAudit(**_valid_audit_values(**updates))


def _valid_candidate() -> SafeStyleCandidate:
    return SafeStyleCandidate(
        text="synthetic-safe-greeting",
        category=CandidateCategory.GREETING,
        scene_tags=[CandidateCategory.GREETING.value],
    )


def test_candidate_scene_tag_must_exactly_match_category() -> None:
    with pytest.raises(ValidationError):
        SafeStyleCandidate(
            text="synthetic-safe-greeting",
            category=CandidateCategory.GREETING,
            scene_tags=[CandidateCategory.COURTESY.value],
        )


def test_calibrated_daily_remains_a_single_exact_category_tag() -> None:
    candidate = SafeStyleCandidate(
        text="synthetic calibrated daily expression",
        category=CandidateCategory.CALIBRATED_DAILY,
        scene_tags=[CandidateCategory.CALIBRATED_DAILY.value],
    )

    assert len(CandidateCategory) == 13
    assert candidate.model_dump(mode="json")["scene_tags"] == ["calibrated_daily"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("allowed_category_counts", {"synthetic-unknown-category": 1}),
        ("allowed_category_counts", {CandidateCategory.GREETING.value: 0}),
        ("rejection_reason_counts", {"synthetic-unknown-reason": 1}),
        ("rejection_reason_counts", {"non_text": 0}),
    ],
)
def test_audit_aggregate_keys_are_approved_and_counts_are_positive(
    field_name: str,
    value: dict[str, int],
) -> None:
    with pytest.raises(ValidationError):
        _valid_audit(**{field_name: value})


def test_audit_counts_must_form_complete_input_partition() -> None:
    with pytest.raises(ValidationError):
        _valid_audit(input_count=3)


def test_audit_allowed_counts_must_equal_output_count() -> None:
    with pytest.raises(ValidationError):
        _valid_audit(input_count=3, owner_input_count=2, output_count=2)


@pytest.mark.parametrize(
    "updates",
    [
        {"rejection_reason_counts": {"non_text": 1}},
        {"unknown_count": 1},
        {"deduplicated_count": 1},
    ],
)
def test_audit_owner_outcome_partition_rejects_each_inconsistent_outcome(
    updates: dict[str, object],
) -> None:
    values = _valid_audit_values(**updates)
    input_count = int(values["input_count"])
    owner_input_count = int(values["owner_input_count"])
    dropped_other_count = int(values["dropped_other_count"])
    allowed_counts = dict(values["allowed_category_counts"])
    rejection_counts = dict(values["rejection_reason_counts"])
    output_count = int(values["output_count"])
    unknown_count = int(values["unknown_count"])
    deduplicated_count = int(values["deduplicated_count"])

    assert input_count == owner_input_count + dropped_other_count
    assert sum(allowed_counts.values()) == output_count
    assert owner_input_count != (
        output_count + sum(rejection_counts.values()) + unknown_count + deduplicated_count
    )
    with pytest.raises(ValidationError):
        CandidateSelectionAudit(**values)


@pytest.mark.parametrize(
    "updates",
    [
        {"candidates": []},
        {
            "candidates": [
                SafeStyleCandidate(
                    text="synthetic-safe-courtesy",
                    category=CandidateCategory.COURTESY,
                    scene_tags=[CandidateCategory.COURTESY.value],
                )
            ]
        },
        {"created_at": datetime(2026, 7, 17, 12, 1, tzinfo=UTC)},
    ],
)
def test_package_candidates_counts_and_timestamps_must_agree(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "created_at": datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
        "candidates": [_valid_candidate()],
        "audit": _valid_audit(),
    }
    values.update(updates)
    with pytest.raises(ValidationError):
        SafeCandidatePackage(**values)
