from datetime import UTC, datetime

import pytest

from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)


@pytest.fixture
def synthetic_candidate_package() -> SafeCandidatePackage:
    timestamp = datetime(2026, 7, 18, tzinfo=UTC)
    candidates = [
        SafeStyleCandidate(
            text="合成晚安呀",
            category=CandidateCategory.SLEEP,
            scene_tags=[CandidateCategory.SLEEP.value],
        ),
        SafeStyleCandidate(
            text="合成早点睡哦",
            category=CandidateCategory.SLEEP,
            scene_tags=[CandidateCategory.SLEEP.value],
        ),
        SafeStyleCandidate(
            text="合成吃饭啦",
            category=CandidateCategory.MEALS,
            scene_tags=[CandidateCategory.MEALS.value],
        ),
    ]
    audit = CandidateSelectionAudit(
        processed_at=timestamp,
        input_count=3,
        owner_input_count=3,
        dropped_other_count=0,
        allowed_category_counts={"sleep": 2, "meals": 1},
        rejection_reason_counts={},
        unknown_count=0,
        deduplicated_count=0,
        output_count=3,
        succeeded=True,
    )
    return SafeCandidatePackage(created_at=timestamp, candidates=candidates, audit=audit)
