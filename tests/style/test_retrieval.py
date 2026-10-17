from datetime import UTC, datetime

import pytest

from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.style import CoverageStatus, StyleExampleIndex


def _package(
    *items: tuple[str, CandidateCategory],
) -> SafeCandidatePackage:
    timestamp = datetime(2026, 7, 18, tzinfo=UTC)
    candidates = [
        SafeStyleCandidate(
            text=text,
            category=category,
            scene_tags=[category.value],
        )
        for text, category in items
    ]
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.category.value] = counts.get(candidate.category.value, 0) + 1
    return SafeCandidatePackage(
        created_at=timestamp,
        candidates=candidates,
        audit=CandidateSelectionAudit(
            processed_at=timestamp,
            input_count=len(candidates),
            owner_input_count=len(candidates),
            dropped_other_count=0,
            allowed_category_counts=counts,
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=len(candidates),
            succeeded=True,
        ),
    )


def test_retrieval_is_category_scoped_and_similarity_ranked() -> None:
    index = StyleExampleIndex(
        _package(
            ("今晚早点睡", CandidateCategory.SLEEP),
            ("晚安好梦呀", CandidateCategory.SLEEP),
            ("今晚吃饭啦", CandidateCategory.MEALS),
        )
    )

    result = index.retrieve("今晚记得休息", CandidateCategory.SLEEP)

    assert result.examples == ["今晚早点睡", "晚安好梦呀"]
    assert "今晚吃饭啦" not in result.examples
    assert result.coverage_status is CoverageStatus.SPARSE
    assert result.category_count == 2


def test_retrieval_never_falls_back_to_another_category() -> None:
    index = StyleExampleIndex(_package(("晚安呀", CandidateCategory.SLEEP)))

    result = index.retrieve("出门带伞", CandidateCategory.WEATHER)

    assert result.examples == []
    assert result.coverage_status is CoverageStatus.MISSING
    assert result.category_count == 0


def test_retrieval_preserves_input_order_for_similarity_ties() -> None:
    index = StyleExampleIndex(
        _package(
            ("甲乙", CandidateCategory.GREETING),
            ("丙丁", CandidateCategory.GREETING),
            ("戊己", CandidateCategory.GREETING),
        )
    )

    result = index.retrieve("完全无关", CandidateCategory.GREETING)

    assert result.examples == ["甲乙", "丙丁"]
    assert result.coverage_status is CoverageStatus.TESTABLE


def test_retrieval_marks_one_example_as_mechanical_risk() -> None:
    index = StyleExampleIndex(_package(("回来啦", CandidateCategory.HOUSEHOLD)))

    result = index.retrieve("到家了吗", CandidateCategory.HOUSEHOLD, limit=1)

    assert result.examples == ["回来啦"]
    assert result.coverage_status is CoverageStatus.MECHANICAL_RISK


def test_retrieval_returns_distinct_examples_when_candidate_texts_repeat() -> None:
    index = StyleExampleIndex(
        _package(
            ("重复示例", CandidateCategory.GREETING),
            ("重复示例", CandidateCategory.GREETING),
            ("不同表达", CandidateCategory.GREETING),
        )
    )

    result = index.retrieve("重复", CandidateCategory.GREETING)

    assert result.examples == ["重复示例", "不同表达"]
    assert result.category_count == 2
    assert result.coverage_status is CoverageStatus.SPARSE


def test_retrieval_coverage_counts_distinct_texts_only() -> None:
    index = StyleExampleIndex(
        _package(
            ("同一句话", CandidateCategory.GREETING),
            ("同一句话", CandidateCategory.GREETING),
            ("同一句话", CandidateCategory.GREETING),
        )
    )

    result = index.retrieve("问候", CandidateCategory.GREETING)

    assert result.examples == ["同一句话"]
    assert result.category_count == 1
    assert result.coverage_status is CoverageStatus.MECHANICAL_RISK


def test_calibrated_daily_unique_matches_augment_semantic_index() -> None:
    package = _package(
        ("早安", CandidateCategory.CALIBRATED_DAILY),
        ("你好呀", CandidateCategory.CALIBRATED_DAILY),
        ("晚上好", CandidateCategory.CALIBRATED_DAILY),
    )
    original = package.model_dump(mode="json")

    result = StyleExampleIndex.from_package(package).retrieve(
        "见面问候", CandidateCategory.GREETING
    )

    assert result.coverage_status is CoverageStatus.TESTABLE
    assert result.category_count == 3
    assert package.model_dump(mode="json") == original


def test_multi_category_calibrated_text_is_not_derived() -> None:
    index = StyleExampleIndex.from_package(
        _package(("下课去拿快递", CandidateCategory.CALIBRATED_DAILY))
    )

    assert index.retrieve("学习", CandidateCategory.WORK_STUDY).examples == []
    assert index.retrieve("购物", CandidateCategory.SHOPPING).examples == []


@pytest.mark.parametrize("limit", [0, 3])
def test_retrieval_rejects_limits_outside_one_or_two(limit: int) -> None:
    index = StyleExampleIndex(_package(("晚安呀", CandidateCategory.SLEEP)))

    with pytest.raises(ValueError, match="limit"):
        index.retrieve("早点睡", CandidateCategory.SLEEP, limit=limit)
