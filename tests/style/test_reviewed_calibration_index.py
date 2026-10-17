import traceback
from datetime import UTC, datetime
from pathlib import Path

import pytest

from companion.contracts import StyleIntent
from companion.privacy import (
    CALIBRATION_CANDIDATE_RELATIVE_PATH,
    CalibrationCandidate,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CandidateCategory,
    calibration_candidate_id,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
    publish_calibration_candidate_package,
)
from companion.style import (
    ReviewedCalibrationIndex,
    StyleValidationError,
    load_reviewed_calibration_package,
)

TRUE_NAMES = ["合成真实姓名"]
APPROVED_NICKNAMES = ["合成爱称"]


def _style_traceback_retains(error: BaseException, sentinel: str) -> bool:
    return any(
        "/companion/style/" in frame.f_code.co_filename.replace("\\", "/")
        and sentinel in repr(frame.f_locals)
        for frame, _line_number in traceback.walk_tb(error.__traceback__)
    )


def _package(*items: tuple[CandidateCategory, str, str]) -> CalibrationCandidatePackage:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    candidates = [
        CalibrationCandidate(
            candidate_id=calibration_candidate_id(
                category=category,
                intent=intent,
                text=text,
            ),
            category=category,
            intent=intent,
            text=text,
        )
        for category, intent, text in items
    ]
    return CalibrationCandidatePackage(
        created_at=now,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=calibration_identity_config_fingerprint(
            TRUE_NAMES,
            APPROVED_NICKNAMES,
        ),
        candidate_count=len(candidates),
        candidates=candidates,
        audit=CalibrationCandidateAudit(
            reviewed_at=now,
            submission_count=len(candidates),
            approved_count=len(candidates),
            rejected_count=0,
            privacy_rejected_count=0,
            revoked_count=0,
            output_count=len(candidates),
        ),
    )


def _publish(package: CalibrationCandidatePackage, private_root: Path) -> Path:
    return publish_calibration_candidate_package(
        package,
        private_data_dir=private_root,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=APPROVED_NICKNAMES,
    )


def test_loader_requires_formal_path_fingerprint_and_runtime_identity_config(
    tmp_path: Path,
) -> None:
    package = _package(
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, "合成安安"),
    )
    source = _publish(package, tmp_path)

    loaded = load_reviewed_calibration_package(
        source,
        private_data_dir=tmp_path,
        expected_fingerprint=package.content_fingerprint,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=APPROVED_NICKNAMES,
    )

    assert loaded == package
    assert source == tmp_path / CALIBRATION_CANDIDATE_RELATIVE_PATH


def test_index_retrieves_only_exact_category_and_intent_deterministically() -> None:
    package = _package(
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER.value, "合成早点睡"),
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, "合成安安二"),
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, "合成安安一"),
        (CandidateCategory.MEALS, StyleIntent.CATEGORY_DEFAULT.value, "合成吃饭"),
    )

    index = ReviewedCalibrationIndex(package)

    first = index.retrieve(
        CandidateCategory.SLEEP,
        StyleIntent.SLEEP_FAREWELL,
    )
    second = index.retrieve(
        CandidateCategory.SLEEP,
        StyleIntent.SLEEP_FAREWELL,
    )
    assert first == second
    assert len(first) == 2
    assert set(first) == {"合成安安一", "合成安安二"}
    assert index.retrieve(CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER) == ["合成早点睡"]
    assert index.retrieve(CandidateCategory.MEALS, StyleIntent.SLEEP_FAREWELL) == []


def test_index_exposes_only_long_examples_in_deterministic_order() -> None:
    long_first = "合成完整示例甲" + "呀" * 26
    long_second = "合成完整示例乙" + "呀" * 26
    assert len(long_first) == len(long_second) == 33
    package = _package(
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, "合成短表达"),
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, long_second),
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, long_first),
    )
    index = ReviewedCalibrationIndex(package)

    assert index.retrieve_long_examples(
        CandidateCategory.SLEEP,
        StyleIntent.SLEEP_FAREWELL,
    ) == [long_second, long_first]
    assert index.retrieve_long_examples(
        CandidateCategory.SLEEP,
        StyleIntent.SLEEP_FAREWELL,
        limit=1,
    ) == [long_second]
    with pytest.raises(StyleValidationError, match="invalid_calibration_limit"):
        index.retrieve_long_examples(
            CandidateCategory.SLEEP,
            StyleIntent.SLEEP_FAREWELL,
            limit=0,
        )


def test_long_first_retrieval_preserves_order_and_limit() -> None:
    short_first = "a" * 32
    short_second = "b" * 32
    long_first = "c" * 33
    long_second = "d" * 33
    category = CandidateCategory.SLEEP
    intent = StyleIntent.SLEEP_FAREWELL
    index = ReviewedCalibrationIndex(
        _package(
            (category, intent.value, short_first),
            (category, intent.value, long_first),
            (category, intent.value, long_second),
        )
    )

    assert index.retrieve_long_first_examples(category, intent, limit=1) == [long_first]
    assert index.retrieve_long_first_examples(category, intent, limit=2) == [
        long_first,
        long_second,
    ]

    short_index = ReviewedCalibrationIndex(
        _package(
            (category, intent.value, short_first),
            (category, intent.value, short_second),
        )
    )
    assert short_index.retrieve_long_first_examples(category, intent) == [
        short_first,
        short_second,
    ]
    assert ReviewedCalibrationIndex(_package()).retrieve_long_first_examples(category, intent) == []
    for limit in (0, 3):
        with pytest.raises(StyleValidationError, match="invalid_calibration_limit"):
            index.retrieve_long_first_examples(category, intent, limit=limit)


def test_index_exposes_all_short_expressions_without_omission() -> None:
    package = _package(
        (CandidateCategory.WEATHER, StyleIntent.WEATHER_COLD_REMINDER.value, "B合成短表达"),
        (CandidateCategory.WEATHER, StyleIntent.WEATHER_COLD_REMINDER.value, "A合成短表达"),
        (
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER.value,
            "合成长示例" + "呀" * 28,
        ),
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER.value, "合成错意图"),
    )
    index = ReviewedCalibrationIndex(package)

    assert index.retrieve_short_expressions(
        CandidateCategory.WEATHER,
        StyleIntent.WEATHER_COLD_REMINDER,
    ) == ["A合成短表达", "B合成短表达"]
    assert (
        index.retrieve_short_expressions(
            CandidateCategory.SLEEP,
            StyleIntent.WEATHER_COLD_REMINDER,
        )
        == []
    )


def test_index_deduplicates_text_and_honors_limit() -> None:
    original = _package(
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, "合成安安"),
    )
    duplicate = original.candidates[0].model_copy(
        update={"candidate_id": "f" * 64},
    )
    forged = original.model_copy(
        update={
            "candidate_count": 2,
            "candidates": [original.candidates[0], duplicate],
            "audit": original.audit.model_copy(
                update={"submission_count": 2, "approved_count": 2, "output_count": 2}
            ),
            "content_fingerprint": calibration_review_fingerprint(
                [original.candidates[0], duplicate]
            ),
        }
    )

    with pytest.raises(StyleValidationError, match="invalid_reviewed_calibration"):
        ReviewedCalibrationIndex(forged)

    assert ReviewedCalibrationIndex(original).retrieve(
        CandidateCategory.SLEEP,
        StyleIntent.SLEEP_FAREWELL,
        limit=1,
    ) == ["合成安安"]


def test_unknown_intent_fails_the_whole_index() -> None:
    package = _package(
        (CandidateCategory.SLEEP, "synthetic-unknown-intent", "合成文本"),
    )

    with pytest.raises(StyleValidationError, match="invalid_calibration_intent"):
        ReviewedCalibrationIndex(package)


def test_loader_rejects_wrong_path_fingerprint_and_empty_identity_config(
    tmp_path: Path,
) -> None:
    package = _package(
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, "合成安安"),
    )
    source = _publish(package, tmp_path)

    with pytest.raises(StyleValidationError, match="unsafe_reviewed_calibration_path"):
        load_reviewed_calibration_package(
            tmp_path / "other.json",
            private_data_dir=tmp_path,
            expected_fingerprint=package.content_fingerprint,
            true_name_terms=TRUE_NAMES,
            approved_nicknames=APPROVED_NICKNAMES,
        )
    with pytest.raises(StyleValidationError, match="stale_calibration_package"):
        load_reviewed_calibration_package(
            source,
            private_data_dir=tmp_path,
            expected_fingerprint="0" * 64,
            true_name_terms=TRUE_NAMES,
            approved_nicknames=APPROVED_NICKNAMES,
        )
    with pytest.raises(StyleValidationError, match="invalid_candidate_terms"):
        load_reviewed_calibration_package(
            source,
            private_data_dir=tmp_path,
            expected_fingerprint=package.content_fingerprint,
            true_name_terms=[],
            approved_nicknames=APPROVED_NICKNAMES,
        )


def test_empty_formal_package_has_zero_results() -> None:
    package = _package()

    assert (
        ReviewedCalibrationIndex(package).retrieve(
            CandidateCategory.SLEEP,
            StyleIntent.SLEEP_FAREWELL,
        )
        == []
    )


def test_loader_failure_traceback_does_not_retain_path_or_identity(tmp_path: Path) -> None:
    sentinel = "SYNTHETIC_PRIVATE_IDENTITY_SENTINEL"
    private_root = tmp_path / sentinel
    package = _package(
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL.value, "合成安安"),
    )
    source = _publish(package, private_root)

    with pytest.raises(StyleValidationError) as caught:
        load_reviewed_calibration_package(
            source,
            private_data_dir=private_root,
            expected_fingerprint="0" * 64,
            true_name_terms=[sentinel],
            approved_nicknames=APPROVED_NICKNAMES,
        )

    assert sentinel not in str(caught.value)
    assert not _style_traceback_retains(caught.value, sentinel)


def test_index_failure_traceback_does_not_retain_candidate_text() -> None:
    sentinel = "SYNTHETIC_PRIVATE_CANDIDATE_SENTINEL"
    package = _package(
        (CandidateCategory.SLEEP, "synthetic-unknown-intent", sentinel),
    )

    with pytest.raises(StyleValidationError) as caught:
        ReviewedCalibrationIndex(package)

    assert sentinel not in str(caught.value)
    assert not _style_traceback_retains(caught.value, sentinel)
