import json
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest

from companion.contracts import StyleIntent
from companion.privacy import CandidateCategory, calibration_submission_id
from companion.style import (
    AbsoluteBlindRatings,
    AbsoluteScenarioRating,
    CoverageStatus,
    EvaluationPlan,
    EvaluationPlanItem,
    StyleValidationError,
    build_ideal_response_submissions,
)


def _plan() -> EvaluationPlan:
    return _fixed_plan()


def _fixed_plan() -> EvaluationPlan:
    scenarios = json.loads(
        Path("fixtures/synthetic/style_evaluation_scenarios.json").read_text(encoding="utf-8")
    )
    return EvaluationPlan(
        style_version="synthetic-v1",
        items=[
            EvaluationPlanItem(
                scenario_id=scenario["scenario_id"],
                category=scenario["category"],
                intent=scenario["intent"],
                baseline_prompt="synthetic baseline",
                styled_prompt="synthetic styled",
                styled_examples=[],
                coverage_status=CoverageStatus.MISSING,
            )
            for scenario in scenarios
        ],
    )


def _rating(scenario_id: str, ideal: str | None) -> AbsoluteScenarioRating:
    return AbsoluteScenarioRating(
        scenario_id=scenario_id,
        preferred="tie",
        similarity_a=1,
        similarity_b=1,
        neither_like_owner=True,
        context_correct_a=True,
        context_correct_b=True,
        mechanical_a=False,
        mechanical_b=False,
        ideal_response=ideal,
    )


def _ratings(ideal_by_scenario: dict[str, str | None]) -> AbsoluteBlindRatings:
    return AbsoluteBlindRatings(
        items=[
            _rating(item.scenario_id, ideal_by_scenario.get(item.scenario_id))
            for item in _fixed_plan().items
        ]
    )


def _style_traceback_retains(error: BaseException, sentinel: str) -> bool:
    return any(
        "/companion/style/" in frame.f_code.co_filename.replace("\\", "/")
        and sentinel in repr(frame.f_locals)
        for frame, _line_number in traceback.walk_tb(error.__traceback__)
    )


def test_builds_only_explicit_ideal_responses_with_plan_binding() -> None:
    submitted_at = datetime(2026, 7, 22, tzinfo=UTC)
    package = build_ideal_response_submissions(
        _plan(),
        _ratings({"sleep-1": "  合成安安  "}),
        submitted_at=submitted_at,
    )

    assert package.submission_count == 1
    submission = package.submissions[0]
    assert submission.text == "合成安安"
    assert submission.category is CandidateCategory.SLEEP
    assert submission.intent == StyleIntent.SLEEP_FAREWELL.value
    assert submission.source == "explicit_owner_ideal_response"
    assert submission.submission_id == calibration_submission_id(
        scenario_id="sleep-1",
        category=CandidateCategory.SLEEP,
        intent=StyleIntent.SLEEP_FAREWELL.value,
    )


def test_output_is_deterministic_and_sorted_by_scenario() -> None:
    submitted_at = datetime(2026, 7, 22, tzinfo=UTC)
    ratings = _ratings({"meals-1": "合成吃饭", "sleep-1": "合成安安"})

    first = build_ideal_response_submissions(_plan(), ratings, submitted_at=submitted_at)
    second = build_ideal_response_submissions(_plan(), ratings, submitted_at=submitted_at)

    assert first == second
    assert [item.scenario_id for item in first.submissions] == ["meals-1", "sleep-1"]


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("no_ideals", "no_ideal_responses"),
        ("missing_rating", "scenario_set_mismatch"),
        ("unknown_rating", "scenario_set_mismatch"),
    ],
)
def test_rejects_absent_ideals_or_scenario_mismatch(
    case: str,
    code: str,
) -> None:
    ratings = _ratings({})
    if case == "missing_rating":
        ratings = ratings.model_copy(update={"items": ratings.items[:-1]})
    elif case == "unknown_rating":
        ratings = ratings.model_copy(
            update={
                "items": [
                    _rating("unknown", "合成文本"),
                    *ratings.items[1:],
                ]
            }
        )
    with pytest.raises(StyleValidationError, match=code):
        build_ideal_response_submissions(
            _plan(), ratings, submitted_at=datetime(2026, 7, 22, tzinfo=UTC)
        )


@pytest.mark.parametrize(
    "migrated_scenario_id",
    [
        "greeting-2",
        "courtesy-1",
        "courtesy-2",
        "weather-1",
        "work_study-1",
    ],
)
def test_rejects_complete_old_plan_when_migrated_scenario_uses_default_intent(
    migrated_scenario_id: str,
) -> None:
    plan = _fixed_plan()
    plan = plan.model_copy(
        update={
            "items": [
                item.model_copy(update={"intent": StyleIntent.CATEGORY_DEFAULT})
                if item.scenario_id == migrated_scenario_id
                else item
                for item in plan.items
            ]
        }
    )
    ratings = _ratings(
        {
            migrated_scenario_id: "synthetic ideal",
        }
    )

    with pytest.raises(StyleValidationError, match="invalid_scenario_set"):
        build_ideal_response_submissions(
            plan,
            ratings,
            submitted_at=datetime(2026, 7, 26, tzinfo=UTC),
        )


def test_rejects_blank_ideal_without_leaking_text() -> None:
    rating = _rating("sleep-1", "合成安安").model_copy(update={"ideal_response": "   "})
    ratings = _ratings({})
    ratings = ratings.model_copy(
        update={
            "items": [
                rating if item.scenario_id == rating.scenario_id else item for item in ratings.items
            ]
        }
    )

    with pytest.raises(StyleValidationError, match="invalid_ideal_response") as caught:
        build_ideal_response_submissions(
            _plan(), ratings, submitted_at=datetime(2026, 7, 22, tzinfo=UTC)
        )

    assert "   " not in str(caught.value)


def test_api_has_no_response_or_baseline_text_input() -> None:
    import inspect

    assert list(inspect.signature(build_ideal_response_submissions).parameters) == [
        "plan",
        "ratings",
        "submitted_at",
    ]


def test_submission_failure_does_not_retain_or_emit_ideal_response(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "SYNTHETIC_PRIVATE_IDEAL_SENTINEL"
    ratings = _ratings({"sleep-1": sentinel})
    ratings = ratings.model_copy(
        update={
            "items": [
                *ratings.items[:-1],
                _rating("unknown", None),
            ]
        }
    )

    with (
        warnings.catch_warnings(record=True) as caught_warnings,
        pytest.raises(StyleValidationError) as caught,
    ):
        build_ideal_response_submissions(
            _plan(), ratings, submitted_at=datetime(2026, 7, 22, tzinfo=UTC)
        )

    assert sentinel not in str(caught.value)
    assert sentinel not in "\n".join(str(item.message) for item in caught_warnings)
    assert sentinel not in capsys.readouterr().err
    assert not _style_traceback_retains(caught.value, sentinel)


def test_invalid_timestamp_is_sanitized_without_retaining_ideal_response() -> None:
    sentinel = "SYNTHETIC_PRIVATE_IDEAL_TIMESTAMP_SENTINEL"
    ratings = _ratings({"sleep-1": sentinel})

    with pytest.raises(StyleValidationError, match="invalid_ideal_response") as caught:
        build_ideal_response_submissions(_plan(), ratings, submitted_at=datetime(2026, 7, 22))

    assert sentinel not in str(caught.value)
    assert not _style_traceback_retains(caught.value, sentinel)
