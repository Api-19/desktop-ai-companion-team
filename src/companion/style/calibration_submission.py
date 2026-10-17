from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError

from companion.privacy import (
    CalibrationReviewPackage,
    CalibrationReviewSubmission,
    calibration_review_fingerprint,
    calibration_submission_id,
)
from companion.style.errors import StyleValidationError
from companion.style.evaluation import _fixed_scenario_pairs
from companion.style.models import AbsoluteBlindRatings, EvaluationPlan


def _build_ideal_response_submissions(
    plan: EvaluationPlan,
    ratings: AbsoluteBlindRatings,
    *,
    submitted_at: datetime,
) -> CalibrationReviewPackage:
    """Prepare explicit owner ideals for privacy review, never for direct learning."""
    try:
        validated_plan = EvaluationPlan.model_validate(plan.model_dump(mode="python"))
        validated_ratings = AbsoluteBlindRatings.model_validate(ratings.model_dump(mode="python"))
    except (AttributeError, ValidationError, ValueError):
        raise StyleValidationError(
            "invalid_ideal_response",
            "ideal response submission input is invalid",
        ) from None

    actual_scenarios = {
        (item.scenario_id, item.category, item.intent) for item in validated_plan.items
    }
    if len(validated_plan.items) != 20 or actual_scenarios != _fixed_scenario_pairs():
        raise StyleValidationError(
            "invalid_scenario_set",
            "ideal response plan does not match the fixed scenario set",
        )

    plan_by_id = {item.scenario_id: item for item in validated_plan.items}
    rating_by_id = {item.scenario_id: item for item in validated_ratings.items}
    if (
        len(plan_by_id) != len(validated_plan.items)
        or len(rating_by_id) != len(validated_ratings.items)
        or set(plan_by_id) != set(rating_by_id)
    ):
        raise StyleValidationError(
            "scenario_set_mismatch",
            "ideal response scenario set is invalid",
        )

    submissions = []
    for scenario_id in sorted(plan_by_id):
        plan_item = plan_by_id[scenario_id]
        ideal_response = rating_by_id[scenario_id].ideal_response
        if ideal_response is None:
            continue
        text = ideal_response.strip()
        if not text or len(text) > 160:
            raise StyleValidationError(
                "invalid_ideal_response",
                "ideal response submission input is invalid",
            )
        submissions.append(
            CalibrationReviewSubmission(
                submission_id=calibration_submission_id(
                    scenario_id=scenario_id,
                    category=plan_item.category,
                    intent=plan_item.intent.value,
                ),
                scenario_id=scenario_id,
                category=plan_item.category,
                intent=plan_item.intent.value,
                text=text,
                source="explicit_owner_ideal_response",
            )
        )

    if not submissions:
        raise StyleValidationError(
            "no_ideal_responses",
            "no explicit ideal responses were provided",
        )
    return CalibrationReviewPackage(
        submitted_at=submitted_at,
        content_fingerprint=calibration_review_fingerprint(submissions),
        submission_count=len(submissions),
        submissions=submissions,
    )


def build_ideal_response_submissions(
    plan: EvaluationPlan,
    ratings: AbsoluteBlindRatings,
    *,
    submitted_at: datetime,
) -> CalibrationReviewPackage:
    try:
        return _build_ideal_response_submissions(
            plan,
            ratings,
            submitted_at=submitted_at,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = (
            "invalid_ideal_response",
            "ideal response submission input is invalid",
        )
    finally:
        plan = None  # type: ignore[assignment]
        ratings = None  # type: ignore[assignment]
        submitted_at = None  # type: ignore[assignment]
    raise StyleValidationError(*failure) from None
