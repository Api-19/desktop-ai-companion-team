import pytest
from pydantic import ValidationError

from companion.contracts import StyleIntent
from companion.privacy import CandidateCategory
from companion.style.models import EvaluationScenario


def test_evaluation_scenario_rejects_unknown_intent() -> None:
    with pytest.raises(ValidationError):
        EvaluationScenario(
            scenario_id="sleep-1",
            category=CandidateCategory.SLEEP,
            intent="synthetic-unknown",
            prompt="synthetic prompt",
        )


def test_evaluation_scenario_defaults_to_category_intent() -> None:
    scenario = EvaluationScenario(
        scenario_id="synthetic",
        category=CandidateCategory.SLEEP,
        prompt="synthetic prompt",
    )

    assert scenario.intent == "category_default"


def test_style_intent_contract_contains_only_approved_values() -> None:
    assert {intent.value for intent in StyleIntent} == {
        "category_default",
        "sleep_farewell",
        "sleep_reminder",
        "conversation_reopen",
        "courtesy_thanks",
        "courtesy_apology",
        "weather_cold_reminder",
        "work_study_completed",
        "morning_greeting",
        "missed_meal_care",
        "unfinished_household_reassurance",
        "shopping_comparison_request",
        "work_fatigue_care",
        "positive_emotion_share",
        "negative_emotion_check_in",
        "missed_contact_affection",
        "not_eaten_yet_care",
        "rain_reminder",
        "household_completed_acknowledgement",
        "going_out_care",
        "shopping_desire_acknowledgement",
    }
