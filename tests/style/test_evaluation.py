import json
import traceback
import warnings
from collections import Counter
from pathlib import Path

import pytest

from companion.contracts import StyleIntent, StylePack
from companion.contracts.style import StylePreference
from companion.privacy import CandidateCategory
from companion.style import (
    EvaluationScenario,
    StyleExampleIndex,
    StyleValidationError,
    build_evaluation_plan,
    build_style_pack,
    create_blind_review,
    extract_style_features,
    score_evaluation,
)
from companion.style.models import (
    BlindAnswerKey,
    BlindRatings,
    CoverageStatus,
    EvaluationPlan,
    EvaluationPlanItem,
    EvaluationResponses,
    ScenarioRating,
    ScenarioResponses,
)

SCENARIO_SOURCE = Path("fixtures/synthetic/style_evaluation_scenarios.json")
EVALUATED_CATEGORIES = (
    CandidateCategory.GREETING,
    CandidateCategory.COURTESY,
    CandidateCategory.MEALS,
    CandidateCategory.SLEEP,
    CandidateCategory.WEATHER,
    CandidateCategory.HOUSEHOLD,
    CandidateCategory.SHOPPING,
    CandidateCategory.WORK_STUDY,
    CandidateCategory.GENERAL_EMOTION,
    CandidateCategory.GENTLE_AFFECTION,
)


def _traceback_retains(error: BaseException, sentinel: str) -> bool:
    for frame, _ in traceback.walk_tb(error.__traceback__):
        if "/companion/style/" in frame.f_code.co_filename.replace("\\", "/") and sentinel in repr(
            frame.f_locals
        ):
            return True
    return False


def test_evaluation_plan_scrubs_forged_style_pack_example_sentinel() -> None:
    sentinel = "synthetic-forged-style-example-sentinel"
    forged = StylePack.model_construct(
        version="synthetic",
        statistics={},
        scene_tags=[],
        forbidden_traits=[],
        examples=[sentinel],
        preferences=[],
    )
    scenario = EvaluationScenario(
        scenario_id="synthetic",
        category=CandidateCategory.SLEEP,
        prompt="synthetic prompt",
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(StyleValidationError) as caught:
            build_evaluation_plan(forged, "invalid-index", [scenario])  # type: ignore[arg-type]

    assert caught.value.code == "invalid_evaluation_plan"
    assert sentinel not in str(caught.value)
    assert sentinel not in "\n".join(str(item.message) for item in caught_warnings)
    assert not _traceback_retains(caught.value, sentinel)


def test_evaluation_plan_rejects_runtime_invalid_inputs_fail_closed() -> None:
    with pytest.raises(StyleValidationError, match="invalid_evaluation_plan"):
        build_evaluation_plan("invalid-style-pack", "invalid-index", [])  # type: ignore[arg-type]


def test_fixture_has_twenty_unique_scenarios_across_ten_categories() -> None:
    raw = json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
    scenarios = [EvaluationScenario.model_validate(item) for item in raw]

    assert len(scenarios) == 20
    assert len({item.scenario_id for item in scenarios}) == 20
    assert Counter(item.category for item in scenarios) == {
        category: 2 for category in EVALUATED_CATEGORIES
    }
    assert CandidateCategory.CALIBRATED_DAILY not in {item.category for item in scenarios}


def test_fixed_scenarios_have_only_approved_fine_grained_intents() -> None:
    scenarios = [
        EvaluationScenario.model_validate(item)
        for item in json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
    ]
    expected = {
        "greeting-1": StyleIntent.MORNING_GREETING,
        "greeting-2": StyleIntent.CONVERSATION_REOPEN,
        "courtesy-1": StyleIntent.COURTESY_THANKS,
        "courtesy-2": StyleIntent.COURTESY_APOLOGY,
        "meals-1": StyleIntent.MISSED_MEAL_CARE,
        "meals-2": StyleIntent.NOT_EATEN_YET_CARE,
        "sleep-1": StyleIntent.SLEEP_FAREWELL,
        "sleep-2": StyleIntent.SLEEP_REMINDER,
        "weather-1": StyleIntent.WEATHER_COLD_REMINDER,
        "weather-2": StyleIntent.RAIN_REMINDER,
        "household-1": StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
        "household-2": StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE,
        "shopping-1": StyleIntent.SHOPPING_COMPARISON_REQUEST,
        "shopping-2": StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
        "work_study-1": StyleIntent.WORK_STUDY_COMPLETED,
        "work_study-2": StyleIntent.WORK_FATIGUE_CARE,
        "general_emotion-1": StyleIntent.POSITIVE_EMOTION_SHARE,
        "general_emotion-2": StyleIntent.NEGATIVE_EMOTION_CHECK_IN,
        "gentle_affection-1": StyleIntent.MISSED_CONTACT_AFFECTION,
        "gentle_affection-2": StyleIntent.GOING_OUT_CARE,
    }

    assert {
        item.scenario_id: item.intent for item in scenarios if item.scenario_id in expected
    } == expected
    assert all(
        item.intent is StyleIntent.CATEGORY_DEFAULT
        for item in scenarios
        if item.scenario_id not in expected
    )


def test_shopping_desire_fixed_prompt_matches_acknowledgement_policy() -> None:
    scenarios = json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
    shopping = next(item for item in scenarios if item["scenario_id"] == "shopping-2")

    assert shopping["prompt"] == "简短回应当前的购物想法\uff0c不主动扩展未请求的购买建议。"


def test_sleep_farewell_prefers_anan_without_affecting_reminder(
    synthetic_candidate_package,
) -> None:
    features = extract_style_features(synthetic_candidate_package)
    pack = build_style_pack(synthetic_candidate_package, features).model_copy(
        update={
            "preferences": [
                StylePreference(
                    rule_id="synthetic-anan",
                    category="sleep",
                    expression="安安",
                    trigger_tags=["sleep"],
                    intent_tags=[StyleIntent.SLEEP_FAREWELL],
                    max_consecutive_uses=1,
                )
            ]
        }
    )
    index = StyleExampleIndex.from_package(synthetic_candidate_package)
    scenarios = [
        EvaluationScenario.model_validate(item)
        for item in json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
    ]

    plan = build_evaluation_plan(pack, index, scenarios)

    farewell = next(item for item in plan.items if item.scenario_id == "sleep-1")
    reminder = next(item for item in plan.items if item.scenario_id == "sleep-2")
    assert farewell.styled_preferences == ["安安"]
    assert len(farewell.styled_preferences) + len(farewell.styled_examples) <= 2
    assert "已批准意图偏好" in farewell.styled_prompt
    assert "仅在语境自然时优先" in farewell.styled_prompt
    assert "连续使用上限 1" in farewell.styled_prompt
    assert reminder.styled_preferences == []


def test_sleep_farewell_selects_first_matching_preference_by_rule_id(
    synthetic_candidate_package,
) -> None:
    features = extract_style_features(synthetic_candidate_package)
    pack = build_style_pack(synthetic_candidate_package, features).model_copy(
        update={
            "preferences": [
                StylePreference(
                    rule_id=rule_id,
                    category="sleep",
                    expression=expression,
                    trigger_tags=["sleep"],
                    intent_tags=[StyleIntent.SLEEP_FAREWELL],
                    max_consecutive_uses=1,
                )
                for rule_id, expression in (
                    ("synthetic-z", "后选表达"),
                    ("synthetic-a", "先选表达"),
                )
            ]
        }
    )
    index = StyleExampleIndex.from_package(synthetic_candidate_package)
    scenario = EvaluationScenario(
        scenario_id="sleep-ranking",
        category=CandidateCategory.SLEEP,
        intent=StyleIntent.SLEEP_FAREWELL,
        prompt="合成睡前告别场景",
    )

    item = build_evaluation_plan(pack, index, [scenario]).items[0]

    assert item.styled_preferences == ["先选表达"]
    assert "后选表达" not in item.styled_prompt


def test_style_prompt_preserves_full_baseline_and_uses_same_category_examples(
    synthetic_candidate_package,
) -> None:
    scenario = EvaluationScenario(
        scenario_id="sleep-test",
        category=CandidateCategory.SLEEP,
        prompt="合成睡眠场景",
    )
    features = extract_style_features(synthetic_candidate_package)
    pack = build_style_pack(synthetic_candidate_package, features)
    index = StyleExampleIndex.from_package(synthetic_candidate_package)

    item = build_evaluation_plan(pack, index, [scenario]).items[0]

    assert item.styled_prompt.startswith(item.baseline_prompt)
    assert item.styled_prompt != item.baseline_prompt
    assert item.coverage_status == "sparse"
    assert len(item.styled_examples) == 2
    assert all("吃饭" not in example for example in item.styled_examples)
    assert "独立 AI" in item.baseline_prompt
    assert "不得冒充主人" in item.baseline_prompt
    assert "不得补写私人经历" in item.baseline_prompt
    assert "一句" in item.baseline_prompt
    assert "不要复制" in item.styled_prompt


def test_evaluation_plan_rejects_duplicate_scenario_ids(
    synthetic_candidate_package,
) -> None:
    scenario = EvaluationScenario(
        scenario_id="duplicate",
        category=CandidateCategory.SLEEP,
        prompt="合成场景",
    )
    features = extract_style_features(synthetic_candidate_package)
    pack = build_style_pack(synthetic_candidate_package, features)
    index = StyleExampleIndex.from_package(synthetic_candidate_package)

    with pytest.raises(StyleValidationError, match="invalid_evaluation_plan"):
        build_evaluation_plan(pack, index, [scenario, scenario])


def _scoring_inputs(
    *,
    styled_wins: int,
    testable: int,
    styled_mechanical: int,
    baseline_mechanical: int = 0,
) -> tuple[EvaluationPlan, EvaluationResponses, BlindAnswerKey, BlindRatings]:
    pairs = [
        (f"{category.value}-{number}", category)
        for category in EVALUATED_CATEGORIES
        for number in (1, 2)
    ]
    fixed_intents = {
        "greeting-1": StyleIntent.MORNING_GREETING,
        "greeting-2": StyleIntent.CONVERSATION_REOPEN,
        "courtesy-1": StyleIntent.COURTESY_THANKS,
        "courtesy-2": StyleIntent.COURTESY_APOLOGY,
        "meals-1": StyleIntent.MISSED_MEAL_CARE,
        "meals-2": StyleIntent.NOT_EATEN_YET_CARE,
        "sleep-1": StyleIntent.SLEEP_FAREWELL,
        "sleep-2": StyleIntent.SLEEP_REMINDER,
        "weather-1": StyleIntent.WEATHER_COLD_REMINDER,
        "weather-2": StyleIntent.RAIN_REMINDER,
        "household-1": StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
        "household-2": StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE,
        "shopping-1": StyleIntent.SHOPPING_COMPARISON_REQUEST,
        "shopping-2": StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
        "work_study-1": StyleIntent.WORK_STUDY_COMPLETED,
        "work_study-2": StyleIntent.WORK_FATIGUE_CARE,
        "general_emotion-1": StyleIntent.POSITIVE_EMOTION_SHARE,
        "general_emotion-2": StyleIntent.NEGATIVE_EMOTION_CHECK_IN,
        "gentle_affection-1": StyleIntent.MISSED_CONTACT_AFFECTION,
        "gentle_affection-2": StyleIntent.GOING_OUT_CARE,
    }
    identifiers = [identifier for identifier, _ in pairs]
    plan = EvaluationPlan(
        style_version="dev-style-synthetic",
        items=[
            EvaluationPlanItem(
                scenario_id=identifier,
                category=category,
                intent=fixed_intents.get(identifier, StyleIntent.CATEGORY_DEFAULT),
                baseline_prompt="synthetic baseline",
                styled_prompt="synthetic styled",
                styled_examples=[],
                coverage_status=(
                    CoverageStatus.TESTABLE if index < testable else CoverageStatus.MISSING
                ),
            )
            for index, (identifier, category) in enumerate(pairs)
        ],
    )
    responses = EvaluationResponses(
        items=[
            ScenarioResponses(
                scenario_id=identifier,
                baseline=f"baseline-{index}",
                styled=f"styled-{index}",
            )
            for index, identifier in enumerate(identifiers)
        ]
    )
    key = BlindAnswerKey(styled_labels=dict.fromkeys(identifiers, "A"))
    ratings = BlindRatings(
        items=[
            ScenarioRating(
                scenario_id=identifier,
                preferred="A" if index < styled_wins else "B",
                mechanical_a=index < styled_mechanical,
                mechanical_b=index < baseline_mechanical,
            )
            for index, identifier in enumerate(identifiers)
        ]
    )
    return plan, responses, key, ratings


def test_scoring_rejects_fixed_scenario_with_wrong_intent() -> None:
    plan, responses, key, ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=0,
    )
    wrong_sleep = plan.items[6].model_copy(update={"intent": "category_default"})
    plan = plan.model_copy(update={"items": [*plan.items[:6], wrong_sleep, *plan.items[7:]]})

    with pytest.raises(StyleValidationError, match="invalid_scenario_set"):
        score_evaluation(plan, responses, key, ratings)


def test_blind_review_is_reproducible_and_keeps_labels_only_in_key() -> None:
    plan, responses, _key, _ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=2,
    )

    first_review, first_key = create_blind_review(plan, responses, seed=20260718)
    second_review, second_key = create_blind_review(plan, responses, seed=20260718)

    assert first_review == second_review
    assert first_key == second_key
    assert len(first_review.items) == len(plan.items)
    assert all(not hasattr(item, "styled_label") for item in first_review.items)
    assert set(first_key.styled_labels) == {item.scenario_id for item in plan.items}
    assert {item.response_a for item in first_review.items} != {
        item.response_b for item in first_review.items
    }


@pytest.mark.parametrize(
    ("styled_wins", "testable", "mechanical", "expected"),
    [
        (16, 16, 2, "overall_sufficient"),
        (15, 16, 2, "locally_sufficient"),
        (16, 15, 2, "locally_sufficient"),
        (4, 5, 0, "locally_sufficient"),
        (3, 4, 0, "insufficient"),
        (16, 16, 3, "locally_sufficient"),
        (16, 16, 4, "locally_sufficient"),
        (20, 20, 5, "insufficient"),
    ],
)
def test_scoring_uses_conservative_thresholds(
    styled_wins: int,
    testable: int,
    mechanical: int,
    expected: str,
) -> None:
    plan, responses, key, ratings = _scoring_inputs(
        styled_wins=styled_wins,
        testable=testable,
        styled_mechanical=mechanical,
    )

    report = score_evaluation(plan, responses, key, ratings)

    assert report.conclusion == expected
    assert report.styled_wins == styled_wins
    assert report.testable_scenarios == testable
    assert report.styled_mechanical_scenarios == mechanical


def test_baseline_mechanical_flags_do_not_lower_style_conclusion() -> None:
    plan, responses, key, ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=0,
        baseline_mechanical=5,
    )

    report = score_evaluation(plan, responses, key, ratings)

    assert report.baseline_mechanical_scenarios == 5
    assert report.styled_mechanical_scenarios == 0
    assert report.conclusion == "overall_sufficient"


def test_automatic_mechanical_checks_cover_exact_span_and_cross_scene_repeat() -> None:
    plan, responses, key, ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=0,
    )
    exact = plan.items[0].model_copy(update={"styled_examples": ["完全相同示例"]})
    span = plan.items[1].model_copy(update={"styled_examples": ["前缀连续复制八个字符后缀"]})
    plan = plan.model_copy(update={"items": [exact, span, *plan.items[2:]]})
    updated_responses = list(responses.items)
    updated_responses[0] = updated_responses[0].model_copy(update={"styled": "完全相同示例"})
    updated_responses[1] = updated_responses[1].model_copy(update={"styled": "连续复制八个字符"})
    for index in (2, 3, 4):
        updated_responses[index] = updated_responses[index].model_copy(
            update={"styled": "跨场景重复回答"}
        )
    responses = responses.model_copy(update={"items": updated_responses})

    report = score_evaluation(plan, responses, key, ratings)

    assert report.styled_mechanical_scenarios == 5
    assert report.conclusion == "insufficient"


@pytest.mark.parametrize("mismatch", ["responses", "key", "ratings"])
def test_evaluation_rejects_scenario_set_mismatches(mismatch: str) -> None:
    plan, responses, key, ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=0,
    )
    if mismatch == "responses":
        responses = responses.model_copy(update={"items": responses.items[:-1]})
    elif mismatch == "key":
        key = key.model_copy(update={"styled_labels": dict(list(key.styled_labels.items())[:-1])})
    else:
        ratings = ratings.model_copy(update={"items": ratings.items[:-1]})

    with pytest.raises(StyleValidationError, match="scenario_set_mismatch"):
        score_evaluation(plan, responses, key, ratings)


def test_scoring_rejects_truncated_sixteen_scenario_plan() -> None:
    plan, responses, key, ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=0,
    )
    identifiers = {item.scenario_id for item in plan.items[:16]}
    plan = plan.model_copy(update={"items": plan.items[:16]})
    responses = responses.model_copy(update={"items": responses.items[:16]})
    key = key.model_copy(
        update={
            "styled_labels": {
                identifier: label
                for identifier, label in key.styled_labels.items()
                if identifier in identifiers
            }
        }
    )
    ratings = ratings.model_copy(update={"items": ratings.items[:16]})

    with pytest.raises(StyleValidationError, match="invalid_scenario_count"):
        score_evaluation(plan, responses, key, ratings)


def test_scoring_rejects_arbitrary_twenty_scenario_set() -> None:
    plan, responses, key, ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=0,
    )
    identifiers = [f"arbitrary-{index}" for index in range(20)]
    plan = plan.model_copy(
        update={
            "items": [
                item.model_copy(update={"scenario_id": identifier})
                for item, identifier in zip(plan.items, identifiers, strict=True)
            ]
        }
    )
    responses = responses.model_copy(
        update={
            "items": [
                item.model_copy(update={"scenario_id": identifier})
                for item, identifier in zip(responses.items, identifiers, strict=True)
            ]
        }
    )
    key = key.model_copy(update={"styled_labels": dict.fromkeys(identifiers, "A")})
    ratings = ratings.model_copy(
        update={
            "items": [
                item.model_copy(update={"scenario_id": identifier})
                for item, identifier in zip(ratings.items, identifiers, strict=True)
            ]
        }
    )

    with pytest.raises(StyleValidationError, match="invalid_scenario_set"):
        score_evaluation(plan, responses, key, ratings)


def test_blind_review_rejects_duplicate_response_ids() -> None:
    plan, responses, _key, _ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=0,
    )
    responses = responses.model_copy(update={"items": [responses.items[0], *responses.items[:-1]]})

    with pytest.raises(StyleValidationError, match="duplicate_scenario_id"):
        create_blind_review(plan, responses, seed=1)


def test_evaluation_failure_traceback_does_not_retain_private_models() -> None:
    plan, responses, key, ratings = _scoring_inputs(
        styled_wins=16,
        testable=16,
        styled_mechanical=0,
    )
    secret = "SYNTHETIC_PRIVATE_RESPONSE_7F31"
    secret_item = responses.items[0].model_copy(update={"styled": secret})
    responses = responses.model_copy(update={"items": [secret_item, *responses.items[1:-1]]})

    with pytest.raises(StyleValidationError) as caught:
        score_evaluation(plan, responses, key, ratings)

    traceback = caught.value.__traceback__
    checked_product_frame = False
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/companion/style/" in filename:
            checked_product_frame = True
            assert secret not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    assert checked_product_frame
