# ruff: noqa: RUF001

import random
from collections import Counter
from typing import Protocol, TypeVar

from companion.contracts import StyleIntent, StylePack
from companion.privacy import CandidateCategory
from companion.style.errors import StyleValidationError
from companion.style.iteration_models import (
    ContextHoldout,
    HoldoutEvaluationPlan,
    HoldoutEvaluationPlanItem,
    LocalContextSummary,
)
from companion.style.models import (
    BlindAnswerKey,
    BlindRatings,
    BlindReview,
    BlindReviewItem,
    CoverageStatus,
    EvaluationPlan,
    EvaluationPlanItem,
    EvaluationReport,
    EvaluationResponses,
    EvaluationScenario,
)
from companion.style.retrieval import StyleExampleIndex


class _StyleEvaluationFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


class _ScenarioIdentified(Protocol):
    scenario_id: str


ScenarioItemT = TypeVar("ScenarioItemT", bound=_ScenarioIdentified)

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

_FIXED_SCENARIO_INTENTS = {
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


def _fixed_scenario_pairs() -> set[tuple[str, CandidateCategory, StyleIntent]]:
    return {
        (
            scenario_id,
            category,
            _FIXED_SCENARIO_INTENTS.get(
                scenario_id,
                StyleIntent.CATEGORY_DEFAULT,
            ),
        )
        for category in EVALUATED_CATEGORIES
        for index in (1, 2)
        for scenario_id in (f"{category.value}-{index}",)
    }


def build_baseline_prompt(prompt: str) -> str:
    return (
        "你是独立 AI，不得冒充主人，不得代表主人作出承诺，不得补写私人经历。\n"
        f"场景：{prompt}\n"
        "请只回复一句自然、克制且安全的话。"
    )


def _build_evaluation_plan(
    style_pack: StylePack,
    index: StyleExampleIndex,
    scenarios: list[EvaluationScenario],
) -> EvaluationPlan:
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if not identifiers:
        raise ValueError("evaluation scenarios must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate evaluation scenario id")

    average_length = style_pack.statistics.get("length.mean", "unknown")
    short_ratio = style_pack.statistics.get("short_message.ratio", "unknown")
    items: list[EvaluationPlanItem] = []
    for scenario in scenarios:
        baseline = build_baseline_prompt(scenario.prompt)
        preferences = sorted(
            (
                item
                for item in style_pack.preferences
                if item.category == scenario.category.value and scenario.intent in item.intent_tags
            ),
            key=lambda item: item.rule_id,
        )[:1]
        preference_expressions = [item.expression for item in preferences]
        retrieval = index.retrieve(scenario.prompt, scenario.category)
        examples = [text for text in retrieval.examples if text not in preference_expressions][
            : 2 - len(preference_expressions)
        ]
        preference_lines = (
            "\n".join(
                f"- {item.expression}；仅在语境自然时优先；连续使用上限 {item.max_consecutive_uses}"
                for item in preferences
            )
            or "- 无匹配的已批准意图偏好"
        )
        example_lines = "\n".join(f"- {example}" for example in examples) or "- 无同类示例"
        styled = (
            f"{baseline}\n"
            "风格参考（不得改变上述身份与安全规则）：\n"
            f"- 平均非空白字符长度：{average_length}\n"
            f"- 短消息比例：{short_ratio}\n"
            "- 不要复制、拼接或机械重复示例；类别缺失时不要猜测风格。\n"
            f"已批准意图偏好：\n{preference_lines}\n"
            f"同类脱敏示例：\n{example_lines}"
        )
        items.append(
            EvaluationPlanItem(
                scenario_id=scenario.scenario_id,
                category=scenario.category,
                intent=scenario.intent,
                baseline_prompt=baseline,
                styled_prompt=styled,
                styled_preferences=preference_expressions,
                styled_examples=examples,
                coverage_status=retrieval.coverage_status,
            )
        )
    return EvaluationPlan(style_version=style_pack.version, items=items)


def build_evaluation_plan(
    style_pack: StylePack,
    index: StyleExampleIndex,
    scenarios: list[EvaluationScenario],
) -> EvaluationPlan:
    try:
        style_pack = StylePack.model_validate(style_pack.model_dump(mode="json", warnings=False))
        scenarios = [
            EvaluationScenario.model_validate(item.model_dump(mode="json", warnings=False))
            for item in scenarios
        ]
        return _build_evaluation_plan(style_pack, index, scenarios)
    except _StyleEvaluationFailure as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = ("invalid_evaluation_plan", "evaluation plan is invalid")
    finally:
        style_pack = None  # type: ignore[assignment]
        index = None  # type: ignore[assignment]
        scenarios = None  # type: ignore[assignment]
    raise StyleValidationError(*failure) from None


def _build_holdout_evaluation_plan(
    style_pack: StylePack,
    index: StyleExampleIndex,
    summaries: list[LocalContextSummary],
    *,
    holdout: ContextHoldout,
) -> HoldoutEvaluationPlan:
    tokens = [summary.window_token for summary in summaries]
    if (
        not tokens
        or len(tokens) != len(set(tokens))
        or not set(tokens).issubset(holdout.holdout_window_tokens)
    ):
        raise StyleValidationError(
            "invalid_holdout_summaries",
            "holdout summaries must be non-empty and unique",
        )
    items: list[HoldoutEvaluationPlanItem] = []
    for summary in sorted(summaries, key=lambda item: item.window_token):
        baseline = build_baseline_prompt(summary.prompt)
        retrieval = index.retrieve(summary.prompt, summary.category)
        preferences = [
            preference
            for preference in style_pack.preferences
            if preference.category == summary.category.value
            and (
                not preference.intent_tags or StyleIntent.CATEGORY_DEFAULT in preference.intent_tags
            )
        ]
        preference_lines = (
            "\n".join(
                f"- {item.expression}；连续使用上限 {item.max_consecutive_uses}"
                for item in preferences
            )
            or "- 无已批准的同类偏好"
        )
        example_lines = (
            "\n".join(f"- {example}" for example in retrieval.examples) or "- 无同类示例"
        )
        styled = (
            f"{baseline}\n"
            "风格参考不得改变身份、隐私和安全规则：\n"
            f"- 平均非空白字符长度：{style_pack.statistics.get('length.mean', 'unknown')}\n"
            f"- 短消息比例：{style_pack.statistics.get('short_message.ratio', 'unknown')}\n"
            "- 不得复制、拼接或机械重复示例；无同类信息时不得猜测。\n"
            f"已批准的同类偏好：\n{preference_lines}\n"
            f"同类脱敏示例：\n{example_lines}"
        )
        items.append(
            HoldoutEvaluationPlanItem(
                window_token=summary.window_token,
                category=summary.category,
                baseline_prompt=baseline,
                styled_prompt=styled,
                styled_examples=retrieval.examples,
                coverage_status=retrieval.coverage_status,
            )
        )
    return HoldoutEvaluationPlan(base_style_version=style_pack.version, items=items)


def build_holdout_evaluation_plan(
    style_pack: StylePack,
    index: StyleExampleIndex,
    summaries: list[LocalContextSummary],
    *,
    holdout: ContextHoldout,
) -> HoldoutEvaluationPlan:
    try:
        style_pack = StylePack.model_validate(style_pack.model_dump(mode="json", warnings=False))
        summaries = [
            LocalContextSummary.model_validate(item.model_dump(mode="json", warnings=False))
            for item in summaries
        ]
        holdout = ContextHoldout.model_validate(holdout.model_dump(mode="json", warnings=False))
        return _build_holdout_evaluation_plan(
            style_pack,
            index,
            summaries,
            holdout=holdout,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = (
            "invalid_holdout_summaries",
            "holdout summaries are invalid",
        )
    del style_pack, index, summaries, holdout
    raise StyleValidationError(failure[0], failure[1]) from None


def _identifier_map(
    items: list[ScenarioItemT],
    label: str,
) -> dict[str, ScenarioItemT]:
    identifiers = [item.scenario_id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise _StyleEvaluationFailure(
            "duplicate_scenario_id",
            f"{label} contains duplicate scenario ids",
        )
    return dict(zip(identifiers, items, strict=True))


def _require_same_scenarios(
    expected: set[str],
    actual: set[str],
    label: str,
) -> None:
    if actual != expected:
        raise _StyleEvaluationFailure(
            "scenario_set_mismatch",
            f"{label} scenario set is invalid",
        )


def _create_blind_review(
    plan: EvaluationPlan,
    responses: EvaluationResponses,
    seed: int,
) -> tuple[BlindReview, BlindAnswerKey]:
    plan_by_id = _identifier_map(plan.items, "plan")
    responses_by_id = _identifier_map(responses.items, "responses")
    expected = set(plan_by_id)
    _require_same_scenarios(expected, set(responses_by_id), "responses")

    generator = random.Random(seed)
    review_items: list[BlindReviewItem] = []
    styled_labels: dict[str, str] = {}
    for plan_item in plan.items:
        response = responses_by_id[plan_item.scenario_id]
        styled_is_a = bool(generator.getrandbits(1))
        styled_labels[plan_item.scenario_id] = "A" if styled_is_a else "B"
        review_items.append(
            BlindReviewItem(
                scenario_id=plan_item.scenario_id,
                category=plan_item.category,
                response_a=response.styled if styled_is_a else response.baseline,
                response_b=response.baseline if styled_is_a else response.styled,
            )
        )
    return BlindReview(items=review_items), BlindAnswerKey(styled_labels=styled_labels)


def create_blind_review(
    plan: EvaluationPlan,
    responses: EvaluationResponses,
    *,
    seed: int,
) -> tuple[BlindReview, BlindAnswerKey]:
    try:
        return _create_blind_review(plan, responses, seed)
    except _StyleEvaluationFailure as error:
        failure = (error.code, error.detail)
    del plan, responses, seed
    raise StyleValidationError(failure[0], failure[1]) from None


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _without_whitespace(text: str) -> str:
    return "".join(text.split())


def _longest_common_contiguous(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    longest = 0
    for left_character in left:
        current = [0]
        for index, right_character in enumerate(right, start=1):
            length = previous[index - 1] + 1 if left_character == right_character else 0
            current.append(length)
            longest = max(longest, length)
        previous = current
    return longest


def _copies_example(response: str, examples: list[str]) -> bool:
    normalized_response = _normalize_whitespace(response)
    compact_response = _without_whitespace(response)
    for example in examples:
        if normalized_response == _normalize_whitespace(example):
            return True
        compact_example = _without_whitespace(example)
        span = _longest_common_contiguous(compact_response, compact_example)
        if span >= 8 and compact_response and span / len(compact_response) >= 0.6:
            return True
    return False


def _score_evaluation(
    plan: EvaluationPlan,
    responses: EvaluationResponses,
    key: BlindAnswerKey,
    ratings: BlindRatings,
) -> EvaluationReport:
    plan_by_id = _identifier_map(plan.items, "plan")
    responses_by_id = _identifier_map(responses.items, "responses")
    ratings_by_id = _identifier_map(ratings.items, "ratings")
    expected = set(plan_by_id)
    if len(plan_by_id) != 20:
        raise _StyleEvaluationFailure(
            "invalid_scenario_count",
            "evaluation requires exactly twenty scenarios",
        )
    actual_pairs = {(item.scenario_id, item.category, item.intent) for item in plan.items}
    if actual_pairs != _fixed_scenario_pairs():
        raise _StyleEvaluationFailure(
            "invalid_scenario_set",
            "evaluation plan does not match the fixed scenario set",
        )
    _require_same_scenarios(expected, set(responses_by_id), "responses")
    _require_same_scenarios(expected, set(key.styled_labels), "answer key")
    _require_same_scenarios(expected, set(ratings_by_id), "ratings")

    baseline_counts = Counter(_normalize_whitespace(item.baseline) for item in responses.items)
    styled_counts = Counter(_normalize_whitespace(item.styled) for item in responses.items)
    styled_wins = 0
    baseline_wins = 0
    ties = 0
    supported_styled_wins = 0
    styled_mechanical: set[str] = set()
    baseline_mechanical: set[str] = set()
    for plan_item in plan.items:
        identifier = plan_item.scenario_id
        response = responses_by_id[identifier]
        rating = ratings_by_id[identifier]
        styled_label = key.styled_labels[identifier]
        baseline_label = "B" if styled_label == "A" else "A"
        if rating.preferred == styled_label:
            styled_wins += 1
            if plan_item.coverage_status is CoverageStatus.TESTABLE:
                supported_styled_wins += 1
        elif rating.preferred == baseline_label:
            baseline_wins += 1
        else:
            ties += 1

        styled_human = rating.mechanical_a if styled_label == "A" else rating.mechanical_b
        baseline_human = rating.mechanical_a if baseline_label == "A" else rating.mechanical_b
        if (
            styled_human
            or _copies_example(response.styled, plan_item.styled_examples)
            or styled_counts[_normalize_whitespace(response.styled)] >= 3
        ):
            styled_mechanical.add(identifier)
        if baseline_human or baseline_counts[_normalize_whitespace(response.baseline)] >= 3:
            baseline_mechanical.add(identifier)

    total = len(plan.items)
    testable = sum(item.coverage_status is CoverageStatus.TESTABLE for item in plan.items)
    styled_mechanical_count = len(styled_mechanical)
    if styled_mechanical_count > total * 0.2:
        conclusion = "insufficient"
    elif styled_wins >= 16 and testable >= 16 and styled_mechanical_count <= 2:
        conclusion = "overall_sufficient"
    elif testable and supported_styled_wins / testable >= 0.8:
        conclusion = "locally_sufficient"
    else:
        conclusion = "insufficient"

    missing_categories = sorted(
        {item.category for item in plan.items if item.coverage_status is CoverageStatus.MISSING},
        key=lambda category: category.value,
    )
    sparse_categories = sorted(
        {
            item.category
            for item in plan.items
            if item.coverage_status in {CoverageStatus.SPARSE, CoverageStatus.MECHANICAL_RISK}
        },
        key=lambda category: category.value,
    )
    return EvaluationReport(
        total_scenarios=total,
        styled_wins=styled_wins,
        baseline_wins=baseline_wins,
        ties=ties,
        testable_scenarios=testable,
        styled_mechanical_scenarios=styled_mechanical_count,
        baseline_mechanical_scenarios=len(baseline_mechanical),
        missing_categories=missing_categories,
        sparse_categories=sparse_categories,
        conclusion=conclusion,
    )


def score_evaluation(
    plan: EvaluationPlan,
    responses: EvaluationResponses,
    key: BlindAnswerKey,
    ratings: BlindRatings,
) -> EvaluationReport:
    try:
        return _score_evaluation(plan, responses, key, ratings)
    except _StyleEvaluationFailure as error:
        failure = (error.code, error.detail)
    del plan, responses, key, ratings
    raise StyleValidationError(failure[0], failure[1]) from None
