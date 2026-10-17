# ruff: noqa: RUF001

from collections import Counter

from companion.contracts import StyleIntent, StylePack
from companion.privacy import CandidateCategory, SafeCandidatePackage
from companion.style.calibration import ReviewedCalibrationIndex, apply_reviewed_calibration
from companion.style.errors import StyleValidationError
from companion.style.evaluation import (
    _copies_example,
    _fixed_scenario_pairs,
    _normalize_whitespace,
    build_evaluation_plan,
)
from companion.style.iteration_models import (
    ContextualCorrectionEvidence,
    ContextualResponsePattern,
)
from companion.style.models import (
    AbsoluteBlindRatings,
    BlindAnswerKey,
    ContextualEvaluationReport,
    EvaluationPlan,
    EvaluationResponses,
    EvaluationScenario,
)
from companion.style.retrieval import StyleExampleIndex


def _listed(values: list[str]) -> str:
    return "、".join(values) if values else "无"


def _build_contextual_evaluation_plan(
    style_pack: StylePack,
    index: StyleExampleIndex,
    scenarios: list[EvaluationScenario],
    patterns: list[ContextualResponsePattern],
    candidate_package: SafeCandidatePackage,
    reviewed_calibration: ReviewedCalibrationIndex | None,
) -> EvaluationPlan:
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise StyleValidationError(
            "duplicate_contextual_scenario_id",
            "contextual evaluation scenario ids must be unique",
        )
    pattern_ids = [pattern.pattern_id for pattern in patterns]
    if len(pattern_ids) != len(set(pattern_ids)):
        raise StyleValidationError(
            "duplicate_contextual_pattern_id",
            "contextual response pattern ids must be unique",
        )
    candidate_bindings = {
        (candidate.category, candidate.text) for candidate in candidate_package.candidates
    }
    if any(
        (pattern.category, expression) not in candidate_bindings
        for pattern in patterns
        for expression in pattern.required_expressions
    ):
        raise StyleValidationError(
            "unsafe_contextual_required_expression",
            "contextual required expressions must exactly match same-category safe candidates",
        )

    plan = build_evaluation_plan(style_pack, index, scenarios)
    items = []
    for scenario, item in zip(scenarios, plan.items, strict=True):
        matching = sorted(
            (
                pattern
                for pattern in patterns
                if pattern.category is scenario.category and pattern.intent is scenario.intent
            ),
            key=lambda pattern: pattern.pattern_id,
        )
        if len(matching) > 1:
            raise StyleValidationError(
                "ambiguous_contextual_pattern",
                "a scenario may match at most one contextual pattern",
            )
        if not matching:
            items.append(item)
            continue

        pattern = matching[0]
        required_expressions = set(pattern.required_expressions)
        styled_examples = [
            example for example in item.styled_examples if example not in required_expressions
        ]
        styled_prompt = item.styled_prompt
        if styled_examples != item.styled_examples:
            prompt_prefix, example_header, _existing_examples = styled_prompt.rpartition(
                "同类脱敏示例：\n"
            )
            if not example_header:
                raise StyleValidationError(
                    "invalid_contextual_evaluation",
                    "contextual evaluation input is invalid",
                )
            example_lines = (
                "\n".join(f"- {example}" for example in styled_examples) or "- 无同类示例"
            )
            styled_prompt = f"{prompt_prefix}{example_header}{example_lines}"
        required_rule = (
            f"- 必须使用以下主人明确确认的表达：{_listed(pattern.required_expressions)}"
            if pattern.required_expressions
            else "- 无必须表达规则。"
        )
        prohibited_rule = (
            f"- 不得使用以下主人明确否定的表达：{_listed(pattern.prohibited_expressions)}"
            if pattern.prohibited_expressions
            else "- 无禁止表达规则。"
        )
        contextual_block = (
            "精确意图风格要求：\n"
            f"{required_rule}\n"
            f"{prohibited_rule}\n"
            f"- 回复长度范围：{pattern.min_length}–{pattern.max_length} 个非空白字符\n"
            f"- 语气标签：{_listed(pattern.tone_tags)}\n"
            f"- 标点标签：{_listed(pattern.punctuation_tags)}\n"
            "- 若风格要求与安全或语义正确性冲突，安全和语义正确性优先于风格要求。"
        )
        items.append(
            item.model_copy(
                update={
                    "styled_prompt": f"{styled_prompt}\n{contextual_block}",
                    "styled_preferences": pattern.required_expressions[:2],
                    "styled_examples": styled_examples,
                }
            )
        )
    result = EvaluationPlan.model_validate(
        plan.model_copy(update={"items": items}).model_dump(mode="json", warnings=False)
    )
    return (
        result
        if reviewed_calibration is None
        else apply_reviewed_calibration(result, reviewed_calibration)
    )


def build_contextual_evaluation_plan(
    style_pack: StylePack,
    index: StyleExampleIndex,
    scenarios: list[EvaluationScenario],
    patterns: list[ContextualResponsePattern],
    candidate_package: SafeCandidatePackage,
    *,
    reviewed_calibration: ReviewedCalibrationIndex | None = None,
) -> EvaluationPlan:
    try:
        style_pack = StylePack.model_validate(style_pack.model_dump(mode="json", warnings=False))
        scenarios = [
            EvaluationScenario.model_validate(item.model_dump(mode="json", warnings=False))
            for item in scenarios
        ]
        patterns = [
            ContextualResponsePattern.model_validate(item.model_dump(mode="json", warnings=False))
            for item in patterns
        ]
        candidate_package = SafeCandidatePackage.model_validate(
            candidate_package.model_dump(mode="json", warnings=False)
        )
        return _build_contextual_evaluation_plan(
            style_pack,
            index,
            scenarios,
            patterns,
            candidate_package,
            reviewed_calibration,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = (
            "invalid_contextual_evaluation",
            "contextual evaluation input is invalid",
        )
    finally:
        style_pack = None  # type: ignore[assignment]
        index = None  # type: ignore[assignment]
        scenarios = None  # type: ignore[assignment]
        patterns = None  # type: ignore[assignment]
        candidate_package = None  # type: ignore[assignment]
        reviewed_calibration = None  # type: ignore[assignment]
    raise StyleValidationError(*failure) from None


def _items_by_scenario(items: list[object], label: str) -> dict[str, object]:
    identifiers = [item.scenario_id for item in items]  # type: ignore[attr-defined]
    if len(identifiers) != len(set(identifiers)):
        raise StyleValidationError(
            "duplicate_scenario_id",
            f"{label} contains duplicate scenario ids",
        )
    return dict(zip(identifiers, items, strict=True))


def _require_matching_scenarios(
    expected: set[str],
    actual: set[str],
    label: str,
) -> None:
    if actual != expected:
        raise StyleValidationError(
            "scenario_set_mismatch",
            f"{label} scenario set is invalid",
        )


def _score_contextual_evaluation(
    plan: EvaluationPlan,
    responses: EvaluationResponses,
    key: BlindAnswerKey,
    ratings: AbsoluteBlindRatings,
) -> ContextualEvaluationReport:
    plan_by_id = _items_by_scenario(plan.items, "plan")
    responses_by_id = _items_by_scenario(responses.items, "responses")
    ratings_by_id = _items_by_scenario(ratings.items, "ratings")
    expected = set(plan_by_id)
    if len(plan_by_id) != 20:
        raise StyleValidationError(
            "invalid_scenario_count",
            "evaluation requires exactly twenty scenarios",
        )
    actual_pairs = {(item.scenario_id, item.category, item.intent) for item in plan.items}
    if actual_pairs != _fixed_scenario_pairs():
        raise StyleValidationError(
            "invalid_scenario_set",
            "evaluation plan does not match the fixed scenario set",
        )
    _require_matching_scenarios(expected, set(responses_by_id), "responses")
    _require_matching_scenarios(expected, set(key.styled_labels), "answer key")
    _require_matching_scenarios(expected, set(ratings_by_id), "ratings")

    styled_counts = Counter(_normalize_whitespace(item.styled) for item in responses.items)
    styled_relative_wins = 0
    baseline_relative_wins = 0
    ties = 0
    styled_absolute_successes = 0
    styled_context_failures = 0
    styled_mechanical: set[str] = set()
    neither_like_owner_scenarios = 0
    for plan_item in plan.items:
        identifier = plan_item.scenario_id
        response = responses_by_id[identifier]
        rating = ratings_by_id[identifier]
        styled_label = key.styled_labels[identifier]
        baseline_label = "B" if styled_label == "A" else "A"
        if rating.neither_like_owner:
            ties += 1
            neither_like_owner_scenarios += 1
        elif rating.preferred == styled_label:
            styled_relative_wins += 1
        elif rating.preferred == baseline_label:
            baseline_relative_wins += 1
        else:
            ties += 1

        styled_similarity = rating.similarity_a if styled_label == "A" else rating.similarity_b
        styled_context_correct = (
            rating.context_correct_a if styled_label == "A" else rating.context_correct_b
        )
        styled_human_mechanical = (
            rating.mechanical_a if styled_label == "A" else rating.mechanical_b
        )
        automatic_styled_mechanical = (
            _copies_example(response.styled, plan_item.styled_examples)
            or styled_counts[_normalize_whitespace(response.styled)] >= 3
        )
        if not styled_context_correct:
            styled_context_failures += 1
        if styled_human_mechanical or automatic_styled_mechanical:
            styled_mechanical.add(identifier)
        if (
            rating.preferred == styled_label
            and styled_similarity >= 4
            and not rating.neither_like_owner
            and styled_context_correct
            and not styled_human_mechanical
            and not automatic_styled_mechanical
        ):
            styled_absolute_successes += 1

    total = len(plan.items)
    styled_mechanical_scenarios = len(styled_mechanical)
    if (
        styled_absolute_successes >= 16
        and neither_like_owner_scenarios == 0
        and styled_context_failures == 0
        and styled_mechanical_scenarios <= 2
    ):
        conclusion = "overall_sufficient"
    elif (
        styled_absolute_successes > 0
        and styled_mechanical_scenarios <= total * 0.2
        and neither_like_owner_scenarios < total * 0.2
    ):
        conclusion = "locally_sufficient"
    else:
        conclusion = "insufficient"

    return ContextualEvaluationReport(
        total_scenarios=total,
        styled_relative_wins=styled_relative_wins,
        baseline_relative_wins=baseline_relative_wins,
        ties=ties,
        styled_absolute_successes=styled_absolute_successes,
        styled_context_failures=styled_context_failures,
        styled_mechanical_scenarios=styled_mechanical_scenarios,
        neither_like_owner_scenarios=neither_like_owner_scenarios,
        conclusion=conclusion,
    )


def score_contextual_evaluation(
    plan: EvaluationPlan,
    responses: EvaluationResponses,
    key: BlindAnswerKey,
    ratings: AbsoluteBlindRatings,
) -> ContextualEvaluationReport:
    try:
        plan = EvaluationPlan.model_validate(plan.model_dump(mode="json", warnings=False))
        responses = EvaluationResponses.model_validate(
            responses.model_dump(mode="json", warnings=False)
        )
        key = BlindAnswerKey.model_validate(key.model_dump(mode="json", warnings=False))
        ratings = AbsoluteBlindRatings.model_validate(
            ratings.model_dump(mode="json", warnings=False)
        )
        return _score_contextual_evaluation(plan, responses, key, ratings)
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = (
            "invalid_contextual_evaluation",
            "contextual evaluation input is invalid",
        )
    finally:
        plan = None  # type: ignore[assignment]
        responses = None  # type: ignore[assignment]
        key = None  # type: ignore[assignment]
        ratings = None  # type: ignore[assignment]
    raise StyleValidationError(*failure) from None


def _normalized_prohibited_expressions(values: list[str]) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise StyleValidationError(
            "invalid_prohibited_expressions",
            "prohibited expressions must be explicit owner-confirmed strings",
        )
    normalized = [value.strip() for value in values]
    if any(not value or len(value) > 32 for value in normalized) or len(normalized) != len(
        set(normalized)
    ):
        raise StyleValidationError(
            "invalid_prohibited_expressions",
            "prohibited expressions must be unique nonblank strings of at most 32 characters",
        )
    return normalized


def _analyze_contextual_differences(
    plan: EvaluationPlan,
    responses: EvaluationResponses,
    key: BlindAnswerKey,
    ratings: AbsoluteBlindRatings,
    candidate_package: SafeCandidatePackage,
    prohibited_expressions: list[str],
) -> list[ContextualCorrectionEvidence]:
    plan_by_id = _items_by_scenario(plan.items, "plan")
    responses_by_id = _items_by_scenario(responses.items, "responses")
    ratings_by_id = _items_by_scenario(ratings.items, "ratings")
    expected = set(plan_by_id)
    _require_matching_scenarios(expected, set(responses_by_id), "responses")
    _require_matching_scenarios(expected, set(key.styled_labels), "answer key")
    _require_matching_scenarios(expected, set(ratings_by_id), "ratings")

    candidate_bindings = {
        (candidate.category, candidate.text.strip()) for candidate in candidate_package.candidates
    }
    raw_evidence: list[tuple[str, CandidateCategory, StyleIntent, str, str]] = []
    for plan_item in sorted(plan.items, key=lambda item: item.scenario_id):
        scenario_id = plan_item.scenario_id
        response = responses_by_id[scenario_id]
        rating = ratings_by_id[scenario_id]
        ideal_response = rating.ideal_response.strip() if rating.ideal_response is not None else ""
        if (
            0 < len(ideal_response) <= 32
            and (plan_item.category, ideal_response) in candidate_bindings
        ):
            raw_evidence.append(
                (
                    scenario_id,
                    plan_item.category,
                    plan_item.intent,
                    "required_expression",
                    ideal_response,
                )
            )

        styled_label = key.styled_labels[scenario_id]
        styled_similarity = rating.similarity_a if styled_label == "A" else rating.similarity_b
        if styled_similarity >= 4 and not rating.neither_like_owner:
            continue
        for expression in prohibited_expressions:
            if expression in response.styled:
                raw_evidence.append(
                    (
                        scenario_id,
                        plan_item.category,
                        plan_item.intent,
                        "prohibited_expression",
                        expression,
                    )
                )

    correction_order = {"required_expression": 0, "prohibited_expression": 1}
    raw_evidence.sort(key=lambda item: (item[0], correction_order[item[3]], item[4]))
    aggregated: dict[tuple[CandidateCategory, StyleIntent, str, str], tuple[str, int]] = {}
    for scenario_id, category, intent, correction_type, expression in raw_evidence:
        correction = (category, intent, correction_type, expression)
        first_scenario, count = aggregated.get(correction, (scenario_id, 0))
        aggregated[correction] = (first_scenario, count + 1)

    evidence = [
        ContextualCorrectionEvidence(
            scenario_id=scenario_id,
            category=category,
            intent=intent,
            correction_type=correction_type,
            expression=expression,
            aggregate_count=count,
            source="explicit_owner_feedback",
        )
        for (category, intent, correction_type, expression), (
            scenario_id,
            count,
        ) in aggregated.items()
    ]
    return sorted(
        evidence,
        key=lambda item: (
            item.scenario_id,
            correction_order[item.correction_type],
            item.expression,
        ),
    )[:5]


def analyze_contextual_differences(
    plan: EvaluationPlan,
    responses: EvaluationResponses,
    key: BlindAnswerKey,
    ratings: AbsoluteBlindRatings,
    candidate_package: SafeCandidatePackage,
    prohibited_expressions: list[str],
) -> list[ContextualCorrectionEvidence]:
    try:
        plan = EvaluationPlan.model_validate(plan.model_dump(mode="json", warnings=False))
        responses = EvaluationResponses.model_validate(
            responses.model_dump(mode="json", warnings=False)
        )
        key = BlindAnswerKey.model_validate(key.model_dump(mode="json", warnings=False))
        ratings = AbsoluteBlindRatings.model_validate(
            ratings.model_dump(mode="json", warnings=False)
        )
        candidate_package = SafeCandidatePackage.model_validate(
            candidate_package.model_dump(mode="json", warnings=False)
        )
        prohibited_expressions = _normalized_prohibited_expressions(prohibited_expressions)
        return _analyze_contextual_differences(
            plan,
            responses,
            key,
            ratings,
            candidate_package,
            prohibited_expressions,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = (
            "invalid_contextual_corrections",
            "contextual correction input is invalid",
        )
    finally:
        plan = None  # type: ignore[assignment]
        responses = None  # type: ignore[assignment]
        key = None  # type: ignore[assignment]
        ratings = None  # type: ignore[assignment]
        candidate_package = None  # type: ignore[assignment]
        prohibited_expressions = None  # type: ignore[assignment]
    raise StyleValidationError(*failure) from None
