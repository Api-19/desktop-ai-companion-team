import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from companion.contracts import StylePack
from companion.contracts.style import StyleIntent
from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.style import (
    BlindAnswerKey,
    EvaluationResponses,
    ScenarioResponses,
    StyleExampleIndex,
    StyleValidationError,
    activate_style_iteration,
    analyze_contextual_differences,
    build_context_holdout,
    build_contextual_evaluation_plan,
    build_holdout_evaluation_plan,
    build_local_context_summary,
    build_style_pack,
    extract_style_features,
    invalidate_style_iteration,
    prepare_style_iteration,
    prepare_style_test_version,
    propose_style_adjustments,
    score_contextual_evaluation,
)
from companion.style.iteration_models import (
    AdjustmentDecision,
    AdjustmentDecisionPackage,
    ContextualResponsePattern,
    ContextWindowReference,
    LocalContextSummary,
    StyleAdjustmentEvidence,
    StyleIterationManifest,
)
from companion.style.iteration_pipeline import load_active_style_iteration
from companion.style.models import (
    AbsoluteBlindRatings,
    AbsoluteScenarioRating,
    EvaluationPlan,
    EvaluationScenario,
)
from companion.style.private_io import write_private_model


def _semantic_acceptance_package() -> SafeCandidatePackage:
    timestamp = datetime(2026, 7, 20, tzinfo=UTC)
    expressions_by_category = {
        CandidateCategory.CALIBRATED_DAILY: (
            "合成早安呀",
            "合成哈喽呀",
            "合成在吗呀",
            "合成谢谢你",
            "合成感谢啦",
            "合成辛苦啦",
            "合成带伞哦",
            "合成降温啦",
            "合成天气好",
            "合成洗碗啦",
            "合成打扫好",
            "合成收拾呀",
            "合成快递到啦",
            "合成下单啦",
            "合成外卖到啦",
            "合成上课啦",
            "合成下班啦",
            "合成作业好",
        ),
        CandidateCategory.MEALS: ("合成吃饭啦", "合成午餐好", "合成晚餐呀"),
        CandidateCategory.SLEEP: ("合成安安", "合成早点睡", "合成好梦呀"),
        CandidateCategory.GENERAL_EMOTION: ("合成开心呀", "合成别难过", "合成陪着呢"),
        CandidateCategory.GENTLE_AFFECTION: ("合成想你呀", "合成路上小心", "合成抱抱呀"),
    }
    candidates = [
        SafeStyleCandidate(
            text=expression,
            category=category,
            scene_tags=[category.value],
        )
        for category, expressions in expressions_by_category.items()
        for expression in expressions
    ]
    category_counts = {
        category.value: len(expressions)
        for category, expressions in expressions_by_category.items()
    }
    candidate_count = len(candidates)
    audit = CandidateSelectionAudit(
        processed_at=timestamp,
        input_count=candidate_count,
        owner_input_count=candidate_count,
        dropped_other_count=0,
        allowed_category_counts=category_counts,
        rejection_reason_counts={},
        unknown_count=0,
        deduplicated_count=0,
        output_count=candidate_count,
        succeeded=True,
    )
    return SafeCandidatePackage(created_at=timestamp, candidates=candidates, audit=audit)


def _contextual_acceptance_package() -> SafeCandidatePackage:
    timestamp = datetime(2026, 7, 22, tzinfo=UTC)
    return SafeCandidatePackage(
        created_at=timestamp,
        candidates=[
            SafeStyleCandidate(
                text="安安",
                category=CandidateCategory.SLEEP,
                scene_tags=[CandidateCategory.SLEEP.value],
            )
        ],
        audit=CandidateSelectionAudit(
            processed_at=timestamp,
            input_count=2,
            owner_input_count=1,
            dropped_other_count=1,
            allowed_category_counts={CandidateCategory.SLEEP.value: 1},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=1,
            succeeded=True,
        ),
    )


def _contextual_acceptance_scenarios() -> list[EvaluationScenario]:
    categories = (
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
    scenarios = []
    for category in categories:
        for number in (1, 2):
            scenario_id = f"{category.value}-{number}"
            intent = fixed_intents.get(scenario_id, StyleIntent.CATEGORY_DEFAULT)
            prompt = (
                "SYNTHETIC_HOLDOUT_FAREWELL_CONTEXT_DO_NOT_LEARN"
                if scenario_id == "sleep-1"
                else (
                    "SYNTHETIC_HOLDOUT_REMINDER_CONTEXT_DO_NOT_LEARN"
                    if scenario_id == "sleep-2"
                    else f"合成固定场景 {scenario_id}"
                )
            )
            scenarios.append(
                EvaluationScenario(
                    scenario_id=scenario_id,
                    category=category,
                    intent=intent,
                    prompt=prompt,
                )
            )
    return scenarios


def test_contextual_style_calibration_lifecycle_acceptance() -> None:
    package = _contextual_acceptance_package()
    generic_model_response = "祝你一切顺利, 今晚早点休息"
    mandatory_directive = "必须使用以下主人明确确认的表达\uff1a安安"
    package_snapshot = package.model_dump_json()
    scenarios = _contextual_acceptance_scenarios()
    pattern = ContextualResponsePattern(
        pattern_id="synthetic-sleep-farewell-pattern",
        category=CandidateCategory.SLEEP,
        intent=StyleIntent.SLEEP_FAREWELL,
        min_length=2,
        max_length=4,
        tone_tags=["brief"],
        punctuation_tags=["plain"],
        required_expressions=["安安"],
        prohibited_expressions=["祝你一切顺利"],
        aggregate_count=1,
        source="explicit_owner_feedback",
        raw_text_retained=False,
        partner_text_retained=False,
    )
    pattern_snapshot = pattern.model_dump_json()
    pack = build_style_pack(package, extract_style_features(package))
    plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(package),
        scenarios,
        [pattern],
        candidate_package=package,
    )
    farewell = next(item for item in plan.items if item.scenario_id == "sleep-1")
    reminder = next(item for item in plan.items if item.scenario_id == "sleep-2")

    assert mandatory_directive in farewell.styled_prompt
    assert mandatory_directive not in reminder.styled_prompt
    assert farewell.styled_preferences == ["安安"]
    assert farewell.styled_examples == []
    assert reminder.styled_preferences == []
    assert len(package.candidates) == 1
    assert package.audit.dropped_other_count == 1

    styled_labels = {item.scenario_id: "A" for item in plan.items}
    styled_labels["sleep-1"] = "B"
    key = BlindAnswerKey(styled_labels=styled_labels)
    responses = EvaluationResponses(
        items=[
            ScenarioResponses(
                scenario_id=item.scenario_id,
                baseline=f"合成基线回答 {index}",
                styled=(
                    generic_model_response
                    if item.scenario_id == "sleep-1"
                    else f"合成通用回答 {index}"
                ),
            )
            for index, item in enumerate(plan.items)
        ]
    )
    ratings = AbsoluteBlindRatings(
        items=[
            AbsoluteScenarioRating(
                scenario_id=item.scenario_id,
                preferred="B",
                similarity_a=2,
                similarity_b=1,
                neither_like_owner=False,
                context_correct_a=True,
                context_correct_b=True,
                mechanical_a=False,
                mechanical_b=False,
                ideal_response="安安" if item.scenario_id == "sleep-1" else None,
            )
            for item in plan.items
        ]
    )

    initial_report = score_contextual_evaluation(plan, responses, key, ratings)
    assert initial_report.styled_relative_wins == 1
    assert initial_report.styled_absolute_successes == 0
    assert initial_report.conclusion == "insufficient"

    evidence = analyze_contextual_differences(
        plan,
        responses,
        key,
        ratings,
        candidate_package=package,
        prohibited_expressions=["祝你一切顺利"],
    )
    assert [(item.correction_type, item.expression) for item in evidence] == [
        ("required_expression", "安安"),
        ("prohibited_expression", "祝你一切顺利"),
    ]

    sleep_index = next(
        index for index, item in enumerate(responses.items) if item.scenario_id == "sleep-1"
    )
    corrected_responses = responses.model_copy(
        update={
            "items": [
                item.model_copy(update={"styled": "安安"}) if index == sleep_index else item
                for index, item in enumerate(responses.items)
            ]
        }
    )
    corrected_ratings = ratings.model_copy(
        update={
            "items": [
                item.model_copy(update={"similarity_b": 5}) if index == sleep_index else item
                for index, item in enumerate(ratings.items)
            ]
        }
    )
    corrected_report = score_contextual_evaluation(
        plan,
        corrected_responses,
        key,
        corrected_ratings,
    )

    assert corrected_report.styled_absolute_successes == 1
    assert corrected_report.styled_mechanical_scenarios == 0
    assert corrected_report.conclusion == "locally_sufficient"
    serialized_learning_material = package.model_dump_json() + pattern.model_dump_json()
    assert package.model_dump_json() == package_snapshot
    assert pattern.model_dump_json() == pattern_snapshot
    assert generic_model_response not in serialized_learning_material
    assert "SYNTHETIC_HOLDOUT" not in serialized_learning_material


def test_semantic_style_routing_lifecycle_acceptance(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    package = _semantic_acceptance_package()
    package_snapshot = package.model_dump(mode="json", warnings=False)
    private_root = tmp_path / "synthetic-private"
    private_root.mkdir()
    candidate_source = private_root / "synthetic-candidates.json"
    candidate_source.write_text(package.model_dump_json(), encoding="utf-8")
    candidate_snapshot = candidate_source.read_text(encoding="utf-8")

    base_pack = build_style_pack(package, extract_style_features(package))
    proposal = propose_style_adjustments(
        [
            StyleAdjustmentEvidence(
                category=CandidateCategory.SLEEP,
                expression="合成安安",
                trigger_tags=[CandidateCategory.SLEEP.value],
                intent_tags=[StyleIntent.SLEEP_FAREWELL],
                aggregate_count=1,
                source="explicit_owner_feedback",
            )
        ],
        candidate_package=package,
        proposal_id="synthetic-semantic-routing-proposal",
        base_style_version=base_pack.version,
    )
    decisions = AdjustmentDecisionPackage(
        proposal_id=proposal.proposal_id,
        decisions=[AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="approve")],
    )
    base_source = private_root / "synthetic-base.json"
    proposal_source = private_root / "synthetic-proposal.json"
    decisions_source = private_root / "synthetic-decisions.json"
    active_style_source = private_root / "synthetic-active-style.json"
    active_manifest_source = private_root / "synthetic-active-manifest.json"
    style_pack_destination = private_root / "synthetic-style-pack.json"
    evaluation_plan_destination = private_root / "synthetic-evaluation-plan.json"
    for model, source in (
        (base_pack, base_source),
        (proposal, proposal_source),
        (decisions, decisions_source),
    ):
        write_private_model(model, source, private_root)
    prepare_style_iteration(
        base_pack_source=base_source,
        proposal_source=proposal_source,
        decisions_source=decisions_source,
        private_data_dir=private_root,
        iteration_round=1,
        created_at=now,
        candidate_destination=active_style_source,
        manifest_destination=active_manifest_source,
    )

    with pytest.raises(StyleValidationError, match="inactive_style_iteration"):
        prepare_style_test_version(
            candidate_source=candidate_source,
            private_data_dir=private_root,
            expected_count=len(package.candidates),
            scenario_source=Path("fixtures/synthetic/style_evaluation_scenarios.json"),
            style_pack_destination=style_pack_destination,
            evaluation_plan_destination=evaluation_plan_destination,
            active_style_source=active_style_source,
            active_manifest_source=active_manifest_source,
        )
    assert not style_pack_destination.exists()
    assert not evaluation_plan_destination.exists()

    activate_style_iteration(
        manifest_source=active_manifest_source,
        private_data_dir=private_root,
        user_confirmed=True,
        activated_at=now,
    )
    summary = prepare_style_test_version(
        candidate_source=candidate_source,
        private_data_dir=private_root,
        expected_count=len(package.candidates),
        scenario_source=Path("fixtures/synthetic/style_evaluation_scenarios.json"),
        style_pack_destination=style_pack_destination,
        evaluation_plan_destination=evaluation_plan_destination,
        active_style_source=active_style_source,
        active_manifest_source=active_manifest_source,
    )

    assert summary.testable_count == 20
    assert summary.sparse_count == 0
    assert summary.mechanical_risk_count == 0
    assert summary.missing_count == 0
    plan = EvaluationPlan.model_validate_json(
        evaluation_plan_destination.read_text(encoding="utf-8")
    )
    preference_scenarios = {
        item.scenario_id for item in plan.items if "合成安安" in item.styled_preferences
    }
    assert preference_scenarios == {"sleep-1"}
    assert package.model_dump(mode="json", warnings=False) == package_snapshot
    assert candidate_source.read_text(encoding="utf-8") == candidate_snapshot


def test_offline_style_iteration_requires_approval_and_supports_invalidation(
    tmp_path: Path,
    synthetic_candidate_package: SafeCandidatePackage,
) -> None:
    now = datetime(2026, 7, 19, tzinfo=UTC)
    references = [
        ContextWindowReference(
            window_token=hashlib.sha256(f"window-{index}".encode()).hexdigest(),
            group_token=hashlib.sha256(f"group-{index}".encode()).hexdigest(),
        )
        for index in range(10)
    ]
    holdout = build_context_holdout(references, dataset_fingerprint="a" * 64)
    holdout_reference = next(
        item for item in references if item.window_token in holdout.holdout_window_tokens
    )
    safe_summary = "synthetic-safe-summary-unique-sentinel"
    summary_result = build_local_context_summary(
        reference=holdout_reference,
        category=CandidateCategory.SLEEP,
        safe_summary=safe_summary,
        holdout=holdout,
        privacy_approved=True,
        speaker_mapping_confirmed=True,
        contains_sensitive_context=False,
        contains_identity_placeholder=False,
    )
    assert summary_result.summary is not None
    assert summary_result.summary.partner_text_retained is False
    assert "partner_text" not in LocalContextSummary.model_fields
    base_pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    plan = build_holdout_evaluation_plan(
        base_pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [summary_result.summary],
        holdout=holdout,
    )
    assert len(plan.items) == 1
    assert safe_summary in plan.items[0].baseline_prompt

    proposal = propose_style_adjustments(
        [
            StyleAdjustmentEvidence(
                category=CandidateCategory.SLEEP,
                expression="合成晚安呀",
                trigger_tags=["sleep"],
                intent_tags=[StyleIntent.SLEEP_FAREWELL],
                aggregate_count=2,
                source="explicit_owner_feedback",
            )
        ],
        candidate_package=synthetic_candidate_package,
        proposal_id="synthetic-proposal",
        base_style_version=base_pack.version,
    )
    decisions = AdjustmentDecisionPackage(
        proposal_id=proposal.proposal_id,
        decisions=[AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="approve")],
    )
    private_root = tmp_path / "private"
    private_root.mkdir()
    base_source = private_root / "base.json"
    proposal_source = private_root / "proposal.json"
    decisions_source = private_root / "decisions.json"
    candidate_source = private_root / "candidate.json"
    manifest_source = private_root / "manifest.json"
    write_private_model(base_pack, base_source, private_root)
    write_private_model(proposal, proposal_source, private_root)
    write_private_model(decisions, decisions_source, private_root)
    prepare_style_iteration(
        base_pack_source=base_source,
        proposal_source=proposal_source,
        decisions_source=decisions_source,
        private_data_dir=private_root,
        iteration_round=1,
        created_at=now,
        candidate_destination=candidate_source,
        manifest_destination=manifest_source,
    )
    with pytest.raises(StyleValidationError, match="inactive_style_iteration"):
        load_active_style_iteration(
            manifest_source=manifest_source,
            candidate_source=candidate_source,
            private_data_dir=private_root,
            dataset_fingerprint=proposal.dataset_fingerprint,
        )
    activate_style_iteration(
        manifest_source=manifest_source,
        private_data_dir=private_root,
        user_confirmed=True,
        activated_at=now,
    )
    assert load_active_style_iteration(
        manifest_source=manifest_source,
        candidate_source=candidate_source,
        private_data_dir=private_root,
        dataset_fingerprint=proposal.dataset_fingerprint,
    ).preferences
    invalidate_style_iteration(
        manifest_source=manifest_source,
        private_data_dir=private_root,
        invalidated_at=now + timedelta(seconds=1),
    )
    with pytest.raises(StyleValidationError, match="inactive_style_iteration"):
        load_active_style_iteration(
            manifest_source=manifest_source,
            candidate_source=candidate_source,
            private_data_dir=private_root,
            dataset_fingerprint=proposal.dataset_fingerprint,
        )
    serialized = candidate_source.read_text(encoding="utf-8") + manifest_source.read_text(
        encoding="utf-8"
    )
    assert safe_summary not in serialized
    assert str(private_root) not in serialized
    assert holdout_reference.window_token not in serialized
    assert {"partner_text", "raw_text"}.isdisjoint(StylePack.model_fields)
    assert {"partner_text", "raw_text"}.isdisjoint(StyleIterationManifest.model_fields)
