# ruff: noqa: RUF001

from datetime import UTC, datetime
from pathlib import Path

import pytest

from companion.contracts import StyleIntent
from companion.contracts.style import StylePreference
from companion.privacy import (
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
    StyleExampleIndex,
    StyleValidationError,
    build_contextual_evaluation_plan,
    build_style_pack,
    extract_style_features,
    prepare_style_test_version,
)

SCENARIOS = Path("fixtures/synthetic/style_evaluation_scenarios.json")
TRUE_NAMES = ["合成真实姓名"]
NICKNAMES = ["合成爱称"]


def _calibration(*items: tuple[CandidateCategory, StyleIntent, str]) -> CalibrationCandidatePackage:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    candidates = [
        CalibrationCandidate(
            candidate_id=calibration_candidate_id(
                category=category, intent=intent.value, text=text
            ),
            category=category,
            intent=intent.value,
            text=text,
        )
        for category, intent, text in items
    ]
    return CalibrationCandidatePackage(
        created_at=now,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=calibration_identity_config_fingerprint(
            TRUE_NAMES,
            NICKNAMES,
        ),
        candidate_count=len(candidates),
        candidates=candidates,
        audit=CalibrationCandidateAudit(
            reviewed_at=now,
            submission_count=len(candidates),
            approved_count=len(candidates),
            rejected_count=0,
            privacy_rejected_count=0,
            output_count=len(candidates),
        ),
    )


def test_reviewed_examples_apply_only_to_exact_category_and_intent(
    synthetic_candidate_package,
) -> None:
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package, extract_style_features(synthetic_candidate_package)
    )
    long_example = "合成" * 20
    reviewed = ReviewedCalibrationIndex(
        _calibration(
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, long_example),
            (CandidateCategory.MEALS, StyleIntent.CATEGORY_DEFAULT, "合成吃饭回复"),
        )
    )
    scenarios = [
        EvaluationScenario(
            scenario_id="sleep-1",
            category=CandidateCategory.SLEEP,
            intent=StyleIntent.SLEEP_FAREWELL,
            prompt="合成睡眠场景",
        ),
        EvaluationScenario(
            scenario_id="sleep-2",
            category=CandidateCategory.SLEEP,
            intent=StyleIntent.SLEEP_REMINDER,
            prompt="合成提醒场景",
        ),
    ]

    plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        scenarios,
        [],
        synthetic_candidate_package,
        reviewed_calibration=reviewed,
    )

    assert long_example in plan.items[0].styled_examples
    assert long_example in plan.items[0].styled_prompt
    assert long_example not in plan.items[0].styled_preferences
    assert long_example not in plan.items[1].styled_examples


def test_reviewed_short_expression_soft_lock_is_exact_and_lexically_deterministic(
    synthetic_candidate_package,
) -> None:
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package, extract_style_features(synthetic_candidate_package)
    )
    reviewed = ReviewedCalibrationIndex(
        _calibration(
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, "synthetic-short-b"),
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, "synthetic-short-a"),
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER, "synthetic-reminder"),
        )
    )
    scenarios = [
        EvaluationScenario(
            scenario_id="sleep-farewell",
            category=CandidateCategory.SLEEP,
            intent=StyleIntent.SLEEP_FAREWELL,
            prompt="synthetic farewell scenario",
        ),
        EvaluationScenario(
            scenario_id="sleep-reminder",
            category=CandidateCategory.SLEEP,
            intent=StyleIntent.SLEEP_REMINDER,
            prompt="synthetic reminder scenario",
        ),
    ]

    assert (
        reviewed.retrieve_short_expression(CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL)
        == "synthetic-short-a"
    )
    assert (
        reviewed.retrieve_short_expression(CandidateCategory.MEALS, StyleIntent.SLEEP_FAREWELL)
        is None
    )

    plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        scenarios,
        [],
        synthetic_candidate_package,
        reviewed_calibration=reviewed,
    )

    exact_item, different_intent_item = plan.items
    assert "主人确认优先表达" in exact_item.styled_prompt
    assert "synthetic-short-a" in exact_item.styled_prompt
    assert "synthetic-short-a" not in exact_item.styled_preferences
    assert "synthetic-short-a" not in different_intent_item.styled_prompt


def test_category_default_short_calibration_preserves_existing_examples(
    synthetic_candidate_package,
) -> None:
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    default_expression = "synthetic-default-expression"
    index = ReviewedCalibrationIndex(
        _calibration(
            (
                CandidateCategory.COURTESY,
                StyleIntent.CATEGORY_DEFAULT,
                default_expression,
            )
        )
    )
    scenarios = [
        EvaluationScenario(
            scenario_id="courtesy-default",
            category=CandidateCategory.COURTESY,
            intent=StyleIntent.CATEGORY_DEFAULT,
            prompt="synthetic courtesy scene",
        )
    ]

    unreviewed_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        scenarios,
        [],
        synthetic_candidate_package,
    )
    reviewed_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        scenarios,
        [],
        synthetic_candidate_package,
        reviewed_calibration=index,
    )

    assert (
        index.retrieve_short_expression(
            CandidateCategory.COURTESY,
            StyleIntent.CATEGORY_DEFAULT,
        )
        is None
    )
    assert reviewed_plan.items[0].styled_examples == unreviewed_plan.items[0].styled_examples
    assert reviewed_plan.items[0].styled_prompt == unreviewed_plan.items[0].styled_prompt
    assert default_expression not in reviewed_plan.items[0].styled_examples
    assert default_expression not in reviewed_plan.items[0].styled_prompt


def test_repeated_specific_short_calibration_preserves_existing_examples(
    synthetic_candidate_package,
) -> None:
    from companion.style.iteration_models import ContextualResponsePattern
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    specific_expression = "synthetic-specific-expression"
    index = ReviewedCalibrationIndex(
        _calibration(
            (
                CandidateCategory.SLEEP,
                StyleIntent.SLEEP_FAREWELL,
                specific_expression,
            )
        )
    )
    scenarios = [
        EvaluationScenario(
            scenario_id=f"synthetic-farewell-{number}",
            category=CandidateCategory.SLEEP,
            intent=StyleIntent.SLEEP_FAREWELL,
            prompt=f"synthetic farewell scene {number}",
        )
        for number in (1, 2)
    ]
    patterns = [
        ContextualResponsePattern(
            pattern_id="synthetic-existing-preference",
            category=CandidateCategory.SLEEP,
            intent=StyleIntent.SLEEP_FAREWELL,
            min_length=1,
            max_length=32,
            tone_tags=["brief"],
            punctuation_tags=["plain"],
            required_expressions=["合成晚安呀"],
            prohibited_expressions=[],
            aggregate_count=1,
            source="explicit_owner_feedback",
            raw_text_retained=False,
            partner_text_retained=False,
        )
    ]

    unreviewed_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        scenarios,
        patterns,
        synthetic_candidate_package,
    )
    reviewed_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        scenarios,
        patterns,
        synthetic_candidate_package,
        reviewed_calibration=index,
    )

    assert [item.styled_examples for item in reviewed_plan.items] == [
        item.styled_examples for item in unreviewed_plan.items
    ]
    assert [item.styled_prompt for item in reviewed_plan.items] == [
        item.styled_prompt for item in unreviewed_plan.items
    ]
    assert all(specific_expression not in item.styled_examples for item in reviewed_plan.items)
    assert all(specific_expression not in item.styled_prompt for item in reviewed_plan.items)


def test_reviewed_calibration_replaces_only_the_generated_example_block(
    synthetic_candidate_package,
) -> None:
    from companion.style.calibration import apply_reviewed_calibration
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    reviewed_expression = "synthetic-reviewed-farewell"
    seed_scenario = EvaluationScenario(
        scenario_id="synthetic-farewell-seed",
        category=CandidateCategory.SLEEP,
        intent=StyleIntent.SLEEP_FAREWELL,
        prompt="synthetic seed scenario",
    )
    seed_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [seed_scenario],
        [],
        synthetic_candidate_package,
    )
    original_item = seed_plan.items[0]
    original_example_block = "同类脱敏示例：\n" + "\n".join(
        f"- {example}" for example in original_item.styled_examples
    )
    style_section = original_item.styled_prompt.removeprefix(f"{original_item.baseline_prompt}\n")
    collision_baseline = f"{original_item.baseline_prompt}\n{original_example_block}"
    collision_item = original_item.model_copy(
        update={
            "baseline_prompt": collision_baseline,
            "styled_prompt": f"{collision_baseline}\n{style_section}",
        }
    )
    collision_plan = seed_plan.model_copy(update={"items": [collision_item]})
    index = ReviewedCalibrationIndex(
        _calibration(
            (
                CandidateCategory.SLEEP,
                StyleIntent.SLEEP_FAREWELL,
                reviewed_expression,
            )
        )
    )

    reviewed_plan = apply_reviewed_calibration(collision_plan, index)

    item = reviewed_plan.items[0]
    baseline_prefix = f"{collision_baseline}\n"
    assert item.styled_prompt.startswith(baseline_prefix)
    style_section = item.styled_prompt.removeprefix(baseline_prefix)
    assert original_example_block not in style_section
    assert f"同类脱敏示例：\n- {reviewed_expression}" in style_section


def test_reviewed_calibration_does_not_replace_a_preference_with_example_markup(
    synthetic_candidate_package,
) -> None:
    from companion.style.models import EvaluationScenario

    preference_expression = "同类脱敏示例：\n- 无同类示例"
    pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    ).model_copy(
        update={
            "preferences": [
                StylePreference(
                    rule_id="synthetic-colliding-preference",
                    category=CandidateCategory.COURTESY.value,
                    expression=preference_expression,
                    trigger_tags=[CandidateCategory.COURTESY.value],
                    intent_tags=[StyleIntent.COURTESY_THANKS],
                    max_consecutive_uses=1,
                )
            ]
        }
    )
    reviewed_expression = "synthetic-reviewed-thanks"
    scenario = EvaluationScenario(
        scenario_id="synthetic-courtesy-collision",
        category=CandidateCategory.COURTESY,
        intent=StyleIntent.COURTESY_THANKS,
        prompt="synthetic courtesy scenario",
    )
    index = ReviewedCalibrationIndex(
        _calibration(
            (
                CandidateCategory.COURTESY,
                StyleIntent.COURTESY_THANKS,
                reviewed_expression,
            )
        )
    )

    reviewed_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [scenario],
        [],
        synthetic_candidate_package,
        reviewed_calibration=index,
    )

    item = reviewed_plan.items[0]
    assert item.styled_preferences == [preference_expression]
    assert f"- {preference_expression}；仅在语境自然时优先；连续使用上限 1" in item.styled_prompt
    assert f"同类脱敏示例：\n- {reviewed_expression}" in item.styled_prompt


@pytest.mark.parametrize(
    ("scenario_id", "category", "intent"),
    [
        (
            "greeting-2",
            CandidateCategory.GREETING,
            StyleIntent.CONVERSATION_REOPEN,
        ),
        ("courtesy-1", CandidateCategory.COURTESY, StyleIntent.COURTESY_THANKS),
        ("courtesy-2", CandidateCategory.COURTESY, StyleIntent.COURTESY_APOLOGY),
        (
            "weather-1",
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
        ),
        (
            "work_study-1",
            CandidateCategory.WORK_STUDY,
            StyleIntent.WORK_STUDY_COMPLETED,
        ),
    ],
)
def test_precise_reviewed_intent_soft_locks_only_its_unique_scene(
    synthetic_candidate_package,
    scenario_id: str,
    category: CandidateCategory,
    intent: StyleIntent,
) -> None:
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    expression = f"synthetic-{scenario_id}"
    index = ReviewedCalibrationIndex(_calibration((category, intent, expression)))
    scenarios = [
        EvaluationScenario(
            scenario_id=scenario_id,
            category=category,
            intent=intent,
            prompt=f"synthetic exact scene {scenario_id}",
        ),
        EvaluationScenario(
            scenario_id=f"{scenario_id}-other",
            category=category,
            intent=StyleIntent.CATEGORY_DEFAULT,
            prompt=f"synthetic other scene {scenario_id}",
        ),
    ]

    plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        scenarios,
        [],
        synthetic_candidate_package,
        reviewed_calibration=index,
    )

    exact_item, *other_items = plan.items
    assert "主人确认优先表达" in exact_item.styled_prompt
    assert expression in exact_item.styled_prompt
    assert all(expression not in item.styled_prompt for item in other_items)


def test_reviewed_33_character_expression_remains_only_a_styled_example(
    synthetic_candidate_package,
) -> None:
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package, extract_style_features(synthetic_candidate_package)
    )
    long_example = "x" * 33
    reviewed = ReviewedCalibrationIndex(
        _calibration(
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, "synthetic-short-a"),
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, long_example),
        )
    )
    scenario = EvaluationScenario(
        scenario_id="sleep-farewell",
        category=CandidateCategory.SLEEP,
        intent=StyleIntent.SLEEP_FAREWELL,
        prompt="synthetic farewell scenario",
    )

    plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [scenario],
        [],
        synthetic_candidate_package,
        reviewed_calibration=reviewed,
    )

    item = plan.items[0]
    priority_block = item.styled_prompt.split("主人确认优先表达", maxsplit=1)[1]
    assert long_example in item.styled_examples
    assert long_example not in priority_block
    assert long_example not in item.styled_preferences


def test_short_soft_lock_does_not_displace_long_example_with_existing_preference(
    synthetic_candidate_package,
) -> None:
    from companion.style.iteration_models import ContextualResponsePattern
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package, extract_style_features(synthetic_candidate_package)
    )
    short_expression = "a-short-expression"
    long_example = "z" * 33
    reviewed = ReviewedCalibrationIndex(
        _calibration(
            (
                CandidateCategory.SLEEP,
                StyleIntent.SLEEP_FAREWELL,
                short_expression,
            ),
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, long_example),
        )
    )
    scenario = EvaluationScenario(
        scenario_id="sleep-farewell",
        category=CandidateCategory.SLEEP,
        intent=StyleIntent.SLEEP_FAREWELL,
        prompt="synthetic farewell scenario",
    )
    patterns = [
        ContextualResponsePattern(
            pattern_id="synthetic-existing-preference",
            category=CandidateCategory.SLEEP,
            intent=StyleIntent.SLEEP_FAREWELL,
            min_length=1,
            max_length=32,
            tone_tags=["brief"],
            punctuation_tags=["plain"],
            required_expressions=["合成晚安呀"],
            prohibited_expressions=[],
            aggregate_count=1,
            source="explicit_owner_feedback",
            raw_text_retained=False,
            partner_text_retained=False,
        )
    ]

    plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [scenario],
        patterns,
        synthetic_candidate_package,
        reviewed_calibration=reviewed,
    )

    item = plan.items[0]
    priority_block = item.styled_prompt.split("主人确认优先表达", maxsplit=1)[1]
    assert item.styled_preferences == ["合成晚安呀"]
    assert item.styled_examples == [long_example]
    assert short_expression in priority_block
    assert short_expression not in item.styled_examples


def test_unique_reviewed_examples_use_deterministic_long_first_capacity_order(
    synthetic_candidate_package,
) -> None:
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package, extract_style_features(synthetic_candidate_package)
    )
    short_expression = "a-short-expression"
    long_example = "z" * 33
    reviewed = ReviewedCalibrationIndex(
        _calibration(
            (
                CandidateCategory.SLEEP,
                StyleIntent.SLEEP_FAREWELL,
                short_expression,
            ),
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, long_example),
        )
    )
    scenario = EvaluationScenario(
        scenario_id="sleep-farewell",
        category=CandidateCategory.SLEEP,
        intent=StyleIntent.SLEEP_FAREWELL,
        prompt="synthetic farewell scenario",
    )

    first_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [scenario],
        [],
        synthetic_candidate_package,
        reviewed_calibration=reviewed,
    )
    second_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [scenario],
        [],
        synthetic_candidate_package,
        reviewed_calibration=reviewed,
    )

    assert first_plan.items[0].styled_examples == [long_example, short_expression]
    assert second_plan.items[0].styled_examples == first_plan.items[0].styled_examples


def test_reviewed_32_character_expression_soft_locks_without_changing_preferences(
    synthetic_candidate_package,
) -> None:
    from companion.style.iteration_models import ContextualResponsePattern
    from companion.style.models import EvaluationScenario

    pack = build_style_pack(
        synthetic_candidate_package, extract_style_features(synthetic_candidate_package)
    )
    expression = "q" * 32
    reviewed = ReviewedCalibrationIndex(
        _calibration(
            (CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, expression),
        )
    )
    scenario = EvaluationScenario(
        scenario_id="sleep-farewell",
        category=CandidateCategory.SLEEP,
        intent=StyleIntent.SLEEP_FAREWELL,
        prompt="synthetic farewell scenario",
    )
    patterns = [
        ContextualResponsePattern(
            pattern_id="synthetic-existing-preference",
            category=CandidateCategory.SLEEP,
            intent=StyleIntent.SLEEP_FAREWELL,
            min_length=1,
            max_length=32,
            tone_tags=["brief"],
            punctuation_tags=["plain"],
            required_expressions=["合成晚安呀"],
            prohibited_expressions=[],
            aggregate_count=1,
            source="explicit_owner_feedback",
            raw_text_retained=False,
            partner_text_retained=False,
        )
    ]

    unreviewed_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [scenario],
        patterns,
        synthetic_candidate_package,
    )
    reviewed_plan = build_contextual_evaluation_plan(
        pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [scenario],
        patterns,
        synthetic_candidate_package,
        reviewed_calibration=reviewed,
    )

    assert (
        reviewed.retrieve_short_expression(CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL)
        == expression
    )
    assert expression in reviewed_plan.items[0].styled_prompt
    assert unreviewed_plan.items[0].styled_preferences == ["合成晚安呀"]
    assert [item.styled_preferences for item in reviewed_plan.items] == [
        item.styled_preferences for item in unreviewed_plan.items
    ]


def test_pipeline_binds_fingerprint_and_preserves_outputs_on_failed_refresh(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    base_source = root / "safe.json"
    base_source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    reviewed = _calibration((CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, "合成安安"))
    calibration_source = publish_calibration_candidate_package(
        reviewed,
        private_data_dir=root,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
    )
    pack = root / "pack.json"
    plan = root / "plan.json"
    args = dict(
        candidate_source=base_source,
        private_data_dir=root,
        expected_count=3,
        scenario_source=SCENARIOS,
        style_pack_destination=pack,
        evaluation_plan_destination=plan,
        calibration_source=calibration_source,
        expected_calibration_fingerprint=reviewed.content_fingerprint,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
    )

    prepare_style_test_version(**args)
    before = (pack.read_bytes(), plan.read_bytes())
    assert reviewed.content_fingerprint.encode() in before[1]

    with pytest.raises(StyleValidationError, match="stale_calibration_package"):
        prepare_style_test_version(**(args | {"expected_calibration_fingerprint": "0" * 64}))

    assert (pack.read_bytes(), plan.read_bytes()) == before


def test_pipeline_rebuilds_distinct_plan_bytes_for_distinct_formal_calibrations(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    base_source = root / "safe.json"
    base_source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    formal_text = "synthetic-formal-courtesy"
    first_reviewed = _calibration(
        (CandidateCategory.COURTESY, StyleIntent.CATEGORY_DEFAULT, formal_text),
    )
    second_reviewed = _calibration(
        (CandidateCategory.COURTESY, StyleIntent.COURTESY_THANKS, formal_text),
    )
    calibration_source = publish_calibration_candidate_package(
        first_reviewed,
        private_data_dir=root,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
    )
    pack = root / "pack.json"
    plan = root / "plan.json"
    args = dict(
        candidate_source=base_source,
        private_data_dir=root,
        expected_count=3,
        scenario_source=SCENARIOS,
        style_pack_destination=pack,
        evaluation_plan_destination=plan,
        calibration_source=calibration_source,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
    )

    prepare_style_test_version(
        **(args | {"expected_calibration_fingerprint": first_reviewed.content_fingerprint})
    )
    first_artifacts = (pack.read_bytes(), plan.read_bytes())

    publish_calibration_candidate_package(
        second_reviewed,
        private_data_dir=root,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
    )
    prepare_style_test_version(
        **(args | {"expected_calibration_fingerprint": second_reviewed.content_fingerprint})
    )
    second_artifacts = (pack.read_bytes(), plan.read_bytes())

    assert first_reviewed.content_fingerprint != second_reviewed.content_fingerprint
    assert first_artifacts != second_artifacts
    assert first_artifacts[0] != second_artifacts[0]
    assert first_artifacts[1] != second_artifacts[1]


def test_pipeline_never_overwrites_the_formal_calibration_input(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    base_source = root / "safe.json"
    base_source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    reviewed = _calibration((CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL, "合成安安"))
    calibration_source = publish_calibration_candidate_package(
        reviewed,
        private_data_dir=root,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
    )
    before = calibration_source.read_bytes()

    with pytest.raises(StyleValidationError, match="style_path_collision"):
        prepare_style_test_version(
            candidate_source=base_source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIOS,
            style_pack_destination=calibration_source,
            evaluation_plan_destination=root / "plan.json",
            calibration_source=calibration_source,
            expected_calibration_fingerprint=reviewed.content_fingerprint,
            true_name_terms=TRUE_NAMES,
            approved_nicknames=NICKNAMES,
        )

    assert calibration_source.read_bytes() == before
