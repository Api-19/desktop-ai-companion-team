import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import companion.style.iteration_pipeline as iteration_pipeline
import companion.style.pipeline as style_pipeline
import companion.style.private_io as style_private_io
from companion.contracts import StyleIntent, StylePack
from companion.privacy import CandidateCategory
from companion.style import (
    StyleValidationError,
    build_style_pack,
    extract_style_features,
    finalize_style_test_version,
    prepare_style_test_version,
)
from companion.style.adjustments import propose_style_adjustments
from companion.style.iteration_models import (
    AdjustmentDecision,
    AdjustmentDecisionPackage,
    StyleAdjustmentEvidence,
    StyleIterationManifest,
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
NOW = datetime(2026, 7, 20, tzinfo=UTC)


def _write_iteration_sources(
    root: Path,
    synthetic_candidate_package,
    *,
    status: str,
) -> tuple[Path, Path]:
    base_pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    proposal = propose_style_adjustments(
        [
            StyleAdjustmentEvidence(
                category=CandidateCategory.SLEEP,
                expression="合成晚安呀",
                trigger_tags=["sleep"],
                intent_tags=[StyleIntent.SLEEP_FAREWELL],
                aggregate_count=1,
                source="explicit_owner_feedback",
            )
        ],
        candidate_package=synthetic_candidate_package,
        proposal_id="synthetic-active-preference",
        base_style_version=base_pack.version,
    )
    decisions = AdjustmentDecisionPackage(
        proposal_id=proposal.proposal_id,
        decisions=[AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="approve")],
    )
    base_source = root / "iteration-base.json"
    proposal_source = root / "iteration-proposal.json"
    decisions_source = root / "iteration-decisions.json"
    candidate_source = root / "iteration-candidate.json"
    manifest_source = root / "iteration-manifest.json"
    for model, path in (
        (base_pack, base_source),
        (proposal, proposal_source),
        (decisions, decisions_source),
    ):
        style_private_io.write_private_model(model, path, root)
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base_source,
        proposal_source=proposal_source,
        decisions_source=decisions_source,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=candidate_source,
        manifest_destination=manifest_source,
    )
    if status in {"active", "invalidated"}:
        iteration_pipeline.activate_style_iteration(
            manifest_source=manifest_source,
            private_data_dir=root,
            user_confirmed=True,
            activated_at=NOW,
        )
    if status == "invalidated":
        iteration_pipeline.invalidate_style_iteration(
            manifest_source=manifest_source,
            private_data_dir=root,
            invalidated_at=NOW + timedelta(seconds=1),
        )
    return candidate_source, manifest_source


@pytest.mark.parametrize("provided", ["candidate", "manifest"])
def test_preparation_requires_active_sources_together_before_writing(
    tmp_path: Path,
    synthetic_candidate_package,
    provided: str,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "safe-candidates.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    pack = root / "style-pack.json"
    plan = root / "evaluation-plan.json"

    with pytest.raises(StyleValidationError, match="incomplete_active_style_sources"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
            active_style_source=(root / "active-style.json" if provided == "candidate" else None),
            active_manifest_source=(
                root / "active-manifest.json" if provided == "manifest" else None
            ),
        )

    assert not pack.exists()
    assert not plan.exists()


@pytest.mark.parametrize("status", ["candidate", "invalidated"])
def test_preparation_rejects_inactive_iteration_before_writing(
    tmp_path: Path,
    synthetic_candidate_package,
    status: str,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "safe-candidates.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    active_style, active_manifest = _write_iteration_sources(
        root,
        synthetic_candidate_package,
        status=status,
    )
    pack = root / "style-pack.json"
    plan = root / "evaluation-plan.json"

    with pytest.raises(StyleValidationError, match="inactive_style_iteration"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
            active_style_source=active_style,
            active_manifest_source=active_manifest,
        )

    assert not pack.exists()
    assert not plan.exists()


def test_preparation_rejects_active_iteration_dataset_fingerprint_mismatch(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "safe-candidates.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    active_style, active_manifest = _write_iteration_sources(
        root,
        synthetic_candidate_package,
        status="active",
    )
    manifest = style_private_io.read_private_model(
        StyleIterationManifest,
        active_manifest,
        root,
    )
    mismatched = StyleIterationManifest.model_validate(
        manifest.model_dump(mode="json") | {"dataset_fingerprint": "f" * 64}
    )
    style_private_io.write_private_model(mismatched, active_manifest, root)
    pack = root / "style-pack.json"
    plan = root / "evaluation-plan.json"

    with pytest.raises(StyleValidationError, match="inactive_style_iteration"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
            active_style_source=active_style,
            active_manifest_source=active_manifest,
        )

    assert not pack.exists()
    assert not plan.exists()


def test_preparation_publishes_valid_active_pack_and_intent_preference(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "safe-candidates.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    active_style, active_manifest = _write_iteration_sources(
        root,
        synthetic_candidate_package,
        status="active",
    )
    pack_destination = root / "style-pack.json"
    plan_destination = root / "evaluation-plan.json"

    prepare_style_test_version(
        candidate_source=source,
        private_data_dir=root,
        expected_count=3,
        scenario_source=SCENARIO_SOURCE,
        style_pack_destination=pack_destination,
        evaluation_plan_destination=plan_destination,
        active_style_source=active_style,
        active_manifest_source=active_manifest,
    )

    selected = style_private_io.read_private_model(StylePack, pack_destination, root)
    active = style_private_io.read_private_model(StylePack, active_style, root)
    plan = style_private_io.read_private_model(EvaluationPlan, plan_destination, root)
    farewell = next(item for item in plan.items if item.scenario_id == "sleep-1")
    reminder = next(item for item in plan.items if item.scenario_id == "sleep-2")
    assert selected == active
    assert farewell.styled_preferences == ["合成晚安呀"]
    assert len(farewell.styled_preferences) + len(farewell.styled_examples) <= 2
    assert reminder.styled_preferences == []


def test_preparation_builds_base_index_before_loading_active_iteration(
    tmp_path: Path,
    synthetic_candidate_package,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "safe-candidates.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    calls: list[str] = []
    base_pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )

    class FailingStyleExampleIndex:
        @classmethod
        def from_package(cls, package):
            calls.append("index")
            raise StyleValidationError("synthetic_index_failure", "synthetic index failure")

    def load_active(**_kwargs) -> StylePack:
        calls.append("active")
        return base_pack

    monkeypatch.setattr(style_pipeline, "StyleExampleIndex", FailingStyleExampleIndex)
    monkeypatch.setattr(style_pipeline, "load_active_style_iteration", load_active)

    with pytest.raises(StyleValidationError, match="synthetic_index_failure"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=root / "style-pack.json",
            evaluation_plan_destination=root / "evaluation-plan.json",
            active_style_source=root / "active-style.json",
            active_manifest_source=root / "active-manifest.json",
        )

    assert calls == ["index"]


@pytest.mark.parametrize("outside_source", ["style", "manifest"])
def test_preparation_rejects_each_active_source_outside_private_data(
    tmp_path: Path,
    synthetic_candidate_package,
    outside_source: str,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "safe-candidates.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    pack = root / "style-pack.json"
    plan = root / "evaluation-plan.json"
    outside = tmp_path / f"synthetic-outside-{outside_source}.json"

    with pytest.raises(StyleValidationError, match="unsafe_style_path"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
            active_style_source=(
                outside if outside_source == "style" else root / "active-style.json"
            ),
            active_manifest_source=(
                outside if outside_source == "manifest" else root / "active-manifest.json"
            ),
        )

    assert not pack.exists()
    assert not plan.exists()


@pytest.mark.parametrize(
    ("active_source", "collision_target"),
    [
        ("style", "manifest"),
        ("style", "candidate"),
        ("style", "scenario"),
        ("style", "pack"),
        ("style", "plan"),
        ("manifest", "candidate"),
        ("manifest", "scenario"),
        ("manifest", "pack"),
        ("manifest", "plan"),
    ],
)
def test_preparation_rejects_all_active_source_collisions_before_writing(
    tmp_path: Path,
    synthetic_candidate_package,
    active_source: str,
    collision_target: str,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "safe-candidates.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    scenario = root / "scenarios.json"
    scenario.write_text(SCENARIO_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    pack = root / "style-pack.json"
    plan = root / "evaluation-plan.json"
    paths = {
        "candidate": source,
        "scenario": scenario,
        "pack": pack,
        "plan": plan,
    }
    active_style = root / "active-style.json"
    active_manifest = root / "active-manifest.json"
    if collision_target == "manifest":
        active_style = active_manifest
    elif active_source == "style":
        active_style = paths[collision_target]
    else:
        active_manifest = paths[collision_target]

    with pytest.raises(StyleValidationError, match="style_path_collision"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=scenario,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
            active_style_source=active_style,
            active_manifest_source=active_manifest,
        )

    assert source.read_text(encoding="utf-8") == synthetic_candidate_package.model_dump_json()
    assert scenario.read_text(encoding="utf-8") == SCENARIO_SOURCE.read_text(encoding="utf-8")
    assert not pack.exists()
    assert not plan.exists()


def test_preparation_failure_traceback_scrubs_both_active_source_paths(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "safe-candidates.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    style_sentinel = "SYNTHETIC_ACTIVE_STYLE_PATH_91D4"
    manifest_sentinel = "SYNTHETIC_ACTIVE_MANIFEST_PATH_7F31"
    pack = root / "style-pack.json"
    plan = root / "evaluation-plan.json"

    with pytest.raises(StyleValidationError) as caught:
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
            active_style_source=root / f"{style_sentinel}.json",
            active_manifest_source=root / f"{manifest_sentinel}.json",
        )

    assert style_sentinel not in str(caught.value)
    assert manifest_sentinel not in str(caught.value)
    traceback = caught.value.__traceback__
    checked_product_frame = False
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/companion/style/" in filename:
            checked_product_frame = True
            locals_text = repr(traceback.tb_frame.f_locals)
            assert style_sentinel not in locals_text
            assert manifest_sentinel not in locals_text
            assert str(root) not in locals_text
        traceback = traceback.tb_next
    assert checked_product_frame
    assert not pack.exists()
    assert not plan.exists()


def test_prepares_private_pack_and_plan(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "candidate.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")

    summary = prepare_style_test_version(
        candidate_source=source,
        private_data_dir=root,
        expected_count=3,
        scenario_source=SCENARIO_SOURCE,
        style_pack_destination=root / "style-pack.json",
        evaluation_plan_destination=root / "evaluation-plan.json",
    )

    assert summary.candidate_count == 3
    assert summary.scenario_count == 20
    assert summary.testable_count == 0
    assert summary.sparse_count == 2
    assert summary.mechanical_risk_count == 2
    assert summary.missing_count == 16
    assert (root / "style-pack.json").exists()
    assert (root / "evaluation-plan.json").exists()
    assert set(summary.model_dump()) == {
        "candidate_count",
        "scenario_count",
        "testable_count",
        "sparse_count",
        "mechanical_risk_count",
        "missing_count",
    }


def test_preparation_rejects_incompatible_candidate_schema_before_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "candidate.json"
    source.write_text(
        json.dumps({"candidate_count": 82, "results": ["synthetic"]}),
        encoding="utf-8",
    )
    pack = root / "style-pack.json"
    plan = root / "evaluation-plan.json"

    with pytest.raises(StyleValidationError, match="invalid_private_model"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=78,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
        )

    assert not pack.exists()
    assert not plan.exists()


def test_preparation_rejects_unsafe_output_before_any_formal_write(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "candidate.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    plan = root / "evaluation-plan.json"

    with pytest.raises(StyleValidationError, match="unsafe_style_path"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=tmp_path / "outside.json",
            evaluation_plan_destination=plan,
        )

    assert not plan.exists()


def test_preparation_prevalidates_both_destinations_before_writing(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "candidate.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    pack = root / "style-pack.json"

    with pytest.raises(StyleValidationError, match="unsafe_style_path"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=tmp_path / "outside-plan.json",
        )

    assert not pack.exists()


@pytest.mark.parametrize("collision", ["candidate", "same_outputs"])
def test_preparation_rejects_private_path_collisions_before_writing(
    tmp_path: Path,
    synthetic_candidate_package,
    collision: str,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "candidate.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    original = source.read_bytes()
    pack = source if collision == "candidate" else root / "shared-output.json"
    plan = pack if collision == "same_outputs" else root / "evaluation-plan.json"

    with pytest.raises(StyleValidationError, match="style_path_collision"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
        )

    assert source.read_bytes() == original


def test_preparation_second_write_failure_leaves_no_formal_output(
    tmp_path: Path,
    synthetic_candidate_package,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "candidate.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    pack = root / "style-pack.json"
    plan = root / "evaluation-plan.json"
    real_write = style_private_io.write_private_model

    def fail_plan_write(model, destination: Path, private_data_dir: Path) -> Path:
        if not isinstance(model, StylePack):
            raise StyleValidationError("private_write_failed", "synthetic failure")
        return real_write(model, destination, private_data_dir)

    monkeypatch.setattr(style_private_io, "write_private_model", fail_plan_write)

    with pytest.raises(StyleValidationError, match="private_write_failed"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=SCENARIO_SOURCE,
            style_pack_destination=pack,
            evaluation_plan_destination=plan,
        )

    assert not pack.exists()
    assert not plan.exists()


def test_preparation_rejects_duplicate_scenarios_with_scrubbed_error(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "candidate.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    scenarios = json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
    scenarios[1]["scenario_id"] = scenarios[0]["scenario_id"]
    scenario_source = tmp_path / "duplicate-scenarios.json"
    scenario_source.write_text(json.dumps(scenarios), encoding="utf-8")

    with pytest.raises(StyleValidationError, match="invalid_evaluation_scenarios"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=scenario_source,
            style_pack_destination=root / "style-pack.json",
            evaluation_plan_destination=root / "evaluation-plan.json",
        )


def test_preparation_rejects_pre_migration_fixed_intents(
    tmp_path: Path,
    synthetic_candidate_package,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    source = root / "candidate.json"
    source.write_text(synthetic_candidate_package.model_dump_json(), encoding="utf-8")
    scenarios = json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
    greeting_reopen = next(item for item in scenarios if item["scenario_id"] == "greeting-2")
    greeting_reopen["intent"] = StyleIntent.CATEGORY_DEFAULT.value
    old_scenario_source = tmp_path / "old-fixed-intents.json"
    old_scenario_source.write_text(json.dumps(scenarios), encoding="utf-8")

    with pytest.raises(StyleValidationError, match="invalid_evaluation_scenarios"):
        prepare_style_test_version(
            candidate_source=source,
            private_data_dir=root,
            expected_count=3,
            scenario_source=old_scenario_source,
            style_pack_destination=root / "style-pack.json",
            evaluation_plan_destination=root / "evaluation-plan.json",
        )


def _write_finalization_inputs(
    root: Path,
    *,
    missing_response: bool,
    fixed_plan: bool = True,
) -> tuple[Path, ...]:
    if fixed_plan:
        raw_scenarios = json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
        scenarios = [
            (item["scenario_id"], CandidateCategory(item["category"]), item["intent"])
            for item in raw_scenarios
        ]
    else:
        scenarios = [
            (f"arbitrary-{index}", CandidateCategory.SLEEP, "category_default")
            for index in range(20)
        ]
    identifiers = [identifier for identifier, _, _ in scenarios]
    plan = EvaluationPlan(
        style_version="dev-style-synthetic",
        items=[
            EvaluationPlanItem(
                scenario_id=identifier,
                category=category,
                intent=intent,
                baseline_prompt="synthetic baseline",
                styled_prompt="synthetic styled",
                styled_examples=[],
                coverage_status=CoverageStatus.TESTABLE,
            )
            for identifier, category, intent in scenarios
        ],
    )
    response_ids = identifiers[:-1] if missing_response else identifiers
    responses = EvaluationResponses(
        items=[
            ScenarioResponses(
                scenario_id=identifier,
                baseline=f"baseline-{identifier}",
                styled=(
                    "SYNTHETIC_PRIVATE_RESPONSE_91D4"
                    if identifier == identifiers[0]
                    else f"styled-{identifier}"
                ),
            )
            for identifier in response_ids
        ]
    )
    key = BlindAnswerKey(styled_labels=dict.fromkeys(identifiers, "A"))
    ratings = BlindRatings(
        items=[
            ScenarioRating(
                scenario_id=identifier,
                preferred="A",
                mechanical_a=False,
                mechanical_b=False,
            )
            for identifier in identifiers
        ]
    )
    models = [plan, responses, key, ratings]
    paths = tuple(root / f"input-{index}.json" for index in range(len(models)))
    for path, model in zip(paths, models, strict=True):
        path.write_text(model.model_dump_json(), encoding="utf-8")
    return paths


def test_finalization_returns_aggregate_report_only(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    plan, responses, key, ratings = _write_finalization_inputs(
        root,
        missing_response=False,
    )

    report = finalize_style_test_version(
        private_data_dir=root,
        evaluation_plan_source=plan,
        responses_source=responses,
        answer_key_source=key,
        ratings_source=ratings,
    )

    payload = report.model_dump_json()
    assert report.total_scenarios == 20
    assert "prompt" not in payload
    assert "response" not in payload
    assert "styled_labels" not in payload
    assert str(root) not in payload


def test_finalization_rejects_nonfixed_twenty_scenario_plan(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    plan, responses, key, ratings = _write_finalization_inputs(
        root,
        missing_response=False,
        fixed_plan=False,
    )

    with pytest.raises(StyleValidationError, match="invalid_evaluation_plan"):
        finalize_style_test_version(
            private_data_dir=root,
            evaluation_plan_source=plan,
            responses_source=responses,
            answer_key_source=key,
            ratings_source=ratings,
        )


def test_finalization_failure_traceback_retains_no_response_or_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    plan, responses, key, ratings = _write_finalization_inputs(
        root,
        missing_response=True,
    )
    secret = "SYNTHETIC_PRIVATE_RESPONSE_91D4"

    with pytest.raises(StyleValidationError) as caught:
        finalize_style_test_version(
            private_data_dir=root,
            evaluation_plan_source=plan,
            responses_source=responses,
            answer_key_source=key,
            ratings_source=ratings,
        )

    traceback = caught.value.__traceback__
    checked_product_frame = False
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/companion/style/" in filename:
            checked_product_frame = True
            locals_text = repr(traceback.tb_frame.f_locals)
            assert secret not in locals_text
            assert str(root) not in locals_text
        traceback = traceback.tb_next
    assert checked_product_frame
