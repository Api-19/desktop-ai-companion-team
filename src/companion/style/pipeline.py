import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from companion.style.adjustments import candidate_package_fingerprint
from companion.style.calibration import (
    ReviewedCalibrationIndex,
    apply_reviewed_calibration,
    load_reviewed_calibration_package,
)
from companion.style.errors import StyleValidationError
from companion.style.evaluation import (
    _fixed_scenario_pairs,
    build_evaluation_plan,
    score_evaluation,
)
from companion.style.features import extract_style_features
from companion.style.iteration_pipeline import load_active_style_iteration
from companion.style.loader import load_safe_candidate_package
from companion.style.models import (
    BlindAnswerKey,
    BlindRatings,
    CoverageStatus,
    EvaluationPlan,
    EvaluationReport,
    EvaluationResponses,
    EvaluationScenario,
    PreparationSummary,
)
from companion.style.pack import build_style_pack
from companion.style.private_io import (
    _private_path,
    _StyleIoFailure,
    read_private_model,
    write_private_models_atomically,
)
from companion.style.retrieval import StyleExampleIndex


class _StylePipelineFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


def _load_scenarios(source: Path) -> list[EvaluationScenario]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("scenario payload must be a list")
        scenarios = [EvaluationScenario.model_validate(item) for item in payload]
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError):
        raise _StylePipelineFailure(
            "invalid_evaluation_scenarios",
            "evaluation scenarios are invalid",
        ) from None
    expected = _fixed_scenario_pairs()
    actual = {(scenario.scenario_id, scenario.category, scenario.intent) for scenario in scenarios}
    if len(scenarios) != 20 or actual != expected:
        raise _StylePipelineFailure(
            "invalid_evaluation_scenarios",
            "evaluation scenario set is invalid",
        )
    return scenarios


def _prepare_style_test_version(
    candidate_source: Path,
    private_data_dir: Path,
    expected_count: int,
    scenario_source: Path,
    style_pack_destination: Path,
    evaluation_plan_destination: Path,
    active_style_source: Path | None,
    active_manifest_source: Path | None,
    calibration_source: Path | None,
    expected_calibration_fingerprint: str | None,
    true_name_terms: Sequence[str] | None,
    approved_nicknames: Sequence[str] | None,
) -> PreparationSummary:
    try:
        if (active_style_source is None) != (active_manifest_source is None):
            raise _StylePipelineFailure(
                "incomplete_active_style_sources",
                "active style candidate and manifest must be provided together",
            )
        calibration_values = (
            calibration_source,
            expected_calibration_fingerprint,
            true_name_terms,
            approved_nicknames,
        )
        if any(value is not None for value in calibration_values) and not all(
            value is not None for value in calibration_values
        ):
            raise _StylePipelineFailure(
                "incomplete_calibration_sources",
                "reviewed calibration inputs must be provided together",
            )
        resolved_candidate = _private_path(candidate_source, private_data_dir)
        resolved_pack = _private_path(style_pack_destination, private_data_dir)
        resolved_plan = _private_path(evaluation_plan_destination, private_data_dir)
        resolved_scenario = scenario_source.resolve(strict=False)
        resolved_paths = [
            resolved_candidate,
            resolved_pack,
            resolved_plan,
            resolved_scenario,
        ]
        if calibration_source is not None:
            resolved_paths.append(_private_path(calibration_source, private_data_dir))
        if active_style_source is not None and active_manifest_source is not None:
            resolved_paths.extend(
                [
                    _private_path(active_style_source, private_data_dir),
                    _private_path(active_manifest_source, private_data_dir),
                ]
            )
        if len(set(resolved_paths)) != len(resolved_paths):
            raise _StylePipelineFailure(
                "style_path_collision",
                "style input and output paths must be distinct",
            )
        scenarios = _load_scenarios(scenario_source)
        package = load_safe_candidate_package(
            candidate_source,
            private_data_dir=private_data_dir,
            expected_count=expected_count,
        )
        features = extract_style_features(package)
        base_pack = build_style_pack(package, features)
        index = StyleExampleIndex.from_package(package)
        style_pack = base_pack
        if active_style_source is not None and active_manifest_source is not None:
            style_pack = load_active_style_iteration(
                manifest_source=active_manifest_source,
                candidate_source=active_style_source,
                private_data_dir=private_data_dir,
                dataset_fingerprint=candidate_package_fingerprint(package),
            )
        plan = build_evaluation_plan(style_pack, index, scenarios)
        if calibration_source is not None:
            reviewed_package = load_reviewed_calibration_package(
                calibration_source,
                private_data_dir=private_data_dir,
                expected_fingerprint=expected_calibration_fingerprint,  # type: ignore[arg-type]
                true_name_terms=true_name_terms,  # type: ignore[arg-type]
                approved_nicknames=approved_nicknames,  # type: ignore[arg-type]
            )
            reviewed_index = ReviewedCalibrationIndex(reviewed_package)
            plan = apply_reviewed_calibration(plan, reviewed_index)
            version_digest = hashlib.sha256(
                f"{style_pack.version}:{reviewed_package.content_fingerprint}".encode()
            ).hexdigest()[:12]
            style_pack = style_pack.model_copy(
                update={"version": f"dev-style-cal-{version_digest}"}
            )
            plan = plan.model_copy(update={"style_version": style_pack.version})
        write_private_models_atomically(
            [
                (style_pack, style_pack_destination),
                (plan, evaluation_plan_destination),
            ],
            private_data_dir,
        )
    except (_StyleIoFailure, StyleValidationError) as error:
        raise _StylePipelineFailure(error.code, error.detail) from None

    counts = {status: 0 for status in CoverageStatus}
    for item in plan.items:
        counts[item.coverage_status] += 1
    return PreparationSummary(
        candidate_count=len(package.candidates),
        scenario_count=len(plan.items),
        testable_count=counts[CoverageStatus.TESTABLE],
        sparse_count=counts[CoverageStatus.SPARSE],
        mechanical_risk_count=counts[CoverageStatus.MECHANICAL_RISK],
        missing_count=counts[CoverageStatus.MISSING],
    )


def prepare_style_test_version(
    *,
    candidate_source: Path,
    private_data_dir: Path,
    expected_count: int,
    scenario_source: Path,
    style_pack_destination: Path,
    evaluation_plan_destination: Path,
    active_style_source: Path | None = None,
    active_manifest_source: Path | None = None,
    calibration_source: Path | None = None,
    expected_calibration_fingerprint: str | None = None,
    true_name_terms: Sequence[str] | None = None,
    approved_nicknames: Sequence[str] | None = None,
) -> PreparationSummary:
    try:
        return _prepare_style_test_version(
            candidate_source,
            private_data_dir,
            expected_count,
            scenario_source,
            style_pack_destination,
            evaluation_plan_destination,
            active_style_source,
            active_manifest_source,
            calibration_source,
            expected_calibration_fingerprint,
            true_name_terms,
            approved_nicknames,
        )
    except _StylePipelineFailure as error:
        failure = (error.code, error.detail)
    del (
        candidate_source,
        private_data_dir,
        expected_count,
        scenario_source,
        style_pack_destination,
        evaluation_plan_destination,
        active_style_source,
        active_manifest_source,
        calibration_source,
        expected_calibration_fingerprint,
        true_name_terms,
        approved_nicknames,
    )
    raise StyleValidationError(failure[0], failure[1]) from None


def _finalize_style_test_version(
    private_data_dir: Path,
    evaluation_plan_source: Path,
    responses_source: Path,
    answer_key_source: Path,
    ratings_source: Path,
) -> EvaluationReport:
    try:
        plan = read_private_model(
            EvaluationPlan,
            evaluation_plan_source,
            private_data_dir,
        )
        actual_pairs = {(item.scenario_id, item.category, item.intent) for item in plan.items}
        if len(plan.items) != 20 or actual_pairs != _fixed_scenario_pairs():
            raise _StylePipelineFailure(
                "invalid_evaluation_plan",
                "evaluation plan does not match the fixed scenario set",
            )
        responses = read_private_model(
            EvaluationResponses,
            responses_source,
            private_data_dir,
        )
        key = read_private_model(BlindAnswerKey, answer_key_source, private_data_dir)
        ratings = read_private_model(BlindRatings, ratings_source, private_data_dir)
        return score_evaluation(plan, responses, key, ratings)
    except StyleValidationError as error:
        raise _StylePipelineFailure(error.code, error.detail) from None


def finalize_style_test_version(
    *,
    private_data_dir: Path,
    evaluation_plan_source: Path,
    responses_source: Path,
    answer_key_source: Path,
    ratings_source: Path,
) -> EvaluationReport:
    try:
        return _finalize_style_test_version(
            private_data_dir,
            evaluation_plan_source,
            responses_source,
            answer_key_source,
            ratings_source,
        )
    except _StylePipelineFailure as error:
        failure = (error.code, error.detail)
    del (
        private_data_dir,
        evaluation_plan_source,
        responses_source,
        answer_key_source,
        ratings_source,
    )
    raise StyleValidationError(failure[0], failure[1]) from None
