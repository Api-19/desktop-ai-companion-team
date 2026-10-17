import json
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest

import companion.style.iteration_pipeline as iteration_pipeline
import companion.style.private_io as style_private_io
from companion.contracts import StylePack
from companion.contracts.style import StyleIntent, StylePreference
from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.style.adjustments import propose_style_adjustments
from companion.style.errors import StyleValidationError
from companion.style.iteration_models import (
    AdjustmentDecision,
    AdjustmentDecisionPackage,
    ApprovedAdjustmentSet,
    StyleAdjustmentEvidence,
    StyleIterationManifest,
)
from companion.style.private_io import (
    read_private_model,
    write_private_model,
    write_private_models_atomically,
)

NOW = datetime(2026, 7, 19, tzinfo=UTC)


def synthetic_style_pack() -> StylePack:
    return StylePack(
        version="synthetic-v1",
        statistics={"candidate_count": 3},
        scene_tags=["sleep"],
        forbidden_traits=["do_not_mechanically_repeat"],
        examples=["synthetic safe example"],
    )


def synthetic_approved_set(*, approved: int, rejected: int) -> ApprovedAdjustmentSet:
    rules = [
        StylePreference(
            rule_id=f"synthetic-approved-{index}",
            category="sleep",
            expression=f"synthetic-expression-{index}",
            trigger_tags=["sleep"],
            max_consecutive_uses=1,
        )
        for index in range(approved)
    ]
    return ApprovedAdjustmentSet(
        proposal_id="synthetic-proposal",
        dataset_fingerprint="a" * 64,
        base_style_version="synthetic-v1",
        rules=rules,
        rejected_rule_ids=[f"synthetic-rejected-{index}" for index in range(rejected)],
    )


def synthetic_manifest() -> StyleIterationManifest:
    return StyleIterationManifest(
        dataset_fingerprint="a" * 64,
        base_style_version="synthetic-v1",
        candidate_style_version="synthetic-v2",
        candidate_digest="b" * 64,
        approved_rule_ids=["synthetic-approved-0"],
        base_preference_rule_ids=[],
        iteration_round=1,
        previous_manifest_digest=None,
        lineage_id="c" * 64,
        status="candidate",
        created_at=NOW,
        activated_at=None,
        invalidated_at=None,
    )


def assert_product_traceback_scrubbed(error: BaseException, sentinel: str) -> None:
    for frame, _ in traceback.walk_tb(error.__traceback__):
        if "/companion/style/" in frame.f_code.co_filename.replace("\\", "/"):
            assert sentinel not in repr(frame.f_locals)


def test_candidate_pack_contains_only_approved_rules() -> None:
    base = synthetic_style_pack()
    approved = synthetic_approved_set(approved=1, rejected=1)

    candidate = iteration_pipeline.build_iterated_style_pack(base, approved)

    assert candidate.version != base.version
    assert [item.rule_id for item in candidate.preferences] == [approved.rules[0].rule_id]
    assert candidate.examples == base.examples
    assert candidate.statistics == base.statistics


def test_candidate_version_is_deterministic_and_deduplicates_existing_rule() -> None:
    existing = synthetic_approved_set(approved=1, rejected=0).rules[0]
    base = synthetic_style_pack().model_copy(update={"preferences": [existing]})
    approved = synthetic_approved_set(approved=1, rejected=0)

    first = iteration_pipeline.build_iterated_style_pack(base, approved)
    second = iteration_pipeline.build_iterated_style_pack(base, approved)

    assert first == second
    assert first.preferences == [existing]


def test_candidate_rejects_conflicting_existing_rule() -> None:
    approved = synthetic_approved_set(approved=1, rejected=0)
    conflicting = approved.rules[0].model_copy(update={"expression": "different"})
    base = synthetic_style_pack().model_copy(update={"preferences": [conflicting]})

    with pytest.raises(StyleValidationError, match="style_rule_conflict"):
        iteration_pipeline.build_iterated_style_pack(base, approved)


def test_candidate_conflict_scrubs_expression_from_traceback() -> None:
    sentinel = "iter-style-expression-sentinel"
    approved = synthetic_approved_set(approved=1, rejected=0)
    base = synthetic_style_pack().model_copy(
        update={"preferences": [approved.rules[0].model_copy(update={"expression": sentinel})]}
    )

    with pytest.raises(StyleValidationError) as caught:
        iteration_pipeline.build_iterated_style_pack(base, approved)

    assert caught.value.code == "style_rule_conflict"
    assert_product_traceback_scrubbed(caught.value, sentinel)


def test_candidate_runtime_type_failure_uses_stable_scrubbed_error() -> None:
    sentinel = "synthetic-runtime-type-sentinel"

    with pytest.raises(StyleValidationError) as caught:
        iteration_pipeline.build_iterated_style_pack(  # type: ignore[arg-type]
            sentinel,
            synthetic_approved_set(approved=1, rejected=0),
        )

    assert caught.value.code == "invalid_style_iteration"
    assert sentinel not in str(caught.value)
    assert_product_traceback_scrubbed(caught.value, sentinel)


def _write_preparation_inputs(
    root: Path,
    *,
    base_pack: StylePack | None = None,
    suffix: str = "",
    expression: str = "synthetic-expression",
    candidate_expressions: tuple[str, ...] | None = None,
) -> tuple[Path, Path, Path]:
    base_pack = base_pack or synthetic_style_pack()
    candidate_expressions = candidate_expressions or (expression,)
    evidence = [
        StyleAdjustmentEvidence(
            category="sleep",
            expression=expression,
            trigger_tags=["sleep"],
            intent_tags=[StyleIntent.CATEGORY_DEFAULT],
            aggregate_count=1,
            source="explicit_owner_feedback",
        )
    ]
    package = SafeCandidatePackage(
        created_at=NOW,
        candidates=[
            SafeStyleCandidate(
                text=candidate_expression,
                category=CandidateCategory.SLEEP,
                scene_tags=["sleep"],
            )
            for candidate_expression in candidate_expressions
        ],
        audit=CandidateSelectionAudit(
            processed_at=NOW,
            input_count=len(candidate_expressions),
            owner_input_count=len(candidate_expressions),
            dropped_other_count=0,
            allowed_category_counts={"sleep": len(candidate_expressions)},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=len(candidate_expressions),
            succeeded=True,
        ),
    )
    proposal = propose_style_adjustments(
        evidence,
        candidate_package=package,
        proposal_id="synthetic-proposal",
        base_style_version=base_pack.version,
    )
    decision = AdjustmentDecisionPackage(
        proposal_id=proposal.proposal_id,
        decisions=[AdjustmentDecision(rule_id=proposal.rules[0].rule_id, decision="approve")],
    )
    base = root / f"base{suffix}.json"
    proposal_path = root / f"proposal{suffix}.json"
    decisions = root / f"decisions{suffix}.json"
    for model, path in (
        (base_pack, base),
        (proposal, proposal_path),
        (decision, decisions),
    ):
        write_private_model(model, path, root)
    return base, proposal_path, decisions


def test_preparation_rejects_round_four_before_reading_private_inputs(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()

    with pytest.raises(StyleValidationError, match="style_iteration_limit_reached"):
        iteration_pipeline.prepare_style_iteration(
            base_pack_source=root / "missing-base.json",
            proposal_source=root / "missing-proposal.json",
            decisions_source=root / "missing-decisions.json",
            private_data_dir=root,
            iteration_round=4,
            created_at=NOW,
            candidate_destination=root / "candidate.json",
            manifest_destination=root / "manifest.json",
        )


def test_preparation_publishes_candidate_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    base, proposal, decisions = _write_preparation_inputs(root)
    candidate, manifest = root / "candidate.json", root / "manifest.json"

    summary = iteration_pipeline.prepare_style_iteration(
        base_pack_source=base,
        proposal_source=proposal,
        decisions_source=decisions,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=candidate,
        manifest_destination=manifest,
    )

    loaded_candidate = read_private_model(StylePack, candidate, root)
    loaded_manifest = read_private_model(StyleIterationManifest, manifest, root)
    assert summary.approved_count == 1
    assert summary.rejected_count == 0
    assert summary.candidate_preference_count == 1
    assert loaded_manifest.candidate_style_version == loaded_candidate.version


def test_later_round_requires_an_active_previous_manifest(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    package_expressions = ("synthetic-expression", "synthetic-expression-two")
    base, proposal, decisions = _write_preparation_inputs(
        root,
        suffix="-1",
        candidate_expressions=package_expressions,
    )
    candidate_1, manifest_1 = root / "candidate-1.json", root / "manifest-1.json"
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base,
        proposal_source=proposal,
        decisions_source=decisions,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=candidate_1,
        manifest_destination=manifest_1,
    )
    candidate_pack = read_private_model(StylePack, candidate_1, root)
    base_2, proposal_2, decisions_2 = _write_preparation_inputs(
        root,
        base_pack=candidate_pack,
        suffix="-2",
        expression="synthetic-expression-two",
        candidate_expressions=package_expressions,
    )

    with pytest.raises(StyleValidationError, match="invalid_iteration_lineage"):
        iteration_pipeline.prepare_style_iteration(
            base_pack_source=base_2,
            proposal_source=proposal_2,
            decisions_source=decisions_2,
            private_data_dir=root,
            iteration_round=2,
            created_at=NOW,
            candidate_destination=root / "candidate-2.json",
            manifest_destination=root / "manifest-2.json",
            previous_manifest_source=manifest_1,
        )


def test_iterated_pack_cannot_reset_lineage_to_round_one(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    package_expressions = ("synthetic-expression", "synthetic-expression-two")
    base, proposal, decisions = _write_preparation_inputs(
        root,
        suffix="-1",
        candidate_expressions=package_expressions,
    )
    candidate_1, manifest_1 = root / "candidate-1.json", root / "manifest-1.json"
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base,
        proposal_source=proposal,
        decisions_source=decisions,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=candidate_1,
        manifest_destination=manifest_1,
    )
    candidate_pack = read_private_model(StylePack, candidate_1, root)
    base_reset, proposal_reset, decisions_reset = _write_preparation_inputs(
        root,
        base_pack=candidate_pack,
        suffix="-reset",
        expression="synthetic-reset-expression",
        candidate_expressions=(
            "synthetic-expression",
            "synthetic-reset-expression",
        ),
    )

    with pytest.raises(StyleValidationError, match="invalid_iteration_lineage"):
        iteration_pipeline.prepare_style_iteration(
            base_pack_source=base_reset,
            proposal_source=proposal_reset,
            decisions_source=decisions_reset,
            private_data_dir=root,
            iteration_round=1,
            created_at=NOW,
            candidate_destination=root / "candidate-reset.json",
            manifest_destination=root / "manifest-reset.json",
        )


def test_second_round_links_to_active_previous_manifest(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    package_expressions = ("synthetic-expression", "synthetic-expression-two")
    base, proposal, decisions = _write_preparation_inputs(
        root,
        suffix="-1",
        candidate_expressions=package_expressions,
    )
    candidate_1, manifest_1 = root / "candidate-1.json", root / "manifest-1.json"
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base,
        proposal_source=proposal,
        decisions_source=decisions,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=candidate_1,
        manifest_destination=manifest_1,
    )
    iteration_pipeline.activate_style_iteration(
        manifest_source=manifest_1,
        private_data_dir=root,
        user_confirmed=True,
        activated_at=NOW,
    )
    first_manifest = read_private_model(StyleIterationManifest, manifest_1, root)
    candidate_pack = read_private_model(StylePack, candidate_1, root)
    base_2, proposal_2, decisions_2 = _write_preparation_inputs(
        root,
        base_pack=candidate_pack,
        suffix="-2",
        expression="synthetic-expression-two",
        candidate_expressions=package_expressions,
    )
    manifest_2 = root / "manifest-2.json"

    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base_2,
        proposal_source=proposal_2,
        decisions_source=decisions_2,
        private_data_dir=root,
        iteration_round=2,
        created_at=NOW,
        candidate_destination=root / "candidate-2.json",
        manifest_destination=manifest_2,
        previous_manifest_source=manifest_1,
    )

    second_manifest = read_private_model(StyleIterationManifest, manifest_2, root)
    assert second_manifest.iteration_round == 2
    assert second_manifest.lineage_id == first_manifest.lineage_id
    assert second_manifest.previous_manifest_digest is not None


def test_second_round_rejects_changed_candidate_dataset(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    base, proposal, decisions = _write_preparation_inputs(
        root,
        suffix="-1",
        candidate_expressions=("synthetic-expression", "synthetic-expression-two"),
    )
    candidate_1, manifest_1 = root / "candidate-1.json", root / "manifest-1.json"
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base,
        proposal_source=proposal,
        decisions_source=decisions,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=candidate_1,
        manifest_destination=manifest_1,
    )
    iteration_pipeline.activate_style_iteration(
        manifest_source=manifest_1,
        private_data_dir=root,
        user_confirmed=True,
        activated_at=NOW,
    )
    candidate_pack = read_private_model(StylePack, candidate_1, root)
    base_2, proposal_2, decisions_2 = _write_preparation_inputs(
        root,
        base_pack=candidate_pack,
        suffix="-2",
        expression="synthetic-expression-two",
        candidate_expressions=("synthetic-expression-two", "changed-dataset-expression"),
    )

    with pytest.raises(StyleValidationError, match="invalid_iteration_lineage"):
        iteration_pipeline.prepare_style_iteration(
            base_pack_source=base_2,
            proposal_source=proposal_2,
            decisions_source=decisions_2,
            private_data_dir=root,
            iteration_round=2,
            created_at=NOW,
            candidate_destination=root / "candidate-2.json",
            manifest_destination=root / "manifest-2.json",
            previous_manifest_source=manifest_1,
        )


def test_initial_lineage_id_binds_base_content_and_candidate_dataset(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    base_a, proposal_a, decisions_a = _write_preparation_inputs(
        root,
        suffix="-a",
        candidate_expressions=("synthetic-expression",),
    )
    base_b, proposal_b, decisions_b = _write_preparation_inputs(
        root,
        suffix="-b",
        candidate_expressions=("synthetic-expression", "synthetic-other-expression"),
    )
    manifest_a, manifest_b = root / "manifest-a.json", root / "manifest-b.json"
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base_a,
        proposal_source=proposal_a,
        decisions_source=decisions_a,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=root / "candidate-a.json",
        manifest_destination=manifest_a,
    )
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base_b,
        proposal_source=proposal_b,
        decisions_source=decisions_b,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=root / "candidate-b.json",
        manifest_destination=manifest_b,
    )

    first = read_private_model(StyleIterationManifest, manifest_a, root)
    second = read_private_model(StyleIterationManifest, manifest_b, root)

    assert first.lineage_id != second.lineage_id


def test_active_load_allows_approved_rule_already_present_in_base(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    package_expressions = ("synthetic-expression",)
    base, proposal, decisions = _write_preparation_inputs(
        root,
        suffix="-1",
        candidate_expressions=package_expressions,
    )
    candidate_1, manifest_1 = root / "candidate-1.json", root / "manifest-1.json"
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base,
        proposal_source=proposal,
        decisions_source=decisions,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=candidate_1,
        manifest_destination=manifest_1,
    )
    iteration_pipeline.activate_style_iteration(
        manifest_source=manifest_1,
        private_data_dir=root,
        user_confirmed=True,
        activated_at=NOW,
    )
    candidate_pack = read_private_model(StylePack, candidate_1, root)
    base_2, proposal_2, decisions_2 = _write_preparation_inputs(
        root,
        base_pack=candidate_pack,
        suffix="-2",
        candidate_expressions=package_expressions,
    )
    candidate_2, manifest_2 = root / "candidate-2.json", root / "manifest-2.json"
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base_2,
        proposal_source=proposal_2,
        decisions_source=decisions_2,
        private_data_dir=root,
        iteration_round=2,
        created_at=NOW,
        candidate_destination=candidate_2,
        manifest_destination=manifest_2,
        previous_manifest_source=manifest_1,
    )
    active = iteration_pipeline.activate_style_iteration(
        manifest_source=manifest_2,
        private_data_dir=root,
        user_confirmed=True,
        activated_at=NOW,
    )

    loaded = iteration_pipeline.load_active_style_iteration(
        manifest_source=manifest_2,
        candidate_source=candidate_2,
        private_data_dir=root,
        dataset_fingerprint=active.dataset_fingerprint,
    )

    assert loaded == read_private_model(StylePack, candidate_2, root)


def test_active_load_rejects_candidate_content_tampering_with_preserved_version(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    base, proposal, decisions = _write_preparation_inputs(root)
    candidate_source, manifest_source = root / "candidate.json", root / "manifest.json"
    iteration_pipeline.prepare_style_iteration(
        base_pack_source=base,
        proposal_source=proposal,
        decisions_source=decisions,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=candidate_source,
        manifest_destination=manifest_source,
    )
    iteration_pipeline.activate_style_iteration(
        manifest_source=manifest_source,
        private_data_dir=root,
        user_confirmed=True,
        activated_at=NOW,
    )
    manifest = read_private_model(StyleIterationManifest, manifest_source, root)
    payload = json.loads(candidate_source.read_text(encoding="utf-8"))
    original_version = payload["version"]
    payload["examples"].append("synthetic-tampered-example")
    candidate_source.write_text(json.dumps(payload), encoding="utf-8")
    assert payload["version"] == original_version

    with pytest.raises(StyleValidationError, match="inactive_style_iteration"):
        iteration_pipeline.load_active_style_iteration(
            manifest_source=manifest_source,
            candidate_source=candidate_source,
            private_data_dir=root,
            dataset_fingerprint=manifest.dataset_fingerprint,
        )


def test_preparation_rolls_back_both_outputs_when_second_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    base, proposal, decisions = _write_preparation_inputs(root)
    candidate, manifest = root / "candidate.json", root / "manifest.json"
    real_write = style_private_io.write_private_model

    def fail_second_write(model, destination: Path, private_data_dir: Path) -> Path:
        if isinstance(model, StyleIterationManifest):
            raise StyleValidationError("private_write_failed", "synthetic failure")
        return real_write(model, destination, private_data_dir)

    monkeypatch.setattr(style_private_io, "write_private_model", fail_second_write)
    with pytest.raises(StyleValidationError, match="private_write_failed"):
        iteration_pipeline.prepare_style_iteration(
            base_pack_source=base,
            proposal_source=proposal,
            decisions_source=decisions,
            private_data_dir=root,
            iteration_round=1,
            created_at=NOW,
            candidate_destination=candidate,
            manifest_destination=manifest,
        )
    assert not candidate.exists()
    assert not manifest.exists()


def test_activation_requires_explicit_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    manifest_source = root / "iterations" / "manifest.json"
    write_private_model(synthetic_manifest(), manifest_source, root)

    with pytest.raises(StyleValidationError, match="style_activation_not_confirmed"):
        iteration_pipeline.activate_style_iteration(
            manifest_source=manifest_source,
            private_data_dir=root,
            user_confirmed=False,
            activated_at=NOW,
        )


def test_unconfirmed_activation_scrubs_private_path_from_traceback(tmp_path: Path) -> None:
    root = tmp_path / "private-path-sentinel"
    root.mkdir()

    with pytest.raises(StyleValidationError) as caught:
        iteration_pipeline.activate_style_iteration(
            manifest_source=root / "manifest.json",
            private_data_dir=root,
            user_confirmed=False,
            activated_at=NOW,
        )

    assert_product_traceback_scrubbed(caught.value, str(root))


def test_atomic_writer_rejects_forged_model_without_warning_or_traceback_retention(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "synthetic-forged-atomic-artifact-sentinel"
    root = tmp_path / "private"
    root.mkdir()
    forged = StylePack.model_construct(
        version=[sentinel],
        statistics={},
        scene_tags=[],
        forbidden_traits=[],
        examples=[],
        preferences=[],
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(StyleValidationError) as caught:
            write_private_models_atomically([(forged, root / "artifact.json")], root)

    assert sentinel not in "\n".join(str(item.message) for item in caught_warnings)
    assert sentinel not in capsys.readouterr().err
    assert_product_traceback_scrubbed(caught.value, sentinel)


def test_atomic_writer_scrubs_model_dump_runtime_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "synthetic-model-dump-runtime-sentinel"
    root = tmp_path / "private"
    root.mkdir()

    class FailingDumpStylePack(StylePack):
        def model_dump_json(self, **kwargs: object) -> str:
            del kwargs
            raise RuntimeError(sentinel)

    model = FailingDumpStylePack(
        version="synthetic-v1",
        statistics={},
        scene_tags=[],
        forbidden_traits=[],
        examples=[],
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(StyleValidationError) as caught:
            write_private_models_atomically([(model, root / "artifact.json")], root)

    assert caught.value.code == "private_write_failed"
    assert sentinel not in str(caught.value)
    assert sentinel not in "\n".join(str(item.message) for item in caught_warnings)
    assert sentinel not in capsys.readouterr().err
    assert_product_traceback_scrubbed(caught.value, sentinel)


def test_active_load_rejects_fingerprint_mismatch_and_invalidated_manifest(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    manifest_source, candidate_source = root / "manifest.json", root / "candidate.json"
    candidate = synthetic_style_pack().model_copy(update={"version": "synthetic-v2"})
    write_private_model(synthetic_manifest(), manifest_source, root)
    write_private_model(candidate, candidate_source, root)
    iteration_pipeline.activate_style_iteration(
        manifest_source=manifest_source,
        private_data_dir=root,
        user_confirmed=True,
        activated_at=NOW,
    )

    with pytest.raises(StyleValidationError, match="inactive_style_iteration"):
        iteration_pipeline.load_active_style_iteration(
            manifest_source=manifest_source,
            candidate_source=candidate_source,
            private_data_dir=root,
            dataset_fingerprint="b" * 64,
        )

    iteration_pipeline.invalidate_style_iteration(
        manifest_source=manifest_source,
        private_data_dir=root,
        invalidated_at=NOW,
    )
    with pytest.raises(StyleValidationError, match="inactive_style_iteration"):
        iteration_pipeline.load_active_style_iteration(
            manifest_source=manifest_source,
            candidate_source=candidate_source,
            private_data_dir=root,
            dataset_fingerprint="a" * 64,
        )
