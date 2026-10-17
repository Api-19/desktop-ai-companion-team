import hashlib
import json
from datetime import datetime
from pathlib import Path

from companion.contracts import StylePack
from companion.style.adjustments import review_style_adjustments
from companion.style.errors import StyleValidationError
from companion.style.iteration_models import (
    AdjustmentDecisionPackage,
    ApprovedAdjustmentSet,
    StyleAdjustmentProposal,
    StyleIterationManifest,
    StyleIterationSummary,
)
from companion.style.private_io import (
    _private_path,
    _StyleIoFailure,
    read_private_model,
    write_private_model,
    write_private_models_atomically,
)


def _model_digest(model: object) -> str:
    payload = json.dumps(
        model.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _candidate_version(base_version: str, preferences: list[object]) -> str:
    payload = json.dumps(
        [base_version, [item.model_dump(mode="json", warnings=False) for item in preferences]],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"dev-style-iter-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _initial_lineage_id(base_pack: StylePack, dataset_fingerprint: str) -> str:
    payload = json.dumps(
        [base_pack.version, _model_digest(base_pack), dataset_fingerprint],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_iterated_style_pack(
    base_pack: StylePack,
    approved: ApprovedAdjustmentSet,
) -> StylePack:
    merged = {rule.rule_id: rule for rule in base_pack.preferences}
    for rule in approved.rules:
        existing = merged.get(rule.rule_id)
        if existing is not None and existing != rule:
            raise StyleValidationError(
                "style_rule_conflict",
                "approved style rule conflicts with the current pack",
            )
        merged[rule.rule_id] = rule
    preferences = [merged[rule_id] for rule_id in sorted(merged)]
    version = _candidate_version(base_pack.version, preferences)
    return StylePack(
        version=version,
        statistics=base_pack.statistics,
        scene_tags=base_pack.scene_tags,
        forbidden_traits=base_pack.forbidden_traits,
        examples=base_pack.examples,
        preferences=preferences,
    )


def build_iterated_style_pack(
    base_pack: StylePack,
    approved: ApprovedAdjustmentSet,
) -> StylePack:
    try:
        validated_base = StylePack.model_validate(base_pack.model_dump(mode="json", warnings=False))
        validated_approved = ApprovedAdjustmentSet.model_validate(
            approved.model_dump(mode="json", warnings=False)
        )
        return _build_iterated_style_pack(validated_base, validated_approved)
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        raise StyleValidationError(
            "invalid_style_iteration",
            "style iteration input is invalid",
        ) from None
    finally:
        base_pack = None  # type: ignore[assignment]
        approved = None  # type: ignore[assignment]
        validated_base = None
        validated_approved = None
    raise StyleValidationError(*failure) from None


def _prepare_style_iteration(
    *,
    base_pack_source: Path,
    proposal_source: Path,
    decisions_source: Path,
    private_data_dir: Path,
    iteration_round: int,
    created_at: datetime,
    candidate_destination: Path,
    manifest_destination: Path,
    previous_manifest_source: Path | None,
) -> StyleIterationSummary:
    if iteration_round not in (1, 2, 3):
        raise StyleValidationError(
            "style_iteration_limit_reached",
            "style iteration round must be between one and three",
        )
    paths = [
        _private_path(path, private_data_dir)
        for path in (
            base_pack_source,
            proposal_source,
            decisions_source,
            candidate_destination,
            manifest_destination,
            *(() if previous_manifest_source is None else (previous_manifest_source,)),
        )
    ]
    if len(set(paths)) != len(paths):
        raise StyleValidationError("style_path_collision", "style iteration paths must differ")
    base_pack = read_private_model(StylePack, base_pack_source, private_data_dir)
    previous = None
    if iteration_round == 1:
        if previous_manifest_source is not None or base_pack.preferences:
            raise StyleValidationError(
                "invalid_iteration_lineage", "style iteration lineage is invalid"
            )
    else:
        if previous_manifest_source is None:
            raise StyleValidationError(
                "invalid_iteration_lineage", "style iteration lineage is invalid"
            )
        previous = read_private_model(
            StyleIterationManifest, previous_manifest_source, private_data_dir
        )
        if (
            previous.status != "active"
            or previous.iteration_round != iteration_round - 1
            or previous.candidate_style_version != base_pack.version
            or previous.candidate_digest != _model_digest(base_pack)
        ):
            raise StyleValidationError(
                "invalid_iteration_lineage", "style iteration lineage is invalid"
            )
    proposal = read_private_model(StyleAdjustmentProposal, proposal_source, private_data_dir)
    decision_package = read_private_model(
        AdjustmentDecisionPackage, decisions_source, private_data_dir
    )
    if decision_package.proposal_id != proposal.proposal_id:
        raise StyleValidationError(
            "adjustment_proposal_mismatch",
            "adjustment decisions do not match the proposal",
        )
    approved = review_style_adjustments(proposal, decision_package.decisions)
    if approved.base_style_version != base_pack.version:
        raise StyleValidationError(
            "base_style_version_mismatch",
            "style adjustment base version is invalid",
        )
    if previous is not None and approved.dataset_fingerprint != previous.dataset_fingerprint:
        raise StyleValidationError(
            "invalid_iteration_lineage", "style iteration lineage is invalid"
        )
    candidate = _build_iterated_style_pack(base_pack, approved)
    manifest = StyleIterationManifest(
        dataset_fingerprint=approved.dataset_fingerprint,
        base_style_version=base_pack.version,
        candidate_style_version=candidate.version,
        candidate_digest=_model_digest(candidate),
        approved_rule_ids=[rule.rule_id for rule in approved.rules],
        base_preference_rule_ids=[rule.rule_id for rule in base_pack.preferences],
        iteration_round=iteration_round,
        previous_manifest_digest=None if previous is None else _model_digest(previous),
        lineage_id=(
            _initial_lineage_id(base_pack, approved.dataset_fingerprint)
            if previous is None
            else previous.lineage_id
        ),
        status="candidate",
        created_at=created_at,
        activated_at=None,
        invalidated_at=None,
    )
    write_private_models_atomically(
        [(candidate, candidate_destination), (manifest, manifest_destination)],
        private_data_dir,
    )
    return StyleIterationSummary(
        approved_count=len(approved.rules),
        rejected_count=len(approved.rejected_rule_ids),
        candidate_preference_count=len(candidate.preferences),
    )


def prepare_style_iteration(
    *,
    base_pack_source: Path,
    proposal_source: Path,
    decisions_source: Path,
    private_data_dir: Path,
    iteration_round: int,
    created_at: datetime,
    candidate_destination: Path,
    manifest_destination: Path,
    previous_manifest_source: Path | None = None,
) -> StyleIterationSummary:
    try:
        return _prepare_style_iteration(
            base_pack_source=base_pack_source,
            proposal_source=proposal_source,
            decisions_source=decisions_source,
            private_data_dir=private_data_dir,
            iteration_round=iteration_round,
            created_at=created_at,
            candidate_destination=candidate_destination,
            manifest_destination=manifest_destination,
            previous_manifest_source=previous_manifest_source,
        )
    except (_StyleIoFailure, StyleValidationError) as error:
        failure = (error.code, error.detail)
    except Exception:
        failure = ("invalid_style_iteration", "style iteration input is invalid")
    del (
        base_pack_source,
        proposal_source,
        decisions_source,
        private_data_dir,
        iteration_round,
        created_at,
        candidate_destination,
        manifest_destination,
        previous_manifest_source,
    )
    raise StyleValidationError(*failure) from None


def activate_style_iteration(
    *,
    manifest_source: Path,
    private_data_dir: Path,
    user_confirmed: bool,
    activated_at: datetime,
) -> StyleIterationManifest:
    try:
        if user_confirmed is not True:
            raise StyleValidationError(
                "style_activation_not_confirmed",
                "style activation requires explicit owner confirmation",
            )
        manifest = read_private_model(StyleIterationManifest, manifest_source, private_data_dir)
        if manifest.status != "candidate":
            raise StyleValidationError(
                "invalid_iteration_state", "style iteration state is invalid"
            )
        activated = manifest.model_copy(
            update={"status": "active", "activated_at": activated_at, "invalidated_at": None}
        )
        activated = StyleIterationManifest.model_validate(activated.model_dump(warnings=False))
        write_private_model(activated, manifest_source, private_data_dir)
        return activated
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        raise StyleValidationError(
            "invalid_iteration_state", "style iteration state is invalid"
        ) from None
    finally:
        manifest_source = None  # type: ignore[assignment]
        private_data_dir = None  # type: ignore[assignment]
        activated_at = None  # type: ignore[assignment]
        manifest = None
        activated = None
    raise StyleValidationError(*failure) from None


def invalidate_style_iteration(
    *,
    manifest_source: Path,
    private_data_dir: Path,
    invalidated_at: datetime,
) -> StyleIterationManifest:
    try:
        manifest = read_private_model(StyleIterationManifest, manifest_source, private_data_dir)
        if manifest.status == "invalidated":
            raise StyleValidationError(
                "invalid_iteration_state", "style iteration state is invalid"
            )
        invalidated = manifest.model_copy(
            update={"status": "invalidated", "invalidated_at": invalidated_at}
        )
        invalidated = StyleIterationManifest.model_validate(invalidated.model_dump(warnings=False))
        write_private_model(invalidated, manifest_source, private_data_dir)
        return invalidated
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        raise StyleValidationError(
            "invalid_iteration_state", "style iteration state is invalid"
        ) from None
    finally:
        manifest_source = None  # type: ignore[assignment]
        private_data_dir = None  # type: ignore[assignment]
        invalidated_at = None  # type: ignore[assignment]
        manifest = None
        invalidated = None
    raise StyleValidationError(*failure) from None


def _require_active_style_iteration_snapshot(
    manifest: StyleIterationManifest,
    candidate: StylePack,
    dataset_fingerprint: str,
) -> None:
    candidate_ids = {rule.rule_id for rule in candidate.preferences}
    base_ids = set(manifest.base_preference_rule_ids)
    approved_ids = set(manifest.approved_rule_ids)
    if (
        manifest.status != "active"
        or manifest.dataset_fingerprint != dataset_fingerprint
        or manifest.candidate_style_version != candidate.version
        or manifest.candidate_digest != _model_digest(candidate)
        or candidate.version
        != _candidate_version(manifest.base_style_version, candidate.preferences)
        or candidate_ids != base_ids | approved_ids
    ):
        raise StyleValidationError("inactive_style_iteration", "style iteration is not active")


def load_active_style_iteration(
    *,
    manifest_source: Path,
    candidate_source: Path,
    private_data_dir: Path,
    dataset_fingerprint: str,
) -> StylePack:
    try:
        manifest = read_private_model(StyleIterationManifest, manifest_source, private_data_dir)
        candidate = read_private_model(StylePack, candidate_source, private_data_dir)
        _require_active_style_iteration_snapshot(manifest, candidate, dataset_fingerprint)
        return candidate
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = ("inactive_style_iteration", "style iteration is not active")
    finally:
        manifest_source = None  # type: ignore[assignment]
        candidate_source = None  # type: ignore[assignment]
        private_data_dir = None  # type: ignore[assignment]
        dataset_fingerprint = None  # type: ignore[assignment]
        manifest = None
        candidate = None
    raise StyleValidationError(*failure) from None


def _load_verified_active_style_iteration(
    *,
    manifest_source: Path,
    candidate_source: Path,
    private_data_dir: Path,
    dataset_fingerprint: str,
    expected_manifest_digest: str,
) -> tuple[StylePack, str]:
    if (
        not isinstance(expected_manifest_digest, str)
        or len(expected_manifest_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_manifest_digest)
    ):
        raise StyleValidationError(
            "invalid_active_manifest_digest",
            "active style manifest digest is invalid",
        )
    manifest = read_private_model(StyleIterationManifest, manifest_source, private_data_dir)
    candidate = read_private_model(StylePack, candidate_source, private_data_dir)
    manifest_digest = _model_digest(manifest)
    if manifest_digest != expected_manifest_digest:
        raise StyleValidationError(
            "inactive_style_iteration",
            "style iteration is not active",
        )
    _require_active_style_iteration_snapshot(manifest, candidate, dataset_fingerprint)
    return candidate, manifest_digest


def load_verified_active_style_iteration(
    *,
    manifest_source: Path,
    candidate_source: Path,
    private_data_dir: Path,
    dataset_fingerprint: str,
    expected_manifest_digest: str,
) -> tuple[StylePack, str]:
    """Load and bind one active-manifest snapshot to a caller-trusted digest."""
    try:
        return _load_verified_active_style_iteration(
            manifest_source=manifest_source,
            candidate_source=candidate_source,
            private_data_dir=private_data_dir,
            dataset_fingerprint=dataset_fingerprint,
            expected_manifest_digest=expected_manifest_digest,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = ("inactive_style_iteration", "style iteration is not active")
    finally:
        manifest_source = None  # type: ignore[assignment]
        candidate_source = None  # type: ignore[assignment]
        private_data_dir = None  # type: ignore[assignment]
        dataset_fingerprint = None  # type: ignore[assignment]
        expected_manifest_digest = None  # type: ignore[assignment]
    raise StyleValidationError(*failure) from None
