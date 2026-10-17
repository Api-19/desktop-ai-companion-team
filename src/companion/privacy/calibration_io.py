from __future__ import annotations

import hmac
import json
import os
import stat
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from companion.privacy.calibration_models import (
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CalibrationReviewDecision,
    CalibrationReviewPackage,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
)
from companion.privacy.candidate_io import _CandidateIoFailure, _validated_runtime_terms
from companion.privacy.candidate_rules import hard_rejection_reason
from companion.privacy.errors import (
    PrivacyValidationError,
    _sanitized_public_privacy_failure,
)

CALIBRATION_CANDIDATE_RELATIVE_PATH = Path(
    "style_candidates/calibration/current/calibration-candidates.json"
)


class _CalibrationIoFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


_CALIBRATION_IO_FAILURE_DETAILS = {
    "calibration_candidate_missing": ("calibration candidate could not be revoked",),
    "calibration_publish_failed": ("calibration file could not be published",),
    "calibration_rollback_failed": ("calibration file rollback failed",),
    "calibration_verification_failed": (
        "calibration candidate package verification failed",
        "calibration file verification failed",
    ),
    "invalid_calibration_package": ("calibration candidate package is invalid",),
    "invalid_calibration_revocation": ("calibration candidate revocation is invalid",),
    "invalid_candidate_terms": ("candidate terms are invalid",),
    "stale_calibration_identity_config": ("calibration identity configuration does not match",),
    "stale_calibration_package": ("calibration candidate package fingerprint does not match",),
    "unsafe_calibration_path": (
        "calibration path could not be validated",
        "calibration path must stay inside private data",
    ),
}
_PUBLISH_FALLBACK = (
    "calibration_verification_failed",
    "calibration candidate package verification failed",
)
_LOAD_FALLBACK = (
    "invalid_calibration_package",
    "calibration candidate package is invalid",
)
_REVOKE_FALLBACK = (
    "invalid_calibration_revocation",
    "calibration candidate revocation is invalid",
)


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        raise _CalibrationIoFailure(
            "unsafe_calibration_path",
            "calibration path could not be validated",
        ) from None
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _safe_private_path(
    path: Path,
    private_data_dir: Path,
    *,
    include_leaf: bool,
) -> Path:
    try:
        private_root = private_data_dir.absolute()
        requested = path.absolute()
    except (AttributeError, OSError):
        raise _CalibrationIoFailure(
            "unsafe_calibration_path",
            "calibration path could not be validated",
        ) from None
    if not requested.is_relative_to(private_root):
        raise _CalibrationIoFailure(
            "unsafe_calibration_path",
            "calibration path must stay inside private data",
        )
    checked = requested if include_leaf else requested.parent
    component = private_root
    if _is_reparse_point(component):
        raise _CalibrationIoFailure(
            "unsafe_calibration_path",
            "calibration path must stay inside private data",
        )
    for part in checked.relative_to(private_root).parts:
        component /= part
        if _is_reparse_point(component):
            raise _CalibrationIoFailure(
                "unsafe_calibration_path",
                "calibration path must stay inside private data",
            )
    try:
        resolved_root = private_root.resolve()
        resolved = requested.resolve()
    except OSError:
        raise _CalibrationIoFailure(
            "unsafe_calibration_path",
            "calibration path could not be validated",
        ) from None
    if not resolved.is_relative_to(resolved_root):
        raise _CalibrationIoFailure(
            "unsafe_calibration_path",
            "calibration path must stay inside private data",
        )
    return resolved


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _CalibrationIoFailure(
            "invalid_calibration_decisions",
            "calibration decisions are invalid",
        ) from None


def _atomic_write_model(
    model: BaseModel,
    destination: Path,
    private_data_dir: Path,
    model_type: type[BaseModel],
) -> Path:
    resolved = _safe_private_path(
        destination,
        private_data_dir,
        include_leaf=destination.exists(),
    )
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise _CalibrationIoFailure(
            "calibration_publish_failed",
            "calibration file could not be published",
        ) from None
    resolved = _safe_private_path(
        destination,
        private_data_dir,
        include_leaf=destination.exists(),
    )
    temporary: Path | None = None
    backup: Path | None = None
    replaced = False
    descriptor: int | None = None

    def raise_after_rollback(failure: _CalibrationIoFailure) -> None:
        nonlocal backup
        if replaced:
            try:
                if backup is None:
                    resolved.unlink(missing_ok=True)
                    raise failure from None
                os.replace(backup, resolved)
                backup = None
            except OSError:
                failure = _CalibrationIoFailure(
                    "calibration_rollback_failed",
                    "calibration file rollback failed",
                )
        raise failure from None

    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            dir=resolved.parent,
        )
        temporary = Path(raw_temporary)
        payload = model.model_dump_json(indent=2).encode("utf-8")
        with os.fdopen(descriptor, "w+b") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            output.seek(0)
            readback = output.read()
        verified = model_type.model_validate_json(readback)
        if verified != model:
            raise _CalibrationIoFailure(
                "calibration_verification_failed",
                "calibration file verification failed",
            )
        _safe_private_path(
            destination,
            private_data_dir,
            include_leaf=destination.exists(),
        )
        if resolved.exists():
            backup_descriptor, raw_backup = tempfile.mkstemp(
                prefix=f".{resolved.name}.",
                suffix=".backup",
                dir=resolved.parent,
            )
            os.close(backup_descriptor)
            backup = Path(raw_backup)
            backup.unlink()
            os.link(resolved, backup)
        os.replace(temporary, resolved)
        replaced = True
        temporary = None
        try:
            formal = model_type.model_validate_json(resolved.read_bytes())
        except (OSError, UnicodeError, ValidationError, ValueError):
            raise _CalibrationIoFailure(
                "calibration_verification_failed",
                "calibration file verification failed",
            ) from None
        if formal != model:
            raise _CalibrationIoFailure(
                "calibration_verification_failed",
                "calibration file verification failed",
            )
        if backup is not None:
            backup.unlink()
            backup = None
        return resolved
    except _CalibrationIoFailure as error:
        raise_after_rollback(error)
    except (OSError, UnicodeError, ValidationError, ValueError):
        raise_after_rollback(
            _CalibrationIoFailure(
                "calibration_publish_failed",
                "calibration file could not be published",
            )
        )
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        if backup is not None:
            with suppress(OSError):
                backup.unlink(missing_ok=True)


def _validated_review_package(package: CalibrationReviewPackage) -> CalibrationReviewPackage:
    try:
        return CalibrationReviewPackage.model_validate(package.model_dump())
    except (AttributeError, ValidationError, ValueError):
        raise _CalibrationIoFailure(
            "invalid_calibration_review_package",
            "calibration review package is invalid",
        ) from None


def write_calibration_review_package(
    package: CalibrationReviewPackage,
    *,
    destination: Path,
    private_data_dir: Path,
) -> Path:
    try:
        validated = _validated_review_package(package)
        return _atomic_write_model(
            validated,
            destination,
            private_data_dir,
            CalibrationReviewPackage,
        )
    except _CalibrationIoFailure as error:
        failure = (error.code, error.detail)
    del package, destination, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _read_calibration_review_decisions(
    source: Path,
    private_data_dir: Path,
) -> list[CalibrationReviewDecision]:
    resolved = _safe_private_path(source, private_data_dir, include_leaf=True)
    payload = _read_json(resolved)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"decisions"}
        or not isinstance(payload["decisions"], list)
    ):
        raise _CalibrationIoFailure(
            "invalid_calibration_decisions",
            "calibration decisions are invalid",
        )
    try:
        return [
            CalibrationReviewDecision.model_validate(decision) for decision in payload["decisions"]
        ]
    except (ValidationError, ValueError):
        raise _CalibrationIoFailure(
            "invalid_calibration_decisions",
            "calibration decisions are invalid",
        ) from None


def read_calibration_review_decisions(
    source: Path,
    *,
    private_data_dir: Path,
) -> list[CalibrationReviewDecision]:
    try:
        return _read_calibration_review_decisions(source, private_data_dir)
    except _CalibrationIoFailure as error:
        failure = (error.code, error.detail)
    del source, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _validated_candidate_package(
    package: CalibrationCandidatePackage,
    *,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    try:
        validated = CalibrationCandidatePackage.model_validate(package.model_dump())
    except (AttributeError, ValidationError, ValueError):
        raise _CalibrationIoFailure(
            "calibration_verification_failed",
            "calibration candidate package verification failed",
        ) from None
    try:
        names, nicknames = _validated_runtime_terms(
            true_name_terms,
            approved_nicknames,
        )
    except _CandidateIoFailure as error:
        raise _CalibrationIoFailure(error.code, error.detail) from None
    expected_identity_fingerprint = calibration_identity_config_fingerprint(
        names,
        nicknames,
    )
    if not hmac.compare_digest(
        validated.identity_config_fingerprint,
        expected_identity_fingerprint,
    ):
        raise _CalibrationIoFailure(
            "stale_calibration_identity_config",
            "calibration identity configuration does not match",
        )
    if any(
        hard_rejection_reason(candidate.text) is not None
        or any(name in candidate.text for name in names)
        for candidate in validated.candidates
    ):
        raise _CalibrationIoFailure(
            "calibration_verification_failed",
            "calibration candidate package verification failed",
        )
    return validated


def publish_calibration_candidate_package(
    package: CalibrationCandidatePackage,
    *,
    private_data_dir: Path,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> Path:
    try:
        return _atomic_write_model(
            _validated_candidate_package(
                package,
                true_name_terms=true_name_terms,
                approved_nicknames=approved_nicknames,
            ),
            private_data_dir / CALIBRATION_CANDIDATE_RELATIVE_PATH,
            private_data_dir,
            CalibrationCandidatePackage,
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_CalibrationIoFailure,
            allowed_details=_CALIBRATION_IO_FAILURE_DETAILS,
            fallback=_PUBLISH_FALLBACK,
        )
    del package, private_data_dir, true_name_terms, approved_nicknames
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _load_calibration_candidate_package(
    private_data_dir: Path,
    expected_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
        raise _CalibrationIoFailure(
            "stale_calibration_package",
            "calibration candidate package fingerprint does not match",
        )
    source = private_data_dir / CALIBRATION_CANDIDATE_RELATIVE_PATH
    resolved = _safe_private_path(source, private_data_dir, include_leaf=True)
    try:
        package = CalibrationCandidatePackage.model_validate_json(resolved.read_bytes())
    except (OSError, UnicodeError, ValidationError, ValueError):
        raise _CalibrationIoFailure(
            "invalid_calibration_package",
            "calibration candidate package is invalid",
        ) from None
    validated = _validated_candidate_package(
        package,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    if validated.content_fingerprint != expected_fingerprint:
        raise _CalibrationIoFailure(
            "stale_calibration_package",
            "calibration candidate package fingerprint does not match",
        )
    return validated


def load_calibration_candidate_package(
    *,
    private_data_dir: Path,
    expected_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    try:
        return _load_calibration_candidate_package(
            private_data_dir,
            expected_fingerprint,
            true_name_terms,
            approved_nicknames,
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_CalibrationIoFailure,
            allowed_details=_CALIBRATION_IO_FAILURE_DETAILS,
            fallback=_LOAD_FALLBACK,
        )
    del private_data_dir, expected_fingerprint, true_name_terms, approved_nicknames
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _revoke_calibration_candidates(
    candidate_ids: Sequence[str],
    *,
    private_data_dir: Path,
    revoked_at: datetime,
    expected_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    if (
        isinstance(candidate_ids, (str, bytes))
        or not isinstance(candidate_ids, Sequence)
        or not candidate_ids
        or any(
            not isinstance(candidate_id, str) or len(candidate_id) != 64
            for candidate_id in candidate_ids
        )
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise _CalibrationIoFailure(
            "invalid_calibration_revocation",
            "calibration candidate revocation is invalid",
        )
    current = _load_calibration_candidate_package(
        private_data_dir,
        expected_fingerprint,
        true_name_terms,
        approved_nicknames,
    )
    requested = set(candidate_ids)
    available = {candidate.candidate_id for candidate in current.candidates}
    if not requested.issubset(available):
        raise _CalibrationIoFailure(
            "calibration_candidate_missing",
            "calibration candidate could not be revoked",
        )
    remaining = [
        candidate for candidate in current.candidates if candidate.candidate_id not in requested
    ]
    audit = CalibrationCandidateAudit(
        reviewed_at=revoked_at,
        submission_count=current.audit.submission_count,
        approved_count=len(remaining),
        rejected_count=current.audit.rejected_count,
        privacy_rejected_count=current.audit.privacy_rejected_count,
        revoked_count=current.audit.revoked_count + len(requested),
        output_count=len(remaining),
    )
    regenerated = CalibrationCandidatePackage(
        created_at=revoked_at,
        content_fingerprint=calibration_review_fingerprint(remaining),
        identity_config_fingerprint=current.identity_config_fingerprint,
        candidate_count=len(remaining),
        candidates=remaining,
        audit=audit,
    )
    _atomic_write_model(
        regenerated,
        private_data_dir / CALIBRATION_CANDIDATE_RELATIVE_PATH,
        private_data_dir,
        CalibrationCandidatePackage,
    )
    return regenerated


def revoke_calibration_candidates(
    candidate_ids: Sequence[str],
    *,
    private_data_dir: Path,
    revoked_at: datetime,
    expected_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    try:
        return _revoke_calibration_candidates(
            candidate_ids,
            private_data_dir=private_data_dir,
            revoked_at=revoked_at,
            expected_fingerprint=expected_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_CalibrationIoFailure,
            allowed_details=_CALIBRATION_IO_FAILURE_DETAILS,
            fallback=_REVOKE_FALLBACK,
        )
    del (
        candidate_ids,
        private_data_dir,
        revoked_at,
        expected_fingerprint,
        true_name_terms,
        approved_nicknames,
    )
    raise PrivacyValidationError(failure[0], failure[1]) from None
