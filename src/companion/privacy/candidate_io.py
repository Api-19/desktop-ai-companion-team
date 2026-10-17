import os
import stat
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from pydantic import ValidationError

from companion.privacy.ambiguous_models import CalibrationProfile
from companion.privacy.candidate_models import CandidateCategory, SafeCandidatePackage
from companion.privacy.candidate_rules import (
    allowed_category,
    classification_text,
    hard_rejection_reason,
)
from companion.privacy.candidate_selection import _calibrated_category
from companion.privacy.errors import PrivacyValidationError


class _CandidateIoFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


_SAFE_CANDIDATE_FAILURE_DETAILS = {
    "candidate_verification_failed": "candidate package verification failed",
    "invalid_candidate_terms": "candidate terms are invalid",
}


def _sanitized_candidate_failure(
    error: Exception,
    *,
    fallback_code: str,
) -> tuple[str, str]:
    fallback = (fallback_code, _SAFE_CANDIDATE_FAILURE_DETAILS[fallback_code])
    try:
        code = error.code
        if type(code) is not str:
            return fallback
        detail = _SAFE_CANDIDATE_FAILURE_DETAILS.get(code)
        if detail is None:
            return fallback
        if type(error) is _CandidateIoFailure:
            candidate_detail = error.detail
            if type(candidate_detail) is str and candidate_detail == detail:
                return code, detail
        elif type(error) is PrivacyValidationError:
            args = error.args
            if (
                type(args) is tuple
                and len(args) == 1
                and type(args[0]) is str
                and args[0] == f"{code}: {detail}"
            ):
                return code, detail
    except Exception:
        return fallback
    return fallback


def _absolute_path(path: Path) -> Path:
    try:
        return path.absolute()
    except (AttributeError, OSError):
        raise _CandidateIoFailure(
            "unsafe_candidate_path",
            "candidate path could not be validated",
        ) from None


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        raise _CandidateIoFailure(
            "unsafe_candidate_path",
            "candidate path could not be validated",
        ) from None
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _has_reparse_component(private_root: Path, path: Path, *, include_leaf: bool) -> bool:
    checked_path = path if include_leaf else path.parent
    try:
        relative_parts = checked_path.relative_to(private_root).parts
    except ValueError:
        return True
    component = private_root
    if _is_reparse_point(component):
        return True
    for part in relative_parts:
        component /= part
        if _is_reparse_point(component):
            return True
    return False


def _private_path(path: Path, private_data_dir: Path, *, include_leaf: bool) -> Path:
    private_root = _absolute_path(private_data_dir)
    requested = _absolute_path(path)
    if not requested.is_relative_to(private_root) or _has_reparse_component(
        private_root,
        requested,
        include_leaf=include_leaf,
    ):
        raise _CandidateIoFailure(
            "unsafe_candidate_path",
            "candidate paths must stay inside the private-data directory",
        )
    try:
        resolved_root = private_root.resolve()
        resolved = requested.resolve()
    except OSError:
        raise _CandidateIoFailure(
            "unsafe_candidate_path",
            "candidate path could not be validated",
        ) from None
    if not resolved.is_relative_to(resolved_root):
        raise _CandidateIoFailure(
            "unsafe_candidate_path",
            "candidate paths must stay inside the private-data directory",
        )
    return resolved


def _validated_paths(
    destination: Path,
    unsafe_result: Path,
    private_data_dir: Path,
    *,
    destination_exists: bool,
) -> tuple[Path, Path]:
    resolved_destination = _private_path(
        destination,
        private_data_dir,
        include_leaf=destination_exists,
    )
    resolved_unsafe = _private_path(unsafe_result, private_data_dir, include_leaf=True)
    if resolved_destination == resolved_unsafe or (
        not destination_exists and resolved_destination.exists()
    ):
        raise _CandidateIoFailure(
            "unsafe_candidate_path",
            "candidate paths are invalid",
        )
    if not resolved_unsafe.exists():
        raise _CandidateIoFailure(
            "unsafe_result_missing",
            "unsafe result is missing",
        )
    return resolved_destination, resolved_unsafe


def _remove_temporary(temporary: Path | None) -> bool:
    if temporary is None:
        return True
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _close_descriptor(descriptor: int | None) -> None:
    if descriptor is not None:
        with suppress(OSError):
            os.close(descriptor)


def _temporary_identity(descriptor: int) -> tuple[int, int]:
    try:
        metadata = os.fstat(descriptor)
    except OSError:
        raise _CandidateIoFailure(
            "candidate_temporary_invalid",
            "candidate temporary file could not be verified",
        ) from None
    if not stat.S_ISREG(metadata.st_mode):
        raise _CandidateIoFailure(
            "candidate_temporary_invalid",
            "candidate temporary file could not be verified",
        )
    return metadata.st_dev, metadata.st_ino


def _verify_temporary_identity(temporary: Path, expected: tuple[int, int]) -> None:
    try:
        metadata = os.lstat(temporary)
    except OSError:
        raise _CandidateIoFailure(
            "candidate_temporary_invalid",
            "candidate temporary file could not be verified",
        ) from None
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or bool(attributes & reparse_flag)
        or metadata.st_dev != expected[0]
        or metadata.st_ino != expected[1]
    ):
        raise _CandidateIoFailure(
            "candidate_temporary_invalid",
            "candidate temporary file could not be verified",
        )


def _verify_descriptor_identity(descriptor: int, expected: tuple[int, int]) -> None:
    if _temporary_identity(descriptor) != expected:
        raise _CandidateIoFailure(
            "candidate_temporary_invalid",
            "candidate temporary file could not be verified",
        )


def _normalise_runtime_terms(
    terms: Sequence[str],
    *,
    required: bool,
) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise _CandidateIoFailure(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )
    raw_values = tuple(terms)
    if (required and not raw_values) or any(not isinstance(term, str) for term in raw_values):
        raise _CandidateIoFailure(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )
    values = tuple(dict.fromkeys(term.strip() for term in raw_values))
    if any(not term or hard_rejection_reason(term) is not None for term in values):
        raise _CandidateIoFailure(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )
    return values


def _validated_runtime_terms(
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = _normalise_runtime_terms(true_name_terms, required=True)
    nicknames = _normalise_runtime_terms(approved_nicknames, required=False)
    if set(names) & set(nicknames):
        raise _CandidateIoFailure(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )
    return names, nicknames


def _validate_runtime_content(
    package: SafeCandidatePackage,
    true_name_terms: tuple[str, ...],
    approved_nicknames: tuple[str, ...],
    calibration_profile: CalibrationProfile | None,
    *,
    require_calibration_profile: bool = True,
) -> None:
    for candidate in package.candidates:
        if hard_rejection_reason(candidate.text) is not None or any(
            term in candidate.text for term in true_name_terms
        ):
            raise _CandidateIoFailure(
                "candidate_verification_failed",
                "candidate package verification failed",
            )
        if candidate.category is CandidateCategory.CALIBRATED_DAILY:
            existing_category = allowed_category(
                classification_text(candidate.text, approved_nicknames)
            )
            if existing_category is not None:
                raise _CandidateIoFailure(
                    "candidate_verification_failed",
                    "candidate package verification failed",
                )
            if require_calibration_profile and (
                _calibrated_category(candidate.text, calibration_profile)
                is not CandidateCategory.CALIBRATED_DAILY
            ):
                raise _CandidateIoFailure(
                    "candidate_verification_failed",
                    "candidate package verification failed",
                )
            continue
        category = allowed_category(classification_text(candidate.text, approved_nicknames))
        if category is not candidate.category:
            raise _CandidateIoFailure(
                "candidate_verification_failed",
                "candidate package verification failed",
            )


def _validate_safe_candidate_runtime_material(
    package: SafeCandidatePackage,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> None:
    try:
        validated = SafeCandidatePackage.model_validate(package.model_dump())
    except (AttributeError, ValidationError, ValueError):
        raise _CandidateIoFailure(
            "candidate_verification_failed",
            "candidate package verification failed",
        ) from None
    names, nicknames = _validated_runtime_terms(true_name_terms, approved_nicknames)
    _validate_runtime_content(
        validated,
        names,
        nicknames,
        None,
        require_calibration_profile=False,
    )


def _write_and_readback_temporary(
    descriptor: int,
    package: SafeCandidatePackage,
) -> SafeCandidatePackage:
    try:
        payload = package.model_dump_json(indent=2).encode("utf-8")
        with os.fdopen(descriptor, "w+b", closefd=False) as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(descriptor)
            temporary_file.seek(0)
            readback = temporary_file.read()
    except (OSError, UnicodeError, ValueError):
        raise _CandidateIoFailure(
            "candidate_replace_failed",
            "candidate package could not be written safely",
        ) from None
    try:
        return SafeCandidatePackage.model_validate_json(readback)
    except (UnicodeError, ValidationError, ValueError):
        raise _CandidateIoFailure(
            "candidate_verification_failed",
            "candidate package verification failed",
        ) from None


def _open_and_verify_formal_destination(
    destination: Path,
    expected_identity: tuple[int, int],
    package: SafeCandidatePackage,
    true_name_terms: tuple[str, ...],
    approved_nicknames: tuple[str, ...],
    calibration_profile: CalibrationProfile | None,
) -> int:
    _verify_temporary_identity(destination, expected_identity)
    descriptor: int | None = None
    try:
        descriptor = os.open(destination, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        _verify_descriptor_identity(descriptor, expected_identity)
        with os.fdopen(descriptor, "rb", closefd=False) as formal_file:
            payload = formal_file.read()
        verified = SafeCandidatePackage.model_validate_json(payload)
        if verified != package:
            raise _CandidateIoFailure(
                "candidate_verification_failed",
                "candidate package verification failed",
            )
        _validate_runtime_content(
            verified,
            true_name_terms,
            approved_nicknames,
            calibration_profile,
        )
        _verify_temporary_identity(destination, expected_identity)
        return descriptor
    except _CandidateIoFailure:
        _close_descriptor(descriptor)
        raise
    except (OSError, UnicodeError, ValidationError, ValueError):
        _close_descriptor(descriptor)
        raise _CandidateIoFailure(
            "candidate_verification_failed",
            "candidate package verification failed",
        ) from None


def _quarantine_destination(destination: Path) -> Path:
    quarantine: Path | None = None
    try:
        descriptor, raw_quarantine = tempfile.mkstemp(
            prefix=f".{destination.name}.quarantine-",
            dir=destination.parent,
        )
        os.close(descriptor)
        quarantine = Path(raw_quarantine)
        os.replace(destination, quarantine)
        return quarantine
    except OSError:
        if not _remove_temporary(quarantine):
            raise _CandidateIoFailure(
                "candidate_cleanup_failed",
                "candidate cleanup could not be completed safely",
            ) from None
        raise _CandidateIoFailure(
            "candidate_rollback_failed",
            "candidate rollback could not isolate the formal package",
        ) from None


def _rollback_formal_destination(destination: Path, private_data_dir: Path) -> None:
    try:
        destination = _private_path(destination, private_data_dir, include_leaf=True)
    except _CandidateIoFailure:
        raise _CandidateIoFailure(
            "candidate_rollback_failed",
            "candidate rollback could not isolate the formal package",
        ) from None
    try:
        destination.unlink()
        return
    except FileNotFoundError:
        return
    except OSError:
        quarantine = _quarantine_destination(destination)
        if not _remove_temporary(quarantine):
            raise _CandidateIoFailure(
                "candidate_rollback_failed",
                "candidate rollback could not clean up the isolated package",
            ) from None


def _replace_unsafe_result_with_candidate_package(
    package: SafeCandidatePackage,
    destination: Path,
    unsafe_result: Path,
    private_data_dir: Path,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    calibration_profile: CalibrationProfile | None,
) -> Path:
    temporary: Path | None = None
    descriptor: int | None = None
    formal_descriptor: int | None = None
    temporary_identity: tuple[int, int] | None = None
    formal_published = False
    cleanup_failure_reported = False
    try:
        names, nicknames = _validated_runtime_terms(true_name_terms, approved_nicknames)
        _validate_runtime_content(package, names, nicknames, calibration_profile)
        resolved_destination, resolved_unsafe = _validated_paths(
            destination,
            unsafe_result,
            private_data_dir,
            destination_exists=False,
        )
        try:
            resolved_destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise _CandidateIoFailure(
                "candidate_replace_failed",
                "candidate package could not be written safely",
            ) from None
        resolved_destination, resolved_unsafe = _validated_paths(
            destination,
            unsafe_result,
            private_data_dir,
            destination_exists=False,
        )

        try:
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{resolved_destination.name}.",
                suffix=".tmp",
                dir=resolved_destination.parent,
            )
            temporary = Path(raw_temporary)
            temporary_identity = _temporary_identity(descriptor)
        except _CandidateIoFailure:
            raise
        except OSError:
            raise _CandidateIoFailure(
                "candidate_replace_failed",
                "candidate package could not be written safely",
            ) from None
        _validated_paths(
            destination,
            unsafe_result,
            private_data_dir,
            destination_exists=False,
        )
        assert temporary_identity is not None
        _verify_temporary_identity(temporary, temporary_identity)
        assert descriptor is not None
        _verify_descriptor_identity(descriptor, temporary_identity)

        verified = _write_and_readback_temporary(descriptor, package)
        if verified != package:
            raise _CandidateIoFailure(
                "candidate_verification_failed",
                "candidate package verification failed",
            )
        _validate_runtime_content(verified, names, nicknames, calibration_profile)
        _validated_paths(
            destination,
            unsafe_result,
            private_data_dir,
            destination_exists=False,
        )
        _verify_descriptor_identity(descriptor, temporary_identity)
        _verify_temporary_identity(temporary, temporary_identity)

        try:
            os.link(temporary, resolved_destination)
        except FileExistsError:
            raise _CandidateIoFailure(
                "unsafe_candidate_path",
                "candidate paths are invalid",
            ) from None
        except OSError:
            raise _CandidateIoFailure(
                "candidate_replace_failed",
                "candidate package could not be written safely",
            ) from None
        formal_published = True
        _verify_temporary_identity(resolved_destination, temporary_identity)
        _close_descriptor(descriptor)
        descriptor = None
        if not _remove_temporary(temporary):
            cleanup_failure_reported = True
            _rollback_formal_destination(resolved_destination, private_data_dir)
            formal_published = False
            raise _CandidateIoFailure(
                "candidate_cleanup_failed",
                "candidate temporary file could not be removed safely",
            )
        temporary = None

        resolved_destination, resolved_unsafe = _validated_paths(
            destination,
            unsafe_result,
            private_data_dir,
            destination_exists=True,
        )
        formal_descriptor = _open_and_verify_formal_destination(
            resolved_destination,
            temporary_identity,
            package,
            names,
            nicknames,
            calibration_profile,
        )
        _verify_descriptor_identity(formal_descriptor, temporary_identity)
        _verify_temporary_identity(resolved_destination, temporary_identity)
        resolved_unsafe.unlink()
        return resolved_destination
    except _CandidateIoFailure:
        if formal_published:
            _close_descriptor(formal_descriptor)
            formal_descriptor = None
            _close_descriptor(descriptor)
            descriptor = None
            _rollback_formal_destination(resolved_destination, private_data_dir)
            formal_published = False
        raise
    except OSError:
        if formal_published:
            _close_descriptor(formal_descriptor)
            formal_descriptor = None
            _close_descriptor(descriptor)
            descriptor = None
            _rollback_formal_destination(resolved_destination, private_data_dir)
            formal_published = False
        raise _CandidateIoFailure(
            "candidate_replace_failed",
            "candidate replacement failed",
        ) from None
    finally:
        _close_descriptor(formal_descriptor)
        _close_descriptor(descriptor)
        if (
            temporary is not None
            and not _remove_temporary(temporary)
            and not cleanup_failure_reported
        ):
            raise _CandidateIoFailure(
                "candidate_cleanup_failed",
                "candidate temporary file could not be removed safely",
            )


def replace_unsafe_result_with_candidate_package(
    package: SafeCandidatePackage,
    *,
    destination: Path,
    unsafe_result: Path,
    private_data_dir: Path,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    calibration_profile: CalibrationProfile | None = None,
) -> Path:
    try:
        return _replace_unsafe_result_with_candidate_package(
            package,
            destination,
            unsafe_result,
            private_data_dir,
            true_name_terms,
            approved_nicknames,
            calibration_profile,
        )
    except _CandidateIoFailure as error:
        failure = (error.code, error.detail)
    del (
        package,
        destination,
        unsafe_result,
        private_data_dir,
        true_name_terms,
        approved_nicknames,
        calibration_profile,
    )
    raise PrivacyValidationError(failure[0], failure[1]) from None


def validate_safe_candidate_runtime_material(
    package: SafeCandidatePackage,
    *,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> None:
    try:
        _validate_safe_candidate_runtime_material(
            package,
            true_name_terms,
            approved_nicknames,
        )
        return
    except Exception as error:
        failure = _sanitized_candidate_failure(
            error,
            fallback_code="candidate_verification_failed",
        )
    del package, true_name_terms, approved_nicknames
    raise PrivacyValidationError(failure[0], failure[1]) from None
