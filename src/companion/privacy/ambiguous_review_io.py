import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from pydantic import ValidationError

from companion.privacy.ambiguous_models import (
    AmbiguousDecisionPackage,
    AmbiguousReviewPackage,
)
from companion.privacy.ambiguous_review_html import render_ambiguous_review_html
from companion.privacy.candidate_io import (
    _CandidateIoFailure,
    _private_path,
    _temporary_identity,
    _verify_descriptor_identity,
    _verify_temporary_identity,
)
from companion.privacy.errors import PrivacyValidationError


class _ReviewIoFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


def _review_path(path: Path, private_data_dir: Path, *, include_leaf: bool) -> Path:
    try:
        return _private_path(path, private_data_dir, include_leaf=include_leaf)
    except _CandidateIoFailure as error:
        raise _ReviewIoFailure(
            "unsafe_review_path" if error.code == "unsafe_candidate_path" else error.code,
            "review path must stay inside the private-data directory",
        ) from None


def _review_helper_failure(error: _CandidateIoFailure) -> _ReviewIoFailure:
    code = (
        "unsafe_review_path"
        if error.code == "unsafe_candidate_path"
        else "review_verification_failed"
    )
    detail = (
        "review path must stay inside the private-data directory"
        if code == "unsafe_review_path"
        else "review package verification failed"
    )
    return _ReviewIoFailure(code, detail)


def _publish_text(payload: str, destination: Path, private_data_dir: Path) -> Path:
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        resolved = _review_path(destination, private_data_dir, include_leaf=False)
        if resolved.exists():
            raise _ReviewIoFailure("unsafe_review_path", "review path already exists")
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise _ReviewIoFailure(
                "review_write_failed",
                "review package could not be written safely",
            ) from None
        resolved = _review_path(destination, private_data_dir, include_leaf=False)
        if resolved.exists():
            raise _ReviewIoFailure("unsafe_review_path", "review path already exists")

        try:
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{resolved.name}.",
                suffix=".tmp",
                dir=resolved.parent,
            )
            temporary = Path(raw_temporary)
            identity = _temporary_identity(descriptor)
            _verify_temporary_identity(temporary, identity)
            _verify_descriptor_identity(descriptor, identity)
            with os.fdopen(descriptor, "w+b", closefd=False) as temporary_file:
                encoded = payload.encode("utf-8")
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(descriptor)
                temporary_file.seek(0)
                if temporary_file.read() != encoded:
                    raise _ReviewIoFailure(
                        "review_verification_failed",
                        "review package verification failed",
                    )
            _verify_descriptor_identity(descriptor, identity)
            _verify_temporary_identity(temporary, identity)
            resolved = _review_path(destination, private_data_dir, include_leaf=False)
            os.link(temporary, resolved)
            resolved = _review_path(destination, private_data_dir, include_leaf=True)
            _verify_temporary_identity(resolved, identity)
            return resolved
        except FileExistsError:
            raise _ReviewIoFailure("unsafe_review_path", "review path already exists") from None
        except _CandidateIoFailure as error:
            raise _review_helper_failure(error) from None
        except _ReviewIoFailure:
            raise
        except (OSError, UnicodeError, ValueError):
            raise _ReviewIoFailure(
                "review_write_failed",
                "review package could not be written safely",
            ) from None
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _write_ambiguous_review(
    package: AmbiguousReviewPackage,
    *,
    package_path: Path,
    html_path: Path,
    private_data_dir: Path,
) -> tuple[Path, Path]:
    package_payload = package.model_dump_json(indent=2)
    html_payload = render_ambiguous_review_html(package)
    written_package = _publish_text(package_payload, package_path, private_data_dir)
    try:
        if (
            AmbiguousReviewPackage.model_validate_json(written_package.read_text(encoding="utf-8"))
            != package
        ):
            raise _ReviewIoFailure(
                "review_verification_failed",
                "review package verification failed",
            )
        written_html = _publish_text(html_payload, html_path, private_data_dir)
        if written_html.read_text(encoding="utf-8") != html_payload:
            raise _ReviewIoFailure(
                "review_verification_failed",
                "review package verification failed",
            )
        return written_package, written_html
    except _ReviewIoFailure:
        raise
    except (OSError, UnicodeError, ValidationError, ValueError):
        raise _ReviewIoFailure(
            "review_verification_failed",
            "review package verification failed",
        ) from None


def write_ambiguous_review(
    package: AmbiguousReviewPackage,
    *,
    package_path: Path,
    html_path: Path,
    private_data_dir: Path,
) -> tuple[Path, Path]:
    try:
        return _write_ambiguous_review(
            package,
            package_path=package_path,
            html_path=html_path,
            private_data_dir=private_data_dir,
        )
    except _ReviewIoFailure as error:
        failure = (error.code, error.detail)
    del package, package_path, html_path, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _read_ambiguous_decisions(
    path: Path,
    *,
    package: AmbiguousReviewPackage,
    private_data_dir: Path,
) -> AmbiguousDecisionPackage:
    resolved = _review_path(path, private_data_dir, include_leaf=True)
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _ReviewIoFailure(
            "invalid_review_package",
            "review decisions could not be read safely",
        ) from None
    if not isinstance(document, dict):
        raise _ReviewIoFailure("invalid_review_package", "review decisions are invalid")
    if set(document) != {"package_token", "content_fingerprint", "decisions"}:
        raise _ReviewIoFailure("invalid_review_package", "review decisions are invalid")
    if (
        document.get("package_token") != package.package_token
        or document.get("content_fingerprint") != package.content_fingerprint
    ):
        raise _ReviewIoFailure(
            "stale_review_package",
            "review decisions do not match the review package",
        )
    raw_decisions = document.get("decisions")
    if not isinstance(raw_decisions, list):
        raise _ReviewIoFailure("invalid_review_package", "review decisions are invalid")
    expected_ids = {item.review_item_id for item in package.items}
    actual_ids = [
        item.get("review_item_id") if isinstance(item, dict) else None for item in raw_decisions
    ]
    if len(actual_ids) != len(set(actual_ids)):
        raise _ReviewIoFailure("invalid_review_package", "review decisions are invalid")
    if set(actual_ids) != expected_ids or len(raw_decisions) != package.item_count:
        raise _ReviewIoFailure(
            "incomplete_review_decisions",
            "review decisions must cover every item exactly once",
        )
    try:
        return AmbiguousDecisionPackage.model_validate(
            {
                "package_token": document["package_token"],
                "content_fingerprint": document["content_fingerprint"],
                "item_count": len(raw_decisions),
                "decisions": raw_decisions,
            }
        )
    except (ValidationError, ValueError, TypeError):
        raise _ReviewIoFailure("invalid_review_package", "review decisions are invalid") from None


def read_ambiguous_decisions(
    path: Path,
    *,
    package: AmbiguousReviewPackage,
    private_data_dir: Path,
) -> AmbiguousDecisionPackage:
    try:
        return _read_ambiguous_decisions(
            path,
            package=package,
            private_data_dir=private_data_dir,
        )
    except _ReviewIoFailure as error:
        failure = (error.code, error.detail)
    del path, package, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None
