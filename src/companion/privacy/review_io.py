import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from companion.contracts.base import ContractModel
from companion.privacy.errors import PrivacyValidationError
from companion.privacy.models import PrivacyReviewPackage, ReviewDecision


class _DecisionDocument(ContractModel):
    decisions: list[ReviewDecision]


class _ReviewIoFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


def _private_path(path: Path, private_data_dir: Path) -> Path:
    try:
        private_root = private_data_dir.resolve()
        resolved = path.resolve()
    except OSError:
        raise _ReviewIoFailure(
            "unsafe_review_path",
            "review path could not be validated",
        ) from None
    if not resolved.is_relative_to(private_root):
        raise _ReviewIoFailure(
            "unsafe_review_path",
            "review path must stay inside the private-data directory",
        )
    return resolved


def _write_review_package(
    package: PrivacyReviewPackage,
    destination: Path,
    private_data_dir: Path,
) -> Path:
    try:
        for component in (destination, *destination.parents):
            if component.is_symlink():
                raise _ReviewIoFailure("unsafe_review_path", "review path must not use links")
    except OSError:
        raise _ReviewIoFailure("unsafe_review_path", "review path could not be validated") from None
    resolved = _private_path(destination, private_data_dir)
    temporary: Path | None = None
    try:
        resolved.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=resolved.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(package.model_dump_json(indent=2))
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(resolved)
        return resolved
    except (OSError, UnicodeError):
        raise _ReviewIoFailure(
            "review_write_failed",
            "review package could not be written safely",
        ) from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                raise _ReviewIoFailure(
                    "review_write_failed", "review package cleanup failed"
                ) from None


def write_review_package(
    package: PrivacyReviewPackage,
    *,
    destination: Path,
    private_data_dir: Path,
) -> Path:
    try:
        return _write_review_package(package, destination, private_data_dir)
    except _ReviewIoFailure as error:
        failure = (error.code, error.detail)
    del package, destination, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _read_review_decisions(path: Path, private_data_dir: Path) -> list[ReviewDecision]:
    resolved = _private_path(path, private_data_dir)
    try:
        document = _DecisionDocument.model_validate_json(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError):
        raise _ReviewIoFailure(
            "invalid_review_decisions",
            "review decisions could not be read safely",
        ) from None

    segment_ids = [decision.segment_id for decision in document.decisions]
    if len(segment_ids) != len(set(segment_ids)):
        raise _ReviewIoFailure(
            "invalid_review_decisions",
            "review decisions contain duplicate segment IDs",
        )
    return document.decisions


def read_review_decisions(
    path: Path,
    *,
    private_data_dir: Path,
) -> list[ReviewDecision]:
    try:
        return _read_review_decisions(path, private_data_dir)
    except _ReviewIoFailure as error:
        failure = (error.code, error.detail)
    del path, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None
