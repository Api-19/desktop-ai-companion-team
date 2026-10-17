import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from pydantic import ValidationError

from companion.contracts.base import ContractModel
from companion.style.errors import StyleValidationError

ModelT = TypeVar("ModelT", bound=ContractModel)


class _StyleIoFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


def _private_path(path: Path, private_data_dir: Path) -> Path:
    try:
        root = private_data_dir.resolve(strict=True)
        resolved = path.resolve(strict=False)
    except OSError:
        raise _StyleIoFailure(
            "unsafe_style_path",
            "style path could not be validated",
        ) from None
    if not resolved.is_relative_to(root):
        raise _StyleIoFailure(
            "unsafe_style_path",
            "style path must stay inside private data",
        )
    return resolved


def _read_private_model(
    model_type: type[ModelT],
    source: Path,
    private_data_dir: Path,
) -> ModelT:
    resolved = _private_path(source, private_data_dir)
    try:
        payload = resolved.read_bytes()
        return model_type.model_validate_json(payload)
    except (OSError, UnicodeError, ValidationError, ValueError):
        raise _StyleIoFailure(
            "invalid_private_model",
            "private model is invalid",
        ) from None


def read_private_model(
    model_type: type[ModelT],
    source: Path,
    private_data_dir: Path,
) -> ModelT:
    try:
        return _read_private_model(model_type, source, private_data_dir)
    except _StyleIoFailure as error:
        failure = (error.code, error.detail)
    del model_type, source, private_data_dir
    raise StyleValidationError(failure[0], failure[1]) from None


def _write_private_model(
    model: ContractModel,
    destination: Path,
    private_data_dir: Path,
) -> Path:
    resolved = _private_path(destination, private_data_dir)
    temporary: Path | None = None
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            dir=resolved.parent,
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(model.model_dump_json(indent=2, warnings=False).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        verified = type(model).model_validate_json(temporary.read_bytes())
        if verified != model:
            raise ValueError("private model readback mismatch")
        os.replace(temporary, resolved)
        return resolved
    except (OSError, UnicodeError, ValidationError, ValueError):
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise _StyleIoFailure(
            "private_write_failed",
            "private model write failed",
        ) from None


def write_private_model(
    model: ContractModel,
    destination: Path,
    private_data_dir: Path,
) -> Path:
    try:
        return _write_private_model(model, destination, private_data_dir)
    except _StyleIoFailure as error:
        failure = (error.code, error.detail)
    del model, destination, private_data_dir
    raise StyleValidationError(failure[0], failure[1]) from None


def _artifact_side_path(destination: Path, purpose: str) -> Path:
    return destination.parent / f".{destination.name}.{purpose}.{uuid4().hex}"


def _write_private_models_atomically(
    artifacts: list[tuple[ContractModel, Path]],
    private_data_dir: Path,
) -> None:
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        resolved = [
            (model, _private_path(destination, private_data_dir))
            for model, destination in artifacts
        ]
        if len({destination for _, destination in resolved}) != len(resolved):
            raise ValueError("private artifact destinations must be distinct")
        for model, destination in resolved:
            stage = _artifact_side_path(destination, "staged")
            write_private_model(model, stage, private_data_dir)
            staged.append((stage, destination))
        for _, destination in staged:
            if destination.exists():
                backup = _artifact_side_path(destination, "backup")
                os.replace(destination, backup)
                backups.append((backup, destination))
        for stage, destination in staged:
            os.replace(stage, destination)
            published.append(destination)
    except (OSError, StyleValidationError, ValueError):
        for destination in published:
            with suppress(OSError):
                destination.unlink(missing_ok=True)
        for backup, destination in reversed(backups):
            with suppress(OSError):
                os.replace(backup, destination)
        raise _StyleIoFailure(
            "private_write_failed",
            "private style artifacts could not be published",
        ) from None
    finally:
        for stage, _ in staged:
            with suppress(OSError):
                stage.unlink(missing_ok=True)
        for backup, destination in backups:
            if destination.exists():
                with suppress(OSError):
                    backup.unlink(missing_ok=True)


def write_private_models_atomically(
    artifacts: list[tuple[ContractModel, Path]],
    private_data_dir: Path,
) -> None:
    try:
        _write_private_models_atomically(artifacts, private_data_dir)
        return
    except _StyleIoFailure as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = ("private_write_failed", "private style artifacts could not be published")
    finally:
        artifacts = None  # type: ignore[assignment]
        private_data_dir = None  # type: ignore[assignment]
    raise StyleValidationError(*failure) from None
