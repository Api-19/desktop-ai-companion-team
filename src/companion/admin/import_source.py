from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from companion.admin.config import AdminConfigError, _safe_ancestry, _safe_private_directory
from companion.importer import (
    CipherTalkJsonAdapter,
    ImportBatch,
    ImportSeverity,
    StandardJsonAdapter,
)


@dataclass(frozen=True)
class _LoadedBatch:
    batch: ImportBatch
    source_fingerprint: bytes


def _batch_identity(loaded: ImportBatch | _LoadedBatch) -> tuple[ImportBatch, str]:
    if type(loaded) is _LoadedBatch:
        batch = loaded.batch
        source_fingerprint = loaded.source_fingerprint
    else:
        batch = loaded
        source_fingerprint = b""
    if type(batch) is not ImportBatch or type(source_fingerprint) is not bytes:
        raise ValueError("batch unavailable")
    validated = ImportBatch.model_validate(
        batch.model_dump(mode="python", warnings="error"), strict=True
    )
    if not validated.records or any(
        issue.severity is ImportSeverity.ERROR for issue in validated.issues
    ):
        raise ValueError("batch unavailable")
    digest = hashlib.sha256(
        source_fingerprint + validated.model_dump_json(warnings="error").encode()
    ).hexdigest()
    return validated, digest


MAX_IMPORT_BYTES = 64 * 1024 * 1024


class _FixedImportBatchSource:
    def __init__(self, private_root: Path, import_format: str) -> None:
        self._private_root = private_root
        self._import_format = import_format

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def _snapshot(self) -> tuple[tuple[int, ...], bytes]:
        root = self._private_root
        admin = root / "admin"
        folder = admin / "import"
        path = folder / "current.json"
        if (
            not _safe_ancestry(root)
            or not _safe_private_directory(root)
            or not _safe_private_directory(admin)
            or not _safe_private_directory(folder)
        ):
            raise AdminConfigError("import source unavailable")
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o077
            or before.st_nlink != 1
            or before.st_size > MAX_IMPORT_BYTES
        ):
            raise AdminConfigError("import source unavailable")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(descriptor)
            if self._identity(opened) != self._identity(before):
                raise AdminConfigError("import source unavailable")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(MAX_IMPORT_BYTES + 1)
            after_read = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
        if (
            len(raw) > MAX_IMPORT_BYTES
            or self._identity(after_read) != self._identity(before)
            or self._identity(after_path) != self._identity(before)
        ):
            raise AdminConfigError("import source unavailable")
        return self._identity(before), raw

    def load(self) -> _LoadedBatch:
        try:
            before_identity, before_content = self._snapshot()
            path = self._private_root / "admin" / "import" / "current.json"
            adapter = (
                CipherTalkJsonAdapter()
                if self._import_format == "ciphertalk_json"
                else StandardJsonAdapter()
            )
            batch = adapter.parse(path)
            after_identity, after_content = self._snapshot()
            if before_identity != after_identity or before_content != after_content:
                raise AdminConfigError("import source unavailable")
            fingerprint = hashlib.sha256(repr(after_identity).encode() + after_content).digest()
            return _LoadedBatch(batch=batch, source_fingerprint=fingerprint)
        except Exception:
            raise AdminConfigError("import source unavailable") from None
