"""Private safe candidate seal used by explicit Admin retain and delete actions.

The service does not authenticate an owner or grant Phase 1 consent.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from companion.admin.config import (
    AdminConfig,
    _safe_ancestry,
    _safe_private_directory,
    _valid_owner_id,
)
from companion.admin.import_source import _batch_identity, _FixedImportBatchSource
from companion.privacy import SafeCandidatePackage
from companion.runtime.settings import Settings

SEAL_RELATIVE_PATH = Path("admin/style-source/current.json")
MAX_SEAL_BYTES = 16 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class StyleSourceSealError(RuntimeError):
    """Fixed, non-sensitive seal failure."""


@dataclass(frozen=True)
class SealedStyleSource:
    owner_id: str
    batch_digest: str
    package: SafeCandidatePackage


def _fail() -> None:
    raise StyleSourceSealError("style source unavailable")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


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


class StyleSourceSealStore:
    """Read and write one fixed private candidate package, without authenticating callers."""

    def __init__(self, settings: Settings, config: AdminConfig) -> None:
        if (
            os.name != "posix"
            or type(settings) is not Settings
            or type(config) is not AdminConfig
            or not _valid_owner_id(config.owner_id)
            or type(config.import_format) is not str
            or config.import_format not in {"ciphertalk_json", "standard_json"}
        ):
            _fail()
        try:
            self._root = settings.private_data_dir
            if not isinstance(self._root, Path):
                _fail()
            self._owner_id = config.owner_id
            self._source = _FixedImportBatchSource(self._root, config.import_format)
            self._path = self._root / SEAL_RELATIVE_PATH
            self._pending_path = self._path.parent / ".current-pending"
            self._directory(create=False)
        except Exception:
            raise StyleSourceSealError("style source unavailable") from None

    def _directory(self, *, create: bool) -> Path | None:
        root = self._root
        admin = root / "admin"
        folder = self._path.parent
        if (
            not _safe_ancestry(root)
            or not _safe_private_directory(root)
            or not _safe_private_directory(admin)
        ):
            _fail()
        if create:
            try:
                folder.mkdir(mode=0o700)
            except FileExistsError:
                pass
            except OSError:
                _fail()
        try:
            folder.lstat()
        except FileNotFoundError:
            return None
        except OSError:
            _fail()
        if not _safe_private_directory(folder):
            _fail()
        return folder

    def _current_digest(self) -> str:
        return _batch_identity(self._source.load())[1]

    def _pending_exists(self) -> bool:
        if self._directory(create=False) is None:
            return False
        try:
            info = self._pending_path.lstat()
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o777 != 0o600
            or info.st_nlink != 1
            or info.st_size != 0
        ):
            _fail()
        return True

    def _fsync_directory(self) -> None:
        descriptor = os.open(self._path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _begin_pending(self) -> None:
        self._directory(create=True)
        descriptor = os.open(
            self._pending_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._fsync_directory()

    def _end_pending(self) -> None:
        if not self._pending_exists():
            _fail()
        self._fsync_directory()
        self._pending_path.unlink()

    def _read_path(self) -> bytes | None:
        if self._directory(create=False) is None:
            return None
        path = self._path
        try:
            before = path.lstat()
        except FileNotFoundError:
            return None
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o777 != 0o600
            or before.st_nlink != 1
            or before.st_size > MAX_SEAL_BYTES
        ):
            _fail()
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(descriptor)
            if _identity(opened) != _identity(before):
                _fail()
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(MAX_SEAL_BYTES + 1)
            after_read = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
        if (
            len(raw) > MAX_SEAL_BYTES
            or _identity(after_read) != _identity(before)
            or _identity(after_path) != _identity(before)
        ):
            _fail()
        return raw

    @staticmethod
    def _parse(raw: bytes) -> SealedStyleSource:
        if type(raw) is not bytes or len(raw) > MAX_SEAL_BYTES:
            _fail()
        value = json.loads(raw)
        if (
            type(value) is not dict
            or set(value) != {"owner_id", "batch_digest", "package"}
            or not _valid_owner_id(value["owner_id"])
            or type(value["batch_digest"]) is not str
            or _DIGEST.fullmatch(value["batch_digest"]) is None
            or type(value["package"]) is not dict
            or _canonical(value) != raw
        ):
            _fail()
        package = SafeCandidatePackage.model_validate_json(
            _canonical(value["package"]), strict=True
        )
        if package.model_dump(mode="json", warnings="error") != value["package"]:
            _fail()
        return SealedStyleSource(value["owner_id"], value["batch_digest"], package)

    @staticmethod
    def _serialize(owner_id: str, batch_digest: str, package: SafeCandidatePackage) -> bytes:
        if (
            not _valid_owner_id(owner_id)
            or type(batch_digest) is not str
            or _DIGEST.fullmatch(batch_digest) is None
            or type(package) is not SafeCandidatePackage
        ):
            _fail()
        verified = SafeCandidatePackage.model_validate(
            package.model_dump(mode="python", warnings="error"), strict=True
        )
        raw = _canonical(
            {
                "owner_id": owner_id,
                "batch_digest": batch_digest,
                "package": verified.model_dump(mode="json", warnings="error"),
            }
        )
        if len(raw) > MAX_SEAL_BYTES:
            _fail()
        return raw

    def _atomic_write(self, raw: bytes) -> None:
        folder = self._directory(create=True)
        if folder is None:
            _fail()
        descriptor, temporary = tempfile.mkstemp(prefix=".current-", dir=folder)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, self._path)
            self._fsync_directory()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def _restore(self, previous: bytes | None) -> bool:
        try:
            if previous is None:
                self._path.unlink()
            else:
                self._atomic_write(previous)
            if self._read_path() != previous:
                _fail()
            return True
        except Exception:
            with suppress(OSError):
                self._path.unlink(missing_ok=True)
            try:
                self._path.lstat()
            except FileNotFoundError:
                return True
            return False

    def seal(
        self, owner_id: str, batch_digest: str, package: SafeCandidatePackage
    ) -> SealedStyleSource:
        try:
            if owner_id != self._owner_id:
                _fail()
            raw = self._serialize(owner_id, batch_digest, package)
            current = self._current_digest()
            if current != batch_digest:
                _fail()
            if self._pending_exists():
                _fail()
            previous = self._read_path()
            if previous is not None:
                existing = self._parse(previous)
                if existing.owner_id != owner_id or existing.batch_digest != current:
                    _fail()
            self._begin_pending()
            try:
                self._atomic_write(raw)
                published = self._read_path()
                after = self._current_digest()
                if published != raw or after != current:
                    _fail()
                result = self._parse(published)
                if result.owner_id != owner_id or result.batch_digest != current:
                    _fail()
                self._end_pending()
                return result
            except Exception:
                if self._restore(previous):
                    self._end_pending()
                raise
        except Exception:
            raise StyleSourceSealError("style source unavailable") from None

    def read(self, owner_id: str) -> SealedStyleSource | None:
        try:
            if owner_id != self._owner_id:
                _fail()
            if self._pending_exists():
                _fail()
            before = self._current_digest()
            raw = self._read_path()
            after = self._current_digest()
            if before != after or self._pending_exists():
                _fail()
            if raw is None:
                return None
            result = self._parse(raw)
            if result.owner_id != owner_id or result.batch_digest != before:
                _fail()
            return result
        except Exception:
            raise StyleSourceSealError("style source unavailable") from None

    def exists_for_owner(self, owner_id: str) -> bool:
        """Check a private seal without consulting the original import source."""
        try:
            if owner_id != self._owner_id or self._pending_exists():
                _fail()
            raw = self._read_path()
            if raw is None:
                return False
            if self._parse(raw).owner_id != owner_id:
                _fail()
            return True
        except Exception:
            raise StyleSourceSealError("style source unavailable") from None

    def delete(self, owner_id: str) -> None:
        try:
            if owner_id != self._owner_id:
                _fail()
            if self._pending_exists():
                _fail()
            raw = self._read_path()
            if raw is None:
                return
            result = self._parse(raw)
            if result.owner_id != owner_id:
                _fail()
            self._path.unlink()
            self._fsync_directory()
            if self._read_path() is not None:
                _fail()
        except Exception:
            raise StyleSourceSealError("style source unavailable") from None
