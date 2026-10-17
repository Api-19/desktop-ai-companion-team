"""Private, current-package owner decisions for a future trusted Admin caller.

The caller authenticates the owner and confirms the source mapping. This store does
neither; a package or an owner string alone is never an authentication credential.
"""

import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from companion.privacy.candidate_models import SafeCandidatePackage
from companion.privacy.errors import PrivacyValidationError
from companion.runtime.settings import Settings

APPROVAL_RELATIVE_PATH = Path("privacy/style-approvals/current.json")
_MAX_FILE_BYTES = 16 * 1024 * 1024
_ERROR = ("style_approval_unavailable", "style approval is unavailable")
_REVISION_PATTERN = re.compile(r"v1-[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ApprovalWitness:
    package_digest: str
    approved_indices: tuple[int, ...]
    revision: str


class _InvalidApproval(Exception):
    pass


def _fail() -> None:
    raise _InvalidApproval


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validated_package(package: SafeCandidatePackage) -> SafeCandidatePackage:
    if type(package) is not SafeCandidatePackage:
        _fail()
    return SafeCandidatePackage.model_validate(package.model_dump(mode="json", warnings=False))


def _valid_owner(owner_id: str) -> bool:
    return (
        type(owner_id) is str
        and 1 <= len(owner_id) <= 120
        and owner_id == owner_id.strip()
        and owner_id.isprintable()
    )


def _identity(owner_id: str, package: SafeCandidatePackage) -> dict:
    if not _valid_owner(owner_id):
        _fail()
    verified = _validated_package(package)
    return {
        "version": 1,
        "owner_id": owner_id,
        "package_digest": _digest(verified.model_dump(mode="json", warnings=False)),
        "items": [
            {
                "index": index,
                "category": candidate.category.value,
                "digest": _digest(candidate.model_dump(mode="json", warnings=False)),
                "state": "pending",
            }
            for index, candidate in enumerate(verified.candidates)
        ],
    }


def _directory(path: Path, *, create: bool) -> None:
    try:
        if create:
            path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
        or metadata.st_mode & 0o700 != 0o700
    ):
        _fail()


def _safe_paths(root: Path, *, create: bool) -> Path:
    if not isinstance(root, Path) or not root.is_absolute() or ".." in root.parts:
        _fail()
    for ancestor in root.parents:
        metadata = os.lstat(ancestor)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail()
    _directory(root, create=False)
    first = root / "privacy"
    second = first / "style-approvals"
    if create:
        _directory(first, create=True)
        _directory(second, create=True)
    else:
        if not first.exists() and not first.is_symlink():
            return second / "current.json"
        _directory(first, create=False)
        if not second.exists() and not second.is_symlink():
            return second / "current.json"
        _directory(second, create=False)
    return second / "current.json"


def _file_identity(metadata: os.stat_result) -> tuple[int, int]:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_mode & 0o600 != 0o600
        or metadata.st_size > _MAX_FILE_BYTES
    ):
        _fail()
    return metadata.st_dev, metadata.st_ino


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def _read_sidecar(path: Path, expected: dict) -> tuple[dict, bytes] | None:
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return None
    identity = _file_identity(before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        if _file_identity(os.fstat(handle.fileno())) != identity:
            _fail()
        raw = handle.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES or _file_identity(os.fstat(handle.fileno())) != identity:
            _fail()
    if _file_identity(os.lstat(path)) != identity:
        _fail()
    record = json.loads(raw, object_pairs_hook=_unique_pairs)
    if type(record) is not dict or set(record) != set(expected) | {"revision"}:
        _fail()
    if (
        type(record["revision"]) is not str
        or _REVISION_PATTERN.fullmatch(record["revision"]) is None
    ):
        _fail()
    if any(record[key] != expected[key] for key in ("version", "owner_id", "package_digest")):
        _fail()
    items = record["items"]
    reference = expected["items"]
    if type(items) is not list or len(items) != len(reference):
        _fail()
    for item, template in zip(items, reference, strict=True):
        if type(item) is not dict or set(item) != set(template):
            _fail()
        if any(
            type(item[key]) is not type(template[key]) or item[key] != template[key]
            for key in ("index", "category", "digest")
        ):
            _fail()
        if type(item["state"]) is not str or item["state"] not in {"pending", "approved"}:
            _fail()
    return record, raw


def _read_owner_for_stale_deletion(path: Path) -> str | None:
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return None
    identity = _file_identity(before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        if _file_identity(os.fstat(handle.fileno())) != identity:
            _fail()
        raw = handle.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES or _file_identity(os.fstat(handle.fileno())) != identity:
            _fail()
    if _file_identity(os.lstat(path)) != identity:
        _fail()
    record = json.loads(raw, object_pairs_hook=_unique_pairs)
    if type(record) is not dict or set(record) not in (
        {
            "version",
            "owner_id",
            "package_digest",
            "items",
        },
        {
            "version",
            "owner_id",
            "package_digest",
            "items",
            "revision",
        },
    ):
        _fail()
    if not _valid_owner(record["owner_id"]) or record["version"] != 1:
        _fail()
    if "revision" in record and (
        type(record["revision"]) is not str
        or _REVISION_PATTERN.fullmatch(record["revision"]) is None
    ):
        _fail()
    return record["owner_id"]


def _atomic_write(path: Path, raw: bytes) -> None:
    if len(raw) > _MAX_FILE_BYTES:
        _fail()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".approval-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _file_identity(os.lstat(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _published_indices(record: dict) -> tuple[int, ...]:
    return tuple(item["index"] for item in record["items"] if item["state"] == "approved")


def _remove_failed_publication(path: Path) -> None:
    path.unlink(missing_ok=True)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    _fail()


class StyleCandidateApprovalStore:
    """Caller-owned service for exact current-package decisions, without authentication."""

    def __init__(self, settings: Settings) -> None:
        if type(settings) is not Settings:
            del self, settings
            raise PrivacyValidationError(*_ERROR) from None
        verified = root = None
        try:
            verified = Settings.model_validate(settings.model_dump())
            root = verified.private_data_dir
            _safe_paths(root, create=False)
        except Exception:
            pass
        else:
            self.private_data_dir = root
            return
        del self, settings, verified, root
        raise PrivacyValidationError(*_ERROR) from None

    def approved_indices(self, owner_id: str, package: SafeCandidatePackage) -> tuple[int, ...]:
        expected = path = result = None
        try:
            expected = _identity(owner_id, package)
            path = _safe_paths(self.private_data_dir, create=False)
            result = _read_sidecar(path, expected)
            return () if result is None else _published_indices(result[0])
        except Exception:
            pass
        del self, owner_id, package, expected, path, result
        raise PrivacyValidationError(*_ERROR) from None

    def approval_witness(
        self, owner_id: str, package: SafeCandidatePackage
    ) -> ApprovalWitness | None:
        expected = path = result = None
        try:
            expected = _identity(owner_id, package)
            path = _safe_paths(self.private_data_dir, create=False)
            result = _read_sidecar(path, expected)
            if result is None:
                return None
            return ApprovalWitness(
                package_digest=expected["package_digest"],
                approved_indices=_published_indices(result[0]),
                revision=result[0]["revision"],
            )
        except Exception:
            pass
        del self, owner_id, package, expected, path, result
        raise PrivacyValidationError(*_ERROR) from None

    def decisions(
        self, owner_id: str, package: SafeCandidatePackage
    ) -> tuple[tuple[int, str], ...]:
        expected = path = result = record = None
        try:
            expected = _identity(owner_id, package)
            path = _safe_paths(self.private_data_dir, create=False)
            result = _read_sidecar(path, expected)
            record = expected if result is None else result[0]
            return tuple((item["index"], item["state"]) for item in record["items"])
        except Exception:
            pass
        del self, owner_id, package, expected, path, result, record
        raise PrivacyValidationError(*_ERROR) from None

    def set_approved(
        self, owner_id: str, package: SafeCandidatePackage, *, index: int, approved: bool
    ) -> tuple[int, ...]:
        expected = path = previous = record = new_raw = readback = restored = None
        try:
            expected = _identity(owner_id, package)
            if type(index) is not int or index < 0 or index >= len(expected["items"]):
                _fail()
            if type(approved) is not bool:
                _fail()
            path = _safe_paths(self.private_data_dir, create=True)
            previous = _read_sidecar(path, expected)
            record = expected if previous is None else previous[0]
            record["items"][index]["state"] = "approved" if approved else "pending"
            previous_revision = None if previous is None else previous[0]["revision"]
            revision = f"v1-{secrets.token_hex(32)}"
            while revision == previous_revision:
                revision = f"v1-{secrets.token_hex(32)}"
            record["revision"] = revision
            new_raw = _canonical(record)
            try:
                _atomic_write(path, new_raw)
                readback = _read_sidecar(path, expected)
                if readback is None or readback[1] != new_raw:
                    _fail()
            except Exception:
                if previous is None:
                    _remove_failed_publication(path)
                else:
                    try:
                        _atomic_write(path, previous[1])
                        restored = _read_sidecar(path, expected)
                        if restored is None or restored[1] != previous[1]:
                            _fail()
                    except Exception:
                        _remove_failed_publication(path)
                raise
            return _published_indices(record)
        except Exception:
            pass
        del self, owner_id, package, index, approved, expected, path, previous, record, new_raw
        del readback, restored
        raise PrivacyValidationError(*_ERROR) from None

    def delete(self, owner_id: str, package: SafeCandidatePackage) -> None:
        expected = path = existing = None
        try:
            expected = _identity(owner_id, package)
            path = _safe_paths(self.private_data_dir, create=False)
            existing = _read_sidecar(path, expected)
            if existing is not None:
                path.unlink()
            if _read_sidecar(path, expected) is not None:
                _fail()
            return
        except Exception:
            pass
        del self, owner_id, package, expected, path, existing
        raise PrivacyValidationError(*_ERROR) from None

    def delete_stale(self, owner_id: str) -> None:
        """Explicitly remove an old sidecar before reviewing a changed package."""
        path = recorded_owner = None
        try:
            if not _valid_owner(owner_id):
                _fail()
            path = _safe_paths(self.private_data_dir, create=False)
            recorded_owner = _read_owner_for_stale_deletion(path)
            if recorded_owner is None:
                return
            if recorded_owner != owner_id:
                _fail()
            path.unlink()
            if _read_owner_for_stale_deletion(path) is not None:
                _fail()
            return
        except Exception:
            pass
        del self, owner_id, path, recorded_owner
        raise PrivacyValidationError(*_ERROR) from None
