"""Internal source approval adapter and fixed Phase 1 paths for trusted Admin."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import suppress
from pathlib import Path

from companion.admin.approval import _validated_witness
from companion.admin.config import _safe_ancestry, _safe_private_directory, _valid_owner_id
from companion.admin.source_seal import StyleSourceSealStore
from companion.privacy.style_candidate_approval import StyleCandidateApprovalStore
from companion.style import SourceApprovalState


class AdminSourceApprovalGateway:
    def __init__(
        self,
        owner_id: str,
        source_seal: StyleSourceSealStore,
        approval_store: StyleCandidateApprovalStore,
    ) -> None:
        if not _valid_owner_id(owner_id):
            raise ValueError("style source unavailable")
        self._owner_id = owner_id
        self._source_seal = source_seal
        self._approval_store = approval_store

    def current_state(self, user_id: str) -> SourceApprovalState | None:
        if type(user_id) is not str or user_id != self._owner_id:
            return None
        try:
            sealed = self._source_seal.read(user_id)
            if sealed is None or sealed.owner_id != user_id:
                raise ValueError("style source unavailable")
            witness = _validated_witness(
                self._approval_store.approval_witness(user_id, sealed.package),
                len(sealed.package.candidates),
            )
            package_digest = hashlib.sha256(
                json.dumps(
                    sealed.package.model_dump(mode="json", warnings=False),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if (
                witness.package_digest != package_digest
                or self._source_seal.read(user_id) != sealed
            ):
                raise ValueError("style source unavailable")
            return SourceApprovalState(
                owner_id=user_id,
                batch_digest=sealed.batch_digest,
                package_digest=witness.package_digest,
                approved_indices=witness.approved_indices,
                revision=witness.revision,
            )
        except Exception:
            raise RuntimeError("style source unavailable") from None


def phase1_paths(private_root: Path) -> tuple[Path, ...]:
    folder = private_root / "style-framework/phase1"
    return tuple(
        folder / f"{name}.json"
        for name in ("consent", "profile", "evidence", "statistics", "index", "cache")
    )


def phase5_paths(private_root: Path) -> tuple[Path, ...]:
    folder = private_root / "style-framework/phase5"
    return tuple(folder / f"{name}.json" for name in ("evidence", "index", "cache", "version"))


def _validate_paths(
    private_root: Path, phase: str, paths: tuple[Path, ...], *, create: bool
) -> None:
    try:
        if not _safe_ancestry(private_root) or not _safe_private_directory(private_root):
            raise ValueError("style management unavailable")
        for folder in (private_root / "style-framework", private_root / "style-framework" / phase):
            if create:
                with suppress(FileExistsError):
                    folder.mkdir(mode=0o700)
            try:
                folder.lstat()
            except FileNotFoundError:
                return
            if not _safe_private_directory(folder):
                raise ValueError("style management unavailable")
        for path in paths:
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o777 != 0o600
                or info.st_nlink != 1
            ):
                raise ValueError("style management unavailable")
    except Exception:
        raise RuntimeError("style management unavailable") from None


def validate_phase1_paths(private_root: Path, *, create: bool = False) -> None:
    """Keep product-managed Phase 1 files inside private, unlinked directories."""
    _validate_paths(private_root, "phase1", phase1_paths(private_root), create=create)


def validate_phase5_paths(private_root: Path, *, create: bool = False) -> None:
    """Same private-file rules for the four approved-expression Phase 5 files."""
    _validate_paths(private_root, "phase5", phase5_paths(private_root), create=create)
