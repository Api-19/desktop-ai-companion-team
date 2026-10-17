"""Synthetic POSIX file-boundary checks for the private Memory factory."""

from __future__ import annotations

import inspect
import os
import sqlite3
import stat
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType

import pytest
from pydantic import BaseModel

import companion.memory.private_storage as private_storage
import companion.memory.store as memory_store
from companion.contracts import ConsentRecord
from companion.memory import (
    CorrectMemoryRequest,
    CreateMemoryRequest,
    MemoryAuthorizationError,
    MemoryField,
    MemoryService,
    MemoryStoreError,
    build_private_memory_service,
)
from companion.runtime.settings import Settings

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
USER_A = "synthetic-storage-a"
USER_B = "synthetic-storage-b"
VALUE = "synthetic-storage-value"
AUX_SUFFIXES = ("-journal", "-wal", "-shm")

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX private storage only")


def _settings(tmp_path: Path, private_root: Path | None = None) -> Settings:
    return Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=private_root if private_root is not None else tmp_path / "private",
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path / "private"
    root.mkdir()
    root.chmod(0o700)
    return root


def _build(settings: Settings) -> MemoryService:
    return build_private_memory_service(settings, clock=lambda: NOW)


def _paths(root: Path) -> tuple[Path, Path]:
    directory = root / "memory"
    return directory, directory / "memory.sqlite3"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _snapshot(base: Path) -> dict[str, tuple[object, ...]]:
    entries: dict[str, tuple[object, ...]] = {}
    for current, directories, files in os.walk(base, followlinks=False):
        for name in [*directories, *files]:
            path = Path(current) / name
            info = path.lstat()
            key = str(path.relative_to(base))
            if stat.S_ISLNK(info.st_mode):
                entries[key] = ("link", os.readlink(path))
            elif stat.S_ISREG(info.st_mode):
                entries[key] = ("file", info.st_mode, info.st_ino, info.st_nlink, path.read_bytes())
            else:
                entries[key] = ("other", info.st_mode, info.st_ino)
    return entries


def _grant(service: MemoryService, user_id: str, fields: list[str] | None = None) -> None:
    service.grant_consent(
        ConsentRecord(
            user_id=user_id,
            granted=True,
            granted_at=NOW,
            allowed_fields=fields if fields is not None else [f.value for f in MemoryField],
        )
    )


def _create(service: MemoryService, user_id: str, field: MemoryField = MemoryField.PREFERENCE):
    return service.create_memory(
        CreateMemoryRequest(
            user_id=user_id,
            field=field,
            value=f"{VALUE}-{field.value}",
            source_timestamp=NOW,
            explicitly_requested=True,
        )
    )


def _raw_counts(database: Path) -> tuple[int, int]:
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
        memories = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
        consents = connection.execute("SELECT COUNT(*) FROM consents").fetchone()
    assert memories is not None and consents is not None
    return int(memories[0]), int(consents[0])


def _memory_traceback_material(traceback: TracebackType | None) -> str:
    material: list[str] = []
    while traceback is not None:
        frame = traceback.tb_frame
        if "companion/memory" in frame.f_code.co_filename.replace("\\", "/"):
            for value in frame.f_locals.values():
                if isinstance(value, str | Path):
                    material.append(str(value))
                elif isinstance(value, BaseModel):
                    material.append(value.model_dump_json())
        traceback = traceback.tb_next
    return "\n".join(material)


def _assert_fixed_store_error(
    captured: pytest.ExceptionInfo[MemoryStoreError], tmp_path: Path
) -> None:
    assert str(captured.value) == "memory store unavailable"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    material = _memory_traceback_material(captured.value.__traceback__)
    assert "synthetic-storage" not in material
    assert str(tmp_path) not in material
    assert "memory.sqlite3" not in material
    assert "SELECT" not in material


def test_factory_signature_is_settings_and_keyword_clock_only() -> None:
    parameters = inspect.signature(build_private_memory_service).parameters
    assert list(parameters) == ["settings", "clock"]
    assert parameters["settings"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["clock"].kind is inspect.Parameter.KEYWORD_ONLY


def test_factory_creates_private_directory_and_database_at_fixed_path(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    service = _build(_settings(tmp_path))

    directory, database = _paths(root)
    assert isinstance(service, MemoryService)
    assert sorted(path.name for path in root.iterdir()) == ["memory"]
    assert sorted(path.name for path in directory.iterdir()) == ["memory.sqlite3"]
    assert _mode(directory) == 0o700
    assert _mode(database) == 0o600
    assert stat.S_ISREG(database.lstat().st_mode)
    assert database.lstat().st_nlink == 1
    assert _mode(root) == 0o700
    assert _raw_counts(database) == (0, 0)
    assert not (tmp_path / "runtime").exists()


def test_factory_has_no_umask_dependent_post_creation_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private_root(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(os, "chmod", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(os, "fchmod", lambda *args, **kwargs: calls.append(args))
    old_umask = os.umask(0o077)
    try:
        _build(_settings(tmp_path))
    finally:
        os.umask(old_umask)
    assert calls == []
    directory, database = _paths(root)
    assert (_mode(directory), _mode(database)) == (0o700, 0o600)


def test_permissive_umask_cannot_widen_created_artifacts(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    old_umask = os.umask(0)
    try:
        service = _build(_settings(tmp_path))
        _grant(service, USER_A)
        _create(service, USER_A)
    finally:
        os.umask(old_umask)
    directory, database = _paths(root)
    assert (_mode(directory), _mode(database)) == (0o700, 0o600)


def test_rollback_journal_is_private_while_transaction_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private_root(tmp_path)
    service = _build(_settings(tmp_path))
    directory, database = _paths(root)
    observed: list[tuple[str, int, int, int]] = []
    original_connect = sqlite3.connect

    def tracing_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)  # type: ignore[arg-type]

        def trace(statement: str) -> None:
            if statement.strip().upper() == "COMMIT":
                for path in directory.iterdir():
                    info = path.lstat()
                    observed.append(
                        (path.name, stat.S_IMODE(info.st_mode), info.st_uid, info.st_nlink)
                    )

        connection.set_trace_callback(trace)
        return connection

    _grant(service, USER_A)
    monkeypatch.setattr(memory_store.sqlite3, "connect", tracing_connect)
    _create(service, USER_A)

    journal = f"{database.name}-journal"
    assert (journal, 0o600, os.geteuid(), 1) in observed
    assert all(mode == 0o600 and uid == os.geteuid() for _, mode, uid, _ in observed)
    assert sorted(path.name for path in directory.iterdir()) == [database.name]


def test_wal_and_shm_follow_private_mode_and_are_rechecked(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    service = _build(_settings(tmp_path))
    directory, database = _paths(root)
    _grant(service, USER_A)
    # Synthetic external WAL switch: production code never changes the journal mode.
    holder = sqlite3.connect(database)
    try:
        assert holder.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        holder.execute("SELECT COUNT(*) FROM memories").fetchone()
        _create(service, USER_A)
        for suffix in ("-wal", "-shm"):
            sidecar = directory / f"{database.name}{suffix}"
            assert _mode(sidecar) == 0o600
            assert sidecar.lstat().st_nlink == 1
        assert len(service.list_memories(USER_A)) == 1

        shm = directory / f"{database.name}-shm"
        shm.chmod(0o644)
        before = _snapshot(tmp_path)
        with pytest.raises(MemoryStoreError) as captured:
            service.list_memories(USER_A)
        _assert_fixed_store_error(captured, tmp_path)
        assert _snapshot(tmp_path) == before
    finally:
        holder.close()


def test_factory_restart_lifecycle_grant_correct_delete_revoke(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    settings = _settings(tmp_path)
    service = _build(settings)
    _grant(service, USER_A)
    _grant(service, USER_B)
    entry = _create(service, USER_A)
    _create(service, USER_B)

    restarted = _build(settings)
    assert [item.memory_id for item in restarted.list_memories(USER_A)] == [entry.memory_id]
    corrected = restarted.correct_memory(
        CorrectMemoryRequest(
            user_id=USER_A,
            memory_id=entry.memory_id,
            value="synthetic-storage-corrected",
            source_timestamp=NOW + timedelta(seconds=1),
        )
    )
    assert corrected.value == "synthetic-storage-corrected"
    with pytest.raises(MemoryAuthorizationError):
        restarted.correct_memory(
            CorrectMemoryRequest(
                user_id=USER_B,
                memory_id=entry.memory_id,
                value="synthetic-storage-cross-user",
                source_timestamp=NOW,
            )
        )

    restarted = _build(settings)
    retrieved = restarted.retrieve_memories(USER_A, "synthetic-storage-corrected")
    assert [item.value for item in retrieved] == ["synthetic-storage-corrected"]
    assert restarted.delete_memory(USER_A, entry.memory_id).deleted_count == 1

    restarted = _build(settings)
    assert restarted.list_memories(USER_A) == ()
    assert restarted.retrieve_memories(USER_A, "synthetic-storage") == ()
    assert len(restarted.list_memories(USER_B)) == 1
    assert restarted.revoke_consent(USER_B).deleted_count == 1

    restarted = _build(settings)
    assert restarted.list_memories(USER_B) == ()
    assert restarted.retrieve_memories(USER_B, "synthetic-storage") == ()
    with pytest.raises(MemoryAuthorizationError):
        _create(restarted, USER_B)
    _directory, database = _paths(root)
    assert _raw_counts(database) == (0, 2)
    assert VALUE.encode() not in database.read_bytes()
    assert b"synthetic-storage-corrected" not in database.read_bytes()


def test_factory_narrowing_consent_removes_only_dropped_fields(tmp_path: Path) -> None:
    _private_root(tmp_path)
    settings = _settings(tmp_path)
    service = _build(settings)
    _grant(service, USER_A)
    _grant(service, USER_B)
    for user in (USER_A, USER_B):
        _create(service, user, MemoryField.PREFERENCE)
        _create(service, user, MemoryField.EXPLICIT_NOTE)

    _grant(service, USER_A, [MemoryField.PREFERENCE.value])
    restarted = _build(settings)
    assert {item.field for item in restarted.list_memories(USER_A)} == {"preference"}
    assert len(restarted.list_memories(USER_B)) == 2
    with pytest.raises(MemoryAuthorizationError):
        _create(restarted, USER_A, MemoryField.EXPLICIT_NOTE)


def test_factory_without_consent_writes_zero_memories(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    service = _build(_settings(tmp_path))
    _directory, database = _paths(root)
    before = database.read_bytes()

    with pytest.raises(MemoryAuthorizationError):
        _create(service, USER_A)

    assert service.list_memories(USER_A) == ()
    assert service.retrieve_memories(USER_A, VALUE) == ()
    assert service.get_consent(USER_A) is None
    assert _raw_counts(database) == (0, 0)
    assert database.read_bytes() == before


def _existing_store(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = _private_root(tmp_path)
    service = _build(_settings(tmp_path))
    _grant(service, USER_A)
    directory, database = _paths(root)
    return root, directory, database


def _safe_target_dir(tmp_path: Path) -> Path:
    target = tmp_path / "target-dir"
    target.mkdir()
    target.chmod(0o700)
    return target


def _safe_target_file(tmp_path: Path) -> Path:
    target = tmp_path / "target-file"
    target.write_bytes(b"synthetic-target-bytes")
    target.chmod(0o600)
    return target


def _missing_root(tmp_path: Path) -> Settings:
    return _settings(tmp_path)


def _dangling_root(tmp_path: Path) -> Settings:
    (tmp_path / "private").symlink_to(tmp_path / "missing-target")
    return _settings(tmp_path)


def _root_symlink(tmp_path: Path) -> Settings:
    (tmp_path / "private").symlink_to(_safe_target_dir(tmp_path))
    return _settings(tmp_path)


def _ancestor_symlink(tmp_path: Path) -> Settings:
    real = tmp_path / "real-parent"
    real.mkdir()
    (real / "private").mkdir()
    (real / "private").chmod(0o700)
    (tmp_path / "alias-parent").symlink_to(real)
    return _settings(tmp_path, tmp_path / "alias-parent" / "private")


def _dangling_ancestor(tmp_path: Path) -> Settings:
    (tmp_path / "alias-parent").symlink_to(tmp_path / "missing-parent")
    return _settings(tmp_path, tmp_path / "alias-parent" / "private")


def _dotdot_root(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    (tmp_path / "other").mkdir()
    return _settings(tmp_path, tmp_path / "other" / ".." / root.name)


def _relative_root(tmp_path: Path) -> Settings:
    return _settings(tmp_path, Path("synthetic-relative-private"))


def _root_mode(mode: int) -> Callable[[Path], Settings]:
    def setup(tmp_path: Path) -> Settings:
        _private_root(tmp_path).chmod(mode)
        return _settings(tmp_path)

    return setup


def _root_is_file(tmp_path: Path) -> Settings:
    (tmp_path / "private").write_bytes(b"synthetic")
    return _settings(tmp_path)


def _memory_dir_symlink(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    (root / "memory").symlink_to(_safe_target_dir(tmp_path))
    return _settings(tmp_path)


def _memory_dir_dangling(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    (root / "memory").symlink_to(tmp_path / "missing-target")
    return _settings(tmp_path)


def _memory_dir_mode(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    (root / "memory").mkdir()
    (root / "memory").chmod(0o755)
    return _settings(tmp_path)


def _memory_dir_is_file(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    (root / "memory").write_bytes(b"synthetic")
    (root / "memory").chmod(0o600)
    return _settings(tmp_path)


def _db_symlink(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    directory, database = _paths(root)
    directory.mkdir(mode=0o700)
    database.symlink_to(_safe_target_file(tmp_path))
    return _settings(tmp_path)


def _db_dangling(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    directory, database = _paths(root)
    directory.mkdir(mode=0o700)
    database.symlink_to(tmp_path / "missing-target")
    return _settings(tmp_path)


def _db_mode(tmp_path: Path) -> Settings:
    _root, _directory, database = _existing_store(tmp_path)
    database.chmod(0o644)
    return _settings(tmp_path)


def _db_hardlink(tmp_path: Path) -> Settings:
    _root, _directory, database = _existing_store(tmp_path)
    os.link(database, tmp_path / "alias-db")
    return _settings(tmp_path)


def _db_is_directory(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    directory, database = _paths(root)
    directory.mkdir(mode=0o700)
    database.mkdir(mode=0o700)
    return _settings(tmp_path)


def _db_is_fifo(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    directory, database = _paths(root)
    directory.mkdir(mode=0o700)
    os.mkfifo(database, 0o600)
    return _settings(tmp_path)


def _aux(suffix: str, kind: str) -> Callable[[Path], Settings]:
    def setup(tmp_path: Path) -> Settings:
        _root, _directory, database = _existing_store(tmp_path)
        sidecar = database.with_name(f"{database.name}{suffix}")
        if kind == "symlink":
            sidecar.symlink_to(_safe_target_file(tmp_path))
        elif kind == "dangling":
            sidecar.symlink_to(tmp_path / "missing-target")
        elif kind == "hardlink":
            os.link(_safe_target_file(tmp_path), sidecar)
        elif kind == "mode":
            sidecar.write_bytes(b"")
            sidecar.chmod(0o644)
        else:
            sidecar.mkdir(mode=0o700)
        return _settings(tmp_path)

    return setup


def _orphan_journal(tmp_path: Path) -> Settings:
    root = _private_root(tmp_path)
    directory, database = _paths(root)
    directory.mkdir(mode=0o700)
    sidecar = database.with_name(f"{database.name}-journal")
    sidecar.write_bytes(b"synthetic-journal")
    sidecar.chmod(0o600)
    return _settings(tmp_path)


REJECTED_LAYOUTS: dict[str, Callable[[Path], Settings]] = {
    "missing-root": _missing_root,
    "dangling-root": _dangling_root,
    "root-symlink": _root_symlink,
    "ancestor-symlink": _ancestor_symlink,
    "dangling-ancestor": _dangling_ancestor,
    "dotdot-root": _dotdot_root,
    "relative-root": _relative_root,
    "root-0755": _root_mode(0o755),
    "root-0750": _root_mode(0o750),
    "root-file": _root_is_file,
    "memory-dir-symlink": _memory_dir_symlink,
    "memory-dir-dangling": _memory_dir_dangling,
    "memory-dir-0755": _memory_dir_mode,
    "memory-dir-file": _memory_dir_is_file,
    "db-symlink": _db_symlink,
    "db-dangling": _db_dangling,
    "db-0644": _db_mode,
    "db-hardlink": _db_hardlink,
    "db-directory": _db_is_directory,
    "db-fifo": _db_is_fifo,
    "orphan-journal": _orphan_journal,
    **{
        f"aux{suffix}-{kind}": _aux(suffix, kind)
        for suffix in AUX_SUFFIXES
        for kind in ("symlink", "dangling", "hardlink", "mode", "directory")
    },
}


@pytest.mark.parametrize("layout", sorted(REJECTED_LAYOUTS))
def test_unsafe_layout_is_rejected_before_initialization_without_changes(
    tmp_path: Path, layout: str
) -> None:
    settings = REJECTED_LAYOUTS[layout](tmp_path)
    before = _snapshot(tmp_path)

    with pytest.raises(MemoryStoreError) as captured:
        _build(settings)

    _assert_fixed_store_error(captured, tmp_path)
    assert _snapshot(tmp_path) == before


def test_wrong_owner_is_rejected_before_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _private_root(tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(private_storage, "_current_uid", lambda: os.geteuid() + 1)

    with pytest.raises(MemoryStoreError) as captured:
        _build(_settings(tmp_path))

    _assert_fixed_store_error(captured, tmp_path)
    assert _snapshot(tmp_path) == before


def test_unsupported_platform_is_fixed_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _private_root(tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(private_storage, "_OS_NAME", "nt")

    with pytest.raises(MemoryStoreError) as captured:
        _build(_settings(tmp_path))

    _assert_fixed_store_error(captured, tmp_path)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("value", [None, "synthetic-storage-root", object()])
def test_non_settings_input_is_rejected(tmp_path: Path, value: object) -> None:
    with pytest.raises(MemoryStoreError) as captured:
        build_private_memory_service(value, clock=lambda: NOW)  # type: ignore[arg-type]
    _assert_fixed_store_error(captured, tmp_path)


def _drift_db_mode(tmp_path: Path, root: Path) -> None:
    _paths(root)[1].chmod(0o644)


def _drift_db_swapped(tmp_path: Path, root: Path) -> None:
    _directory, database = _paths(root)
    replacement = database.with_name("replacement")
    fd = os.open(replacement, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(database.read_bytes())
    os.replace(replacement, database)


def _drift_db_hardlink(tmp_path: Path, root: Path) -> None:
    os.link(_paths(root)[1], tmp_path / "alias-db")


def _drift_db_removed(tmp_path: Path, root: Path) -> None:
    _paths(root)[1].unlink()


def _drift_db_symlink(tmp_path: Path, root: Path) -> None:
    _directory, database = _paths(root)
    os.replace(database, tmp_path / "moved-db")
    database.symlink_to(tmp_path / "moved-db")


def _drift_memory_dir_symlink(tmp_path: Path, root: Path) -> None:
    directory, _database = _paths(root)
    os.replace(directory, tmp_path / "moved-memory")
    directory.symlink_to(tmp_path / "moved-memory")


def _drift_memory_dir_mode(tmp_path: Path, root: Path) -> None:
    _paths(root)[0].chmod(0o755)


def _drift_root_mode(tmp_path: Path, root: Path) -> None:
    root.chmod(0o755)


def _drift_journal_symlink(tmp_path: Path, root: Path) -> None:
    _directory, database = _paths(root)
    database.with_name(f"{database.name}-journal").symlink_to(tmp_path / "missing-target")


def _drift_wal_mode(tmp_path: Path, root: Path) -> None:
    _directory, database = _paths(root)
    sidecar = database.with_name(f"{database.name}-wal")
    sidecar.write_bytes(b"")
    sidecar.chmod(0o644)


DRIFTS: dict[str, Callable[[Path, Path], None]] = {
    "db-mode": _drift_db_mode,
    "db-swapped": _drift_db_swapped,
    "db-hardlink": _drift_db_hardlink,
    "db-removed": _drift_db_removed,
    "db-symlink": _drift_db_symlink,
    "memory-dir-symlink": _drift_memory_dir_symlink,
    "memory-dir-mode": _drift_memory_dir_mode,
    "root-mode": _drift_root_mode,
    "journal-symlink": _drift_journal_symlink,
    "wal-mode": _drift_wal_mode,
}


@pytest.mark.parametrize("drift", sorted(DRIFTS))
@pytest.mark.parametrize("operation", ["read", "write", "revoke"])
def test_boundary_drift_after_construction_fails_next_operation_closed(
    tmp_path: Path, drift: str, operation: str
) -> None:
    root = _private_root(tmp_path)
    service = _build(_settings(tmp_path))
    _grant(service, USER_A)
    _create(service, USER_A)
    DRIFTS[drift](tmp_path, root)
    before = _snapshot(tmp_path)
    actions: dict[str, Callable[[], object]] = {
        "read": lambda: service.list_memories(USER_A),
        "write": lambda: _create(service, USER_A, MemoryField.EXPLICIT_NOTE),
        "revoke": lambda: service.revoke_consent(USER_A),
    }

    with pytest.raises(MemoryStoreError) as captured:
        actions[operation]()

    _assert_fixed_store_error(captured, tmp_path)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("drift", sorted(DRIFTS))
def test_drift_is_rejected_before_sqlite_opens_and_opens_never_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    root = _private_root(tmp_path)
    service = _build(_settings(tmp_path))
    targets: list[object] = []
    original_connect = sqlite3.connect

    def recording_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        targets.append((args, kwargs))
        return original_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(memory_store.sqlite3, "connect", recording_connect)
    _grant(service, USER_A)
    assert targets
    for args, kwargs in targets:  # type: ignore[misc]
        assert str(args[0]).endswith("memory.sqlite3?mode=rw")
        assert kwargs == {"uri": True}
    targets.clear()

    DRIFTS[drift](tmp_path, root)
    with pytest.raises(MemoryStoreError):
        service.list_memories(USER_A)
    assert targets == []


def test_post_connect_identity_recheck_rejects_swap_during_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private_root(tmp_path)
    service = _build(_settings(tmp_path))
    _grant(service, USER_A)
    _create(service, USER_A)
    original_connect = sqlite3.connect

    def swapping_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)  # type: ignore[arg-type]
        _drift_db_swapped(tmp_path, root)
        return connection

    # A read is used because SQLite itself rejects writes to a moved database.
    monkeypatch.setattr(memory_store.sqlite3, "connect", swapping_connect)
    with pytest.raises(MemoryStoreError) as captured:
        service.list_memories(USER_A)
    _assert_fixed_store_error(captured, tmp_path)
    monkeypatch.undo()
    with pytest.raises(MemoryStoreError):
        service.list_memories(USER_A)


def test_legacy_constructor_path_is_unchanged(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "synthetic-legacy.sqlite3"
    service = MemoryService(database, clock=lambda: NOW)
    _grant(service, USER_A)
    _create(service, USER_A)
    assert len(MemoryService(database, clock=lambda: NOW).list_memories(USER_A)) == 1
