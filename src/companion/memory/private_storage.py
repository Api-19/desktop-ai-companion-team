"""Production Memory factory with a POSIX private file boundary."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from companion.memory.errors import MemoryStoreError
from companion.memory.service import (
    MemoryService,
    _sanitize_public_errors,
    _service_with_store,
)
from companion.memory.store import _MemoryStore
from companion.runtime.settings import Settings

_OS_NAME = os.name
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_AUXILIARY_SUFFIXES = ("-journal", "-wal", "-shm")
_CREATE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_CREATE_FLAGS |= getattr(os, "O_CLOEXEC", 0)


def _current_uid() -> int:
    return os.geteuid()


class _PrivateBoundary:
    def __init__(self, private_root: Path) -> None:
        self._root = private_root
        self._directory = private_root / "memory"
        self.database_path = self._directory / "memory.sqlite3"
        self._identity: tuple[int, int] | None = None

    def prepare(self) -> None:
        self._check_root()
        with suppress(FileExistsError):
            os.mkdir(self._directory, _DIRECTORY_MODE)
        _check_directory(self._directory)
        try:
            os.lstat(self.database_path)
        except FileNotFoundError:
            if any(os.path.lexists(path) for path in self._auxiliary_paths()):
                raise ValueError("orphan auxiliary file") from None
            descriptor = os.open(self.database_path, _CREATE_FLAGS, _FILE_MODE)
            os.close(descriptor)
        info = _check_file(self.database_path)
        self._identity = (info.st_dev, info.st_ino)

    def verify(self) -> None:
        try:
            self._check_root()
            _check_directory(self._directory)
            info = _check_file(self.database_path)
            if self._identity != (info.st_dev, info.st_ino):
                raise ValueError("database identity changed")
            self._check_auxiliary_files()
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def _check_root(self) -> None:
        if not self._root.is_absolute() or os.pardir in self._root.parts:
            raise ValueError("private root must be absolute")
        for ancestor in reversed(self._root.parents):
            if not stat.S_ISDIR(os.lstat(ancestor).st_mode):
                raise ValueError("unsafe private root ancestor")
        _check_directory(self._root)

    def _auxiliary_paths(self) -> tuple[Path, ...]:
        name = self.database_path.name
        return tuple(self._directory / f"{name}{suffix}" for suffix in _AUXILIARY_SUFFIXES)

    def _check_auxiliary_files(self) -> None:
        for path in self._auxiliary_paths():
            try:
                _check_file(path)
            except FileNotFoundError:
                continue


def _check_directory(path: Path) -> os.stat_result:
    info = os.lstat(path)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != _current_uid()
        or stat.S_IMODE(info.st_mode) != _DIRECTORY_MODE
    ):
        raise ValueError("unsafe private directory")
    return info


def _check_file(path: Path) -> os.stat_result:
    info = os.lstat(path)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != _current_uid()
        or stat.S_IMODE(info.st_mode) != _FILE_MODE
        or info.st_nlink != 1
    ):
        raise ValueError("unsafe private file")
    return info


@_sanitize_public_errors
def build_private_memory_service(
    settings: Settings, *, clock: Callable[[], datetime]
) -> MemoryService:
    try:
        if _OS_NAME != "posix" or not isinstance(settings, Settings):
            raise ValueError("private memory storage unavailable")
        boundary = _PrivateBoundary(Path(settings.private_data_dir))
        boundary.prepare()
    except Exception:
        raise MemoryStoreError("memory store unavailable") from None
    store = _MemoryStore(boundary.database_path, boundary_check=boundary.verify)
    return _service_with_store(store, clock)
