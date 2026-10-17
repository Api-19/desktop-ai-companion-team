from __future__ import annotations

import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path

from companion.runtime.settings import Settings


class AdminConfigError(RuntimeError):
    """A fixed, non-sensitive local configuration failure."""


@dataclass(frozen=True)
class AdminConfig:
    owner_id: str
    import_format: str | None = None


def _valid_owner_id(value: object) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= 120
        and value == value.strip()
        and value.isprintable()
    )


def _safe_ancestry(path: Path) -> bool:
    if not path.is_absolute():
        return False
    try:
        return all(stat.S_ISDIR(ancestor.lstat().st_mode) for ancestor in (path, *path.parents))
    except OSError:
        return False


def _safe_private_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0


def load_admin_config(settings: Settings) -> AdminConfig:
    """Read only the fixed POSIX private-root/admin/config.toml path."""
    if os.name != "posix" or type(settings) is not Settings:
        raise AdminConfigError("admin configuration unavailable")
    root = settings.private_data_dir
    folder = root / "admin"
    path = folder / "config.toml"
    if (
        not _safe_ancestry(root)
        or not _safe_private_directory(root)
        or not _safe_private_directory(folder)
    ):
        raise AdminConfigError("admin configuration unavailable")
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o077
            or before.st_nlink != 1
        ):
            raise AdminConfigError("admin configuration unavailable")
        flags = os.O_RDONLY | os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            current = os.fstat(descriptor)
            if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
                raise AdminConfigError("admin configuration unavailable")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(4097)
        finally:
            os.close(descriptor)
        if len(raw) > 4096:
            raise AdminConfigError("admin configuration unavailable")
        values = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise AdminConfigError("admin configuration unavailable") from None
    if (
        type(values) is not dict
        or not {"enabled", "owner_id"}.issubset(values)
        or not set(values).issubset({"enabled", "owner_id", "import_format"})
        or values["enabled"] is not True
        or not _valid_owner_id(values["owner_id"])
        or (
            "import_format" in values
            and (
                type(values["import_format"]) is not str
                or values["import_format"] not in {"ciphertalk_json", "standard_json"}
            )
        )
    ):
        raise AdminConfigError("admin configuration unavailable")
    return AdminConfig(owner_id=values["owner_id"], import_format=values.get("import_format"))
