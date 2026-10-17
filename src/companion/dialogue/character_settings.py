"""Edit the fixed character name without loading provider or style configuration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from companion.dialogue.models import DEFAULT_CHARACTER_NAME, validate_character_name
from companion.runtime.settings import Settings


class CharacterSettingsUnavailable(RuntimeError):
    """The fixed configuration cannot be read or written."""


class CharacterSettingsConflict(ValueError):
    """The form refers to an older configuration digest."""


class CharacterSettingsInvalidName(ValueError):
    """The submitted character name is invalid."""


def _read_config(settings: Settings) -> tuple[Path, dict[str, Any], str, str]:
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid configuration object")
        name = validate_character_name(payload.get("character_name", DEFAULT_CHARACTER_NAME))
    except (OSError, UnicodeError, ValueError):
        raise CharacterSettingsUnavailable("角色名称配置不可用") from None
    return path, payload, name, hashlib.sha256(raw).hexdigest()


def read_character_name(settings: Settings) -> tuple[str, str]:
    """Return the normalized name and the digest of the single raw file read."""
    _, _, name, digest = _read_config(settings)
    return name, digest


def write_character_name(settings: Settings, name: str, *, expected_digest: str) -> str:
    """Save a changed name for the next restart; the digest rejects stale forms.

    This read-then-replace operation does not provide cross-process compare-and-swap.
    """
    try:
        name = validate_character_name(name)
    except ValueError:
        raise CharacterSettingsInvalidName("角色名称无效") from None

    path, payload, current_name, current_digest = _read_config(settings)
    if current_digest != expected_digest:
        raise CharacterSettingsConflict("角色名称配置已变化，请重新查看")  # noqa: RUF001
    if name == current_name:
        return current_digest

    payload["character_name"] = name
    temporary_path: Path | None = None
    try:
        raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        try:
            with NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(raw)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except (OSError, UnicodeError):
        raise CharacterSettingsUnavailable("角色名称配置不可用") from None
    return hashlib.sha256(raw).hexdigest()
