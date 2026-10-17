from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class SettingsError(RuntimeError):
    """Raised when runtime configuration cannot be loaded safely."""


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: Literal["development", "test", "production"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    private_data_dir: Path
    runtime_data_dir: Path
    test_results_dir: Path


ENV_TO_FIELD = {
    "COMPANION_ENVIRONMENT": "environment",
    "COMPANION_LOG_LEVEL": "log_level",
    "COMPANION_PRIVATE_DATA_DIR": "private_data_dir",
    "COMPANION_RUNTIME_DATA_DIR": "runtime_data_dir",
    "COMPANION_TEST_RESULTS_DIR": "test_results_dir",
}

PATH_FIELDS = ("private_data_dir", "runtime_data_dir", "test_results_dir")


def _read_companion_table(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    table = document["companion"]
    if not isinstance(table, dict):
        raise TypeError("companion configuration must be a table")
    return dict(table)


def _reject_private_symlink_ancestors(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise ValueError("private data path contains symlink")
        if current == current.parent:
            return
        current = current.parent


def load_settings(
    project_root: Path,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    root = project_root.resolve()
    environment = dict(os.environ if environ is None else environ)
    unknown = sorted(
        key for key in environment if key.startswith("COMPANION_") and key not in ENV_TO_FIELD
    )
    if unknown:
        raise SettingsError(f"unknown COMPANION_ setting: {unknown[0]}")

    try:
        values = _read_companion_table(root / "config" / "default.toml")
        if config_path is not None:
            values.update(_read_companion_table(config_path.resolve()))
        for environment_name, field_name in ENV_TO_FIELD.items():
            if environment_name in environment:
                values[field_name] = environment[environment_name]
        for field_name in PATH_FIELDS:
            path = Path(values[field_name])
            expanded = root / path if not path.is_absolute() else path
            if field_name == "private_data_dir":
                _reject_private_symlink_ancestors(expanded)
            values[field_name] = expanded.resolve()
        return Settings.model_validate(values)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        tomllib.TOMLDecodeError,
        ValidationError,
    ) as error:
        raise SettingsError(f"invalid configuration ({type(error).__name__})") from None
