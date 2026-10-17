from __future__ import annotations

import os
import stat
import tomllib
import weakref
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, NoReturn

from pydantic import BaseModel, ConfigDict

from companion.runtime.settings import Settings
from companion.speech.local_errors import (
    LocalSpeechAdapterErrorCode,
    raise_local_speech_error,
)

if TYPE_CHECKING:
    from companion.speech.local_audio_devices import LocalAudioDeviceConfig
    from companion.speech.local_model_config import LocalSpeechModelConfig


class SpeechRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    model_root: str
    run_root: str
    host_api: str
    input_selector: str
    output_selector: str


@dataclass(frozen=True, slots=True, repr=False, weakref_slot=True)
class _ValidatedSpeechRuntimeConfig:
    config: SpeechRuntimeConfig
    speech_root: Path
    model_root: Path
    run_root: Path


_RUNTIME_FIELDS = ("model_root", "run_root", "host_api", "input_selector", "output_selector")
_SELECTOR_MAX_LENGTH = 255
_PATH_TYPE = type(Path())
_ISSUED_RUNTIME_CONFIGS: weakref.WeakValueDictionary[int, _ValidatedSpeechRuntimeConfig] = (
    weakref.WeakValueDictionary()
)
_BOUND_MODEL_CONFIGS: dict[int, weakref.ReferenceType[object]] = {}


def _require_exact_str(value: object) -> str:
    """Reject any ``str`` subclass or non-string value."""
    if type(value) is not str:
        raise TypeError("value must be a built-in str")
    return value


def _require_selector(value: object) -> str:
    text = _require_exact_str(value)
    if not text or text != text.strip() or len(text) > _SELECTOR_MAX_LENGTH:
        raise ValueError("selector must be non-blank with no surrounding whitespace")
    return text


def _absolute_no_follow(path: Path, *, directory: bool) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        metadata = current.lstat()
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError("link or reparse point")
    resolved = absolute.resolve(strict=True)
    metadata = resolved.stat()
    if directory and not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("root is not a directory")
    if not directory and not stat.S_ISREG(metadata.st_mode):
        raise ValueError("runtime config is not a regular file")
    return resolved


def _resolve_contained_root(speech_root: Path, value: str) -> Path:
    """Resolve a relative root beneath ``speech_root`` without following links."""
    value = _require_exact_str(value)
    source = Path(value)
    windows_source = PureWindowsPath(value)
    if (
        source.is_absolute()
        or source.drive
        or windows_source.is_absolute()
        or windows_source.drive
        or not source.parts
    ):
        raise ValueError("root must be a relative path")
    if any(part in {"", ".", ".."} for part in source.parts):
        raise ValueError("invalid path segment")
    root = _absolute_no_follow(speech_root, directory=True)
    candidate = Path(os.path.abspath(root / source))
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("root escapes speech directory")
    return _absolute_no_follow(candidate, directory=True)


def _raise_configuration_failed() -> NoReturn:
    raise_local_speech_error(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)


def _load_validated_speech_runtime_config(settings: Settings) -> _ValidatedSpeechRuntimeConfig:
    speech_root: Path | None = None
    runtime_path: Path | None = None
    handle: object = None
    document: object = None
    model_literal: str | None = None
    run_literal: str | None = None
    config: SpeechRuntimeConfig | None = None
    model_root: Path | None = None
    run_root: Path | None = None
    result: _ValidatedSpeechRuntimeConfig | None = None
    failed = False
    try:
        if type(settings) is not Settings:
            raise TypeError("settings must be an exact Settings")
        if type(settings.private_data_dir) is not _PATH_TYPE:
            raise TypeError("private data root must be an exact Path")
        speech_root = settings.private_data_dir / "speech"
        speech_root = _absolute_no_follow(speech_root, directory=True)
        runtime_path = _absolute_no_follow(speech_root / "runtime.toml", directory=False)
        with runtime_path.open("rb") as handle:
            document = tomllib.load(handle)
        if type(document) is not dict:
            raise TypeError("runtime.toml must be a table")
        if set(document) != set(_RUNTIME_FIELDS):
            raise ValueError("runtime.toml must contain exactly five fields")

        model_literal = _require_exact_str(document["model_root"])
        run_literal = _require_exact_str(document["run_root"])
        if model_literal != "models":
            raise ValueError("invalid model root")
        if run_literal != "run":
            raise ValueError("invalid run root")

        config = SpeechRuntimeConfig(
            model_root=model_literal,
            run_root=run_literal,
            host_api=_require_selector(document["host_api"]),
            input_selector=_require_selector(document["input_selector"]),
            output_selector=_require_selector(document["output_selector"]),
        )
        model_root = _resolve_contained_root(speech_root, model_literal)
        run_root = _resolve_contained_root(speech_root, run_literal)
        result = _ValidatedSpeechRuntimeConfig(
            config=config,
            speech_root=speech_root,
            model_root=model_root,
            run_root=run_root,
        )
        _ISSUED_RUNTIME_CONFIGS[id(result)] = result
    except Exception:
        failed = True
    settings = None  # type: ignore[assignment]
    speech_root = None
    runtime_path = None
    handle = None
    document = None
    model_literal = None
    run_literal = None
    config = None
    model_root = None
    run_root = None
    if failed or result is None:
        result = None
        _raise_configuration_failed()
    return result


def _require_current_validated_runtime(
    settings: Settings,
    value: object,
) -> _ValidatedSpeechRuntimeConfig:
    result: _ValidatedSpeechRuntimeConfig | None = None
    current_speech_root: Path | None = None
    current_model_root: Path | None = None
    current_run_root: Path | None = None
    failed = False
    try:
        if type(settings) is not Settings or type(value) is not _ValidatedSpeechRuntimeConfig:
            raise TypeError("untrusted runtime configuration")
        if type(settings.private_data_dir) is not _PATH_TYPE:
            raise TypeError("private data root must be an exact Path")
        if _ISSUED_RUNTIME_CONFIGS.get(id(value)) is not value:
            raise ValueError("unissued runtime configuration")
        if type(value.config) is not SpeechRuntimeConfig:
            raise TypeError("invalid runtime document")
        if any(
            type(path) is not _PATH_TYPE
            for path in (value.speech_root, value.model_root, value.run_root)
        ):
            raise TypeError("invalid runtime root")
        current_speech_root = _absolute_no_follow(
            settings.private_data_dir / "speech",
            directory=True,
        )
        current_model_root = _resolve_contained_root(current_speech_root, "models")
        current_run_root = _resolve_contained_root(current_speech_root, "run")
        if (
            value.speech_root != current_speech_root
            or value.model_root != current_model_root
            or value.run_root != current_run_root
            or value.config.model_dump()
            != {
                "model_root": "models",
                "run_root": "run",
                "host_api": _require_selector(value.config.host_api),
                "input_selector": _require_selector(value.config.input_selector),
                "output_selector": _require_selector(value.config.output_selector),
            }
        ):
            raise ValueError("runtime configuration drift")
        result = value
    except Exception:
        failed = True
    settings = None  # type: ignore[assignment]
    value = None
    current_speech_root = None
    current_model_root = None
    current_run_root = None
    if failed or result is None:
        result = None
        _raise_configuration_failed()
    return result


def _local_model_config_from_runtime(
    settings: Settings,
    value: object,
) -> LocalSpeechModelConfig:
    from companion.speech.local_model_config import _model_config_from_validated_runtime

    validated: _ValidatedSpeechRuntimeConfig | None = None
    model_config: LocalSpeechModelConfig | None = None
    failed = False
    key: int | None = None
    try:
        validated = _require_current_validated_runtime(settings, value)
        model_config = _model_config_from_validated_runtime(settings, validated)
        key = id(validated)
    except Exception:
        failed = True
    settings = None  # type: ignore[assignment]
    value = None
    validated = None
    if failed or model_config is None or key is None:
        model_config = None
        _raise_configuration_failed()

    def discard(_reference: object) -> None:
        assert key is not None
        _BOUND_MODEL_CONFIGS.pop(key, None)

    _BOUND_MODEL_CONFIGS[key] = weakref.ref(model_config, discard)
    return model_config


def _require_current_model_config(
    settings: Settings,
    validated: _ValidatedSpeechRuntimeConfig,
    value: object,
) -> LocalSpeechModelConfig:
    from companion.speech.local_model_config import (
        LocalSpeechModelConfig,
        _config_matches_manifest,
        _declarations_from_config,
        _relative_paths_from_config,
    )

    result: LocalSpeechModelConfig | None = None
    failed = False
    try:
        current = _require_current_validated_runtime(settings, validated)
        reference = _BOUND_MODEL_CONFIGS.get(id(current))
        if (
            type(value) is not LocalSpeechModelConfig
            or reference is None
            or reference() is not value
            or type(value.model_root) is not _PATH_TYPE
            or type(value.run_root) is not _PATH_TYPE
            or any(type(path) is not _PATH_TYPE for path in _relative_paths_from_config(value))
            or any(type(item) is not str for item in _declarations_from_config(value))
            or value.model_root != current.model_root
            or value.run_root != current.run_root
            or not _config_matches_manifest(value)
        ):
            raise ValueError("untrusted model configuration")
        result = value
    except Exception:
        failed = True
    settings = None  # type: ignore[assignment]
    validated = None  # type: ignore[assignment]
    value = None
    current = None  # type: ignore[assignment]
    reference = None  # type: ignore[assignment]
    if failed or result is None:
        result = None
        _raise_configuration_failed()
    return result


def _local_audio_config_from_runtime(
    settings: Settings,
    value: object,
    model_config: object,
) -> LocalAudioDeviceConfig:
    from companion.speech.local_audio_devices import _audio_config_from_validated_runtime

    validated: _ValidatedSpeechRuntimeConfig | None = None
    bound_model: LocalSpeechModelConfig | None = None
    result: LocalAudioDeviceConfig | None = None
    failed = False
    try:
        validated = _require_current_validated_runtime(settings, value)
        bound_model = _require_current_model_config(settings, validated, model_config)
        result = _audio_config_from_validated_runtime(settings, validated, bound_model)
    except Exception:
        failed = True
    settings = None  # type: ignore[assignment]
    value = None
    model_config = None
    validated = None
    bound_model = None
    if failed or result is None:
        result = None
        _raise_configuration_failed()
    return result


def preflight_local_speech_runtime(settings: Settings) -> None:
    from companion.speech.local_model_config import verify_local_speech_config

    validated: _ValidatedSpeechRuntimeConfig | None = None
    model_config: object = None
    audio_config: object = None
    failed = False
    try:
        validated = _load_validated_speech_runtime_config(settings)
        model_config = _local_model_config_from_runtime(settings, validated)
        audio_config = _local_audio_config_from_runtime(settings, validated, model_config)
        if audio_config.run_root != model_config.run_root:  # type: ignore[attr-defined]
            raise ValueError("runtime roots differ")
        verify_local_speech_config(model_config)  # type: ignore[arg-type]
    except Exception:
        failed = True
    settings = None  # type: ignore[assignment]
    validated = None
    model_config = None
    audio_config = None
    if failed:
        _raise_configuration_failed()
