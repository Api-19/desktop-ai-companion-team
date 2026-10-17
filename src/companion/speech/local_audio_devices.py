from __future__ import annotations

import importlib
import importlib.metadata
import stat
import sys
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import getattr_static
from pathlib import Path
from typing import NoReturn

from companion.speech.local_errors import (
    LocalSpeechAdapterError,
    LocalSpeechAdapterErrorCode,
    raise_local_speech_error,
)

SOUNDDEVICE_VERSION = "0.5.5"
INPUT_SAMPLE_RATE_HZ = 16_000
OUTPUT_SAMPLE_RATE_HZ = 44_100
DEVICE_NATIVE_SAMPLE_RATE_HZ = 48_000
_INPUT_SAMPLE_RATE_CANDIDATES = (INPUT_SAMPLE_RATE_HZ, DEVICE_NATIVE_SAMPLE_RATE_HZ)
_OUTPUT_SAMPLE_RATE_CANDIDATES = (OUTPUT_SAMPLE_RATE_HZ, DEVICE_NATIVE_SAMPLE_RATE_HZ)
_SAMPLE_DTYPE = "int16"
_SELECTOR_MAX_LENGTH = 255
_PATH_TYPE = type(Path())
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
_SUPPORTED_DEVICE_PLATFORMS = frozenset({"darwin", "linux", "win32"})


def _raise_device_configuration_error() -> NoReturn:
    raise_local_speech_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
        return stat.S_ISLNK(metadata.st_mode) or bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    except Exception:
        return True


def _validated_external_run_root(value: object) -> Path | None:
    if type(value) is not _PATH_TYPE:
        return None
    path = value
    try:
        if not path.is_absolute() or _contains_link_or_reparse(path):
            return None
        resolved = path.resolve(strict=True)
        worktree = _WORKTREE_ROOT.resolve(strict=True)
        if not resolved.is_dir() or resolved == worktree or worktree in resolved.parents:
            return None
        if resolved == Path(resolved.anchor):
            return None
        return resolved
    except Exception:
        return None


def _valid_selector(value: object) -> bool:
    return type(value) is str and bool(value.strip()) and len(value) <= _SELECTOR_MAX_LENGTH


def _contains_link_or_reparse(path: Path) -> bool:
    current = path
    while True:
        if _is_link_or_reparse(current):
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


@dataclass(frozen=True, slots=True, repr=False, init=False, weakref_slot=True)
class LocalAudioDeviceConfig:
    run_root: Path
    input_selector: str
    output_selector: str
    hostapi_selector: str | None

    def __init__(
        self,
        *,
        run_root: Path,
        input_selector: str,
        output_selector: str,
        hostapi_selector: str | None = None,
    ) -> None:
        resolved_root = _validated_external_run_root(run_root)
        if (
            resolved_root is None
            or not _valid_selector(input_selector)
            or not _valid_selector(output_selector)
            or (hostapi_selector is not None and not _valid_selector(hostapi_selector))
        ):
            del run_root
            del input_selector
            del output_selector
            del hostapi_selector
            del resolved_root
            _raise_device_configuration_error()
        object.__setattr__(self, "run_root", resolved_root)
        object.__setattr__(self, "input_selector", input_selector)
        object.__setattr__(self, "output_selector", output_selector)
        object.__setattr__(self, "hostapi_selector", hostapi_selector)


def _audio_config_from_validated_runtime(
    settings: object,
    value: object,
    model_config: object,
) -> LocalAudioDeviceConfig:
    from companion.speech.runtime_config import (
        _require_current_model_config,
        _require_current_validated_runtime,
    )

    validated: object = None
    bound_model: object = None
    result: LocalAudioDeviceConfig | None = None
    failed = False
    try:
        validated = _require_current_validated_runtime(settings, value)  # type: ignore[arg-type]
        bound_model = _require_current_model_config(  # type: ignore[arg-type]
            settings,
            validated,
            model_config,
        )
        result = object.__new__(LocalAudioDeviceConfig)
        object.__setattr__(result, "run_root", bound_model.run_root)  # type: ignore[attr-defined]
        object.__setattr__(
            result,
            "input_selector",
            validated.config.input_selector,  # type: ignore[attr-defined]
        )
        object.__setattr__(
            result,
            "output_selector",
            validated.config.output_selector,  # type: ignore[attr-defined]
        )
        object.__setattr__(
            result,
            "hostapi_selector",
            validated.config.host_api,  # type: ignore[attr-defined]
        )
    except Exception:
        failed = True
    settings = None
    value = None
    model_config = None
    validated = None
    bound_model = None
    if failed or result is None:
        result = None
        raise_local_speech_error(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)
    return result


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedAudioDevicePair:
    input_index: int
    output_index: int
    input_sample_rate_hz: int = INPUT_SAMPLE_RATE_HZ
    output_sample_rate_hz: int = OUTPUT_SAMPLE_RATE_HZ

    def __post_init__(self) -> None:
        if (
            type(self.input_index) is not int
            or self.input_index < 0
            or type(self.output_index) is not int
            or self.output_index < 0
            or type(self.input_sample_rate_hz) is not int
            or self.input_sample_rate_hz not in _INPUT_SAMPLE_RATE_CANDIDATES
            or type(self.output_sample_rate_hz) is not int
            or self.output_sample_rate_hz not in _OUTPUT_SAMPLE_RATE_CANDIDATES
        ):
            _raise_device_configuration_error()


def _first_supported_rate(
    check: object,
    *,
    device: int,
    candidates: tuple[int, ...],
) -> int | None:
    if not callable(check):
        return None
    for sample_rate_hz in candidates:
        try:
            result = check(
                device=device,
                channels=1,
                dtype=_SAMPLE_DTYPE,
                samplerate=sample_rate_hz,
            )
        except Exception:
            continue
        if result is None:
            return sample_rate_hz
    return None


def _load_sounddevice_runtime() -> object:
    try:
        version = importlib.metadata.version("sounddevice")
        if type(version) is not str or version != SOUNDDEVICE_VERSION:
            del version
            _raise_device_configuration_error()
        runtime = importlib.import_module("sounddevice")
    except LocalSpeechAdapterError:
        raise
    except Exception as error:
        del error
        _raise_device_configuration_error()
    return runtime


def _valid_device_entry(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        name = value["name"]
        input_channels = value["max_input_channels"]
        output_channels = value["max_output_channels"]
    except Exception:
        return False
    return (
        type(name) is str
        and type(input_channels) is int
        and input_channels >= 0
        and type(output_channels) is int
        and output_channels >= 0
    )


def _hostapi_resolution(hostapis: object, selector: str) -> tuple[int, int] | None:
    if not isinstance(hostapis, Sequence) or isinstance(hostapis, str | bytes | bytearray):
        return None
    matches: list[int] = []
    for index, entry in enumerate(hostapis):
        if not isinstance(entry, Mapping):
            return None
        try:
            name = entry["name"]
        except Exception:
            return None
        if type(name) is not str:
            return None
        if name == selector:
            matches.append(index)
    return (matches[0], len(hostapis)) if len(matches) == 1 else None


def _matching_indices(
    devices: Sequence[object],
    selector: str,
    *,
    channel_field: str,
    hostapi_index: int | None,
    hostapi_count: int | None,
) -> tuple[int, ...] | None:
    matches: list[int] = []
    for index, entry in enumerate(devices):
        if not _valid_device_entry(entry):
            return None
        try:
            if hostapi_index is not None:
                entry_hostapi = entry["hostapi"]  # type: ignore[index]
                if (
                    type(entry_hostapi) is not int
                    or hostapi_count is None
                    or not 0 <= entry_hostapi < hostapi_count
                ):
                    return None
                if entry_hostapi != hostapi_index:
                    continue
            if entry["name"] == selector and entry[channel_field] >= 1:  # type: ignore[index,operator]
                matches.append(index)
        except Exception:
            return None
    return tuple(matches)


def _resolve_with_runtime(
    runtime: object,
    input_selector: str,
    output_selector: str,
    hostapi_selector: str | None,
) -> ResolvedAudioDevicePair | None:
    try:
        query_devices = runtime.query_devices  # type: ignore[attr-defined]
        check_input = runtime.check_input_settings  # type: ignore[attr-defined]
        check_output = runtime.check_output_settings  # type: ignore[attr-defined]
        if not all(callable(item) for item in (query_devices, check_input, check_output)):
            return None
        devices = query_devices()
        if not isinstance(devices, Sequence) or isinstance(devices, str | bytes | bytearray):
            return None
        hostapi_index: int | None = None
        hostapi_count: int | None = None
        if hostapi_selector is not None:
            query_hostapis = runtime.query_hostapis  # type: ignore[attr-defined]
            if not callable(query_hostapis):
                return None
            hostapi_resolution = _hostapi_resolution(query_hostapis(), hostapi_selector)
            if hostapi_resolution is None:
                return None
            hostapi_index, hostapi_count = hostapi_resolution
        input_matches = _matching_indices(
            devices,
            input_selector,
            channel_field="max_input_channels",
            hostapi_index=hostapi_index,
            hostapi_count=hostapi_count,
        )
        output_matches = _matching_indices(
            devices,
            output_selector,
            channel_field="max_output_channels",
            hostapi_index=hostapi_index,
            hostapi_count=hostapi_count,
        )
        if input_matches is None or output_matches is None:
            return None
        if len(input_matches) != 1 or len(output_matches) != 1:
            return None
        input_index = input_matches[0]
        output_index = output_matches[0]
        input_sample_rate_hz = _first_supported_rate(
            check_input,
            device=input_index,
            candidates=_INPUT_SAMPLE_RATE_CANDIDATES,
        )
        if input_sample_rate_hz is None:
            return None
        output_sample_rate_hz = _first_supported_rate(
            check_output,
            device=output_index,
            candidates=_OUTPUT_SAMPLE_RATE_CANDIDATES,
        )
        if output_sample_rate_hz is None:
            return None
        return ResolvedAudioDevicePair(
            input_index=input_index,
            output_index=output_index,
            input_sample_rate_hz=input_sample_rate_hz,
            output_sample_rate_hz=output_sample_rate_hz,
        )
    except Exception:
        return None


def _resolve_audio_devices_for_runtime(
    config: LocalAudioDeviceConfig, runtime: object
) -> ResolvedAudioDevicePair:
    if (
        type(config) is not LocalAudioDeviceConfig
        or sys.platform not in _SUPPORTED_DEVICE_PLATFORMS
    ):
        del config
        del runtime
        _raise_device_configuration_error()
    pair = _resolve_with_runtime(
        runtime,
        config.input_selector,
        config.output_selector,
        config.hostapi_selector,
    )
    if pair is None:
        del config
        del runtime
        del pair
        _raise_device_configuration_error()
    return pair


_RUNTIME_SEAM_NAMES = (
    "query_devices",
    "check_input_settings",
    "check_output_settings",
    "RawInputStream",
    "RawOutputStream",
)


def _runtime_seams(runtime: object) -> tuple[object, ...] | None:
    try:
        seams = tuple(getattr_static(runtime, name) for name in _RUNTIME_SEAM_NAMES)
        return seams if all(callable(seam) for seam in seams) else None
    except Exception:
        return None


@dataclass(frozen=True, slots=True, repr=False, init=False, weakref_slot=True, eq=False)
class _ResolvedAudioRuntime:
    runtime: object
    pair: ResolvedAudioDevicePair
    run_root: Path

    @classmethod
    def _create(
        cls,
        *,
        runtime: object,
        pair: ResolvedAudioDevicePair,
        run_root: Path,
    ) -> _ResolvedAudioRuntime:
        result: _ResolvedAudioRuntime | None = None
        failed = False
        try:
            if cls is not _ResolvedAudioRuntime:
                raise TypeError
            if type(pair) is not ResolvedAudioDevicePair:
                raise TypeError
            if (
                type(pair.input_index) is not int
                or pair.input_index < 0
                or type(pair.output_index) is not int
                or pair.output_index < 0
                or type(pair.input_sample_rate_hz) is not int
                or pair.input_sample_rate_hz not in _INPUT_SAMPLE_RATE_CANDIDATES
                or type(pair.output_sample_rate_hz) is not int
                or pair.output_sample_rate_hz not in _OUTPUT_SAMPLE_RATE_CANDIDATES
            ):
                raise ValueError
            if type(run_root) is not _PATH_TYPE:
                raise TypeError
            if _runtime_seams(runtime) is None:
                raise ValueError
            result = object.__new__(cls)
            object.__setattr__(result, "runtime", runtime)
            object.__setattr__(result, "pair", pair)
            object.__setattr__(result, "run_root", run_root)
        except Exception:
            failed = True
        runtime = None
        pair = None
        run_root = None
        if failed or result is None:
            result = None
            _raise_device_configuration_error()
        return result


@dataclass(frozen=True, slots=True, repr=False)
class _ResolvedRuntimeBinding:
    config_reference: weakref.ReferenceType[LocalAudioDeviceConfig]
    runtime: object
    pair: ResolvedAudioDevicePair
    input_index: int
    output_index: int
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    run_root: Path
    input_selector: str
    output_selector: str
    hostapi_selector: str | None
    runtime_seams: tuple[object, ...]


_RESOLVED_RUNTIME_BINDINGS: weakref.WeakKeyDictionary[
    _ResolvedAudioRuntime,
    _ResolvedRuntimeBinding,
] = weakref.WeakKeyDictionary()


def _validate_resolved_audio_runtime(
    config: object,
    resolved: object,
) -> _ResolvedAudioRuntime:
    result: _ResolvedAudioRuntime | None = None
    failed = False
    try:
        if type(config) is not LocalAudioDeviceConfig:
            raise TypeError
        if type(resolved) is not _ResolvedAudioRuntime:
            raise TypeError
        binding = _RESOLVED_RUNTIME_BINDINGS.get(resolved)
        if binding is None or binding.config_reference() is not config:
            raise ValueError
        pair = resolved.pair
        if (
            type(pair) is not ResolvedAudioDevicePair
            or pair is not binding.pair
            or type(pair.input_index) is not int
            or pair.input_index < 0
            or pair.input_index != binding.input_index
            or type(pair.output_index) is not int
            or pair.output_index < 0
            or pair.output_index != binding.output_index
            or type(pair.input_sample_rate_hz) is not int
            or pair.input_sample_rate_hz != binding.input_sample_rate_hz
            or type(pair.output_sample_rate_hz) is not int
            or pair.output_sample_rate_hz != binding.output_sample_rate_hz
        ):
            raise ValueError
        if (
            resolved.runtime is not binding.runtime
            or type(resolved.run_root) is not _PATH_TYPE
            or resolved.run_root != binding.run_root
            or type(config.run_root) is not _PATH_TYPE
            or config.run_root != binding.run_root
            or type(config.input_selector) is not str
            or config.input_selector != binding.input_selector
            or type(config.output_selector) is not str
            or config.output_selector != binding.output_selector
            or (config.hostapi_selector is not None and type(config.hostapi_selector) is not str)
            or config.hostapi_selector != binding.hostapi_selector
        ):
            raise ValueError
        current_seams = _runtime_seams(resolved.runtime)
        if current_seams is None or any(
            current is not original
            for current, original in zip(
                current_seams,
                binding.runtime_seams,
                strict=True,
            )
        ):
            raise ValueError
        result = resolved
    except Exception:
        failed = True
    config = None
    resolved = None
    binding = None  # type: ignore[assignment]
    if failed or result is None:
        result = None
        _raise_device_configuration_error()
    return result


def _resolve_audio_runtime(config: object) -> _ResolvedAudioRuntime:
    result: _ResolvedAudioRuntime | None = None
    runtime: object = None
    failed = False
    try:
        if type(config) is not LocalAudioDeviceConfig:
            raise TypeError
        runtime = _load_sounddevice_runtime()
        pair = _resolve_audio_devices_for_runtime(config, runtime)
        result = _ResolvedAudioRuntime._create(
            runtime=runtime,
            pair=pair,
            run_root=config.run_root,
        )
        runtime_seams = _runtime_seams(runtime)
        if runtime_seams is None:
            raise ValueError
        _RESOLVED_RUNTIME_BINDINGS[result] = _ResolvedRuntimeBinding(
            config_reference=weakref.ref(config),
            runtime=runtime,
            pair=pair,
            input_index=pair.input_index,
            output_index=pair.output_index,
            input_sample_rate_hz=pair.input_sample_rate_hz,
            output_sample_rate_hz=pair.output_sample_rate_hz,
            run_root=config.run_root,
            input_selector=config.input_selector,
            output_selector=config.output_selector,
            hostapi_selector=config.hostapi_selector,
            runtime_seams=runtime_seams,
        )
    except Exception:
        failed = True
    config = None
    runtime = None
    if failed or result is None:
        result = None
        _raise_device_configuration_error()
    return result


def resolve_audio_devices(config: LocalAudioDeviceConfig) -> ResolvedAudioDevicePair:
    if type(config) is not LocalAudioDeviceConfig:
        del config
        _raise_device_configuration_error()
    if sys.platform not in _SUPPORTED_DEVICE_PLATFORMS:
        del config
        _raise_device_configuration_error()
    try:
        runtime = _load_sounddevice_runtime()
    except Exception:
        runtime = None
    if runtime is None:
        del config
        del runtime
        _raise_device_configuration_error()
    return _resolve_audio_devices_for_runtime(config, runtime)


# Public names for the body process (companion.bodylink); same objects, same behaviour.
load_sounddevice_runtime = _load_sounddevice_runtime
resolve_with_runtime = _resolve_with_runtime
