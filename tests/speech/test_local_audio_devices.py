from __future__ import annotations

import copy
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

import companion.speech.local_audio_devices as devices_module
from companion.runtime.settings import Settings
from companion.speech.local_audio_devices import (
    INPUT_SAMPLE_RATE_HZ,
    OUTPUT_SAMPLE_RATE_HZ,
    LocalAudioDeviceConfig,
    ResolvedAudioDevicePair,
    resolve_audio_devices,
)
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode


class _FakeSoundDevice:
    def __init__(self, devices: object, hostapis: object | None = None) -> None:
        self.devices = devices
        self.hostapis = [{"name": "synthetic-api"}] if hostapis is None else hostapis
        self.query_calls = 0
        self.hostapi_query_calls = 0
        self.input_checks: list[dict[str, object]] = []
        self.output_checks: list[dict[str, object]] = []
        self.input_failure: Exception | None = None
        self.output_failure: Exception | None = None
        self.unsupported_input_rates: set[int] = set()
        self.unsupported_output_rates: set[int] = set()
        self.input_open_calls = 0
        self.output_open_calls = 0

    def query_devices(self) -> object:
        self.query_calls += 1
        return self.devices

    def query_hostapis(self) -> object:
        self.hostapi_query_calls += 1
        return self.hostapis

    def check_input_settings(self, **kwargs: object) -> None:
        self.input_checks.append(kwargs)
        if self.input_failure is not None:
            raise self.input_failure
        if kwargs.get("samplerate") in self.unsupported_input_rates:
            raise ValueError("unsupported")

    def check_output_settings(self, **kwargs: object) -> None:
        self.output_checks.append(kwargs)
        if self.output_failure is not None:
            raise self.output_failure
        if kwargs.get("samplerate") in self.unsupported_output_rates:
            raise ValueError("unsupported")

    def RawInputStream(self, **_kwargs: object) -> object:
        self.input_open_calls += 1
        return object()

    def RawOutputStream(self, **_kwargs: object) -> object:
        self.output_open_calls += 1
        return object()


def _make_config(tmp_path: Path, **overrides: object) -> LocalAudioDeviceConfig:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    values: dict[str, object] = {
        "run_root": run_root,
        "input_selector": "synthetic-input",
        "output_selector": "synthetic-output",
    }
    values.update(overrides)
    return LocalAudioDeviceConfig(**values)  # type: ignore[arg-type]


def _valid_runtime() -> _FakeSoundDevice:
    return _FakeSoundDevice(
        [
            {
                "name": "unrelated",
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {
                "name": "synthetic-input",
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "synthetic-output",
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
        ]
    )


def _install_runtime(monkeypatch: pytest.MonkeyPatch, runtime: object) -> None:
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)


def test_unique_compatible_pair_resolves_with_frozen_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    runtime = _valid_runtime()
    _install_runtime(monkeypatch, runtime)

    pair = resolve_audio_devices(config)

    assert type(pair) is ResolvedAudioDevicePair
    assert (pair.input_index, pair.output_index) == (1, 2)
    assert (pair.input_sample_rate_hz, pair.output_sample_rate_hz) == (16_000, 44_100)
    assert runtime.query_calls == 1
    assert runtime.input_checks == [
        {"device": 1, "channels": 1, "dtype": "int16", "samplerate": 16_000}
    ]
    assert runtime.output_checks == [
        {"device": 2, "channels": 1, "dtype": "int16", "samplerate": 44_100}
    ]
    assert INPUT_SAMPLE_RATE_HZ == 16_000
    assert OUTPUT_SAMPLE_RATE_HZ == 44_100


def test_macos_resolution_uses_the_same_exact_fake_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _valid_runtime()
    monkeypatch.setattr(devices_module.sys, "platform", "darwin")
    _install_runtime(monkeypatch, runtime)

    pair = resolve_audio_devices(_make_config(tmp_path))

    assert (pair.input_index, pair.output_index) == (1, 2)
    assert (pair.input_sample_rate_hz, pair.output_sample_rate_hz) == (16_000, 44_100)
    assert runtime.query_calls == 1


def test_linux_resolution_uses_the_same_exact_fake_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _valid_runtime()
    monkeypatch.setattr(devices_module.sys, "platform", "linux")
    _install_runtime(monkeypatch, runtime)

    pair = resolve_audio_devices(_make_config(tmp_path))

    assert (pair.input_index, pair.output_index) == (1, 2)
    assert (pair.input_sample_rate_hz, pair.output_sample_rate_hz) == (16_000, 44_100)
    assert runtime.query_calls == 1


def test_pair_falls_back_to_device_native_48khz_in_fixed_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    runtime = _valid_runtime()
    runtime.unsupported_input_rates.add(16_000)
    runtime.unsupported_output_rates.add(44_100)
    _install_runtime(monkeypatch, runtime)

    pair = resolve_audio_devices(config)

    assert (pair.input_sample_rate_hz, pair.output_sample_rate_hz) == (48_000, 48_000)
    assert [item["samplerate"] for item in runtime.input_checks] == [16_000, 48_000]
    assert [item["samplerate"] for item in runtime.output_checks] == [44_100, 48_000]
    assert runtime.query_calls == 1


def test_pair_fails_when_no_frozen_sample_rate_is_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    runtime = _valid_runtime()
    runtime.unsupported_input_rates.update((16_000, 48_000))
    _install_runtime(monkeypatch, runtime)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        resolve_audio_devices(config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED
    assert runtime.output_checks == []


def test_config_and_pair_repr_do_not_expose_selectors_or_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    runtime = _valid_runtime()
    _install_runtime(monkeypatch, runtime)

    pair = resolve_audio_devices(config)

    assert "synthetic-input" not in repr(config)
    assert "synthetic-output" not in repr(config)
    assert "synthetic-api" not in repr(config)
    assert str(config.run_root) not in repr(config)
    assert "synthetic" not in repr(pair)


@pytest.mark.parametrize("field", ["input_selector", "output_selector"])
@pytest.mark.parametrize("value", ["", "   ", "x" * 256])
def test_config_rejects_blank_or_overlong_selectors(tmp_path: Path, field: str, value: str) -> None:
    with pytest.raises(LocalSpeechAdapterError) as caught:
        _make_config(tmp_path, **{field: value})

    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED


def test_config_rejects_selector_subclasses_and_invalid_roots(tmp_path: Path) -> None:
    class SelectorSubclass(str):
        pass

    with pytest.raises(LocalSpeechAdapterError):
        _make_config(tmp_path, input_selector=SelectorSubclass("synthetic-input"))
    with pytest.raises(LocalSpeechAdapterError):
        _make_config(tmp_path, hostapi_selector=SelectorSubclass("synthetic-api"))
    with pytest.raises(LocalSpeechAdapterError):
        _make_config(tmp_path, hostapi_selector=" ")
    with pytest.raises(LocalSpeechAdapterError):
        _make_config(tmp_path, run_root="not-a-path")
    with pytest.raises(LocalSpeechAdapterError):
        _make_config(tmp_path, run_root=Path.cwd())


def test_explicit_hostapi_disambiguates_duplicate_device_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeSoundDevice(
        [
            {
                "name": "synthetic-input",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "synthetic-input",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "synthetic-output",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 1,
            },
            {
                "name": "synthetic-output",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 1,
            },
        ],
        [{"name": "other-api"}, {"name": "synthetic-api"}],
    )
    _install_runtime(monkeypatch, runtime)

    pair = resolve_audio_devices(_make_config(tmp_path, hostapi_selector="synthetic-api"))

    assert (pair.input_index, pair.output_index) == (1, 3)
    assert runtime.hostapi_query_calls == 1
    assert "synthetic-api" not in repr(_make_config(tmp_path, hostapi_selector="synthetic-api"))


@pytest.mark.parametrize(
    "hostapis",
    [[], [{"name": "other-api"}], [{"name": "synthetic-api"}, {"name": "synthetic-api"}]],
)
def test_explicit_hostapi_missing_or_ambiguous_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hostapis: object
) -> None:
    runtime = _FakeSoundDevice(_valid_runtime().devices, hostapis)
    _install_runtime(monkeypatch, runtime)

    with pytest.raises(LocalSpeechAdapterError):
        resolve_audio_devices(_make_config(tmp_path, hostapi_selector="synthetic-api"))


@pytest.mark.parametrize("bad_hostapi", [True, -1, 2, "1"])
def test_explicit_hostapi_rejects_malformed_device_hostapi_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_hostapi: object
) -> None:
    devices = [
        {
            "name": "unrelated",
            "hostapi": bad_hostapi,
            "max_input_channels": 0,
            "max_output_channels": 1,
        },
        {
            "name": "synthetic-input",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
        },
        {
            "name": "synthetic-output",
            "hostapi": 0,
            "max_input_channels": 0,
            "max_output_channels": 1,
        },
    ]
    runtime = _FakeSoundDevice(devices)
    _install_runtime(monkeypatch, runtime)

    with pytest.raises(LocalSpeechAdapterError):
        resolve_audio_devices(_make_config(tmp_path, hostapi_selector="synthetic-api"))


def test_invalid_config_construction_does_not_retain_selectors_or_root(tmp_path: Path) -> None:
    marker = "CONSTRUCTOR_" + "SENSITIVE_MARKER"
    run_root = tmp_path / marker / "run"
    run_root.mkdir(parents=True)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        LocalAudioDeviceConfig(
            run_root=run_root,
            input_selector=f"{marker}-input",
            output_selector=" ",
        )

    production_locals: list[str] = []
    for frame, _lineno in traceback.walk_tb(caught.value.__traceback__):
        if Path(frame.f_code.co_filename).name == "local_audio_devices.py":
            production_locals.extend(repr(value) for value in frame.f_locals.values())
    assert marker not in "".join(production_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_config_rejects_an_ancestor_reparse_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ancestor = tmp_path / "synthetic-reparse-parent"
    run_root = ancestor / "run"
    run_root.mkdir(parents=True)
    real_check = devices_module._is_link_or_reparse

    monkeypatch.setattr(
        devices_module,
        "_is_link_or_reparse",
        lambda path: path == ancestor or real_check(path),
    )

    with pytest.raises(LocalSpeechAdapterError):
        LocalAudioDeviceConfig(
            run_root=run_root,
            input_selector="synthetic-input",
            output_selector="synthetic-output",
        )


@pytest.mark.parametrize(
    "devices",
    [
        [],
        [
            {"name": "synthetic-input", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "synthetic-input", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "synthetic-output", "max_input_channels": 0, "max_output_channels": 1},
        ],
        [
            {"name": "synthetic-input", "max_input_channels": 0, "max_output_channels": 1},
            {"name": "synthetic-output", "max_input_channels": 1, "max_output_channels": 0},
        ],
        [
            {"name": "synthetic-input", "max_input_channels": True, "max_output_channels": 0},
            {"name": "synthetic-output", "max_input_channels": 0, "max_output_channels": 1},
        ],
        [
            {"name": "synthetic-input", "max_input_channels": 1},
            {"name": "synthetic-output", "max_input_channels": 0, "max_output_channels": 1},
        ],
        "not-a-device-sequence",
    ],
)
def test_missing_ambiguous_incompatible_or_malformed_devices_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, devices: object
) -> None:
    runtime = _FakeSoundDevice(devices)
    _install_runtime(monkeypatch, runtime)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        resolve_audio_devices(_make_config(tmp_path))

    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("direction", ["input", "output"])
def test_unsupported_device_format_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    marker = "runtime-format-failure"
    runtime = _valid_runtime()
    setattr(runtime, f"{direction}_failure", RuntimeError(marker))
    _install_runtime(monkeypatch, runtime)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        resolve_audio_devices(_make_config(tmp_path))

    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED
    assert marker not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_resolution_exception_does_not_retain_selector_runtime_or_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "SENSITIVE_" + "DEVICE_MARKER"
    config = _make_config(
        tmp_path / marker,
        input_selector=f"{marker}-input",
        output_selector=f"{marker}-output",
    )
    runtime = _FakeSoundDevice(
        [
            {"name": f"{marker}-input", "max_input_channels": 1, "max_output_channels": 0},
            {"name": f"{marker}-output", "max_input_channels": 0, "max_output_channels": 1},
        ]
    )
    runtime.output_failure = RuntimeError(marker)
    _install_runtime(monkeypatch, runtime)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        resolve_audio_devices(config)

    production_locals: list[str] = []
    for frame, _lineno in traceback.walk_tb(caught.value.__traceback__):
        if Path(frame.f_code.co_filename).name == "local_audio_devices.py":
            production_locals.extend(repr(value) for value in frame.f_locals.values())
    assert marker not in str(caught.value)
    assert marker not in "".join(production_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_runtime_load_failure_does_not_retain_device_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "LOAD_" + "SENSITIVE_MARKER"
    config = _make_config(
        tmp_path / marker,
        input_selector=f"{marker}-input",
        output_selector=f"{marker}-output",
    )

    def fail_load() -> object:
        raise RuntimeError(marker)

    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", fail_load)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        resolve_audio_devices(config)

    production_locals: list[str] = []
    for frame, _lineno in traceback.walk_tb(caught.value.__traceback__):
        if Path(frame.f_code.co_filename).name == "local_audio_devices.py":
            production_locals.extend(repr(value) for value in frame.f_locals.values())
    assert marker not in "".join(production_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_resolver_requires_exact_config_type(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    with pytest.raises(LocalSpeechAdapterError):
        resolve_audio_devices(object())  # type: ignore[arg-type]
    with pytest.raises(LocalSpeechAdapterError):
        replace(config, input_selector=" ")


def test_unsupported_platform_resolution_fails_before_runtime_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads = 0

    def forbidden_load() -> object:
        nonlocal loads
        loads += 1
        raise AssertionError("runtime must not load")

    monkeypatch.setattr(devices_module.sys, "platform", "freebsd")
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", forbidden_load)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        resolve_audio_devices(_make_config(tmp_path))

    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED
    assert loads == 0


def test_runtime_loader_is_lazy_and_enforces_pinned_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(devices_module.importlib.metadata, "version", lambda _name: "0.5.4")
    monkeypatch.setattr(
        devices_module.importlib,
        "import_module",
        lambda name: imports.append(name) or object(),
    )

    with pytest.raises(LocalSpeechAdapterError):
        devices_module._load_sounddevice_runtime()
    assert imports == []

    monkeypatch.setattr(devices_module.importlib.metadata, "version", lambda _name: "0.5.5")
    runtime = devices_module._load_sounddevice_runtime()
    assert type(runtime) is object
    assert imports == ["sounddevice"]


def test_import_does_not_load_sounddevice_in_an_isolated_process() -> None:
    script = """
import importlib
real_import = importlib.import_module
def guarded(name, package=None):
    if name == 'sounddevice':
        raise SystemExit(91)
    return real_import(name, package)
importlib.import_module = guarded
import companion.speech.local_audio_devices
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_private_audio_constructor_requires_runtime_and_model_capabilities() -> None:
    helper = getattr(devices_module, "_audio_config_from_validated_runtime", None)
    assert callable(helper)

    with pytest.raises(LocalSpeechAdapterError):
        helper(object(), object(), object())


def test_private_audio_constructor_failure_scrubs_all_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "PRIVATE_AUDIO_FACTORY_SENTINEL"
    settings = Settings.model_validate(
        {
            "environment": "test",
            "log_level": "INFO",
            "private_data_dir": tmp_path / marker,
            "runtime_data_dir": tmp_path / "runtime",
            "test_results_dir": tmp_path / "results",
        }
    )
    capability = {"marker": marker}
    model = {"marker": marker}

    with pytest.raises(LocalSpeechAdapterError) as caught:
        devices_module._audio_config_from_validated_runtime(settings, capability, model)

    production_locals: list[str] = []
    for frame, _lineno in traceback.walk_tb(caught.value.__traceback__):
        if Path(frame.f_code.co_filename).name == "local_audio_devices.py":
            production_locals.extend(repr(item) for item in frame.f_locals.values())
    captured = capsys.readouterr()
    assert marker not in "".join(production_locals)
    assert marker not in captured.out
    assert marker not in captured.err
    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def _resolve_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, LocalAudioDeviceConfig, _FakeSoundDevice]:
    config = _make_config(tmp_path)
    runtime = _valid_runtime()
    _install_runtime(monkeypatch, runtime)
    resolved = devices_module._resolve_audio_runtime(config)
    return resolved, config, runtime


def test_resolve_audio_runtime_resolves_once_and_stores_exact_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    runtime = _valid_runtime()
    _install_runtime(monkeypatch, runtime)

    resolved = devices_module._resolve_audio_runtime(config)

    assert type(resolved) is devices_module._ResolvedAudioRuntime
    assert type(resolved.pair) is ResolvedAudioDevicePair
    assert (resolved.pair.input_index, resolved.pair.output_index) == (1, 2)
    assert resolved.run_root == config.run_root
    assert resolved.runtime is runtime
    assert runtime.query_calls == 1
    assert runtime.hostapi_query_calls == 0
    assert runtime.input_open_calls == 0
    assert runtime.output_open_calls == 0


def test_resolve_audio_runtime_with_hostapi_queries_hostapi_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path, hostapi_selector="synthetic-api")
    runtime = _FakeSoundDevice(
        [
            {
                "name": "synthetic-input",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "synthetic-output",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 1,
            },
        ],
        [{"name": "synthetic-api"}],
    )
    _install_runtime(monkeypatch, runtime)

    resolved = devices_module._resolve_audio_runtime(config)

    assert (resolved.pair.input_index, resolved.pair.output_index) == (0, 1)
    assert runtime.query_calls == 1
    assert runtime.hostapi_query_calls == 1
    assert runtime.input_open_calls == 0
    assert runtime.output_open_calls == 0


def test_resolve_audio_runtime_requires_exact_config_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = 0

    def forbidden_load() -> object:
        nonlocal loads
        loads += 1
        raise AssertionError("runtime must not load")

    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", forbidden_load)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        devices_module._resolve_audio_runtime(object())

    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED
    assert loads == 0


def test_resolved_audio_runtime_repr_does_not_expose_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved, config, _runtime = _resolve_once(tmp_path, monkeypatch)

    text = repr(resolved)

    assert "synthetic-input" not in text
    assert "synthetic-output" not in text
    assert "synthetic-api" not in text
    assert str(config.run_root) not in text
    assert "input_index" not in text
    assert "output_index" not in text


def test_resolved_audio_runtime_create_rejects_malformed_pairs(
    tmp_path: Path,
) -> None:
    runtime = _valid_runtime()

    bad_input = object.__new__(ResolvedAudioDevicePair)
    object.__setattr__(bad_input, "input_index", True)
    object.__setattr__(bad_input, "output_index", 0)
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._ResolvedAudioRuntime._create(
            runtime=runtime, pair=bad_input, run_root=tmp_path / "run"
        )

    bad_output = object.__new__(ResolvedAudioDevicePair)
    object.__setattr__(bad_output, "input_index", 0)
    object.__setattr__(bad_output, "output_index", -1)
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._ResolvedAudioRuntime._create(
            runtime=runtime, pair=bad_output, run_root=tmp_path / "run"
        )


def test_resolved_audio_runtime_create_rejects_subclass(tmp_path: Path) -> None:
    runtime = _valid_runtime()
    pair = ResolvedAudioDevicePair(input_index=0, output_index=1)

    class Subclass(devices_module._ResolvedAudioRuntime):
        pass

    with pytest.raises(LocalSpeechAdapterError):
        Subclass._create(runtime=runtime, pair=pair, run_root=tmp_path / "run")


def test_validate_resolved_audio_runtime_rejects_wrong_types_and_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved, config, runtime = _resolve_once(tmp_path, monkeypatch)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        devices_module._validate_resolved_audio_runtime(object(), resolved)
    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED

    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, object())

    class Lookalike:
        pass

    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, Lookalike())

    forged = object.__new__(devices_module._ResolvedAudioRuntime)
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, forged)

    hand_copied = object.__new__(devices_module._ResolvedAudioRuntime)
    object.__setattr__(hand_copied, "runtime", runtime)
    object.__setattr__(hand_copied, "pair", resolved.pair)
    object.__setattr__(hand_copied, "run_root", config.run_root)
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, hand_copied)

    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, copy.copy(resolved))


def test_direct_create_is_not_a_parser_issued_capability(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    runtime = _valid_runtime()
    pair = ResolvedAudioDevicePair(input_index=1, output_index=2)
    direct = devices_module._ResolvedAudioRuntime._create(
        runtime=runtime,
        pair=pair,
        run_root=config.run_root,
    )

    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, direct)


def test_resolved_runtime_binds_exact_config_not_only_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved, config, _runtime = _resolve_once(tmp_path, monkeypatch)
    same_root_other_selectors = LocalAudioDeviceConfig(
        run_root=config.run_root,
        input_selector="other-input",
        output_selector="other-output",
    )
    copied_config = copy.copy(config)

    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(same_root_other_selectors, resolved)
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(copied_config, resolved)


def test_validate_resolved_audio_runtime_rejects_mismatched_config_or_mutated_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved, config, runtime = _resolve_once(tmp_path, monkeypatch)

    other = _make_config(tmp_path / "other")
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(other, resolved)

    runtime.RawInputStream = None  # type: ignore[method-assign]
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, resolved)


def test_validate_resolved_audio_runtime_rejects_in_place_binding_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved, config, runtime = _resolve_once(tmp_path / "config", monkeypatch)
    object.__setattr__(config, "input_selector", "changed-input")
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, resolved)

    resolved, config, runtime = _resolve_once(tmp_path / "pair", monkeypatch)
    object.__setattr__(resolved.pair, "input_index", 77)
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, resolved)

    resolved, config, runtime = _resolve_once(tmp_path / "runtime", monkeypatch)
    object.__setattr__(resolved, "runtime", _valid_runtime())
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, resolved)

    resolved, config, runtime = _resolve_once(tmp_path / "seam", monkeypatch)
    runtime.query_devices = lambda: []  # type: ignore[method-assign]
    with pytest.raises(LocalSpeechAdapterError):
        devices_module._validate_resolved_audio_runtime(config, resolved)


def test_runtime_seam_validation_never_invokes_descriptors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "HOSTILE_RUNTIME_DESCRIPTOR_SENTINEL"

    class HostileRuntime:
        @property
        def query_devices(self) -> object:
            print(marker)
            raise RuntimeError(marker)

        def check_input_settings(self) -> None:
            raise AssertionError

        def check_output_settings(self) -> None:
            raise AssertionError

        def RawInputStream(self) -> None:
            raise AssertionError

        def RawOutputStream(self) -> None:
            raise AssertionError

    pair = ResolvedAudioDevicePair(input_index=0, output_index=1)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        devices_module._ResolvedAudioRuntime._create(
            runtime=HostileRuntime(),
            pair=pair,
            run_root=tmp_path,
        )

    captured = capsys.readouterr()
    assert marker not in captured.out
    assert marker not in captured.err
    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_validate_resolved_audio_runtime_scrubs_hostile_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "HOSTILE_RESOLVED_MARKER"
    _resolved, config, _runtime = _resolve_once(tmp_path, monkeypatch)

    class HostileResolved:
        @property
        def pair(self) -> object:
            raise RuntimeError(marker)

        def __repr__(self) -> str:
            return marker

    hostile = HostileResolved()
    with pytest.raises(LocalSpeechAdapterError) as caught:
        devices_module._validate_resolved_audio_runtime(config, hostile)

    production_locals: list[str] = []
    for frame, _lineno in traceback.walk_tb(caught.value.__traceback__):
        if Path(frame.f_code.co_filename).name == "local_audio_devices.py":
            production_locals.extend(repr(value) for value in frame.f_locals.values())
    assert marker not in str(caught.value)
    assert marker not in "".join(production_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
