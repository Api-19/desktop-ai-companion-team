from __future__ import annotations

import importlib
import json
from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

import companion.speech.local_audio_devices as audio_devices_module
import companion.speech.local_model_config as model_config_module
from companion.runtime.settings import Settings
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_model_config import ALL_MODEL_FILES, LOCAL_SPEECH_MANIFEST
from companion.speech.runtime_config import (
    _load_validated_speech_runtime_config,
    _local_audio_config_from_runtime,
    _local_model_config_from_runtime,
    _require_exact_str,
    _resolve_contained_root,
    _ValidatedSpeechRuntimeConfig,
    preflight_local_speech_runtime,
)


def _settings(private_data_dir: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "test",
        "log_level": "INFO",
        "private_data_dir": private_data_dir,
        "runtime_data_dir": private_data_dir / "runtime_data",
        "test_results_dir": private_data_dir / "test-results",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def _toml_literal(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{" + ", ".join(f"{key} = {_toml_literal(item)}" for key, item in value.items()) + "}"
        )
    raise TypeError(f"unsupported TOML literal: {value!r}")


def _write_runtime_toml(settings: Settings, **fields: object) -> None:
    lines = [f"{key} = {_toml_literal(value)}" for key, value in fields.items()]
    speech_dir = settings.private_data_dir / "speech"
    speech_dir.mkdir(parents=True, exist_ok=True)
    (speech_dir / "runtime.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_valid_runtime(settings: Settings, **overrides: object) -> None:
    _write_runtime_toml(settings, **_valid_fields(**overrides))
    speech_dir = settings.private_data_dir / "speech"
    (speech_dir / "models").mkdir()
    (speech_dir / "run").mkdir()


def _write_manifest_shaped_model_tree(settings: Settings) -> None:
    model_root = settings.private_data_dir / "speech" / "models"
    for item in ALL_MODEL_FILES:
        path = model_root / item.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"public-synthetic-model:" + item.relative_path.as_posix().encode())


def _install_preflight_digest_results(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    *,
    failing_gate: str | None = None,
) -> None:
    model_root = settings.private_data_dir / "speech" / "models"
    expected_by_path = {
        (model_root / item.relative_path).resolve(): (item.identifier, item.sha256)
        for item in ALL_MODEL_FILES
    }

    def file_digest(path: Path) -> str:
        identifier, expected = expected_by_path[path.resolve()]
        return "0" * 64 if failing_gate == f"file:{identifier}" else expected

    def sensevoice_tree_digest(_root: Path) -> str:
        if failing_gate == "tree:sensevoice":
            return "0" * 64
        return LOCAL_SPEECH_MANIFEST.sensevoice_tree_sha256

    def vits_tree_digest(_root: Path, *, required_paths: object = None) -> str:
        gate = "tree:vits-required" if required_paths is not None else "tree:vits"
        if failing_gate == gate:
            return "0" * 64
        return (
            LOCAL_SPEECH_MANIFEST.vits_required_tree_sha256
            if required_paths is not None
            else LOCAL_SPEECH_MANIFEST.vits_tree_sha256
        )

    monkeypatch.setattr(model_config_module, "_file_sha256", file_digest)
    monkeypatch.setattr(model_config_module, "_sensevoice_tree_sha256", sensevoice_tree_digest)
    monkeypatch.setattr(model_config_module, "_vits_tree_sha256", vits_tree_digest)


def _install_preflight_side_effect_tripwires(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None) -> object:
        if name in {"sounddevice", "sherpa_onnx"}:
            pytest.fail(f"runtime import forbidden during preflight: {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    monkeypatch.setattr(
        audio_devices_module,
        "_load_sounddevice_runtime",
        lambda: pytest.fail("sounddevice runtime must not load during preflight"),
    )
    monkeypatch.setattr(
        audio_devices_module,
        "resolve_audio_devices",
        lambda _config: pytest.fail("devices must not resolve during preflight"),
    )
    monkeypatch.setattr(
        model_config_module,
        "load_sherpa_runtime",
        lambda _config: pytest.fail("sherpa runtime must not load during preflight"),
    )


def _valid_fields(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_root": "models",
        "run_root": "run",
        "host_api": "synthetic-host-api",
        "input_selector": "synthetic-input",
        "output_selector": "synthetic-output",
    }
    values.update(overrides)
    return values


def _assert_no_production_traceback_leak(error: BaseException, secret: str) -> None:
    checked = False
    traceback = error.__traceback__
    while traceback is not None:
        if (
            Path(traceback.tb_frame.f_code.co_filename)
            .as_posix()
            .endswith("src/companion/speech/runtime_config.py")
        ):
            checked = True
            assert secret not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    assert checked


def test_loader_uses_settings_private_root_and_exact_five_field_toml(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)

    result = _load_validated_speech_runtime_config(settings)

    assert type(result) is _ValidatedSpeechRuntimeConfig
    assert result.speech_root == settings.private_data_dir / "speech"
    assert result.model_root == result.speech_root / "models"
    assert result.run_root == result.speech_root / "run"
    assert result.config.model_dump() == {
        "model_root": "models",
        "run_root": "run",
        "host_api": "synthetic-host-api",
        "input_selector": "synthetic-input",
        "output_selector": "synthetic-output",
    }


def test_validated_result_repr_hides_all_fields(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)
    result = _load_validated_speech_runtime_config(settings)

    assert str(settings.private_data_dir) not in repr(result)
    assert "synthetic-host-api" not in repr(result)
    assert "synthetic-input" not in repr(result)
    assert "synthetic-output" not in repr(result)


def _assert_fails_closed(settings: Settings, secret: str) -> None:
    with pytest.raises(LocalSpeechAdapterError) as exc_info:
        _load_validated_speech_runtime_config(settings)
    error = exc_info.value
    assert type(error) is LocalSpeechAdapterError
    assert error.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert str(error) == LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED.value
    assert secret not in str(error)
    assert secret not in repr(error)


@pytest.mark.parametrize(
    "fields",
    [
        {"run_root": "run", "host_api": "h", "input_selector": "i", "output_selector": "o"},
        {"model_root": "models", "host_api": "h", "input_selector": "i", "output_selector": "o"},
        {**_valid_fields(), "unknown": "value"},
    ],
)
def test_missing_or_unknown_fields_fail_closed(tmp_path: Path, fields: dict[str, object]) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **fields)
    _assert_fails_closed(settings, str(settings.private_data_dir))


def test_duplicate_field_fails_closed(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    speech_dir = settings.private_data_dir / "speech"
    speech_dir.mkdir(parents=True)
    (speech_dir / "runtime.toml").write_text(
        'model_root = "models"\n'
        'model_root = "models"\n'
        'run_root = "run"\n'
        'host_api = "h"\n'
        'input_selector = "i"\n'
        'output_selector = "o"\n',
        encoding="utf-8",
    )
    _assert_fails_closed(settings, str(settings.private_data_dir))


@pytest.mark.parametrize(
    "field",
    [
        "model_root",
        "run_root",
        "host_api",
        "input_selector",
        "output_selector",
    ],
)
@pytest.mark.parametrize("value", [123, True, ["a"], {"nested": "value"}])
def test_non_string_values_fail_closed(tmp_path: Path, field: str, value: object) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **_valid_fields(**{field: value}))
    _assert_fails_closed(settings, str(settings.private_data_dir))


def test_loader_does_not_mask_process_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **_valid_fields())

    def interrupt(_handle: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr("companion.speech.runtime_config.tomllib.load", interrupt)

    with pytest.raises(KeyboardInterrupt):
        _load_validated_speech_runtime_config(settings)


def test_exact_settings_subclass_fails_closed(tmp_path: Path) -> None:
    class SettingsSubclass(Settings):
        pass

    settings = SettingsSubclass.model_validate(
        {
            "environment": "test",
            "log_level": "INFO",
            "private_data_dir": tmp_path / "private",
            "runtime_data_dir": tmp_path / "private" / "runtime_data",
            "test_results_dir": tmp_path / "private" / "test-results",
        }
    )
    _write_runtime_toml(settings, **_valid_fields())
    _assert_fails_closed(settings, str(settings.private_data_dir))


def test_require_exact_str_rejects_str_subclass() -> None:
    class StrSubclass(str):
        pass

    with pytest.raises(TypeError):
        _require_exact_str(StrSubclass("models"))


@pytest.mark.parametrize("field", ["host_api", "input_selector", "output_selector"])
def test_blank_selector_fails_closed(tmp_path: Path, field: str) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **_valid_fields(**{field: ""}))
    _assert_fails_closed(settings, str(settings.private_data_dir))


@pytest.mark.parametrize("field", ["host_api", "input_selector", "output_selector"])
def test_overlong_selector_fails_closed(tmp_path: Path, field: str) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **_valid_fields(**{field: "x" * 256}))
    _assert_fails_closed(settings, str(settings.private_data_dir))


@pytest.mark.parametrize("field", ["host_api", "input_selector", "output_selector"])
@pytest.mark.parametrize("value", [" padded", "padded ", " padded "])
def test_selector_surrounding_whitespace_fails_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **_valid_fields(**{field: value}))
    _assert_fails_closed(settings, str(settings.private_data_dir))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_root", "other"),
        ("run_root", "other"),
        ("model_root", "run"),
        ("run_root", "models"),
        ("model_root", "Models"),
    ],
)
def test_wrong_model_or_run_literal_fails_closed(tmp_path: Path, field: str, value: str) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **_valid_fields(**{field: value}))
    _assert_fails_closed(settings, str(settings.private_data_dir))


def test_absent_speech_root_fails_closed(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _assert_fails_closed(settings, str(settings.private_data_dir))


def test_non_directory_speech_root_fails_closed(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    (settings.private_data_dir).mkdir(parents=True)
    (settings.private_data_dir / "speech").write_text("not a directory", encoding="utf-8")
    _assert_fails_closed(settings, str(settings.private_data_dir))


@pytest.mark.parametrize("root_name", ["models", "run"])
def test_missing_model_or_run_root_fails_closed(tmp_path: Path, root_name: str) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **_valid_fields())
    speech_dir = settings.private_data_dir / "speech"
    for name in {"models", "run"} - {root_name}:
        (speech_dir / name).mkdir()
    _assert_fails_closed(settings, str(settings.private_data_dir))


@pytest.mark.parametrize("root_name", ["models", "run"])
def test_non_directory_model_or_run_root_fails_closed(tmp_path: Path, root_name: str) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_runtime_toml(settings, **_valid_fields())
    speech_dir = settings.private_data_dir / "speech"
    for name in {"models", "run"}:
        path = speech_dir / name
        if name == root_name:
            path.write_text("not a directory", encoding="utf-8")
        else:
            path.mkdir()
    _assert_fails_closed(settings, str(settings.private_data_dir))


@pytest.mark.parametrize(
    "value",
    [
        "/absolute",
        "C:\\absolute",
        "\\\\server\\share",
        "..",
        "../escape",
        "a/../../escape",
    ],
)
def test_contained_root_rejects_escape(tmp_path: Path, value: str) -> None:
    (tmp_path / "speech").mkdir()
    with pytest.raises(ValueError):
        _resolve_contained_root(tmp_path / "speech", value)


def test_contained_root_rejects_speech_root_itself(tmp_path: Path) -> None:
    speech_root = tmp_path / "speech"
    speech_root.mkdir()
    with pytest.raises(ValueError):
        _resolve_contained_root(speech_root, ".")


def test_contained_root_rejects_symlink_component(tmp_path: Path) -> None:
    speech_root = tmp_path / "speech"
    speech_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (speech_root / "models").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ValueError):
        _resolve_contained_root(speech_root, "models")


def test_link_component_in_model_root_fails_closed(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    speech_dir = settings.private_data_dir / "speech"
    speech_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (speech_dir / "models").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    _write_runtime_toml(settings, **_valid_fields())
    _assert_fails_closed(settings, str(settings.private_data_dir))


def test_two_private_roots_derive_distinct_speech_roots(tmp_path: Path) -> None:
    first = _settings(private_data_dir=tmp_path / "first")
    second = _settings(private_data_dir=tmp_path / "second")
    _write_valid_runtime(first)
    _write_valid_runtime(second)

    first_result = _load_validated_speech_runtime_config(first)
    second_result = _load_validated_speech_runtime_config(second)

    assert first_result.speech_root == first.private_data_dir / "speech"
    assert second_result.speech_root == second.private_data_dir / "speech"
    assert first_result.speech_root != second_result.speech_root


def test_speech_ignores_environment_and_non_private_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    first = _settings(
        private,
        runtime_data_dir=tmp_path / "runtime-a",
        test_results_dir=tmp_path / "results-a",
        log_level="DEBUG",
    )
    second = _settings(
        private,
        runtime_data_dir=tmp_path / "runtime-b",
        test_results_dir=tmp_path / "results-b",
        log_level="WARNING",
    )
    monkeypatch.setenv("COMPANION_PRIVATE_DATA_DIR", str(tmp_path / "decoy"))
    _write_valid_runtime(first)

    first_result = _load_validated_speech_runtime_config(first)
    second_result = _load_validated_speech_runtime_config(second)

    assert first_result.speech_root == private / "speech"
    assert second_result.speech_root == private / "speech"
    assert first_result.speech_root == second_result.speech_root


def test_configuration_failure_does_not_leak_path_or_document(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    secret_selector = "top-secret-selector"
    _write_runtime_toml(settings, **_valid_fields(host_api=secret_selector, model_root="wrong"))

    with pytest.raises(LocalSpeechAdapterError) as exc_info:
        _load_validated_speech_runtime_config(settings)
    error = exc_info.value

    assert type(error) is LocalSpeechAdapterError
    assert error.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert str(error) == LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert str(settings.private_data_dir) not in str(error)
    assert str(settings.private_data_dir) not in repr(error)
    assert secret_selector not in str(error)
    assert secret_selector not in repr(error)
    assert secret_selector not in caplog.text
    captured = capsys.readouterr()
    assert secret_selector not in captured.out
    assert secret_selector not in captured.err
    _assert_no_production_traceback_leak(error, secret_selector)
    _assert_no_production_traceback_leak(error, str(settings.private_data_dir))


def test_private_factories_bind_manifest_and_exact_runtime_config(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)
    validated = _load_validated_speech_runtime_config(settings)

    model_config = _local_model_config_from_runtime(settings, validated)
    audio_config = _local_audio_config_from_runtime(settings, validated, model_config)

    assert model_config.model_root == validated.model_root
    assert model_config.run_root == validated.run_root
    assert model_config.sensevoice_identifier == LOCAL_SPEECH_MANIFEST.sensevoice_identifier
    assert model_config.vits_identifier == LOCAL_SPEECH_MANIFEST.vits_identifier
    assert model_config.sensevoice_model_sha256 == LOCAL_SPEECH_MANIFEST.sensevoice_model.sha256
    assert model_config.vits_required_tree_sha256 == (
        LOCAL_SPEECH_MANIFEST.vits_required_tree_sha256
    )
    assert audio_config.run_root == model_config.run_root
    assert audio_config.hostapi_selector == "synthetic-host-api"
    assert audio_config.input_selector == "synthetic-input"
    assert audio_config.output_selector == "synthetic-output"


def test_private_factories_reject_stale_or_copied_runtime_capability(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "first")
    _write_valid_runtime(settings)
    validated = _load_validated_speech_runtime_config(settings)
    for copied in (
        replace(validated),
        copy(validated),
        object.__new__(_ValidatedSpeechRuntimeConfig),
    ):
        with pytest.raises(LocalSpeechAdapterError):
            _local_model_config_from_runtime(settings, copied)

    second = tmp_path / "second"
    (second / "speech" / "models").mkdir(parents=True)
    (second / "speech" / "run").mkdir()
    settings.private_data_dir = second

    with pytest.raises(LocalSpeechAdapterError):
        _local_model_config_from_runtime(settings, validated)


def test_audio_factory_requires_exact_model_instance_and_current_settings(tmp_path: Path) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)
    validated = _load_validated_speech_runtime_config(settings)
    model_config = _local_model_config_from_runtime(settings, validated)
    for candidate in (replace(model_config), copy(model_config), object()):
        with pytest.raises(LocalSpeechAdapterError):
            _local_audio_config_from_runtime(settings, validated, candidate)

    object.__setattr__(model_config, "run_root", tmp_path / "wrong-run")
    with pytest.raises(LocalSpeechAdapterError):
        _local_audio_config_from_runtime(settings, validated, model_config)


def test_factories_reject_path_subclasses_without_invoking_comparison_hooks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "PRIVATE_PATH_COMPARISON_SENTINEL"

    class EvilPath(type(Path())):
        def __eq__(self, other: object) -> bool:
            print(marker)
            return super().__eq__(other)

        def __ne__(self, other: object) -> bool:
            print(marker)
            return super().__ne__(other)

    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)
    validated = _load_validated_speech_runtime_config(settings)
    object.__setattr__(validated, "speech_root", EvilPath(validated.speech_root))

    with pytest.raises(LocalSpeechAdapterError):
        _local_model_config_from_runtime(settings, validated)

    captured = capsys.readouterr()
    assert marker not in captured.out
    assert marker not in captured.err


def test_factories_reject_settings_path_subclass_without_invoking_path_hooks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "PRIVATE_SETTINGS_PATH_SENTINEL"

    class EvilPath(type(Path())):
        def __truediv__(self, key: object) -> Path:
            print(marker)
            return super().__truediv__(key)

    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)
    validated = _load_validated_speech_runtime_config(settings)
    settings.private_data_dir = EvilPath(settings.private_data_dir)

    with pytest.raises(LocalSpeechAdapterError):
        _local_model_config_from_runtime(settings, validated)

    captured = capsys.readouterr()
    assert marker not in captured.out
    assert marker not in captured.err


def test_preflight_verifies_models_without_loading_runtimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)
    calls: list[object] = []

    def verify(config: object) -> object:
        calls.append(config)
        return object()

    monkeypatch.setattr("companion.speech.local_model_config.verify_local_speech_config", verify)
    monkeypatch.setattr(
        "companion.speech.local_model_config.load_sherpa_runtime",
        lambda _config: pytest.fail("model runtime must not load during preflight"),
    )
    monkeypatch.setattr(
        "companion.speech.local_audio_devices._load_sounddevice_runtime",
        lambda: pytest.fail("device runtime must not load during preflight"),
    )

    assert preflight_local_speech_runtime(settings) is None
    assert len(calls) == 1


def test_preflight_consumes_manifest_tree_without_runtime_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)
    _write_manifest_shaped_model_tree(settings)
    _install_preflight_digest_results(monkeypatch, settings)
    _install_preflight_side_effect_tripwires(monkeypatch)

    assert preflight_local_speech_runtime(settings) is None


@pytest.mark.parametrize(
    "failing_gate",
    [
        *(f"file:{item.identifier}" for item in ALL_MODEL_FILES),
        "tree:sensevoice",
        "tree:vits",
        "tree:vits-required",
    ],
)
def test_preflight_fails_closed_for_each_manifest_digest_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_gate: str,
) -> None:
    settings = _settings(private_data_dir=tmp_path / "private")
    _write_valid_runtime(settings)
    _write_manifest_shaped_model_tree(settings)
    _install_preflight_digest_results(monkeypatch, settings, failing_gate=failing_gate)
    _install_preflight_side_effect_tripwires(monkeypatch)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        preflight_local_speech_runtime(settings)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED


def test_preflight_rejects_settings_drift_before_model_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(private_data_dir=tmp_path / "first")
    _write_valid_runtime(settings)
    real_loader = _load_validated_speech_runtime_config
    second = tmp_path / "second"
    (second / "speech" / "models").mkdir(parents=True)
    (second / "speech" / "run").mkdir()
    verify_calls = 0

    def drifting_loader(exact_settings: Settings) -> _ValidatedSpeechRuntimeConfig:
        result = real_loader(exact_settings)
        exact_settings.private_data_dir = second
        return result

    def forbidden_verify(_config: object) -> object:
        nonlocal verify_calls
        verify_calls += 1
        raise AssertionError("verification must not run for stale settings")

    monkeypatch.setattr(
        "companion.speech.runtime_config._load_validated_speech_runtime_config",
        drifting_loader,
    )
    monkeypatch.setattr(
        "companion.speech.local_model_config.verify_local_speech_config",
        forbidden_verify,
    )

    with pytest.raises(LocalSpeechAdapterError):
        preflight_local_speech_runtime(settings)
    assert verify_calls == 0
