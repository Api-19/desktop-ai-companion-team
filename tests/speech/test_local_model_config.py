from __future__ import annotations

import hashlib
import importlib.metadata
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

import companion.speech.local_model_config as config_module
from companion.runtime.settings import Settings
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_model_config import (
    ALL_MODEL_FILES,
    LOCAL_SPEECH_MANIFEST,
    SHERPA_ONNX_CORE_VERSION,
    SHERPA_ONNX_VERSION,
    LocalSpeechModelConfig,
    VerifiedLocalSpeechPaths,
    load_sherpa_runtime,
    verify_local_speech_config,
)


def _write_model_files(model_root: Path) -> None:
    for item in ALL_MODEL_FILES:
        path = model_root / item.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-model-file:" + item.relative_path.as_posix().encode())


def make_config(tmp_path: Path, **overrides: object) -> LocalSpeechModelConfig:
    model_root = tmp_path / "models"
    run_root = tmp_path / "run"
    _write_model_files(model_root)
    run_root.mkdir(parents=True, exist_ok=True)
    declarations: dict[str, object] = {
        "model_root": model_root,
        "run_root": run_root,
        "sensevoice_identifier": LOCAL_SPEECH_MANIFEST.sensevoice_identifier,
        "silero_identifier": LOCAL_SPEECH_MANIFEST.silero_identifier,
        "vits_identifier": LOCAL_SPEECH_MANIFEST.vits_identifier,
        "sensevoice_model_relative_path": LOCAL_SPEECH_MANIFEST.sensevoice_model.relative_path,
        "sensevoice_tokens_relative_path": LOCAL_SPEECH_MANIFEST.sensevoice_tokens.relative_path,
        "silero_model_relative_path": LOCAL_SPEECH_MANIFEST.silero_model.relative_path,
        "vits_model_relative_path": LOCAL_SPEECH_MANIFEST.vits_model.relative_path,
        "vits_lexicon_relative_path": LOCAL_SPEECH_MANIFEST.vits_lexicon.relative_path,
        "vits_tokens_relative_path": LOCAL_SPEECH_MANIFEST.vits_tokens.relative_path,
        "vits_date_fst_relative_path": LOCAL_SPEECH_MANIFEST.vits_date_fst.relative_path,
        "vits_number_fst_relative_path": LOCAL_SPEECH_MANIFEST.vits_number_fst.relative_path,
        "sherpa_onnx_version": SHERPA_ONNX_VERSION,
        "sherpa_onnx_core_version": SHERPA_ONNX_CORE_VERSION,
        "sensevoice_model_sha256": LOCAL_SPEECH_MANIFEST.sensevoice_model.sha256,
        "sensevoice_tokens_sha256": LOCAL_SPEECH_MANIFEST.sensevoice_tokens.sha256,
        "silero_model_sha256": LOCAL_SPEECH_MANIFEST.silero_model.sha256,
        "vits_model_sha256": LOCAL_SPEECH_MANIFEST.vits_model.sha256,
        "vits_lexicon_sha256": LOCAL_SPEECH_MANIFEST.vits_lexicon.sha256,
        "vits_tokens_sha256": LOCAL_SPEECH_MANIFEST.vits_tokens.sha256,
        "vits_date_fst_sha256": LOCAL_SPEECH_MANIFEST.vits_date_fst.sha256,
        "vits_number_fst_sha256": LOCAL_SPEECH_MANIFEST.vits_number_fst.sha256,
        "sensevoice_tree_sha256": LOCAL_SPEECH_MANIFEST.sensevoice_tree_sha256,
        "vits_tree_sha256": LOCAL_SPEECH_MANIFEST.vits_tree_sha256,
        "vits_required_tree_sha256": LOCAL_SPEECH_MANIFEST.vits_required_tree_sha256,
    }
    declarations.update(overrides)
    return LocalSpeechModelConfig(**declarations)  # type: ignore[arg-type]


def _patch_passing_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_by_path = {item.relative_path.as_posix(): item.sha256 for item in ALL_MODEL_FILES}

    def fake_file_digest(path: Path) -> str:
        for relative_path, digest in expected_by_path.items():
            if path.as_posix().endswith(relative_path):
                return digest
        raise AssertionError("unexpected model path")

    monkeypatch.setattr(config_module, "_file_sha256", fake_file_digest)
    monkeypatch.setattr(
        config_module,
        "_sensevoice_tree_sha256",
        lambda _root: LOCAL_SPEECH_MANIFEST.sensevoice_tree_sha256,
    )
    monkeypatch.setattr(
        config_module,
        "_vits_tree_sha256",
        lambda _root, *, required_paths=None: (
            LOCAL_SPEECH_MANIFEST.vits_required_tree_sha256
            if required_paths is not None
            else LOCAL_SPEECH_MANIFEST.vits_tree_sha256
        ),
    )


def test_error_uses_only_a_fixed_code() -> None:
    error = LocalSpeechAdapterError(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)

    assert str(error) == "configuration-failed"
    assert error.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert error.args == ("configuration-failed",)
    with pytest.raises(TypeError):
        LocalSpeechAdapterError("configuration-failed")  # type: ignore[arg-type]


def test_manifest_freezes_the_approved_model_facts() -> None:
    assert SHERPA_ONNX_VERSION == "1.13.4"
    assert SHERPA_ONNX_CORE_VERSION == "1.13.4"
    assert LOCAL_SPEECH_MANIFEST.sensevoice_model.sha256 == (
        "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"
    )
    assert LOCAL_SPEECH_MANIFEST.sensevoice_tokens.sha256 == (
        "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"
    )
    assert LOCAL_SPEECH_MANIFEST.silero_model.sha256 == (
        "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
    )
    assert LOCAL_SPEECH_MANIFEST.sensevoice_tree_sha256 == (
        "782cbe9421d975b92f8484df9a7c751cdd2f475866f595c16034f9a978881e19"
    )
    assert LOCAL_SPEECH_MANIFEST.vits_tree_sha256 == (
        "7c4491c59e61659fca2a1e1555bef674a098ddde4be131300437e8fad0dfe1f4"
    )
    assert LOCAL_SPEECH_MANIFEST.vits_required_tree_sha256 == (
        "e4683fd2a372235d02972eb6f85619597892ca10ac00bd4daf66f22390356ca5"
    )
    assert LOCAL_SPEECH_MANIFEST.vits_archive_sha256 == (
        "e58351ed7149f290a54534538badd4077cdbe6fddc964b24d0bee870415d1514"
    )
    assert LOCAL_SPEECH_MANIFEST.vits_archive_license_id == "MIT"
    assert LOCAL_SPEECH_MANIFEST.vits_archive_source_url == (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
        "vits-melo-tts-zh_en.tar.bz2"
    )
    assert [item.sha256 for item in LOCAL_SPEECH_MANIFEST.vits_required_files] == [
        "bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d",
        "7236884b02435ac5d10cf69b4be40a61b45aa676b5300f0e412f185748fee528",
        "d18664a7e12bd7ea1022ddaf951e534e136815016c5a809d6b64156bffb4369d",
        "eb8aa079ae3cb81d8f4404992f39d61a0cb990947512b5b8d1e54d1f6980e718",
        "743f402181fcfebf76cc2f0546b71fa26476e626fbe4e460fb7b4c3a7a8bd5bd",
    ]
    assert [item.license_id for item in ALL_MODEL_FILES] == [
        "FunASR-Model-License-1.1",
        "FunASR-Model-License-1.1",
        "MIT",
        "MIT",
        "MIT",
        "MIT",
        "MIT",
        "MIT",
    ]
    assert all(
        item.source_url.startswith("https://github.com/k2-fsa/sherpa-onnx/")
        for item in ALL_MODEL_FILES
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sensevoice_model_sha256", "0" * 64),
        ("sherpa_onnx_version", "1.13.3"),
    ],
)
def test_config_requires_the_approved_manifest(tmp_path: Path, field: str, value: str) -> None:
    with pytest.raises(LocalSpeechAdapterError) as caught:
        make_config(tmp_path, **{field: value})

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert caught.value.__cause__ is None
    assert str(tmp_path) not in str(caught.value)


def test_config_rejects_relative_or_actual_worktree_roots_outside_the_checkout_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    actual_worktree = Path(config_module.__file__).resolve().parents[3]
    monkeypatch.chdir(tmp_path)

    with pytest.raises(LocalSpeechAdapterError):
        replace(config, model_root=Path("relative-models"))
    with pytest.raises(LocalSpeechAdapterError):
        replace(config, run_root=Path("relative-run"))
    with pytest.raises(LocalSpeechAdapterError):
        replace(config, model_root=actual_worktree / "models")
    with pytest.raises(LocalSpeechAdapterError):
        replace(config, run_root=actual_worktree / "run")


def test_config_requires_exact_manifest_identifiers_at_construction(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    class IdentifierSubclass(str):
        pass

    with pytest.raises(LocalSpeechAdapterError):
        replace(config, sensevoice_identifier="other-sensevoice")
    with pytest.raises(LocalSpeechAdapterError):
        replace(config, silero_identifier=IdentifierSubclass("silero-vad-onnx"))
    with pytest.raises(LocalSpeechAdapterError):
        replace(config, vits_identifier="other-vits")


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_root", "not-a-path"),
        ("run_root", Path()),
        ("sensevoice_model_relative_path", Path()),
        ("sensevoice_model_relative_path", Path("../outside.onnx")),
    ],
)
def test_config_rejects_non_exact_or_empty_declarations(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError, LocalSpeechAdapterError)):
        make_config(tmp_path, **{field: value})


def test_config_rejects_roots_that_are_equal_or_inside_the_git_worktree(tmp_path: Path) -> None:
    valid_config = make_config(tmp_path)

    with pytest.raises(LocalSpeechAdapterError):
        replace(valid_config, model_root=Path.cwd())
    with pytest.raises(LocalSpeechAdapterError):
        replace(valid_config, run_root=valid_config.model_root)


def test_verification_returns_only_resolved_internal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    _patch_passing_digests(monkeypatch)

    paths = verify_local_speech_config(config)

    assert type(paths) is VerifiedLocalSpeechPaths
    assert (
        paths.sensevoice_model
        == (config.model_root / config.sensevoice_model_relative_path).resolve()
    )
    assert paths.vits_rule_fsts == (
        (config.model_root / config.vits_date_fst_relative_path).resolve(),
        (config.model_root / config.vits_number_fst_relative_path).resolve(),
    )
    assert paths.run_root == config.run_root.resolve()
    assert str(config.model_root) not in repr(paths)
    assert str(config.run_root) not in repr(paths)


def test_verification_fails_closed_for_missing_or_non_regular_model_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    (config.model_root / config.silero_model_relative_path).unlink()

    with pytest.raises(LocalSpeechAdapterError) as caught:
        verify_local_speech_config(config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    (config.model_root / config.silero_model_relative_path).mkdir()
    with pytest.raises(LocalSpeechAdapterError):
        verify_local_speech_config(config)


def test_verification_rejects_actual_file_digest_mismatch(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(LocalSpeechAdapterError):
        verify_local_speech_config(config)


def test_real_file_digest_changes_after_a_one_byte_mutation(tmp_path: Path) -> None:
    file = tmp_path / "tiny.bin"
    file.write_bytes(b"a")
    before = config_module._file_sha256(file)
    file.write_bytes(b"b")

    assert config_module._file_sha256(file) != before


def test_model_tree_order_is_casefolded_and_cross_platform_stable(tmp_path: Path) -> None:
    root = tmp_path / "model"
    (root / "test_wavs").mkdir(parents=True)
    for relative in (
        "README.md",
        "tokens.txt",
        "LICENSE",
        "model.int8.onnx",
        "export-onnx.py",
        "test_wavs/zh.wav",
    ):
        (root / relative).write_bytes(relative.encode("ascii"))

    ordered = tuple(
        relative.as_posix()
        for relative, _path in config_module._regular_tree_files(root, casefold_order=True)
    )

    expected = (
        "export-onnx.py",
        "LICENSE",
        "model.int8.onnx",
        "README.md",
        "test_wavs/zh.wav",
        "tokens.txt",
    )
    assert ordered == expected
    digest = hashlib.sha256()
    for relative in expected:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256((root / relative).read_bytes()).digest())
    assert config_module._sensevoice_tree_sha256(root) == digest.hexdigest()
    assert tuple(
        relative.as_posix() for relative, _path in config_module._regular_tree_files(root)
    ) == tuple(sorted(ordered))


def test_verification_rejects_symbolic_link_model_file_when_supported(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    target = tmp_path / "outside.onnx"
    target.write_bytes(b"synthetic")
    linked = config.model_root / config.silero_model_relative_path
    linked.unlink()
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable in this Windows test environment")

    with pytest.raises(LocalSpeechAdapterError):
        verify_local_speech_config(config)


def test_failed_verification_has_no_path_or_traceback_output_leakage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "MODEL_ROOT_SENTINEL"
    config = make_config(tmp_path / marker)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        verify_local_speech_config(config)

    formatted = "".join(traceback.format_exception(caught.value))
    captured = capsys.readouterr()
    assert marker not in str(caught.value)
    assert marker not in formatted
    assert marker not in captured.out
    assert marker not in captured.err
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert not hasattr(caught.value, "detail")


def test_missing_or_wrong_sherpa_version_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.13.3")

    with pytest.raises(LocalSpeechAdapterError) as caught:
        load_sherpa_runtime(config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_runtime_checks_both_packages_then_imports_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    seen_packages: list[str] = []
    runtime = object()

    def fake_version(name: str) -> str:
        seen_packages.append(name)
        return "1.13.4"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    monkeypatch.setattr(config_module.importlib, "import_module", lambda name: runtime)

    assert load_sherpa_runtime(config) is runtime
    assert seen_packages == ["sherpa-onnx", "sherpa-onnx-core"]


def test_private_model_constructor_requires_runtime_capability() -> None:
    helper = getattr(config_module, "_model_config_from_validated_runtime", None)
    assert callable(helper)

    with pytest.raises(LocalSpeechAdapterError):
        helper(object(), object())


def test_private_model_constructor_failure_scrubs_settings_and_capability(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "PRIVATE_MODEL_FACTORY_SENTINEL"
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

    with pytest.raises(LocalSpeechAdapterError) as caught:
        config_module._model_config_from_validated_runtime(settings, capability)

    production_locals: list[str] = []
    for frame, _lineno in traceback.walk_tb(caught.value.__traceback__):
        if Path(frame.f_code.co_filename).name == "local_model_config.py":
            production_locals.extend(repr(item) for item in frame.f_locals.values())
    captured = capsys.readouterr()
    assert marker not in "".join(production_locals)
    assert marker not in captured.out
    assert marker not in captured.err
    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
