# ruff: noqa: RUF001

"""Character settings use only disposable, synthetic configuration files."""

import builtins
import hashlib
import io
import json
import os
import socket
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from test_config import make_settings

import companion.dialogue.character_settings as character_settings
import companion.dialogue.config as dialogue_config
from companion.dialogue.character_settings import (
    CharacterSettingsConflict,
    CharacterSettingsInvalidName,
    CharacterSettingsUnavailable,
    read_character_name,
    write_character_name,
)
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.provider import OpenAICompatibleProvider
from companion.runtime.settings import Settings


@pytest.fixture
def config_file(tmp_path: Path) -> tuple[Settings, Path]:
    settings = make_settings(tmp_path)
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"character_name": "kiki"}')
    return settings, path


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected side effect")


def test_character_write_preserves_other_fields(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    path.parent.mkdir(parents=True)
    payload = {
        "character_name": "kiki",
        "provider": {"nested": {"api_key_env": "SYNTHETIC_KEY", "options": [1, True, None]}},
        "budget": {"monthly_cap_cny": "100.00", "unknown": {"enabled": False}},
        "style": {"source": "synthetic/style.json", "nested": ["合成", {"count": 2}]},
        "synthetic_unknown": {"array": [None, 1.5, {"preserve": "合成值"}]},
    }
    original = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(original)

    assert read_character_name(settings) == ("kiki", hashlib.sha256(original).hexdigest())
    digest = write_character_name(
        settings, " 合成角色 ", expected_digest=hashlib.sha256(original).hexdigest()
    )

    expected = {**payload, "character_name": "合成角色"}
    written = path.read_bytes()
    assert json.loads(written) == expected
    assert written == (json.dumps(expected, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert digest == hashlib.sha256(written).hexdigest()
    assert read_character_name(settings) == ("合成角色", digest)


@pytest.mark.parametrize("raw", (b"{}", b'{ "character_name" : " kiki " }\n'))
def test_character_read_uses_one_raw_read(
    config_file: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    settings, path = config_file
    path.write_bytes(raw)
    reads = []
    original = Path.read_bytes

    def read_bytes(candidate):
        reads.append(candidate)
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert read_character_name(settings) == ("kiki", hashlib.sha256(raw).hexdigest())
    assert reads == [path]


@pytest.mark.parametrize("new_name", ("kiki", "合成角色"))
@pytest.mark.parametrize("change", (b"\n", b' , "synthetic_unknown": true'))
def test_character_write_rejects_stale_digest(
    config_file: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch,
    new_name: str, change: bytes,
) -> None:
    settings, path = config_file
    _, digest = read_character_name(settings)
    changed = path.read_bytes()[:-1] + change + b"}"
    path.write_bytes(changed)
    monkeypatch.setattr(character_settings, "NamedTemporaryFile", forbidden)
    with pytest.raises(CharacterSettingsConflict, match=r"^角色名称配置已变化，请重新查看$"):
        write_character_name(settings, new_name, expected_digest=digest)
    assert path.read_bytes() == changed


@pytest.mark.parametrize(
    ("payload", "name"),
    (({"character_name": "kiki"}, " kiki "),
     ({"character_name": " kiki "}, "kiki"), ({}, "kiki"), ({}, " kiki ")),
)
def test_same_character_name_is_noop(
    config_file: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch,
    payload: dict, name: str,
) -> None:
    settings, path = config_file
    original = json.dumps(payload).encode("utf-8") + b"\n\n"
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    monkeypatch.setattr(character_settings, "NamedTemporaryFile", forbidden)
    monkeypatch.setattr(character_settings.os, "replace", forbidden)
    assert write_character_name(settings, name, expected_digest=digest) == digest
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "fault",
    ("missing_directory", "missing_file", "unreadable", "directory", "utf8", "json",
     "array", "null", "invalid_name", "non_string_name", "control_name", "format_name"),
)
def test_character_settings_reject_missing_or_corrupt_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    settings = make_settings(tmp_path)
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    if fault != "missing_directory":
        path.parent.mkdir(parents=True)
    raw = {
        "utf8": b"\xff", "json": b"{", "array": b"[]", "null": b"null",
        "invalid_name": b'{"character_name": ""}',
        "non_string_name": b'{"character_name": null}',
        "control_name": b'{"character_name": "ki\\nki"}',
        "format_name": b'{"character_name": "ki\\u200bki"}',
    }
    if fault in raw:
        path.write_bytes(raw[fault])
    elif fault == "directory":
        path.mkdir()
    elif fault == "unreadable":
        path.write_bytes(b"{}")

        def fail_read(candidate):
            assert candidate == path
            raise PermissionError(str(path))

        monkeypatch.setattr(Path, "read_bytes", fail_read)
    before = sorted(tmp_path.rglob("*"))
    for operation in (
        lambda: read_character_name(settings),
        lambda: write_character_name(settings, "合成角色", expected_digest="stale"),
    ):
        with pytest.raises(CharacterSettingsUnavailable, match=r"^角色名称配置不可用$"):
            operation()
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("name", ("", " ", "x" * 33, "ki\nki", "ki\u200bki", None, 123))
def test_character_write_validates_name_before_reading(tmp_path: Path, name: object) -> None:
    with pytest.raises(CharacterSettingsInvalidName, match=r"^角色名称无效$"):
        write_character_name(make_settings(tmp_path), name, expected_digest="stale")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", ("write", "close", "replace"))
def test_character_write_cleans_temporary_on_failure(
    config_file: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    settings, path = config_file
    original = path.read_bytes()
    _, digest = read_character_name(settings)
    created = []

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream
            self.name = stream.name

        def __enter__(self):
            self.stream.__enter__()
            return self

        def write(self, data):
            if fault == "write":
                self.stream.write(data[:3])
                raise OSError("synthetic write failure")
            return self.stream.write(data)

        def __exit__(self, *args):
            self.stream.__exit__(*args)
            if fault == "close":
                raise OSError("synthetic close failure")

    def temporary_file(*args, **kwargs):
        assert kwargs["dir"] == path.parent
        assert kwargs["prefix"] == f".{path.name}."
        assert kwargs["delete"] is False
        stream = NamedTemporaryFile(*args, **kwargs)  # noqa: SIM115 - caller manages context
        created.append(Path(stream.name))
        return FailingStream(stream)

    def replace(source, destination):
        assert fault == "replace"
        assert Path(source) == created[0]
        assert Path(destination) == path
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(character_settings, "NamedTemporaryFile", temporary_file)
    monkeypatch.setattr(character_settings.os, "replace", replace)
    with pytest.raises(CharacterSettingsUnavailable, match=r"^角色名称配置不可用$"):
        write_character_name(settings, "合成角色", expected_digest=digest)
    assert len(created) == 1
    assert not created[0].exists()
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("replace_fails", (False, True))
def test_character_write_is_confined_and_cleans_temporary(
    config_file: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch, replace_fails: bool
) -> None:
    settings, path = config_file
    sentinels = [
        settings.private_data_dir / "memory" / "synthetic.json",
        settings.private_data_dir / "style" / "synthetic.json",
        settings.runtime_data_dir / "dialogue" / "usage.json",
        settings.test_results_dir / "synthetic.json",
    ]
    for sentinel in sentinels:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(b"synthetic untouched")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    created, opened, replaced, unlinked = [], [], [], []
    original_io_open, original_open = io.open, os.open
    original_replace, original_unlink = os.replace, os.unlink

    def confined_open(file, mode="r", *args, **kwargs):
        if "opener" in kwargs:
            # NamedTemporaryFile delegates actual creation to its exclusive os.open opener.
            assert Path(file) == path.parent
            assert mode == "wb"
        else:
            assert Path(file) == path
            assert mode == "rb"
            opened.append(path)
        return original_io_open(file, mode, *args, **kwargs)

    def exclusive_open(file, flags, mode=0o777, *, dir_fd=None):
        temporary = Path(file)
        assert dir_fd is None
        assert temporary.parent == path.parent
        assert temporary.name.startswith(f".{path.name}.")
        assert flags & os.O_EXCL and flags & os.O_CREAT
        created.append(temporary)
        return original_open(file, flags, mode)

    def replace(source, destination):
        assert Path(source) == created[0]
        assert Path(destination) == path
        replaced.append((Path(source), Path(destination)))
        if replace_fails:
            raise OSError("synthetic replace failure")
        return original_replace(source, destination)

    def unlink(file, *args, **kwargs):
        assert Path(file) == created[0]
        unlinked.append(Path(file))
        return original_unlink(file, *args, **kwargs)

    class NoEnvironment(dict):
        __getitem__ = get = __iter__ = forbidden

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", confined_open)
        patch.setattr(builtins, "open", confined_open)
        patch.setattr(os, "open", exclusive_open)
        patch.setattr(os, "replace", replace)
        patch.setattr(os, "unlink", unlink)
        patch.setattr(os, "mkdir", forbidden)
        patch.setattr(os, "environ", NoEnvironment())
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(dialogue_config, "load_dialogue_config", forbidden)
        patch.setattr(OpenAICompatibleProvider, "__init__", forbidden)
        patch.setattr(DialogueOrchestrator, "__init__", forbidden)
        if replace_fails:
            with pytest.raises(CharacterSettingsUnavailable):
                write_character_name(settings, "合成角色", expected_digest=digest)
        else:
            result = write_character_name(settings, "合成角色", expected_digest=digest)
    assert opened == [path]
    assert len(created) == len(replaced) == len(unlinked) == 1
    assert list(path.parent.iterdir()) == [path]
    for sentinel in sentinels:
        assert sentinel.read_bytes() == b"synthetic untouched"
    if not replace_fails:
        assert result == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("operation", "fault"),
    (("read", "read"), ("read", "json"), ("read", "utf8"), ("read", "invalid_name"),
     ("write", "read"), ("write", "json"), ("write", "replace"), ("write", "encode"),
     ("write", "create")),
)
def test_character_settings_errors_hide_config_and_paths(
    config_file: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture,
    operation: str, fault: str,
) -> None:
    settings, path = config_file
    config_canary, key_canary = "SYNTHETIC_CONFIG_CANARY_72", "SYNTHETIC_KEY_CANARY_93"
    payload = {"character_name": "kiki", config_canary: key_canary}
    if fault == "json":
        path.write_text(f"{{{config_canary} {key_canary}", encoding="utf-8")
    elif fault == "utf8":
        path.write_bytes(f"{config_canary} {key_canary}".encode() + b"\xff")
    elif fault == "invalid_name":
        payload["character_name"] = config_canary + "\n" + key_canary
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif fault == "encode":
        payload[config_canary] = "\ud800" + key_canary
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def fail(*args, **kwargs):
        raise OSError(f"{path}: {config_canary} {key_canary}")

    if fault == "read":
        monkeypatch.setattr(Path, "read_bytes", fail)
    elif fault == "replace":
        monkeypatch.setattr(character_settings.os, "replace", fail)
    elif fault == "create":
        monkeypatch.setattr(character_settings, "NamedTemporaryFile", fail)

    with pytest.raises(CharacterSettingsUnavailable, match=r"^角色名称配置不可用$") as caught:
        if operation == "read":
            read_character_name(settings)
        else:
            write_character_name(settings, "合成角色", expected_digest=digest)
    captured = capsys.readouterr()
    exposed = (
        str(caught.value) + repr(caught.value)
        + "".join(traceback.format_exception(caught.value))
        + captured.out + captured.err + caplog.text
    )
    for secret in (config_canary, key_canary, str(path), str(settings.private_data_dir)):
        assert secret not in exposed
    assert caught.value.__cause__ is None
    assert captured.out == captured.err == caplog.text == ""
