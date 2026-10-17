"""Only synthetic fresh-run outputs; never read real blind-evaluation artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from companion.dialogue.ab import _make_private_run_dir, _write_private_json


def test_fresh_private_run_dir_permissions_under_permissive_umask(tmp_path: Path) -> None:
    previous = os.umask(0)
    try:
        directory = _make_private_run_dir(tmp_path, "synthetic")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


@pytest.mark.parametrize("name", ("answers.json", "answer-key.json", "smoke.json"))
def test_fresh_private_json_permissions_under_permissive_umask(tmp_path: Path, name: str) -> None:
    previous = os.umask(0)
    try:
        target = tmp_path / name
        _write_private_json(target, {"synthetic": True})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.parametrize("kind", ("file", "symlink", "hardlink"))
def test_private_json_refuses_existing_targets_without_overwrite(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "answer-key.json"
    external = tmp_path / "synthetic-external.json"
    external.write_bytes(b"synthetic preserved bytes")
    if kind == "file":
        target.write_bytes(b"synthetic preserved bytes")
    elif kind == "symlink":
        target.symlink_to(external)
    else:
        os.link(external, target)
    with pytest.raises(FileExistsError):
        _write_private_json(target, {"synthetic": "replacement"})
    assert target.read_bytes() == b"synthetic preserved bytes"
    assert external.read_bytes() == b"synthetic preserved bytes"


def test_private_run_dir_refuses_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import companion.dialogue.ab as ab

    monkeypatch.setattr(ab.secrets, "token_hex", lambda _: "synthetic-fixed")
    target = tmp_path / "synthetic-synthetic-fixed"
    target.mkdir()
    with pytest.raises(FileExistsError):
        _make_private_run_dir(tmp_path, "synthetic")


def test_private_run_dir_refuses_dangling_symlink_without_creating_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import companion.dialogue.ab as ab

    monkeypatch.setattr(ab.secrets, "token_hex", lambda _: "synthetic-fixed")
    target = tmp_path / "synthetic-synthetic-fixed"
    destination = tmp_path / "synthetic-other"
    target.symlink_to(destination, target_is_directory=True)
    with pytest.raises(FileExistsError):
        _make_private_run_dir(tmp_path, "synthetic")
    assert target.is_symlink()
    assert not destination.exists()
