"""The collaboration preflight must inspect new files before their first commit."""

import subprocess
from pathlib import Path

import pytest

from scripts.verify_no_private_data import Finding, scan_repository


def repository(path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def test_untracked_secret_is_checked_without_echoing_the_secret(tmp_path):
    root = repository(tmp_path)
    secret = "sk-" + "a" * 30
    (root / "accidental.txt").write_text(secret)
    findings = scan_repository(root, include_untracked=True)
    assert findings == [Finding("accidental.txt", 1, "api-key-pattern")]
    assert secret not in repr(findings)


@pytest.mark.parametrize("name", [
    "test-results/run.txt", "private_data/style/card.txt", "backup/private_data/style.json",
    "backup/habit_card.txt", "backup/owner.sqlite3", "backup/client.key",
])
def test_unignored_private_artifacts_are_rejected(tmp_path, name):
    root = repository(tmp_path)
    file = root / name
    file.parent.mkdir(parents=True)
    file.write_text("synthetic")
    assert scan_repository(root, include_untracked=True) == [
        Finding(name, None, "forbidden-path")
    ]


def test_symlink_is_reported_without_reading_its_target(tmp_path):
    root = repository(tmp_path / "repo")
    (root / "outside.txt").symlink_to(tmp_path / "does-not-exist")
    assert scan_repository(root, include_untracked=True) == [
        Finding("outside.txt", None, "symlink")
    ]


def test_ignored_private_material_is_not_opened(tmp_path):
    root = repository(tmp_path)
    (root / ".gitignore").write_text("private_data/\n")
    folder = root / "private_data"
    folder.mkdir()
    (folder / "unreadable").symlink_to(tmp_path / "does-not-exist")
    assert scan_repository(root, include_untracked=True) == []


def test_machine_account_path_is_rejected(tmp_path):
    root = repository(tmp_path)
    (root / "notes.md").write_text("/" + "Users/" + "synthetic-account/notes")
    assert scan_repository(root, include_untracked=True) == [
        Finding("notes.md", 1, "unix-user-directory")
    ]
