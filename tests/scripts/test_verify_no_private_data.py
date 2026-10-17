import subprocess
from pathlib import Path

from scripts.verify_no_private_data import Finding, scan_repository


def git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def init_repo(path: Path) -> None:
    git(path, "init", "-q")


def test_scanner_reports_rule_without_echoing_matched_value(tmp_path: Path) -> None:
    init_repo(tmp_path)
    secret = "sk-" + "synthetic-secret-material-1234567890"
    (tmp_path / "tracked.txt").write_text(secret, encoding="utf-8")
    git(tmp_path, "add", "tracked.txt")
    findings = scan_repository(tmp_path)
    assert len(findings) == 1
    assert findings[0].rule == "api-key-pattern"
    assert secret not in str(findings[0])


def test_scanner_ignores_untracked_private_directory_contents(tmp_path: Path) -> None:
    init_repo(tmp_path)
    private = tmp_path / "private_data"
    private.mkdir()
    (private / "raw.txt").write_text("private synthetic content", encoding="utf-8")
    assert scan_repository(tmp_path) == []


def test_scanner_allows_tracked_env_example(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".env.example").write_text("COMPANION_LOG_LEVEL=INFO\n", encoding="utf-8")
    git(tmp_path, "add", ".env.example")
    assert scan_repository(tmp_path) == []


def test_scanner_rejects_tracked_env_file(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".env").write_text("SYNTHETIC=value\n", encoding="utf-8")
    git(tmp_path, "add", ".env")
    findings = scan_repository(tmp_path)
    assert findings == [Finding(path=".env", line=None, rule="forbidden-path")]


def test_scanner_rejects_tracked_private_directory(tmp_path: Path) -> None:
    init_repo(tmp_path)
    private = tmp_path / "private_data"
    private.mkdir()
    (private / "raw.txt").write_text("synthetic content", encoding="utf-8")
    git(tmp_path, "add", "private_data/raw.txt")
    findings = scan_repository(tmp_path)
    assert findings == [Finding(path="private_data/raw.txt", line=None, rule="forbidden-path")]


def test_scanner_rejects_tracked_windows_user_directory_without_echoing_value(
    tmp_path: Path,
) -> None:
    init_repo(tmp_path)
    leaked_path = "C:" + "\\Users\\synthetic-account\\Documents\\tracked-repo"
    (tmp_path / "tracked-plan.md").write_text(leaked_path, encoding="utf-8")
    git(tmp_path, "add", "tracked-plan.md")

    findings = scan_repository(tmp_path)

    assert findings == [Finding(path="tracked-plan.md", line=1, rule="windows-user-directory")]
    assert leaked_path not in str(findings[0])


def test_scanner_accepts_clean_tracked_text(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / "README.md").write_text("synthetic clean content\n", encoding="utf-8")
    git(tmp_path, "add", "README.md")
    assert scan_repository(tmp_path) == []
