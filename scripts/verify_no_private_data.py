from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

FORBIDDEN_DIRECTORIES = (
    "private_data/",
    "runtime_data/",
    "test-results/",
    "data/raw/",
    "data/private/",
    "secrets/",
)
PRIVATE_ARTIFACT_NAMES = {
    "habit_card.txt", "habit_card.md", "library.jsonl", "deleted_ids.json",
    "active_manifest.json", "active-style.json", "shared_terms.json", "partner_profile.json",
}
PRIVATE_ARTIFACT_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx"}
CONTENT_RULES = {
    "private-key-header": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "api-key-pattern": re.compile(r"sk-[A-Za-z0-9_-]{24,}"),
    "github-token-pattern": re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"
    ),
    "unix-user-directory": re.compile(r"/(?:Us" + r"ers|ho" + r"me)/[^/\s]+/"),
    "real-private-data-marker": re.compile("REAL_" + "PRIVATE_DATA"),
    "windows-user-directory": re.compile(
        r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]+Users[\\/]+[^\\/\r\n]+[\\/]"
    ),
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int | None
    rule: str


def _tracked_files(repo_root: Path, *, include_untracked: bool = False) -> list[str]:
    arguments = ["git", "ls-files", "-z"]
    if include_untracked:
        arguments.extend(["--cached", "--others", "--exclude-standard"])
    result = subprocess.run(
        arguments,
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return sorted({item.decode("utf-8") for item in result.stdout.split(b"\0") if item})


def _forbidden_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parsed = PurePosixPath(normalized)
    name = parsed.name
    forbidden_environment_file = name == ".env" or (
        name.startswith(".env.") and name != ".env.example"
    )
    return (
        forbidden_environment_file
        or name in PRIVATE_ARTIFACT_NAMES
        or parsed.suffix.lower() in PRIVATE_ARTIFACT_SUFFIXES
        or any(
            f"/{prefix}" in f"/{normalized}"
            for prefix in FORBIDDEN_DIRECTORIES
        )
    )


def scan_repository(repo_root: Path, *, include_untracked: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    for relative_path in _tracked_files(repo_root, include_untracked=include_untracked):
        if _forbidden_path(relative_path):
            findings.append(Finding(relative_path, None, "forbidden-path"))
            continue
        if (repo_root / relative_path).is_symlink():
            findings.append(Finding(relative_path, None, "symlink"))
            continue
        try:
            text = (repo_root / relative_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule, pattern in CONTENT_RULES.items():
                if pattern.search(line):
                    findings.append(Finding(relative_path, line_number, rule))
    return findings


def main() -> int:
    findings = scan_repository(Path.cwd(), include_untracked=True)
    if findings:
        for finding in findings:
            location = finding.path if finding.line is None else f"{finding.path}:{finding.line}"
            print(f"{location}: {finding.rule}")
        return 1
    print("No private paths or secret patterns found in tracked and unignored new files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
