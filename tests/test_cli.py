import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from companion.cli import app
from companion.runtime.settings import SettingsError


def write_config(root: Path, log_level: str = "INFO") -> None:
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "default.toml").write_text(
        '[companion]\nenvironment="test"\n'
        f'log_level="{log_level}"\n'
        'private_data_dir="private"\nruntime_data_dir="runtime"\n'
        'test_results_dir="results"\n',
        encoding="utf-8",
    )


def test_healthcheck_command_returns_machine_readable_json(tmp_path: Path) -> None:
    write_config(tmp_path)
    result = CliRunner().invoke(app, ["healthcheck", "--project-root", str(tmp_path)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "healthy"
    assert payload["environment"] == "test"


def test_healthcheck_configuration_error_is_safe(tmp_path: Path) -> None:
    secret = "synthetic-secret-value"
    write_config(tmp_path, log_level=secret)
    result = CliRunner().invoke(app, ["healthcheck", "--project-root", str(tmp_path)])
    assert result.exit_code == 2
    assert "Configuration error" in result.stderr
    assert secret not in result.stderr
    assert secret not in result.stdout


@pytest.mark.parametrize(
    "case,expected",
    [
        ("missing", "configuration file not found"),
        ("syntax", "invalid TOML syntax"),
        ("value", "invalid configuration fields or paths"),
        ("environment", "unrecognized COMPANION_ environment variable"),
    ],
)
def test_configuration_errors_explain_category_without_private_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, expected: str
) -> None:
    secret = "synthetic-private-canary"
    if case != "missing":
        write_config(tmp_path, log_level=secret if case == "value" else "INFO")
    if case == "syntax":
        (tmp_path / "config" / "default.toml").write_text(
            f'[companion]\nlog_level="{secret}', encoding="utf-8"
        )
    if case == "environment":
        monkeypatch.setenv("COMPANION_SYNTHETIC_UNKNOWN", secret)
    result = CliRunner().invoke(app, ["healthcheck", "--project-root", str(tmp_path)])
    assert result.exit_code == 2
    assert expected in result.stderr
    assert "README" in result.stderr
    assert str(tmp_path) not in result.stderr
    assert secret not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("file_access", "configuration file could not be accessed"),
        ("synthetic-private-canary", "invalid configuration"),
    ],
)
def test_configuration_error_never_prints_exception_or_unknown_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str, expected: str
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise SettingsError("synthetic-private-canary", reason=reason)

    monkeypatch.setattr("companion.cli.load_settings", fail)
    result = CliRunner().invoke(app, ["healthcheck", "--project-root", str(tmp_path)])
    assert result.exit_code == 2
    assert expected in result.stderr
    assert "synthetic-private-canary" not in result.stderr
    assert result.stdout == ""
