import json
from pathlib import Path

from typer.testing import CliRunner

from companion.cli import app


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
