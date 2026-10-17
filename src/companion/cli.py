from pathlib import Path
from typing import Annotated

import typer

from companion.runtime.health import check_health
from companion.runtime.logging import configure_logging
from companion.runtime.settings import SettingsError, load_settings

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Run local desktop AI companion utilities."""


@app.command()
def healthcheck(
    project_root: Annotated[Path | None, typer.Option("--project-root")] = None,
) -> None:
    """Validate local configuration and report runtime health."""
    try:
        settings = load_settings(project_root or Path.cwd())
        configure_logging(settings.log_level)
        typer.echo(check_health(settings).model_dump_json())
    except SettingsError as error:
        # Only fixed messages may reach stderr; exception text can contain private data.
        explanations = {
            "missing_file": "configuration file not found; check --project-root",
            "invalid_toml": "invalid TOML syntax",
            "file_access": "configuration file could not be accessed",
            "invalid_fields": "invalid configuration fields or paths",
            "unknown_environment": "unrecognized COMPANION_ environment variable",
        }
        explanation = explanations.get(error.reason, "invalid configuration")
        typer.echo(f"Configuration error: {explanation}. See README configuration help.", err=True)
        raise typer.Exit(code=2) from None


if __name__ == "__main__":
    app()
