from pathlib import Path

import pytest

from companion.runtime.settings import SettingsError, load_settings


def write_default_config(root: Path, log_level: str = "INFO") -> None:
    config = root / "config"
    config.mkdir()
    (config / "default.toml").write_text(
        "[companion]\n"
        f'log_level = "{log_level}"\n'
        'environment = "development"\n'
        'private_data_dir = "private_data"\n'
        'runtime_data_dir = "runtime_data"\n'
        'test_results_dir = "test-results"\n',
        encoding="utf-8",
    )


def test_load_settings_resolves_paths_and_environment_override(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    settings = load_settings(
        tmp_path,
        environ={"COMPANION_LOG_LEVEL": "WARNING"},
    )
    assert settings.environment == "development"
    assert settings.log_level == "WARNING"
    assert settings.private_data_dir == (tmp_path / "private_data").resolve()
    assert settings.runtime_data_dir == (tmp_path / "runtime_data").resolve()
    assert settings.test_results_dir == (tmp_path / "test-results").resolve()


def test_explicit_config_overrides_only_supplied_defaults(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    override = tmp_path / "local.toml"
    override.write_text('[companion]\nlog_level = "DEBUG"\n', encoding="utf-8")
    settings = load_settings(tmp_path, config_path=override, environ={})
    assert settings.log_level == "DEBUG"
    assert settings.environment == "development"
    assert settings.private_data_dir == (tmp_path / "private_data").resolve()


def test_absolute_paths_remain_absolute(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    absolute = (tmp_path.parent / "separate-private").resolve()
    settings = load_settings(
        tmp_path,
        environ={"COMPANION_PRIVATE_DATA_DIR": str(absolute)},
    )
    assert settings.private_data_dir == absolute


@pytest.mark.parametrize("source", ("default", "override", "environment"))
@pytest.mark.parametrize("absolute", (False, True))
@pytest.mark.parametrize("symlink_position", ("ancestor", "leaf"))
def test_private_data_path_rejects_original_symlink_before_resolving(
    tmp_path: Path,
    source: str,
    absolute: bool,
    symlink_position: str,
) -> None:
    write_default_config(tmp_path)
    real = tmp_path / "real"
    private = real / "private"
    private.mkdir(parents=True)
    if symlink_position == "ancestor":
        alias = tmp_path / "alias"
        alias.symlink_to(real, target_is_directory=True)
        raw_path = alias / "private"
    else:
        alias = tmp_path / "private-alias"
        alias.symlink_to(private, target_is_directory=True)
        raw_path = alias

    configured_path = str(raw_path if absolute else raw_path.relative_to(tmp_path))
    config_path = None
    environment: dict[str, str] = {}
    if source == "default":
        default = tmp_path / "config" / "default.toml"
        default.write_text(
            default.read_text(encoding="utf-8").replace(
                'private_data_dir = "private_data"',
                f'private_data_dir = "{configured_path}"',
            ),
            encoding="utf-8",
        )
    elif source == "override":
        config_path = tmp_path / "override.toml"
        config_path.write_text(
            f'[companion]\nprivate_data_dir = "{configured_path}"\n',
            encoding="utf-8",
        )
    else:
        environment["COMPANION_PRIVATE_DATA_DIR"] = configured_path

    with pytest.raises(SettingsError) as error:
        load_settings(tmp_path, config_path=config_path, environ=environment)
    assert configured_path not in str(error.value)
    assert str(raw_path) not in str(error.value)


def test_private_data_path_accepts_regular_absolute_and_relative_paths(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    private = tmp_path / "regular" / "private"
    private.mkdir(parents=True)
    relative = load_settings(
        tmp_path,
        environ={"COMPANION_PRIVATE_DATA_DIR": "regular/private"},
    )
    absolute = load_settings(
        tmp_path,
        environ={"COMPANION_PRIVATE_DATA_DIR": str(private)},
    )
    assert relative.private_data_dir == private.resolve()
    assert absolute.private_data_dir == private.resolve()


def test_other_data_paths_keep_existing_symlink_resolution(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    settings = load_settings(
        tmp_path,
        environ={
            "COMPANION_RUNTIME_DATA_DIR": "alias/runtime",
            "COMPANION_TEST_RESULTS_DIR": "alias/results",
        },
    )

    assert settings.runtime_data_dir == real / "runtime"
    assert settings.test_results_dir == real / "results"


def test_unknown_companion_environment_variable_is_rejected(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    with pytest.raises(SettingsError, match="unknown COMPANION_ setting"):
        load_settings(tmp_path, environ={"COMPANION_SECRET_VALUE": "do-not-print"})


def test_settings_error_does_not_echo_secret_value(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    secret = "synthetic-secret-value"
    with pytest.raises(SettingsError) as error:
        load_settings(tmp_path, environ={"COMPANION_SECRET_VALUE": secret})
    assert secret not in str(error.value)


def test_missing_default_configuration_has_safe_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="invalid configuration") as error:
        load_settings(tmp_path, environ={})
    assert str(tmp_path) not in str(error.value)


def test_invalid_log_level_is_rejected(tmp_path: Path) -> None:
    write_default_config(tmp_path, log_level="VERBOSE")
    with pytest.raises(SettingsError, match="invalid configuration"):
        load_settings(tmp_path, environ={})


def test_unknown_toml_field_is_rejected(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    override = tmp_path / "local.toml"
    override.write_text('[companion]\nunknown = "value"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match="invalid configuration"):
        load_settings(tmp_path, config_path=override, environ={})
