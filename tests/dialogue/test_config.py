import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.dialogue.config import (
    DialogueConfigError,
    load_dialogue_config,
    validate_framework_sources,
)
from companion.dialogue.models import ProductionStyleSources, ProviderConfig
from companion.runtime.settings import Settings


@pytest.mark.parametrize(
    "base_url",
    (
        "http://api.deepseek.com/v1",
        "http://example.invalid/v1",
        "http://localhost:8080/v1",
    ),
)
def test_remote_http_base_url_rejected(base_url: str) -> None:
    with pytest.raises(ValidationError, match="远端 provider 必须使用 https") as exc:
        ProviderConfig(base_url=base_url, model="synthetic-model", timeout_seconds=120)
    assert base_url not in str(exc.value)


def test_base_url_userinfo_rejected_without_echo() -> None:
    base_url = "https://user:synthetic-password@example.invalid/v1"
    with pytest.raises(ValidationError, match="provider base_url 不能包含用户信息") as exc:
        ProviderConfig(base_url=base_url, model="synthetic-model", timeout_seconds=120)
    assert "synthetic-password" not in str(exc.value)
    assert base_url not in str(exc.value)


@pytest.mark.parametrize("base_url", ("http://127.0.0.1:8080/v1", "http://[::1]:8080/v1"))
def test_loopback_http_base_url_allowed(base_url: str) -> None:
    config = ProviderConfig(base_url=base_url, model="synthetic-model", timeout_seconds=120)
    assert str(config.base_url) == base_url


@pytest.mark.parametrize("base_url", ("https://api.deepseek.com/v1",))
def test_https_base_url_allowed(base_url: str) -> None:
    config = ProviderConfig(base_url=base_url, model="synthetic-model", timeout_seconds=120)
    assert str(config.base_url) == base_url


def make_settings(root: Path) -> Settings:
    return Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=root / "private",
        runtime_data_dir=root / "runtime",
        test_results_dir=root / "results",
    )


def write_private_dialogue_config(
    private_root: Path,
    *,
    api_key_env: str | None = None,
) -> None:
    destination = private_root / "dialogue" / "runtime-config.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(
            {
                "provider": {
                    "base_url": "https://jojocode.com/v1",
                    "model": "gpt-5.6-terra",
                    "timeout_seconds": 120,
                    **({"api_key_env": api_key_env} if api_key_env is not None else {}),
                },
                "budget": {
                    "monthly_cap_cny": "100.00",
                    "worst_case_request_cny": "1.00",
                    "input_cny_per_million": "1.00",
                    "output_cny_per_million": "2.00",
                    "max_output_tokens": 256,
                },
                "style": {
                    "candidate_source": "style_candidates/current/safe-candidates.json",
                    "expected_count": 7,
                    "active_style_source": "style_runtime/active.json",
                    "active_manifest_source": "style_runtime/manifest.json",
                    "calibration_source": (
                        "style_candidates/calibration/current/calibration-candidates.json"
                    ),
                    "expected_calibration_fingerprint": "b" * 64,
                    "true_name_terms": ["synthetic-owner-name"],
                    "approved_nicknames": ["synthetic-approved-nickname"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_dialogue_config_reads_secret_and_trusted_digest_separately(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)

    config, secrets = load_dialogue_config(
        settings,
        {
            "OPENAI_API_KEY": "synthetic-api-key",
            "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
        },
    )

    assert str(config.provider.base_url).rstrip("/") == "https://jojocode.com/v1"
    assert config.provider.model == "gpt-5.6-terra"
    assert config.provider.timeout_seconds == 120
    assert config.style.expected_active_manifest_digest == "a" * 64
    assert secrets.api_key.get_secret_value() == "synthetic-api-key"
    assert "synthetic-api-key" not in repr(secrets)
    assert config.style_framework is None


def _framework_paths(root: Path) -> ProductionStyleSources:
    return ProductionStyleSources.model_validate(
        {
            field: root / "framework" / f"{field}.json"
            for field in ProductionStyleSources.model_fields
        }
    )


@pytest.mark.parametrize("style_payload", (None, "invalid", {"unexpected": "ignored"}))
def test_optional_style_config_ignores_style_and_framework_without_reading_digest(
    tmp_path: Path, style_payload: object
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir, api_key_env="DEEPSEEK_API_KEY")
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("style")
    if style_payload is not None:
        payload["style"] = style_payload
    payload["style_framework"] = "invalid"
    payload["style_framework_owner_id"] = "synthetic-owner"
    path.write_text(json.dumps(payload), encoding="utf-8")

    class ApiKeyOnlyEnvironment(dict):
        def __getitem__(self, key):
            assert key != "STYLE_ACTIVE_MANIFEST_DIGEST"
            return super().__getitem__(key)

    config, secrets = load_dialogue_config(
        settings,
        ApiKeyOnlyEnvironment(DEEPSEEK_API_KEY="synthetic-key"),
        require_style_manifest=False,
    )
    assert config.style is None
    assert config.style_framework is None
    assert config.style_framework_owner_id == "synthetic-owner"
    assert config.provider.api_key_env == "DEEPSEEK_API_KEY"
    assert config.budget.max_output_tokens == 256
    assert secrets.api_key.get_secret_value() == "synthetic-key"


@pytest.mark.parametrize("environment", ({}, {"OPENAI_API_KEY": " "}))
def test_optional_style_config_still_requires_api_key(tmp_path: Path, environment: dict) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)
    with pytest.raises(DialogueConfigError):
        load_dialogue_config(settings, environment, require_style_manifest=False)


@pytest.mark.parametrize(
    ("section", "invalid"),
    (("provider", {}), ("budget", {}), ("style_framework_owner_id", " owner")),
)
def test_optional_style_config_still_validates_other_sections(
    tmp_path: Path, section: str, invalid: object
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[section] = invalid
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DialogueConfigError):
        load_dialogue_config(
            settings, {"OPENAI_API_KEY": "synthetic-key"}, require_style_manifest=False
        )


def test_required_style_config_still_requires_digest(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)
    with pytest.raises(DialogueConfigError):
        load_dialogue_config(
            settings, {"OPENAI_API_KEY": "synthetic-key"}, require_style_manifest=True
        )


def test_production_framework_fixed_paths_reject_escape_symlink_and_overlap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    sources = _framework_paths(root)
    validate_framework_sources(sources, root)
    with pytest.raises(ValueError):
        validate_framework_sources(
            sources.model_copy(update={"consent_source": tmp_path / "outside.json"}), root
        )
    with pytest.raises(ValueError):
        validate_framework_sources(
            sources.model_copy(update={"profile_source": sources.consent_source}), root
        )
    (root / "framework").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_framework_sources(sources, root)


def test_load_dialogue_config_accepts_explicit_framework_paths_only_when_valid(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["style_framework"] = {
        field: f"framework/{field}.json" for field in ProductionStyleSources.model_fields
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    environment = {
        "OPENAI_API_KEY": "synthetic-key",
        "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
    }
    config, _ = load_dialogue_config(settings, environment)
    assert config.style_framework is not None
    assert config.style_framework.consent_source == (
        settings.private_data_dir / "framework" / "consent_source.json"
    )

    payload["style_framework"]["profile_source"] = "framework/consent_source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DialogueConfigError):
        load_dialogue_config(settings, environment)


@pytest.mark.parametrize("owner", ("", " owner", "owner ", "x" * 121, 4, True))
def test_framework_owner_config_requires_exact_nonempty_string(
    tmp_path: Path, owner: object
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["style_framework_owner_id"] = owner
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DialogueConfigError):
        load_dialogue_config(
            settings,
            {"OPENAI_API_KEY": "synthetic-key", "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64},
        )


def test_framework_sources_reject_symlink_in_private_root_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    private = real / "private"
    private.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    root = alias / "private"
    sources = _framework_paths(root.resolve())

    with pytest.raises(ValueError):
        validate_framework_sources(sources, root)


def test_load_dialogue_config_rejects_symlink_in_private_root_ancestor(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    private = real / "private"
    private.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    settings = make_settings(alias)
    write_private_dialogue_config(settings.private_data_dir)
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["style_framework"] = {
        field: f"framework/{field}.json" for field in ProductionStyleSources.model_fields
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DialogueConfigError):
        load_dialogue_config(
            settings,
            {
                "OPENAI_API_KEY": "synthetic-key",
                "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
            },
        )


def test_load_dialogue_config_selects_the_configured_deepseek_secret(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(
        settings.private_data_dir,
        api_key_env="DEEPSEEK_API_KEY",
    )

    config, secrets = load_dialogue_config(
        settings,
        {
            "OPENAI_API_KEY": "synthetic-jojo-key",
            "DEEPSEEK_API_KEY": "synthetic-deepseek-key",
            "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
        },
    )

    assert config.provider.api_key_env == "DEEPSEEK_API_KEY"
    assert secrets.api_key.get_secret_value() == "synthetic-deepseek-key"
    assert "synthetic-deepseek-key" not in repr(secrets)


@pytest.mark.parametrize(
    ("api_key_env", "environ"),
    (
        (
            "UNTRUSTED_API_KEY",
            {
                "UNTRUSTED_API_KEY": "synthetic-untrusted-key",
                "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
            },
        ),
        (
            "DEEPSEEK_API_KEY",
            {
                "OPENAI_API_KEY": "synthetic-jojo-key",
                "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
            },
        ),
        (
            "DEEPSEEK_API_KEY",
            {
                "OPENAI_API_KEY": "synthetic-jojo-key",
                "DEEPSEEK_API_KEY": " ",
                "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
            },
        ),
    ),
)
def test_load_dialogue_config_rejects_an_invalid_or_missing_selected_secret(
    tmp_path: Path,
    api_key_env: str,
    environ: dict[str, str],
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(
        settings.private_data_dir,
        api_key_env=api_key_env,
    )

    with pytest.raises(
        DialogueConfigError,
        match="dialogue configuration unavailable",
    ):
        load_dialogue_config(settings, environ)


@pytest.mark.parametrize(
    "environ",
    (
        {},
        {"OPENAI_API_KEY": "synthetic-api-key"},
        {
            "OPENAI_API_KEY": "synthetic-api-key",
            "STYLE_ACTIVE_MANIFEST_DIGEST": "A" * 64,
        },
    ),
)
def test_load_dialogue_config_fails_closed_without_valid_independent_anchors(
    tmp_path: Path,
    environ: dict[str, str],
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)

    with pytest.raises(DialogueConfigError, match="dialogue configuration unavailable") as error:
        load_dialogue_config(settings, environ)

    assert str(tmp_path) not in str(error.value)


def test_load_dialogue_config_rejects_private_path_escape(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)
    config_path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["style"]["candidate_source"] = "../outside.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DialogueConfigError, match="dialogue configuration unavailable"):
        load_dialogue_config(
            settings,
            {
                "OPENAI_API_KEY": "synthetic-api-key",
                "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
            },
        )


def test_load_dialogue_config_rejects_a_manifest_digest_in_private_json(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)
    config_path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["style"]["expected_active_manifest_digest"] = "b" * 64
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DialogueConfigError, match="dialogue configuration unavailable"):
        load_dialogue_config(
            settings,
            {
                "OPENAI_API_KEY": "synthetic-api-key",
                "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
            },
        )


def test_load_dialogue_config_clears_private_config_after_secret_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    write_private_dialogue_config(settings.private_data_dir)
    config_path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["style"]["candidate_source"] = "private-path-sentinel/candidates.json"
    payload["style"]["true_name_terms"] = ["true-name-sentinel"]
    payload["style"]["approved_nicknames"] = ["nickname-sentinel"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    payload = None

    def fail_secret_construction(*, api_key: str) -> None:
        raise RuntimeError("synthetic secret construction failure")

    monkeypatch.setattr("companion.dialogue.config.DialogueSecrets", fail_secret_construction)
    with pytest.raises(DialogueConfigError) as error:
        load_dialogue_config(
            settings,
            {
                "OPENAI_API_KEY": "synthetic-api-key",
                "STYLE_ACTIVE_MANIFEST_DIGEST": "a" * 64,
            },
        )

    traceback = error.value.__traceback__
    while traceback is not None:
        for local_value in traceback.tb_frame.f_locals.values():
            local_text = repr(local_value)
            assert "private-path-sentinel" not in local_text
            assert "true-name-sentinel" not in local_text
            assert "nickname-sentinel" not in local_text
        traceback = traceback.tb_next
