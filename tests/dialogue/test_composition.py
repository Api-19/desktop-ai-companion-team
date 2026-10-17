"""Production composition exercised with public synthetic resources only."""

from __future__ import annotations

from pathlib import Path

import pytest
from dialogue_support import FIXED_NOW, FakeProvider, sample_request, synthetic_dialogue_config
from pydantic import SecretStr
from strict_framework_env import OWNER, build_strict_env

import companion.dialogue as dialogue
import companion.dialogue.ab as ab
import companion.dialogue.composition as composition
from companion.contracts import DialogueRequest
from companion.dialogue.models import DialogueConfig, DialogueSecrets
from companion.memory import MemoryService
from companion.runtime.settings import Settings


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object):
    provider = FakeProvider()
    monkeypatch.setattr(composition, "OpenAICompatibleProvider", lambda **_: provider)
    settings = Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=tmp_path,
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )
    orchestrator = composition.build_real_orchestrator(
        settings,
        synthetic_dialogue_config(tmp_path),
        DialogueSecrets(api_key=SecretStr("synthetic-key")),
        trusted_user_id="synthetic-user",
        **kwargs,
    )
    orchestrator._clock = lambda: FIXED_NOW
    return orchestrator, provider


def test_builder_is_one_implementation_exported_at_all_compatible_paths() -> None:
    assert composition.build_real_orchestrator is ab.build_real_orchestrator
    assert dialogue.build_real_orchestrator is composition.build_real_orchestrator
    assert composition.build_real_orchestrator.__module__ == "companion.dialogue.composition"


def test_dialogue_config_defaults_to_no_style(tmp_path: Path) -> None:
    original = synthetic_dialogue_config(tmp_path)
    config = DialogueConfig(provider=original.provider, budget=original.budget)
    assert config.style is None
    assert config.style_framework is None


@pytest.mark.parametrize("personalization", (False, True))
def test_composition_rejects_no_style_without_snippets(
    tmp_path: Path, personalization: bool
) -> None:
    original = synthetic_dialogue_config(tmp_path)
    config = original.model_copy(update={"style": None})
    settings = Settings(
        environment="test", log_level="INFO", private_data_dir=tmp_path,
        runtime_data_dir=tmp_path / "runtime", test_results_dir=tmp_path / "results",
    )
    with pytest.raises(ValueError):
        composition.build_real_orchestrator(
            settings, config, DialogueSecrets(api_key=SecretStr("synthetic-key")),
            trusted_user_id="synthetic-user",
            enable_production_personalization=personalization,
        )


@pytest.mark.parametrize("missing", ("style", "style_framework"))
def test_strict_composition_rejects_missing_style_sources(
    tmp_path: Path, missing: str
) -> None:
    env = build_strict_env(tmp_path)
    config = env.config.model_copy(update={missing: None})
    with pytest.raises(ValueError, match="production personalization configuration unavailable"):
        composition.build_real_orchestrator(
            env.settings, config, env.secrets, trusted_user_id=OWNER,
            enable_production_personalization=True,
        )


def test_default_composition_does_not_load_admin_or_create_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import companion.admin.app as admin

    monkeypatch.setattr(admin, "load_admin_config", lambda _: pytest.fail("Admin read"))
    runtime, provider = _build(tmp_path, monkeypatch)
    assert runtime._memory is None
    assert runtime._style_personalization is None
    assert runtime._style_verifier is None
    assert provider.call_count == 0
    assert not (tmp_path / "memory").exists()


def test_real_composition_forwards_explicit_revision_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class FakeRevisionSource:
        def __init__(self, name: str, revision: int | str) -> None:
            self.name = name
            self.revision = revision

        def content_revision(self, user_id: str) -> int | str:
            calls.append((self.name, user_id))
            return self.revision

    runtime, _ = _build(
        tmp_path,
        monkeypatch,
        privacy_revision_sources=[
            FakeRevisionSource("memory", 12),
            FakeRevisionSource("snippets", "synthetic-digest"),
        ],
    )
    assert calls == []
    assert runtime.privacy_revision("synthetic-user") == (12, "synthetic-digest")
    assert calls == [("memory", "synthetic-user"), ("snippets", "synthetic-user")]


def test_default_composition_rejects_memory_command_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, provider = _build(tmp_path, monkeypatch)
    request = sample_request().model_copy(update={"text": "记住我喜欢合成蓝色"})

    run = runtime.reply(request, trusted_user_id=request.user_id)

    assert run.response.blocked_reason == "memory_disabled"
    assert run.response.billable is False
    assert provider.call_count == 0
    assert not (tmp_path / "memory").exists()


def test_explicit_memory_gateway_retains_unauthorized_and_identity_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = MemoryService(tmp_path / "synthetic-memory.sqlite3", clock=lambda: FIXED_NOW)
    runtime, provider = _build(tmp_path, monkeypatch, memory_gateway=memory)
    assert runtime._memory is memory
    request = DialogueRequest(
        user_id="synthetic-user", text="记住我喜欢合成苹果", recent_context=[]
    )
    rejected = runtime.reply(request, trusted_user_id="synthetic-user")
    assert rejected.response.blocked_reason == "memory_write_failed"
    assert memory.list_memories("synthetic-user") == ()
    assert memory.get_consent("synthetic-user") is None
    cross_user = runtime.reply(sample_request(), trusted_user_id="synthetic-other")
    assert cross_user.response.blocked_reason == "untrusted_user"
    assert provider.call_count == 0


def test_strict_composition_keeps_shared_source_gateway_and_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_strict_env(tmp_path)
    provider = FakeProvider()
    monkeypatch.setattr(composition, "OpenAICompatibleProvider", lambda **_: provider)
    runtime = composition.build_real_orchestrator(
        env.settings,
        env.config,
        env.secrets,
        trusted_user_id=OWNER,
        enable_production_personalization=True,
    )
    personalization = runtime._style_personalization
    assert personalization._profile._strict_source_binding is True
    assert personalization._evidence._strict_source_binding is True
    assert personalization._profile._source_gateway is personalization._evidence._source_gateway
    assert runtime._style_verifier is not None
    assert runtime._memory is None
    assert provider.call_count == 0
