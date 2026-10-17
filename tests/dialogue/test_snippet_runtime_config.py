"""snippet_style runtime-config segment and product runtime assembly (synthetic data only)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from dialogue_support import FIXED_NOW, FakeProvider
from pydantic import SecretStr, ValidationError

import companion.dialogue.composition as composition
from companion.contracts import AddresseeRegister, DialogueRequest
from companion.dialogue.config import DialogueConfigError, load_dialogue_config
from companion.dialogue.context import SNIPPETS_END, SNIPPETS_START
from companion.dialogue.models import DialogueSecrets, SnippetStyleSources
from companion.privacy.paired_snippets import PairedSnippet, write_snippets
from companion.runtime.settings import Settings
from companion.style.snippet_source import FileSnippetStyleSource

PARTNER = "synthetic-partner"
CARD = "合成习惯 短句 常用语气词"


def _settings(root: Path) -> Settings:
    return Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=root / "private",
        runtime_data_dir=root / "runtime",
        test_results_dir=root / "results",
    )


def _snippet(index: int, *, owner_reply: list[str] | None = None) -> PairedSnippet:
    return PairedSnippet(
        snippet_id=f"snip-{index:016x}",
        conversation_id="synthetic-conversation",
        started_at=datetime(2026, 1, 1, 12, index, tzinfo=UTC),
        partner_turns=[f"合成她说{index}"],
        owner_reply=owner_reply or [f"合成他回{index}"],
        source_fingerprint="f" * 64,
    )


def _write_sources(private_root: Path, *, card: str = CARD) -> None:
    snippets_dir = private_root / "style" / "snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    library = [_snippet(index) for index in range(9)]
    # Four owner lines exceed the StyleSnippet limit and must be filtered out.
    too_long = ["合成超长甲", "合成超长乙", "合成超长丙", "合成超长丁"]
    library.append(_snippet(20, owner_reply=too_long))
    write_snippets(snippets_dir / "library.jsonl", library)
    (snippets_dir / "habit_card.txt").write_text(card, encoding="utf-8")


def _write_config(private_root: Path, snippet_style: object, **extra: object) -> None:
    destination = private_root / "dialogue" / "runtime-config.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "provider": {
                    "base_url": "https://jojocode.com/v1",
                    "model": "gpt-5.6-terra",
                    "timeout_seconds": 120,
                },
                "budget": {
                    "monthly_cap_cny": "100.00",
                    "worst_case_request_cny": "1.00",
                    "input_cny_per_million": "1.00",
                    "output_cny_per_million": "2.00",
                    "max_output_tokens": 256,
                },
                "snippet_style": snippet_style,
                **extra,
            }
        ),
        encoding="utf-8",
    )


def _segment(**overrides: object) -> dict[str, object]:
    return {
        "library_source": "style/snippets/library.jsonl",
        "habit_card_source": "style/snippets/habit_card.txt",
        "partner_user_id": PARTNER,
        **overrides,
    }


class _NoDigestEnvironment(dict):
    def __getitem__(self, key: str) -> str:
        assert key != "STYLE_ACTIVE_MANIFEST_DIGEST"
        return super().__getitem__(key)


def _load(settings: Settings):
    return load_dialogue_config(settings, _NoDigestEnvironment(OPENAI_API_KEY="synthetic-key"))


def test_snippet_style_config_loads_without_digest_and_resolves_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_sources(settings.private_data_dir)
    _write_config(settings.private_data_dir, _segment(), style="ignored-legacy-segment")

    config, secrets = _load(settings)

    root = settings.private_data_dir.resolve()
    assert config.style is None
    assert config.style_framework is None
    assert config.snippet_style == SnippetStyleSources(
        library_source=root / "style/snippets/library.jsonl",
        habit_card_source=root / "style/snippets/habit_card.txt",
        partner_user_id=PARTNER,
    )
    assert config.snippet_style.library_source.is_absolute()
    assert config.snippet_style.style_version == "snippet-runtime-v1"
    assert secrets.api_key.get_secret_value() == "synthetic-key"


def test_snippet_style_habit_card_is_optional(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    segment = _segment()
    del segment["habit_card_source"]
    _write_config(settings.private_data_dir, segment)

    config, _ = _load(settings)

    assert config.snippet_style is not None
    assert config.snippet_style.habit_card_source is None


@pytest.mark.parametrize(
    "overrides",
    (
        {"library_source": "../outside/library.jsonl"},
        {"habit_card_source": "../outside/habit_card.txt"},
        {"library_source": "/etc/library.jsonl"},
        {"library_source": ""},
        {"partner_user_id": " padded"},
        {"partner_user_id": ""},
        {"partner_user_id": "x" * 121},
        {"unexpected": "field"},
    ),
)
def test_invalid_snippet_style_segment_fails_closed(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    settings = _settings(tmp_path)
    _write_config(settings.private_data_dir, _segment(**overrides))

    with pytest.raises(DialogueConfigError):
        _load(settings)


def test_snippet_style_segment_must_be_object(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_config(settings.private_data_dir, "style/snippets/library.jsonl")

    with pytest.raises(DialogueConfigError):
        _load(settings)


def test_snippet_style_sources_model_rejects_whitespace_partner() -> None:
    with pytest.raises(ValidationError):
        SnippetStyleSources(library_source=Path("/x"), partner_user_id="partner ")


def _loaded_config(tmp_path: Path, *, card: str = CARD, with_card: bool = True):
    settings = _settings(tmp_path)
    _write_sources(settings.private_data_dir, card=card)
    segment = _segment()
    if not with_card:
        del segment["habit_card_source"]
    _write_config(settings.private_data_dir, segment)
    config, secrets = _load(settings)
    return settings, config, secrets


def test_build_snippet_style_runtime_matches_eval_card_arm(tmp_path: Path) -> None:
    _, config, _ = _loaded_config(tmp_path)

    adapter, card = composition.build_snippet_style_runtime(config, clock=lambda: FIXED_NOW)

    assert isinstance(adapter, FileSnippetStyleSource)
    assert card == CARD
    partner = adapter.evidence_for(PARTNER, "合成她说1", [])
    assert partner.addressee_register is AddresseeRegister.INTIMATE
    assert partner.style_version == "snippet-runtime-v1"
    assert partner.similar or partner.general
    replies = {line for item in [*partner.similar, *partner.general] for line in item.owner_reply}
    assert "合成超长甲" not in replies
    other = adapter.evidence_for("synthetic-other", "合成她说1", [])
    assert other.addressee_register is AddresseeRegister.NEUTRAL
    assert other.similar == []
    assert other.general == []


def test_build_snippet_style_runtime_without_card_returns_none(tmp_path: Path) -> None:
    _, config, _ = _loaded_config(tmp_path, with_card=False)

    _, card = composition.build_snippet_style_runtime(config, clock=lambda: FIXED_NOW)

    assert card is None


@pytest.mark.parametrize("card", ("", " \n\t "))
def test_build_snippet_style_runtime_rejects_empty_card(tmp_path: Path, card: str) -> None:
    _, config, _ = _loaded_config(tmp_path, card=card)

    with pytest.raises(ValueError):
        composition.build_snippet_style_runtime(config, clock=lambda: FIXED_NOW)


def test_build_snippet_style_runtime_requires_snippet_segment(tmp_path: Path) -> None:
    _, config, _ = _loaded_config(tmp_path)

    with pytest.raises(ValueError):
        composition.build_snippet_style_runtime(
            config.model_copy(update={"snippet_style": None}), clock=lambda: FIXED_NOW
        )


def test_snippet_runtime_wires_into_real_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, config, secrets = _loaded_config(tmp_path)
    provider = FakeProvider("合成安全回复")
    monkeypatch.setattr(composition, "OpenAICompatibleProvider", lambda **_: provider)
    adapter, card = composition.build_snippet_style_runtime(config, clock=lambda: FIXED_NOW)

    runtime = composition.build_real_orchestrator(
        settings,
        config,
        DialogueSecrets(api_key=SecretStr(secrets.api_key.get_secret_value())),
        trusted_user_id=PARTNER,
        snippet_style=adapter,
        snippet_habit_card=card,
    )
    runtime._clock = lambda: FIXED_NOW
    run = runtime.reply(
        DialogueRequest(user_id=PARTNER, text="合成她说2", recent_context=[]),
        trusted_user_id=PARTNER,
    )

    assert provider.call_count == 1
    system = provider.system_text
    assert SNIPPETS_START in system
    assert SNIPPETS_END in system
    assert f"他的说话习惯：\n{CARD}" in system  # noqa: RUF001
    assert "- 她：合成她说" in system  # noqa: RUF001
    assert run.trace.style_version == "snippet-runtime-v1"
    assert run.trace.style_evidence_count
