# ruff: noqa: RUF001
"""Owner-approved shared-terms dictionary injected on demand (synthetic data only)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from dialogue_support import FIXED_NOW, FakeProvider, synthetic_dialogue_config
from pydantic import SecretStr
from test_snippet_runtime_config import PARTNER, _load, _segment, _settings, _write_config
from test_snippet_runtime_config import _write_sources as _write_snippet_sources
from test_snippet_style import FakeSnippetStyle

import companion.dialogue.composition as composition
from companion.contracts import DialogueRequest
from companion.dialogue.models import DialogueSecrets
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.shared_terms import SharedTerm, load_shared_terms
from companion.dialogue.usage import MonthlyUsageLedger

HEADING = "你们之间的专属说法（她这句话里出现了，按这里的意思理解）："


def _entry(
    term: str,
    meaning: str,
    *,
    occasion: str = "合成场合",
    permission: str = "allowed",
    speaker: str = "both",
) -> dict[str, str]:
    return {
        "term": term,
        "meaning": meaning,
        "occasion": occasion,
        "speaker": speaker,
        "addressee": "合成对象",
        "permission": permission,
    }


def _terms(*entries: dict[str, str]) -> tuple[SharedTerm, ...]:
    return tuple(SharedTerm.model_validate(entry) for entry in entries)


def _orchestrator(
    root: Path, provider: FakeProvider, terms: Sequence[SharedTerm] | None
) -> DialogueOrchestrator:
    config = synthetic_dialogue_config(root)
    return DialogueOrchestrator(
        config=config,
        snippet_style=FakeSnippetStyle(),
        snippet_shared_terms=terms,
        provider=provider,
        ledger=MonthlyUsageLedger(root / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    )


def _system(
    root: Path,
    terms: Sequence[SharedTerm] | None,
    *,
    text: str = "合成当前消息",
    recent: Sequence[str] = (),
) -> str:
    provider = FakeProvider("合成安全回复")
    run = _orchestrator(root, provider, terms).reply(
        DialogueRequest(user_id="synthetic-user", text=text, recent_context=list(recent)),
        trusted_user_id="synthetic-user",
    )
    assert run.response.blocked_reason is None
    assert provider.call_count == 1
    return provider.system_text


def test_allowed_and_understand_only_wording_and_dictionary_order(tmp_path: Path) -> None:
    terms = _terms(
        _entry("合成乙词", "合成乙义", permission="understand_only"),
        _entry("合成甲词", "合成甲义", occasion=""),
        _entry("合成未出现", "合成丙义"),
    )

    system = _system(tmp_path, terms, text="合成甲词和合成乙词都说了")

    assert (
        f"{HEADING}\n"
        "- 合成乙词：合成乙义；场合：合成场合；只用来理解她的话，你不要这样说\n"
        "- 合成甲词：合成甲义；合适时你也可以这样说，不必硬用\n"
        "相似：" in system
    )
    assert "合成未出现" not in system


def test_block_sits_after_habit_card_and_is_escaped(tmp_path: Path) -> None:
    config = synthetic_dialogue_config(tmp_path)
    provider = FakeProvider("合成安全回复")
    DialogueOrchestrator(
        config=config,
        snippet_style=FakeSnippetStyle(),
        snippet_habit_card="合成习惯",
        snippet_shared_terms=_terms(_entry("合成[词]", "合成<义>")),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    ).reply(
        DialogueRequest(user_id="synthetic-user", text="合成[词]", recent_context=[]),
        trusted_user_id="synthetic-user",
    )

    assert (
        f"他的说话习惯：\n合成习惯\n{HEADING}\n"
        "- 合成\\[词\\]：合成\\<义\\>；场合：合成场合；合适时你也可以这样说，不必硬用\n相似："
    ) in provider.system_text


def test_partner_context_hit_injects_but_own_line_and_older_lines_do_not(tmp_path: Path) -> None:
    terms = _terms(_entry("合成前文词", "合成前义"), _entry("合成自述词", "合成自义"),
                   _entry("合成旧词", "合成旧义"))

    system = _system(
        tmp_path,
        terms,
        recent=("对方：合成旧词", "对方：合成中间", "你：合成自述词", "对方：合成前文词"),
    )

    assert "- 合成前文词：合成前义" in system
    assert "合成自义" not in system
    assert "合成旧义" not in system


def test_disabled_entries_never_inject_and_matches_cap_at_eight(tmp_path: Path) -> None:
    entries = [_entry("合成停用", "合成停义", permission="disabled")]
    entries += [_entry(f"合成词{index}", f"合成义{index}") for index in range(10)]

    system = _system(
        tmp_path, _terms(*entries), text="合成停用 " + " ".join(f"合成词{i}" for i in range(10))
    )

    assert "合成停义" not in system
    assert [f"合成义{i}" in system for i in range(10)] == [True] * 8 + [False] * 2


@pytest.mark.parametrize("terms", (None, ()))
def test_no_dictionary_or_no_hit_keeps_prompt_byte_identical(
    tmp_path: Path, terms: tuple[SharedTerm, ...] | None
) -> None:
    baseline = _system(tmp_path / "baseline", terms, recent=("对方：合成上文",))
    unmatched = _system(
        tmp_path / "unmatched",
        _terms(
            _entry("合成没出现", "合成义"),
            _entry("合成上文", "合成停义", permission="disabled"),
        ),
        recent=("对方：合成上文",),
    )

    assert unmatched == baseline
    assert HEADING not in baseline


def _write_dictionary(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "payload",
    (
        "{not json",
        [],
        {"version": 2, "entries": []},
        {"version": 1},
        {"version": 1, "entries": [{**_entry("合成", "合成义"), "permission": "maybe"}]},
        {"version": 1, "entries": [{**_entry("合成", "合成义"), "speaker": "other"}]},
        {"version": 1, "entries": [_entry("", "合成义")]},
        {"version": 1, "entries": [{**_entry("合成", "合成义"), "extra": "x"}]},
    ),
)
def test_malformed_dictionary_is_configuration_error(tmp_path: Path, payload: object) -> None:
    path = _write_dictionary(tmp_path / "terms.json", payload)

    with pytest.raises(ValueError, match="shared terms configuration unavailable"):
        load_shared_terms(path)


def test_missing_dictionary_is_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shared terms configuration unavailable"):
        load_shared_terms(tmp_path / "absent.json")


def test_shared_terms_require_snippet_style(tmp_path: Path) -> None:
    from dialogue_support import FakeStyleProvider

    config = synthetic_dialogue_config(tmp_path)
    with pytest.raises(ValueError, match="shared terms require snippet style"):
        DialogueOrchestrator(
            config=config,
            style_adapter=FakeStyleProvider(),
            snippet_shared_terms=(),
            provider=FakeProvider(),
            ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
            clock=lambda: FIXED_NOW,
        )


# --- runtime config and real construction path -------------------------------------------


def test_config_resolves_optional_shared_terms_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_config(settings.private_data_dir, _segment(shared_terms_source="style/terms.json"))

    config, _ = _load(settings)

    root = settings.private_data_dir.resolve()
    assert config.snippet_style.shared_terms_source == root / "style/terms.json"
    assert composition.load_snippet_shared_terms(
        config.model_copy(
            update={
                "snippet_style": config.snippet_style.model_copy(
                    update={"shared_terms_source": None}
                )
            }
        )
    ) is None


def test_load_snippet_shared_terms_reports_missing_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_config(settings.private_data_dir, _segment(shared_terms_source="style/terms.json"))
    config, _ = _load(settings)

    with pytest.raises(ValueError, match="shared terms configuration unavailable"):
        composition.load_snippet_shared_terms(config)


def test_real_orchestrator_construction_injects_matched_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _write_snippet_sources(settings.private_data_dir)
    _write_dictionary(
        settings.private_data_dir / "style/terms.json",
        {"version": 1, "entries": [
            _entry("合成暗号", "合成暗号的意思", permission="understand_only"),
            _entry("合成没提", "合成没提的意思"),
        ]},
    )
    _write_config(settings.private_data_dir, _segment(shared_terms_source="style/terms.json"))
    config, secrets = _load(settings)
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
        snippet_shared_terms=composition.load_snippet_shared_terms(config),
    )
    runtime._clock = lambda: FIXED_NOW
    runtime.reply(
        DialogueRequest(user_id=PARTNER, text="合成她说2 合成暗号", recent_context=[]),
        trusted_user_id=PARTNER,
    )

    assert provider.call_count == 1
    assert (
        f"{HEADING}\n- 合成暗号：合成暗号的意思；场合：合成场合；只用来理解她的话，你不要这样说\n"
    ) in provider.system_text
    assert "合成没提的意思" not in provider.system_text


@pytest.mark.parametrize(
    "recent, expected",
    [
        ((), ["当前"]),
        (("对方：早", "你：自己", "对方：近", "原样", "你：最后"),
         ["当前", "近", "原样"]),
        (("你：自己", "对方：唯一"), ["当前", "唯一"]),
    ],
)
def test_shared_term_query_texts_rules(recent, expected) -> None:
    from companion.dialogue.shared_terms import shared_term_query_texts

    assert shared_term_query_texts("当前", recent) == expected
