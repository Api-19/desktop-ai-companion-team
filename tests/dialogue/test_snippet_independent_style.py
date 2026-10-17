# ruff: noqa: RUF001
"""Synthetic regressions for style material that does not require chat snippets."""

from hashlib import sha256
from pathlib import Path

import pytest
from dialogue_support import FIXED_NOW, FakeProvider, synthetic_dialogue_config
from test_shared_terms import _entry, _terms
from test_snippet_style import FakeSnippetStyle, evidence, prepared, snippet, system_of

import companion.dialogue.context as context
from companion.contracts import AddresseeRegister, DialogueRequest, MemoryEntry
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger

CARD = "合成习惯\n合成第二行"


@pytest.mark.parametrize("card", (None, "", " \n\t ", CARD))
@pytest.mark.parametrize("with_terms", (False, True))
def test_intimate_without_snippets_keeps_independent_style(
    card: str | None, with_terms: bool
) -> None:
    terms = _terms(_entry("合成词", "合成词义")) if with_terms else ()
    system = system_of(
        context.build_snippet_prompt(prepared(), evidence(), habit_card=card, shared_terms=terms)
    )
    if card != CARD and not terms:
        assert context.SNIPPETS_START not in system
        return

    expected = [context.SNIPPETS_START]
    if card == CARD:
        expected.extend(["他的说话习惯：", *CARD.split("\n")])
    if terms:
        expected.extend([
            context.SHARED_TERMS_HEADING,
            "- 合成词：合成词义；场合：合成场合；合适时你也可以这样说，不必硬用",
        ])
    expected.append(context.SNIPPETS_END)
    assert system[system.index(context.SNIPPETS_START):] == "\n".join(expected)
    assert context.SNIPPET_INSTRUCTIONS[0] not in system
    assert "相似：" not in system
    assert "整体：" not in system


def test_neutral_without_snippets_keeps_original_prompt_bytes() -> None:
    terms = _terms(
        _entry("合成词", "合成词义"),
        _entry("合成理解词", "合成理解义", permission="understand_only"),
    )
    system = system_of(context.build_snippet_prompt(
        prepared(), evidence(register=AddresseeRegister.NEUTRAL),
        habit_card=CARD, shared_terms=terms,
    ))
    assert context.SNIPPETS_START not in system
    assert "他的说话习惯：" not in system
    assert CARD not in system
    assert context.SHARED_TERMS_HEADING not in system
    # Captured from the pre-change prompt using these same synthetic inputs.
    assert sha256(system.encode()).hexdigest() == (
        "53bd72a3827ab7e3154b6ad912646111b911daddf00903a0c2ad1d602d2dd4bb"
    )


def test_with_snippets_only_allowed_permission_wording_changes() -> None:
    terms = _terms(
        _entry("合成词", "合成词义"),
        _entry("合成理解词", "合成理解义", permission="understand_only"),
    )
    system = system_of(context.build_snippet_prompt(
        prepared(),
        evidence(similar=[snippet("合成她", "合成他")],
                 general=[snippet("合成她二", "合成他二")]),
        habit_card=CARD, shared_terms=terms,
    ))
    assert "合适时你也可以这样说，不必硬用" in system
    original = system.replace("合适时你也可以这样说，不必硬用", "你也可以这样说")
    assert sha256(original.encode()).hexdigest() == (
        "53ea227e90d14ad05b25bc6db79c73ea94f2948d6daa3dd3cc52d1ac67bf35ea"
    )


@pytest.mark.parametrize("with_snippets", (False, True))
def test_budget_can_keep_card_and_terms_after_all_snippets_are_trimmed(
    monkeypatch: pytest.MonkeyPatch, with_snippets: bool
) -> None:
    terms = _terms(_entry("合成词", "合成词义"))
    bare = context.build_snippet_prompt(prepared(), evidence())
    block = "\n".join([
        context.SNIPPETS_START, "他的说话习惯：", CARD, context.SHARED_TERMS_HEADING,
        "- 合成词：合成词义；场合：合成场合；合适时你也可以这样说，不必硬用",
        context.SNIPPETS_END,
    ])
    expected_system = system_of(bare) + "\n\n" + block
    budget = len(expected_system.encode()) + len(prepared().text.encode())
    monkeypatch.setattr(context, "MAX_PROMPT_UTF8_BYTES", budget)
    source = evidence(
        similar=[snippet("合成她", "合成他")] if with_snippets else [],
        general=[snippet("合成她二", "合成他二")] if with_snippets else [],
    )
    assert system_of(context.build_snippet_prompt(
        prepared(), source, habit_card=CARD, shared_terms=terms,
    )) == expected_system


@pytest.mark.parametrize("budget", (1, 5000))
@pytest.mark.parametrize("with_snippets", (False, True))
def test_over_budget_card_is_dropped_without_popping_empty_lists(
    monkeypatch: pytest.MonkeyPatch, budget: int, with_snippets: bool
) -> None:
    monkeypatch.setattr(context, "MAX_PROMPT_UTF8_BYTES", budget)
    source = evidence(
        similar=[snippet("合成她", "合成他")] if with_snippets else [],
        general=[snippet("合成她二", "合成他二")] if with_snippets else [],
    )
    result = context.build_snippet_prompt(prepared(), source, habit_card=CARD * 1000)
    assert context.SNIPPETS_START not in system_of(result)
    assert "他的说话习惯：" not in system_of(result)


def test_dropped_card_budget_is_available_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = MemoryEntry(
        memory_id="synthetic-memory", user_id="synthetic-user", field="合成字段", value="合成值",
        source_timestamp=FIXED_NOW, deleted_at=None,
    )
    expected = context.build_snippet_prompt(prepared(), evidence(), memories=[memory])
    budget = sum(len(message.content.encode()) for message in expected.messages)
    monkeypatch.setattr(context, "MAX_PROMPT_UTF8_BYTES", budget)
    actual = context.build_snippet_prompt(
        prepared(), evidence(similar=[snippet("合成她", "合成他")]),
        memories=[memory], habit_card=CARD * 1000,
    )
    assert actual == expected
    assert context.MEMORY_START in system_of(actual)


def test_fake_provider_receives_card_and_matched_terms_without_snippets(tmp_path: Path) -> None:
    provider = FakeProvider("合成安全回复")
    config = synthetic_dialogue_config(tmp_path)
    runtime = DialogueOrchestrator(
        config=config, snippet_style=FakeSnippetStyle(evidence()),
        snippet_habit_card=CARD, snippet_shared_terms=_terms(_entry("合成词", "合成词义")),
        provider=provider, ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    )
    run = runtime.reply(
        DialogueRequest(user_id="synthetic-user", text="合成词", recent_context=[]),
        trusted_user_id="synthetic-user",
    )
    assert run.response.blocked_reason is None
    assert provider.call_count == 1
    assert CARD in provider.system_text
    assert "- 合成词：合成词义" in provider.system_text
    assert context.SNIPPET_INSTRUCTIONS[0] not in provider.system_text
    assert run.trace.style_evidence_count == 0
