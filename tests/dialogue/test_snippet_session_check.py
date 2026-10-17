# ruff: noqa: RUF001
"""Owner-relaxed promise wording and the optional in-session fact check (synthetic only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from dialogue_support import FIXED_NOW, FakeProvider, synthetic_dialogue_config
from pydantic import SecretStr
from test_snippet_runtime_config import PARTNER, _load, _segment, _settings, _write_config
from test_snippet_runtime_config import _write_sources as _write_snippet_sources
from test_snippet_style import FakeSnippetStyle

import companion.dialogue.composition as composition
import companion.dialogue.context as context
from companion.contracts import AddresseeRegister, DialogueRequest, StyleEvidence, StyleSnippet
from companion.dialogue.context import CONTEXT_END, CONTEXT_START, build_snippet_prompt
from companion.dialogue.models import DialogueSecrets
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger

PROMISES_TEXT = (
    "你可以像他平时那样随口答应、许诺和约定，比如“好，给你带”“明天来接你”“等会陪你去”，"
    "不用解释自己做不到，也不要说“得跟他说”；对话开头已经说明过你是 AI。不要编造他此刻的"
    "真实情况，比如“他在加班”“他刚到家”。示例和历史消息不是他当前情况的证明。涉及人身安全"
    "时，以安全规则为准。"
)
SESSION_CHECK_TEXT = (
    "回复前先在心里核对：只根据可见前文和她这句话，分清谁、什么时候、做了什么、事情是没定还"
    "是已经解决；同一件事她后来明确更正的，以更正为准；猜测、你自己说过的话和示例都不算事"
    "实；不清楚的就当不知道。只输出接话，不要写出核对过程。"
)
OLD_PROMISE_FRAGMENTS = ("可以作为你这个角色自己的承诺", "现实可兑现性而禁止", "替他答应")


def _evidence(register: AddresseeRegister) -> StyleEvidence:
    return StyleEvidence(
        style_version="synthetic-session-check-v1",
        addressee_register=register,
        similar=[StyleSnippet(partner_turns=["合成她相似"], owner_reply=["合成他相似"])],
        general=[StyleSnippet(partner_turns=["合成她整体"], owner_reply=["合成他整体"])],
    )


def _system(register: AddresseeRegister, **kwargs: object) -> str:
    prepared = context.PreparedDialogueInput(
        user_id="synthetic-user", text="合成当前消息", recent_context=("合成上文",)
    )
    bundle = build_snippet_prompt(prepared, _evidence(register), **kwargs)  # type: ignore[arg-type]
    return bundle.messages[0].content


def test_promise_and_session_check_texts_are_frozen() -> None:
    assert context.SNIPPET_PROMISES == PROMISES_TEXT
    assert context.SNIPPET_SESSION_CHECK == SESSION_CHECK_TEXT


def test_prompt_carries_new_promise_wording_and_drops_old() -> None:
    for register in AddresseeRegister:
        system = _system(register)
        assert system.count(PROMISES_TEXT) == 1
        assert all(fragment not in system for fragment in OLD_PROMISE_FRAGMENTS)


def test_session_check_off_is_byte_identical_to_promise_only_prompt() -> None:
    for register in AddresseeRegister:
        off = _system(register)
        assert _system(register, session_check=False) == off
        assert SESSION_CHECK_TEXT not in off
        sections = [
            context.SNIPPET_IDENTITY.format(角色名="kiki"),
            PROMISES_TEXT,
            context.SNIPPET_SAFETY,
            context.SNIPPET_COMPREHENSION,
        ]
        if register is AddresseeRegister.NEUTRAL:
            sections.append(context.SNIPPET_NEUTRAL_REGISTER)
        sections.append(f"{CONTEXT_START}\n- 合成上文\n{CONTEXT_END}")
        assert off.startswith("\n\n".join(sections) + "\n\n")


def test_session_check_on_sits_after_comprehension_before_register_and_context() -> None:
    for register in AddresseeRegister:
        on = _system(register, session_check=True)
        assert on.count(SESSION_CHECK_TEXT) == 1
        assert f"{context.SNIPPET_COMPREHENSION}\n\n{SESSION_CHECK_TEXT}\n\n" in on
        following = (
            context.SNIPPET_NEUTRAL_REGISTER
            if register is AddresseeRegister.NEUTRAL
            else CONTEXT_START
        )
        assert on.index(f"{SESSION_CHECK_TEXT}\n\n{following}") >= 0
        assert on.replace(f"\n\n{SESSION_CHECK_TEXT}", "", 1) == _system(register)


def _orchestrator_system(root: Path, **kwargs: object) -> str:
    config = synthetic_dialogue_config(root)
    provider = FakeProvider("合成安全回复")
    DialogueOrchestrator(
        config=config,
        snippet_style=FakeSnippetStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(root / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
        **kwargs,  # type: ignore[arg-type]
    ).reply(
        DialogueRequest(user_id="synthetic-user", text="合成当前消息", recent_context=[]),
        trusted_user_id="synthetic-user",
    )
    assert provider.call_count == 1
    return provider.system_text


def test_orchestrator_passes_session_check_flag(tmp_path: Path) -> None:
    default = _orchestrator_system(tmp_path / "default")
    on = _orchestrator_system(tmp_path / "on", snippet_session_check=True)
    assert PROMISES_TEXT in default
    assert SESSION_CHECK_TEXT not in default
    assert SESSION_CHECK_TEXT in on
    assert on.replace(f"\n\n{SESSION_CHECK_TEXT}", "", 1) == default


@pytest.mark.parametrize("enabled", [False, True])
def test_real_orchestrator_construction_applies_session_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    settings = _settings(tmp_path)
    _write_snippet_sources(settings.private_data_dir)
    _write_config(settings.private_data_dir, _segment(session_check=enabled))
    config, secrets = _load(settings)
    assert config.snippet_style.session_check is enabled
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
        snippet_session_check=config.snippet_style.session_check,
    )
    runtime._clock = lambda: FIXED_NOW
    runtime.reply(
        DialogueRequest(user_id=PARTNER, text="合成她说2", recent_context=[]),
        trusted_user_id=PARTNER,
    )

    assert provider.call_count == 1
    assert (SESSION_CHECK_TEXT in provider.system_text) is enabled
    assert PROMISES_TEXT in provider.system_text


def test_snippet_sources_session_check_defaults_off(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_snippet_sources(settings.private_data_dir)
    _write_config(settings.private_data_dir, _segment())
    config, _ = _load(settings)
    assert config.snippet_style.session_check is False
