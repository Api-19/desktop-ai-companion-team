from __future__ import annotations

import pytest
from dialogue_support import FIXED_NOW

import companion.dialogue.context as context
from companion.contracts import AddresseeRegister, MemoryEntry, StyleEvidence, StyleSnippet
from companion.dialogue.context import (
    CONTEXT_START,
    PreparedDialogueInput,
    build_snippet_prompt,
)

FROZEN_TEXT = (
    "接话要贴着她刚说的事：分清时间（今天、明天、刚才）和谁做了什么，留意她刚说的限制"  # noqa: RUF001
    "或更正。玩笑和跳着接话都可以，但别和她说的事实冲突：别把未来说成已经发生，别换掉"  # noqa: RUF001
    "做事的人，别把猜的事当成真的。"  # noqa: RUF001
)


def _evidence(register: AddresseeRegister, *, count: int = 1) -> StyleEvidence:
    def snippets(label: str) -> list[StyleSnippet]:
        return [
            StyleSnippet(partner_turns=[f"合成她{label}{i}"], owner_reply=[f"合成他{label}{i}"])
            for i in range(count)
        ]

    return StyleEvidence(
        style_version="synthetic-comprehension-v1",
        addressee_register=register,
        similar=snippets("相似"),
        general=snippets("整体"),
    )


def _prepared() -> PreparedDialogueInput:
    return PreparedDialogueInput(
        user_id="synthetic-user", text="合成当前消息", recent_context=("合成上文",)
    )


def _system(register: AddresseeRegister) -> str:
    return build_snippet_prompt(_prepared(), _evidence(register)).messages[0].content


def test_comprehension_text_is_frozen() -> None:
    assert context.SNIPPET_COMPREHENSION == FROZEN_TEXT


def test_comprehension_follows_safety_and_precedes_context_in_both_registers() -> None:
    comprehension = context.SNIPPET_COMPREHENSION
    for register in (AddresseeRegister.INTIMATE, AddresseeRegister.NEUTRAL):
        system = _system(register)
        assert system.count(comprehension) == 1
        safety_end = system.index(context.SNIPPET_SAFETY) + len(context.SNIPPET_SAFETY)
        comprehension_at = system.index(comprehension)
        assert safety_end + len("\n\n") == comprehension_at
        assert comprehension_at + len(comprehension) < system.index(CONTEXT_START)


def test_empty_comprehension_is_omitted_without_blank_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context, "SNIPPET_COMPREHENSION", "")
    for register in (AddresseeRegister.INTIMATE, AddresseeRegister.NEUTRAL):
        system = _system(register)
        assert FROZEN_TEXT not in system
        assert "\n\n\n" not in system
        safety_end = system.index(context.SNIPPET_SAFETY) + len(context.SNIPPET_SAFETY)
        next_section = (
            context.SNIPPET_NEUTRAL_REGISTER
            if register is AddresseeRegister.NEUTRAL
            else CONTEXT_START
        )
        assert system.index(next_section) == safety_end + len("\n\n")


def test_comprehension_survives_budget_trimming(monkeypatch: pytest.MonkeyPatch) -> None:
    memories = [
        MemoryEntry(
            memory_id=f"m-{i}",
            user_id="synthetic-user",
            field="合成字段",
            value="合成值" * 20,
            source_timestamp=FIXED_NOW,
            deleted_at=None,
        )
        for i in range(5)
    ]
    evidence = _evidence(AddresseeRegister.INTIMATE, count=5)
    user_bytes = len(_prepared().text.encode())
    bare = build_snippet_prompt(_prepared(), _evidence(AddresseeRegister.INTIMATE, count=0))
    budget = len(bare.messages[0].content.encode()) + user_bytes + 300
    monkeypatch.setattr(context, "MAX_PROMPT_UTF8_BYTES", budget)

    system = build_snippet_prompt(_prepared(), evidence, memories).messages[0].content

    assert system.count("\n- 她：") < len(evidence.similar) + len(evidence.general)  # noqa: RUF001
    assert system.count(context.SNIPPET_COMPREHENSION) == 1
    assert system.index(context.SNIPPET_COMPREHENSION) < system.index(CONTEXT_START)
    assert len(system.encode()) + user_bytes <= budget
