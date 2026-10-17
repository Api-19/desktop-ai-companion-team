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


def _evidence(register: AddresseeRegister, *, count: int = 1) -> StyleEvidence:
    def snippets(label: str) -> list[StyleSnippet]:
        return [
            StyleSnippet(partner_turns=[f"合成她{label}{i}"], owner_reply=[f"合成他{label}{i}"])
            for i in range(count)
        ]

    return StyleEvidence(
        style_version="synthetic-safety-v1",
        addressee_register=register,
        similar=snippets("相似"),
        general=snippets("整体"),
    )


def _prepared() -> PreparedDialogueInput:
    return PreparedDialogueInput(
        user_id="synthetic-user", text="合成当前消息", recent_context=("合成上文",)
    )


def _system(register: AddresseeRegister, **kwargs: object) -> str:
    bundle = build_snippet_prompt(_prepared(), _evidence(register), **kwargs)  # type: ignore[arg-type]
    return bundle.messages[0].content


def test_safety_section_follows_promises_and_precedes_context_in_both_registers() -> None:
    safety = context.SNIPPET_SAFETY
    for register in (AddresseeRegister.INTIMATE, AddresseeRegister.NEUTRAL):
        system = _system(register)
        assert system.count(safety) == 1
        promises_end = system.index(context.SNIPPET_PROMISES) + len(context.SNIPPET_PROMISES)
        safety_at = system.index(safety)
        assert promises_end + len("\n\n") == safety_at
        assert safety_at + len(safety) < system.index(CONTEXT_START)


def test_safety_section_survives_budget_trimming(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert "memory[5]" not in system
    assert system.count(context.SNIPPET_SAFETY) == 1
    assert system.index(context.SNIPPET_SAFETY) < system.index(CONTEXT_START)
    assert len(system.encode()) + user_bytes <= budget


def test_promises_defer_to_safety_rules() -> None:
    assert "涉及人身安全时，以安全规则为准" in context.SNIPPET_PROMISES  # noqa: RUF001
