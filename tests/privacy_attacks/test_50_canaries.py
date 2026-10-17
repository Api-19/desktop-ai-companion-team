"""Synthetic third-user canaries across Memory and the real Dialogue path."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from companion.contracts import ConsentRecord, DialogueRequest, StyleReplyGuidance
from companion.dialogue.config import DialogueConfig
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import (
    BudgetConfig,
    ProviderConfig,
    ProviderResult,
    StyleRuntimeSources,
)
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.memory import CreateMemoryRequest, MemoryField, MemoryService

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
VICTIM = "synthetic-canary-owner"
THIRD_USER = "synthetic-canary-third-user"
CANARIES = tuple(f"SYNTHETIC_PRIVATE_{index:02d}_VALUE" for index in range(50))


class RecordingProvider:
    def __init__(self) -> None:
        self.prompts: list[tuple[PromptMessage, ...]] = []

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.prompts.append(tuple(messages))
        return ProviderResult(text="合成安全回复")


class UnavailableStyle:
    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def guidance_for(self, *args: object, **kwargs: object) -> StyleReplyGuidance:
        return StyleReplyGuidance(status="unavailable")


def _config(root: Path) -> DialogueConfig:
    return DialogueConfig(
        provider=ProviderConfig(
            base_url="https://example.invalid/v1",
            model="synthetic-canary-provider",
            timeout_seconds=120,
        ),
        budget=BudgetConfig(
            monthly_cap_cny=Decimal("100.00"),
            worst_case_request_cny=Decimal("1.00"),
            input_cny_per_million=Decimal("1.00"),
            output_cny_per_million=Decimal("2.00"),
            max_output_tokens=256,
        ),
        style=StyleRuntimeSources(
            candidate_source=root / "style-candidates.json",
            expected_count=1,
            active_style_source=root / "active-style.json",
            active_manifest_source=root / "style-manifest.json",
            expected_active_manifest_digest="a" * 64,
            calibration_source=root / "calibration-candidates.json",
            expected_calibration_fingerprint="b" * 64,
            private_data_dir=root,
            true_name_terms=("synthetic-owner",),
            approved_nicknames=("synthetic-nickname",),
        ),
    )


def _setup(tmp_path: Path) -> tuple[MemoryService, DialogueOrchestrator, RecordingProvider]:
    memory = MemoryService(tmp_path / "synthetic-memory.sqlite3", clock=lambda: NOW)
    for user_id in (VICTIM, THIRD_USER):
        memory.grant_consent(
            ConsentRecord(
                user_id=user_id,
                granted=True,
                granted_at=NOW,
                allowed_fields=[field.value for field in MemoryField],
            )
        )
    for index, canary in enumerate(CANARIES):
        memory.create_memory(
            CreateMemoryRequest(
                user_id=VICTIM,
                field=MemoryField.EXPLICIT_NOTE,
                value=canary,
                source_timestamp=NOW + timedelta(seconds=index),
                explicitly_requested=True,
            )
        )
    provider = RecordingProvider()
    config = _config(tmp_path)
    dialogue = DialogueOrchestrator(
        config=config,
        style_adapter=UnavailableStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "usage", config.budget),
        clock=lambda: NOW,
        memory_gateway=memory,
    )
    return memory, dialogue, provider


def _reply(dialogue: DialogueOrchestrator, user_id: str, query: str):
    return dialogue.reply(
        DialogueRequest(user_id=user_id, text=query, recent_context=[]),
        trusted_user_id=user_id,
    )


def _prompt_text(provider: RecordingProvider) -> str:
    return "\n".join(message.content for message in provider.prompts[-1])


def test_fifty_canaries_do_not_leak_to_third_user(tmp_path: Path) -> None:
    memory, dialogue, provider = _setup(tmp_path)
    assert len(CANARIES) == len(set(CANARIES)) == 50
    assert len(memory.list_memories(VICTIM)) == 50
    assert memory.list_memories(THIRD_USER) == ()

    owner_query = "SYNTHETIC_PRIVATE_00"
    assert CANARIES[0] in {entry.value for entry in memory.retrieve_memories(VICTIM, owner_query)}
    owner_run = _reply(dialogue, VICTIM, owner_query)
    assert owner_run.response.blocked_reason is None
    assert CANARIES[0] in _prompt_text(provider)

    checked = 0
    for index, canary in enumerate(CANARIES):
        query = f"SYNTHETIC_PRIVATE_{index:02d}"
        assert canary not in query
        before = len(provider.prompts)
        retrieved = memory.retrieve_memories(THIRD_USER, query)
        assert retrieved == (), f"case {index}: third-user retrieval"
        run = _reply(dialogue, THIRD_USER, query)
        assert len(provider.prompts) == before + 1, f"case {index}: provider call"
        assert run.response.blocked_reason is None, f"case {index}: dialogue blocked"
        prompt_text = _prompt_text(provider)
        response_and_trace = run.model_dump_json()
        leaks = {
            "retrieval": [
                value for value in CANARIES if any(value in entry.value for entry in retrieved)
            ],
            "provider_prompt": [value for value in CANARIES if value in prompt_text],
            "response_or_trace": [value for value in CANARIES if value in response_and_trace],
        }
        assert not any(leaks.values()), f"case {index}: {leaks}"
        checked += 1

    assert checked == 50


def test_revocation_removes_canary_from_next_owner_request(tmp_path: Path) -> None:
    memory, dialogue, provider = _setup(tmp_path)
    query = "SYNTHETIC_PRIVATE_00"
    owner_run = _reply(dialogue, VICTIM, query)
    assert owner_run.response.blocked_reason is None
    assert CANARIES[0] in _prompt_text(provider)

    receipt = memory.revoke_consent(VICTIM)
    assert receipt.deleted_count == 50
    assert memory.retrieve_memories(VICTIM, query) == ()
    assert memory.list_memories(VICTIM) == ()
    run = _reply(dialogue, VICTIM, query)

    assert run.response.blocked_reason is None
    assert CANARIES[0] not in _prompt_text(provider)
    assert CANARIES[0] not in run.model_dump_json()
