"""Synthetic Memory-to-Dialogue lifecycle across reconstructed services."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

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
from companion.memory import (
    CorrectMemoryRequest,
    MemoryField,
    MemoryService,
    build_private_memory_service,
)
from companion.runtime.settings import Settings

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
USER_A = "synthetic-lifecycle-a"
USER_B = "synthetic-lifecycle-b"
QUERY = "合成蓝色"
A_OLD = "喜欢合成蓝色甲旧"
A_NEW = "喜欢合成蓝色甲新"
B_VALUE = "喜欢合成蓝色乙"


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[PromptMessage, ...]] = []

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        return ProviderResult(text="合成安全回复")

    @property
    def last_prompt(self) -> str:
        return "\n".join(message.content for message in self.requests[-1])


class UnavailableStyle:
    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def guidance_for(self, *args: object, **kwargs: object) -> StyleReplyGuidance:
        return StyleReplyGuidance(status="unavailable")


def _config(root: Path) -> DialogueConfig:
    return DialogueConfig(
        provider=ProviderConfig(
            base_url="https://example.invalid/v1",
            model="synthetic-lifecycle-provider",
            timeout_seconds=10,
        ),
        budget=BudgetConfig(
            monthly_cap_cny=Decimal("100.00"),
            worst_case_request_cny=Decimal("1.00"),
            input_cny_per_million=Decimal("1.00"),
            output_cny_per_million=Decimal("2.00"),
            max_output_tokens=256,
        ),
        style=StyleRuntimeSources(
            candidate_source=root / "synthetic-candidates.json",
            expected_count=1,
            active_style_source=root / "synthetic-style.json",
            active_manifest_source=root / "synthetic-manifest.json",
            expected_active_manifest_digest="a" * 64,
            calibration_source=root / "synthetic-calibration.json",
            expected_calibration_fingerprint="b" * 64,
            private_data_dir=root,
            true_name_terms=("synthetic-owner",),
            approved_nicknames=("synthetic-nickname",),
        ),
    )


def _system(root: Path) -> tuple[MemoryService, DialogueOrchestrator, RecordingProvider]:
    memory = MemoryService(root / "synthetic-memory.sqlite3", clock=lambda: NOW)
    provider = RecordingProvider()
    config = _config(root)
    dialogue = DialogueOrchestrator(
        config=config,
        style_adapter=UnavailableStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(root / "usage", config.budget),
        clock=lambda: NOW,
        memory_gateway=memory,
    )
    return memory, dialogue, provider


def _reply(dialogue: DialogueOrchestrator, user_id: str, text: str):
    return dialogue.reply(
        DialogueRequest(user_id=user_id, text=text, recent_context=[]),
        trusted_user_id=user_id,
    )


def _grant(memory: MemoryService, user_id: str) -> None:
    memory.grant_consent(
        ConsentRecord(
            user_id=user_id,
            granted=True,
            granted_at=NOW,
            allowed_fields=[field.value for field in MemoryField],
        )
    )


def test_authorized_memory_lifecycle_survives_restart_without_old_prompt_values(
    tmp_path: Path,
) -> None:
    memory, dialogue, provider = _system(tmp_path)

    denied = _reply(dialogue, USER_A, f"记住我{A_OLD}")
    assert denied.trace.provider_attempted is False
    assert memory.list_memories(USER_A) == ()
    assert provider.requests == []

    _grant(memory, USER_A)
    _grant(memory, USER_B)
    assert _reply(dialogue, USER_A, f"记住我{A_OLD}").trace.provider_attempted is False
    assert _reply(dialogue, USER_B, f"记住我{B_VALUE}").trace.provider_attempted is False
    assert provider.requests == []
    owner_entries = memory.list_memories(USER_A)
    other_entries = memory.list_memories(USER_B)
    assert len(owner_entries) == len(other_entries) == 1
    assert owner_entries[0].value == A_OLD
    assert other_entries[0].value == B_VALUE

    assert {entry.value for entry in memory.retrieve_memories(USER_A, QUERY)} == {A_OLD}
    assert {entry.value for entry in memory.retrieve_memories(USER_B, QUERY)} == {B_VALUE}
    assert _reply(dialogue, USER_A, QUERY).response.blocked_reason is None
    assert A_OLD in provider.last_prompt
    assert B_VALUE not in provider.last_prompt
    assert _reply(dialogue, USER_B, QUERY).response.blocked_reason is None
    assert B_VALUE in provider.last_prompt
    assert A_OLD not in provider.last_prompt

    corrected = memory.correct_memory(
        CorrectMemoryRequest(
            user_id=USER_A,
            memory_id=owner_entries[0].memory_id,
            value=A_NEW,
            source_timestamp=NOW + timedelta(seconds=1),
        )
    )
    assert corrected.value == A_NEW
    memory, dialogue, provider = _system(tmp_path)
    assert {entry.value for entry in memory.list_memories(USER_A)} == {A_NEW}
    assert {entry.value for entry in memory.retrieve_memories(USER_A, QUERY)} == {A_NEW}
    assert memory.retrieve_memories(USER_A, A_OLD) == ()
    assert _reply(dialogue, USER_A, QUERY).response.blocked_reason is None
    assert A_NEW in provider.last_prompt
    assert A_OLD not in provider.last_prompt
    assert B_VALUE not in provider.last_prompt

    receipt = memory.delete_memory(USER_A, owner_entries[0].memory_id)
    assert receipt.deleted_count == 1
    memory, dialogue, provider = _system(tmp_path)
    assert memory.list_memories(USER_A) == ()
    assert memory.retrieve_memories(USER_A, QUERY) == ()
    assert memory.retrieve_memories(USER_A, A_OLD) == ()
    assert memory.retrieve_memories(USER_A, A_NEW) == ()
    assert _reply(dialogue, USER_A, QUERY).response.blocked_reason is None
    assert A_OLD not in provider.last_prompt
    assert A_NEW not in provider.last_prompt
    assert B_VALUE not in provider.last_prompt
    assert {entry.value for entry in memory.list_memories(USER_B)} == {B_VALUE}
    assert _reply(dialogue, USER_B, QUERY).response.blocked_reason is None
    assert B_VALUE in provider.last_prompt
    assert A_OLD not in provider.last_prompt
    assert A_NEW not in provider.last_prompt


def _factory_system(
    root: Path,
) -> tuple[MemoryService, DialogueOrchestrator, RecordingProvider]:
    settings = Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=root / "private",
        runtime_data_dir=root / "runtime",
        test_results_dir=root / "results",
    )
    memory = build_private_memory_service(settings, clock=lambda: NOW)
    provider = RecordingProvider()
    config = _config(root)
    dialogue = DialogueOrchestrator(
        config=config,
        style_adapter=UnavailableStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(root / "usage", config.budget),
        clock=lambda: NOW,
        memory_gateway=memory,
    )
    return memory, dialogue, provider


@pytest.mark.skipif(os.name != "posix", reason="POSIX private storage only")
def test_private_factory_lifecycle_survives_restart_through_dialogue(tmp_path: Path) -> None:
    (tmp_path / "private").mkdir(mode=0o700)
    (tmp_path / "private").chmod(0o700)
    memory, dialogue, provider = _factory_system(tmp_path)

    assert _reply(dialogue, USER_A, f"记住我{A_OLD}").trace.provider_attempted is False
    assert memory.list_memories(USER_A) == ()
    _grant(memory, USER_A)
    _grant(memory, USER_B)
    assert _reply(dialogue, USER_A, f"记住我{A_OLD}").trace.provider_attempted is False
    assert _reply(dialogue, USER_B, f"记住我{B_VALUE}").trace.provider_attempted is False
    owner_entry = memory.list_memories(USER_A)[0]
    memory.correct_memory(
        CorrectMemoryRequest(
            user_id=USER_A,
            memory_id=owner_entry.memory_id,
            value=A_NEW,
            source_timestamp=NOW + timedelta(seconds=1),
        )
    )

    memory, dialogue, provider = _factory_system(tmp_path)
    assert _reply(dialogue, USER_A, QUERY).response.blocked_reason is None
    assert A_NEW in provider.last_prompt
    assert A_OLD not in provider.last_prompt
    assert B_VALUE not in provider.last_prompt
    assert memory.delete_memory(USER_A, owner_entry.memory_id).deleted_count == 1
    assert memory.revoke_consent(USER_B).deleted_count == 1

    memory, dialogue, provider = _factory_system(tmp_path)
    assert _reply(dialogue, USER_A, QUERY).response.blocked_reason is None
    assert _reply(dialogue, USER_B, QUERY).response.blocked_reason is None
    for value in (A_OLD, A_NEW, B_VALUE):
        assert value not in "\n".join(m.content for r in provider.requests for m in r)
    database = tmp_path / "private" / "memory" / "memory.sqlite3"
    assert sorted(path.name for path in database.parent.iterdir()) == [database.name]
    assert stat.S_IMODE(database.parent.lstat().st_mode) == 0o700
    assert stat.S_IMODE(database.lstat().st_mode) == 0o600
