"""Frozen checks 1-9: synthetic extraction, confirmation, dialogue and lifecycle.

Owner review and simulated subject confirmation use the public service methods.
M2 scenarios are prepared and validated locally; no real provider is called.
"""

# Chinese punctuation is intentional in synthetic messages.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import SecretStr

import companion.dialogue.composition as composition
from companion.contracts import ConsentRecord, DialogueRequest, StyleEvidence, StyleSnippet
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import (
    BudgetConfig,
    DialogueConfig,
    DialogueSecrets,
    ProviderConfig,
    ProviderResult,
)
from companion.dialogue.orchestrator import UNSAFE_OUTPUT_FALLBACK, DialogueOrchestrator
from companion.memory import CreateMemoryRequest, MemoryService
from companion.memory.profile import ProfileItem
from companion.runtime.settings import Settings
from scripts import extract_partner_profile

ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / "fixtures/synthetic/partner_profile_chat_v1.json"
SCENARIOS = ROOT / "fixtures/synthetic/partner_profile_m2_scenarios_v1.json"
PARTNER = "synthetic-partner"
OTHER = "synthetic-other"
NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)
PROFILE_START = "[BEGIN UNTRUSTED PARTNER PROFILE]"
PROFILE_END = "[END UNTRUSTED PARTNER PROFILE]"


class RecordingProvider:
    def __init__(self) -> None:
        self.text = "合成安全回复"
        self.requests: list[tuple[PromptMessage, ...]] = []

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        return ProviderResult(text=self.text)


class SyntheticSnippetAdapter:
    def accepts_user(self, user_id: str) -> bool:
        return user_id in {PARTNER, OTHER}

    def evidence_for(
        self, user_id: str, text: str, recent_context: Sequence[str]
    ) -> StyleEvidence:
        return StyleEvidence(
            style_version="synthetic-profile-e2e-v1",
            addressee_register="intimate" if user_id == PARTNER else "neutral",
            similar=[StyleSnippet(partner_turns=["合成问候"], owner_reply=["合成回应示例"])],
        )


def grant(service: MemoryService, user: str, now: datetime, *, profile: bool = True) -> None:
    service.grant_consent(
        ConsentRecord(
            user_id=user,
            granted=True,
            granted_at=now,
            allowed_fields=["preference", "profile"] if profile else ["preference"],
        )
    )


def confirm(service: MemoryService, item: ProfileItem) -> ProfileItem:
    return service.confirm_profile(
        user_id=item.user_id,
        profile_id=item.profile_id,
        version=item.version,
        subject_id=item.user_id,
        simulated=True,
        use_in_chat=True,
    )


class SyntheticChain:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.now = NOW
        self.database = root / "synthetic-profiles.sqlite3"
        self.service = MemoryService(self.database, clock=lambda: self.now)
        self.provider = RecordingProvider()
        chain = self

        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return chain.now if tz is None else chain.now.astimezone(tz)

        monkeypatch.setattr(extract_partner_profile, "datetime", FixedDatetime)
        monkeypatch.setattr(composition, "datetime", FixedDatetime)
        monkeypatch.setattr(composition, "OpenAICompatibleProvider", lambda **_: self.provider)
        grant(self.service, PARTNER, self.now)
        grant(self.service, OTHER, self.now)
        counts = extract_partner_profile._run(
            argparse.Namespace(
                chat=CHAT,
                memory_db=self.database,
                ledger=root / "synthetic-extraction-ledger.json",
                user_id=PARTNER,
                dry_run=False,
            )
        )
        assert counts["scanned"] == 18
        assert counts["candidates_added"] == 8
        items = self.service.list_profile(PARTNER)
        assert all(item.status == "candidate" and item.source == "self_stated" for item in items)
        self.quotes = [item.evidence_quote for item in items]
        assert all(self.quotes)
        self.items = {item.topic: item for item in items}

    def build(self, user: str = PARTNER) -> DialogueOrchestrator:
        config = DialogueConfig(
            provider=ProviderConfig(
                base_url="https://example.invalid/v1",
                model="synthetic-profile-provider",
                timeout_seconds=10,
            ),
            budget=BudgetConfig(
                monthly_cap_cny=Decimal("100.00"),
                worst_case_request_cny=Decimal("1.00"),
                input_cny_per_million=Decimal("1.00"),
                output_cny_per_million=Decimal("2.00"),
                max_output_tokens=256,
            ),
        )
        return composition.build_real_orchestrator(
            Settings(
                environment="test",
                log_level="INFO",
                private_data_dir=self.root,
                runtime_data_dir=self.root / "synthetic-runtime",
                test_results_dir=self.root / "synthetic-results",
            ),
            config,
            DialogueSecrets(api_key=SecretStr("synthetic-unused-key")),
            trusted_user_id=user,
            snippet_style=SyntheticSnippetAdapter(),
            snippet_habit_card="合成主人习惯：关心时说短句",
            profile_source=self.service,
        )

    def reply(
        self,
        runtime: DialogueOrchestrator,
        *,
        text: str = "今天喝水了吗",
        user: str = PARTNER,
        expected: Sequence[ProfileItem] = (),
        blocked: bool = False,
    ):
        before = len(self.provider.requests)
        run = runtime.reply(
            DialogueRequest(user_id=user, text=text, recent_context=[]), trusted_user_id=user
        )
        assert len(self.provider.requests) == before + 1
        prompt = "\n".join(message.content for message in self.provider.requests[-1])
        trace = run.trace.model_dump_json()
        for quote in self.quotes:
            assert quote not in prompt
            assert quote not in trace
        if expected:
            block = "\n".join(f"- {item.topic}：{item.text}" for item in expected)
            assert f"{PROFILE_START}\n{block}\n{PROFILE_END}" in prompt
            assert prompt.count(PROFILE_START) == 1
        else:
            assert PROFILE_START not in prompt
            assert PROFILE_END not in prompt
        reason = "profile_assertion" if blocked else None
        assert run.response.blocked_reason == reason
        assert run.trace.output_block_reason == reason
        return run

    def approve_and_confirm(self, topic: str) -> ProfileItem:
        approved = self.service.approve_profile(
            user_id=PARTNER, profile_id=self.items[topic].profile_id
        )
        assert approved.status == "owner_approved"
        assert self.service.retrieve_profile(user_id=PARTNER, query_texts=[topic]) == ()
        item = confirm(self.service, approved)
        assert item.status == "confirmed"
        assert item.confirmed_by == PARTNER and item.confirmed_version == item.version
        assert item.simulated and item.use_in_chat and item.evidence_quote is None
        return item


@pytest.fixture
def chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticChain:
    return SyntheticChain(tmp_path, monkeypatch)


def test_extraction_to_dialogue_correction_conflict_and_expiry(chain: SyntheticChain) -> None:
    """Checks 1-7, 9: real assembly, isolation, guard, edit, conflict and expiry."""
    service = chain.service
    runtime = chain.build()
    chain.reply(runtime)  # Candidates and evidence never enter prompts.
    water = chain.approve_and_confirm("喝水")
    sleep = chain.approve_and_confirm("作息")
    assert len(service.retrieve_profile(user_id=PARTNER, query_texts=["喝水"])) == 1
    chain.reply(runtime, expected=[water])
    chain.reply(runtime, text="合成拼图终于完成了")
    chain.reply(chain.build(OTHER), user=OTHER)
    assert service.retrieve_profile(user_id=OTHER, query_texts=["喝水"]) == ()

    chain.provider.text = "你又没喝水吧"
    blocked = chain.reply(runtime, expected=[water], blocked=True)
    assert UNSAFE_OUTPUT_FALLBACK in blocked.response.text
    assert chain.provider.text not in blocked.response.text
    chain.provider.text = "今天喝水了吗"
    allowed = chain.reply(runtime, expected=[water])
    assert chain.provider.text in allowed.response.text

    edited = service.edit_profile(
        user_id=PARTNER, profile_id=water.profile_id, text="喝水时喜欢小口慢喝"
    )
    assert edited.version == water.version + 1
    assert edited.status == "owner_approved" and edited.confirmed_version is None
    assert service.retrieve_profile(user_id=PARTNER, query_texts=["喝水"]) == ()
    chain.reply(runtime)
    water = confirm(service, edited)
    assert len(service.retrieve_profile(user_id=PARTNER, query_texts=["喝水"])) == 1
    chain.reply(runtime, expected=[water])

    conflicting = service.add_profile_candidate(
        user_id=PARTNER,
        topic="喝水",
        text="喝水时喜欢一口喝完",
        category="habit",
        source="owner_added",
        evidence_date=chain.now.date(),
        evidence_quote="合成冲突证据：我喜欢一口喝完",
        confidence="med",
    )
    chain.quotes.append(conflicting.evidence_quote)
    chain.reply(runtime, expected=[water])
    approved = service.approve_profile(user_id=PARTNER, profile_id=conflicting.profile_id)
    confirm(service, approved)
    conflicts = [item for item in service.list_profile(PARTNER) if item.topic == "喝水"]
    assert len(conflicts) == 2
    assert all(
        item.status == "disabled" and item.disabled_reason == "conflict" for item in conflicts
    )
    assert service.retrieve_profile(user_id=PARTNER, query_texts=["喝水"]) == ()
    chain.reply(runtime)

    # The nonconflicting sleep profile proves expiry independently of conflict.
    assert sleep.valid_until == date(2027, 2, 10)
    chain.now = datetime(2027, 2, 9, 23, 59, 59, tzinfo=UTC)
    chain.reply(runtime, text="作息怎么样", expected=[sleep])
    chain.now += timedelta(seconds=1)
    assert service.retrieve_profile(user_id=PARTNER, query_texts=["作息"]) == ()
    expired = service.list_profile(PARTNER)
    assert len(expired) == 1 and expired[0].profile_id == sleep.profile_id
    assert expired[0].status == "disabled" and expired[0].disabled_reason == "expired"
    chain.reply(runtime, text="作息怎么样")
    chain.now += timedelta(days=30, seconds=-1)
    assert len(service.list_profile(PARTNER)) == 1
    chain.now += timedelta(seconds=1)
    assert service.list_profile(PARTNER) == ()
    chain.reply(runtime, text="作息怎么样")


def test_delete_narrow_revoke_and_restart_preserve_other_user(chain: SyntheticChain) -> None:
    """Checks 1-3, 8-9 plus single deletion: both users survive scoped operations."""
    service = chain.service
    water = chain.approve_and_confirm("喝水")
    sleep = chain.approve_and_confirm("作息")
    memories = {}
    for user in (PARTNER, OTHER):
        memories[user] = service.create_memory(
            CreateMemoryRequest(
                user_id=user,
                field="preference",
                value=f"合成拼图偏好-{user}",
                source_timestamp=chain.now,
                explicitly_requested=True,
            )
        )
    other = service.add_profile_candidate(
        user_id=OTHER,
        topic="喝水",
        text="喝水时喜欢用合成方杯",
        category="preference",
        source="self_stated",
        evidence_date=chain.now.date(),
        evidence_quote="合成乙证据：我喜欢方杯",
        confidence="high",
    )
    chain.quotes.append(other.evidence_quote)
    other = confirm(service, service.approve_profile(user_id=OTHER, profile_id=other.profile_id))
    runtime = chain.build()
    chain.reply(runtime, expected=[water])
    chain.reply(chain.build(OTHER), user=OTHER, expected=[other])

    receipt = service.delete_profile(user_id=PARTNER, profile_id=sleep.profile_id)
    assert receipt.deleted_count == 1
    assert service.retrieve_profile(user_id=PARTNER, query_texts=["作息"]) == ()
    chain.reply(runtime, text="作息怎么样")
    chain.service = MemoryService(chain.database, clock=lambda: chain.now)
    service = chain.service
    assert sleep.profile_id not in {item.profile_id for item in service.list_profile(PARTNER)}
    chain.reply(chain.build(), expected=[water])
    chain.reply(chain.build(), text="作息怎么样")

    grant(service, PARTNER, chain.now, profile=False)
    assert service.list_profile(PARTNER) == ()
    assert service.retrieve_profile(user_id=PARTNER, query_texts=["喝水"]) == ()
    assert service.list_memories(PARTNER) == (memories[PARTNER],)
    chain.reply(runtime)
    chain.service = MemoryService(chain.database, clock=lambda: chain.now)
    service = chain.service
    assert service.list_profile(PARTNER) == ()
    assert service.list_memories(PARTNER) == (memories[PARTNER],)
    assert service.get_consent(PARTNER).allowed_fields == ["preference"]
    chain.reply(chain.build())

    # Regrant and add a fresh confirmed row so total withdrawal deletes both stores.
    grant(service, PARTNER, chain.now)
    fresh = service.add_profile_candidate(
        user_id=PARTNER,
        topic="喝水",
        text="喝水时喜欢合成圆杯",
        category="preference",
        source="owner_added",
        evidence_date=chain.now.date(),
        evidence_quote="合成新证据：我喜欢圆杯",
        confidence="med",
    )
    chain.quotes.append(fresh.evidence_quote)
    fresh = confirm(service, service.approve_profile(user_id=PARTNER, profile_id=fresh.profile_id))
    chain.reply(chain.build(), expected=[fresh])
    assert service.revoke_consent(PARTNER).deleted_count == 2
    assert service.list_profile(PARTNER) == ()
    assert service.list_memories(PARTNER) == ()
    chain.reply(chain.build())
    chain.service = MemoryService(chain.database, clock=lambda: chain.now)
    service = chain.service
    assert service.list_profile(PARTNER) == ()
    assert service.list_memories(PARTNER) == ()
    assert service.retrieve_profile(user_id=PARTNER, query_texts=["喝水"]) == ()
    assert service.get_consent(PARTNER).granted is False
    chain.reply(chain.build())
    assert service.list_profile(OTHER) == (other,)
    assert service.list_memories(OTHER) == (memories[OTHER],)
    chain.reply(chain.build(OTHER), user=OTHER, expected=[other])


def test_m2_scenarios_structure_and_content() -> None:
    """Acceptance 2: 24 synthetic M2 inputs; evaluation date is 2026-10-10.

    Profile evidence is assumed no later than confirmation in these scenarios.
    These cases are prepared only and are never sent to a real model here.
    """
    raw = SCENARIOS.read_text(encoding="utf-8")
    scenarios = json.loads(raw)
    assert isinstance(scenarios, list) and len(scenarios) == 24
    assert Counter(row["group"] for row in scenarios) == {
        "relevant": 12, "stale": 4, "conflict": 4, "out_of_scope": 4
    }
    assert len({row["id"] for row in scenarios}) == 24
    assert re.search("自杀|自伤|杀|伤害", raw) is None
    for row in scenarios:
        assert set(row) == {"id", "group", "profile", "recent_context", "text", "expect"}
        assert isinstance(row["id"], str) and row["id"].startswith("synthetic-m2-")
        assert isinstance(row["profile"], list) and row["profile"]
        assert isinstance(row["recent_context"], list) and 0 <= len(row["recent_context"]) <= 4
        assert all(isinstance(text, str) and text for text in row["recent_context"])
        assert isinstance(row["text"], str) and row["text"]
        assert isinstance(row["expect"], str) and row["expect"]
        for item in row["profile"]:
            assert set(item) == {"topic", "text", "category", "confirmed"}
            assert isinstance(item["topic"], str) and 1 <= len(item["topic"]) <= 12
            assert isinstance(item["text"], str) and 1 <= len(item["text"]) <= 60
            assert item["category"] in {"habit", "preference", "reminder_pref"}
            assert date.fromisoformat(item["confirmed"]).isoformat() == item["confirmed"]
