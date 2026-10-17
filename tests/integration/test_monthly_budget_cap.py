# ruff: noqa: RUF001
"""Synthetic monthly budget cap through the real Dialogue and ledger path."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

from companion.contracts import DialogueRequest, StyleReplyGuidance
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
from companion.runtime import budget_notice
from companion.runtime.speech_dialogue import SpeechDialogueBridge

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
USER = "synthetic-budget-user"
EXPECTED_FALLBACK = "这个月的云端聊天额度用完啦，云端回复先停一下，让主人看看额度吧。"


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[PromptMessage, ...]] = []

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        return ProviderResult(text="合成安全答复", usage=None)


class UnavailableStyle:
    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def guidance_for(self, *args: object, **kwargs: object) -> StyleReplyGuidance:
        return StyleReplyGuidance(status="unavailable")


def _config(root: Path) -> DialogueConfig:
    return DialogueConfig(
        provider=ProviderConfig(
            base_url="https://example.invalid/v1",
            model="synthetic-budget-provider",
            timeout_seconds=10,
        ),
        budget=BudgetConfig(
            monthly_cap_cny=Decimal("100.00"),
            worst_case_request_cny=Decimal("50.00"),
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


def _orchestrator(root: Path, provider: RecordingProvider) -> DialogueOrchestrator:
    config = _config(root)
    return DialogueOrchestrator(
        config=config,
        style_adapter=UnavailableStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(root / "usage", config.budget),
        clock=lambda: NOW,
    )


def _reply(orchestrator: DialogueOrchestrator, text: str):
    return orchestrator.reply(
        DialogueRequest(user_id=USER, text=text, recent_context=[]),
        trusted_user_id=USER,
    )


def _state(root: Path) -> dict[str, object]:
    path = root / "usage" / "dialogue-usage-2026-09.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_cap_transition_notifies_after_settlement_without_extra_provider_call(
    tmp_path: Path, monkeypatch,
) -> None:
    provider = RecordingProvider()
    orchestrator = _orchestrator(tmp_path, provider)
    ledger = MonthlyUsageLedger(tmp_path / "usage", _config(tmp_path).budget)
    path = tmp_path / "usage" / "dialogue-usage-2026-09.json"
    delivered_states = []

    def delivered() -> None:
        snapshot = ledger.snapshot(NOW)
        assert snapshot.settled_cny == Decimal("100")
        assert snapshot.reserved_cny == 0
        assert snapshot.status == "blocked"
        state = _state(tmp_path)
        assert state["reservations"] == {}
        assert state["settled_cny"] == "100.000000"
        assert len(provider.requests) == 2
        delivered_states.append(state)

    sender = Mock(side_effect=delivered)
    monkeypatch.setattr(budget_notice, "send_mac_budget_notice", sender)
    notifier = budget_notice.MacBudgetNotifier(ledger=ledger, clock=lambda: NOW)
    bridge = SpeechDialogueBridge(
        orchestrator=orchestrator, privacy_revision=orchestrator.privacy_revision,
        budget_notice=notifier.check,
    )
    notifier.check()
    sender.assert_not_called()
    assert provider.requests == []
    assert not path.exists()

    first = bridge.reply(user_id=USER, text="第一条合成普通问题")
    assert first.blocked_reason is None and first.billable
    assert ledger.snapshot(NOW).settled_cny == Decimal("50")
    sender.assert_not_called()
    second = bridge.reply(user_id=USER, text="第二条合成普通问题")
    assert second.blocked_reason is None and second.billable
    assert ledger.snapshot(NOW).settled_cny == Decimal("100")
    sender.assert_called_once_with()
    at_cap = path.read_bytes()

    for text in ("第三条合成普通问题", "第四条合成普通问题"):
        response = bridge.reply(user_id=USER, text=text)
        assert response.blocked_reason == "budget_exceeded"
        assert response.text.endswith(EXPECTED_FALLBACK)
        assert not response.billable
        assert len(provider.requests) == 2
        assert path.read_bytes() == at_cap
        sender.assert_called_once_with()

    restarted = budget_notice.MacBudgetNotifier(
        ledger=MonthlyUsageLedger(tmp_path / "usage", _config(tmp_path).budget), clock=lambda: NOW,
    )
    restarted.check()
    restarted.check()
    assert sender.call_count == 2
    assert len(delivered_states) == 2
    assert len(provider.requests) == 2
    assert path.read_bytes() == at_cap
    bridge.close()


def test_cap_blocks_two_followups_and_reconstructed_orchestrator(tmp_path: Path) -> None:
    provider = RecordingProvider()
    orchestrator = _orchestrator(tmp_path, provider)

    first = _reply(orchestrator, "第一条合成普通问题")
    assert first.response.blocked_reason is None
    assert first.response.billable is True
    assert first.trace.provider_attempted is True
    assert first.trace.usage_reliable is False
    assert len(provider.requests) == 1
    assert provider.requests[0][1].content == "第一条合成普通问题"
    assert _state(tmp_path)["settled_cny"] == "50.000000"

    second = _reply(orchestrator, "第二条合成普通问题")
    assert second.response.blocked_reason is None
    assert second.response.billable is True
    assert second.trace.provider_attempted is True
    assert second.trace.usage_reliable is False
    assert len(provider.requests) == 2
    state_at_cap = _state(tmp_path)
    assert state_at_cap == {
        "schema_version": 1,
        "month": "2026-09",
        "settled_cny": "100.000000",
        "reservations": {},
        "requires_review": False,
    }

    for text in ("第三条合成普通问题", "第四条合成普通问题"):
        blocked = _reply(orchestrator, text)
        assert blocked.response.blocked_reason == "budget_exceeded"
        assert blocked.response.text.endswith(EXPECTED_FALLBACK)
        assert text not in blocked.response.text
        assert blocked.response.billable is False
        assert blocked.trace.provider_attempted is False
        assert blocked.trace.provider_call_count == 0
        assert len(provider.requests) == 2
        assert _state(tmp_path) == state_at_cap

    reconstructed = _orchestrator(tmp_path, provider)
    after_reconstruction = _reply(reconstructed, "重建后的合成普通问题")
    assert after_reconstruction.response.blocked_reason == "budget_exceeded"
    assert after_reconstruction.response.text.endswith(EXPECTED_FALLBACK)
    assert after_reconstruction.response.billable is False
    assert after_reconstruction.trace.provider_attempted is False
    assert after_reconstruction.trace.provider_call_count == 0
    assert len(provider.requests) == 2
    assert _state(tmp_path) == state_at_cap
