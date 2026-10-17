"""Live composition at the outbound provider boundary, using disposable synthetic files."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from dialogue_support import FIXED_NOW, FakeProvider
from test_snippet_runtime_config import PARTNER, _loaded_config, _snippet

import companion.dialogue.composition as composition
from companion.contracts import DialogueRequest
from companion.dialogue.usage import MonthlyUsageLedger
from companion.privacy.paired_snippets import delete_snippets, write_snippets

DELETED_CANARY = "SYNTHETIC_DELETED_EVIDENCE"
RETAINED_CANARY = "SYNTHETIC_RETAINED_EVIDENCE"
CORRUPT_CANARY = "SYNTHETIC_CORRUPT_SOURCE"


@pytest.fixture
def live_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings, config, secrets = _loaded_config(tmp_path)
    library = config.snippet_style.library_source
    # More evidence than one prompt can consume prevents the reuse window from
    # hiding stale evidence when the old one-shot adapter is used.
    deleted = [
        _snippet(index, owner_reply=[f"{DELETED_CANARY}_{index}"])
        for index in range(24)
    ]
    retained = _snippet(30, owner_reply=[RETAINED_CANARY])
    write_snippets(library, [*deleted, retained])
    provider = FakeProvider("合成安全回复")
    monkeypatch.setattr(composition, "OpenAICompatibleProvider", lambda **_: provider)
    source, card = composition.build_snippet_style_runtime(config, clock=lambda: FIXED_NOW)
    runtime = composition.build_real_orchestrator(
        settings,
        config,
        secrets,
        trusted_user_id=PARTNER,
        snippet_style=source,
        snippet_habit_card=card,
    )
    request = DialogueRequest(user_id=PARTNER, text="合成她说", recent_context=[])
    first = runtime.reply(request, trusted_user_id=PARTNER)
    assert first.response.blocked_reason is None
    assert provider.call_count == 1
    assert DELETED_CANARY in provider.system_text
    provider.text = "合成新的安全回答"
    return runtime, provider, request, library, deleted, settings, config


def test_live_composition_removes_deleted_evidence_from_next_prompt(live_composition) -> None:
    runtime, provider, request, library, deleted, _, _ = live_composition

    # The deletion service atomically rewrites the same configured synthetic file.
    assert delete_snippets(
        [library], snippet_ids=frozenset(item.snippet_id for item in deleted)
    ) == len(deleted)
    run = runtime.reply(request, trusted_user_id=PARTNER)

    assert provider.call_count == 2
    outbound = "\n".join(message.content for message in provider.requests[-1])
    assert DELETED_CANARY not in outbound
    assert RETAINED_CANARY in outbound
    assert run.response.blocked_reason is None
    assert DELETED_CANARY not in run.model_dump_json()


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_live_composition_source_failure_blocks_provider_and_reservation(
    live_composition, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    runtime, provider, request, library, _, settings, config = live_composition
    if failure == "missing":
        library.unlink()
    else:
        library.write_text("{" + CORRUPT_CANARY, encoding="utf-8")
    complete = Mock(wraps=provider.complete)
    monkeypatch.setattr(provider, "complete", complete)
    reserve = MonthlyUsageLedger.reserve
    reservations = []

    def record_reservation(self, count, now):
        reservations.append(count)
        return reserve(self, count=count, now=now)

    monkeypatch.setattr(MonthlyUsageLedger, "reserve", record_reservation)
    ledger = MonthlyUsageLedger(settings.runtime_data_dir / "dialogue", config.budget)
    now = datetime.now(UTC)
    before = ledger.snapshot(now)

    run = runtime.reply(request, trusted_user_id=PARTNER)

    complete.assert_not_called()
    assert provider.call_count == 1
    assert reservations == []
    assert ledger.snapshot(now) == before
    assert run.response.blocked_reason == "configuration_unavailable"
    assert run.response.billable is False
    output = run.model_dump_json()
    assert DELETED_CANARY not in output
    assert CORRUPT_CANARY not in output
    assert str(library) not in output
