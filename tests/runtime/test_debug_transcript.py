"""Actual debug JSONL writes using synthetic data and a local fake provider."""

import json
import stat
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest

from companion.contracts import ConsentRecord, DialogueRequest, StyleEvidence, StyleSnippet
from companion.dialogue.observer import TurnObservation
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.memory import MemoryField, MemoryService
from companion.memory.profile import ProfileView
from companion.runtime import speech_chat_mac
from companion.runtime.settings import Settings
from companion.runtime.speech_chat_mac import _open_debug_transcript
from tests.dialogue.dialogue_support import FIXED_NOW, FakeProvider, synthetic_dialogue_config

TURN_KEYS = {
    "observed_at", "blocked_reason", "input_chars", "context_count", "context_chars",
    "reply_chars", "similar_count", "general_count", "profile_count", "provider_attempted",
    "provider_failure", "provider_elapsed_ms",
}
INPUT = "SYNTHETIC_INPUT_CANARY"
CONTEXT = "SYNTHETIC_CONTEXT_CANARY"
REPLY = "SYNTHETIC_REPLY_CANARY"
SIMILAR_PARTNER = "SYNTHETIC_SIMILAR_PARTNER_CANARY"
SIMILAR_OWNER = "SYNTHETIC_SIMILAR_OWNER_CANARY"
GENERAL_PARTNER = "SYNTHETIC_GENERAL_PARTNER_CANARY"
GENERAL_OWNER = "SYNTHETIC_GENERAL_OWNER_CANARY"
TOPIC = "合成主题暗号"
PROFILE = "SYNTHETIC_PROFILE_TEXT_CANARY"
CONTENT = (
    INPUT, CONTEXT, REPLY, SIMILAR_PARTNER, SIMILAR_OWNER,
    GENERAL_PARTNER, GENERAL_OWNER, TOPIC, PROFILE,
)


class SyntheticSnippetStyle:
    def __init__(self) -> None:
        self.calls = 0

    def accepts_user(self, user_id: str) -> bool:
        return user_id == "synthetic-user"

    def evidence_for(
        self, user_id: str, text: str, recent_context: Sequence[str]
    ) -> StyleEvidence:
        self.calls += 1
        return StyleEvidence(
            style_version="synthetic-debug-v1", addressee_register="intimate",
            similar=[StyleSnippet(partner_turns=[SIMILAR_PARTNER], owner_reply=[SIMILAR_OWNER])],
            general=[StyleSnippet(partner_turns=[GENERAL_PARTNER], owner_reply=[GENERAL_OWNER])],
        )


@pytest.fixture
def debug_settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test", log_level="INFO", private_data_dir=tmp_path / "synthetic-private",
        runtime_data_dir=tmp_path / "synthetic-runtime",
        test_results_dir=tmp_path / "synthetic-results",
    )


@pytest.fixture
def debug(debug_settings: Settings):
    with ExitStack() as resources:
        path, observer, _ = _open_debug_transcript(debug_settings, resources)
        yield path, observer
    with pytest.raises(ValueError, match="closed file"):
        observer(observation())


def test_same_second_debug_runs_use_distinct_files(debug_settings, monkeypatch):
    clock = Mock(wraps=datetime)
    clock.now.return_value = FIXED_NOW
    monkeypatch.setattr(speech_chat_mac, "datetime", clock)

    with ExitStack() as resources:
        first, _, _ = _open_debug_transcript(debug_settings, resources)
        second, _, _ = _open_debug_transcript(debug_settings, resources)
        assert first != second
        for path in (first, second):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert set(first.parent.iterdir()) == {first, second}


def test_debug_never_follows_existing_symlink(debug_settings, tmp_path, monkeypatch):
    fixed_uuid = UUID("a" * 32)
    monkeypatch.setattr(speech_chat_mac, "uuid4", lambda: fixed_uuid)
    directory = debug_settings.private_data_dir / "speech" / "run" / "debug-transcripts"
    directory.mkdir(parents=True, mode=0o700)
    target = tmp_path / "synthetic-target"
    original = b"synthetic target must remain unchanged\n"
    target.write_bytes(original)
    link = directory / f"transcript-{fixed_uuid.hex}.jsonl"
    link.symlink_to(target)

    with ExitStack() as resources, pytest.raises(OSError):
        _open_debug_transcript(debug_settings, resources)

    assert link.is_symlink()
    assert target.read_bytes() == original


def test_debug_rejects_existing_target_without_changing_it(debug_settings, monkeypatch):
    fixed_uuid = UUID("b" * 32)
    monkeypatch.setattr(speech_chat_mac, "uuid4", lambda: fixed_uuid)
    directory = debug_settings.private_data_dir / "speech" / "run" / "debug-transcripts"
    directory.mkdir(parents=True, mode=0o700)
    target = directory / f"transcript-{fixed_uuid.hex}.jsonl"
    original = b"synthetic existing diagnostic file\n"
    target.write_bytes(original)
    target.chmod(0o640)

    with ExitStack() as resources, pytest.raises(FileExistsError):
        _open_debug_transcript(debug_settings, resources)

    assert target.read_bytes() == original
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_debug_preserves_existing_directory_permissions(debug_settings):
    directory = debug_settings.private_data_dir / "speech" / "run" / "debug-transcripts"
    directory.mkdir(parents=True, mode=0o700)
    directory.chmod(0o750)

    with ExitStack() as resources:
        path, _, _ = _open_debug_transcript(debug_settings, resources)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(directory.stat().st_mode) == 0o750


def observation() -> TurnObservation:
    return TurnObservation(
        blocked_reason="synthetic-block", input_chars=23, context_count=1, context_chars=25,
        reply_chars=23, similar_count=1, general_count=1, profile_count=1,
        provider_attempted=True, provider_failure="synthetic-failure", provider_elapsed_ms=17,
        system_prompt_sha256="a" * 64, observed_at=FIXED_NOW,
    )


def request(text: str = INPUT) -> DialogueRequest:
    return DialogueRequest(user_id="synthetic-user", text=text, recent_context=[CONTEXT])


def build(root, observer, *, memory=None, clock=lambda: FIXED_NOW):
    config = synthetic_dialogue_config(root)
    provider = FakeProvider(REPLY)
    style = SyntheticSnippetStyle()
    profile = Mock()
    profile.retrieve_profile.return_value = (
        ProfileView(topic=TOPIC, text=PROFILE, category="habit"),
    )
    ledger = MonthlyUsageLedger(root / "synthetic-ledger", config.budget)
    runtime = DialogueOrchestrator(
        config=config, snippet_style=style, profile_source=profile, provider=provider,
        ledger=ledger, clock=clock, memory_gateway=memory, turn_observer=observer,
    )
    return runtime, provider, style, profile, ledger


def read_diagnostics(path, capsys, *, count, forbidden=()):
    # Read while ExitStack is still open: every callback must flush its line.
    raw = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == count
    for row in rows:
        assert set(row) == TURN_KEYS
        assert datetime.fromisoformat(row["observed_at"]).tzinfo is not None
    captured = capsys.readouterr()
    for value in (*CONTENT, *forbidden):
        for line in value.splitlines():
            assert line not in raw
            assert line not in captured.out
            assert line not in captured.err
    return rows


def test_memory_success_and_denial_leave_no_debug_content(tmp_path, debug, capsys):
    path, observer = debug
    for index, authorized in enumerate((True, False), start=1):
        root = tmp_path / str(index)
        service = MemoryService(root / "synthetic-memory.sqlite3", clock=lambda: FIXED_NOW)
        if authorized:
            service.grant_consent(ConsentRecord(
                user_id="synthetic-user", granted=True, granted_at=FIXED_NOW,
                allowed_fields=[MemoryField.PREFERENCE.value],
            ))
        value = f"喜欢SYNTHETIC_MEMORY_{index}_CANARY"
        command = f"记住我{value}"
        runtime, provider, _, _, _ = build(root, observer, memory=service)
        run = runtime.reply(request(command), trusted_user_id="synthetic-user")
        assert run.response.blocked_reason == (None if authorized else "memory_write_failed")
        assert provider.call_count == 0
        entries = service.list_memories("synthetic-user")
        assert [entry.value for entry in entries] == ([value] if authorized else [])
        rows = read_diagnostics(
            path, capsys, count=index, forbidden=(command, value, run.response.text),
        )
        assert rows[-1]["blocked_reason"] == run.response.blocked_reason
        assert rows[-1]["provider_attempted"] is False
        assert rows[-1]["reply_chars"] == len(run.response.text)


def test_disabled_memory_command_leaves_no_debug_content(tmp_path, debug, capsys):
    path, observer = debug
    runtime, provider, _, _, _ = build(tmp_path, observer)
    value = "喜欢SYNTHETIC_DISABLED_MEMORY_CANARY"
    command = f"记住我{value}"
    run = runtime.reply(request(command), trusted_user_id="synthetic-user")
    assert run.response.blocked_reason == "memory_disabled"
    assert provider.call_count == 0
    row, = read_diagnostics(path, capsys, count=1, forbidden=(command, value, run.response.text))
    assert row["blocked_reason"] == "memory_disabled"
    assert row["provider_attempted"] is False


def test_local_blocks_leave_no_debug_content(tmp_path, debug, capsys):
    path, observer = debug
    for index, reason in enumerate((
        "invalid_request", "untrusted_user", "configuration_unavailable",
    ), start=1):
        clock = (lambda: datetime(2026, 10, 12)) if index == 3 else lambda: FIXED_NOW
        runtime, provider, _, _, _ = build(tmp_path / str(index), observer, clock=clock)
        current = request().model_copy(update={"text": ""}) if index == 1 else request()
        trusted = "synthetic-other" if index == 2 else "synthetic-user"
        run = runtime.reply(current, trusted_user_id=trusted)
        assert run.response.blocked_reason == reason
        assert provider.call_count == 0
        rows = read_diagnostics(path, capsys, count=index, forbidden=(run.response.text,))
        assert rows[-1]["blocked_reason"] == reason
        assert rows[-1]["provider_attempted"] is False
        assert rows[-1]["similar_count"] == rows[-1]["profile_count"] == 0


def test_budget_block_leaves_no_debug_content(tmp_path, debug, capsys):
    path, observer = debug
    runtime, provider, style, profile, ledger = build(tmp_path, observer)
    ledger.reserve(count=100, now=FIXED_NOW)
    run = runtime.reply(request(), trusted_user_id="synthetic-user")
    assert run.response.blocked_reason == "budget_exceeded"
    assert provider.call_count == 0
    assert style.calls == 1
    profile.retrieve_profile.assert_called_once()
    row, = read_diagnostics(path, capsys, count=1, forbidden=(run.response.text,))
    assert row["blocked_reason"] == "budget_exceeded"
    assert row["provider_attempted"] is False
    assert row["similar_count"] == row["general_count"] == row["profile_count"] == 1


def test_normal_reply_and_evidence_leave_no_debug_content(tmp_path, debug, capsys):
    path, observer = debug
    runtime, provider, _, _, _ = build(tmp_path, observer)
    run = runtime.reply(request(), trusted_user_id="synthetic-user")
    assert run.response.blocked_reason is None
    assert REPLY in run.response.text
    assert provider.call_count == 1
    prompt = "\n".join(message.content for message in provider.requests[0])
    for value in CONTENT:
        if value != REPLY:
            assert value in prompt
    row, = read_diagnostics(path, capsys, count=1, forbidden=(run.response.text,))
    assert row["blocked_reason"] is None
    assert row["provider_attempted"] is True
    assert row["input_chars"] == len(INPUT)
    assert row["context_count"] == 1
    assert row["context_chars"] == len(CONTEXT)
    assert row["reply_chars"] == len(run.response.text)
    assert row["similar_count"] == row["general_count"] == row["profile_count"] == 1
    assert row["provider_failure"] is None
    assert isinstance(row["provider_elapsed_ms"], int)


def test_duplicate_callbacks_write_diagnostics_only(debug, capsys):
    path, observer = debug
    observed = observation()
    for count in (1, 2):
        observer(observed)
        rows = read_diagnostics(path, capsys, count=count)
        assert rows[-1] == {
            "observed_at": FIXED_NOW.isoformat(), "blocked_reason": "synthetic-block",
            "input_chars": 23, "context_count": 1, "context_chars": 25, "reply_chars": 23,
            "similar_count": 1, "general_count": 1, "profile_count": 1,
            "provider_attempted": True, "provider_failure": "synthetic-failure",
            "provider_elapsed_ms": 17,
        }
    assert rows[0] == rows[1]
