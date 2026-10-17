"""Live privacy invalidation at the real bridge/provider and debug file boundaries.

All persistent inputs are disposable synthetic files. No device or paid provider
is involved; the disconnected callback cases are deliberate regression controls.
"""

# Chinese punctuation is part of the real bridge's speaker labels.
# ruff: noqa: RUF001

import json
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from companion.contracts import ConsentRecord
from companion.dialogue.context import CONTEXT_END, CONTEXT_START
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.memory import (
    CorrectMemoryRequest,
    CreateMemoryRequest,
    MemoryService,
    MemoryStoreError,
    build_private_memory_service,
)
from companion.runtime.settings import Settings
from companion.runtime.speech_chat_mac import _open_debug_transcript
from companion.runtime.speech_dialogue import SpeechDialogueBridge, SpeechDialogueBridgeError
from companion.speech.controller import SpeechController
from companion.speech.controller_models import CONNECTION_RETRY_PROMPT
from companion.style.snippet_source import FileSnippetStyleSource
from tests.integration.test_memory_lifecycle_restart import _config
from tests.integration.test_speech_dialogue_runtime import (
    CaptureProvider,
    _complete_turn,
    _recognition,
)
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)

NOW = datetime(2026, 10, 16, 8, tzinfo=UTC)
A = "synthetic-live-a"
B = "synthetic-live-b"
QUERY = "合成蓝色"
OLD = "SYNTHETIC_OLD_MEMORY"
NEW = "SYNTHETIC_NEW_MEMORY"
PROFILE = "合成画像旧暗号"
TOPIC = "合成蓝色主题"
SAFE_REPLY = "合成下一轮安全回答"
EXCEPTION = "SYNTHETIC_LOW_LEVEL_EXCEPTION"
TURN_KEYS = {
    "observed_at", "blocked_reason", "input_chars", "context_count", "context_chars",
    "reply_chars", "similar_count", "general_count", "profile_count", "provider_attempted",
    "provider_failure", "provider_elapsed_ms",
}


def settings_for(root: Path) -> Settings:
    private = root / "synthetic-private"
    private.mkdir(parents=True, mode=0o700)
    return Settings(
        environment="test", log_level="INFO", private_data_dir=private,
        runtime_data_dir=root / "synthetic-runtime", test_results_dir=root / "synthetic-results",
    )


def grant(service, user=A, fields=("preference", "explicit_note", "profile")):
    return service.grant_consent(ConsentRecord(
        user_id=user, granted=True, granted_at=NOW, allowed_fields=list(fields),
    ))


def memory(service, user=A, canary=OLD):
    return service.create_memory(CreateMemoryRequest(
        user_id=user, field="preference",
        value=f"喜欢{QUERY}，其中的独立暗号是{canary}，常用于拼图",
        source_timestamp=NOW, explicitly_requested=True,
    ))


def profile(service, user=A):
    item = service.add_profile_candidate(
        user_id=user, topic=TOPIC, text=f"{QUERY}时喜欢{PROFILE}，并且慢慢完成拼图",
        category="preference", source="self_stated", evidence_date=NOW.date(), confidence="high",
    )
    service.approve_profile(user_id=user, profile_id=item.profile_id)
    return service.confirm_profile(
        user_id=user, profile_id=item.profile_id, version=item.version,
        subject_id=user, simulated=True, use_in_chat=True,
    )


def outbound(provider) -> str:
    return "\n".join(message.content for message in provider.requests[-1])


def history(provider) -> str:
    return outbound(provider).split(CONTEXT_START, 1)[1].split(CONTEXT_END, 1)[0].replace(
        "\\n", "\n"
    )


def controller_failure(bridge, timing_observer=None):
    synth = SpeechSynthesizerFake()
    player = AudioPlayerFake()
    controller = SpeechController(
        trusted_user_id=A, recorder=AudioRecorderFake(),
        recognizer=SpeechRecognizerFake(result=_recognition(QUERY)),
        dialogue=bridge, synthesizer=synth, player=player,
        mute_gate=MuteGateFake(), clock=ClockFake(), timing_observer=timing_observer,
    )
    _complete_turn(controller, player)
    assert synth.inputs == [CONNECTION_RETRY_PROMPT]
    controller.shutdown()


class LiveSystem:
    def __init__(self, root, *, connected=True, snippet=None, observer=None):
        self.now = NOW
        self.settings = settings_for(root)
        self.database = self.settings.private_data_dir / "memory" / "memory.sqlite3"
        self.source = build_private_memory_service(self.settings, clock=lambda: self.now)
        # A distinct writer commits to the same database while all readers live.
        self.writer = MemoryService(self.database, clock=lambda: self.now)
        self.provider = CaptureProvider()
        self.observations = []

        def observe(value):
            self.observations.append(value)
            if observer is not None:
                observer(value)

        self.config = _config(root)
        if snippet is None:
            library = root / "synthetic-empty-snippets.jsonl"
            library.write_text("", encoding="utf-8")
            snippet = FileSnippetStyleSource(
                library_source=library, partner_user_id=A, style_version="synthetic-live-v1",
                clock=lambda: self.now,
            )
        self.ledger = MonthlyUsageLedger(root / "synthetic-ledger", self.config.budget)
        self.runtime = DialogueOrchestrator(
            config=self.config,
            snippet_style=snippet, provider=self.provider, ledger=self.ledger,
            memory_gateway=self.source, profile_source=self.source, clock=lambda: self.now,
            privacy_revision_sources=(self.source, snippet),
            turn_observer=observe,
        )
        self.bridge = SpeechDialogueBridge(
            orchestrator=self.runtime,
            privacy_revision=self.runtime.privacy_revision if connected else None,
        )

    def reply(self, text=QUERY, user=A, reply=SAFE_REPLY):
        self.provider.text = reply
        before = self.provider.call_count
        response = self.bridge.reply(user_id=user, text=text)
        assert response.blocked_reason is None
        assert response.billable is True
        assert self.observations[-1].provider_attempted is True
        assert self.provider.call_count == before + 1
        return response

    def seed_echo(self, canary=OLD, user=A):
        response = self.reply(user=user, reply=f"合成回答提到{canary}，稍后继续")
        assert canary in response.text
        assert canary in outbound(self.provider)
        # An unchanged round proves the accepted answer is really in history,
        # independently of the source evidence still being available.
        self.reply(user=user)
        assert f"你：{response.text}" in history(self.provider)
        assert f"对方：{QUERY}" in history(self.provider)


@pytest.mark.parametrize("connected", [False, True], ids=["disconnected-control", "live"])
def test_deleted_memory_echo_does_not_reenter_next_prompt(tmp_path, connected):
    system = LiveSystem(tmp_path, connected=connected)
    grant(system.writer)
    item = memory(system.writer)
    system.seed_echo()
    assert system.writer.delete_memory(A, item.memory_id).deleted_count == 1
    assert system.source.list_memories(A) == ()
    system.reply()
    assert (OLD in outbound(system.provider)) is (not connected)
    assert (OLD in history(system.provider)) is (not connected)


def test_corrected_memory_removes_old_echo_and_uses_new_value(tmp_path):
    system = LiveSystem(tmp_path)
    grant(system.writer)
    item = memory(system.writer)
    system.seed_echo()
    corrected = system.writer.correct_memory(CorrectMemoryRequest(
        user_id=A, memory_id=item.memory_id, value=f"喜欢{QUERY}和{NEW}，继续完成合成拼图",
        source_timestamp=NOW,
    ))
    assert system.source.retrieve_memories(A, QUERY) == (corrected,)
    system.reply()
    assert OLD not in outbound(system.provider)
    assert corrected.value in outbound(system.provider)


@pytest.mark.parametrize("operation", ["revoke", "narrow"])
def test_revocation_and_narrowing_remove_old_echo_from_next_prompt(tmp_path, operation):
    system = LiveSystem(tmp_path)
    grant(system.writer)
    memory(system.writer)
    system.seed_echo()
    if operation == "revoke":
        assert system.writer.revoke_consent(A).deleted_count == 1
    else:
        grant(system.writer, fields=("explicit_note",))
    assert system.source.list_memories(A) == ()
    system.reply()
    assert OLD not in outbound(system.provider)


@pytest.mark.parametrize("operation", ["delete", "disable", "revoke", "expire"])
@pytest.mark.parametrize("connected", [False, True], ids=["disconnected-control", "live"])
def test_profile_change_clears_old_echo_before_next_prompt(tmp_path, operation, connected):
    system = LiveSystem(tmp_path, connected=connected)
    grant(system.writer)
    item = profile(system.writer)
    system.seed_echo(PROFILE)
    target = dict(user_id=A, profile_id=item.profile_id)
    if operation == "delete":
        assert system.writer.delete_profile(**target).deleted_count == 1
    elif operation == "disable":
        assert system.writer.disable_profile(**target, reason="owner").status == "disabled"
    elif operation == "revoke":
        assert system.writer.revoke_consent(A).deleted_count == 1
    else:
        system.now = datetime.combine(item.valid_until, datetime.min.time(), UTC)
        # Commit expiry cleanup through a different service before the next round.
        system.writer.content_revision(A)
    assert system.source.retrieve_profile(user_id=A, query_texts=[QUERY]) == ()
    system.reply()
    assert (PROFILE in outbound(system.provider)) is (not connected)
    assert (PROFILE in history(system.provider)) is (not connected)
    assert TOPIC not in outbound(system.provider)


def break_database(system, failure):
    if failure == "permission":
        system.database.chmod(0o644)
    elif failure == "missing":
        system.database.unlink()
    else:
        system.database.write_text(EXCEPTION, encoding="utf-8")


@pytest.mark.parametrize("failure", ["permission", "missing", "corrupt"])
def test_revision_read_failure_prevents_provider_and_reservation(tmp_path, monkeypatch, failure):
    system = LiveSystem(tmp_path / EXCEPTION)
    grant(system.writer)
    memory(system.writer)
    system.seed_echo()
    break_database(system, failure)
    # The actual file boundary must fail, rather than a substituted revision callback.
    with pytest.raises(MemoryStoreError, match=r"^memory store unavailable$"):
        system.source.content_revision(A)
    reserve = Mock(wraps=system.ledger.reserve)
    monkeypatch.setattr(system.ledger, "reserve", reserve)
    before = system.provider.call_count
    with pytest.raises(
        SpeechDialogueBridgeError, match=r"^speech-dialogue-bridge-failed$"
    ) as error:
        system.bridge.reply(user_id=A, text=QUERY)
    assert error.value.__cause__ is None
    controller_failure(system.bridge)
    assert system.provider.call_count == before
    reserve.assert_not_called()


@pytest.mark.parametrize("failure", ["permission", "missing", "corrupt"])
def test_source_recovery_never_restores_old_context(tmp_path, failure):
    system = LiveSystem(tmp_path)
    grant(system.writer)
    item = memory(system.writer)
    system.seed_echo()
    backup = tmp_path / "synthetic-database-backup"
    original = system.database.read_bytes()
    if failure == "missing":
        # Restore the same inode after the outage: the private factory pins identity.
        system.database.rename(backup)
    else:
        break_database(system, failure)
    with pytest.raises(SpeechDialogueBridgeError):
        system.bridge.reply(user_id=A, text=QUERY)
    if failure == "missing":
        backup.rename(system.database)
    elif failure == "corrupt":
        system.database.write_bytes(original)
    else:
        system.database.chmod(0o600)
    corrected = system.writer.correct_memory(CorrectMemoryRequest(
        user_id=A, memory_id=item.memory_id, value=f"喜欢{QUERY}和{NEW}", source_timestamp=NOW,
    ))
    assert system.source.retrieve_memories(A, QUERY) == (corrected,)
    system.reply()
    assert OLD not in outbound(system.provider)
    assert NEW in outbound(system.provider)
    assert history(system.provider).strip() == ""


@pytest.mark.parametrize("operation", ["missing-delete", "repeat-revoke", "same-correction"])
def test_repeated_noop_changes_preserve_current_history(tmp_path, operation):
    system = LiveSystem(tmp_path)
    grant(system.writer)
    item = memory(system.writer)
    if operation == "repeat-revoke":
        system.writer.revoke_consent(A)
    elif operation == "missing-delete":
        system.writer.delete_memory(A, item.memory_id)
    # Seed after the initial real mutation: the following operations are no-ops.
    first = system.reply(reply="合成保留的双边回答")
    before = system.source.content_revision(A)
    for _ in range(2):
        if operation == "repeat-revoke":
            assert system.writer.revoke_consent(A).deleted_count == 0
        elif operation == "missing-delete":
            assert system.writer.delete_memory(A, item.memory_id).deleted_count == 0
        else:
            assert system.writer.correct_memory(CorrectMemoryRequest(
                user_id=A, memory_id=item.memory_id, value=item.value,
                source_timestamp=item.source_timestamp,
            )) == item
        assert system.source.content_revision(A) == before
        system.reply()
        assert f"你：{first.text}" in history(system.provider)
        assert f"对方：{QUERY}" in history(system.provider)


def test_other_user_change_preserves_unaffected_user_history(tmp_path):
    system = LiveSystem(tmp_path)
    for user in (A, B):
        grant(system.writer, user)
    item = memory(system.writer)
    memory(system.writer, B, "SYNTHETIC_OTHER_MEMORY")
    system.seed_echo()
    other = system.reply(user=B, reply="合成乙的双边回答")
    before_b = system.source.content_revision(B)
    system.writer.delete_memory(A, item.memory_id)
    system.reply()
    assert OLD not in outbound(system.provider)
    assert history(system.provider).strip() == ""
    assert system.source.content_revision(B) == before_b
    system.reply(user=B)
    assert f"你：{other.text}" in history(system.provider)
    assert f"对方：{QUERY}" in history(system.provider)
    assert OLD not in outbound(system.provider)


@pytest.mark.parametrize("failure", [None, "permission", "missing", "corrupt"])
def test_live_deletion_debug_contains_diagnostics_only(tmp_path, capsys, monkeypatch, failure):
    settings = settings_for(tmp_path / "debug")
    with ExitStack() as stack:
        path, observer, timing_observer = _open_debug_transcript(settings, stack)
        system = LiveSystem(tmp_path / EXCEPTION, observer=observer)
        grant(system.writer)
        item = memory(system.writer)
        profile_item = profile(system.writer)
        system.seed_echo()
        system.reply(reply=f"合成画像回应{PROFILE}")
        system.reply()
        assert PROFILE in history(system.provider)
        assert system.writer.delete_memory(A, item.memory_id).deleted_count == 1
        assert system.writer.delete_profile(user_id=A, profile_id=profile_item.profile_id)
        system.reply()
        assert OLD not in outbound(system.provider)
        assert PROFILE not in outbound(system.provider)
        expected_turns = len(system.observations)
        if failure is not None:
            break_database(system, failure)
            before = system.provider.call_count
            reserve = Mock(wraps=system.ledger.reserve)
            monkeypatch.setattr(system.ledger, "reserve", reserve)
            controller_failure(system.bridge, timing_observer)
            assert system.provider.call_count == before
            reserve.assert_not_called()
        raw = path.read_text(encoding="utf-8")  # Also proves per-record flush.
        rows = [json.loads(line) for line in raw.splitlines()]
        turns = [row for row in rows if row.get("kind") != "speech_timing"]
        assert len(turns) == expected_turns
        assert all(set(row) == TURN_KEYS for row in turns)
        if failure is not None:
            assert rows[-1]["kind"] == "speech_timing"
            assert rows[-1]["issue"] == "dialogue-failed"
        captured = capsys.readouterr()
        for canary in (QUERY, OLD, PROFILE, TOPIC, SAFE_REPLY, EXCEPTION, "合成画像回应"):
            assert canary not in raw + captured.out + captured.err
