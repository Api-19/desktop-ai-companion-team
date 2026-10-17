"""Synthetic live snippet invalidation, including the formal Mac/Body wiring."""

# ruff: noqa: RUF001

import hashlib
import json
import os
import sys
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

import companion.admin.snippets as admin_snippets
import companion.dialogue.composition as composition
from companion.admin.snippets import SnippetStore
from companion.dialogue.models import DialogueSecrets, SnippetStyleSources
from companion.memory import MemoryService, build_private_memory_service
from companion.privacy.paired_snippets import PairedSnippet, delete_snippets, write_snippets
from companion.runtime import budget_notice, speech_chat_body, speech_chat_mac
from companion.runtime.speech_dialogue import SpeechDialogueBridge, SpeechDialogueBridgeError
from companion.speech import runtime_assembly
from companion.style.snippet_source import FileSnippetStyleSource
from tests.integration.test_live_memory_invalidation import (
    NOW,
    PROFILE,
    QUERY,
    SAFE_REPLY,
    TOPIC,
    TURN_KEYS,
    A,
    B,
    LiveSystem,
    controller_failure,
    grant,
    history,
    outbound,
    profile,
    settings_for,
)
from tests.integration.test_memory_lifecycle_restart import _config
from tests.integration.test_speech_dialogue_runtime import CaptureProvider

OLD_SNIPPET = "SYNTHETIC_OLD_SNIPPET"
NEW_SNIPPET = "SYNTHETIC_NEW_SNIPPET"
RETAINED = "SYNTHETIC_RETAINED_SNIPPET"
PARTNER_TEXT = "合成片段对方正文暗号"


def pair(index=0, canary=OLD_SNIPPET):
    return PairedSnippet(
        snippet_id=f"synthetic-live-{index}", conversation_id="synthetic-live-conversation",
        started_at=NOW + timedelta(minutes=index), source_fingerprint="synthetic-fingerprint",
        partner_turns=[f"{QUERY}，{PARTNER_TEXT}{index}，继续看合成拼图"],
        owner_reply=[f"先把合成拼图放好，独立暗号{canary}，我们稍后再慢慢看{index}"],
    )


def snippet_system(root, *, connected=True, observer=None):
    root.mkdir(parents=True, exist_ok=True)
    library = root / "synthetic-library.jsonl"
    removed, retained = pair(), pair(1, RETAINED)
    write_snippets(library, [removed, retained])
    source = FileSnippetStyleSource(
        library_source=library, partner_user_id=A, style_version="synthetic-live-v1",
        clock=lambda: NOW,
    )
    system = LiveSystem(root, connected=connected, snippet=source, observer=observer)
    return system, source, library, removed, retained


@pytest.mark.parametrize("method", ["id", "time-range"])
@pytest.mark.parametrize("connected", [False, True], ids=["disconnected-control", "live"])
def test_deleted_snippet_echo_does_not_reenter_next_prompt(tmp_path, method, connected):
    system, source, library, removed, _ = snippet_system(tmp_path, connected=connected)
    system.seed_echo(OLD_SNIPPET)
    if method == "id":
        count = delete_snippets([library], snippet_ids=frozenset([removed.snippet_id]))
    else:
        count = delete_snippets([library], started_between=(NOW, NOW))
    assert count == 1
    system.reply()
    assert RETAINED in outbound(system.provider)
    assert (OLD_SNIPPET in outbound(system.provider)) is (not connected)
    assert (OLD_SNIPPET in history(system.provider)) is (not connected)
    evidence = source.evidence_for(A, QUERY, [])
    assert OLD_SNIPPET not in evidence.model_dump_json()
    assert removed.to_style_snippet() not in evidence.similar + evidence.general


@pytest.mark.parametrize("method", ["id", "time-range"])
def test_partial_admin_delete_excludes_next_prompt_and_old_echo(
    tmp_path, monkeypatch, capsys, caplog, method,
):
    root = tmp_path / "synthetic-admin"
    root.mkdir(mode=0o700)
    store = SnippetStore(root)
    removed, retained = pair(), pair(1, RETAINED)
    for path in store.snippet_paths:
        write_snippets(path, [removed, retained])
        path.chmod(0o600)
    original = [path.read_bytes() for path in store.snippet_paths]
    source = FileSnippetStyleSource(
        library_source=store.library_path, partner_user_id=A,
        style_version="synthetic-live-v1", clock=lambda: NOW,
    )
    revision = source.content_revision(A)
    ids = frozenset([removed.snippet_id]) if method == "id" else frozenset()
    interval = (NOW, NOW) if method == "time-range" else None
    rewrite = Mock(side_effect=OSError("SYNTHETIC_ADMIN_REWRITE_EXCEPTION"))
    monkeypatch.setattr(admin_snippets, "delete_snippets", rewrite)

    with ExitStack() as resources:
        debug, observer, _ = speech_chat_mac._open_debug_transcript(
            settings_for(tmp_path / "debug"), resources,
        )
        system = LiveSystem(tmp_path / "live", snippet=source, observer=observer)
        system.seed_echo(OLD_SNIPPET)
        with pytest.raises(OSError, match="SYNTHETIC_ADMIN_REWRITE_EXCEPTION"):
            store.delete(ids, interval)
        rewrite.assert_called_once_with(
            store.snippet_paths, snippet_ids=ids, started_between=interval,
        )
        assert store.exclusions() == ({removed.snippet_id}, [interval] if interval else [])
        assert [path.read_bytes() for path in store.snippet_paths] == original

        system.reply()
        assert source.content_revision(A) != revision
        assert OLD_SNIPPET not in outbound(system.provider)
        assert RETAINED in outbound(system.provider)
        assert history(system.provider).strip() == ""
        raw = debug.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == 3
        assert all(set(row) == TURN_KEYS and row["provider_attempted"] for row in rows)
        captured = capsys.readouterr()
        for canary in (
            QUERY, OLD_SNIPPET, RETAINED, PARTNER_TEXT, SAFE_REPLY, "合成回答提到",
            "SYNTHETIC_ADMIN_REWRITE_EXCEPTION",
        ):
            assert canary not in raw + captured.out + captured.err + caplog.text


def test_corrupt_exclusions_prevent_provider_and_reservation(
    tmp_path, monkeypatch, capsys, caplog,
):
    with ExitStack() as resources:
        debug, observer, timing = speech_chat_mac._open_debug_transcript(
            settings_for(tmp_path / "debug"), resources,
        )
        system, _, library, _, _ = snippet_system(tmp_path / "live", observer=observer)
        exclusions = library.parent / "deleted_ids.json"
        exclusions.write_text("{SYNTHETIC_EXCLUSIONS_PARSE_EXCEPTION", encoding="utf-8")
        reserve = Mock(wraps=system.ledger.reserve)
        monkeypatch.setattr(system.ledger, "reserve", reserve)

        with pytest.raises(SpeechDialogueBridgeError, match=r"^speech-dialogue-bridge-failed$"):
            system.bridge.reply(user_id=A, text=QUERY)
        controller_failure(system.bridge, timing)
        assert system.provider.call_count == 0
        reserve.assert_not_called()
        raw = debug.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == 1
        assert rows[0]["kind"] == "speech_timing"
        assert rows[0]["issue"] == "dialogue-failed"
        captured = capsys.readouterr()
        for canary in (
            QUERY, OLD_SNIPPET, RETAINED, PARTNER_TEXT, SAFE_REPLY,
            "SYNTHETIC_EXCLUSIONS_PARSE_EXCEPTION", str(exclusions),
        ):
            assert canary not in raw + captured.out + captured.err + caplog.text


def test_snippet_content_version_uses_bytes_instead_of_file_metadata(tmp_path):
    system, source, library, _, _ = snippet_system(tmp_path)
    system.seed_echo(OLD_SNIPPET)
    before = source.content_revision(A)
    raw = library.read_bytes()
    original = library.stat()
    replacement = tmp_path / "synthetic-replacement.jsonl"
    replacement.write_bytes(raw)
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(library)
    assert library.stat().st_ino != original.st_ino
    assert source.content_revision(A) == before == hashlib.sha256(raw).hexdigest()
    system.reply()
    assert OLD_SNIPPET in history(system.provider)

    # Equal-size bytes with exactly the old timestamps must still invalidate.
    unchanged_inode = library.stat().st_ino
    library.write_bytes(raw.replace(OLD_SNIPPET.encode(), NEW_SNIPPET.encode()))
    os.utime(library, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert library.stat().st_ino == unchanged_inode
    assert library.stat().st_size == original.st_size
    assert library.stat().st_mtime_ns == original.st_mtime_ns
    assert source.content_revision(A) != before
    system.reply()
    assert OLD_SNIPPET not in outbound(system.provider)
    assert NEW_SNIPPET in outbound(system.provider)
    assert history(system.provider).strip() == ""


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_revision_read_failure_prevents_provider_and_reservation(tmp_path, monkeypatch, failure):
    system, _, library, _, _ = snippet_system(tmp_path)
    system.seed_echo(OLD_SNIPPET)
    if failure == "missing":
        library.unlink()
    else:
        library.write_text("{SYNTHETIC_SNIPPET_PARSE_EXCEPTION", encoding="utf-8")
    reserve = Mock(wraps=system.ledger.reserve)
    monkeypatch.setattr(system.ledger, "reserve", reserve)
    before = system.provider.call_count
    with pytest.raises(SpeechDialogueBridgeError, match=r"^speech-dialogue-bridge-failed$"):
        system.bridge.reply(user_id=A, text=QUERY)
    controller_failure(system.bridge)
    assert system.provider.call_count == before
    reserve.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_source_recovery_never_restores_old_context(tmp_path, failure):
    system, _, library, _, retained = snippet_system(tmp_path)
    system.seed_echo(OLD_SNIPPET)
    if failure == "missing":
        library.unlink()
    else:
        library.write_text("{SYNTHETIC_SNIPPET_PARSE_EXCEPTION", encoding="utf-8")
    with pytest.raises(SpeechDialogueBridgeError):
        system.bridge.reply(user_id=A, text=QUERY)
    write_snippets(library, [pair(canary=NEW_SNIPPET), retained])
    system.reply()
    assert OLD_SNIPPET not in outbound(system.provider)
    assert NEW_SNIPPET in outbound(system.provider)
    assert history(system.provider).strip() == ""


def test_repeated_noop_changes_preserve_current_history(tmp_path):
    system, source, library, _, _ = snippet_system(tmp_path)
    system.seed_echo(OLD_SNIPPET)
    raw = library.read_bytes()
    revision = source.content_revision(A)
    for _ in range(2):
        replacement = tmp_path / "synthetic-same-bytes.jsonl"
        replacement.write_bytes(raw)
        replacement.replace(library)
        assert source.content_revision(A) == revision
        system.reply()
        assert OLD_SNIPPET in history(system.provider)
        assert f"对方：{QUERY}" in history(system.provider)


def test_other_user_change_preserves_unaffected_user_history(tmp_path):
    system, source, library, removed, _ = snippet_system(tmp_path)
    assert source.content_revision(B) == ""
    system.seed_echo(OLD_SNIPPET)
    other = system.reply(user=B, reply="合成非伴侣用户的回答")
    assert OLD_SNIPPET not in outbound(system.provider)
    assert delete_snippets([library], snippet_ids=frozenset([removed.snippet_id])) == 1
    assert source.content_revision(B) == ""
    system.reply()
    assert OLD_SNIPPET not in outbound(system.provider)
    assert history(system.provider).strip() == ""
    system.reply(user=B)
    assert f"你：{other.text}" in history(system.provider)
    assert f"对方：{QUERY}" in history(system.provider)


@pytest.mark.parametrize("failure", [None, "missing", "corrupt"])
def test_live_deletion_debug_contains_diagnostics_only(tmp_path, capsys, monkeypatch, failure):
    settings = settings_for(tmp_path / "debug")
    with ExitStack() as resources:
        path, observer, timing = speech_chat_mac._open_debug_transcript(settings, resources)
        system, _, library, removed, _ = snippet_system(tmp_path / "live", observer=observer)
        system.seed_echo(OLD_SNIPPET)
        assert delete_snippets([library], snippet_ids=frozenset([removed.snippet_id])) == 1
        system.reply()
        assert OLD_SNIPPET not in outbound(system.provider)
        if failure is not None:
            if failure == "missing":
                library.unlink()
            else:
                library.write_text("{SYNTHETIC_SNIPPET_PARSE_EXCEPTION", encoding="utf-8")
            before = system.provider.call_count
            reserve = Mock(wraps=system.ledger.reserve)
            monkeypatch.setattr(system.ledger, "reserve", reserve)
            controller_failure(system.bridge, timing)
            assert system.provider.call_count == before
            reserve.assert_not_called()
        raw = path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == (3 if failure is None else 4)
        assert all(set(row) == TURN_KEYS for row in rows[:3])
        assert rows[0]["similar_count"] + rows[0]["general_count"] > 0
        assert all(row["provider_attempted"] for row in rows[:3])
        if failure is not None:
            assert rows[-1]["kind"] == "speech_timing"
            assert rows[-1]["issue"] == "dialogue-failed"
        captured = capsys.readouterr()
        for canary in (
            QUERY, OLD_SNIPPET, RETAINED, PARTNER_TEXT, SAFE_REPLY, "合成回答提到",
            "SYNTHETIC_SNIPPET_PARSE_EXCEPTION",
        ):
            assert canary not in raw + captured.out + captured.err


@pytest.mark.parametrize("entry", ["local", "body"])
def test_formal_mac_and_body_bind_live_revision_sources(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(budget_notice.subprocess, "run", Mock())
    settings = settings_for(tmp_path)
    user = A if entry == "local" else "posix-uid-909"
    library = tmp_path / "synthetic-formal-library.jsonl"
    write_snippets(library, [pair(), pair(1, RETAINED)])
    config = _config(tmp_path).model_copy(update={
        "snippet_style": SnippetStyleSources(library_source=library, partner_user_id=user),
    })
    secrets = DialogueSecrets(api_key=SecretStr("synthetic-unused-key"))
    provider = CaptureProvider()
    monkeypatch.setattr(composition, "OpenAICompatibleProvider", lambda **_: provider)
    monkeypatch.setattr(speech_chat_mac, "load_dialogue_config", lambda _: (config, secrets))
    # Prepare the actual private database before the formal builder opens it.
    reader = build_private_memory_service(settings, clock=lambda: NOW)
    writer = MemoryService(
        settings.private_data_dir / "memory" / "memory.sqlite3", clock=lambda: NOW,
    )
    grant(writer, user)
    item = profile(writer, user)
    observations = []
    exercised = []

    def exercise(bridge):
        assert type(bridge) is SpeechDialogueBridge
        provider.text = f"合成回显{OLD_SNIPPET}以及{PROFILE}，稍后继续"
        first = bridge.reply(user_id=user, text=QUERY)
        assert first.blocked_reason is None and first.billable
        assert observations[-1].provider_attempted
        assert OLD_SNIPPET in first.text and PROFILE in first.text
        assert TOPIC in outbound(provider)
        provider.text = SAFE_REPLY
        bridge.reply(user_id=user, text=QUERY)
        assert OLD_SNIPPET in history(provider) and PROFILE in history(provider)
        # A profile-only change proves the memory source is bound separately.
        assert writer.delete_profile(user_id=user, profile_id=item.profile_id).deleted_count == 1
        bridge.reply(user_id=user, text=QUERY)
        assert PROFILE not in outbound(provider)
        assert history(provider).strip() == ""
        provider.text = f"合成再次回显{OLD_SNIPPET}，稍后继续"
        bridge.reply(user_id=user, text=QUERY)
        provider.text = SAFE_REPLY
        bridge.reply(user_id=user, text=QUERY)
        assert OLD_SNIPPET in history(provider)
        assert delete_snippets([library], snippet_ids=frozenset([pair().snippet_id])) == 1
        bridge.reply(user_id=user, text=QUERY)
        assert OLD_SNIPPET not in outbound(provider)
        assert RETAINED in outbound(provider)
        assert history(provider).strip() == ""
        exercised.append(bridge)

    if entry == "local":
        runtime, notifier = speech_chat_mac._build_orchestrator(settings, user, observations.append)
        assert runtime.privacy_revision(user) == (
            reader.content_revision(user), hashlib.sha256(library.read_bytes()).hexdigest(),
        )
        bridge = runtime_assembly._construct_bridge(runtime, budget_notice=notifier.check)
        exercise(bridge)
        bridge.close()
    else:
        # Body main still invokes the real Mac builder and creates its own bridge.
        # Only terminal, certificate, hardware/connection and loop factories are stubs.
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr(speech_chat_body, "_cbreak_console", nullcontext)
        monkeypatch.setattr(speech_chat_body, "load_settings", lambda _: settings)
        monkeypatch.setattr(
            speech_chat_body, "speech_run_root", lambda _: tmp_path / "synthetic-run"
        )
        monkeypatch.setattr(speech_chat_body.os, "getuid", lambda: 909)
        monkeypatch.setattr(speech_chat_body, "brain_ssl_context", lambda _: object())
        link = SimpleNamespace(
            recorder=object(), player=SimpleNamespace(start_for_session=Mock()),
            mute_gate=object(), close=Mock(), start=Mock(), port=5000,
        )
        monkeypatch.setattr(speech_chat_body, "BrainBodyLink", lambda **_: link)
        monkeypatch.setattr(speech_chat_body, "SpeechExpressionDriver", lambda *_: SimpleNamespace(
            processing=Mock(),
        ))
        assembly = SimpleNamespace(close=Mock())

        def assemble(**kwargs):
            exercise(kwargs["dialogue"])
            return assembly

        monkeypatch.setattr(speech_chat_body, "assemble_body_speech_runtime", assemble)
        monkeypatch.setattr(speech_chat_body, "_run_body_loop", lambda **_: None)
        # Supply the real observer through the normal debug-enabled main branch.
        opener = speech_chat_mac._open_debug_transcript

        def open_debug(settings, resources):
            path, observe, timing = opener(settings, resources)

            def record(value):
                observations.append(value)
                observe(value)

            return path, record, timing

        monkeypatch.setattr(speech_chat_mac, "_open_debug_transcript", open_debug)
        assert speech_chat_body.main(["--debug-log"]) == 0
        link.start.assert_called_once()
        assembly.close.assert_called_once()
        exercised[0].close()
    assert len(exercised) == 1
    assert provider.call_count == 6
