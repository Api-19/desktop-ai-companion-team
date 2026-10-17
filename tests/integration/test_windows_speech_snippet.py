"""Windows platform-entry software evidence on macOS, using synthetic audio/provider.

Exercises one live main/console/assembly per case; this is not real Windows,
Windows Memory, real provider, or persistent identity authorization evidence.
"""

# ruff: noqa: RUF001

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
from pydantic import SecretStr

import companion.dialogue.composition as composition
from companion.dialogue.models import DialogueSecrets, SnippetStyleSources
from companion.dialogue.usage import MonthlyUsageLedger
from companion.privacy.paired_snippets import delete_snippets, write_snippets
from companion.runtime import speech_chat
from companion.runtime.settings import Settings
from companion.speech import runtime_assembly
from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechControllerIssue, SpeechState
from companion.speech.sounddevice_player import PlaybackDeviceEvent
from companion.style.snippet_source import FileSnippetStyleSource
from tests.integration.test_live_memory_invalidation import (
    NOW,
    QUERY,
    SAFE_REPLY,
    history,
    outbound,
)
from tests.integration.test_live_snippet_invalidation import (
    NEW_SNIPPET,
    OLD_SNIPPET,
    RETAINED,
    pair,
)
from tests.integration.test_memory_lifecycle_restart import _config
from tests.integration.test_speech_dialogue_runtime import CaptureProvider, _recognition
from tests.speech.controller_fakes import AudioRecorderFake, ClockFake
from tests.speech.test_runtime_assembly import (
    RuntimePlayerFake,
    RuntimeRecognizerFake,
    RuntimeSynthesizerFake,
)


class WindowsHost:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.settings = Settings(
            environment="test", log_level="INFO",
            private_data_dir=tmp_path / "private",
            runtime_data_dir=tmp_path / "runtime",
            test_results_dir=tmp_path / "results",
        )
        self.library = tmp_path / "synthetic-library.jsonl"
        self.removed, self.retained = pair(), pair(1, RETAINED)
        write_snippets(self.library, [self.removed, self.retained])
        user_id = speech_chat._trusted_user_id_from_sid_bytes(b"synthetic-windows-sid")
        config = _config(tmp_path).model_copy(update={
            "style": None,
            "snippet_style": SnippetStyleSources(
                library_source=self.library, partner_user_id=user_id,
            ),
        })
        secrets = DialogueSecrets(api_key=SecretStr("synthetic-unused-key"))
        monkeypatch.setattr(speech_chat, "load_settings", lambda _: self.settings)
        monkeypatch.setattr(speech_chat, "load_dialogue_config", lambda _: (config, secrets))
        fixed_datetime = SimpleNamespace(now=lambda _: NOW)
        monkeypatch.setattr(speech_chat, "datetime", fixed_datetime)
        monkeypatch.setattr(composition, "datetime", fixed_datetime)
        monkeypatch.setattr(
            speech_chat, "_read_current_windows_sid_bytes", lambda: b"synthetic-windows-sid",
        )
        self.provider = CaptureProvider(text=f"合成回显{OLD_SNIPPET}，稍后继续")
        provider_factory = Mock(return_value=self.provider)
        monkeypatch.setattr(composition, "OpenAICompatibleProvider", provider_factory)
        self.provider_factory = provider_factory
        self.snippet_builder = Mock(wraps=composition.build_snippet_style_runtime)
        monkeypatch.setattr(speech_chat, "build_snippet_style_runtime", self.snippet_builder)

        self.recorder = AudioRecorderFake()
        self.recognizer = RuntimeRecognizerFake(result=_recognition(QUERY))
        self.synthesizer = RuntimeSynthesizerFake()
        self.player = RuntimePlayerFake()
        run_config = SimpleNamespace(run_root=tmp_path / "synthetic-speech-run")
        monkeypatch.setattr(runtime_assembly, "_load_runtime_config", lambda _: object())
        monkeypatch.setattr(runtime_assembly, "_make_model_config", lambda *_: run_config)
        monkeypatch.setattr(runtime_assembly, "_make_audio_config", lambda *_: run_config)
        monkeypatch.setattr(runtime_assembly, "_verify_model_tree", lambda _: None)
        monkeypatch.setattr(runtime_assembly, "_resolve_device_pair", lambda _: object())
        monkeypatch.setattr(
            runtime_assembly, "_construct_recognizer", lambda _: self.recognizer,
        )
        monkeypatch.setattr(
            runtime_assembly, "_construct_synthesizer", lambda _: self.synthesizer,
        )
        monkeypatch.setattr(runtime_assembly, "_construct_recorder", lambda *_: self.recorder)
        monkeypatch.setattr(runtime_assembly, "_construct_player", lambda *_: self.player)
        monkeypatch.setattr(runtime_assembly, "_close_recognizer", lambda owner: owner.close())

        self.clock = ClockFake()
        monkeypatch.setattr(speech_chat, "_SystemClock", lambda: self.clock)
        monkeypatch.setattr(speech_chat, "time", SimpleNamespace(sleep=self.sleep))
        # Transparent spies retain the real construction, loop and close logic.
        self.assemble = Mock(wraps=runtime_assembly.assemble_local_speech_runtime)
        self.loop = Mock(wraps=speech_chat._run_console_loop)
        monkeypatch.setattr(speech_chat, "assemble_local_speech_runtime", self.assemble)
        monkeypatch.setattr(speech_chat, "_run_console_loop", self.loop)
        self.reserve = create_autospec(
            MonthlyUsageLedger.reserve, side_effect=MonthlyUsageLedger.reserve,
        )
        monkeypatch.setattr(MonthlyUsageLedger, "reserve", self.reserve)

    def sleep(self, seconds: float) -> None:
        self.clock.now += seconds

    @property
    def controller(self) -> SpeechController:
        return self.loop.call_args.kwargs["assembly"].controller

    def turn(
        self, issue: SpeechControllerIssue | None = None,
    ) -> Iterator[str | None]:
        controller = self.controller
        assert type(controller) is SpeechController
        assert controller.state is SpeechState.IDLE
        yield " "
        assert controller.state is SpeechState.RECORDING
        yield " "
        assert controller.state is SpeechState.PLAYING
        assert controller.last_issue is issue
        assert self.controller is controller
        # Playback completion goes through the real assembly event drain.
        self.player.events = (PlaybackDeviceEvent(self.player.tokens[-1], True),)
        yield None
        assert controller.state is SpeechState.IDLE
        assert self.synthesizer.live_audio_count == 0

    def run(self, keys: Iterator[str | None]) -> None:
        self.monkeypatch.setattr(speech_chat, "_read_console_key", lambda: next(keys))
        assert speech_chat.main() == 0
        # Exhaustion proves every lifecycle assertion ran before main returned.
        assert list(keys) == []
        self.assemble.assert_called_once()
        self.loop.assert_called_once()
        self.snippet_builder.assert_called_once()
        self.provider_factory.assert_called_once()
        assert self.recognizer.close_calls == self.synthesizer.close_calls == 1
        assert self.recorder.live_session_count == 0
        assert self.recorder.live_recorded_audio_count == 0
        assert self.synthesizer.live_audio_count == 0

    def seed_reply(self) -> Iterator[str | None]:
        yield from self.turn()
        assert self.provider.call_count == self.reserve.call_count == 1
        assert OLD_SNIPPET in outbound(self.provider)
        assert OLD_SNIPPET not in history(self.provider)
        assert OLD_SNIPPET in self.synthesizer.inputs[0]
        self.provider.text = SAFE_REPLY

    def ledger_bytes(self) -> dict[str, bytes]:
        root = self.settings.runtime_data_dir / "dialogue"
        return {path.name: path.read_bytes() for path in root.glob("*.json")}


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WindowsHost:
    return WindowsHost(tmp_path, monkeypatch)


@pytest.mark.parametrize("method", ["id", "time-range"])
@pytest.mark.parametrize("connected", [True, False], ids=["connected", "disconnected-control"])
def test_windows_snippet_delete_clears_live_history(host, method, connected):
    if not connected:
        def disconnect(*args, **kwargs):
            source, = kwargs.pop("privacy_revision_sources")
            assert source is kwargs["snippet_style"]
            assert type(source) is FileSnippetStyleSource
            return composition.build_real_orchestrator(*args, **kwargs)

        host.monkeypatch.setattr(speech_chat, "build_real_orchestrator", disconnect)

    def keys():
        yield from host.seed_reply()
        if method == "id":
            count = delete_snippets(
                [host.library], snippet_ids=frozenset([host.removed.snippet_id]),
            )
        else:
            count = delete_snippets([host.library], started_between=(NOW, NOW))
        assert count == 1
        yield from host.turn()
        assert host.provider.call_count == host.reserve.call_count == 2
        assert RETAINED in outbound(host.provider)
        assert (OLD_SNIPPET in outbound(host.provider)) is (not connected)
        assert (OLD_SNIPPET in history(host.provider)) is (not connected)
        if connected:
            assert history(host.provider).strip() == ""
        yield "q"

    host.run(keys())


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_windows_snippet_revision_failure_stops_outbound(host, failure, capsys):
    def keys():
        yield from host.seed_reply()
        assembly = host.loop.call_args.kwargs["assembly"]
        controller = assembly.controller
        before = host.ledger_bytes()
        assert before  # Real numeric accounting was written under tmp_path.
        if failure == "missing":
            host.library.unlink()
        else:
            host.library.write_text("{SYNTHETIC_SNIPPET_PARSE_EXCEPTION", encoding="utf-8")
        yield from host.turn(SpeechControllerIssue.DIALOGUE_FAILED)
        assert host.provider.call_count == host.reserve.call_count == 1
        assert host.ledger_bytes() == before
        assert "对话连接失败。" in capsys.readouterr().out

        write_snippets(host.library, [pair(canary=NEW_SNIPPET), host.retained])
        yield from host.turn()
        assert host.loop.call_args.kwargs["assembly"] is assembly
        assert assembly.controller is controller
        assert host.assemble.call_count == 1
        assert host.recognizer.close_calls == host.synthesizer.close_calls == 0
        assert host.provider.call_count == host.reserve.call_count == 2
        assert host.ledger_bytes() != before
        assert OLD_SNIPPET not in outbound(host.provider)
        assert NEW_SNIPPET in outbound(host.provider)
        assert RETAINED in outbound(host.provider)
        assert history(host.provider).strip() == ""
        yield "q"

    host.run(keys())


def test_windows_snippet_unchanged_library_preserves_history(host):
    def keys():
        original = host.library.read_bytes()
        yield from host.seed_reply()
        assert host.library.read_bytes() == original
        yield from host.turn()
        assert host.library.read_bytes() == original
        assert host.provider.call_count == host.reserve.call_count == 2
        assert OLD_SNIPPET in history(host.provider)
        assert QUERY in history(host.provider)
        yield "q"

    host.run(keys())
