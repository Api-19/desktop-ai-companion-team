# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
import stat
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from companion.dialogue.models import BudgetConfig
from companion.dialogue.observer import TurnObservation
from companion.dialogue.usage import MonthlyUsageLedger
from companion.expression import Face
from companion.runtime import budget_notice, speech_chat, speech_chat_mac, speech_expression
from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechControllerIssue, SpeechState
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    DialogueFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)


class FakeStdin:
    def __init__(self, *, is_tty: bool = True) -> None:
        self.is_tty = is_tty

    def isatty(self) -> bool:
        return self.is_tty

    def fileno(self) -> int:
        return 42


class FakeController:
    def __init__(self) -> None:
        self.processing_observer: Callable[[], None] | None = None
        self.state = SpeechState.IDLE
        self.last_issue = None
        self.turn_failure = None
        self.button_calls = 0
        self.tick_calls = 0

    def tick(self) -> None:
        self.tick_calls += 1

    def button_pressed(self) -> None:
        self.button_calls += 1
        if self.state is SpeechState.RECORDING:
            self.processing_observer()
        self.state = (
            SpeechState.IDLE if self.state is SpeechState.RECORDING else SpeechState.RECORDING
        )


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    stdin = FakeStdin()
    saved = ["synthetic terminal settings"]
    termios = SimpleNamespace(
        TCSADRAIN=1,
        tcgetattr=Mock(return_value=saved),
        tcsetattr=Mock(),
    )
    tty = SimpleNamespace(setcbreak=Mock())
    assembly = SimpleNamespace(
        controller=FakeController(),
        drain_playback_events=Mock(),
        close=Mock(),
    )
    settings = SimpleNamespace(
        private_data_dir=tmp_path / "synthetic-private", runtime_data_dir=tmp_path / "runtime",
    )
    config = SimpleNamespace(snippet_style=object(), budget=BudgetConfig(
        monthly_cap_cny="100", worst_case_request_cny="50", input_cny_per_million="1",
        output_cny_per_million="2", max_output_tokens=256,
    ))
    secrets = object()
    adapter = object()
    habit_card = "synthetic habit card"
    profile = object()
    orchestrator = object()
    notifiers = []

    def make_notifier(**kwargs):
        notifier = budget_notice.MacBudgetNotifier(**kwargs)
        notifier.check = Mock(wraps=notifier.check)
        notifiers.append(notifier)
        return notifier

    process = Mock()
    monkeypatch.setattr(budget_notice.subprocess, "run", process)
    def assemble(**kwargs: object) -> SimpleNamespace:
        assembly.controller.processing_observer = kwargs.get("processing_observer")
        return assembly

    builders = {
        "load_settings": Mock(return_value=settings),
        "load_dialogue_config": Mock(return_value=(config, secrets)),
        "build_snippet_style_runtime": Mock(return_value=(adapter, habit_card)),
        "build_private_memory_service": Mock(return_value=profile),
        "build_real_orchestrator": Mock(return_value=orchestrator),
        "assemble_local_speech_runtime": Mock(side_effect=assemble),
        "MonthlyUsageLedger": Mock(wraps=MonthlyUsageLedger),
        "MacBudgetNotifier": Mock(side_effect=make_notifier),
    }
    monkeypatch.setattr(speech_chat_mac.sys, "platform", "darwin")
    monkeypatch.setattr(speech_chat_mac.sys, "stdin", stdin)
    monkeypatch.setattr(speech_chat_mac.sys, "argv", ["speech_chat_mac"])
    monkeypatch.setitem(sys.modules, "termios", termios)
    monkeypatch.setitem(sys.modules, "tty", tty)
    monkeypatch.setattr(speech_chat_mac.os, "getuid", lambda: 1234, raising=False)
    read = Mock(return_value=b"q")
    select = Mock(return_value=([42], [], []))
    monkeypatch.setattr(speech_chat_mac.os, "read", read)
    monkeypatch.setattr(speech_chat_mac.select, "select", select)
    monkeypatch.setattr(speech_chat_mac.time, "sleep", lambda _: None)
    for name, builder in builders.items():
        monkeypatch.setattr(speech_chat_mac, name, builder, raising=False)
    return SimpleNamespace(
        stdin=stdin,
        saved=saved,
        termios=termios,
        tty=tty,
        assembly=assembly,
        settings=settings,
        config=config,
        secrets=secrets,
        adapter=adapter,
        habit_card=habit_card,
        profile=profile,
        orchestrator=orchestrator,
        builders=builders,
        read=read,
        select=select,
        notifiers=notifiers,
        process=process,
    )


@pytest.mark.parametrize("snippet", [True, False])
def test_mac_notice_uses_loaded_budget_and_runtime_root(snippet, host) -> None:
    if not snippet:
        host.config.snippet_style = None
    before = datetime.now(UTC)

    orchestrator, notifier = speech_chat_mac._build_orchestrator(
        host.settings, "synthetic-user", None,
    )

    assert orchestrator is host.orchestrator
    assert host.notifiers == [notifier]
    host.builders["load_dialogue_config"].assert_called_once_with(host.settings)
    ledger_factory = host.builders["MonthlyUsageLedger"]
    ledger_factory.assert_called_once_with(
        host.settings.runtime_data_dir / "dialogue", host.config.budget,
    )
    assert ledger_factory.call_args.args[1] is host.config.budget
    kwargs = host.builders["MacBudgetNotifier"].call_args.kwargs
    assert set(kwargs) == {"ledger", "clock"}
    assert isinstance(kwargs["ledger"], MonthlyUsageLedger)
    now = kwargs["clock"]()
    assert now.tzinfo is UTC and before <= now <= datetime.now(UTC)
    notifier.check.assert_not_called()
    host.process.assert_not_called()


def test_startup_blocked_notifies_once(host, monkeypatch) -> None:
    ledger = MonthlyUsageLedger(host.settings.runtime_data_dir / "dialogue", host.config.budget)
    now = datetime.now(UTC)
    for reservation in ledger.reserve(2, now):
        ledger.settle(reservation, usage=None, now=now)
    loop = Mock()

    def sent(*args, **kwargs):
        host.builders["assemble_local_speech_runtime"].assert_called_once()
        loop.assert_not_called()

    host.process.side_effect = sent

    def run(**kwargs):
        (notifier,) = host.notifiers
        notifier.check.assert_called_once_with()
        callback = host.builders["assemble_local_speech_runtime"].call_args.kwargs["budget_notice"]
        assert callback is notifier.check
        callback()
        callback()
        host.process.assert_called_once()

    loop.side_effect = run
    monkeypatch.setattr(speech_chat_mac, "_run_console_loop", loop)
    assert speech_chat_mac.main() == 0
    host.builders["MacBudgetNotifier"].assert_called_once()
    host.process.assert_called_once()


def test_failed_startup_does_not_check_budget(host) -> None:
    host.builders["assemble_local_speech_runtime"].side_effect = RuntimeError("synthetic failure")
    assert speech_chat_mac.main() == 1
    (notifier,) = host.notifiers
    notifier.check.assert_not_called()
    host.process.assert_not_called()


@pytest.mark.parametrize("snippet", [True, False])
def test_main_wires_snippets_and_profile_only_when_configured(
    snippet: bool, host: SimpleNamespace, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not snippet:
        host.config.snippet_style = None
    driver_factory = Mock(wraps=speech_chat_mac.SpeechExpressionDriver)
    loop = Mock(wraps=speech_chat_mac._run_console_loop)
    monkeypatch.setattr(speech_chat_mac, "SpeechExpressionDriver", driver_factory)
    monkeypatch.setattr(speech_chat_mac, "_run_console_loop", loop)
    before = datetime.now(UTC)

    assert speech_chat_mac.main() == 0

    host.builders["load_settings"].assert_called_once_with(Path.cwd())
    host.builders["load_dialogue_config"].assert_called_once_with(host.settings)
    expected = {"trusted_user_id": "posix-uid-1234", "turn_observer": None}
    if snippet:
        expected.update(
            snippet_style=host.adapter,
            snippet_habit_card=host.habit_card,
            profile_source=host.profile,
            privacy_revision_sources=(host.profile, host.adapter),
        )
        for name, value in (
            ("build_snippet_style_runtime", host.config),
            ("build_private_memory_service", host.settings),
        ):
            builder = host.builders[name]
            assert builder.call_count == 1
            assert builder.call_args.args == (value,)
            assert set(builder.call_args.kwargs) == {"clock"}
            now = builder.call_args.kwargs["clock"]()
            assert now.tzinfo is UTC
            assert before <= now <= datetime.now(UTC)
    else:
        host.builders["build_snippet_style_runtime"].assert_not_called()
        host.builders["build_private_memory_service"].assert_not_called()
    host.builders["build_real_orchestrator"].assert_called_once_with(
        host.settings, host.config, host.secrets, **expected
    )
    assert not host.settings.private_data_dir.exists()
    builder = host.builders["assemble_local_speech_runtime"]
    assert builder.call_count == 1
    values = builder.call_args.kwargs
    assert set(values) == {
        "settings", "orchestrator", "trusted_user_id", "mute_gate", "clock", "timing_observer",
        "processing_observer", "budget_notice",
    }
    (notifier,) = host.notifiers
    assert values["budget_notice"] is notifier.check
    notifier.check.assert_called_once_with()
    values["budget_notice"]()
    assert notifier.check.call_count == 2
    host.process.assert_not_called()
    assert values["processing_observer"].__self__ is loop.call_args.kwargs["observe"].__self__
    driver_factory.assert_called_once()
    assert driver_factory.call_args.args[1] is values["mute_gate"]
    assert values["settings"] is host.settings
    assert values["orchestrator"] is host.orchestrator
    assert values["trusted_user_id"] == "posix-uid-1234"
    assert isinstance(values["mute_gate"], speech_chat_mac.MacDevelopmentMuteGate)
    assert values["mute_gate"].physical_mute_verified is False
    assert values["mute_gate"].microphone_enabled() is True
    assert isinstance(values["clock"], speech_chat._SystemClock)
    host.tty.setcbreak.assert_called_once_with(42)
    host.termios.tcsetattr.assert_called_once_with(42, host.termios.TCSADRAIN, host.saved)
    host.assembly.close.assert_called_once_with()
    assert capsys.readouterr().out.splitlines() == [
        "Mac 开发模式：尚未验证机器人实体断麦。",
        speech_chat._STATUS_TEXT["speech-chat-ready"],
        speech_expression.FACE_TEXT[Face.NEUTRAL],
        speech_chat._STATUS_TEXT["speech-chat-stopped"],
    ]


@pytest.mark.parametrize("snippet", [True, False])
def test_debug_log_records_synthetic_turns_and_private_permissions(
    snippet: bool,
    host: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not snippet:
        host.config.snippet_style = None
    observation = TurnObservation(
        blocked_reason=None,
        input_chars=6, context_count=2, context_chars=14, reply_chars=6,
        similar_count=1, general_count=1, profile_count=1, provider_attempted=True,
        provider_failure="synthetic-provider-failure",
        provider_elapsed_ms=20003,
        observed_at=datetime(2026, 10, 12, 1, 2, 3, tzinfo=UTC),
    )
    blocked = observation.model_copy(update={
        "blocked_reason": "synthetic-block", "input_chars": 6,
        "similar_count": 0, "general_count": 0, "profile_count": 0,
        "context_count": 0, "context_chars": 0, "provider_attempted": False,
        "provider_failure": None, "provider_elapsed_ms": None,
    })
    directory = host.settings.private_data_dir / "speech" / "run" / "debug-transcripts"

    def run_loop(**kwargs: object) -> None:
        observer = host.builders["build_real_orchestrator"].call_args.kwargs["turn_observer"]
        assert callable(observer)
        path, = directory.glob("transcript-*.jsonl")
        assert re.fullmatch(r"transcript-[0-9a-f]{32}\.jsonl", path.name)
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        for count, turn in enumerate((observation, blocked), start=1):
            observer(turn)
            raw = path.read_text(encoding="utf-8")
            assert len(raw.splitlines()) == count  # Visible before the file is closed.
        rows = [json.loads(line) for line in raw.splitlines()]
        expected = {
            "observed_at": observation.observed_at.isoformat(),
            "blocked_reason": None,
            "input_chars": 6,
            "context_count": 2,
            "context_chars": 14,
            "reply_chars": 6,
            "similar_count": 1,
            "general_count": 1,
            "profile_count": 1,
            "provider_attempted": True,
            "provider_failure": "synthetic-provider-failure",
            "provider_elapsed_ms": 20003,
        }
        assert rows == [expected, {
            **expected, "blocked_reason": "synthetic-block",
            "similar_count": 0, "general_count": 0, "profile_count": 0,
            "context_count": 0, "context_chars": 0, "provider_attempted": False,
            "provider_failure": None, "provider_elapsed_ms": None,
        }]
        kwargs["write_status"]("speech-chat-development-mode")
        kwargs["write_status"]("speech-chat-ready")

    monkeypatch.setattr(speech_chat_mac, "_run_console_loop", run_loop)

    assert speech_chat_mac.main(["--debug-log"]) == 0

    path, = directory.glob("transcript-*.jsonl")
    assert capsys.readouterr().out.splitlines() == [
        "Mac 开发模式：尚未验证机器人实体断麦。",
        "本地调试记录已开启：" + str(path),
        speech_chat._STATUS_TEXT["speech-chat-ready"],
    ]
    observer = host.builders["build_real_orchestrator"].call_args.kwargs["turn_observer"]
    with pytest.raises(ValueError, match="closed file"):
        observer(observation)


@pytest.mark.parametrize("argument", ["--unknown", "--debug", "--help", "--debug-log=yes"])
def test_unknown_argument_returns_fixed_failure(
    argument: str, host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert speech_chat_mac.main([argument]) == 1
    captured = capsys.readouterr()
    assert captured.out == "语音聊天暂时无法启动。\n"
    assert captured.err == ""
    host.tty.setcbreak.assert_not_called()
    for builder in host.builders.values():
        builder.assert_not_called()
    assert not host.settings.private_data_dir.exists()


@pytest.mark.parametrize(
    "step",
    [
        "load_settings",
        "load_dialogue_config",
        "build_snippet_style_runtime",
        "build_private_memory_service",
        "build_real_orchestrator",
        "assemble_local_speech_runtime",
    ],
)
def test_build_failure_restores_terminal_and_prints_fixed_error(
    step: str, host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    host.builders[step].side_effect = RuntimeError("synthetic failure detail")

    assert speech_chat_mac.main() == 1

    assert capsys.readouterr().out == "语音聊天暂时无法启动。\n"
    host.tty.setcbreak.assert_called_once_with(42)
    host.termios.tcsetattr.assert_called_once_with(42, host.termios.TCSADRAIN, host.saved)
    host.assembly.close.assert_not_called()


@pytest.mark.parametrize("keys", [[b" ", b"q"], [b" ", b" ", b"Q"]])
def test_stdin_space_dispatches_recording_and_q_exits(
    keys: list[bytes], host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    host.read.side_effect = keys

    assert speech_chat_mac.main() == 0

    assert host.assembly.controller.button_calls == len(keys) - 1
    assert host.assembly.controller.tick_calls == len(keys)
    assert host.assembly.drain_playback_events.call_count == len(keys)
    assert host.read.call_count == len(keys)
    host.read.assert_called_with(42, 1)
    host.select.assert_called_with([42], [], [], 0)
    host.assembly.close.assert_called_once_with()
    host.termios.tcsetattr.assert_called_once_with(42, host.termios.TCSADRAIN, host.saved)
    lines = capsys.readouterr().out.splitlines()
    recording = lines.index("正在录音，再按空格结束。")
    assert lines[recording + 1] == speech_expression.FACE_TEXT[Face.ATTENTIVE]
    if len(keys) == 3:
        processing = lines.index("正在处理。")
        assert lines[processing + 1] == speech_expression.FACE_TEXT[Face.THINKING]


def test_key_read_returns_immediately_when_stdin_is_not_ready(host: SimpleNamespace) -> None:
    host.select.return_value = ([], [], [])

    assert speech_chat_mac._read_console_key() is None

    host.select.assert_called_once_with([42], [], [], 0)
    host.read.assert_not_called()


@pytest.mark.parametrize("issue", [None, SpeechControllerIssue.DIALOGUE_FAILED, "synthetic-issue"])
def test_debug_timing_record_has_only_timing_keys_and_uses_existing_private_file(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, issue: str | None,
) -> None:
    from companion.speech.controller_models import TurnTiming

    timing = TurnTiming(
        played=True, issue=issue, stop_ms=1, recognize_ms=2, dialogue_ms=3,
        synthesize_ms=4, playback_start_ms=5, total_ms=20,
    )
    directory = host.settings.private_data_dir / "speech" / "run" / "debug-transcripts"

    def run_loop(**kwargs: object) -> None:
        values = host.builders["assemble_local_speech_runtime"].call_args.kwargs
        (path,) = directory.iterdir()
        turn_observer = host.builders["build_real_orchestrator"].call_args.kwargs["turn_observer"]
        turn_observer(TurnObservation(
            blocked_reason=None, input_chars=20, context_count=0, context_chars=0,
            reply_chars=15, similar_count=0, general_count=0, profile_count=0,
            provider_attempted=True, observed_at=datetime.now(UTC),
        ))
        dialogue_line = path.read_text(encoding="utf-8")
        before = datetime.now(UTC)
        values["timing_observer"](timing)
        (first_line, line) = path.read_text(encoding="utf-8").splitlines()
        assert first_line + "\n" == dialogue_line
        assert "kind" not in json.loads(first_line)
        row = json.loads(line)
        assert set(row) == {
            "kind", "observed_at", "played", "issue", "stop_ms", "recognize_ms",
            "dialogue_ms", "synthesize_ms", "playback_start_ms", "total_ms",
        }
        assert row.pop("kind") == "speech_timing"
        assert before <= datetime.fromisoformat(row.pop("observed_at")) <= datetime.now(UTC)
        assert row == timing.model_dump()
        assert list(directory.iterdir()) == [path]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    monkeypatch.setattr(speech_chat_mac, "_run_console_loop", run_loop)
    assert speech_chat_mac.main(["--debug-log"]) == 0


def test_no_debug_log_passes_no_timing_writer_and_creates_no_directory(
    host: SimpleNamespace,
) -> None:
    assert speech_chat_mac.main([]) == 0
    values = host.builders["assemble_local_speech_runtime"].call_args.kwargs
    assert values["timing_observer"] is None
    assert not (host.settings.private_data_dir / "speech" / "run" / "debug-transcripts").exists()
    assert not host.settings.private_data_dir.exists()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_non_darwin_fails_before_terminal_or_builders(
    platform: str,
    host: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(speech_chat_mac.sys, "platform", platform)

    assert speech_chat_mac.main() == 1

    assert capsys.readouterr().out == "语音聊天暂时无法启动。\n"
    host.termios.tcgetattr.assert_not_called()
    host.tty.setcbreak.assert_not_called()
    for builder in host.builders.values():
        builder.assert_not_called()


def test_non_tty_fails_before_terminal_or_builders(
    host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    host.stdin.is_tty = False

    assert speech_chat_mac.main() == 1

    assert capsys.readouterr().out == "语音聊天暂时无法启动。\n"
    host.termios.tcgetattr.assert_not_called()
    host.tty.setcbreak.assert_not_called()
    for builder in host.builders.values():
        builder.assert_not_called()


@pytest.mark.parametrize("interrupt", [False, True])
def test_loop_exception_closes_assembly_and_restores_terminal(
    interrupt: bool, host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    host.assembly.drain_playback_events.side_effect = (
        KeyboardInterrupt if interrupt else RuntimeError("synthetic failure detail")
    )

    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            speech_chat_mac.main()
    else:
        assert speech_chat_mac.main() == 1
        assert "语音聊天暂时无法启动。" in capsys.readouterr().out

    host.assembly.close.assert_called_once_with()
    host.termios.tcsetattr.assert_called_once_with(42, host.termios.TCSADRAIN, host.saved)


def test_cbreak_failure_still_restores_terminal(
    host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    host.tty.setcbreak.side_effect = OSError("synthetic terminal failure")

    assert speech_chat_mac.main() == 1

    assert capsys.readouterr().out == "语音聊天暂时无法启动。\n"
    host.termios.tcsetattr.assert_called_once_with(42, host.termios.TCSADRAIN, host.saved)
    host.builders["load_settings"].assert_not_called()


def test_all_other_status_text_is_shared(capsys: pytest.CaptureFixture[str]) -> None:
    for status, text in speech_chat._STATUS_TEXT.items():
        if status == "speech-chat-development-mode":
            continue
        speech_chat_mac._write_console_status(status)
        assert capsys.readouterr().out == text + "\n"
    speech_chat_mac._write_console_status("synthetic-unknown-status")
    assert capsys.readouterr().out == ""


def test_face_display_failure_keeps_speech_running(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken_show(self: object, face: Face) -> None:
        raise RuntimeError("synthetic display failure")

    monkeypatch.setattr(speech_expression.TerminalFaceDisplay, "show", broken_show)
    host.read.side_effect = [b" ", b" ", b"q"]

    assert speech_chat_mac.main() == 0

    assert host.assembly.controller.button_calls == 2
    assert host.assembly.controller.tick_calls == 3
    host.assembly.close.assert_called_once_with()
    output = capsys.readouterr().out
    assert "表情：" not in output
    assert output.splitlines()[-1] == speech_chat._STATUS_TEXT["speech-chat-stopped"]


def _run_synthetic_turn(
    host: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    recording_limit: bool,
    mute_at_stop: bool = False,
) -> None:
    clock = ClockFake(now=0.0)
    recorder = AudioRecorderFake()
    recognizer = SpeechRecognizerFake()
    synthesizer = SpeechSynthesizerFake()
    gate = MuteGateFake()
    faces: list[Face] = []
    notifications: list[str] = []
    recognized: list[bool] = []
    processing = speech_expression.SpeechExpressionDriver.processing

    def notify(driver: speech_expression.SpeechExpressionDriver) -> None:
        notifications.append("processing")
        processing(driver)

    def session_closed() -> None:
        notifications.append("closed")
        if mute_at_stop:
            gate.enabled = False

    def recognize() -> None:
        notifications.append("recognize")
        assert faces[-1] is (Face.MUTED if mute_at_stop else Face.THINKING)
        assert notifications == ["stopped", "closed", "processing", "recognize"]
        recognized.append(True)

    recorder.configure_callback("session.stop", lambda: notifications.append("stopped"))
    recorder.configure_callback("session.close", session_closed)
    recognizer.configure_callback("recognizer.recognize", recognize)
    monkeypatch.setattr(speech_expression.SpeechExpressionDriver, "processing", notify)
    monkeypatch.setattr(
        speech_expression.TerminalFaceDisplay, "show", lambda self, face: faces.append(face)
    )
    monkeypatch.setattr(speech_chat_mac, "MacDevelopmentMuteGate", lambda: gate)
    monkeypatch.setattr(speech_chat_mac, "_SystemClock", lambda: clock)

    def assemble(**kwargs: object) -> SimpleNamespace:
        controller = SpeechController(
            trusted_user_id=kwargs["trusted_user_id"],
            recorder=recorder,
            recognizer=recognizer,
            dialogue=DialogueFake(),
            synthesizer=synthesizer,
            player=AudioPlayerFake(),
            mute_gate=kwargs["mute_gate"],
            clock=kwargs["clock"],
            processing_observer=kwargs.get("processing_observer"),
        )
        host.assembly.controller = controller
        host.assembly.close.side_effect = controller.shutdown
        return host.assembly

    host.builders["assemble_local_speech_runtime"].side_effect = assemble
    host.read.side_effect = [b" ", b"", b"q"] if recording_limit else [b" ", b" ", b"q"]
    times = iter([29.9, 30.0] if recording_limit else [0.5, 0.5])

    def advance_clock(_: float) -> None:
        if recording_limit:
            assert recognizer.calls == 0
        clock.now = next(times)

    monkeypatch.setattr(speech_chat_mac.time, "sleep", advance_clock)

    assert speech_chat_mac.main([]) == 0
    assert recognized == [True]
    assert notifications == ["stopped", "closed", "processing", "recognize"]
    assert recognizer.calls == 1
    assert recorder.live_session_count == recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0
    host.assembly.close.assert_called_once_with()


def test_mac_recording_limit_shows_thinking_before_recognition(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_synthetic_turn(host, monkeypatch, recording_limit=True)


def test_mac_manual_stop_notifies_expression_once(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_synthetic_turn(host, monkeypatch, recording_limit=False)


@pytest.mark.parametrize("recording_limit", [False, True])
def test_mac_processing_notification_respects_mute(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, recording_limit: bool,
) -> None:
    _run_synthetic_turn(host, monkeypatch, recording_limit=recording_limit, mute_at_stop=True)
