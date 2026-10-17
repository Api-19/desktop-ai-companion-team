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

from companion.contracts import DialogueResponse
from companion.contracts.body import ButtonPressed, MuteState
from companion.contracts.robot import RobotEvent
from companion.dialogue.models import BudgetConfig
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime import budget_notice, speech_chat, speech_chat_body, speech_chat_mac
from companion.runtime.speech_dialogue import SpeechDialogueBridge
from companion.runtime.speech_expression import BodyDisplayPort, SpeechExpressionDriver
from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechControllerIssue, SpeechState, SpeechTurnFailure
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    DialogueFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)

FAILURE = "语音聊天暂时无法启动。\n"


class FakeStdin:
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 42


class FakeController:
    def __init__(self) -> None:
        self.processing_observer: Callable[[], None] | None = None
        self.state = SpeechState.IDLE
        self.last_issue = None
        self.turn_failure = None
        self.calls: list[str] = []

    def tick(self) -> None:
        self.calls.append("tick")

    def mute_changed(self) -> None:
        self.calls.append("mute")

    def button_pressed(self) -> None:
        self.calls.append("button")
        if self.state is SpeechState.RECORDING:
            self.processing_observer()
        self.state = (
            SpeechState.PLAYING if self.state is SpeechState.RECORDING else SpeechState.RECORDING
        )


class FakeLink:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.recorder = object()
        self.player = SimpleNamespace(start_for_session=Mock())
        self.mute_gate = SimpleNamespace(microphone_enabled=lambda: True)
        self.current_session_id: str | None = "s"
        self.port = kwargs["port"]
        self.inputs: list[tuple] = []
        self.shown: list[str] = []
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def close(self) -> None:
        self.calls.append("close")

    def drain_body_inputs(self) -> tuple:
        return self.inputs.pop(0) if self.inputs else ()

    def show_state(self, event: RobotEvent) -> None:
        if not self.shown or self.shown[-1] != event.state:  # BrainBodyLink sends changes only
            self.shown.append(event.state)


def mute_state(source: str) -> MuteState:
    return MuteState(protocol_version=1, seq=0, session_id="s", microphone_enabled=True,
                     source=source, observed_at_ms=0)  # fmt: skip


def button() -> ButtonPressed:
    return ButtonPressed(protocol_version=1, seq=0, session_id="s", observed_at_ms=0)


class FakeOrchestrator:
    def privacy_revision(self, user_id: str) -> tuple[()]:
        return ()


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    termios = SimpleNamespace(TCSADRAIN=1, tcgetattr=Mock(return_value=["saved"]), tcsetattr=Mock())
    tty = SimpleNamespace(setcbreak=Mock())
    assembly = SimpleNamespace(
        controller=FakeController(), drain_playback_events=Mock(), close=Mock()
    )
    settings = SimpleNamespace(
        private_data_dir=tmp_path / "app" / "private", runtime_data_dir=tmp_path / "runtime",
    )
    config = SimpleNamespace(snippet_style=object(), budget=BudgetConfig(
        monthly_cap_cny="100", worst_case_request_cny="50", input_cny_per_million="1",
        output_cny_per_million="2", max_output_tokens=256,
    ))
    secrets, adapter, profile = object(), object(), object()
    orchestrator = FakeOrchestrator()
    links: list[FakeLink] = []
    notifiers = []

    def make_notifier(**kwargs):
        notifier = budget_notice.MacBudgetNotifier(**kwargs)
        notifier.check = Mock(wraps=notifier.check)
        notifiers.append(notifier)
        return notifier

    process = Mock()
    monkeypatch.setattr(budget_notice.subprocess, "run", process)

    def make_link(**kwargs: object) -> FakeLink:
        links.append(FakeLink(**kwargs))
        return links[-1]

    keys = ["q"]
    mac_builders = {
        "load_dialogue_config": Mock(return_value=(config, secrets)),
        "build_snippet_style_runtime": Mock(return_value=(adapter, "synthetic habit card")),
        "build_private_memory_service": Mock(return_value=profile),
        "build_real_orchestrator": Mock(return_value=orchestrator),
        "MonthlyUsageLedger": Mock(wraps=MonthlyUsageLedger),
        "MacBudgetNotifier": Mock(side_effect=make_notifier),
    }
    def assemble(**kwargs: object) -> SimpleNamespace:
        assembly.controller.processing_observer = kwargs.get("processing_observer")
        return assembly

    body_builders = {
        "load_settings": Mock(return_value=settings),
        "speech_run_root": Mock(return_value=tmp_path / "run"),
        "brain_ssl_context": Mock(return_value="synthetic-ssl"),
        "BrainBodyLink": Mock(side_effect=make_link),
        "assemble_body_speech_runtime": Mock(side_effect=assemble),
    }
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.setitem(sys.modules, "termios", termios)
    monkeypatch.setitem(sys.modules, "tty", tty)
    monkeypatch.setattr(speech_chat_body.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(
        speech_chat_body, "_read_console_key", lambda: keys.pop(0) if keys else None
    )
    monkeypatch.setattr(speech_chat_body.time, "sleep", lambda _: None)
    for name, builder in mac_builders.items():
        monkeypatch.setattr(speech_chat_mac, name, builder, raising=False)
    for name, builder in body_builders.items():
        monkeypatch.setattr(speech_chat_body, name, builder)
    return SimpleNamespace(
        termios=termios, tty=tty, assembly=assembly, settings=settings, config=config,
        secrets=secrets, adapter=adapter, profile=profile, orchestrator=orchestrator,
        mac=mac_builders, body=body_builders, links=links, keys=keys, tmp_path=tmp_path,
        notifiers=notifiers, process=process,
    )  # fmt: skip


def assembled_dialogue(host: SimpleNamespace) -> object:
    builder = host.body["assemble_body_speech_runtime"]
    assert builder.call_count == 1
    return builder.call_args.kwargs["dialogue"]


def test_startup_blocked_notifies_once(host, monkeypatch, capsys) -> None:
    ledger = MonthlyUsageLedger(host.settings.runtime_data_dir / "dialogue", host.config.budget)
    now = datetime.now(UTC)
    for reservation in ledger.reserve(2, now):
        ledger.settle(reservation, usage=None, now=now)
    loop = Mock()

    def sent(*args, **kwargs):
        assert host.links[0].calls == ["start"]
        assert "大脑已启动" not in capsys.readouterr().out
        loop.assert_not_called()

    host.process.side_effect = sent

    def run(**kwargs):
        assert "大脑已启动" in capsys.readouterr().out
        (notifier,) = host.notifiers
        notifier.check.assert_called_once_with()
        callback = assembled_dialogue(host)._budget_notice
        assert callback is notifier.check
        callback()
        callback()
        host.process.assert_called_once()

    loop.side_effect = run
    monkeypatch.setattr(speech_chat_body, "_run_body_loop", loop)
    assert speech_chat_body.main([]) == 0
    host.mac["MacBudgetNotifier"].assert_called_once()
    host.process.assert_called_once()


def test_failed_startup_does_not_check_budget(host, monkeypatch) -> None:
    monkeypatch.setattr(FakeLink, "start", Mock(side_effect=RuntimeError("synthetic failure")))
    assert speech_chat_body.main([]) == 1
    (notifier,) = host.notifiers
    notifier.check.assert_not_called()
    host.process.assert_not_called()


@pytest.mark.parametrize("snippet", [True, False])
def test_main_wires_dialogue_like_mac_entry_and_body_ports(
    snippet: bool, host: SimpleNamespace, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver_factory = Mock(wraps=SpeechExpressionDriver)
    loop = Mock(wraps=speech_chat_body._run_body_loop)
    monkeypatch.setattr(speech_chat_body, "SpeechExpressionDriver", driver_factory)
    monkeypatch.setattr(speech_chat_body, "_run_body_loop", loop)
    if not snippet:
        host.config.snippet_style = None

    assert speech_chat_body.main([]) == 0

    host.body["load_settings"].assert_called_once_with(Path.cwd())
    host.mac["load_dialogue_config"].assert_called_once_with(host.settings)
    expected = {"trusted_user_id": "posix-uid-1234", "turn_observer": None}
    if snippet:
        expected.update(
            snippet_style=host.adapter,
            snippet_habit_card="synthetic habit card",
            profile_source=host.profile,
            privacy_revision_sources=(host.profile, host.adapter),
        )
        now = host.mac["build_private_memory_service"].call_args.kwargs["clock"]()
        assert now.tzinfo is UTC and now <= datetime.now(UTC)
    else:
        host.mac["build_snippet_style_runtime"].assert_not_called()
        host.mac["build_private_memory_service"].assert_not_called()
    host.mac["build_real_orchestrator"].assert_called_once_with(
        host.settings, host.config, host.secrets, **expected
    )
    dialogue = assembled_dialogue(host)
    assert type(dialogue) is SpeechDialogueBridge
    assert dialogue._orchestrator is host.orchestrator
    assert dialogue._privacy_revision == host.orchestrator.privacy_revision
    assert dialogue._privacy_revision.__self__ is host.orchestrator
    (notifier,) = host.notifiers
    assert dialogue._budget_notice is notifier.check
    notifier.check.assert_called_once_with()
    dialogue._budget_notice()
    assert notifier.check.call_count == 2
    host.process.assert_not_called()

    host.body["brain_ssl_context"].assert_called_once_with(
        host.tmp_path / "app" / "bodylink" / "brain"
    )
    (link,) = host.links
    assert link.kwargs == {
        "host": "0.0.0.0", "port": 8765, "ssl_context": "synthetic-ssl",
        "body_id": "kiki-body", "run_root": host.tmp_path / "run",
    }  # fmt: skip
    host.body["speech_run_root"].assert_called_once_with(host.settings)
    values = host.body["assemble_body_speech_runtime"].call_args.kwargs
    assert values["processing_observer"].__self__ is loop.call_args.kwargs["expression"]
    driver_factory.assert_called_once()
    assert driver_factory.call_args.args[1] is link.mute_gate
    assert values["settings"] is host.settings
    assert values["trusted_user_id"] == "posix-uid-1234"
    assert values["recorder"] is link.recorder
    assert values["player"] is link.player
    assert values["mute_gate"] is link.mute_gate
    supplier = values["session_id_supplier"]
    assert supplier() == "s"
    link.current_session_id = "new-session"
    assert supplier() == "new-session"
    assert values["session_playback_starter"] is link.player.start_for_session
    assert isinstance(values["clock"], speech_chat._SystemClock)
    assert link.calls == ["start", "close"]
    host.assembly.close.assert_called_once_with()
    host.tty.setcbreak.assert_called_once_with(42)
    host.termios.tcsetattr.assert_called_once_with(42, host.termios.TCSADRAIN, ["saved"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "大脑已启动：等待身体连接（端口 8765）。"
    assert lines[-1] == speech_chat._STATUS_TEXT["speech-chat-stopped"]


def test_command_line_overrides_link_settings(host: SimpleNamespace) -> None:
    argv = ["--cert-dir", "/synthetic/certs", "--host", "127.0.0.1", "--port", "9000",
            "--body-id", "body-2"]  # fmt: skip

    assert speech_chat_body.main(argv) == 0

    host.body["brain_ssl_context"].assert_called_once_with(Path("/synthetic/certs"))
    kwargs = host.links[0].kwargs
    assert (kwargs["host"], kwargs["port"], kwargs["body_id"]) == ("127.0.0.1", 9000, "body-2")


def test_debug_log_passes_observer_and_reports_private_path(
    host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert speech_chat_body.main(["--debug-log"]) == 0

    observer = host.mac["build_real_orchestrator"].call_args.kwargs["turn_observer"]
    assert callable(observer)
    directory = host.settings.private_data_dir / "speech" / "run" / "debug-transcripts"
    (path,) = directory.glob("transcript-*.jsonl")
    assert re.fullmatch(r"transcript-[0-9a-f]{32}\.jsonl", path.name)
    assert "本地调试记录已开启：" + str(path) in capsys.readouterr().out.splitlines()


def test_fixed_reply_skips_dialogue_config_and_orchestrator(host: SimpleNamespace) -> None:
    assert speech_chat_body.main(["--fixed-reply", "合成固定回复"]) == 0

    for builder in host.mac.values():
        builder.assert_not_called()
    assert host.notifiers == []
    host.process.assert_not_called()
    dialogue = assembled_dialogue(host)
    assert type(dialogue) is speech_chat_body.FixedReplyDialogue
    assert dialogue.reply(user_id="synthetic-user", text="合成输入") == DialogueResponse(
        text="合成固定回复", disclosed_ai_identity=False, billable=False, blocked_reason=None
    )


@pytest.mark.parametrize("issue", [
    None, SpeechControllerIssue.RECOGNITION_FAILED, "synthetic-issue",
])
def test_body_debug_timing_record_has_only_timing_keys_and_uses_existing_private_file(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, issue: str | None,
) -> None:
    from companion.speech.controller_models import TurnTiming

    timing = TurnTiming(
        played=True, issue=issue, stop_ms=1, recognize_ms=2,
        dialogue_ms=None, synthesize_ms=4, playback_start_ms=5, total_ms=20,
    )
    directory = host.settings.private_data_dir / "speech" / "run" / "debug-transcripts"

    def run_loop(**kwargs: object) -> None:
        values = host.body["assemble_body_speech_runtime"].call_args.kwargs
        (path,) = directory.iterdir()
        before = datetime.now(UTC)
        values["timing_observer"](timing)
        (line,) = path.read_text(encoding="utf-8").splitlines()
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

    monkeypatch.setattr(speech_chat_body, "_run_body_loop", run_loop)
    assert speech_chat_body.main(["--debug-log"]) == 0


@pytest.mark.parametrize("argv", [[], ["--fixed-reply", "synthetic reply"]])
def test_body_no_debug_log_passes_no_timing_writer_and_creates_no_directory(
    host: SimpleNamespace, argv: list[str]
) -> None:
    assert speech_chat_body.main(argv) == 0
    assert host.body["assemble_body_speech_runtime"].call_args.kwargs["timing_observer"] is None
    assert not (host.settings.private_data_dir / "speech" / "run" / "debug-transcripts").exists()
    assert not host.settings.private_data_dir.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--fixed-reply", "合成", "--debug-log"],
        ["--unknown"],
        ["--fixed-reply"],
        ["--port", "x"],
        ["--help"],
    ],
)
def test_invalid_arguments_fail_before_any_work(
    argv: list[str], host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert speech_chat_body.main(argv) == 1

    assert capsys.readouterr().out == FAILURE
    host.tty.setcbreak.assert_not_called()
    for builder in (*host.mac.values(), *host.body.values()):
        builder.assert_not_called()


def test_non_darwin_fails(host: SimpleNamespace, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    assert speech_chat_body.main([]) == 1

    assert capsys.readouterr().out == FAILURE
    host.body["load_settings"].assert_not_called()


@pytest.mark.parametrize("step", ["speech_run_root", "assemble_body_speech_runtime", "start"])
def test_startup_failure_releases_resources_and_returns_1(
    step: str, host: SimpleNamespace, monkeypatch, capsys
) -> None:
    if step == "start":
        monkeypatch.setattr(FakeLink, "start", Mock(side_effect=OSError("synthetic bind")))
    else:
        host.body[step].side_effect = RuntimeError("synthetic failure")

    assert speech_chat_body.main(["--fixed-reply", "合成"]) == 1

    assert capsys.readouterr().out == FAILURE
    for link in host.links:
        assert link.calls[-1] == "close"
    assert host.assembly.close.call_count == (1 if step == "start" else 0)
    host.termios.tcsetattr.assert_called_once_with(42, host.termios.TCSADRAIN, ["saved"])


# ---- loop ----


def run_loop(link: FakeLink, controller: FakeController, keys: list[str | None]) -> list[str]:
    statuses: list[str] = []
    assembly = SimpleNamespace(controller=controller, drain_playback_events=Mock(), close=Mock())
    expression = SpeechExpressionDriver(BodyDisplayPort(link), link.mute_gate)
    controller.processing_observer = expression.processing
    speech_chat_body._run_body_loop(
        assembly=assembly,
        link=link,
        expression=expression,
        read_key=lambda: keys.pop(0),
        write_status=statuses.append,
        sleep=lambda _: None,
    )
    return statuses


def test_body_inputs_drive_controller_and_show_state() -> None:
    link = FakeLink(port=0)
    controller = FakeController()
    link.inputs = [(mute_state("hardware_switch"), button()), (button(),)]

    statuses = run_loop(link, controller, [None, None, "Q"])

    assert controller.calls == ["mute", "button", "tick", "button", "tick", "tick"]
    assert link.shown == ["idle", "listening", "thinking", "speaking"]
    assert statuses == [
        "speech-chat-ready",
        "speech-chat-recording",
        "speech-chat-processing",
        "speech-chat-playing",
        "speech-chat-stopped",
    ]


def test_body_loop_reports_cloud_failure_once_and_clears_next_turn() -> None:
    link = FakeLink(port=0)
    link.inputs = [(button(),), (button(),), (), (button(),), (), (button(),), ()]
    controller = SpeechController(
        trusted_user_id="synthetic-user",
        recorder=AudioRecorderFake(), recognizer=SpeechRecognizerFake(),
        dialogue=DialogueFake(result=DialogueResponse(
            text="synthetic cloud fallback", disclosed_ai_identity=False,
            billable=False, blocked_reason="provider_timeout",
        )),
        synthesizer=SpeechSynthesizerFake(failure=RuntimeError("synthetic TTS failure")),
        player=AudioPlayerFake(), mute_gate=MuteGateFake(), clock=ClockFake(),
    )
    observed = []
    statuses: list[str] = []

    def read_key() -> str | None:
        observed.append(controller.turn_failure)
        return "q" if len(observed) == 7 else None

    speech_chat_body._run_body_loop(
        assembly=SimpleNamespace(controller=controller, drain_playback_events=Mock()),
        link=link,
        expression=SpeechExpressionDriver(BodyDisplayPort(link), link.mute_gate),
        read_key=read_key, write_status=statuses.append, sleep=lambda _: None,
    )
    failure = SpeechTurnFailure.PROVIDER_TIMEOUT
    assert observed == [None, failure, failure, None, None, failure, failure]
    assert statuses.count("speech-chat-provider-timeout") == 2
    assert statuses.count("speech-chat-synthesis-failed") == 2
    for index, status in enumerate(statuses):
        if status == "speech-chat-provider-timeout":
            assert statuses[index + 1] == "speech-chat-synthesis-failed"


def test_reconnect_between_drained_inputs_discards_remaining_old_batch(monkeypatch) -> None:
    link = FakeLink(port=0)
    controller = FakeController()
    link.inputs = [(button(), button(), mute_state("development"))]
    button_pressed = controller.button_pressed

    def reconnect() -> None:
        button_pressed()
        link.current_session_id = "B"

    monkeypatch.setattr(controller, "button_pressed", reconnect)
    observe = Mock(wraps=SpeechExpressionDriver.observe)
    monkeypatch.setattr(
        SpeechExpressionDriver, "observe", lambda driver, current: observe(driver, current)
    )

    statuses = run_loop(link, controller, ["q"])

    assert controller.calls == ["button", "tick"]
    assert statuses == ["speech-chat-ready", "speech-chat-recording", "speech-chat-stopped"]
    assert link.shown == ["idle", "listening"]
    assert observe.call_count == 3  # initial state, accepted button, end-of-loop tick


def test_development_mute_source_is_reported_once() -> None:
    link = FakeLink(port=0)
    controller = FakeController()
    link.inputs = [(mute_state("development"),), (mute_state("development"),)]

    statuses = run_loop(link, controller, [None, None, "q"])

    assert statuses.count("bodylink-development-mute") == 1
    assert controller.calls.count("mute") == 2


def test_development_mute_text(capsys: pytest.CaptureFixture[str]) -> None:
    speech_chat_body._write_console_status("bodylink-development-mute")
    speech_chat_body._write_console_status("speech-chat-ready")
    speech_chat_body._write_console_status("speech-chat-dialogue-failed")
    assert capsys.readouterr().out.splitlines() == [
        "身体为开发模式：未接物理断麦开关",
        "等待身体按键开始录音，按 Q 退出。",
        speech_chat._STATUS_TEXT["speech-chat-dialogue-failed"],
    ]


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
    notifications: list[str] = []
    recognized: list[bool] = []
    processing = SpeechExpressionDriver.processing

    def notify(driver: SpeechExpressionDriver) -> None:
        notifications.append("processing")
        processing(driver)

    def session_closed() -> None:
        notifications.append("closed")
        if mute_at_stop:
            gate.enabled = False

    def recognize() -> None:
        notifications.append("recognize")
        assert host.links[0].shown[-1] == ("muted" if mute_at_stop else "thinking")
        assert notifications == ["stopped", "closed", "processing", "recognize"]
        recognized.append(True)

    recorder.configure_callback("session.stop", lambda: notifications.append("stopped"))
    recorder.configure_callback("session.close", session_closed)
    recognizer.configure_callback("recognizer.recognize", recognize)
    monkeypatch.setattr(SpeechExpressionDriver, "processing", notify)
    monkeypatch.setattr(speech_chat_body, "_SystemClock", lambda: clock)

    def make_link(**kwargs: object) -> FakeLink:
        link = FakeLink(**kwargs)
        link.recorder = recorder
        link.player = AudioPlayerFake()
        link.player.start_for_session = lambda audio, token, _session_id: link.player.start(
            audio, token
        )
        link.mute_gate = gate
        link.inputs = [(button(),), () if recording_limit else (button(),)]
        host.links.append(link)
        return link

    def assemble(**kwargs: object) -> SimpleNamespace:
        controller = SpeechController(
            trusted_user_id=kwargs["trusted_user_id"],
            recorder=kwargs["recorder"],
            recognizer=recognizer,
            dialogue=kwargs["dialogue"],
            synthesizer=synthesizer,
            player=kwargs["player"],
            mute_gate=kwargs["mute_gate"],
            clock=kwargs["clock"],
            processing_observer=kwargs.get("processing_observer"),
            session_id_supplier=kwargs["session_id_supplier"],
            session_playback_starter=kwargs["session_playback_starter"],
        )
        host.assembly.controller = controller
        host.assembly.close.side_effect = controller.shutdown
        return host.assembly

    host.body["BrainBodyLink"].side_effect = make_link
    host.body["assemble_body_speech_runtime"].side_effect = assemble
    host.keys[:] = [None, None, "q"]
    times = iter([29.9, 30.0] if recording_limit else [0.5, 0.5])

    def advance_clock(_: float) -> None:
        if recording_limit:
            assert recognizer.calls == 0
        clock.now = next(times)

    monkeypatch.setattr(speech_chat_body.time, "sleep", advance_clock)

    assert speech_chat_body.main(["--fixed-reply", "synthetic reply"]) == 0
    assert recognized == [True]
    assert notifications == ["stopped", "closed", "processing", "recognize"]
    assert recognizer.calls == 1
    assert recorder.live_session_count == recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0
    host.assembly.close.assert_called_once_with()
    assert host.links[0].calls == ["start", "close"]


def test_body_recording_limit_shows_thinking_before_recognition(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_synthetic_turn(host, monkeypatch, recording_limit=True)


def test_body_manual_stop_notifies_expression_once(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_synthetic_turn(host, monkeypatch, recording_limit=False)


@pytest.mark.parametrize("recording_limit", [False, True])
def test_body_processing_notification_respects_mute(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, recording_limit: bool,
) -> None:
    _run_synthetic_turn(host, monkeypatch, recording_limit=recording_limit, mute_at_stop=True)
