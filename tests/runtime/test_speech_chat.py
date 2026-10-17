# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from companion.expression import ExpressionController, Face
from companion.runtime import budget_notice, speech_chat, speech_expression
from companion.runtime.speech_chat import (
    SpeechChatError,
    WindowsDevelopmentMuteGate,
    _read_current_windows_sid_bytes,
    _run_console_loop,
    _trusted_user_id_from_sid_bytes,
    _windows_trusted_user_id,
)
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


def test_same_idle_failure_is_reported_for_each_button_attempt():
    statuses = []
    controller = type("Controller", (), {})()
    controller.state = SpeechState.IDLE
    controller.last_issue = SpeechControllerIssue.RECORDING_FAILED
    controller.turn_failure = None
    for _ in range(2):
        speech_chat._report_console_state(
            controller,
            SpeechState.IDLE,
            controller.last_issue,
            None,
            statuses.append,
            after_button=True,
        )
    speech_chat._report_console_state(
        controller, SpeechState.IDLE, controller.last_issue, None, statuses.append
    )
    assert statuses == ["speech-chat-recording-failed"] * 2


def test_unknown_issue_is_not_rendered():
    controller = type("Controller", (), {})()
    controller.state = SpeechState.IDLE
    controller.last_issue = "PRIVATE_SENTINEL"
    controller.turn_failure = None
    statuses = []
    speech_chat._report_console_state(
        controller, SpeechState.IDLE, None, None, statuses.append, after_button=True
    )
    assert statuses == []


class FakeController:
    def __init__(self) -> None:
        self.state = SpeechState.IDLE
        self.last_issue = None
        self.turn_failure = None
        self.button_calls = 0
        self.tick_calls = 0

    def button_pressed(self) -> None:
        self.button_calls += 1

    def tick(self) -> None:
        self.tick_calls += 1


class FakeAssembly:
    def __init__(self) -> None:
        self.controller = FakeController()
        self.drain_calls = 0
        self.close_calls = 0

    def drain_playback_events(self) -> None:
        self.drain_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize("issue", [None, *SpeechControllerIssue])
def test_processing_return_to_idle_reports_fixed_issue_and_ready(issue: object) -> None:
    assembly = FakeAssembly()
    assembly.controller.state = SpeechState.RECORDING
    assembly.controller.last_issue = None

    def finish() -> None:
        assembly.controller.state = SpeechState.IDLE
        assembly.controller.last_issue = issue

    assembly.controller.button_pressed = finish
    statuses: list[str] = []
    _run_console_loop(
        assembly=assembly,
        read_key=key_reader(" ", None, "q"),
        write_status=statuses.append,
        sleep=lambda _: None,
    )
    processing_index = statuses.index("speech-chat-processing")
    assert "speech-chat-ready" in statuses[processing_index + 1 :]
    if issue is not None:
        assert f"speech-chat-{issue.value}" in statuses[processing_index + 1 :]
    assert statuses.count("speech-chat-ready") == 2


def test_async_playback_failure_reports_failure_and_ready_once() -> None:
    assembly = FakeAssembly()
    assembly.controller.state = SpeechState.PLAYING
    assembly.controller.last_issue = None

    def drain() -> None:
        assembly.controller.state = SpeechState.IDLE
        assembly.controller.last_issue = SpeechControllerIssue.PLAYBACK_FAILED

    assembly.drain_playback_events = drain
    statuses: list[str] = []
    _run_console_loop(
        assembly=assembly,
        read_key=key_reader(None, None, "q"),
        write_status=statuses.append,
        sleep=lambda _: None,
    )
    assert statuses.count("speech-chat-playback-failed") == 1
    assert statuses.count("speech-chat-ready") == 2


@pytest.mark.parametrize("reason", [
    "provider-timeout", "provider-unavailable", "invalid-provider-response",
])
def test_cloud_failure_reports_once_despite_later_speech_failure(reason: str) -> None:
    failure = SpeechTurnFailure(reason)
    controller = FakeController()
    controller.state = SpeechState.PLAYING
    controller.last_issue = None
    controller.turn_failure = failure
    statuses: list[str] = []
    state, issue, turn_failure = speech_chat._report_console_state(
        controller, SpeechState.RECORDING, None, None, statuses.append, after_button=True
    )
    controller.state = SpeechState.IDLE
    controller.last_issue = SpeechControllerIssue.PLAYBACK_FAILED
    state, issue, turn_failure = speech_chat._report_console_state(
        controller, state, issue, turn_failure, statuses.append
    )
    speech_chat._report_console_state(controller, state, issue, turn_failure, statuses.append)
    speech_chat._report_console_state(
        controller, state, issue, turn_failure, statuses.append, after_button=True
    )
    assert statuses == [
        f"speech-chat-{reason}", "speech-chat-playing",
        "speech-chat-playback-failed", "speech-chat-ready", "speech-chat-playback-failed",
    ]


@pytest.mark.parametrize("reason", [
    "provider-timeout", "provider-unavailable", "invalid-provider-response",
])
def test_repeated_cloud_failed_turns_each_report_once(reason: str) -> None:
    failure = SpeechTurnFailure(reason)
    assembly = FakeAssembly()
    controller = assembly.controller
    controller.state = SpeechState.IDLE
    controller.last_issue = None

    def press() -> None:
        if controller.state is SpeechState.IDLE:
            controller.state = SpeechState.RECORDING
            controller.last_issue = None
            controller.turn_failure = None
        else:
            controller.state = SpeechState.IDLE
            controller.last_issue = SpeechControllerIssue.SYNTHESIS_FAILED
            controller.turn_failure = failure

    controller.button_pressed = press
    statuses: list[str] = []
    _run_console_loop(
        assembly=assembly,
        read_key=key_reader(" ", " ", None, " ", " ", None, "q"),
        write_status=statuses.append,
        sleep=lambda _: None,
    )
    assert statuses.count(f"speech-chat-{reason}") == 2
    assert statuses.count("speech-chat-synthesis-failed") == 2
    for index, status in enumerate(statuses):
        if status == f"speech-chat-{reason}":
            assert statuses[index + 1] == "speech-chat-synthesis-failed"


@pytest.mark.parametrize(("status", "text"), [
    ("speech-chat-provider-timeout", "云端回复超时。"),
    ("speech-chat-provider-unavailable", "云端聊天服务暂时不可用。"),
    ("speech-chat-invalid-provider-response", "云端回复无效。"),
])
def test_cloud_failure_status_text(status: str, text: str, capsys) -> None:
    speech_chat._write_console_status(status)
    assert capsys.readouterr().out == text + "\n"


def key_reader(*keys: str | None):
    remaining = list(keys)

    def read() -> str | None:
        return remaining.pop(0)

    return read


def test_sid_bytes_create_stable_private_trusted_user_id() -> None:
    sid = b"synthetic-windows-token-sid"

    result = _trusted_user_id_from_sid_bytes(sid)

    assert result == f"windows-sid-sha256:{hashlib.sha256(sid).hexdigest()}"
    assert "synthetic" not in result


@pytest.mark.parametrize("value", [b"", bytearray(b"sid"), "sid", None, True])
def test_invalid_sid_material_fails_with_fixed_error(value: object) -> None:
    with pytest.raises(SpeechChatError, match=r"^speech chat unavailable$") as captured:
        _trusted_user_id_from_sid_bytes(value)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows token identity only")
def test_current_windows_token_produces_private_trusted_user_id() -> None:
    result = _windows_trusted_user_id()

    assert type(result) is str
    assert result.startswith("windows-sid-sha256:")
    assert len(result) == len("windows-sid-sha256:") + 64


class FakeTokenApi:
    def __init__(
        self,
        *,
        token: object = ...,
        sid: object = b"synthetic-sid",
        close_result: object = True,
    ) -> None:
        self.token = object() if token is ... else token
        self.sid = sid
        self.close_result = close_result
        self.events: list[str] = []

    def open_current_token(self) -> object:
        self.events.append("open")
        return self.token

    def read_token_sid(self, token: object) -> object:
        self.events.append("read")
        return self.sid

    def close_token(self, token: object) -> object:
        self.events.append("close")
        return self.close_result


def test_synthetic_token_api_closes_after_read() -> None:
    api = FakeTokenApi()

    assert _read_current_windows_sid_bytes(api) == b"synthetic-sid"
    assert api.events == ["open", "read", "close"]


def test_synthetic_token_api_rejects_missing_token_before_read() -> None:
    api = FakeTokenApi(token=None)

    with pytest.raises(SpeechChatError, match=r"^speech chat unavailable$"):
        _read_current_windows_sid_bytes(api)

    assert api.events == ["open"]


@pytest.mark.parametrize("close_result", [False, None, 1, "yes"])
def test_token_close_failure_fails_closed(close_result: object) -> None:
    api = FakeTokenApi(close_result=close_result)

    with pytest.raises(SpeechChatError, match=r"^speech chat unavailable$") as captured:
        _read_current_windows_sid_bytes(api)

    assert api.events == ["open", "read", "close"]
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize("sid", [b"", bytearray(b"sid"), None, "sid"])
def test_synthetic_token_api_rejects_invalid_sid_after_closing(sid: object) -> None:
    api = FakeTokenApi(sid=sid)

    with pytest.raises(SpeechChatError, match=r"^speech chat unavailable$"):
        _read_current_windows_sid_bytes(api)

    assert api.events == ["open", "read", "close"]


def test_space_dispatches_button_and_q_exits_with_cleanup() -> None:
    assembly = FakeAssembly()
    statuses: list[str] = []

    _run_console_loop(
        assembly=assembly,
        read_key=key_reader(" ", None, " ", "q"),
        write_status=statuses.append,
        sleep=lambda _: None,
    )

    assert assembly.controller.button_calls == 2
    assert assembly.controller.tick_calls == 4
    assert assembly.drain_calls == 4
    assert assembly.close_calls == 1
    assert statuses == [
        "speech-chat-development-mode",
        "speech-chat-ready",
        "speech-chat-stopped",
    ]


def test_unknown_keys_are_ignored_and_playback_events_keep_draining() -> None:
    assembly = FakeAssembly()

    _run_console_loop(
        assembly=assembly,
        read_key=key_reader("x", None, "Q"),
        write_status=lambda _: None,
        sleep=lambda _: None,
    )

    assert assembly.controller.button_calls == 0
    assert assembly.drain_calls == 3
    assert assembly.close_calls == 1


def test_loop_failure_closes_and_exposes_only_fixed_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    private = "PRIVATE_EXCEPTION_BODY"

    class FailingAssembly(FakeAssembly):
        def drain_playback_events(self) -> None:
            raise RuntimeError(private)

    assembly = FailingAssembly()
    with pytest.raises(SpeechChatError, match=r"^speech chat unavailable$") as captured:
        _run_console_loop(
            assembly=assembly,
            read_key=key_reader(None),
            write_status=lambda _: None,
            sleep=lambda _: None,
        )

    assert assembly.close_calls == 1
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert private not in capsys.readouterr().out + capsys.readouterr().err


@pytest.mark.parametrize("attribute", ["state", "last_issue"])
def test_missing_controller_attribute_closes_with_fixed_error(attribute: str) -> None:
    assembly = FakeAssembly()
    delattr(assembly.controller, attribute)
    statuses: list[str] = []

    with pytest.raises(SpeechChatError, match=r"^speech chat unavailable$") as captured:
        _run_console_loop(
            assembly=assembly,
            read_key=lambda: "q",
            write_status=statuses.append,
            sleep=lambda _: None,
        )

    assert assembly.close_calls == 1
    assert "speech-chat-stopped" not in statuses
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_development_mute_gate_is_explicitly_software_only() -> None:
    gate = WindowsDevelopmentMuteGate()

    assert gate.microphone_enabled() is True
    assert gate.physical_mute_verified is False


def test_main_composes_existing_runtime_and_prints_ready_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    process = Mock()
    monkeypatch.setattr(budget_notice.subprocess, "run", process)
    assembly = FakeAssembly()
    driver_factory = Mock(wraps=speech_expression.SpeechExpressionDriver)
    monkeypatch.setattr(speech_chat, "SpeechExpressionDriver", driver_factory)
    settings = object()
    config = SimpleNamespace(snippet_style=None)
    secrets = object()
    orchestrator = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(speech_chat, "load_settings", lambda _: settings)
    monkeypatch.setattr(speech_chat, "load_dialogue_config", lambda value: (config, secrets))
    monkeypatch.setattr(
        speech_chat,
        "build_real_orchestrator",
        lambda *values, **kwargs: (
            captured.update(builder_values=values, builder_kwargs=kwargs) or orchestrator
        ),
    )
    monkeypatch.setattr(speech_chat, "_windows_trusted_user_id", lambda: "trusted-user")
    monkeypatch.setattr(
        speech_chat,
        "assemble_local_speech_runtime",
        lambda **values: captured.update(assembly_values=values) or assembly,
    )
    monkeypatch.setattr(speech_chat, "_read_console_key", key_reader("q"))
    monkeypatch.setattr(speech_chat.time, "sleep", lambda _: None)

    assert speech_chat.main() == 0

    output = capsys.readouterr().out
    assert output.count("按空格开始录音，按 Q 退出。") == 1
    assert "尚未验证机器人实体断麦" in output
    assert assembly.close_calls == 1
    assert captured["builder_kwargs"] == {"trusted_user_id": "trusted-user"}
    assert captured["assembly_values"]["trusted_user_id"] == "trusted-user"  # type: ignore[index]
    assert "budget_notice" not in captured["assembly_values"]
    processing = captured["assembly_values"]["processing_observer"]
    assert processing.__func__ is speech_expression.SpeechExpressionDriver.processing
    assert isinstance(captured["assembly_values"]["mute_gate"], WindowsDevelopmentMuteGate)
    driver_factory.assert_called_once()
    assert driver_factory.call_args.args[1] is captured["assembly_values"]["mute_gate"]
    process.assert_not_called()


@pytest.mark.parametrize("mode", ["snippet", "legacy", "construction-error"])
def test_main_snippet_style_wiring(
    mode: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    process = Mock()
    monkeypatch.setattr(budget_notice.subprocess, "run", process)
    settings = object()
    config = SimpleNamespace(snippet_style=None if mode == "legacy" else object())
    secrets = object()
    adapter = object()
    habit_card = "synthetic habit card"
    orchestrator = object()
    assembly = object()
    snippet_calls = []
    builder_calls = []
    assembly_calls = []
    loop_calls = []

    def build_snippets(value, *, clock):
        snippet_calls.append(value)
        before = datetime.now(UTC)
        now = clock()
        assert now.tzinfo is UTC
        assert before <= now <= datetime.now(UTC)
        if mode == "construction-error":
            raise ValueError("synthetic construction failure")
        return adapter, habit_card

    def build_orchestrator(*values, **kwargs):
        builder_calls.append((values, kwargs))
        return orchestrator

    def assemble(**kwargs):
        assembly_calls.append(kwargs)
        return assembly

    monkeypatch.setattr(speech_chat, "load_settings", lambda _: settings)
    monkeypatch.setattr(speech_chat, "load_dialogue_config", lambda _: (config, secrets))
    monkeypatch.setattr(speech_chat, "build_real_orchestrator", build_orchestrator)
    monkeypatch.setattr(speech_chat, "build_snippet_style_runtime", build_snippets)
    monkeypatch.setattr(speech_chat, "assemble_local_speech_runtime", assemble)
    monkeypatch.setattr(
        speech_chat, "_run_console_loop", lambda **kwargs: loop_calls.append(kwargs)
    )
    monkeypatch.setattr(speech_chat, "_windows_trusted_user_id", lambda: "trusted-user")

    result = speech_chat.main()

    process.assert_not_called()
    assert snippet_calls == ([] if mode == "legacy" else [config])
    if mode == "construction-error":
        assert result == 1
        assert capsys.readouterr().out == "语音聊天暂时无法启动。\n"
        assert builder_calls == assembly_calls == loop_calls == []
    else:
        assert result == 0
        expected_kwargs = {"trusted_user_id": "trusted-user"}
        if mode == "snippet":
            expected_kwargs.update(
                snippet_style=adapter,
                snippet_habit_card=habit_card,
                privacy_revision_sources=(adapter,),
            )
        assert builder_calls == [((settings, config, secrets), expected_kwargs)]
        if mode == "snippet":
            sources = builder_calls[0][1]["privacy_revision_sources"]
            assert len(sources) == 1
            assert sources[0] is builder_calls[0][1]["snippet_style"] is adapter
        assert len(assembly_calls) == len(loop_calls) == 1
        assert assembly_calls[0]["orchestrator"] is orchestrator
        assert assembly_calls[0]["trusted_user_id"] == "trusted-user"
        assert "budget_notice" not in assembly_calls[0]
        assert loop_calls[0]["assembly"] is assembly
        processing = assembly_calls[0]["processing_observer"]
        observe = loop_calls[0]["observe"]
        assert processing.__self__ is observe.__self__
        assert processing.__func__ is speech_expression.SpeechExpressionDriver.processing
        assert observe.__func__ is speech_expression.SpeechExpressionDriver.observe


@pytest.fixture
def windows_host(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    assembly = SimpleNamespace(drain_playback_events=Mock(), close=Mock())
    assemble = Mock(return_value=assembly)
    monkeypatch.setattr(speech_chat, "load_settings", lambda _: object())
    monkeypatch.setattr(
        speech_chat, "load_dialogue_config",
        lambda _: (SimpleNamespace(snippet_style=None), object()),
    )
    monkeypatch.setattr(speech_chat, "build_real_orchestrator", lambda *args, **kwargs: object())
    monkeypatch.setattr(speech_chat, "_windows_trusted_user_id", lambda: "synthetic-user")
    monkeypatch.setattr(speech_chat, "assemble_local_speech_runtime", assemble)
    return SimpleNamespace(assembly=assembly, assemble=assemble)


def test_windows_entry_wires_expression_driver(windows_host, monkeypatch) -> None:
    driver = Mock(spec=speech_expression.SpeechExpressionDriver)
    driver_factory = Mock(return_value=driver)
    gate = WindowsDevelopmentMuteGate()
    loop = Mock()
    monkeypatch.setattr(speech_chat, "WindowsDevelopmentMuteGate", lambda: gate)
    monkeypatch.setattr(speech_chat, "SpeechExpressionDriver", driver_factory)
    monkeypatch.setattr(speech_chat, "_run_console_loop", loop)

    assert speech_chat.main() == 0

    driver_factory.assert_called_once()
    expression, driver_gate = driver_factory.call_args.args
    assert isinstance(expression, ExpressionController)
    assert driver_gate is windows_host.assemble.call_args.kwargs["mute_gate"] is gate
    assert windows_host.assemble.call_args.kwargs["processing_observer"] is driver.processing
    assert loop.call_args.kwargs["observe"] is driver.observe


def _run_windows_synthetic_turn(
    host, monkeypatch, *, recording_limit: bool, mute_at_stop: bool = False,
    display_failure: bool = False,
) -> None:
    clock = ClockFake(now=0.0)
    recorder = AudioRecorderFake()
    recognizer = SpeechRecognizerFake()
    dialogue = DialogueFake()
    synthesizer = SpeechSynthesizerFake()
    player = AudioPlayerFake()
    gate = MuteGateFake()
    faces = []
    events = []
    processing = speech_expression.SpeechExpressionDriver.processing

    def notify(driver):
        events.append("processing")
        processing(driver)

    def closed():
        events.append("closed")
        if mute_at_stop:
            gate.enabled = False

    def recognize():
        events.append("recognize")
        assert events == ["stopped", "closed", "processing", "recognize"]
        if not display_failure:
            assert faces[-1] is (Face.MUTED if mute_at_stop else Face.THINKING)

    def show(self, face):
        faces.append(face)
        if display_failure:
            raise RuntimeError("synthetic display failure")

    recorder.configure_callback("session.stop", lambda: events.append("stopped"))
    recorder.configure_callback("session.close", closed)
    recognizer.configure_callback("recognizer.recognize", recognize)
    monkeypatch.setattr(speech_expression.SpeechExpressionDriver, "processing", notify)
    monkeypatch.setattr(speech_expression.TerminalFaceDisplay, "show", show)
    monkeypatch.setattr(speech_chat, "WindowsDevelopmentMuteGate", lambda: gate)
    monkeypatch.setattr(speech_chat, "_SystemClock", lambda: clock)

    def assemble(**kwargs):
        controller = SpeechController(
            trusted_user_id=kwargs["trusted_user_id"], recorder=recorder, recognizer=recognizer,
            dialogue=dialogue, synthesizer=synthesizer, player=player,
            mute_gate=kwargs["mute_gate"], clock=kwargs["clock"],
            processing_observer=kwargs.get("processing_observer"),
        )
        host.assembly.controller = controller
        host.assembly.close.side_effect = controller.shutdown
        return host.assembly

    host.assemble.side_effect = assemble
    keys = (" ", None, "Q") if recording_limit else (" ", " ", "Q")
    monkeypatch.setattr(speech_chat, "_read_console_key", key_reader(*keys))
    times = iter([29.9, 30.0] if recording_limit else [0.5, 0.5])

    def advance_clock(_):
        if recording_limit:
            assert recognizer.calls == 0
        clock.now = next(times)

    monkeypatch.setattr(speech_chat.time, "sleep", advance_clock)

    assert speech_chat.main() == 0
    assert events == ["stopped", "closed", "processing", "recognize"]
    assert recognizer.calls == 1
    assert dialogue.calls == [("synthetic-user", "synthetic transcript")]
    assert synthesizer.inputs == ["synthetic safe response"]
    assert player.tokens == [1]
    assert player.stop_calls == 1
    assert recorder.live_session_count == recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0
    host.assembly.close.assert_called_once_with()
    if display_failure:
        assert Face.THINKING in faces


def test_windows_manual_stop_notifies_expression_once(windows_host, monkeypatch) -> None:
    _run_windows_synthetic_turn(windows_host, monkeypatch, recording_limit=False)


def test_windows_recording_limit_shows_thinking_before_recognition(
    windows_host, monkeypatch,
) -> None:
    _run_windows_synthetic_turn(windows_host, monkeypatch, recording_limit=True)


@pytest.mark.parametrize("recording_limit", [False, True])
def test_windows_processing_notification_respects_mute(
    windows_host, monkeypatch, recording_limit: bool,
) -> None:
    _run_windows_synthetic_turn(
        windows_host, monkeypatch, recording_limit=recording_limit, mute_at_stop=True
    )


def test_windows_expression_failure_keeps_speech_running(windows_host, monkeypatch) -> None:
    _run_windows_synthetic_turn(
        windows_host, monkeypatch, recording_limit=False, display_failure=True
    )
