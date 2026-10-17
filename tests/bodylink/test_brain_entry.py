# ruff: noqa: RUF001
"""Brain entry loop (speech_chat_body) + body process core (BodyCore) over real bodylink."""

import logging
import threading
import time

import pytest

import companion.speech.runtime_assembly as assembly_module
from companion.bodylink import body_main
from companion.bodylink.body_main import BodyCore, DevelopmentMuteSource
from companion.bodylink.brain_ports import BrainBodyLink
from companion.bodylink.pairing import brain_ssl_context
from companion.contracts.body import MuteState, ShowState
from companion.expression import HeadMotion
from companion.runtime import speech_chat_body
from companion.runtime.speech_chat_body import FixedReplyDialogue
from companion.runtime.speech_expression import BodyDisplayPort, SpeechExpressionDriver
from companion.speech.controller_models import SpeechState
from companion.speech.runtime_assembly import assemble_body_speech_runtime

from .conftest import BODY_ID
from .test_body_main import (
    SYNTH_PCM,
    Body,
    CheckingRecognizer,
    FakeSynthesizer,
    MonotonicClock,
    SwitchMute,
    pcm_ramp,
    wait_until,
)


def _patch_speech_models(monkeypatch: pytest.MonkeyPatch, spoken: bytes) -> CheckingRecognizer:
    recognizer = CheckingRecognizer(spoken)
    for name, value in {
        "_load_runtime_config": lambda settings: object(),
        "_make_model_config": lambda settings, validated: object(),
        "_verify_model_tree": lambda config: None,
        "_construct_recognizer": lambda config: recognizer,
        "_construct_synthesizer": lambda config: FakeSynthesizer(),
        "_close_recognizer": lambda owner: None,
    }.items():
        monkeypatch.setattr(assembly_module, name, value)
    return recognizer


def _start_brain_loop(assembly, link, expression):
    quit_requested = threading.Event()
    failures: list[Exception] = []

    def loop() -> None:
        try:
            speech_chat_body._run_body_loop(
                assembly=assembly,
                link=link,
                expression=expression,
                read_key=lambda: "q" if quit_requested.is_set() else None,
                write_status=speech_chat_body._write_console_status,
                sleep=lambda _: time.sleep(0.01),
            )
        except Exception as error:  # surface loop failures in the test thread
            failures.append(error)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return quit_requested, failures, thread


def test_fixed_reply_turn_from_body_button_to_body_playback(
    monkeypatch: pytest.MonkeyPatch, pairing, closer, tmp_path, capsys
) -> None:
    # monkeypatch comes first so the seams outlive ``closer`` teardown (assembly.close).
    run_root = tmp_path / "run"
    run_root.mkdir()
    spoken = pcm_ramp(16_000)
    recognizer = _patch_speech_models(monkeypatch, spoken)
    shown: list[str] = []
    on_message = BodyCore.on_message

    def record_show_state(self: BodyCore, message: object) -> None:
        if isinstance(message, ShowState):
            shown.append(message.event.state)
        on_message(self, message)

    monkeypatch.setattr(BodyCore, "on_message", record_show_state)

    link = BrainBodyLink(
        host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=BODY_ID, run_root=run_root,
    )  # fmt: skip
    closer(link)
    expression = SpeechExpressionDriver(BodyDisplayPort(link), link.mute_gate)
    assembly = assemble_body_speech_runtime(
        settings=object(),  # type: ignore[arg-type]
        dialogue=FixedReplyDialogue("合成固定回复"),
        trusted_user_id="synthetic-owner",
        recorder=link.recorder,
        player=link.player,
        mute_gate=link.mute_gate,
        clock=MonotonicClock(),
        session_id_supplier=lambda: link.current_session_id,
        session_playback_starter=link.player.start_for_session,
        processing_observer=expression.processing,
    )
    closer(assembly)
    link.start()
    body = Body(link.port, pairing, input_rate=16000, resend_s=1.0, mute=DevelopmentMuteSource())
    closer(body)

    quit_requested, failures, thread = _start_brain_loop(assembly, link, expression)
    controller = assembly.controller
    wait_until(link.mute_gate.microphone_enabled)
    wait_until(lambda: shown[-1:] == ["idle"])
    body.core.button_pressed()
    wait_until(lambda: controller.state is SpeechState.RECORDING)
    stream = body.wait_input()
    for offset in range(0, len(spoken), 3200):
        stream.feed(spoken[offset : offset + 3200])
    body.core.button_pressed()
    wait_until(lambda: body.runtime.outputs and body.runtime.outputs[0].played() == SYNTH_PCM)
    wait_until(lambda: controller.state is SpeechState.IDLE and shown[-1:] == ["idle"])
    quit_requested.set()
    thread.join(5.0)

    assert failures == []
    assert not thread.is_alive()
    assert controller.last_issue is None
    assert recognizer.matched
    assert list(run_root.iterdir()) == []
    turn = shown[shown.index("listening") :]
    assert turn == ["listening", "thinking", "speaking", "idle"]
    lines = capsys.readouterr().out.splitlines()
    assert lines.count("身体为开发模式：未接物理断麦开关") == 1
    assert "正在录音，再按身体按键结束。" in lines


_FAULT_CANARY = "synthetic-body-servo-fault-canary"


class RecordingFaultServo:
    def __init__(self, error_type: type[Exception], *, disable_fails: bool) -> None:
        self.error_type = error_type
        self.disable_fails = disable_fails
        self.motions: list[HeadMotion] = []
        self.disable_calls = 0
        self.call_threads: list[str] = []

    def perform(self, motion: HeadMotion) -> None:
        self.motions.append(motion)
        self.call_threads.append(threading.current_thread().name)
        if len(self.motions) == 1:
            raise self.error_type(f"{_FAULT_CANARY}: perform")

    def disable(self) -> None:
        self.disable_calls += 1
        self.call_threads.append(threading.current_thread().name)
        if self.disable_fails:
            raise self.error_type(f"{_FAULT_CANARY}: disable")


@pytest.mark.parametrize("error_type", [TimeoutError, OSError, RuntimeError])
@pytest.mark.parametrize("disable_fails", [False, True])
def test_body_servo_fault_keeps_turn_and_worker_alive(
    monkeypatch: pytest.MonkeyPatch,
    pairing,
    closer,
    tmp_path,
    capsys,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
    disable_fails: bool,
) -> None:
    caplog.set_level(logging.DEBUG)
    servo = RecordingFaultServo(error_type, disable_fails=disable_fails)
    monkeypatch.setattr(body_main, "NullHeadServo", lambda: servo)
    run_root = tmp_path / "run"
    run_root.mkdir()
    spoken = pcm_ramp(16_000)
    recognizer = _patch_speech_models(monkeypatch, spoken)
    shown: list[ShowState] = []
    sessions: list[str] = []
    ended: list[str] = []
    on_message = BodyCore.on_message
    on_session_started = BodyCore.on_session_started
    on_session_ended = BodyCore.on_session_ended

    def record_show_state(self: BodyCore, message: object) -> None:
        if isinstance(message, ShowState):
            shown.append(message)
        on_message(self, message)

    def record_session_started(self: BodyCore, session_id: str) -> None:
        sessions.append(session_id)
        on_session_started(self, session_id)

    def record_session_ended(self: BodyCore, session_id: str, reason: str) -> None:
        ended.append(session_id)
        on_session_ended(self, session_id, reason)

    monkeypatch.setattr(BodyCore, "on_message", record_show_state)
    monkeypatch.setattr(BodyCore, "on_session_started", record_session_started)
    monkeypatch.setattr(BodyCore, "on_session_ended", record_session_ended)
    link = BrainBodyLink(
        host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=BODY_ID, run_root=run_root,
    )  # fmt: skip
    closer(link)
    mute_states: list[MuteState] = []
    drain_body_inputs = link.drain_body_inputs

    def record_body_inputs():
        items = drain_body_inputs()
        mute_states.extend(item for item in items if isinstance(item, MuteState))
        return items

    monkeypatch.setattr(link, "drain_body_inputs", record_body_inputs)
    expression = SpeechExpressionDriver(BodyDisplayPort(link), link.mute_gate)
    assembly = assemble_body_speech_runtime(
        settings=object(),  # type: ignore[arg-type]
        dialogue=FixedReplyDialogue("合成固定回复"),
        trusted_user_id="synthetic-owner",
        recorder=link.recorder,
        player=link.player,
        mute_gate=link.mute_gate,
        clock=MonotonicClock(),
        session_id_supplier=lambda: link.current_session_id,
        session_playback_starter=link.player.start_for_session,
        processing_observer=expression.processing,
    )
    closer(assembly)
    link.start()
    mute = SwitchMute()
    body = Body(link.port, pairing, input_rate=16000, resend_s=1.0, mute=mute)
    closer(body)
    quit_requested, failures, thread = _start_brain_loop(assembly, link, expression)
    controller = assembly.controller
    try:
        wait_until(link.mute_gate.microphone_enabled)
        wait_until(lambda: shown and shown[-1].event.state == "idle")
        session_id = sessions[0]
        for count in (1, 2):
            shown_start = len(shown)
            body.core.button_pressed()
            wait_until(lambda: controller.state is SpeechState.RECORDING)
            stream = body.wait_input(count=count)
            for offset in range(0, len(spoken), 3200):
                stream.feed(spoken[offset : offset + 3200])
            body.core.button_pressed()
            wait_until(lambda count=count: len(body.runtime.outputs) == count)
            output = body.runtime.outputs[count - 1]
            wait_until(lambda output=output: output.closed)
            wait_until(
                lambda shown_start=shown_start: controller.state is SpeechState.IDLE
                and len(shown) >= shown_start + 4
                and shown[-1].event.state == "idle"
            )
            assert output.played() == SYNTH_PCM
            assert output.closed
            assert recognizer.matched
            assert controller.last_issue is None
            assert list(run_root.iterdir()) == []
            assert [message.event.state for message in shown[shown_start:]] == [
                "listening", "thinking", "speaking", "idle"
            ]
            assert servo.motions == [HeadMotion.TILT]
            assert servo.disable_calls == 1

        # A fresh changed MuteState must traverse the body worker and real link after playback.
        mute_start = len(mute_states)
        mute.enabled = False
        wait_until(lambda: len(mute_states) > mute_start)
        assert mute_states[-1].microphone_enabled is False
        assert {message.session_id for message in [*shown, *mute_states]} == {session_id}
        assert sessions == [session_id]
        assert ended == []
        assert len(body.runtime.inputs) == 2
        assert len(body.runtime.outputs) == 2
        assert servo.call_threads == ["body-core", "body-core"]
        assert controller.state is SpeechState.IDLE
        assert controller.last_issue is None
        assert list(run_root.iterdir()) == []
    finally:
        quit_requested.set()
        thread.join(5.0)

    assert not thread.is_alive()
    assert failures == []
    captured = capsys.readouterr()
    assert _FAULT_CANARY not in captured.out
    assert _FAULT_CANARY not in captured.err
    assert _FAULT_CANARY not in caplog.text
