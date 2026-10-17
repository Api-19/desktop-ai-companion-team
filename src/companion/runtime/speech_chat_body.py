# ruff: noqa: RUF001
"""Brain entry for the robot body: ``python -m companion.runtime.speech_chat_body`` (Mac).

Recognition, dialogue and synthesis run here as in ``speech_chat_mac``; recording, playback,
the button and the mute switch live on the body and arrive over bodylink. Expression states
go back to the body as ShowState. ``--fixed-reply <text>`` replaces the dialogue with a fixed
answer (no dialogue config, no keys, no paid request).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Protocol

from companion.bodylink.brain_ports import BodyInput, BrainBodyLink
from companion.bodylink.pairing import brain_ssl_context
from companion.contracts import DialogueResponse
from companion.contracts.body import ButtonPressed
from companion.contracts.robot import RobotEvent
from companion.dialogue.observer import TurnObserver
from companion.runtime import speech_chat_mac
from companion.runtime.settings import load_settings
from companion.runtime.speech_chat import (
    _POLL_SECONDS,
    _STATUS_TEXT,
    SpeechChatError,
    _report_console_state,
    _SystemClock,
)
from companion.runtime.speech_chat_mac import _cbreak_console, _read_console_key
from companion.runtime.speech_dialogue import SpeechDialogueBridge
from companion.runtime.speech_expression import BodyDisplayPort, SpeechExpressionDriver
from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechState, TurnTiming
from companion.speech.controller_ports import DialogueCallable, MuteGate
from companion.speech.runtime_assembly import assemble_body_speech_runtime, speech_run_root

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_BODY_ID = "kiki-body"

_BODY_STATUS_TEXT = {
    "speech-chat-ready": "等待身体按键开始录音，按 Q 退出。",
    "speech-chat-recording": "正在录音，再按身体按键结束。",
    "speech-chat-playing": "正在播放；按身体按键可打断并开始下一轮。",
    "bodylink-development-mute": "身体为开发模式：未接物理断麦开关",
}


class FixedReplyDialogue:
    """Development dialogue: every turn answers ``text``; nothing leaves the machine."""

    def __init__(self, text: str) -> None:
        self._text = text

    def reply(self, *, user_id: str, text: str) -> DialogueResponse:
        return DialogueResponse(
            text=self._text, disclosed_ai_identity=False, billable=False, blocked_reason=None
        )


class _Assembly(Protocol):
    @property
    def controller(self) -> SpeechController: ...

    def drain_playback_events(self) -> None: ...


class _Link(Protocol):
    mute_gate: MuteGate

    @property
    def current_session_id(self) -> str | None: ...

    def drain_body_inputs(self) -> tuple[BodyInput, ...]: ...

    def show_state(self, event: RobotEvent) -> None: ...


def _write_console_status(status: str) -> None:
    text = _BODY_STATUS_TEXT.get(status) or _STATUS_TEXT.get(status)
    if text is not None:
        print(text, flush=True)


def _run_body_loop(
    *,
    assembly: _Assembly,
    link: _Link,
    expression: SpeechExpressionDriver,
    read_key: Callable[[], str | None],
    write_status: Callable[[str], None],
    sleep: Callable[[float], None],
) -> None:
    controller = assembly.controller
    write_status("speech-chat-ready")
    state = controller.state
    issue = controller.last_issue
    turn_failure = controller.turn_failure
    expression.observe(controller)
    development_reported = False
    while True:
        for item in link.drain_body_inputs():
            if item.session_id != link.current_session_id:
                continue
            if isinstance(item, ButtonPressed):
                if controller.state is SpeechState.RECORDING:
                    write_status("speech-chat-processing")
                controller.button_pressed()
                state, issue, turn_failure = _report_console_state(
                    controller, state, issue, turn_failure, write_status, after_button=True
                )
                expression.observe(controller)
            else:
                if item.source == "development" and not development_reported:
                    write_status("bodylink-development-mute")
                    development_reported = True
                controller.mute_changed()
        controller.tick()
        assembly.drain_playback_events()
        state, issue, turn_failure = _report_console_state(
            controller, state, issue, turn_failure, write_status
        )
        expression.observe(controller)
        if read_key() in ("q", "Q"):
            break
        sleep(_POLL_SECONDS)
    write_status("speech-chat-stopped")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--debug-log", action="store_true")
    parser.add_argument("--fixed-reply")
    parser.add_argument("--cert-dir", type=Path)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--body-id", default=DEFAULT_BODY_ID)
    args, unknown = parser.parse_known_args(argv)
    if unknown or (args.fixed_reply is not None and args.debug_log):
        raise SpeechChatError("invalid arguments")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if sys.platform != "darwin" or not sys.stdin.isatty():
            raise SpeechChatError("speech chat unavailable")
        with _cbreak_console(), ExitStack() as resources:
            settings = load_settings(Path.cwd())
            trusted_user_id = f"posix-uid-{os.getuid()}"
            dialogue: DialogueCallable
            timing_observer: Callable[[TurnTiming], None] | None = None
            notifier = None
            if args.fixed_reply is not None:
                dialogue = FixedReplyDialogue(args.fixed_reply)
            else:
                turn_observer: TurnObserver | None = None
                if args.debug_log:
                    debug_path, turn_observer, timing_observer = (
                        speech_chat_mac._open_debug_transcript(settings, resources)
                    )
                    print("本地调试记录已开启：" + str(debug_path), flush=True)
                orchestrator, notifier = speech_chat_mac._build_orchestrator(
                    settings, trusted_user_id, turn_observer
                )
                dialogue = SpeechDialogueBridge(
                    orchestrator=orchestrator,
                    privacy_revision=orchestrator.privacy_revision,
                    budget_notice=notifier.check,
                )
            cert_dir = args.cert_dir or settings.private_data_dir.parent / "bodylink" / "brain"
            run_root = speech_run_root(settings)
            link = BrainBodyLink(
                host=args.host,
                port=args.port,
                ssl_context=brain_ssl_context(cert_dir),
                body_id=args.body_id,
                run_root=run_root,
            )
            resources.callback(link.close)
            expression = SpeechExpressionDriver(BodyDisplayPort(link), link.mute_gate)
            assembly = assemble_body_speech_runtime(
                settings=settings,
                dialogue=dialogue,
                trusted_user_id=trusted_user_id,
                recorder=link.recorder,
                player=link.player,
                mute_gate=link.mute_gate,
                clock=_SystemClock(),
                timing_observer=timing_observer,
                session_id_supplier=lambda: link.current_session_id,
                session_playback_starter=link.player.start_for_session,
                processing_observer=expression.processing,
            )
            resources.callback(assembly.close)
            link.start()
            if notifier is not None:
                notifier.check()
            print(f"大脑已启动：等待身体连接（端口 {link.port}）。", flush=True)
            _run_body_loop(
                assembly=assembly,
                link=link,
                expression=expression,
                read_key=_read_console_key,
                write_status=_write_console_status,
                sleep=time.sleep,
            )
        return 0
    except Exception:
        print("语音聊天暂时无法启动。", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
