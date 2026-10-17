# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from companion.dialogue.composition import build_real_orchestrator, build_snippet_style_runtime
from companion.dialogue.config import load_dialogue_config
from companion.dialogue.observer import TurnObservation, TurnObserver
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.expression import ExpressionController
from companion.memory.private_storage import build_private_memory_service
from companion.runtime.budget_notice import MacBudgetNotifier
from companion.runtime.settings import Settings, load_settings
from companion.runtime.speech_chat import (
    SpeechChatError,
    _run_console_loop,
    _SystemClock,
)
from companion.runtime.speech_chat import _write_console_status as _write_shared_console_status
from companion.runtime.speech_expression import (
    NullHeadServo,
    SpeechExpressionDriver,
    TerminalFaceDisplay,
)
from companion.speech.controller_models import TurnTiming
from companion.speech.runtime_assembly import assemble_local_speech_runtime


class MacDevelopmentMuteGate:
    """Software-only Mac development gate; not a physical mute attestation."""

    physical_mute_verified = False

    def microphone_enabled(self) -> bool:
        return True


@contextmanager
def _cbreak_console() -> Iterator[None]:
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _read_console_key() -> str | None:
    fd = sys.stdin.fileno()
    readable, _, _ = select.select([fd], [], [], 0)
    # Read the descriptor directly so buffered text input cannot hide queued keys from select.
    return os.read(fd, 1).decode("ascii", errors="ignore") if readable else None


def _write_console_status(status: str) -> None:
    if status == "speech-chat-development-mode":
        print("Mac 开发模式：尚未验证机器人实体断麦。", flush=True)
    else:
        _write_shared_console_status(status)


def _open_debug_transcript(
    settings: Settings, resources: ExitStack
) -> tuple[Path, TurnObserver, Callable[[TurnTiming], None]]:
    """Private 0600 diagnostic-only JSONL, closed with ``resources``."""
    directory = settings.private_data_dir / "speech" / "run" / "debug-transcripts"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    debug_path = directory / f"transcript-{uuid4().hex}.jsonl"
    fd = os.open(
        debug_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    debug_file = resources.enter_context(os.fdopen(fd, "w", encoding="utf-8"))

    def observe_turn(observation: TurnObservation) -> None:
        record = {
            "observed_at": observation.observed_at.isoformat(),
            "blocked_reason": observation.blocked_reason,
            "input_chars": observation.input_chars,
            "context_count": observation.context_count,
            "context_chars": observation.context_chars,
            "reply_chars": observation.reply_chars,
            "similar_count": observation.similar_count,
            "general_count": observation.general_count,
            "profile_count": observation.profile_count,
            "provider_attempted": observation.provider_attempted,
            "provider_failure": observation.provider_failure,
            "provider_elapsed_ms": observation.provider_elapsed_ms,
        }
        debug_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        debug_file.flush()

    def observe_timing(timing: TurnTiming) -> None:
        record = {
            "kind": "speech_timing",
            "observed_at": datetime.now(UTC).isoformat(),
            "played": timing.played,
            "issue": timing.issue,
            "stop_ms": timing.stop_ms,
            "recognize_ms": timing.recognize_ms,
            "dialogue_ms": timing.dialogue_ms,
            "synthesize_ms": timing.synthesize_ms,
            "playback_start_ms": timing.playback_start_ms,
            "total_ms": timing.total_ms,
        }
        debug_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        debug_file.flush()

    return debug_path, observe_turn, observe_timing


def _build_orchestrator(
    settings: Settings, trusted_user_id: str, turn_observer: TurnObserver | None
) -> tuple[DialogueOrchestrator, MacBudgetNotifier]:
    """Formal dialogue assembly: snippet style and private profile when configured."""
    config, secrets = load_dialogue_config(settings)
    if config.snippet_style is None:
        orchestrator = build_real_orchestrator(
            settings,
            config,
            secrets,
            trusted_user_id=trusted_user_id,
            turn_observer=turn_observer,
        )
    else:
        adapter, habit_card = build_snippet_style_runtime(config, clock=lambda: datetime.now(UTC))
        profile_source = build_private_memory_service(settings, clock=lambda: datetime.now(UTC))
        orchestrator = build_real_orchestrator(
            settings,
            config,
            secrets,
            trusted_user_id=trusted_user_id,
            snippet_style=adapter,
            snippet_habit_card=habit_card,
            profile_source=profile_source,
            turn_observer=turn_observer,
            privacy_revision_sources=(profile_source, adapter),
        )
    notifier = MacBudgetNotifier(
        ledger=MonthlyUsageLedger(settings.runtime_data_dir / "dialogue", config.budget),
        clock=lambda: datetime.now(UTC),
    )
    return orchestrator, notifier


def main(argv: list[str] | None = None) -> int:
    try:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
        parser.add_argument("--debug-log", action="store_true")
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise SpeechChatError("unknown arguments")
        if sys.platform != "darwin" or not sys.stdin.isatty():
            raise SpeechChatError("speech chat unavailable")
        with _cbreak_console(), ExitStack() as resources:
            settings = load_settings(Path.cwd())
            debug_path = None
            turn_observer: TurnObserver | None = None
            timing_observer: Callable[[TurnTiming], None] | None = None
            if args.debug_log:
                debug_path, turn_observer, timing_observer = _open_debug_transcript(
                    settings, resources
                )

            mute_gate = MacDevelopmentMuteGate()
            expression = SpeechExpressionDriver(
                ExpressionController(TerminalFaceDisplay(), NullHeadServo()), mute_gate
            )

            def write_status(status: str) -> None:
                _write_console_status(status)
                if status == "speech-chat-development-mode" and debug_path is not None:
                    print("本地调试记录已开启：" + str(debug_path), flush=True)

            trusted_user_id = f"posix-uid-{os.getuid()}"
            orchestrator, notifier = _build_orchestrator(settings, trusted_user_id, turn_observer)
            assembly = assemble_local_speech_runtime(
                settings=settings,
                orchestrator=orchestrator,
                trusted_user_id=trusted_user_id,
                mute_gate=mute_gate,
                clock=_SystemClock(),
                timing_observer=timing_observer,
                processing_observer=expression.processing,
                budget_notice=notifier.check,
            )
            notifier.check()
            _run_console_loop(
                assembly=assembly,
                read_key=_read_console_key,
                write_status=write_status,
                sleep=time.sleep,
                observe=expression.observe,
            )
        return 0
    except Exception:
        print("语音聊天暂时无法启动。", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
