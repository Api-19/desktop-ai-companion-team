# ruff: noqa: RUF001

from __future__ import annotations

import ctypes
import hashlib
import sys
import time
from collections.abc import Callable
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from companion.dialogue.composition import build_real_orchestrator, build_snippet_style_runtime
from companion.dialogue.config import load_dialogue_config
from companion.expression import ExpressionController
from companion.runtime.settings import load_settings
from companion.runtime.speech_expression import (
    NullHeadServo,
    SpeechExpressionDriver,
    TerminalFaceDisplay,
)
from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechControllerIssue, SpeechState, SpeechTurnFailure
from companion.speech.runtime_assembly import assemble_local_speech_runtime

_FAILURE_MESSAGE = "speech chat unavailable"
_POLL_SECONDS = 0.05


class SpeechChatError(RuntimeError):
    """Fixed, content-free failure at the Windows console host boundary."""


def _raise_unavailable() -> None:
    raise SpeechChatError(_FAILURE_MESSAGE) from None


def _trusted_user_id_from_sid_bytes(value: object) -> str:
    if type(value) is not bytes or not value:
        value = None
        _raise_unavailable()
    digest = hashlib.sha256(value).hexdigest()
    value = None
    return f"windows-sid-sha256:{digest}"


class _WindowsTokenApi(Protocol):
    def open_current_token(self) -> object: ...

    def read_token_sid(self, token: object) -> object: ...

    def close_token(self, token: object) -> object: ...


class _CtypesWindowsTokenApi:
    def __init__(self) -> None:
        if sys.platform != "win32":
            _raise_unavailable()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

        class _SidAndAttributes(ctypes.Structure):
            _fields_ = (("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD))

        class _TokenUser(ctypes.Structure):
            _fields_ = (("user", _SidAndAttributes),)

        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        )
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.GetLengthSid.argtypes = (ctypes.c_void_p,)
        advapi32.GetLengthSid.restype = wintypes.DWORD
        self._kernel32 = kernel32
        self._advapi32 = advapi32
        self._token_user_type = _TokenUser

    def open_current_token(self) -> object:
        token = wintypes.HANDLE()
        if not self._advapi32.OpenProcessToken(
            self._kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
        ):
            raise OSError
        return token

    def read_token_sid(self, token: object) -> object:
        required = wintypes.DWORD(0)
        self._advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(required))
        if required.value <= 0:
            raise OSError
        buffer = ctypes.create_string_buffer(required.value)
        if not self._advapi32.GetTokenInformation(
            token, 1, buffer, required.value, ctypes.byref(required)
        ):
            raise OSError
        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(self._token_user_type)).contents.user.sid
        if sid_pointer is None:
            raise OSError
        sid_length = self._advapi32.GetLengthSid(ctypes.c_void_p(sid_pointer))
        if type(sid_length) is not int or sid_length <= 0 or sid_length > 1024:
            raise OSError
        return ctypes.string_at(sid_pointer, sid_length)

    def close_token(self, token: object) -> object:
        return self._kernel32.CloseHandle(token) != 0


def _read_current_windows_sid_bytes(api: _WindowsTokenApi | None = None) -> bytes:
    token = None
    result: object = None
    selected_api: _WindowsTokenApi | None = None
    failed = False
    try:
        selected_api = _CtypesWindowsTokenApi() if api is None else api
        token = selected_api.open_current_token()
        if token is None:
            raise TypeError
        result = selected_api.read_token_sid(token)
    except Exception:
        failed = True
    finally:
        if token is not None and selected_api is not None:
            try:
                if selected_api.close_token(token) is not True:
                    failed = True
            except Exception:
                failed = True
        token = None
        selected_api = None  # type: ignore[assignment]
    if failed or type(result) is not bytes or not result:
        result = None
        _raise_unavailable()
    return result  # type: ignore[return-value]


def _windows_trusted_user_id() -> str:
    return _trusted_user_id_from_sid_bytes(_read_current_windows_sid_bytes())


class WindowsDevelopmentMuteGate:
    """Software-only Windows development gate; not a physical mute attestation."""

    physical_mute_verified = False

    def microphone_enabled(self) -> bool:
        return True


class _ConsoleAssembly(Protocol):
    @property
    def controller(self) -> SpeechController: ...

    def drain_playback_events(self) -> None: ...

    def close(self) -> None: ...


def _run_console_loop(
    *,
    assembly: _ConsoleAssembly,
    read_key: Callable[[], str | None],
    write_status: Callable[[str], None],
    sleep: Callable[[float], None],
    observe: Callable[[SpeechController], None] | None = None,
) -> None:
    failed = False
    try:
        controller = assembly.controller
        write_status("speech-chat-development-mode")
        write_status("speech-chat-ready")
        observed_state = controller.state
        observed_issue = controller.last_issue
        observed_turn_failure = controller.turn_failure
        if observe is not None:
            observe(controller)
        while True:
            controller.tick()
            assembly.drain_playback_events()
            observed_state, observed_issue, observed_turn_failure = _report_console_state(
                controller, observed_state, observed_issue, observed_turn_failure, write_status
            )
            if observe is not None:
                observe(controller)
            key = read_key()
            if key in ("q", "Q"):
                break
            if key == " ":
                state = controller.state
                if state is SpeechState.RECORDING:
                    write_status("speech-chat-processing")
                controller.button_pressed()
                observed_state, observed_issue, observed_turn_failure = _report_console_state(
                    controller, observed_state, observed_issue, observed_turn_failure,
                    write_status, after_button=True
                )
                if observe is not None:
                    observe(controller)
            sleep(_POLL_SECONDS)
    except Exception:
        failed = True
    finally:
        try:
            assembly.close()
        except Exception:
            failed = True
    if failed:
        assembly = None  # type: ignore[assignment]
        read_key = None  # type: ignore[assignment]
        write_status = None  # type: ignore[assignment]
        sleep = None  # type: ignore[assignment]
        _raise_unavailable()
    write_status("speech-chat-stopped")


def _report_console_state(
    controller: SpeechController,
    previous_state: object,
    previous_issue: object,
    previous_turn_failure: SpeechTurnFailure | None,
    write_status: Callable[[str], None],
    *,
    after_button: bool = False,
) -> tuple[object, object, SpeechTurnFailure | None]:
    turn_failure = controller.turn_failure
    if turn_failure is not None and turn_failure != previous_turn_failure:
        write_status("speech-chat-" + turn_failure.value)
    state = controller.state
    issue = controller.last_issue
    if type(issue) is SpeechControllerIssue and (issue is not previous_issue or after_button):
        write_status(f"speech-chat-{issue.value}")
    if state is not previous_state:
        if state is SpeechState.IDLE:
            write_status("speech-chat-ready")
        elif state is SpeechState.RECORDING:
            write_status("speech-chat-recording")
        elif state is SpeechState.PLAYING:
            write_status("speech-chat-playing")
    return state, issue, turn_failure


_STATUS_TEXT = {
    "speech-chat-development-mode": "Windows 开发模式：尚未验证机器人实体断麦。",
    "speech-chat-ready": "按空格开始录音，按 Q 退出。",
    "speech-chat-recording": "正在录音，再按空格结束。",
    "speech-chat-processing": "正在处理。",
    "speech-chat-playing": "正在播放；按空格可打断并开始下一轮。",
    "speech-chat-stopped": "已退出。",
    "speech-chat-recording-failed": "录音失败。",
    "speech-chat-recognition-rejected": "语音未通过本地检查，请重新录音。",
    "speech-chat-recognition-failed": "语音识别失败。",
    "speech-chat-dialogue-failed": "对话连接失败。",
    "speech-chat-synthesis-failed": "语音合成失败。",
    "speech-chat-playback-failed": "语音播放失败。",
    "speech-chat-cleanup-failed": "语音资源清理失败。",
    "speech-chat-provider-timeout": "云端回复超时。",
    "speech-chat-provider-unavailable": "云端聊天服务暂时不可用。",
    "speech-chat-invalid-provider-response": "云端回复无效。",
}


def _read_console_key() -> str | None:
    import msvcrt

    return msvcrt.getwch() if msvcrt.kbhit() else None


def _write_console_status(status: str) -> None:
    text = _STATUS_TEXT.get(status)
    if text is not None:
        print(text, flush=True)


def main() -> int:
    try:
        settings = load_settings(Path.cwd())
        config, secrets = load_dialogue_config(settings)
        trusted_user_id = _windows_trusted_user_id()
        if config.snippet_style is not None:
            adapter, habit_card = build_snippet_style_runtime(
                config, clock=lambda: datetime.now(UTC)
            )
            orchestrator = build_real_orchestrator(
                settings,
                config,
                secrets,
                trusted_user_id=trusted_user_id,
                snippet_style=adapter,
                snippet_habit_card=habit_card,
                privacy_revision_sources=(adapter,),
            )
        else:
            orchestrator = build_real_orchestrator(
                settings,
                config,
                secrets,
                trusted_user_id=trusted_user_id,
            )
        mute_gate = WindowsDevelopmentMuteGate()
        expression = SpeechExpressionDriver(
            ExpressionController(TerminalFaceDisplay(), NullHeadServo()), mute_gate
        )
        assembly = assemble_local_speech_runtime(
            settings=settings,
            orchestrator=orchestrator,
            trusted_user_id=trusted_user_id,
            mute_gate=mute_gate,
            clock=_SystemClock(),
            processing_observer=expression.processing,
        )
        _run_console_loop(
            assembly=assembly,
            read_key=_read_console_key,
            write_status=_write_console_status,
            sleep=time.sleep,
            observe=expression.observe,
        )
        return 0
    except Exception:
        print("语音聊天暂时无法启动。", flush=True)
        return 1


class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()


if __name__ == "__main__":
    raise SystemExit(main())
