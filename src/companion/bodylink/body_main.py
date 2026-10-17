# ruff: noqa: RUF001
"""Body process: ``python -m companion.bodylink.body_main``.

Captures and plays audio, reads the button, reports the mute switch and shows faces, driven by
the brain over bodylink. Audio lives only in memory; nothing is written to disk.

Threading: callbacks enqueue session-tagged business events or invalidate the session under a
condition. One worker owns device state and sends; playback has a thread per utterance.
"""

import argparse
import os
import select
import signal
import sys
import threading
import time
from array import array
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from companion.bodylink.body import BodyClient
from companion.bodylink.body_devices import (
    DeviceNotFoundError,
    InputHandle,
    Microphone,
    Pcm16StreamResampler,
    SoundDeviceMicrophone,
    SoundDeviceSpeaker,
    Speaker,
    resolve_body_devices,
)
from companion.bodylink.pairing import body_ssl_context
from companion.bodylink.transport import NotConnectedError
from companion.contracts.body import (
    HEARTBEAT_INTERVAL_MS,
    INPUT_SAMPLE_RATE_HZ,
    MAX_AUDIO_FRAME_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    UINT32_MAX,
    AbortRecording,
    AudioFrame,
    ButtonPressed,
    Fault,
    MuteState,
    PlayAudio,
    PlaybackFinished,
    RecordingStopped,
    ShowState,
    StartRecording,
    StopPlayback,
    StopRecording,
)
from companion.contracts.robot import RobotEvent
from companion.expression import ExpressionController, FaceDisplay
from companion.runtime.speech_expression import NullHeadServo, TerminalFaceDisplay
from companion.speech.local_audio_devices import load_sounddevice_runtime
from companion.speech.local_errors import LocalSpeechAdapterError

MUTE_POLL_INTERVAL_S = 0.05
BODY_EVENT_QUEUE_CAPACITY = 64
DOWNLINK_PROGRESS_TIMEOUT_S = 3.0
SELF_TEST_RECORD_SECONDS = 1.0
SELF_TEST_TONE_SECONDS = 0.5
SELF_TEST_TONE_HZ = 440


class ButtonSource(Protocol):
    """Terminal keys or no button; a Raspberry Pi GPIO button later."""

    def start(self, on_press: Callable[[], None]) -> None: ...

    def close(self) -> None: ...


class MuteSource(Protocol):
    """Polled by the body worker; True means the microphone switch is on. ``source`` is sent
    in every MuteState: "hardware_switch" for a physical switch, "development" otherwise."""

    source: Literal["hardware_switch", "development"]

    def microphone_enabled(self) -> bool: ...


class BodyLink(Protocol):
    def send_message(self, message: Any) -> None: ...

    def send_frame(self, session_id: str, frame: AudioFrame) -> None: ...


class DevelopmentMuteSource:
    """No physical mute switch is wired yet: always enabled, reported as "development"."""

    source: Literal["development"] = "development"

    def microphone_enabled(self) -> bool:
        return True


class TerminalButtonSource:
    """Space presses the button, Q/q calls ``on_quit``. ``fd`` should be in cbreak mode."""

    def __init__(self, fd: int, *, on_quit: Callable[[], None]) -> None:
        self._fd = fd
        self._on_quit = on_quit
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_press: Callable[[], None]) -> None:
        self._thread = threading.Thread(target=self._run, args=(on_press,), daemon=True)
        self._thread.start()

    def _run(self, on_press: Callable[[], None]) -> None:
        while not self._closed.is_set():
            readable, _, _ = select.select([self._fd], [], [], 0.1)
            if not readable:
                continue
            key = os.read(self._fd, 1)
            if key == b" ":
                on_press()
            elif key in (b"q", b"Q", b""):
                self._on_quit()
                return

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join()


class NoButtonSource:
    """No local conversation trigger; no callback or thread is retained."""

    def start(self, on_press: Callable[[], None]) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class _QueuedBodyEvent:
    session_id: str
    generation: int
    payload: Any


@dataclass
class _Recording:
    turn_id: int
    max_bytes: int
    resampler: Pcm16StreamResampler
    session_id: str
    generation: int
    handle: InputHandle | None = None
    sent_bytes: int = 0
    next_index: int = 0


@dataclass
class _Playback:
    token: int
    sample_rate: int
    total_bytes: int
    session_id: str
    generation: int
    receive_deadline: float
    chunks: list[bytes] = field(default_factory=list)
    received_bytes: int = 0
    next_index: int = 0
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class BodyCore:
    """Protocol behaviour of the body. Wire the four ``on_*`` methods into a ``BodyClient``,
    then ``start(client)`` before ``client.start()``."""

    def __init__(
        self,
        *,
        microphone: Microphone,
        speaker: Speaker,
        mute_source: MuteSource,
        display: FaceDisplay,
        clock: Callable[[], float] = time.monotonic,
        mute_resend_interval_s: float = HEARTBEAT_INTERVAL_MS / 1000,
    ) -> None:
        self._microphone = microphone
        self._speaker = speaker
        self._mute_source = mute_source
        self._display = display
        self._clock = clock
        self._resend_s = mute_resend_interval_s
        self._expression = ExpressionController(display, NullHeadServo())
        self._condition = threading.Condition()
        self._queue: deque[_QueuedBodyEvent] = deque()
        self._accepting_session_id: str | None = None
        self._generation = 0
        self._applied_generation = 0
        self._closing = False
        self._overflow_generation: int | None = None
        self._link: BodyLink | None = None
        self._thread: threading.Thread | None = None
        self._session_id: str | None = None
        self._recording: _Recording | None = None
        self._playback: _Playback | None = None
        self._mute_enabled: bool | None = None
        self._mute_sent_at: float | None = None

    # ---- inputs (any thread; no device operations or sends) ----

    def on_session_started(self, session_id: str) -> None:
        with self._condition:
            if self._closing or self._accepting_session_id == session_id:
                return
            self._accepting_session_id = session_id
            self._generation += 1
            if self._playback is not None:
                self._playback.stop.set()
            self._condition.notify_all()

    def on_session_ended(self, session_id: str, _reason: str) -> None:
        with self._condition:
            if session_id != self._accepting_session_id:
                return
            self._accepting_session_id = None
            self._generation += 1
            if self._playback is not None:
                self._playback.stop.set()
            self._condition.notify_all()

    def on_message(self, message: Any) -> None:
        with self._condition:
            generation = self._generation
        self._enqueue(message, session_id=message.session_id, generation=generation)

    def on_frame(self, frame: AudioFrame) -> None:
        with self._condition:
            session_id, generation = self._accepting_session_id, self._generation
        if session_id is not None:
            self._enqueue(frame, session_id=session_id, generation=generation)

    def button_pressed(self) -> None:
        with self._condition:
            session_id, generation = self._accepting_session_id, self._generation
        if session_id is not None:
            self._enqueue(("button", self._now_ms()), session_id=session_id, generation=generation)

    def _enqueue(self, payload: Any, *, session_id: str, generation: int) -> None:
        with self._condition:
            if (
                self._closing
                or session_id != self._accepting_session_id
                or generation != self._generation
                or self._overflow_generation == generation
            ):
                return
            if len(self._queue) == BODY_EVENT_QUEUE_CAPACITY:
                self._overflow_generation = generation
                if self._playback is not None:
                    self._playback.stop.set()
            else:
                self._queue.append(_QueuedBodyEvent(session_id, generation, payload))
            self._condition.notify_all()

    # ---- lifecycle ----

    def start(self, link: BodyLink) -> None:
        self._link = link
        self._thread = threading.Thread(target=self._run, name="body-core", daemon=True)
        self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._closing = True
            if self._playback is not None:
                self._playback.stop.set()
            self._condition.notify_all()
        if self._thread is None:
            return
        self._thread.join()
        self._thread = None

    # ---- worker ----

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._closing:
                        self._queue.clear()
                        return
                    if self._generation != self._applied_generation:
                        action = "session"
                    elif self._overflow_generation == self._applied_generation:
                        action = "overflow"
                        self._queue.clear()
                    elif self._queue:
                        action = "event"
                        event = self._queue.popleft()
                    else:
                        if self._condition.wait(MUTE_POLL_INTERVAL_S):
                            continue
                        action = "poll"
                if action == "session":
                    self._apply_session_change()
                elif action == "overflow":
                    self._abort_overflow()
                self._expire_incomplete_playback()
                if action == "event":
                    with self._condition:
                        valid = (
                            not self._closing
                            and event.session_id == self._session_id == self._accepting_session_id
                            and event.generation == self._applied_generation == self._generation
                            and self._overflow_generation != event.generation
                        )
                    if valid:
                        self._handle(event.payload)
                # A lifecycle/overflow notification during a call wins over the next mute poll.
                with self._condition:
                    poll = (
                        not self._closing
                        and self._generation == self._applied_generation
                        and self._overflow_generation is None
                    )
                if poll:
                    self._poll_mute()
        finally:
            self._drop_everything()

    def _apply_session_change(self) -> None:
        previous_session = self._session_id
        self._drop_everything()
        if previous_session is not None:
            print("与大脑断开，等待重连", flush=True)
            self._expression = ExpressionController(self._display, NullHeadServo())
            enabled = self._mute_source.microphone_enabled()
            self._expression.handle(RobotEvent(state="muted"))
            if enabled:
                self._expression.handle(RobotEvent(state="idle"))
        with self._condition:
            self._queue = deque(
                event for event in self._queue if event.generation == self._generation
            )
            self._session_id = self._accepting_session_id
            self._applied_generation = self._generation
            if self._overflow_generation != self._generation:
                self._overflow_generation = None
        self._mute_sent_at = None

    def _abort_overflow(self) -> None:
        generation = self._applied_generation
        if self._recording is not None:
            self._end_recording("local_abort")
        playback = self._playback
        self._stop_playback()
        if playback is not None:
            self._send(PlaybackFinished, playback_token=playback.token, result="failed")
        with self._condition:
            if self._overflow_generation == generation:
                self._overflow_generation = None
            self._condition.notify_all()

    def _handle(self, item: Any) -> None:
        if isinstance(item, AudioFrame):
            self._on_downlink(item)
        elif isinstance(item, tuple):
            kind = item[0]
            if kind == "button":
                self._on_button(item[1])
            elif kind == "audio":
                self._on_audio(item[1], item[2])
            elif kind == "played":
                self._on_played(item[1], item[2])
        elif isinstance(item, StartRecording):
            self._start_recording(item)
        elif isinstance(item, StopRecording | AbortRecording):
            recording = self._recording
            if recording is not None and recording.turn_id == item.turn_id:
                reason = "stop_command" if isinstance(item, StopRecording) else "local_abort"
                self._end_recording(reason)
        elif isinstance(item, PlayAudio):
            self._stop_playback()
            with self._condition:
                self._playback = _Playback(
                    token=item.playback_token,
                    sample_rate=item.sample_rate,
                    total_bytes=item.total_bytes,
                    session_id=item.session_id,
                    generation=self._applied_generation,
                    receive_deadline=self._clock() + DOWNLINK_PROGRESS_TIMEOUT_S,
                )
                if (
                    self._closing
                    or self._generation != self._applied_generation
                    or self._overflow_generation == self._applied_generation
                ):
                    self._playback.stop.set()
        elif isinstance(item, StopPlayback):
            playback = self._playback
            if playback is not None and playback.token == item.playback_token:
                self._stop_playback()
                self._send(PlaybackFinished, playback_token=playback.token, result="stopped")
        elif isinstance(item, ShowState):
            self._poll_mute()
            event = item.event if self._mute_enabled else RobotEvent(state="muted")
            self._expression.handle(event)

    # ---- recording ----

    def _start_recording(self, message: StartRecording) -> None:
        self._discard_recording()
        if not self._mute_enabled:
            self._send(RecordingStopped, turn_id=message.turn_id, reason="muted", total_bytes=0)
            return
        recording = _Recording(
            turn_id=message.turn_id,
            max_bytes=message.max_seconds * INPUT_SAMPLE_RATE_HZ * 2,
            resampler=Pcm16StreamResampler(self._microphone.sample_rate),
            session_id=message.session_id,
            generation=self._applied_generation,
        )
        try:
            recording.handle = self._microphone.open(
                lambda pcm: self._enqueue(
                    ("audio", recording, pcm),
                    session_id=recording.session_id,
                    generation=recording.generation,
                )
            )
        except Exception as exc:  # any device/driver error: report it, keep the body running
            self._send(Fault, kind="microphone_unavailable", detail=type(exc).__name__)
            self._send(
                RecordingStopped, turn_id=message.turn_id, reason="local_abort", total_bytes=0
            )
            return
        self._recording = recording

    def _on_audio(self, recording: _Recording, pcm: bytes) -> None:
        if recording is not self._recording:
            return  # late callback from a recording that already ended
        data = recording.resampler.process(pcm)
        room = recording.max_bytes - recording.sent_bytes
        self._send_uplink(recording, data[:room], last=False)
        if len(data) >= room:
            self._end_recording("limit")

    def _end_recording(self, reason: str) -> None:
        recording = self._recording
        assert recording is not None and recording.handle is not None
        self._recording = None
        recording.handle.close()
        if reason in ("stop_command", "limit"):
            tail = recording.resampler.finish()[: recording.max_bytes - recording.sent_bytes]
            self._send_uplink(recording, tail, last=True)
        self._send(
            RecordingStopped,
            turn_id=recording.turn_id,
            reason=reason,
            total_bytes=recording.sent_bytes,
        )

    def _discard_recording(self) -> None:
        recording = self._recording
        self._recording = None
        if recording is not None and recording.handle is not None:
            recording.handle.close()

    def _send_uplink(self, recording: _Recording, data: bytes, *, last: bool) -> None:
        if not last and not data:
            return
        step = MAX_AUDIO_FRAME_PAYLOAD_BYTES
        pieces = [data[i : i + step] for i in range(0, len(data), step)] or [b""]
        for position, piece in enumerate(pieces):
            frame = AudioFrame(
                direction="uplink",
                stream_id=recording.turn_id,
                frame_index=recording.next_index,
                is_last=last and position == len(pieces) - 1,
                payload=piece,
            )
            recording.next_index += 1
            recording.sent_bytes += len(piece)
            self._send_frame(frame)

    # ---- playback ----

    def _expire_incomplete_playback(self) -> None:
        playback = self._playback
        if (
            playback is not None
            and playback.thread is None
            and self._clock() >= playback.receive_deadline
        ):
            self._stop_playback()
            self._send(PlaybackFinished, playback_token=playback.token, result="failed")

    def _on_downlink(self, frame: AudioFrame) -> None:
        playback = self._playback
        if playback is None or frame.stream_id != playback.token or playback.thread is not None:
            return
        if (
            frame.frame_index != playback.next_index
            or (not frame.payload and not frame.is_last)
            or playback.received_bytes + len(frame.payload) > playback.total_bytes
            or (frame.is_last and playback.received_bytes + len(frame.payload)
                != playback.total_bytes)
        ):  # fmt: skip
            self._stop_playback()
            self._send(PlaybackFinished, playback_token=playback.token, result="failed")
            return
        playback.chunks.append(frame.payload)
        playback.received_bytes += len(frame.payload)
        playback.next_index += 1
        if frame.payload:
            playback.receive_deadline = self._clock() + DOWNLINK_PROGRESS_TIMEOUT_S
        if frame.is_last:
            pcm = b"".join(playback.chunks)
            playback.chunks.clear()
            playback.thread = threading.Thread(
                target=self._play, args=(playback, pcm), name="body-playback", daemon=True
            )
            playback.thread.start()

    def _play(self, playback: _Playback, pcm: bytes) -> None:
        try:
            completed = self._speaker.play(pcm, playback.sample_rate, playback.stop)
            result = "completed" if completed else "stopped"
        except Exception:  # device errors end this playback as "failed"
            result = "failed"
        self._enqueue(
            ("played", playback, result),
            session_id=playback.session_id,
            generation=playback.generation,
        )

    def _on_played(self, playback: _Playback, result: str) -> None:
        if playback is not self._playback:
            return  # already stopped and reported
        with self._condition:
            self._playback = None
        assert playback.thread is not None
        playback.thread.join()
        self._send(PlaybackFinished, playback_token=playback.token, result=result)
        if result == "failed":
            self._send(Fault, kind="speaker_unavailable", detail=None)

    def _stop_playback(self) -> None:
        with self._condition:
            playback = self._playback
        if playback is None:
            return
        playback.stop.set()
        abort_failed = False
        if playback.thread is not None:
            try:
                self._speaker.abort()
            except Exception:
                abort_failed = True
            playback.thread.join()
        with self._condition:
            self._playback = None
            source_valid = (
                playback.session_id == self._session_id == self._accepting_session_id
                and playback.generation == self._applied_generation == self._generation
            )
        playback.chunks.clear()
        if abort_failed and source_valid:
            self._send(Fault, kind="speaker_unavailable", detail=None)

    # ---- button, mute ----

    def _on_button(self, observed_at_ms: int) -> None:
        if self._session_id is None:
            return
        playback = self._playback
        if playback is not None:
            self._stop_playback()
            self._send(PlaybackFinished, playback_token=playback.token, result="stopped")
        self._send(ButtonPressed, observed_at_ms=observed_at_ms)

    def _poll_mute(self) -> None:
        enabled = self._mute_source.microphone_enabled()
        previous_enabled = self._mute_enabled
        changed = enabled != previous_enabled
        self._mute_enabled = enabled
        if changed and not enabled and self._recording is not None:
            self._end_recording("muted")
        if changed and not enabled:
            self._expression.handle(RobotEvent(state="muted"))
        elif previous_enabled is False and enabled:
            self._expression.handle(RobotEvent(state="idle"))
        now = self._clock()
        if self._session_id is not None and (
            changed or self._mute_sent_at is None or now - self._mute_sent_at >= self._resend_s
        ):
            self._mute_sent_at = now
            self._send(
                MuteState,
                microphone_enabled=enabled,
                source=self._mute_source.source,
                observed_at_ms=self._now_ms(),
            )

    # ---- helpers ----

    def _drop_everything(self) -> None:
        self._discard_recording()
        self._stop_playback()

    def _now_ms(self) -> int:
        return int(self._clock() * 1000) % (UINT32_MAX + 1)

    def _send(self, cls: type, **fields: Any) -> None:
        with self._condition:
            if (
                self._closing
                or self._session_id is None
                or self._session_id != self._accepting_session_id
                or self._applied_generation != self._generation
            ):
                return
            session_id = self._session_id
        assert self._link is not None
        message = cls(protocol_version=PROTOCOL_VERSION, seq=0, session_id=session_id,
                      **fields)  # fmt: skip
        # Transport rejects a source invalidated after this capture; never relabel it.
        with suppress(NotConnectedError):
            self._link.send_message(message)

    def _send_frame(self, frame: AudioFrame) -> None:
        with self._condition:
            if (
                self._closing
                or self._session_id is None
                or self._session_id != self._accepting_session_id
                or self._applied_generation != self._generation
            ):
                return
            session_id = self._session_id
        assert self._link is not None
        with suppress(NotConnectedError):
            self._link.send_frame(session_id, frame)


# ---- self-test (no brain) ----


def run_self_test(
    microphone: Microphone, speaker: Speaker, output_rate: int, out: Callable[[str], None]
) -> None:
    """Record 1 s (kept in memory, only size and peak reported), then play a 440 Hz tone."""
    import math

    wanted = int(microphone.sample_rate * SELF_TEST_RECORD_SECONDS) * 2
    chunks: list[bytes] = []
    enough = threading.Event()
    collected = 0

    def on_audio(pcm: bytes) -> None:
        nonlocal collected
        chunks.append(pcm)
        collected += len(pcm)
        if collected >= wanted:
            enough.set()

    handle = microphone.open(on_audio)
    try:
        if not enough.wait(SELF_TEST_RECORD_SECONDS + 3.0):
            raise RuntimeError("microphone delivered no audio")
    finally:
        handle.close()
    resampler = Pcm16StreamResampler(microphone.sample_rate)
    pcm = resampler.process(b"".join(chunks)[:wanted]) + resampler.finish()
    chunks.clear()
    samples = array("h")
    samples.frombytes(pcm)
    peak = max(map(abs, samples), default=0)
    out(f"自检录音：{len(pcm)} 字节（16 kHz 单声道，未保存），峰值 {peak}/32768")
    del pcm, samples

    count = int(output_rate * SELF_TEST_TONE_SECONDS)
    tone = array(
        "h",
        (
            round(6000 * math.sin(2 * math.pi * SELF_TEST_TONE_HZ * i / output_rate))
            for i in range(count)
        ),
    )
    completed = speaker.play(tone.tobytes(), output_rate, threading.Event())
    outcome = "完成" if completed else "中断"
    out(f"自检放音：{SELF_TEST_TONE_SECONDS} 秒 {SELF_TEST_TONE_HZ} Hz 正弦，{outcome}")


# ---- entry point ----


@contextmanager
def _sigterm_quit(quit_requested: threading.Event) -> Iterator[None]:
    saved = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: quit_requested.set())
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, saved)


@contextmanager
def _cbreak(fd: int) -> Iterator[None]:
    import termios
    import tty

    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m companion.bodylink.body_main", description="机器人身体进程"
    )
    parser.add_argument("--input-device", required=True, help="输入设备名（与 query_devices 一致）")
    parser.add_argument("--output-device", required=True, help="输出设备名")
    parser.add_argument("--hostapi", default=None, help="音频后端名，如 ALSA、Core Audio")
    parser.add_argument("--self-test", action="store_true", help="只做设备自检，不连接大脑")
    parser.add_argument(
        "--button", choices=("terminal", "none"), default="terminal",
        help="按键来源：terminal=空格/Q，none=无按键",
    )
    parser.add_argument("--brain-host")
    parser.add_argument("--brain-port", type=int)
    parser.add_argument("--body-id")
    parser.add_argument("--cert-dir", type=Path, help="配对生成的 body/ 目录")
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [
            name
            for name in ("brain_host", "brain_port", "body_id", "cert_dir")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("缺少参数：" + "、".join("--" + m.replace("_", "-") for m in missing))
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        runtime = load_sounddevice_runtime()
        pair = resolve_body_devices(runtime, args.input_device, args.output_device, args.hostapi)
    except (LocalSpeechAdapterError, DeviceNotFoundError):
        print("音频设备不可用：检查 sounddevice 安装与设备名", file=sys.stderr)
        return 2
    microphone = SoundDeviceMicrophone(runtime, pair.input_index, pair.input_sample_rate_hz)
    speaker = SoundDeviceSpeaker(runtime, pair.output_index)
    if args.self_test:
        run_self_test(microphone, speaker, pair.output_sample_rate_hz, print)
        return 0
    if args.button == "terminal" and not sys.stdin.isatty():
        print("需要在终端里运行（空格键=按键，Q=退出）", file=sys.stderr)
        return 2

    print("开发模式：未接物理断麦开关", flush=True)
    quit_requested = threading.Event()
    core = BodyCore(
        microphone=microphone,
        speaker=speaker,
        mute_source=DevelopmentMuteSource(),
        display=TerminalFaceDisplay(),
    )
    client = BodyClient(
        host=args.brain_host,
        port=args.brain_port,
        ssl_context=body_ssl_context(args.cert_dir),
        body_id=args.body_id,
        output_sample_rates=(pair.output_sample_rate_hz,),
        on_session_started=core.on_session_started,
        on_session_ended=core.on_session_ended,
        on_message=core.on_message,
        on_frame=core.on_frame,
    )
    button: ButtonSource
    if args.button == "terminal":
        fd = sys.stdin.fileno()
        button = TerminalButtonSource(fd, on_quit=quit_requested.set)
        terminal_mode = _cbreak(fd)
        prompt = "空格键=按键，Q=退出"
    else:
        button = NoButtonSource()
        terminal_mode = nullcontext()
        prompt = "无按键模式：等待大脑连接；未接物理按键，不能用物理按键发起对话"
    with _sigterm_quit(quit_requested), terminal_mode, ExitStack() as resources:
        resources.callback(core.close)
        resources.callback(client.close)
        resources.callback(button.close)
        try:
            core.start(client)
            client.start()
            button.start(core.button_pressed)
            print(prompt, flush=True)
            quit_requested.wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
