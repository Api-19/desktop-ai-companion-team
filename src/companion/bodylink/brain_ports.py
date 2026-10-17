"""Brain-side speech ports over bodylink: network recorder, player, mute gate and body inputs.

``BrainBodyLink`` owns a ``BrainServer``. Transport callbacks only update state under one
condition lock and enqueue; every send happens on the synchronous caller's thread with the
lock released. Nothing here logs audio or message bodies.
"""

import queue
import ssl
import struct
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from companion.bodylink.brain import BrainServer
from companion.bodylink.transport import EndReason, NotConnectedError, TransportStats
from companion.contracts.body import (
    HEARTBEAT_INTERVAL_MS,
    INPUT_SAMPLE_RATE_HZ,
    LIVENESS_TIMEOUT_MS,
    MAX_AUDIO_FRAME_PAYLOAD_BYTES,
    MAX_RECORDING_BYTES,
    PROTOCOL_VERSION,
    UINT32_MAX,
    AbortRecording,
    AudioFrame,
    ButtonPressed,
    Hello,
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
from companion.speech.controller_ports import SynthesizedAudio
from companion.speech.local_errors import LocalSpeechAdapterErrorCode, raise_local_speech_error
from companion.speech.local_wav import OwnedWavFile, write_validated_pcm_wav
from companion.speech.pcm_audio import PcmAudio, pcm16_samples
from companion.speech.pcm_resampler import resample_pcm16
from companion.speech.sounddevice_player import PlaybackDeviceEvent

MUTE_STATE_MAX_AGE_S = LIVENESS_TIMEOUT_MS / 1000
DEFAULT_RECORDING_MAX_SECONDS = 30
DEFAULT_STOP_TIMEOUT_S = 2.0
PLAYBACK_RECEIPT_GRACE_S = 5.0

BodyInput = ButtonPressed | MuteState


def _recording_failed() -> NoReturn:
    raise_local_speech_error(LocalSpeechAdapterErrorCode.RECORDING_FAILED)


def _playback_failed() -> NoReturn:
    raise_local_speech_error(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED)


@dataclass
class _Recording:
    session_id: str
    turn_id: int
    chunks: list[bytes] = field(default_factory=list)
    received_bytes: int = 0
    next_index: int = 0
    last_frame_seen: bool = False
    stopped: RecordingStopped | None = None
    failed: bool = False

    def fail(self) -> None:
        self.failed = True
        self.chunks.clear()
        self.received_bytes = 0


class BrainBodyLink:
    """One paired body seen as speech ports: ``recorder``, ``player``, ``mute_gate`` and
    ``drain_body_inputs()`` (ButtonPressed and changed MuteState, oldest first)."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        ssl_context: ssl.SSLContext,
        body_id: str,
        run_root: Path,
        clock: Callable[[], float] = time.monotonic,
        recording_max_seconds: int = DEFAULT_RECORDING_MAX_SECONDS,
        stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
        heartbeat_interval_ms: int = HEARTBEAT_INTERVAL_MS,
        liveness_timeout_ms: int = LIVENESS_TIMEOUT_MS,
    ) -> None:
        self._clock = clock
        self._cond = threading.Condition()
        self._session_id: str | None = None
        self._output_rates: tuple[int, ...] = ()
        self._mute: tuple[bool, float] | None = None
        self._shown: RobotEvent | None = None
        self._recording: _Recording | None = None
        self._playback_token: int | None = None
        self._playback_session_id: str | None = None
        self._playback_receipt_deadline: float | None = None
        self._inputs: queue.SimpleQueue[BodyInput] = queue.SimpleQueue()
        self._playback_events: queue.SimpleQueue[PlaybackDeviceEvent] = queue.SimpleQueue()
        self._server = BrainServer(
            host=host,
            port=port,
            ssl_context=ssl_context,
            body_id=body_id,
            on_session_started=self._on_session_started,
            on_session_ended=self._on_session_ended,
            on_message=self._on_message,
            on_frame=self._on_frame,
            heartbeat_interval_ms=heartbeat_interval_ms,
            liveness_timeout_ms=liveness_timeout_ms,
        )
        self.mute_gate = NetworkMuteGate(self)
        self.recorder = NetworkAudioRecorder(
            self,
            run_root=run_root,
            max_seconds=recording_max_seconds,
            stop_timeout_s=stop_timeout_s,
        )
        self.player = NetworkAudioPlayer(self)

    @property
    def port(self) -> int | None:
        return self._server.port

    @property
    def current_session_id(self) -> str | None:
        with self._cond:
            return self._session_id

    def stats_snapshot(self) -> TransportStats:
        return self._server.stats_snapshot()

    def start(self) -> None:
        self._server.start()

    def close(self) -> None:
        self._server.close()

    def drain_body_inputs(self) -> tuple[BodyInput, ...]:
        return _drain(self._inputs)

    def show_state(self, event: RobotEvent) -> None:
        """Send only the state with detail=None, deduplicated by state in this session.

        Without a session the event is dropped; session end discards the shown state.
        """
        event = RobotEvent(state=event.state)
        with self._cond:
            session_id = self._session_id
            if session_id is None or event == self._shown:
                return
            self._shown = event
        # A lost session resets ``_shown``; the next session gets the state again.
        with suppress(NotConnectedError):
            self._server.send_message(
                ShowState(
                    protocol_version=PROTOCOL_VERSION, seq=0, session_id=session_id, event=event
                )
            )

    # ---- state queries (caller holds self._cond) ----

    def _microphone_enabled_locked(self) -> bool:
        if self._session_id is None or self._mute is None:
            return False
        enabled, received_at = self._mute
        return enabled and self._clock() - received_at <= MUTE_STATE_MAX_AGE_S

    # ---- transport callbacks (transport thread; never block, never send) ----

    def _on_session_started(self, session_id: str, hello: Hello) -> None:
        with self._cond:
            self._session_id = session_id
            _drain(self._inputs)
            self._output_rates = hello.output_sample_rates
            self._mute = None
            self._shown = None

    def _on_session_ended(self, session_id: str, reason: EndReason) -> None:
        with self._cond:
            if self._session_id is None or session_id != self._session_id:
                return
            self._session_id = None
            _drain(self._inputs)
            self._shown = None
            self._output_rates = ()
            self._mute = None
            if self._recording is not None:
                self._recording.fail()
                self._recording = None
            if self._playback_token is not None:
                self._playback_events.put(PlaybackDeviceEvent(self._playback_token, False))
            self._playback_token = None
            self._playback_session_id = None
            self._playback_receipt_deadline = None
            self._cond.notify_all()

    def _on_message(self, message: Any) -> None:
        with self._cond:
            if message.session_id != self._session_id:
                return
            if isinstance(message, ButtonPressed):
                self._inputs.put(message)
            elif isinstance(message, MuteState):
                previous = self._mute
                self._mute = (message.microphone_enabled, self._clock())
                if previous is None or previous[0] != message.microphone_enabled:
                    self._inputs.put(message)
            elif isinstance(message, RecordingStopped):
                recording = self._recording
                if recording is not None and recording.turn_id == message.turn_id:
                    recording.stopped = message
                    if message.reason not in ("stop_command", "limit"):
                        recording.fail()
                    self._cond.notify_all()
            elif isinstance(message, PlaybackFinished):
                self.player._on_finished(message)

    def _on_frame(self, frame: AudioFrame) -> None:
        with self._cond:
            recording = self._recording
            if recording is None or recording.failed or frame.stream_id != recording.turn_id:
                return
            if (
                recording.last_frame_seen
                or frame.frame_index != recording.next_index
                or (not frame.payload and not frame.is_last)
                or recording.received_bytes + len(frame.payload) > MAX_RECORDING_BYTES
            ):
                recording.fail()
            else:
                recording.chunks.append(frame.payload)
                recording.received_bytes += len(frame.payload)
                recording.next_index += 1
                recording.last_frame_seen = frame.is_last
            self._cond.notify_all()


class NetworkMuteGate:
    """Open only while a session is live and its latest MuteState (received within the last
    3 s on the brain's monotonic clock) says the microphone is enabled."""

    def __init__(self, link: BrainBodyLink) -> None:
        self._link = link

    def microphone_enabled(self) -> bool:
        with self._link._cond:
            return self._link._microphone_enabled_locked()


class NetworkRecordingSession:
    def __init__(self, recorder: "NetworkAudioRecorder", recording: _Recording) -> None:
        self._recorder = recorder
        self._recording = recording

    def stop(self) -> OwnedWavFile:
        link = self._recorder._link
        recording = self._recording
        with link._cond:
            usable = link._recording is recording and not recording.failed
            already_stopped = recording.stopped is not None
        if not usable:
            self._recorder._discard(recording)
            _recording_failed()
        if not already_stopped:
            try:
                link._server.send_message(
                    StopRecording(
                        protocol_version=PROTOCOL_VERSION,
                        seq=0,
                        session_id=recording.session_id,
                        turn_id=recording.turn_id,
                    )
                )
            except NotConnectedError:
                self._recorder._discard(recording)
                _recording_failed()
        with link._cond:
            link._cond.wait_for(
                lambda: (
                    recording.failed
                    or (recording.stopped is not None and recording.last_frame_seen)
                ),
                timeout=self._recorder._stop_timeout_s,
            )
            complete = (
                not recording.failed
                and recording.stopped is not None
                and recording.last_frame_seen
                and recording.stopped.total_bytes == recording.received_bytes
                and recording.received_bytes > 0
            )
            payload = b"".join(recording.chunks) if complete else b""
            recording.fail()
            if link._recording is recording:
                link._recording = None
        if not complete:
            _recording_failed()
        samples = tuple(sample for (sample,) in struct.iter_unpack("<h", payload))
        del payload
        return write_validated_pcm_wav(self._recorder._run_root, samples, INPUT_SAMPLE_RATE_HZ)

    def abort(self) -> None:
        link = self._recorder._link
        recording = self._recording
        with link._cond:
            if link._recording is not recording:
                return
            link._recording = None
            send_abort = recording.stopped is None and not recording.failed
            recording.fail()
        if send_abort:
            # A lost session means the body has already dropped the turn itself.
            with suppress(NotConnectedError):
                link._server.send_message(
                    AbortRecording(
                        protocol_version=PROTOCOL_VERSION,
                        seq=0,
                        session_id=recording.session_id,
                        turn_id=recording.turn_id,
                    )
                )

    def close(self) -> None:
        self.abort()


class NetworkAudioRecorder:
    """Records 16 kHz PCM on the body; ``stop()`` writes it to an OwnedWavFile in ``run_root``."""

    def __init__(
        self, link: BrainBodyLink, *, run_root: Path, max_seconds: int, stop_timeout_s: float
    ) -> None:
        self._link = link
        self._run_root = run_root
        self._max_seconds = max_seconds
        self._stop_timeout_s = stop_timeout_s
        self._last_turn_id = 0

    def start(self) -> NetworkRecordingSession:
        link = self._link
        with link._cond:
            session_id = link._session_id
            if (
                session_id is None
                or link._recording is not None
                or not link._microphone_enabled_locked()
            ):
                _recording_failed()
            self._last_turn_id = self._last_turn_id % UINT32_MAX + 1
            recording = _Recording(session_id=session_id, turn_id=self._last_turn_id)
            link._recording = recording
        try:
            link._server.send_message(
                StartRecording(
                    protocol_version=PROTOCOL_VERSION,
                    seq=0,
                    session_id=session_id,
                    turn_id=recording.turn_id,
                    max_seconds=self._max_seconds,
                )
            )
        except NotConnectedError:
            self._discard(recording)
            _recording_failed()
        return NetworkRecordingSession(self, recording)

    def _discard(self, recording: _Recording) -> None:
        with self._link._cond:
            recording.fail()
            if self._link._recording is recording:
                self._link._recording = None


class NetworkAudioPlayer:
    """Streams PcmAudio to the body. ``drain_events()`` has SoundDeviceAudioPlayer semantics:
    one PlaybackDeviceEvent per playback that ended on its own; none after ``stop()``."""

    def __init__(self, link: BrainBodyLink) -> None:
        self._link = link

    def start(self, audio: PcmAudio, playback_token: int) -> None:
        session_id = self._link.current_session_id
        if session_id is None:
            _playback_failed()
        self.start_for_session(audio, playback_token, session_id)

    def start_for_session(
        self, audio: SynthesizedAudio, playback_token: int, session_id: str
    ) -> None:
        link = self._link
        if type(audio) is not PcmAudio:
            _playback_failed()
        with link._cond:
            rates = link._output_rates
            if session_id != link._session_id or link._playback_token is not None:
                _playback_failed()
            link._playback_token = playback_token
            link._playback_session_id = session_id
            link._playback_receipt_deadline = None
        try:
            rate = min(rates, key=lambda r: abs(r - audio.sample_rate_hz))
            samples = resample_pcm16(pcm16_samples(audio), audio.sample_rate_hz, rate)
            payload = struct.pack(f"<{len(samples)}h", *samples)
            del samples
            link._server.send_message(
                PlayAudio(
                    protocol_version=PROTOCOL_VERSION,
                    seq=0,
                    session_id=session_id,
                    playback_token=playback_token,
                    sample_rate=rate,
                    channels=1,
                    sample_format="pcm_s16le",
                    total_bytes=len(payload),
                )
            )
            step = MAX_AUDIO_FRAME_PAYLOAD_BYTES
            for index, offset in enumerate(range(0, len(payload), step)):
                link._server.send_frame(
                    session_id,
                    AudioFrame(
                        direction="downlink",
                        stream_id=playback_token,
                        frame_index=index,
                        is_last=offset + step >= len(payload),
                        payload=payload[offset : offset + step],
                    ),
                )
            with link._cond:
                if (
                    link._playback_session_id == session_id
                    and link._playback_token == playback_token
                ):
                    link._playback_receipt_deadline = (
                        link._clock() + len(payload) / (rate * 2) + PLAYBACK_RECEIPT_GRACE_S
                    )
        except (ValueError, RuntimeError):  # resampling, PlayAudio limits, lost session
            self._clear(playback_token, session_id)
            _playback_failed()

    def stop(self) -> None:
        link = self._link
        with link._cond:
            token = link._playback_token
            session_id = link._playback_session_id
            link._playback_token = None
            link._playback_session_id = None
            link._playback_receipt_deadline = None
        if token is None or session_id is None:
            return
        # A lost session means the body has already stopped playing.
        with suppress(NotConnectedError):
            link._server.send_message(
                StopPlayback(
                    protocol_version=PROTOCOL_VERSION,
                    seq=0,
                    session_id=session_id,
                    playback_token=token,
                )
            )

    def drain_events(self) -> tuple[PlaybackDeviceEvent, ...]:
        self._expire_playback_receipt()
        return _drain(self._link._playback_events)

    def _expire_playback_receipt(self) -> None:
        link = self._link
        with link._cond:
            deadline = link._playback_receipt_deadline
            if deadline is None or link._clock() < deadline:
                return
            session_id = link._playback_session_id
            token = link._playback_token
            link._playback_token = None
            link._playback_session_id = None
            link._playback_receipt_deadline = None
            link._playback_events.put(PlaybackDeviceEvent(token, False))
        with suppress(NotConnectedError):
            link._server.send_message(
                StopPlayback(
                    protocol_version=PROTOCOL_VERSION,
                    seq=0,
                    session_id=session_id,
                    playback_token=token,
                )
            )

    def _clear(self, playback_token: int, session_id: str) -> None:
        with self._link._cond:
            if (
                self._link._playback_token == playback_token
                and self._link._playback_session_id == session_id
            ):
                self._link._playback_token = None
                self._link._playback_session_id = None
                self._link._playback_receipt_deadline = None

    def _on_finished(self, message: PlaybackFinished) -> None:
        with self._link._cond:
            if (
                message.playback_token != self._link._playback_token
                or message.session_id != self._link._playback_session_id
            ):
                return  # stale source/token, or stopped by the brain
            self._link._playback_token = None
            self._link._playback_session_id = None
            self._link._playback_receipt_deadline = None
            # "stopped" here means the body stopped on its own (local button): not a failure.
            succeeded = message.result != "failed"
            self._link._playback_events.put(PlaybackDeviceEvent(message.playback_token, succeeded))


def _drain(source: queue.SimpleQueue) -> tuple:
    items = []
    while True:
        try:
            items.append(source.get_nowait())
        except queue.Empty:
            return tuple(items)
