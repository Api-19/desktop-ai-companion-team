from __future__ import annotations

import queue
import struct
import threading
from dataclasses import dataclass, field
from typing import NoReturn

from companion.speech.local_audio_devices import (
    LocalAudioDeviceConfig,
    _load_sounddevice_runtime,
    _resolve_audio_devices_for_runtime,
    _ResolvedAudioRuntime,
    _validate_resolved_audio_runtime,
)
from companion.speech.local_errors import LocalSpeechAdapterErrorCode, raise_local_speech_error
from companion.speech.pcm_audio import PcmAudio, pcm16_samples
from companion.speech.pcm_resampler import resample_pcm16

_SAMPLE_DTYPE = "int16"
_CHUNK_BYTES = 2048
_JOIN_TIMEOUT_SECONDS = 2.0


def _raise_playback_error(code: LocalSpeechAdapterErrorCode) -> NoReturn:
    raise_local_speech_error(code)


@dataclass(frozen=True, slots=True, repr=False)
class PlaybackDeviceEvent:
    playback_token: int
    succeeded: bool

    def __post_init__(self) -> None:
        if (
            type(self.playback_token) is not int
            or self.playback_token <= 0
            or type(self.succeeded) is not bool
        ):
            _raise_playback_error(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED)


@dataclass(slots=True, repr=False)
class _PlaybackReservation:
    playback_token: int
    cancelled: bool = False


@dataclass(slots=True, repr=False)
class _PlaybackRun:
    playback_token: int
    payload: bytes
    sample_rate_hz: int
    stop_event: threading.Event
    thread: threading.Thread | None = None
    stream: object | None = None
    stream_started: bool = False
    stream_interrupt_claimed: bool = False
    stream_stop_claimed: bool = False
    stream_close_claimed: bool = False
    stream_close_deferred: bool = False
    stream_closed: bool = False
    resource_lock: threading.Lock = field(default_factory=threading.Lock)
    interrupt_complete: threading.Event = field(default_factory=threading.Event)


def _release_output_stream(run: _PlaybackRun) -> bool:
    failed = False
    with run.resource_lock:
        stream = run.stream
        if stream is None or run.stream_close_claimed:
            return True
        should_stop = run.stream_started and not run.stream_interrupt_claimed
        if should_stop:
            run.stream_stop_claimed = True
        wait_for_interrupt = run.stream_interrupt_claimed
        run.stream_close_claimed = True
    if wait_for_interrupt and not run.interrupt_complete.wait(timeout=_JOIN_TIMEOUT_SECONDS):
        # Hand cleanup to the interrupt owner without closing during abort.
        with run.resource_lock:
            run.stream_close_deferred = True
            run.stream_close_claimed = False
        if run.interrupt_complete.is_set():
            return _release_output_stream(run)
        return False
    if should_stop:
        try:
            stream.stop()  # type: ignore[attr-defined]
        except Exception:
            failed = True
    try:
        stream.close()  # type: ignore[attr-defined]
    except Exception:
        failed = True
    with run.resource_lock:
        run.stream_closed = True
        run.stream = None
    return not failed


def _interrupt_output_stream(run: _PlaybackRun) -> bool:
    with run.resource_lock:
        stream = run.stream
        if (
            stream is None
            or run.stream_interrupt_claimed
            or run.stream_close_claimed
            or run.stream_closed
        ):
            return True
        run.stream_interrupt_claimed = True
    succeeded = True
    try:
        stream.abort()  # type: ignore[attr-defined]
    except Exception:
        succeeded = False
    finally:
        run.interrupt_complete.set()
        with run.resource_lock:
            deferred = run.stream_close_deferred
        if deferred and not _release_output_stream(run):
            succeeded = False
    return succeeded


def _capture_playback(
    audio: PcmAudio, runtime: object, output_index: int, output_sample_rate_hz: int
) -> tuple[bytes, int] | None:
    payload: bytes | None = None
    samples: tuple[int, ...] | None = None
    try:
        samples = resample_pcm16(
            pcm16_samples(audio),
            audio.sample_rate_hz,
            output_sample_rate_hz,
        )
        payload = struct.pack(f"<{len(samples)}h", *samples)
        sample_rate_hz = output_sample_rate_hz
        check_result = runtime.check_output_settings(  # type: ignore[attr-defined]
            device=output_index,
            channels=1,
            dtype=_SAMPLE_DTYPE,
            samplerate=sample_rate_hz,
        )
        if check_result is not None or not payload:
            raise ValueError
        result = (payload, sample_rate_hz)
    except Exception:
        result = None
    samples = None
    payload = None
    return result


class SoundDeviceAudioPlayer:
    __slots__ = (
        "_active",
        "_events",
        "_last_token",
        "_lock",
        "_output_index",
        "_output_sample_rate_hz",
        "_runtime",
    )

    def __init__(self, config: LocalAudioDeviceConfig) -> None:
        if type(config) is not LocalAudioDeviceConfig:
            del config
            _raise_playback_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        try:
            runtime = _load_sounddevice_runtime()
        except Exception as error:
            del error
            runtime = None
        if runtime is None:
            del config
            del runtime
            _raise_playback_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        try:
            pair = _resolve_audio_devices_for_runtime(config, runtime)
        except Exception as error:
            del error
            pair = None
        if pair is None:
            del config
            del runtime
            del pair
            _raise_playback_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        self._assign_fields(
            runtime=runtime,
            output_index=pair.output_index,
            output_sample_rate_hz=pair.output_sample_rate_hz,
        )

    def _assign_fields(
        self, *, runtime: object, output_index: int, output_sample_rate_hz: int
    ) -> None:
        self._runtime = runtime
        self._output_index = output_index
        self._output_sample_rate_hz = output_sample_rate_hz
        self._lock = threading.Lock()
        self._events: queue.SimpleQueue[PlaybackDeviceEvent] = queue.SimpleQueue()
        self._active: _PlaybackReservation | _PlaybackRun | None = None
        self._last_token = 0

    @classmethod
    def _create(
        cls, *, runtime: object, output_index: int, output_sample_rate_hz: int
    ) -> SoundDeviceAudioPlayer:
        instance = object.__new__(cls)
        instance._assign_fields(
            runtime=runtime,
            output_index=output_index,
            output_sample_rate_hz=output_sample_rate_hz,
        )
        return instance

    @classmethod
    def from_config(cls, config: LocalAudioDeviceConfig) -> SoundDeviceAudioPlayer:
        try:
            player = cls(config)
        except Exception as error:
            del error
            player = None
        if player is None:
            del config
            del player
            _raise_playback_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        return player

    @classmethod
    def from_resolved(
        cls,
        config: LocalAudioDeviceConfig,
        resolved: _ResolvedAudioRuntime,
    ) -> SoundDeviceAudioPlayer:
        try:
            validated = _validate_resolved_audio_runtime(config, resolved)
            player = cls._create(
                runtime=validated.runtime,
                output_index=validated.pair.output_index,
                output_sample_rate_hz=validated.pair.output_sample_rate_hz,
            )
        except Exception as error:
            del error
            player = None
        if player is None:
            del config
            del resolved
            del player
            _raise_playback_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        return player

    def start(self, audio: object, playback_token: int) -> None:
        if type(playback_token) is not int or playback_token <= 0 or type(audio) is not PcmAudio:
            del audio
            del playback_token
            _raise_playback_error(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED)
        with self._lock:
            if self._active is not None or playback_token <= self._last_token:
                _raise_playback_error(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED)
            reservation = _PlaybackReservation(playback_token)
            self._active = reservation
            self._last_token = playback_token
        captured = _capture_playback(
            audio,
            self._runtime,
            self._output_index,
            self._output_sample_rate_hz,
        )
        del audio
        if captured is None:
            with self._lock:
                if self._active is reservation:
                    self._active = None
            del reservation
            del captured
            _raise_playback_error(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED)
        payload, sample_rate_hz = captured
        captured = None
        run = _PlaybackRun(
            playback_token=playback_token,
            payload=payload,
            sample_rate_hz=sample_rate_hz,
            stop_event=threading.Event(),
        )
        thread = threading.Thread(target=self._play, args=(run,), daemon=True)
        run.thread = thread
        start_failed = False
        with self._lock:
            if self._active is not reservation or reservation.cancelled:
                start_failed = True
            else:
                self._active = run
                try:
                    thread.start()
                except Exception:
                    self._active = None
                    start_failed = True
        if start_failed:
            run.stop_event.set()
            run.payload = b""
            payload = b""
            captured = None
            del reservation
            del run
            del thread
            del payload
            _raise_playback_error(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED)
        payload = b""

    def _play(self, run: _PlaybackRun) -> None:
        succeeded = False
        try:
            factory = self._runtime.RawOutputStream  # type: ignore[attr-defined]
            if not callable(factory):
                raise TypeError
            stream = factory(
                device=self._output_index,
                samplerate=run.sample_rate_hz,
                channels=1,
                dtype=_SAMPLE_DTYPE,
            )
            with run.resource_lock:
                run.stream = stream
                should_start = not run.stop_event.is_set()
            if not should_start:
                raise RuntimeError
            stream.start()  # type: ignore[attr-defined]
            with run.resource_lock:
                run.stream_started = True
            if run.stop_event.is_set():
                _interrupt_output_stream(run)
            offset = 0
            while offset < len(run.payload) and not run.stop_event.is_set():
                end = min(offset + _CHUNK_BYTES, len(run.payload))
                stream.write(run.payload[offset:end])  # type: ignore[attr-defined]
                offset = end
            succeeded = offset == len(run.payload) and not run.stop_event.is_set()
        except Exception:
            succeeded = False
        if not _release_output_stream(run):
            succeeded = False
        run.payload = b""
        with self._lock:
            if self._active is run:
                self._active = None
                if not run.stop_event.is_set():
                    self._events.put(
                        PlaybackDeviceEvent(
                            playback_token=run.playback_token,
                            succeeded=succeeded,
                        )
                    )

    def stop(self) -> None:
        with self._lock:
            active = self._active
            if active is None:
                return
            if type(active) is _PlaybackReservation:
                active.cancelled = True
                self._active = None
                return
            if type(active) is not _PlaybackRun:
                invalid_active = True
            else:
                invalid_active = False
                active.stop_event.set()
        if invalid_active:
            active = None
            _raise_playback_error(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED)
        if type(active) is not _PlaybackRun:
            return
        interrupted = _interrupt_output_stream(active)
        thread = active.thread
        if thread is not None:
            thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
        alive = thread is not None and thread.is_alive()
        if not alive:
            with self._lock:
                if self._active is active:
                    self._active = None
        if alive:
            active.payload = b""
        failed = alive or not interrupted
        active = None
        thread = None
        if failed:
            _raise_playback_error(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED)

    def drain_events(self) -> tuple[PlaybackDeviceEvent, ...]:
        events: list[PlaybackDeviceEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return tuple(events)
