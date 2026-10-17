from __future__ import annotations

import struct
import threading
from contextlib import suppress
from pathlib import Path
from typing import NoReturn

from companion.speech.local_audio_devices import (
    INPUT_SAMPLE_RATE_HZ,
    LocalAudioDeviceConfig,
    _load_sounddevice_runtime,
    _resolve_audio_devices_for_runtime,
    _ResolvedAudioRuntime,
    _validate_resolved_audio_runtime,
)
from companion.speech.local_errors import (
    LocalSpeechAdapterError,
    LocalSpeechAdapterErrorCode,
    raise_local_speech_error,
)
from companion.speech.local_wav import OwnedWavFile, write_validated_pcm_wav
from companion.speech.pcm_resampler import resample_pcm16

_SAMPLE_DTYPE = "int16"
_STARTING = object()


def _raise_recording_error(code: LocalSpeechAdapterErrorCode) -> NoReturn:
    raise_local_speech_error(code)


class _CaptureState:
    __slots__ = ("_accepting", "_chunks", "_failed", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: list[bytes] = []
        self._failed = False
        self._accepting = True

    def callback(self, indata: object, frames: object, _time: object, status: object) -> None:
        try:
            payload = bytes(indata)  # type: ignore[arg-type]
            failed = (
                bool(status) or type(frames) is not int or frames < 0 or len(payload) != frames * 2
            )
        except Exception:
            payload = b""
            failed = True
        with self._lock:
            if not self._accepting:
                return
            if failed:
                self._failed = True
                return
            self._chunks.append(payload)

    def finish(self) -> tuple[tuple[bytes, ...], bool]:
        with self._lock:
            self._accepting = False
            chunks = tuple(self._chunks)
            self._chunks.clear()
            failed = self._failed
            self._failed = False
        return chunks, failed


def _release_stream(stream: object, *, started: bool) -> bool:
    failed = False
    if started:
        try:
            stream.stop()  # type: ignore[attr-defined]
        except Exception:
            failed = True
    try:
        stream.close()  # type: ignore[attr-defined]
    except Exception:
        failed = True
    return not failed


class SoundDeviceRecordingSession:
    __slots__ = (
        "_input_sample_rate_hz",
        "_lock",
        "_release_owner",
        "_run_root",
        "_state",
        "_stream",
    )

    def __init__(
        self,
        *,
        stream: object,
        state: _CaptureState,
        run_root: Path,
        input_sample_rate_hz: int,
        release_owner: object,
    ) -> None:
        self._stream: object | None = stream
        self._state = state
        self._run_root = run_root
        self._input_sample_rate_hz = input_sample_rate_hz
        self._release_owner = release_owner
        self._lock = threading.Lock()

    def _detach(self) -> object | None:
        with self._lock:
            stream = self._stream
            self._stream = None
        return stream

    def _notify_released(self) -> None:
        callback = self._release_owner
        self._release_owner = None
        if callable(callback):
            with suppress(Exception):
                callback(self)

    def stop(self) -> OwnedWavFile:
        stream = self._detach()
        if stream is None:
            _raise_recording_error(LocalSpeechAdapterErrorCode.RECORDING_FAILED)
        chunks, capture_failed = self._state.finish()
        released = _release_stream(stream, started=True)
        self._notify_released()
        del stream
        if not released:
            del chunks
            _raise_recording_error(LocalSpeechAdapterErrorCode.CLEANUP_FAILED)
        payload = b"".join(chunks)
        del chunks
        if capture_failed or not payload or len(payload) % 2:
            del payload
            _raise_recording_error(LocalSpeechAdapterErrorCode.RECORDING_FAILED)
        try:
            samples = tuple(item[0] for item in struct.iter_unpack("<h", payload))
        except Exception:
            samples = None
        del payload
        if samples is None or not samples:
            del samples
            _raise_recording_error(LocalSpeechAdapterErrorCode.RECORDING_FAILED)
        try:
            samples = resample_pcm16(
                samples,
                self._input_sample_rate_hz,
                INPUT_SAMPLE_RATE_HZ,
            )
        except Exception:
            samples = None
        if samples is None or not samples:
            samples = None
            _raise_recording_error(LocalSpeechAdapterErrorCode.RECORDING_FAILED)
        write_error_code: LocalSpeechAdapterErrorCode | None = None
        audio: OwnedWavFile | None = None
        try:
            audio = write_validated_pcm_wav(
                self._run_root,
                samples,
                INPUT_SAMPLE_RATE_HZ,
            )
        except LocalSpeechAdapterError as error:
            write_error_code = (
                LocalSpeechAdapterErrorCode.CLEANUP_FAILED
                if error.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
                else LocalSpeechAdapterErrorCode.RECORDING_FAILED
            )
            del error
        except Exception as error:
            del error
            write_error_code = LocalSpeechAdapterErrorCode.RECORDING_FAILED
        if write_error_code is not None or audio is None:
            del samples
            del audio
            _raise_recording_error(write_error_code or LocalSpeechAdapterErrorCode.RECORDING_FAILED)
        del samples
        return audio

    def abort(self) -> None:
        stream = self._detach()
        if stream is None:
            return
        chunks, _failed = self._state.finish()
        del chunks
        released = _release_stream(stream, started=True)
        self._notify_released()
        del stream
        if not released:
            _raise_recording_error(LocalSpeechAdapterErrorCode.CLEANUP_FAILED)

    def close(self) -> None:
        self.abort()


class SoundDeviceAudioRecorder:
    __slots__ = (
        "_active_session",
        "_input_index",
        "_input_sample_rate_hz",
        "_lock",
        "_run_root",
        "_runtime",
    )

    def __init__(self, config: LocalAudioDeviceConfig) -> None:
        if type(config) is not LocalAudioDeviceConfig:
            del config
            _raise_recording_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        try:
            runtime = _load_sounddevice_runtime()
        except Exception as error:
            del error
            runtime = None
        if runtime is None:
            del config
            del runtime
            _raise_recording_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        try:
            pair = _resolve_audio_devices_for_runtime(config, runtime)
        except Exception as error:
            del error
            pair = None
        if pair is None:
            del config
            del runtime
            del pair
            _raise_recording_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        self._assign_fields(
            runtime=runtime,
            input_index=pair.input_index,
            input_sample_rate_hz=pair.input_sample_rate_hz,
            run_root=config.run_root,
        )

    def _assign_fields(
        self, *, runtime: object, input_index: int, input_sample_rate_hz: int, run_root: Path
    ) -> None:
        self._run_root = run_root
        self._input_index = input_index
        self._input_sample_rate_hz = input_sample_rate_hz
        self._runtime = runtime
        self._active_session: object | SoundDeviceRecordingSession | None = None
        self._lock = threading.Lock()

    @classmethod
    def _create(
        cls,
        *,
        runtime: object,
        input_index: int,
        input_sample_rate_hz: int,
        run_root: Path,
    ) -> SoundDeviceAudioRecorder:
        instance = object.__new__(cls)
        instance._assign_fields(
            runtime=runtime,
            input_index=input_index,
            input_sample_rate_hz=input_sample_rate_hz,
            run_root=run_root,
        )
        return instance

    @classmethod
    def from_config(cls, config: LocalAudioDeviceConfig) -> SoundDeviceAudioRecorder:
        try:
            recorder = cls(config)
        except Exception as error:
            del error
            recorder = None
        if recorder is None:
            del config
            del recorder
            _raise_recording_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        return recorder

    @classmethod
    def from_resolved(
        cls,
        config: LocalAudioDeviceConfig,
        resolved: _ResolvedAudioRuntime,
    ) -> SoundDeviceAudioRecorder:
        try:
            validated = _validate_resolved_audio_runtime(config, resolved)
            recorder = cls._create(
                runtime=validated.runtime,
                input_index=validated.pair.input_index,
                input_sample_rate_hz=validated.pair.input_sample_rate_hz,
                run_root=config.run_root,
            )
        except Exception as error:
            del error
            recorder = None
        if recorder is None:
            del config
            del resolved
            del recorder
            _raise_recording_error(LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED)
        return recorder

    def _release_session(self, session: SoundDeviceRecordingSession) -> None:
        with self._lock:
            if self._active_session is session:
                self._active_session = None

    def _clear_starting_reservation(self) -> None:
        with self._lock:
            if self._active_session is _STARTING:
                self._active_session = None

    def start(self) -> SoundDeviceRecordingSession:
        with self._lock:
            if self._active_session is not None:
                _raise_recording_error(LocalSpeechAdapterErrorCode.RECORDING_FAILED)
            self._active_session = _STARTING
        state = _CaptureState()
        try:
            factory = self._runtime.RawInputStream  # type: ignore[attr-defined]
            if not callable(factory):
                raise TypeError
            stream = factory(
                device=self._input_index,
                samplerate=self._input_sample_rate_hz,
                channels=1,
                dtype=_SAMPLE_DTYPE,
                callback=state.callback,
            )
        except Exception:
            stream = None
        if stream is None:
            self._clear_starting_reservation()
            del state
            del stream
            _raise_recording_error(LocalSpeechAdapterErrorCode.RECORDING_FAILED)
        session = SoundDeviceRecordingSession(
            stream=stream,
            state=state,
            run_root=self._run_root,
            input_sample_rate_hz=self._input_sample_rate_hz,
            release_owner=self._release_session,
        )
        start_failed = False
        try:
            stream.start()  # type: ignore[attr-defined]
        except Exception as error:
            del error
            start_failed = True
        if start_failed:
            released = _release_stream(stream, started=False)
            self._clear_starting_reservation()
            del stream
            del state
            del session
            if not released:
                _raise_recording_error(LocalSpeechAdapterErrorCode.CLEANUP_FAILED)
            _raise_recording_error(LocalSpeechAdapterErrorCode.RECORDING_FAILED)
        with self._lock:
            if self._active_session is not _STARTING:
                raise RuntimeError("invalid recorder reservation")
            self._active_session = session
        return session
