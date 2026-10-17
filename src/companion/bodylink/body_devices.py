"""Body-side audio devices over a sounddevice runtime: streaming microphone, speaker, resampler.

Audio only ever lives in memory here. Device selection reuses the speech module's selector
rules (device name, optional host API name); the body has no run directory, so it resolves
devices directly instead of going through ``LocalAudioDeviceConfig``.
"""

import math
import threading
from collections.abc import Callable
from typing import Any, Protocol

from companion.contracts.body import INPUT_SAMPLE_RATE_HZ
from companion.speech.local_audio_devices import ResolvedAudioDevicePair, resolve_with_runtime

_DTYPE = "int16"
_INPUT_BLOCK_SECONDS = 0.1
_OUTPUT_CHUNK_BYTES = 2048


class DeviceNotFoundError(RuntimeError):
    """The configured input/output device (or host API) does not resolve to exactly one device."""


def resolve_body_devices(
    runtime: Any, input_selector: str, output_selector: str, hostapi_selector: str | None
) -> ResolvedAudioDevicePair:
    """Input rate is 16000 or 48000, output rate 44100 or 48000 (first one the device accepts)."""
    pair = resolve_with_runtime(runtime, input_selector, output_selector, hostapi_selector)
    if pair is None:
        raise DeviceNotFoundError("audio device not found or unsupported")
    return pair


class InputHandle(Protocol):
    def close(self) -> None: ...


class Microphone(Protocol):
    sample_rate: int

    def open(self, on_audio: Callable[[bytes], None]) -> InputHandle: ...


class Speaker(Protocol):
    def play(self, pcm: bytes, sample_rate: int, stop: threading.Event) -> bool: ...

    def abort(self) -> None: ...


class _InputStreamHandle:
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class SoundDeviceMicrophone:
    """Mono PCM16 capture at the device rate; ``on_audio`` runs on the audio callback thread."""

    def __init__(self, runtime: Any, device: int, sample_rate: int) -> None:
        self._runtime = runtime
        self._device = device
        self.sample_rate = sample_rate

    def open(self, on_audio: Callable[[bytes], None]) -> InputHandle:
        def callback(indata: Any, _frames: int, _time: Any, _status: Any) -> None:
            on_audio(bytes(indata))

        stream = self._runtime.RawInputStream(
            device=self._device,
            samplerate=self.sample_rate,
            channels=1,
            dtype=_DTYPE,
            blocksize=int(self.sample_rate * _INPUT_BLOCK_SECONDS),
            callback=callback,
        )
        stream.start()
        return _InputStreamHandle(stream)


class SoundDeviceSpeaker:
    def __init__(self, runtime: Any, device: int) -> None:
        self._runtime = runtime
        self._device = device
        self._stream_lock = threading.Lock()
        self._active_stream: Any | None = None
        self._active_stream_aborted = False

    def abort(self) -> None:
        with self._stream_lock:
            if self._active_stream is None or self._active_stream_aborted:
                return
            self._active_stream_aborted = True
            self._active_stream.abort()

    def play(self, pcm: bytes, sample_rate: int, stop: threading.Event) -> bool:
        """Blocks until played (True) or ``stop`` is set (False); device errors propagate."""
        if stop.is_set():
            return False
        stream = None
        try:
            stream = self._runtime.RawOutputStream(
                device=self._device, samplerate=sample_rate, channels=1, dtype=_DTYPE
            )
            with self._stream_lock:
                if stop.is_set():
                    return False
                self._active_stream = stream
                self._active_stream_aborted = False
                stream.start()
            for offset in range(0, len(pcm), _OUTPUT_CHUNK_BYTES):
                if stop.is_set():
                    self.abort()
                    return False
                stream.write(pcm[offset : offset + _OUTPUT_CHUNK_BYTES])
            stream.stop()  # waits until the buffered audio has been played
            return not stop.is_set()
        except Exception:
            if stop.is_set():
                return False
            raise
        finally:
            if stream is not None:
                with self._stream_lock:
                    self._active_stream = None
                    stream.close()


class Pcm16StreamResampler:
    """Streams PCM16 mono from ``source_rate`` to 16 kHz. ``finish()`` returns the tail so the
    total output is exactly ``round(input_frames * 16000 / source_rate)`` samples."""

    def __init__(self, source_rate: int) -> None:
        self._ratio = INPUT_SAMPLE_RATE_HZ / source_rate
        self._input_frames = 0
        self._output_frames = 0
        self._resampler: Any = None
        if source_rate != INPUT_SAMPLE_RATE_HZ:
            import samplerate

            self._resampler = samplerate.Resampler("sinc_best", channels=1)

    def process(self, pcm: bytes) -> bytes:
        if self._resampler is None:
            return pcm
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._input_frames += len(samples)
        return self._emit(self._resampler.process(samples, self._ratio))

    def finish(self) -> bytes:
        if self._resampler is None:
            return b""
        import numpy as np

        expected = round(self._input_frames * self._ratio)
        tail = b""
        # libsamplerate holds back its filter delay; push silence through to release it.
        while (missing := expected - self._output_frames) > 0:
            padding = np.zeros(math.ceil(missing / self._ratio) + 256, dtype=np.float32)
            out = self._resampler.process(padding, self._ratio)
            tail += self._emit(out[:missing])
        return tail

    def _emit(self, out: Any) -> bytes:
        import numpy as np

        self._output_frames += len(out)
        return (np.clip(np.rint(out * 32768.0), -32768, 32767).astype("<i2")).tobytes()
