from __future__ import annotations

import importlib
import importlib.metadata
import math
from collections.abc import Iterable
from typing import NoReturn

from companion.speech.local_errors import LocalSpeechAdapterErrorCode, raise_local_speech_error

SAMPLERATE_VERSION = "0.2.4"
_CONVERTER_TYPE = "sinc_best"
_MAX_AUDIO_SECONDS = 30
_MAX_SAMPLE_RATE_HZ = 48_000
_MAX_INPUT_FRAMES = _MAX_AUDIO_SECONDS * _MAX_SAMPLE_RATE_HZ


def _raise_invalid_audio() -> NoReturn:
    raise_local_speech_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)


def _load_resampler_runtime() -> object:
    runtime: object | None = None
    failed = False
    try:
        version = importlib.metadata.version("samplerate")
        if type(version) is not str or version != SAMPLERATE_VERSION:
            raise ValueError
        candidate = importlib.import_module("samplerate")
        if not callable(getattr(candidate, "resample", None)):
            raise TypeError
        runtime = candidate
    except Exception:
        failed = True
    if failed or runtime is None:
        runtime = None
        _raise_invalid_audio()
    return runtime


def _valid_inputs(
    samples: object, source_rate_hz: object, target_rate_hz: object
) -> tuple[int, ...] | None:
    if (
        type(samples) is not tuple
        or not samples
        or type(source_rate_hz) is not int
        or source_rate_hz <= 0
        or type(target_rate_hz) is not int
        or target_rate_hz <= 0
        or len(samples) > min(_MAX_INPUT_FRAMES, source_rate_hz * _MAX_AUDIO_SECONDS)
    ):
        return None
    if any(type(sample) is not int or not -32768 <= sample <= 32767 for sample in samples):
        return None
    return samples


def _try_resample(
    normalized: tuple[float, ...], source_rate_hz: int, target_rate_hz: int
) -> tuple[int, ...] | None:
    result: tuple[int, ...] | None = None
    try:
        runtime = _load_resampler_runtime()
        output = runtime.resample(  # type: ignore[attr-defined]
            normalized,
            target_rate_hz / source_rate_hz,
            converter_type=_CONVERTER_TYPE,
        )
        if not isinstance(output, Iterable) or isinstance(output, str | bytes | bytearray):
            raise TypeError
        expected_frames = round(len(normalized) * target_rate_hz / source_rate_hz)
        if expected_frames <= 0 or expected_frames > min(
            _MAX_INPUT_FRAMES, target_rate_hz * _MAX_AUDIO_SECONDS
        ):
            raise ValueError
        maximum_frames = min(_MAX_INPUT_FRAMES, expected_frames + 1)
        converted: list[int] = []
        for index, value in enumerate(output):
            if index >= maximum_frames:
                raise ValueError
            if type(value) not in (float, int) and type(value).__module__ != "numpy":
                raise TypeError
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError
            scaled = round(numeric * 32768.0)
            converted.append(min(32767, max(-32768, scaled)))
        if abs(len(converted) - expected_frames) > 1 or not converted:
            raise ValueError
        result = tuple(converted)
    except Exception:
        result = None
    return result


def resample_pcm16(
    samples: tuple[int, ...], source_rate_hz: int, target_rate_hz: int
) -> tuple[int, ...]:
    validated = _valid_inputs(samples, source_rate_hz, target_rate_hz)
    if validated is None:
        samples = ()
        _raise_invalid_audio()
    if source_rate_hz == target_rate_hz:
        return tuple(validated)

    normalized = tuple(sample / 32768.0 for sample in validated)
    result = _try_resample(normalized, source_rate_hz, target_rate_hz)
    samples = ()
    validated = None
    normalized = ()
    source_rate_hz = 0
    target_rate_hz = 0
    if result is None:
        result = None
        _raise_invalid_audio()
    return result
