from __future__ import annotations

import math
import struct
from numbers import Real

from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_model_config import (
    LocalSpeechModelConfig,
    VerifiedLocalSpeechPaths,
    load_sherpa_runtime,
    verify_local_speech_config,
)
from companion.speech.pcm_audio import PcmAudio

_SAMPLE_RATE_HZ = 44_100


def _raise(code: LocalSpeechAdapterErrorCode) -> None:
    raise LocalSpeechAdapterError(code)


def _load_runtime_and_paths(
    config: LocalSpeechModelConfig,
) -> tuple[VerifiedLocalSpeechPaths, object]:
    return verify_local_speech_config(config), load_sherpa_runtime(config)


def _float_to_int16(value: float) -> int:
    if value == -1.0:
        return -32_768
    scaled = value * 32_767.0
    return math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)


def _pcm16le(output: object) -> bytes | None:
    try:
        sample_rate = output.sample_rate  # type: ignore[attr-defined]
        raw_samples = output.samples  # type: ignore[attr-defined]
        samples = tuple(raw_samples)
    except Exception:
        return None
    if type(sample_rate) is not int or sample_rate != _SAMPLE_RATE_HZ or not samples:
        return None

    converted: list[int] = []
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, Real):
            return None
        value = float(sample)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            return None
        converted.append(_float_to_int16(value))
    if not any(converted):
        return None
    return struct.pack(f"<{len(converted)}h", *converted)


class SherpaVitsSynthesizer:
    __slots__ = ("__busy", "__closed", "__engine")

    def __init__(self, config: LocalSpeechModelConfig) -> None:
        self.__busy = False
        self.__closed = True
        self.__engine: object | None = None
        if type(config) is not LocalSpeechModelConfig:
            _raise(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)

        engine: object | None = None
        try:
            paths, runtime = _load_runtime_and_paths(config)
            vits = runtime.OfflineTtsVitsModelConfig(  # type: ignore[attr-defined]
                model=str(paths.vits_model),
                lexicon=str(paths.vits_lexicon),
                tokens=str(paths.vits_tokens),
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0,
            )
            model = runtime.OfflineTtsModelConfig(  # type: ignore[attr-defined]
                vits=vits,
                num_threads=4,
                debug=False,
                provider="cpu",
            )
            tts_config = runtime.OfflineTtsConfig(  # type: ignore[attr-defined]
                model=model,
                rule_fsts=",".join(str(path) for path in paths.vits_rule_fsts),
                max_num_sentences=1,
            )
            if tts_config.validate() is not True:
                raise ValueError("invalid TTS config")
            paths = verify_local_speech_config(config)
            engine = runtime.OfflineTts(tts_config)  # type: ignore[attr-defined]
        except Exception:
            engine = None
        if engine is None:
            runtime = None  # type: ignore[assignment]
            paths = None  # type: ignore[assignment]
            vits = None  # type: ignore[assignment]
            model = None  # type: ignore[assignment]
            tts_config = None  # type: ignore[assignment]
            _raise(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)
        self.__engine = engine
        self.__closed = False

    @classmethod
    def from_config(cls, config: LocalSpeechModelConfig) -> SherpaVitsSynthesizer:
        return cls(config)

    def close(self) -> None:
        self.__engine = None
        self.__closed = True

    def synthesize(self, text: str) -> PcmAudio:
        if self.__closed or self.__busy:
            del text
            _raise(LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED)
        if type(text) is not str or not text.strip():
            del text
            _raise(LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED)

        engine = self.__engine
        if engine is None:
            del text
            _raise(LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED)

        audio: PcmAudio | None = None
        output: object | None = None
        pcm: bytes | None = None
        self.__busy = True
        try:
            output = engine.generate(text=text, sid=0, speed=1.0)  # type: ignore[attr-defined]
            pcm = _pcm16le(output)
            if pcm is None:
                raise ValueError("invalid TTS output")
            audio = PcmAudio(pcm=pcm, sample_rate_hz=_SAMPLE_RATE_HZ)
        except Exception:
            audio = None
        finally:
            self.__busy = False

        output = None
        pcm = None
        engine = None
        if audio is not None:
            return audio
        del text
        _raise(LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED)
