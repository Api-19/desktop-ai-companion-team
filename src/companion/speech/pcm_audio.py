from __future__ import annotations

import struct
from typing import Literal

from companion.contracts.body import MAX_OUTPUT_SAMPLE_RATE_HZ, MIN_OUTPUT_SAMPLE_RATE_HZ
from companion.speech.local_errors import LocalSpeechAdapterErrorCode, raise_local_speech_error


class PcmAudio:
    """In-memory mono PCM16LE synthesis result, field-compatible with body PlayAudio."""

    __slots__ = ("__pcm", "__sample_rate_hz")

    def __init__(self, *, pcm: bytes, sample_rate_hz: int) -> None:
        if (
            type(pcm) is not bytes
            or not pcm
            or len(pcm) % 2
            or type(sample_rate_hz) is not int
            or not MIN_OUTPUT_SAMPLE_RATE_HZ <= sample_rate_hz <= MAX_OUTPUT_SAMPLE_RATE_HZ
        ):
            raise_local_speech_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
        self.__pcm: bytes | None = pcm
        self.__sample_rate_hz = sample_rate_hz

    @property
    def pcm(self) -> bytes:
        pcm = self.__pcm
        if pcm is None:
            raise_local_speech_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
        return pcm

    @property
    def sample_rate_hz(self) -> int:
        return self.__sample_rate_hz

    @property
    def channels(self) -> Literal[1]:
        return 1

    @property
    def sample_format(self) -> Literal["pcm_s16le"]:
        return "pcm_s16le"

    def close(self) -> None:
        self.__pcm = None


def pcm16_samples(audio: PcmAudio) -> tuple[int, ...]:
    if type(audio) is not PcmAudio:
        raise_local_speech_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
    pcm = audio.pcm
    return struct.unpack(f"<{len(pcm) // 2}h", pcm)
