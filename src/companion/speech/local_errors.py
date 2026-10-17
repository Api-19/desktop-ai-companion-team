from __future__ import annotations

from enum import StrEnum
from typing import NoReturn


class LocalSpeechAdapterErrorCode(StrEnum):
    CONFIGURATION_FAILED = "configuration-failed"
    DEVICE_CONFIGURATION_FAILED = "device-configuration-failed"
    INVALID_AUDIO = "invalid-audio"
    PLAYBACK_FAILED = "playback-failed"
    RECORDING_FAILED = "recording-failed"
    RECOGNITION_FAILED = "recognition-failed"
    SYNTHESIS_FAILED = "synthesis-failed"
    CLEANUP_FAILED = "cleanup-failed"


class LocalSpeechAdapterError(RuntimeError):
    def __init__(self, code: LocalSpeechAdapterErrorCode) -> None:
        if type(code) is not LocalSpeechAdapterErrorCode:
            raise TypeError("code must be LocalSpeechAdapterErrorCode")
        self.code = code
        super().__init__(code.value)


def raise_local_speech_error(code: LocalSpeechAdapterErrorCode) -> NoReturn:
    raise LocalSpeechAdapterError(code)
