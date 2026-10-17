from __future__ import annotations

import struct

import pytest

from companion.contracts.body import MAX_OUTPUT_SAMPLE_RATE_HZ, MIN_OUTPUT_SAMPLE_RATE_HZ
from companion.speech.controller_ports import SynthesizedAudio
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.pcm_audio import PcmAudio, pcm16_samples

_PCM = struct.pack("<4h", -2000, 2000, -32_768, 32_767)


def test_pcm_audio_exposes_play_audio_compatible_fields() -> None:
    audio = PcmAudio(pcm=_PCM, sample_rate_hz=44_100)

    assert audio.pcm == _PCM
    assert audio.sample_rate_hz == 44_100
    assert audio.channels == 1
    assert audio.sample_format == "pcm_s16le"
    assert pcm16_samples(audio) == (-2000, 2000, -32_768, 32_767)
    assert isinstance(audio, SynthesizedAudio)


@pytest.mark.parametrize("rate", [MIN_OUTPUT_SAMPLE_RATE_HZ, MAX_OUTPUT_SAMPLE_RATE_HZ])
def test_pcm_audio_accepts_play_audio_rate_bounds(rate: int) -> None:
    assert PcmAudio(pcm=_PCM, sample_rate_hz=rate).sample_rate_hz == rate


@pytest.mark.parametrize(
    ("pcm", "rate"),
    [
        (b"", 44_100),
        (b"\x00\x01\x02", 44_100),
        (bytearray(_PCM), 44_100),
        ("pcm", 44_100),
        (_PCM, MIN_OUTPUT_SAMPLE_RATE_HZ - 1),
        (_PCM, MAX_OUTPUT_SAMPLE_RATE_HZ + 1),
        (_PCM, True),
        (_PCM, 44_100.0),
    ],
)
def test_pcm_audio_rejects_values_play_audio_would_reject(pcm: object, rate: object) -> None:
    with pytest.raises(LocalSpeechAdapterError) as caught:
        PcmAudio(pcm=pcm, sample_rate_hz=rate)  # type: ignore[arg-type]

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO


def test_pcm_audio_fields_are_read_only() -> None:
    audio = PcmAudio(pcm=_PCM, sample_rate_hz=44_100)

    for name, value in (
        ("pcm", b"\x00\x00"),
        ("sample_rate_hz", 16_000),
        ("channels", 2),
        ("sample_format", "f32"),
    ):
        with pytest.raises(AttributeError):
            setattr(audio, name, value)


def test_close_releases_pcm_and_is_idempotent() -> None:
    audio = PcmAudio(pcm=_PCM, sample_rate_hz=44_100)

    assert audio.close() is None
    assert audio.close() is None

    with pytest.raises(LocalSpeechAdapterError) as caught:
        _ = audio.pcm
    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    with pytest.raises(LocalSpeechAdapterError):
        pcm16_samples(audio)


def test_repr_does_not_render_pcm() -> None:
    marker = struct.pack("<2h", 12_345, -23_456)
    audio = PcmAudio(pcm=marker, sample_rate_hz=44_100)

    assert "12345" not in repr(audio)
    assert repr(marker) not in repr(audio)


def test_pcm16_samples_rejects_non_pcm_audio() -> None:
    with pytest.raises(LocalSpeechAdapterError) as caught:
        pcm16_samples(object())  # type: ignore[arg-type]

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
