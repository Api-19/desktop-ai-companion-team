"""Contract every pluggable speech synthesizer must satisfy.

To plug in a new engine, add a factory to ``_SYNTHESIZER_FACTORIES`` that builds
it against a fake runtime (no real model, no network).
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import companion.speech.sherpa_vits_synthesizer as vits_module
from companion.contracts.body import MAX_OUTPUT_SAMPLE_RATE_HZ, MIN_OUTPUT_SAMPLE_RATE_HZ
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.pcm_audio import PcmAudio
from companion.speech.sherpa_vits_synthesizer import SherpaVitsSynthesizer
from tests.speech.test_local_model_config import make_config
from tests.speech.test_sherpa_vits_synthesizer import _paths, _Runtime

PUBLIC_TEXT = "合成公开测试文本"


class FakePcmSynthesizer:
    """Reference engine: a fixed short square wave per request."""

    def __init__(self) -> None:
        self._closed = False

    def synthesize(self, text: str) -> PcmAudio:
        if self._closed or type(text) is not str or not text.strip():
            raise LocalSpeechAdapterError(LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED)
        return PcmAudio(pcm=struct.pack("<4h", 1000, -1000, 1000, -1000), sample_rate_hz=22_050)

    def close(self) -> None:
        self._closed = True


def _fake(_tmp_path: Path, _monkeypatch: pytest.MonkeyPatch) -> object:
    return FakePcmSynthesizer()


def _sherpa_vits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    runtime = _Runtime()
    monkeypatch.setattr(vits_module, "load_sherpa_runtime", lambda _config: runtime)
    monkeypatch.setattr(vits_module, "verify_local_speech_config", _paths)
    return SherpaVitsSynthesizer.from_config(make_config(tmp_path))


_SYNTHESIZER_FACTORIES: dict[str, Callable[[Path, pytest.MonkeyPatch], object]] = {
    "fake": _fake,
    "sherpa-vits": _sherpa_vits,
}


@pytest.fixture(params=sorted(_SYNTHESIZER_FACTORIES))
def synthesizer(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[object]:
    engine = _SYNTHESIZER_FACTORIES[request.param](tmp_path, monkeypatch)
    yield engine
    engine.close()  # type: ignore[attr-defined]


def test_synthesize_returns_play_audio_compatible_pcm(synthesizer: object) -> None:
    audio = synthesizer.synthesize(PUBLIC_TEXT)  # type: ignore[attr-defined]

    assert type(audio) is PcmAudio
    assert audio.sample_format == "pcm_s16le"
    assert audio.channels == 1
    assert MIN_OUTPUT_SAMPLE_RATE_HZ <= audio.sample_rate_hz <= MAX_OUTPUT_SAMPLE_RATE_HZ
    assert len(audio.pcm) >= 2
    assert len(audio.pcm) % 2 == 0
    audio.close()


def test_each_request_returns_an_independent_audio_object(synthesizer: object) -> None:
    first = synthesizer.synthesize(PUBLIC_TEXT)  # type: ignore[attr-defined]
    second = synthesizer.synthesize(PUBLIC_TEXT)  # type: ignore[attr-defined]

    assert first is not second
    first.close()
    assert len(second.pcm) >= 2
    second.close()


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_fails_with_synthesis_failed(synthesizer: object, text: str) -> None:
    with pytest.raises(LocalSpeechAdapterError) as caught:
        synthesizer.synthesize(text)  # type: ignore[attr-defined]

    assert caught.value.code is LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED


def test_close_is_idempotent_and_blocks_later_synthesis(synthesizer: object) -> None:
    audio = synthesizer.synthesize(PUBLIC_TEXT)  # type: ignore[attr-defined]

    assert synthesizer.close() is None  # type: ignore[attr-defined]
    assert synthesizer.close() is None  # type: ignore[attr-defined]

    with pytest.raises(LocalSpeechAdapterError) as caught:
        synthesizer.synthesize(PUBLIC_TEXT)  # type: ignore[attr-defined]
    assert caught.value.code is LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED
    assert len(audio.pcm) >= 2
    audio.close()
