from __future__ import annotations

import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

import companion.speech.pcm_resampler as resampler_module
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.pcm_resampler import resample_pcm16


def test_same_rate_returns_exact_pcm_without_loading_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resampler_module,
        "_load_resampler_runtime",
        lambda: pytest.fail("runtime must not load"),
    )

    assert resample_pcm16((-32768, 0, 32767), 16_000, 16_000) == (-32768, 0, 32767)


def test_resamples_normalized_pcm_with_frozen_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, float, str]] = []

    def resample(values: object, ratio: float, converter_type: str) -> list[float]:
        calls.append((values, ratio, converter_type))
        return [-1.0, 0.0, 0.999969482421875]

    monkeypatch.setattr(
        resampler_module,
        "_load_resampler_runtime",
        lambda: SimpleNamespace(resample=resample),
    )

    samples = (-32768, 0, 32767) * 3
    result = resample_pcm16(samples, 48_000, 16_000)

    assert result == (-32768, 0, 32767)
    assert calls == [(((-1.0, 0.0, 32767 / 32768.0) * 3), 1 / 3, "sinc_best")]


@pytest.mark.parametrize(
    "result",
    [[], [float("nan")], [float("inf")], [[0.0]], object()],
)
def test_malformed_resampler_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    monkeypatch.setattr(
        resampler_module,
        "_load_resampler_runtime",
        lambda: SimpleNamespace(resample=lambda *_args, **_kwargs: result),
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        resample_pcm16((100,), 44_100, 48_000)

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_overlong_or_nonterminating_output_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    produced = 0

    def endless() -> object:
        nonlocal produced
        while True:
            produced += 1
            yield 0.0

    monkeypatch.setattr(
        resampler_module,
        "_load_resampler_runtime",
        lambda: SimpleNamespace(resample=lambda *_args, **_kwargs: endless()),
    )

    with pytest.raises(LocalSpeechAdapterError):
        resample_pcm16((100,), 44_100, 48_000)

    assert produced <= 3


def test_oversized_input_fails_before_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resampler_module,
        "_load_resampler_runtime",
        lambda: pytest.fail("runtime must not load"),
    )
    samples = (0,) * (resampler_module._MAX_INPUT_FRAMES + 1)

    with pytest.raises(LocalSpeechAdapterError):
        resample_pcm16(samples, 48_000, 16_000)


def test_failure_traceback_does_not_retain_pcm_or_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = 12345
    runtime = SimpleNamespace(resample=lambda *_args, **_kwargs: [float("nan"), marker])
    monkeypatch.setattr(resampler_module, "_load_resampler_runtime", lambda: runtime)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        resample_pcm16((marker,), 48_000, 16_000)

    frames = traceback.extract_tb(caught.value.__traceback__)
    assert all(frame.name != "_try_resample" for frame in frames)
    current = caught.value.__traceback__
    while current is not None:
        if Path(current.tb_frame.f_code.co_filename).name == "pcm_resampler.py":
            assert runtime not in current.tb_frame.f_locals.values()
            assert (marker,) not in current.tb_frame.f_locals.values()
        current = current.tb_next


def test_pinned_runtime_resamples_public_tones_with_expected_lengths() -> None:
    pytest.importorskip("samplerate")
    input_48k = tuple(round(8000 * math.sin(2 * math.pi * 440 * i / 48_000)) for i in range(4800))
    input_44k = tuple(round(8000 * math.sin(2 * math.pi * 440 * i / 44_100)) for i in range(4410))

    downsampled = resample_pcm16(input_48k, 48_000, 16_000)
    upsampled = resample_pcm16(input_44k, 44_100, 48_000)

    assert len(downsampled) == 1600
    assert len(upsampled) == 4800
    assert max(map(abs, downsampled)) > 1000
    assert max(map(abs, upsampled)) > 1000


@pytest.mark.parametrize(
    ("samples", "source", "target"),
    [([], 48_000, 16_000), ((0,), True, 16_000), ((0,), 48_000, 0), ((True,), 48_000, 16_000)],
)
def test_invalid_inputs_fail_before_runtime(
    monkeypatch: pytest.MonkeyPatch, samples: object, source: object, target: object
) -> None:
    monkeypatch.setattr(
        resampler_module,
        "_load_resampler_runtime",
        lambda: pytest.fail("runtime must not load"),
    )

    with pytest.raises(LocalSpeechAdapterError):
        resample_pcm16(samples, source, target)  # type: ignore[arg-type]
