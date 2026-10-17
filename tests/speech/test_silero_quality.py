import math

import pytest

from companion.speech.local_wav import PcmWavData
from companion.speech.silero_quality import (
    SileroInterval,
    analyze_silero_quality,
    build_audio_quality_evidence,
    normalize_silero_intervals,
)


def _pcm(samples: tuple[int, ...]) -> PcmWavData:
    return PcmWavData(
        sample_rate_hz=16_000,
        channels=1,
        sample_width_bytes=2,
        frame_count=len(samples),
        samples_int16=samples,
        samples_float=tuple(sample / 32768.0 for sample in samples),
    )


def _guarded_pcm() -> PcmWavData:
    return _pcm((0,) * 800 + (16_384,) * 800 + (8_192,) * 1_600 + (16_384,) * 800 + (0,) * 800)


def test_overlapping_and_adjacent_intervals_sort_clip_and_merge() -> None:
    assert normalize_silero_intervals(
        [(600, 1_100), (250, 400), (400, 500), (-50, 100), (100, 300)], 1_000
    ) == (SileroInterval(0, 500), SileroInterval(600, 1_000))


class _IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "intervals",
    [
        ((True, 10),),
        ((_IntSubclass(1), 10),),
        ((1.0, 10),),
        ((float("nan"), 10),),
        (("1", 10),),
        ((10, 10),),
        ((10, 5),),
        ((1_000, 1_100),),
        ((-100, 0),),
        ((1, 2, 3),),
        "not-interval-pairs",
        ((item for item in [(1, 2)]),),
        (item for item in [(1, 2)]),
    ],
)
def test_interval_normalization_rejects_ambiguous_or_invalid_inputs(intervals: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_silero_intervals(intervals, 1_000)  # type: ignore[arg-type]


@pytest.mark.parametrize("frame_count", [True, _IntSubclass(1_000), 1.0, -1])
def test_interval_normalization_requires_exact_nonnegative_frame_count(frame_count: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_silero_intervals((), frame_count)  # type: ignore[arg-type]


def test_guard_bands_affect_only_noise_and_exact_floor_is_used() -> None:
    evidence, vad_passed = analyze_silero_quality(_guarded_pcm(), [(1_600, 3_200)])

    expected = 20 * math.log10(0.25 / (1 / 32768))
    assert evidence.estimated_snr_db == pytest.approx(expected)
    assert vad_passed is True


def test_snr_uses_exact_minimum_mask_counts() -> None:
    pcm = _pcm((0,) * 800 + (0,) * 800 + (8_192,) * 1_600 + (0,) * 800 + (0,) * 800)

    evidence, vad_passed = analyze_silero_quality(pcm, [(1_600, 3_200)])

    assert vad_passed is True
    assert evidence.estimated_snr_db == pytest.approx(20 * math.log10(0.25 / (1 / 32768)))


def test_snr_is_none_when_speech_has_fewer_than_minimum_frames() -> None:
    pcm = _pcm((0,) * 800 + (8_192,) * 1_599 + (0,) * 800)

    evidence, vad_passed = analyze_silero_quality(pcm, [(800, 2_399)])

    assert vad_passed is True
    assert evidence.estimated_snr_db is None


def test_no_intervals_and_all_speech_keep_vad_and_snr_independent() -> None:
    silent = _pcm((0,) * 2_000)
    all_speech = _pcm((8_192,) * 2_000)

    no_speech_evidence, no_speech_vad = analyze_silero_quality(silent, [])
    all_speech_evidence, all_speech_vad = analyze_silero_quality(all_speech, [(0, 2_000)])

    assert no_speech_vad is False
    assert no_speech_evidence.estimated_snr_db is None
    assert all_speech_vad is True
    assert all_speech_evidence.estimated_snr_db is None


def test_full_file_metrics_derive_from_int16_endpoints() -> None:
    pcm = _pcm((-32_768, 32_767, 0))

    evidence = build_audio_quality_evidence(pcm, ())

    assert evidence.rms == pytest.approx(math.sqrt((1.0 + (32_767 / 32_768) ** 2) / 3))
    assert evidence.peak_abs == 1.0
    assert evidence.clipping_fraction == pytest.approx(2 / 3)
    assert evidence.estimated_snr_db is None


def test_snr_is_not_clamped_or_replaced_when_noise_is_louder() -> None:
    pcm = _pcm((16_384,) * 800 + (0,) * 800 + (8_192,) * 1_600 + (0,) * 800 + (16_384,) * 800)

    evidence, vad_passed = analyze_silero_quality(pcm, [(1_600, 3_200)])

    assert vad_passed is True
    assert evidence.estimated_snr_db == pytest.approx(20 * math.log10(0.25 / 0.5))


def test_builder_requires_exact_pcm_and_normalized_interval_tuple() -> None:
    class _PcmSubclass(PcmWavData):
        pass

    pcm = _pcm((0,) * 3_200)
    pcm_subclass = _PcmSubclass(
        sample_rate_hz=16_000,
        channels=1,
        sample_width_bytes=2,
        frame_count=3_200,
        samples_int16=(0,) * 3_200,
        samples_float=(0.0,) * 3_200,
    )

    with pytest.raises(TypeError):
        build_audio_quality_evidence(pcm_subclass, ())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_audio_quality_evidence(pcm, [SileroInterval(0, 1_600)])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_audio_quality_evidence(pcm, ((0, 1_600),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("attribute", "value", "error_type"),
    [
        ("start", True, TypeError),
        ("start", _IntSubclass(0), TypeError),
        ("start", -1, ValueError),
        ("end", 0, ValueError),
        ("end", 3_201, ValueError),
    ],
)
def test_builder_rechecks_each_normalized_interval_before_mask_construction(
    attribute: str,
    value: object,
    error_type: type[Exception],
) -> None:
    interval = SileroInterval(0, 1_600)
    object.__setattr__(interval, attribute, value)

    with pytest.raises(error_type):
        build_audio_quality_evidence(_pcm((0,) * 3_200), (interval,))
