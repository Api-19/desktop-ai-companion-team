from __future__ import annotations

import math
from dataclasses import dataclass

from companion.speech.local_wav import PcmWavData
from companion.speech.recognition_gate import AudioQualityEvidence

_INT16_MINIMUM = -32_768
_INT16_MAXIMUM = 32_767
_GUARD_FRAMES = 800
_MINIMUM_MASK_FRAMES = 1_600
_NOISE_FLOOR = 1 / 32_768


@dataclass(frozen=True, slots=True)
class SileroInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("Silero interval bounds must be ints")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Silero interval bounds are invalid")


def _validate_frame_count(frame_count: object) -> int:
    if type(frame_count) is not int:
        raise TypeError("frame_count must be an int")
    if frame_count < 0:
        raise ValueError("frame_count must be nonnegative")
    return frame_count


def normalize_silero_intervals(
    intervals: object,
    frame_count: int,
) -> tuple[SileroInterval, ...]:
    frame_count = _validate_frame_count(frame_count)
    if type(intervals) not in (list, tuple):
        raise TypeError("intervals must be a concrete sequence")

    clipped: list[SileroInterval] = []
    for interval in intervals:
        if type(interval) not in (list, tuple) or len(interval) != 2:
            raise TypeError("each interval must be a pair")
        start, end = interval
        if type(start) is not int or type(end) is not int:
            raise TypeError("Silero interval bounds must be ints")
        if end <= start:
            raise ValueError("Silero interval bounds are invalid")
        clipped_start = max(0, start)
        clipped_end = min(frame_count, end)
        if clipped_end <= clipped_start:
            raise ValueError("Silero interval is outside PCM")
        clipped.append(SileroInterval(clipped_start, clipped_end))

    merged: list[SileroInterval] = []
    for interval in sorted(clipped, key=lambda item: (item.start, item.end)):
        if merged and interval.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = SileroInterval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return tuple(merged)


def _validated_pcm(pcm: object) -> PcmWavData:
    if type(pcm) is not PcmWavData:
        raise TypeError("pcm must be PcmWavData")
    if (
        type(pcm.sample_rate_hz) is not int
        or type(pcm.channels) is not int
        or type(pcm.sample_width_bytes) is not int
        or type(pcm.frame_count) is not int
        or pcm.frame_count < 0
        or type(pcm.samples_int16) is not tuple
        or len(pcm.samples_int16) != pcm.frame_count
    ):
        raise ValueError("PCM data is invalid")
    if any(
        type(sample) is not int or not _INT16_MINIMUM <= sample <= _INT16_MAXIMUM
        for sample in pcm.samples_int16
    ):
        raise ValueError("PCM data is invalid")
    return pcm


def _validated_normalized_intervals(
    intervals: object,
    frame_count: int,
) -> tuple[SileroInterval, ...]:
    if type(intervals) is not tuple:
        raise TypeError("intervals must be a normalized tuple")
    previous_end = 0
    for index, interval in enumerate(intervals):
        if type(interval) is not SileroInterval:
            raise TypeError("intervals must contain SileroInterval values")
        if type(interval.start) is not int or type(interval.end) is not int:
            raise TypeError("Silero interval bounds must be ints")
        if not 0 <= interval.start < interval.end <= frame_count:
            raise ValueError("Silero interval bounds are invalid")
        if index and interval.start <= previous_end:
            raise ValueError("intervals are not normalized")
        previous_end = interval.end
    return intervals


def _rms(samples: tuple[int, ...], mask: list[bool] | None = None) -> float:
    selected = (
        samples
        if mask is None
        else tuple(sample for sample, included in zip(samples, mask, strict=True) if included)
    )
    if not selected:
        return 0.0
    return math.sqrt(sum((sample / 32_768.0) ** 2 for sample in selected) / len(selected))


def _estimated_snr_db(
    samples: tuple[int, ...], speech_mask: list[bool], noise_mask: list[bool]
) -> float | None:
    if sum(speech_mask) < _MINIMUM_MASK_FRAMES or sum(noise_mask) < _MINIMUM_MASK_FRAMES:
        return None
    speech_rms = _rms(samples, speech_mask)
    noise_rms = _rms(samples, noise_mask)
    if not math.isfinite(speech_rms) or not math.isfinite(noise_rms):
        return None
    ratio = speech_rms / max(noise_rms, _NOISE_FLOOR)
    if not math.isfinite(ratio) or ratio <= 0.0:
        return None
    snr = 20 * math.log10(ratio)
    return snr if math.isfinite(snr) else None


def _masks(
    frame_count: int, intervals: tuple[SileroInterval, ...]
) -> tuple[list[bool], list[bool]]:
    speech_mask = [False] * frame_count
    for interval in intervals:
        for frame in range(interval.start, interval.end):
            speech_mask[frame] = True

    noise_mask = [not speech for speech in speech_mask]
    for interval in intervals:
        for frame in range(max(0, interval.start - _GUARD_FRAMES), interval.start):
            noise_mask[frame] = False
        for frame in range(interval.end, min(frame_count, interval.end + _GUARD_FRAMES)):
            noise_mask[frame] = False
    return speech_mask, noise_mask


def build_audio_quality_evidence(
    pcm: PcmWavData,
    intervals: tuple[SileroInterval, ...],
) -> AudioQualityEvidence:
    pcm = _validated_pcm(pcm)
    intervals = _validated_normalized_intervals(intervals, pcm.frame_count)
    speech_mask, noise_mask = _masks(pcm.frame_count, intervals)
    samples = pcm.samples_int16
    rms = _rms(samples)
    peak_abs = max((abs(sample / 32_768.0) for sample in samples), default=0.0)
    clipping_fraction = (
        sum(sample in (_INT16_MINIMUM, _INT16_MAXIMUM) for sample in samples) / pcm.frame_count
        if pcm.frame_count
        else 0.0
    )
    return AudioQualityEvidence(
        sample_rate_hz=pcm.sample_rate_hz,
        channels=pcm.channels,
        sample_width_bytes=pcm.sample_width_bytes,
        frame_count=pcm.frame_count,
        rms=rms,
        peak_abs=peak_abs,
        clipping_fraction=clipping_fraction,
        estimated_snr_db=_estimated_snr_db(samples, speech_mask, noise_mask),
    )


def analyze_silero_quality(
    pcm: PcmWavData,
    intervals: object,
) -> tuple[AudioQualityEvidence, bool]:
    pcm = _validated_pcm(pcm)
    normalized_intervals = normalize_silero_intervals(intervals, pcm.frame_count)
    return build_audio_quality_evidence(pcm, normalized_intervals), bool(normalized_intervals)
