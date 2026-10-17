import statistics
import unicodedata
from collections import Counter
from itertools import pairwise
from math import ceil

from companion.privacy import CandidateCategory, SafeCandidatePackage
from companion.style.models import StyleFeatures

TONE_MARKERS = ("呀", "啦", "哦", "呢", "嘛", "哇", "哈", "呐", "哒")
ENDING_MARKS = ("。", "！", "？", "!", "?", "…", "～", "~")  # noqa: RUF001


def _compact(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _phrase_text(text: str) -> str:
    return "".join(
        character
        for character in _compact(text)
        if not unicodedata.category(character).startswith(("P", "S"))
    )


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * fraction) - 1]


def _has_repeated_punctuation(text: str) -> bool:
    compact = _compact(text)
    return any(
        unicodedata.category(left).startswith("P") and unicodedata.category(right).startswith("P")
        for left, right in pairwise(compact)
    )


def extract_style_features(package: SafeCandidatePackage) -> StyleFeatures:
    texts = [candidate.text for candidate in package.candidates]
    lengths = [len(_compact(text)) for text in texts]
    category_counts = Counter(candidate.category.value for candidate in package.candidates)
    phrase_documents: Counter[str] = Counter()
    for text in texts:
        compact = _phrase_text(text)
        phrases = {
            compact[index : index + width]
            for width in range(2, 5)
            for index in range(max(0, len(compact) - width + 1))
        }
        phrase_documents.update(phrases)
    common_phrases = [
        phrase
        for phrase, count in sorted(
            phrase_documents.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )
        if count >= 2
    ][:12]
    count = len(texts)
    statistics_map: dict[str, float | int | str] = {
        "candidate_count": count,
        "length.mean": round(statistics.fmean(lengths), 3),
        "length.median": statistics.median(lengths),
        "length.p25": _percentile(lengths, 0.25),
        "length.p75": _percentile(lengths, 0.75),
        "length.min": min(lengths),
        "length.max": max(lengths),
        "short_message.ratio": round(sum(length <= 4 for length in lengths) / count, 3),
        "emoji_message.ratio": round(
            sum(
                any(unicodedata.category(character) == "So" for character in text) for text in texts
            )
            / count,
            3,
        ),
        "repeated_punctuation.ratio": round(
            sum(_has_repeated_punctuation(text) for text in texts) / count,
            3,
        ),
        "dominant_category.ratio": round(max(category_counts.values()) / count, 3),
    }
    for category in CandidateCategory:
        category_count = category_counts.get(category.value, 0)
        statistics_map[f"category.{category.value}.count"] = category_count
        statistics_map[f"category.{category.value}.ratio"] = round(
            category_count / count,
            3,
        )
    for marker in TONE_MARKERS:
        statistics_map[f"tone.{marker}.count"] = sum(text.count(marker) for text in texts)
    for mark in ENDING_MARKS:
        statistics_map[f"ending.{mark}.count"] = sum(text.endswith(mark) for text in texts)
    for index, phrase in enumerate(common_phrases, start=1):
        statistics_map[f"common_phrase.{index}"] = phrase
    return StyleFeatures(
        statistics=statistics_map,
        common_phrases=common_phrases,
        scene_tags=sorted(category_counts),
    )
