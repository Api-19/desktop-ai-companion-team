from datetime import UTC, datetime

from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.style import build_style_pack, extract_style_features


def _single_candidate_package(text: str) -> SafeCandidatePackage:
    timestamp = datetime(2026, 7, 18, tzinfo=UTC)
    return SafeCandidatePackage(
        created_at=timestamp,
        candidates=[
            SafeStyleCandidate(
                text=text,
                category=CandidateCategory.GREETING,
                scene_tags=[CandidateCategory.GREETING.value],
            )
        ],
        audit=CandidateSelectionAudit(
            processed_at=timestamp,
            input_count=1,
            owner_input_count=1,
            dropped_other_count=0,
            allowed_category_counts={CandidateCategory.GREETING.value: 1},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=1,
            succeeded=True,
        ),
    )


def test_extracts_length_category_tone_and_phrase_features(
    synthetic_candidate_package,
) -> None:
    features = extract_style_features(synthetic_candidate_package)

    assert features.statistics["candidate_count"] == 3
    assert features.statistics["length.mean"] == 5.333
    assert features.statistics["length.median"] == 5
    assert features.statistics["length.min"] == 5
    assert features.statistics["length.max"] == 6
    assert features.statistics["category.sleep.count"] == 2
    assert features.statistics["category.meals.count"] == 1
    assert features.statistics["category.weather.count"] == 0
    assert features.statistics["tone.呀.count"] == 1
    assert features.statistics["tone.啦.count"] == 1
    assert features.statistics["tone.哦.count"] == 1
    assert features.common_phrases[0] == "合成"
    assert features.scene_tags == ["meals", "sleep"]


def test_extracts_punctuation_emoji_and_short_message_ratios() -> None:
    features = extract_style_features(_single_candidate_package("嗨呀😊！"))  # noqa: RUF001

    assert features.statistics["short_message.ratio"] == 1.0
    assert features.statistics["emoji_message.ratio"] == 1.0
    assert features.statistics["ending.！.count"] == 1  # noqa: RUF001
    assert features.statistics["repeated_punctuation.ratio"] == 0.0


def test_detects_repeated_punctuation() -> None:
    features = extract_style_features(_single_candidate_package("你好！！"))  # noqa: RUF001

    assert features.statistics["repeated_punctuation.ratio"] == 1.0


def test_length_quartiles_use_nearest_rank() -> None:
    timestamp = datetime(2026, 7, 18, tzinfo=UTC)
    texts = ["甲", "甲乙", "甲乙丙", "甲乙丙丁"]
    package = SafeCandidatePackage(
        created_at=timestamp,
        candidates=[
            SafeStyleCandidate(
                text=text,
                category=CandidateCategory.GREETING,
                scene_tags=[CandidateCategory.GREETING.value],
            )
            for text in texts
        ],
        audit=CandidateSelectionAudit(
            processed_at=timestamp,
            input_count=4,
            owner_input_count=4,
            dropped_other_count=0,
            allowed_category_counts={CandidateCategory.GREETING.value: 4},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=4,
            succeeded=True,
        ),
    )

    features = extract_style_features(package)

    assert features.statistics["length.p25"] == 1
    assert features.statistics["length.p75"] == 3


def test_builds_deterministic_bounded_style_pack(
    synthetic_candidate_package,
) -> None:
    features = extract_style_features(synthetic_candidate_package)

    first = build_style_pack(synthetic_candidate_package, features)
    second = build_style_pack(synthetic_candidate_package, features)

    assert first == second
    assert first.version.startswith("dev-style-")
    assert len(first.examples) == 3
    assert len(first.examples) <= 12
    assert first.scene_tags == ["meals", "sleep"]
    assert "do_not_mechanically_repeat" in first.forbidden_traits
