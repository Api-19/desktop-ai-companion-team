from __future__ import annotations

import pytest
from pydantic import ValidationError

from companion.dialogue import (
    STYLE_REFRESH_SCENARIO_IDS,
    StyleRefreshAnswerKey,
    StyleRefreshRating,
    StyleRefreshRatings,
    score_style_refresh,
)


def rating(
    scenario_id: str,
    *,
    preferred: str = "A",
    similarity_a: int = 5,
    similarity_b: int = 0,
    semantic_error_a: bool = False,
    semantic_error_b: bool = False,
    mechanical_a: bool = False,
    mechanical_b: bool = False,
) -> StyleRefreshRating:
    return StyleRefreshRating(
        scenario_id=scenario_id,
        preferred=preferred,
        similarity_a=similarity_a,
        similarity_b=similarity_b,
        semantic_error_a=semantic_error_a,
        semantic_error_b=semantic_error_b,
        mechanical_a=mechanical_a,
        mechanical_b=mechanical_b,
    )


def ratings(**overrides: StyleRefreshRating) -> StyleRefreshRatings:
    return StyleRefreshRatings(
        items=[overrides.get(item, rating(item)) for item in STYLE_REFRESH_SCENARIO_IDS]
    )


def answer_key(**overrides: str) -> StyleRefreshAnswerKey:
    return StyleRefreshAnswerKey(
        styled_labels={item: overrides.get(item, "A") for item in STYLE_REFRESH_SCENARIO_IDS}
    )


@pytest.mark.parametrize("similarity", (0, 5))
def test_style_refresh_rating_accepts_similarity_endpoints(similarity: int) -> None:
    item = rating(STYLE_REFRESH_SCENARIO_IDS[0], similarity_a=similarity)

    assert item.similarity_a == similarity


@pytest.mark.parametrize("preferred", ("A", "B", "tie", "neither"))
def test_style_refresh_rating_accepts_exact_preferences(preferred: str) -> None:
    assert rating(STYLE_REFRESH_SCENARIO_IDS[0], preferred=preferred).preferred == preferred


def test_style_refresh_ratings_require_exact_complete_unique_scenario_set() -> None:
    complete = [rating(item) for item in STYLE_REFRESH_SCENARIO_IDS]

    with pytest.raises(ValidationError):
        StyleRefreshRatings(items=complete[:-1])
    with pytest.raises(ValidationError):
        StyleRefreshRatings(items=[*complete[:-1], complete[0]])
    with pytest.raises(ValidationError):
        StyleRefreshRatings(items=[*complete, rating("extra-scenario")])


def test_style_refresh_answer_key_requires_exact_complete_scenario_set() -> None:
    incomplete = {item: "A" for item in STYLE_REFRESH_SCENARIO_IDS[:-1]}

    with pytest.raises(ValidationError):
        StyleRefreshAnswerKey(styled_labels=incomplete)


def test_style_refresh_complete_sets_do_not_depend_on_json_order() -> None:
    reversed_ids = tuple(reversed(STYLE_REFRESH_SCENARIO_IDS))
    shuffled_ratings = StyleRefreshRatings(items=[rating(item) for item in reversed_ids])
    shuffled_key = StyleRefreshAnswerKey(styled_labels={item: "A" for item in reversed_ids})

    result = score_style_refresh(shuffled_ratings, shuffled_key)

    assert result.scenario_count == 7
    assert result.absolute_successes == 7


def test_six_strict_successes_proceed_to_full_twenty() -> None:
    failed = STYLE_REFRESH_SCENARIO_IDS[-1]
    result = score_style_refresh(
        ratings(**{failed: rating(failed, preferred="B")}),
        answer_key(),
    )

    assert result.absolute_successes == 6
    assert result.proceed_to_full_twenty is True
    assert result.provider_requests == 0


def test_five_strict_successes_do_not_proceed() -> None:
    first, second = STYLE_REFRESH_SCENARIO_IDS[-2:]
    result = score_style_refresh(
        ratings(
            **{
                first: rating(first, preferred="B"),
                second: rating(second, preferred="B"),
            }
        ),
        answer_key(),
    )

    assert result.absolute_successes == 5
    assert result.proceed_to_full_twenty is False


def test_one_neither_blocks_progress_even_with_six_successes() -> None:
    failed = STYLE_REFRESH_SCENARIO_IDS[-1]
    result = score_style_refresh(
        ratings(**{failed: rating(failed, preferred="neither")}),
        answer_key(),
    )

    assert result.absolute_successes == 6
    assert result.neither_count == 1
    assert result.proceed_to_full_twenty is False


def test_one_styled_semantic_error_blocks_progress() -> None:
    failed = STYLE_REFRESH_SCENARIO_IDS[-1]
    result = score_style_refresh(
        ratings(**{failed: rating(failed, semantic_error_a=True)}),
        answer_key(),
    )

    assert result.absolute_successes == 6
    assert result.styled_semantic_failures == 1
    assert result.proceed_to_full_twenty is False


def test_styled_label_is_resolved_only_from_complete_answer_key() -> None:
    first = STYLE_REFRESH_SCENARIO_IDS[0]
    result = score_style_refresh(
        ratings(**{first: rating(first, preferred="B", similarity_a=0, similarity_b=5)}),
        answer_key(**{first: "B"}),
    )

    assert result.absolute_successes == 7
    assert result.styled_relative_wins == 7
