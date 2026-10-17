from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from companion.dialogue.ab import STYLE_REFRESH_SCENARIO_IDS

Similarity = Annotated[StrictInt, Field(ge=0, le=5)]
BlindLabel = Literal["A", "B"]
Preference = Literal["A", "B", "tie", "neither"]


class StyleRefreshRating(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: StrictStr
    preferred: Preference
    similarity_a: Similarity
    similarity_b: Similarity
    semantic_error_a: StrictBool
    semantic_error_b: StrictBool
    mechanical_a: StrictBool
    mechanical_b: StrictBool
    reason: StrictStr | None = Field(default=None, max_length=240)

    @model_validator(mode="before")
    @classmethod
    def require_exact_scalar_types(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        string_fields = ("scenario_id", "preferred")
        int_fields = ("similarity_a", "similarity_b")
        bool_fields = (
            "semantic_error_a",
            "semantic_error_b",
            "mechanical_a",
            "mechanical_b",
        )
        if any(type(value.get(field)) is not str for field in string_fields):
            raise ValueError("invalid style refresh rating")
        if any(type(value.get(field)) is not int for field in int_fields):
            raise ValueError("invalid style refresh rating")
        if any(type(value.get(field)) is not bool for field in bool_fields):
            raise ValueError("invalid style refresh rating")
        reason = value.get("reason")
        if reason is not None and type(reason) is not str:
            raise ValueError("invalid style refresh rating")
        return value


class StyleRefreshRatings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[StyleRefreshRating, ...]

    @model_validator(mode="after")
    def require_exact_scenarios(self) -> Self:
        scenario_ids = tuple(item.scenario_id for item in self.items)
        if len(scenario_ids) != 7 or set(scenario_ids) != set(STYLE_REFRESH_SCENARIO_IDS):
            raise ValueError("invalid style refresh ratings")
        return self


class StyleRefreshAnswerKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    styled_labels: dict[StrictStr, BlindLabel]

    @model_validator(mode="before")
    @classmethod
    def require_exact_key_types(cls, value: object) -> object:
        if not isinstance(value, dict) or type(value.get("styled_labels")) is not dict:
            raise ValueError("invalid style refresh answer key")
        labels = value["styled_labels"]
        if any(type(key) is not str or type(label) is not str for key, label in labels.items()):
            raise ValueError("invalid style refresh answer key")
        return value

    @model_validator(mode="after")
    def require_exact_scenarios(self) -> Self:
        if len(self.styled_labels) != 7 or set(self.styled_labels) != set(
            STYLE_REFRESH_SCENARIO_IDS
        ):
            raise ValueError("invalid style refresh answer key")
        return self


class StyleRefreshReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_count: Literal[7] = 7
    styled_relative_wins: int = Field(ge=0, le=7)
    baseline_relative_wins: int = Field(ge=0, le=7)
    ties: int = Field(ge=0, le=7)
    neither_count: int = Field(ge=0, le=7)
    absolute_successes: int = Field(ge=0, le=7)
    styled_semantic_failures: int = Field(ge=0, le=7)
    styled_mechanical_failures: int = Field(ge=0, le=7)
    proceed_to_full_twenty: bool
    provider_requests: Literal[0] = 0


def score_style_refresh(
    ratings: StyleRefreshRatings,
    answer_key: StyleRefreshAnswerKey,
) -> StyleRefreshReviewSummary:
    if type(ratings) is not StyleRefreshRatings or type(answer_key) is not StyleRefreshAnswerKey:
        raise ValueError("invalid style refresh review")
    styled_wins = baseline_wins = ties = neither = 0
    successes = semantic_failures = mechanical_failures = 0
    for rating in ratings.items:
        styled_label = answer_key.styled_labels[rating.scenario_id]
        baseline_label = "B" if styled_label == "A" else "A"
        if rating.preferred == styled_label:
            styled_wins += 1
        elif rating.preferred == baseline_label:
            baseline_wins += 1
        else:
            ties += 1
        if rating.preferred == "neither":
            neither += 1
        styled_similarity = rating.similarity_a if styled_label == "A" else rating.similarity_b
        styled_semantic_error = (
            rating.semantic_error_a if styled_label == "A" else rating.semantic_error_b
        )
        styled_mechanical = rating.mechanical_a if styled_label == "A" else rating.mechanical_b
        semantic_failures += int(styled_semantic_error)
        mechanical_failures += int(styled_mechanical)
        if (
            rating.preferred == styled_label
            and styled_similarity >= 4
            and not styled_semantic_error
            and not styled_mechanical
        ):
            successes += 1
    proceed = (
        successes >= 6 and neither == 0 and semantic_failures == 0 and mechanical_failures == 0
    )
    return StyleRefreshReviewSummary(
        styled_relative_wins=styled_wins,
        baseline_relative_wins=baseline_wins,
        ties=ties,
        neither_count=neither,
        absolute_successes=successes,
        styled_semantic_failures=semantic_failures,
        styled_mechanical_failures=mechanical_failures,
        proceed_to_full_twenty=proceed,
    )
