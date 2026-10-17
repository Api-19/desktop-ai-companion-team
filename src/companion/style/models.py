from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from companion.contracts import StyleIntent
from companion.contracts.base import ContractModel
from companion.privacy import CandidateCategory


class StyleFeatures(ContractModel):
    statistics: dict[str, float | int | str]
    common_phrases: list[str]
    scene_tags: list[str]


class CoverageStatus(StrEnum):
    TESTABLE = "testable"
    SPARSE = "sparse"
    MECHANICAL_RISK = "mechanical_risk"
    MISSING = "missing"


class RetrievalResult(ContractModel):
    examples: list[str]
    coverage_status: CoverageStatus
    category_count: int


class EvaluationScenario(ContractModel):
    scenario_id: str = Field(min_length=1)
    category: CandidateCategory
    intent: StyleIntent = StyleIntent.CATEGORY_DEFAULT
    prompt: str = Field(min_length=1)


class EvaluationPlanItem(ContractModel):
    scenario_id: str
    category: CandidateCategory
    intent: StyleIntent = StyleIntent.CATEGORY_DEFAULT
    baseline_prompt: str
    styled_prompt: str
    styled_preferences: list[str] = Field(default_factory=list, max_length=2)
    styled_examples: list[str]
    coverage_status: CoverageStatus


class EvaluationPlan(ContractModel):
    style_version: str
    calibration_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    items: list[EvaluationPlanItem]


class ScenarioResponses(ContractModel):
    scenario_id: str = Field(min_length=1)
    baseline: str = Field(min_length=1)
    styled: str = Field(min_length=1)


class EvaluationResponses(ContractModel):
    items: list[ScenarioResponses]


class BlindReviewItem(ContractModel):
    scenario_id: str
    category: CandidateCategory
    response_a: str
    response_b: str


class BlindReview(ContractModel):
    items: list[BlindReviewItem]


class BlindAnswerKey(ContractModel):
    styled_labels: dict[str, Literal["A", "B"]]


class ScenarioRating(ContractModel):
    scenario_id: str = Field(min_length=1)
    preferred: Literal["A", "B", "tie"]
    mechanical_a: bool
    mechanical_b: bool
    note: str | None = Field(default=None, max_length=240)


class BlindRatings(ContractModel):
    items: list[ScenarioRating]


class AbsoluteScenarioRating(ContractModel):
    scenario_id: str = Field(min_length=1)
    preferred: Literal["A", "B", "tie"]
    similarity_a: int = Field(ge=1, le=5)
    similarity_b: int = Field(ge=1, le=5)
    neither_like_owner: bool
    context_correct_a: bool
    context_correct_b: bool
    mechanical_a: bool
    mechanical_b: bool
    ideal_response: str | None = Field(default=None, min_length=1, max_length=160)


class AbsoluteBlindRatings(ContractModel):
    items: list[AbsoluteScenarioRating]


class ContextualEvaluationReport(ContractModel):
    total_scenarios: int = Field(ge=0)
    styled_relative_wins: int = Field(ge=0)
    baseline_relative_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    styled_absolute_successes: int = Field(ge=0)
    styled_context_failures: int = Field(ge=0)
    styled_mechanical_scenarios: int = Field(ge=0)
    neither_like_owner_scenarios: int = Field(ge=0)
    conclusion: Literal["overall_sufficient", "locally_sufficient", "insufficient"]

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if (
            self.styled_relative_wins + self.baseline_relative_wins + self.ties
            != self.total_scenarios
        ):
            raise ValueError("contextual relative outcome counts must match total scenarios")
        bounded_counts = (
            self.styled_absolute_successes,
            self.styled_context_failures,
            self.styled_mechanical_scenarios,
            self.neither_like_owner_scenarios,
        )
        if any(count > self.total_scenarios for count in bounded_counts):
            raise ValueError("contextual scenario counts cannot exceed total scenarios")
        if self.styled_absolute_successes > self.styled_relative_wins:
            raise ValueError("contextual absolute successes require styled relative wins")
        if self.neither_like_owner_scenarios > self.ties:
            raise ValueError("neither-like-owner scenarios must be relative ties")
        if any(
            self.styled_absolute_successes + failure_count > self.total_scenarios
            for failure_count in (
                self.styled_context_failures,
                self.styled_mechanical_scenarios,
                self.neither_like_owner_scenarios,
            )
        ):
            raise ValueError("contextual absolute successes cannot overlap failure counts")
        return self


class EvaluationReport(ContractModel):
    total_scenarios: int
    styled_wins: int
    baseline_wins: int
    ties: int
    testable_scenarios: int
    styled_mechanical_scenarios: int
    baseline_mechanical_scenarios: int
    missing_categories: list[CandidateCategory]
    sparse_categories: list[CandidateCategory]
    conclusion: Literal[
        "overall_sufficient",
        "locally_sufficient",
        "insufficient",
    ]


class PreparationSummary(ContractModel):
    candidate_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    testable_count: int = Field(ge=0)
    sparse_count: int = Field(ge=0)
    mechanical_risk_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
