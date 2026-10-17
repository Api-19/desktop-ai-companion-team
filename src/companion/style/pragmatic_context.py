"""Request-scoped, calibrated pragmatic context representation."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import ConfigDict, Field, StrictStr, StringConstraints, model_validator

from companion.contracts.base import TimezoneRequiredModel
from companion.style.errors import StyleValidationError

PUBLIC_PRAGMATIC_CALIBRATION_VERSION = "public-synthetic-pragmatics-v1"
PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION = "public-synthetic-chinese-semantics-v1"
_MAX_TEXT_LENGTH = 2_000
_MAX_RECENT_CONTEXT = 4
_REOPEN_SECONDS = 21_600
_WILSON_Z_95 = 1.959963984540054

UserId = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
CalibrationVersion = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
RuleId = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    ),
]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


class _FrozenTimezoneModel(TimezoneRequiredModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PragmaticDimension(StrEnum):
    SPEECH_ACT = "speech_act"
    EXPRESSED_AFFECT = "expressed_affect"
    RELATIONSHIP_CONTEXT = "relationship_context"
    DIALOGUE_STAGE = "dialogue_stage"


class PragmaticLabel(StrEnum):
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    GRATITUDE = "gratitude"
    APOLOGY = "apology"
    FAREWELL = "farewell"

    JOY = "joy"
    SADNESS = "sadness"
    FATIGUE = "fatigue"
    WORRY = "worry"
    AFFECTION = "affection"

    AFFILIATIVE = "affiliative"
    SUPPORT_SEEKING = "support_seeking"
    CARE_GIVING = "care_giving"
    AFFECTIONATE = "affectionate"
    REPAIR = "repair"

    OPENING = "opening"
    REOPENING = "reopening"
    CONTINUING = "continuing"
    CLOSING = "closing"


_LABEL_DIMENSIONS: dict[PragmaticDimension, frozenset[PragmaticLabel]] = {
    PragmaticDimension.SPEECH_ACT: frozenset(
        {
            PragmaticLabel.GREETING,
            PragmaticLabel.QUESTION,
            PragmaticLabel.REQUEST,
            PragmaticLabel.GRATITUDE,
            PragmaticLabel.APOLOGY,
            PragmaticLabel.FAREWELL,
        }
    ),
    PragmaticDimension.EXPRESSED_AFFECT: frozenset(
        {
            PragmaticLabel.JOY,
            PragmaticLabel.SADNESS,
            PragmaticLabel.FATIGUE,
            PragmaticLabel.WORRY,
            PragmaticLabel.AFFECTION,
        }
    ),
    PragmaticDimension.RELATIONSHIP_CONTEXT: frozenset(
        {
            PragmaticLabel.AFFILIATIVE,
            PragmaticLabel.SUPPORT_SEEKING,
            PragmaticLabel.CARE_GIVING,
            PragmaticLabel.AFFECTIONATE,
            PragmaticLabel.REPAIR,
        }
    ),
    PragmaticDimension.DIALOGUE_STAGE: frozenset(
        {
            PragmaticLabel.OPENING,
            PragmaticLabel.REOPENING,
            PragmaticLabel.CONTINUING,
            PragmaticLabel.CLOSING,
        }
    ),
}


class PragmaticSourceType(StrEnum):
    PUBLIC_SYNTHETIC_CALIBRATION = "public_synthetic_calibration"


class PragmaticObservedFrom(StrEnum):
    CURRENT_REQUEST = "current_request"
    RECENT_CONTEXT_PRESENCE = "recent_context_presence"
    DETERMINISTIC_REQUEST_STATE = "deterministic_request_state"


class PragmaticConfidenceMethod(StrEnum):
    WILSON_LOWER_BOUND_95 = "wilson_lower_bound_95"


class SemanticPredicate(StrEnum):
    GRATITUDE = "gratitude"
    SLEEP_DEPARTURE = "sleep_departure"
    FATIGUE = "fatigue"
    SLEEP_ADVICE = "sleep_advice"
    QUESTION = "question"
    TECHNICAL_REQUEST = "technical_request"
    AFFECTION = "affection"
    CARE_SUPPORT = "care_support"
    SUPPORT_REQUEST = "support_request"
    SADNESS = "sadness"
    APOLOGY = "apology"
    REPORTED_CONTENT = "reported_content"
    THIRD_PARTY_CONTENT = "third_party_content"


class SemanticParticipant(StrEnum):
    OWNER = "owner"
    ADDRESSEE = "addressee"
    SELF = "self"
    THIRD_PARTY = "third_party"


class SemanticPolarity(StrEnum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"


class SemanticDiscourseMode(StrEnum):
    USE = "use"
    MENTION = "mention"
    REPORTED = "reported"


class SemanticContextResolution(StrEnum):
    CURRENT = "current"
    RESOLVED_CONTEXT = "resolved_context"


@dataclass(frozen=True)
class _SemanticFrameRegistration:
    definition_id: str
    predicate: SemanticPredicate
    actor: SemanticParticipant | None = None
    experiencer: SemanticParticipant | None = None
    target: SemanticParticipant | None = None
    addressee: SemanticParticipant | None = None
    polarity: SemanticPolarity = SemanticPolarity.AFFIRMED
    discourse_mode: SemanticDiscourseMode = SemanticDiscourseMode.USE
    context_resolution: SemanticContextResolution = SemanticContextResolution.CURRENT


_SEMANTIC_FRAME_REGISTRATIONS = (
    _SemanticFrameRegistration(
        "semantic-gratitude-owner-addressee-current-v1",
        SemanticPredicate.GRATITUDE,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-sleep-departure-owner-self-current-v1",
        SemanticPredicate.SLEEP_DEPARTURE,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
    ),
    _SemanticFrameRegistration(
        "semantic-fatigue-owner-current-v1",
        SemanticPredicate.FATIGUE,
        experiencer=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
    ),
    _SemanticFrameRegistration(
        "semantic-sleep-advice-owner-addressee-current-v1",
        SemanticPredicate.SLEEP_ADVICE,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-question-use-current-v1",
        SemanticPredicate.QUESTION,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-question-mention-current-v1",
        SemanticPredicate.QUESTION,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
        discourse_mode=SemanticDiscourseMode.MENTION,
    ),
    _SemanticFrameRegistration(
        "semantic-question-reported-current-v1",
        SemanticPredicate.QUESTION,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.THIRD_PARTY,
        addressee=SemanticParticipant.ADDRESSEE,
        discourse_mode=SemanticDiscourseMode.REPORTED,
    ),
    _SemanticFrameRegistration(
        "semantic-reported-content-third-party-current-v1",
        SemanticPredicate.REPORTED_CONTENT,
        actor=SemanticParticipant.THIRD_PARTY,
        target=SemanticParticipant.THIRD_PARTY,
        discourse_mode=SemanticDiscourseMode.REPORTED,
    ),
    _SemanticFrameRegistration(
        "semantic-third-party-content-current-v1",
        SemanticPredicate.THIRD_PARTY_CONTENT,
        actor=SemanticParticipant.THIRD_PARTY,
        target=SemanticParticipant.THIRD_PARTY,
    ),
    _SemanticFrameRegistration(
        "semantic-technical-question-current-v1",
        SemanticPredicate.TECHNICAL_REQUEST,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-technical-request-current-v1",
        SemanticPredicate.TECHNICAL_REQUEST,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-affection-owner-addressee-current-v1",
        SemanticPredicate.AFFECTION,
        experiencer=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-affection-addressee-owner-question-current-v1",
        SemanticPredicate.AFFECTION,
        experiencer=SemanticParticipant.ADDRESSEE,
        target=SemanticParticipant.OWNER,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-care-support-owner-addressee-current-v1",
        SemanticPredicate.CARE_SUPPORT,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-support-request-addressee-owner-current-v1",
        SemanticPredicate.SUPPORT_REQUEST,
        actor=SemanticParticipant.ADDRESSEE,
        target=SemanticParticipant.OWNER,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-sadness-owner-current-v1",
        SemanticPredicate.SADNESS,
        experiencer=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
    ),
    _SemanticFrameRegistration(
        "semantic-apology-owner-addressee-current-v1",
        SemanticPredicate.APOLOGY,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
    ),
    _SemanticFrameRegistration(
        "semantic-affection-owner-addressee-resolved-v1",
        SemanticPredicate.AFFECTION,
        experiencer=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
        context_resolution=SemanticContextResolution.RESOLVED_CONTEXT,
    ),
    _SemanticFrameRegistration(
        "semantic-sleep-departure-owner-self-resolved-v1",
        SemanticPredicate.SLEEP_DEPARTURE,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
        context_resolution=SemanticContextResolution.RESOLVED_CONTEXT,
    ),
    _SemanticFrameRegistration(
        "semantic-care-support-addressee-owner-resolved-v1",
        SemanticPredicate.CARE_SUPPORT,
        actor=SemanticParticipant.ADDRESSEE,
        target=SemanticParticipant.OWNER,
        addressee=SemanticParticipant.OWNER,
        context_resolution=SemanticContextResolution.RESOLVED_CONTEXT,
    ),
    _SemanticFrameRegistration(
        "semantic-sadness-owner-resolved-v1",
        SemanticPredicate.SADNESS,
        experiencer=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
        context_resolution=SemanticContextResolution.RESOLVED_CONTEXT,
    ),
    _SemanticFrameRegistration(
        "semantic-fatigue-owner-negated-current-v1",
        SemanticPredicate.FATIGUE,
        experiencer=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
        polarity=SemanticPolarity.NEGATED,
    ),
    _SemanticFrameRegistration(
        "semantic-gratitude-owner-addressee-negated-current-v1",
        SemanticPredicate.GRATITUDE,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
        polarity=SemanticPolarity.NEGATED,
    ),
    _SemanticFrameRegistration(
        "semantic-sleep-departure-owner-self-negated-current-v1",
        SemanticPredicate.SLEEP_DEPARTURE,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
        polarity=SemanticPolarity.NEGATED,
    ),
    _SemanticFrameRegistration(
        "semantic-sleep-advice-owner-addressee-negated-current-v1",
        SemanticPredicate.SLEEP_ADVICE,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
        polarity=SemanticPolarity.NEGATED,
    ),
    _SemanticFrameRegistration(
        "semantic-apology-owner-addressee-negated-current-v1",
        SemanticPredicate.APOLOGY,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.ADDRESSEE,
        addressee=SemanticParticipant.ADDRESSEE,
        polarity=SemanticPolarity.NEGATED,
    ),
    _SemanticFrameRegistration(
        "semantic-sadness-owner-negated-current-v1",
        SemanticPredicate.SADNESS,
        experiencer=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
        polarity=SemanticPolarity.NEGATED,
    ),
    _SemanticFrameRegistration(
        "semantic-support-request-negated-current-v1",
        SemanticPredicate.SUPPORT_REQUEST,
        actor=SemanticParticipant.ADDRESSEE,
        target=SemanticParticipant.OWNER,
        addressee=SemanticParticipant.ADDRESSEE,
        polarity=SemanticPolarity.NEGATED,
    ),
    _SemanticFrameRegistration(
        "semantic-sleep-departure-negated-reported-current-v1",
        SemanticPredicate.SLEEP_DEPARTURE,
        actor=SemanticParticipant.OWNER,
        target=SemanticParticipant.SELF,
        polarity=SemanticPolarity.NEGATED,
        discourse_mode=SemanticDiscourseMode.REPORTED,
    ),
)
_SEMANTIC_FRAME_REGISTRY = {
    registration.definition_id: registration for registration in _SEMANTIC_FRAME_REGISTRATIONS
}


class SemanticFrame(_FrozenTimezoneModel):
    registry_version: CalibrationVersion = PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION
    semantic_definition_id: RuleId
    predicate: SemanticPredicate
    actor: SemanticParticipant | None = None
    experiencer: SemanticParticipant | None = None
    target: SemanticParticipant | None = None
    addressee: SemanticParticipant | None = None
    polarity: SemanticPolarity
    discourse_mode: SemanticDiscourseMode
    context_resolution: SemanticContextResolution

    @model_validator(mode="after")
    def require_registered_definition(self) -> Self:
        if self.registry_version != PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION:
            raise ValueError("semantic frame registry version is not registered")
        registration = _SEMANTIC_FRAME_REGISTRY.get(self.semantic_definition_id)
        if registration is None:
            raise ValueError("semantic frame definition is not registered")
        expected = (
            registration.predicate,
            registration.actor,
            registration.experiencer,
            registration.target,
            registration.addressee,
            registration.polarity,
            registration.discourse_mode,
            registration.context_resolution,
        )
        actual = (
            self.predicate,
            self.actor,
            self.experiencer,
            self.target,
            self.addressee,
            self.polarity,
            self.discourse_mode,
            self.context_resolution,
        )
        if actual != expected:
            raise ValueError("semantic frame does not match its registered definition")
        return self


@dataclass(frozen=True)
class _RuleRegistration:
    rule_id: str
    dimension: PragmaticDimension
    label: PragmaticLabel
    correct_count: int
    evaluated_count: int


_RULE_REGISTRATIONS = (
    _RuleRegistration(
        "speech-greeting-v1", PragmaticDimension.SPEECH_ACT, PragmaticLabel.GREETING, 4, 5
    ),
    _RuleRegistration(
        "speech-question-v1", PragmaticDimension.SPEECH_ACT, PragmaticLabel.QUESTION, 5, 5
    ),
    _RuleRegistration(
        "speech-request-v1", PragmaticDimension.SPEECH_ACT, PragmaticLabel.REQUEST, 4, 5
    ),
    _RuleRegistration(
        "speech-gratitude-v1", PragmaticDimension.SPEECH_ACT, PragmaticLabel.GRATITUDE, 5, 5
    ),
    _RuleRegistration(
        "speech-apology-v1", PragmaticDimension.SPEECH_ACT, PragmaticLabel.APOLOGY, 5, 5
    ),
    _RuleRegistration(
        "speech-farewell-v1", PragmaticDimension.SPEECH_ACT, PragmaticLabel.FAREWELL, 5, 5
    ),
    _RuleRegistration(
        "affect-joy-v1", PragmaticDimension.EXPRESSED_AFFECT, PragmaticLabel.JOY, 4, 5
    ),
    _RuleRegistration(
        "affect-sadness-v1", PragmaticDimension.EXPRESSED_AFFECT, PragmaticLabel.SADNESS, 5, 5
    ),
    _RuleRegistration(
        "affect-fatigue-v1", PragmaticDimension.EXPRESSED_AFFECT, PragmaticLabel.FATIGUE, 4, 5
    ),
    _RuleRegistration(
        "affect-worry-v1", PragmaticDimension.EXPRESSED_AFFECT, PragmaticLabel.WORRY, 4, 5
    ),
    _RuleRegistration(
        "affect-affection-v1", PragmaticDimension.EXPRESSED_AFFECT, PragmaticLabel.AFFECTION, 5, 5
    ),
    _RuleRegistration(
        "relationship-affiliative-v1",
        PragmaticDimension.RELATIONSHIP_CONTEXT,
        PragmaticLabel.AFFILIATIVE,
        4,
        5,
    ),
    _RuleRegistration(
        "relationship-support-seeking-v1",
        PragmaticDimension.RELATIONSHIP_CONTEXT,
        PragmaticLabel.SUPPORT_SEEKING,
        4,
        5,
    ),
    _RuleRegistration(
        "relationship-care-giving-v1",
        PragmaticDimension.RELATIONSHIP_CONTEXT,
        PragmaticLabel.CARE_GIVING,
        5,
        5,
    ),
    _RuleRegistration(
        "relationship-affectionate-v1",
        PragmaticDimension.RELATIONSHIP_CONTEXT,
        PragmaticLabel.AFFECTIONATE,
        5,
        5,
    ),
    _RuleRegistration(
        "relationship-repair-v1",
        PragmaticDimension.RELATIONSHIP_CONTEXT,
        PragmaticLabel.REPAIR,
        4,
        5,
    ),
    _RuleRegistration(
        "stage-opening-v1", PragmaticDimension.DIALOGUE_STAGE, PragmaticLabel.OPENING, 5, 5
    ),
    _RuleRegistration(
        "stage-reopening-v1",
        PragmaticDimension.DIALOGUE_STAGE,
        PragmaticLabel.REOPENING,
        5,
        5,
    ),
    _RuleRegistration(
        "stage-continuing-v1",
        PragmaticDimension.DIALOGUE_STAGE,
        PragmaticLabel.CONTINUING,
        4,
        5,
    ),
    _RuleRegistration(
        "stage-closing-v1", PragmaticDimension.DIALOGUE_STAGE, PragmaticLabel.CLOSING, 5, 5
    ),
)
_RULE_REGISTRY = {registration.rule_id: registration for registration in _RULE_REGISTRATIONS}


def _wilson_lower_bound(correct: int, total: int) -> float:
    proportion = correct / total
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / total
    centre = proportion + z_squared / (2.0 * total)
    margin = _WILSON_Z_95 * math.sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total
    )
    return round((centre - margin) / denominator, 6)


class PragmaticCalibrationBasis(_FrozenTimezoneModel):
    calibration_version: CalibrationVersion
    correct_count: int = Field(strict=True, ge=1, le=1_000_000)
    evaluated_count: int = Field(strict=True, ge=1, le=1_000_000)
    confidence_method: PragmaticConfidenceMethod
    matched_rule_ids: tuple[RuleId, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if self.calibration_version != PUBLIC_PRAGMATIC_CALIBRATION_VERSION:
            raise ValueError("pragmatic calibration version is not registered")
        if self.correct_count > self.evaluated_count:
            raise ValueError("calibration correct count cannot exceed evaluated examples")
        if len(self.matched_rule_ids) != len(set(self.matched_rule_ids)):
            raise ValueError("matched pragmatic calibration rules must be unique")
        if len(self.matched_rule_ids) != 1:
            raise ValueError("exactly one registered pragmatic rule is required")
        registration = _RULE_REGISTRY.get(self.matched_rule_ids[0])
        if registration is None:
            raise ValueError("pragmatic calibration rule is not registered")
        if (
            self.correct_count != registration.correct_count
            or self.evaluated_count != registration.evaluated_count
        ):
            raise ValueError("pragmatic counts do not match registered calibration counts")
        return self


class PragmaticSignalProvenance(_FrozenTimezoneModel):
    source_type: PragmaticSourceType
    evidence_count: int = Field(strict=True, ge=1, le=16)
    observed_from: PragmaticObservedFrom
    updated_at: datetime
    confidence_method: PragmaticConfidenceMethod


class PragmaticSignal(_FrozenTimezoneModel):
    dimension: PragmaticDimension
    label: PragmaticLabel
    score: UnitInterval
    confidence: UnitInterval
    provenance: PragmaticSignalProvenance
    calibration: PragmaticCalibrationBasis

    @model_validator(mode="after")
    def require_calibrated_compatible_signal(self) -> Self:
        if self.label not in _LABEL_DIMENSIONS[self.dimension]:
            raise ValueError("pragmatic label does not belong to its dimension")
        registration = _RULE_REGISTRY[self.calibration.matched_rule_ids[0]]
        if self.dimension is not registration.dimension or self.label is not registration.label:
            raise ValueError("pragmatic rule does not match its registered dimension and label")
        expected_score = round(
            self.calibration.correct_count / self.calibration.evaluated_count,
            6,
        )
        if self.score != expected_score:
            raise ValueError("pragmatic score must equal empirical calibration accuracy")
        expected_confidence = _wilson_lower_bound(
            self.calibration.correct_count,
            self.calibration.evaluated_count,
        )
        if self.confidence != expected_confidence:
            raise ValueError("pragmatic confidence must equal the calibrated Wilson bound")
        if self.provenance.evidence_count != len(self.calibration.matched_rule_ids):
            raise ValueError("pragmatic evidence count must match distinct rules")
        if self.provenance.confidence_method is not self.calibration.confidence_method:
            raise ValueError("pragmatic confidence methods must match")
        return self


class PragmaticContextDelta(_FrozenTimezoneModel):
    """Ephemeral current-request representation; it is never a stable Profile."""

    user_id: UserId
    calibration_version: CalibrationVersion
    semantic_registry_version: CalibrationVersion = PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION
    created_at: datetime
    speech_act: tuple[PragmaticSignal, ...] = Field(default=(), max_length=8)
    expressed_affect: tuple[PragmaticSignal, ...] = Field(default=(), max_length=8)
    relationship_context: tuple[PragmaticSignal, ...] = Field(default=(), max_length=8)
    dialogue_stage: tuple[PragmaticSignal, ...] = Field(default=(), max_length=4)
    semantic_frames: tuple[SemanticFrame, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def require_dimension_groups(self) -> Self:
        if self.calibration_version != PUBLIC_PRAGMATIC_CALIBRATION_VERSION:
            raise ValueError("pragmatic calibration version is not registered")
        if self.semantic_registry_version != PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION:
            raise ValueError("semantic frame registry version is not registered")
        if len({frame.semantic_definition_id for frame in self.semantic_frames}) != len(
            self.semantic_frames
        ):
            raise ValueError("semantic frame definitions must be unique per request")
        if any(
            frame.registry_version != self.semantic_registry_version
            for frame in self.semantic_frames
        ):
            raise ValueError("semantic frame registry versions must match the context delta")
        groups = (
            (PragmaticDimension.SPEECH_ACT, self.speech_act),
            (PragmaticDimension.EXPRESSED_AFFECT, self.expressed_affect),
            (PragmaticDimension.RELATIONSHIP_CONTEXT, self.relationship_context),
            (PragmaticDimension.DIALOGUE_STAGE, self.dialogue_stage),
        )
        for dimension, signals in groups:
            if any(signal.dimension is not dimension for signal in signals):
                raise ValueError("pragmatic signal is stored under the wrong dimension")
            labels = [signal.label for signal in signals]
            if len(labels) != len(set(labels)):
                raise ValueError("pragmatic labels must be unique within a dimension")
            if any(
                signal.calibration.calibration_version != self.calibration_version
                for signal in signals
            ):
                raise ValueError("pragmatic calibration versions must match the context delta")
        return self

    @property
    def all_signals(self) -> tuple[PragmaticSignal, ...]:
        return (
            *self.speech_act,
            *self.expressed_affect,
            *self.relationship_context,
            *self.dialogue_stage,
        )


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    patterns: tuple[str, ...]
    observed_from: PragmaticObservedFrom = PragmaticObservedFrom.CURRENT_REQUEST

    def matches(self, text: str) -> bool:
        return any(_contains_unnegated_pattern(text, pattern) for pattern in self.patterns)


_LOCAL_NEGATION_MARKERS = (
    "一点也不",
    "并没有",
    "并未",
    "并不",
    "还不",
    "才不",
    "不再",
    "不用",
    "不要",
    "没有",
    "别",
    "不",
    "没",
)
_DOUBLE_NEGATION_MARKERS = ("不是不", "并非不")
_CLAUSE_BOUNDARIES = re.compile(
    r"(?:[\uFF0C\u3002\uFF01\uFF1F!?\uFF1B;]|但是|不过|可是|然后|只是|却|但|而)"
)


def _pattern_is_negated(text: str, index: int) -> bool:
    """Resolve a bounded lexical predicate inside its current clause."""

    prefix = _CLAUSE_BOUNDARIES.split(text[:index])[-1]
    if any(marker in prefix for marker in _DOUBLE_NEGATION_MARKERS):
        return False
    if "不但" in prefix or "不仅" in prefix:
        return False
    return any(marker in prefix for marker in _LOCAL_NEGATION_MARKERS)


def _contains_unnegated_pattern(text: str, pattern: str) -> bool:
    start = 0
    while True:
        index = text.find(pattern, start)
        if index < 0:
            return False
        if not _pattern_is_negated(text, index):
            return True
        start = index + len(pattern)


# Counts are the frozen aggregate results of the public synthetic calibration set.
# Runtime values are calculated from these counts, never from ordinal confidence words.
_TEXT_RULES = (
    _Rule("speech-greeting-v1", ("你好", "早上好", "晚上好")),
    _Rule("speech-question-v1", ("\uff1f", "?")),
    _Rule("speech-request-v1", ("请", "帮我", "陪陪我", "陪我一会儿", "陪我聊聊")),
    _Rule("speech-gratitude-v1", ("谢谢", "谢啦", "多亏你")),
    _Rule("speech-apology-v1", ("对不起", "抱歉")),
    _Rule("speech-farewell-v1", ("晚安", "再见", "休息了", "见周公", "撤了", "先睡了")),
    _Rule("affect-joy-v1", ("开心", "高兴")),
    _Rule("affect-sadness-v1", ("难过", "伤心")),
    _Rule("affect-fatigue-v1", ("有点累", "好累", "疲惫", "困死", "累趴", "我困了")),
    _Rule("affect-worry-v1", ("我担心", "我害怕")),
    _Rule("affect-affection-v1", ("想你", "爱你", "抱抱")),
    _Rule("relationship-affiliative-v1", ("谢谢你", "谢啦")),
    _Rule(
        "relationship-support-seeking-v1",
        ("陪陪我", "陪我一会儿", "陪我聊聊", "安慰我"),
    ),
    _Rule("relationship-care-giving-v1", ("早点休息", "早点睡", "去睡觉", "记得吃饭", "我来陪你")),
    _Rule("relationship-affectionate-v1", ("想你", "抱抱")),
    _Rule("relationship-repair-v1", ("对不起", "抱歉")),
)

# Audited, versioned lexical families provide bounded candidate recall.  They are
# deliberately phrase-level rather than complete-message examples.
_GRATITUDE_TERMS = ("谢谢", "谢啦", "多亏", "感谢", "感激")
_SLEEP_DEPARTURE_TERMS = ("晚安", "休息", "歇", "见周公", "撤", "睡")
_SLEEP_ADVICE_TERMS = ("早点睡", "早点休息", "去睡觉")
_FATIGUE_TERMS = ("有点累", "好累", "疲惫", "困死", "累趴", "困了", "累")
_AFFECTION_OWNER_TERMS = ("想你", "想念你", "惦记你", "爱你", "抱抱")
_AFFECTION_ROLE_SWAP_TERMS = ("你想我", "你想念我", "你惦记我")
_SUPPORT_REQUEST_TERMS = ("陪我",)
_SADNESS_TERMS = ("难过", "伤心", "低落")
_TECHNICAL_TERMS = (
    "量子",
    "公式",
    "方程",
    "相对论",
    "线性代数",
    "矩阵",
    "求逆",
    "算一下",
    "帮我算",
    "乘以",
    "除以",
)
_META_MENTION_MARKERS = (
    "是什么意思",
    "怎么理解",
    "怎么翻译",
    "怎么用",
    "是什么语气",
    "是告别语",
    "是歌词",
    "这句话",
)
_REPORTING_MARKERS = ("说", "问", "提到")
_THIRD_PARTY_MARKERS = ("他", "她", "他们", "她们", "朋友", "同事", "小王")


def _contains_family(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _family_is_negated(text: str, terms: tuple[str, ...]) -> bool:
    indexes = ((text.find(term), term) for term in terms if text.find(term) >= 0)
    return all(_pattern_is_negated(text, index) for index, _term in indexes)


def _has_question_form(text: str) -> bool:
    stripped = text.rstrip("\u3002\uff01!")
    return stripped.endswith(("\uff1f", "?", "吗", "么", "呢", "不"))


def _is_reported_content(text: str) -> bool:
    return _contains_family(text, _THIRD_PARTY_MARKERS) and _contains_family(
        text, _REPORTING_MARKERS
    )


def _is_mention_question(text: str) -> bool:
    has_quotes = any(mark in text for mark in ("“", "”", '"', "\u2018", "\u2019"))
    return _contains_family(text, _META_MENTION_MARKERS) and (
        _has_question_form(text) or has_quotes or "这句话" in text
    )


def _is_owner_care_support(text: str) -> bool:
    return bool(re.search(r"我(?:会|来|可以|能)?陪(?:着)?你", text))


def _is_support_request(text: str) -> bool:
    return _contains_family(text, _SUPPORT_REQUEST_TERMS) and not _is_owner_care_support(text)


def _is_addressee_affection_question(text: str) -> bool:
    return _has_question_form(text) and _contains_family(text, _AFFECTION_ROLE_SWAP_TERMS)


_ALL_SEMANTIC_TERMS = (
    *_GRATITUDE_TERMS,
    *_SLEEP_DEPARTURE_TERMS,
    *_FATIGUE_TERMS,
    *_AFFECTION_OWNER_TERMS,
    *_SUPPORT_REQUEST_TERMS,
    *_SADNESS_TERMS,
)


def _clause_subject(clause: str) -> SemanticParticipant | None:
    normalized = clause.lstrip()
    if normalized.startswith("请问"):
        normalized = normalized.removeprefix("请问").lstrip()
    if normalized.startswith("我"):
        return SemanticParticipant.OWNER
    if normalized.startswith("你"):
        return SemanticParticipant.ADDRESSEE
    if normalized.startswith(_THIRD_PARTY_MARKERS):
        return SemanticParticipant.THIRD_PARTY
    return None


def _semantic_clauses(text: str) -> tuple[str, ...]:
    return tuple(
        clause
        for clause in _CLAUSE_BOUNDARIES.split(text)
        if clause and _contains_family(clause, _ALL_SEMANTIC_TERMS)
    )


def _is_only_third_party_content(text: str) -> bool:
    clauses = _semantic_clauses(text)
    return bool(clauses) and all(
        _clause_subject(clause) is SemanticParticipant.THIRD_PARTY for clause in clauses
    )


def _is_addressee_state_question(text: str) -> bool:
    clauses = _semantic_clauses(text)
    return (
        _has_question_form(text)
        and bool(clauses)
        and all(_clause_subject(clause) is SemanticParticipant.ADDRESSEE for clause in clauses)
        and not _is_support_request(text)
    )


def _is_owner_fatigue(text: str) -> bool:
    for clause in _CLAUSE_BOUNDARIES.split(text):
        if _contains_family(clause, _FATIGUE_TERMS) and not clause.startswith("你"):
            return True
    return False


def _semantic_frame(definition_id: str) -> SemanticFrame:
    registration = _SEMANTIC_FRAME_REGISTRY[definition_id]
    return SemanticFrame(
        semantic_definition_id=definition_id,
        predicate=registration.predicate,
        actor=registration.actor,
        experiencer=registration.experiencer,
        target=registration.target,
        addressee=registration.addressee,
        polarity=registration.polarity,
        discourse_mode=registration.discourse_mode,
        context_resolution=registration.context_resolution,
    )


def _semantic_frame_ids(text: str, context_items: tuple[str, ...]) -> tuple[str, ...]:
    latest_context = context_items[-1] if context_items else ""

    if text in {"我也是", "我也是呀", "也是呢"} and _contains_family(
        latest_context, _AFFECTION_OWNER_TERMS
    ):
        return ("semantic-affection-owner-addressee-resolved-v1",)
    if text in {"那就这样", "那先这样吧"} and _contains_family(
        latest_context, _SLEEP_DEPARTURE_TERMS
    ):
        return ("semantic-sleep-departure-owner-self-resolved-v1",)
    if text == "不用了" and "陪你" in latest_context:
        return ("semantic-support-request-negated-current-v1",)
    if text in {"好呀", "好啊"} and _is_owner_care_support(latest_context):
        return ("semantic-care-support-addressee-owner-resolved-v1",)
    if text == "嗯嗯" and _contains_family(latest_context, _SADNESS_TERMS):
        return ("semantic-sadness-owner-resolved-v1",)

    if "没说晚安" in text:
        return ("semantic-sleep-departure-negated-reported-current-v1",)
    if _is_reported_content(text):
        return (
            "semantic-question-reported-current-v1"
            if _has_question_form(text)
            else "semantic-reported-content-third-party-current-v1",
        )
    if _is_mention_question(text):
        return ("semantic-question-mention-current-v1",)
    if _is_only_third_party_content(text):
        return ("semantic-third-party-content-current-v1",)
    if _contains_family(text, _TECHNICAL_TERMS):
        definition_id = (
            "semantic-technical-question-current-v1"
            if _has_question_form(text)
            else "semantic-technical-request-current-v1"
        )
        return (definition_id,)

    definition_ids: list[str] = []
    if _is_addressee_affection_question(text):
        definition_ids.append("semantic-affection-addressee-owner-question-current-v1")
    if _is_addressee_state_question(text):
        return ("semantic-question-use-current-v1",)
    if _contains_family(text, _GRATITUDE_TERMS):
        definition_ids.append(
            "semantic-gratitude-owner-addressee-negated-current-v1"
            if _family_is_negated(text, _GRATITUDE_TERMS)
            else "semantic-gratitude-owner-addressee-current-v1"
        )
    is_sleep_advice = not _has_question_form(text) and _contains_family(text, _SLEEP_ADVICE_TERMS)
    if is_sleep_advice:
        definition_ids.append(
            "semantic-sleep-advice-owner-addressee-negated-current-v1"
            if _family_is_negated(text, _SLEEP_ADVICE_TERMS)
            else "semantic-sleep-advice-owner-addressee-current-v1"
        )
    if (
        not is_sleep_advice
        and not _has_question_form(text)
        and _contains_family(text, _SLEEP_DEPARTURE_TERMS)
    ):
        definition_ids.append(
            "semantic-sleep-departure-owner-self-negated-current-v1"
            if _family_is_negated(text, _SLEEP_DEPARTURE_TERMS)
            else "semantic-sleep-departure-owner-self-current-v1"
        )
    if _is_owner_fatigue(text):
        definition_ids.append(
            "semantic-fatigue-owner-negated-current-v1"
            if _family_is_negated(text, _FATIGUE_TERMS)
            else "semantic-fatigue-owner-current-v1"
        )
    if _is_owner_care_support(text):
        definition_ids.append("semantic-care-support-owner-addressee-current-v1")
    elif _is_support_request(text):
        definition_ids.append(
            "semantic-support-request-negated-current-v1"
            if _family_is_negated(text, _SUPPORT_REQUEST_TERMS)
            else "semantic-support-request-addressee-owner-current-v1"
        )
    if _contains_family(text, _AFFECTION_OWNER_TERMS) and not _family_is_negated(
        text, _AFFECTION_OWNER_TERMS
    ):
        definition_ids.append("semantic-affection-owner-addressee-current-v1")
    if _contains_family(text, _SADNESS_TERMS):
        definition_ids.append(
            "semantic-sadness-owner-negated-current-v1"
            if _family_is_negated(text, _SADNESS_TERMS)
            else "semantic-sadness-owner-current-v1"
        )
    if any(marker in text for marker in ("对不起", "抱歉")):
        definition_ids.append(
            "semantic-apology-owner-addressee-negated-current-v1"
            if _family_is_negated(text, ("对不起", "抱歉"))
            else "semantic-apology-owner-addressee-current-v1"
        )
    if _has_question_form(text) and not definition_ids:
        definition_ids.append("semantic-question-use-current-v1")
    return tuple(dict.fromkeys(definition_ids))


_SEMANTIC_SIGNAL_RULE_IDS: dict[str, tuple[str, ...]] = {
    "semantic-gratitude-owner-addressee-current-v1": (
        "speech-gratitude-v1",
        "relationship-affiliative-v1",
    ),
    "semantic-sleep-departure-owner-self-current-v1": ("speech-farewell-v1",),
    "semantic-fatigue-owner-current-v1": ("affect-fatigue-v1",),
    "semantic-sleep-advice-owner-addressee-current-v1": ("relationship-care-giving-v1",),
    "semantic-question-use-current-v1": ("speech-question-v1",),
    "semantic-question-mention-current-v1": ("speech-question-v1",),
    "semantic-question-reported-current-v1": ("speech-question-v1",),
    "semantic-affection-addressee-owner-question-current-v1": ("speech-question-v1",),
    "semantic-affection-owner-addressee-current-v1": (
        "affect-affection-v1",
        "relationship-affectionate-v1",
    ),
    "semantic-care-support-owner-addressee-current-v1": ("relationship-care-giving-v1",),
    "semantic-support-request-addressee-owner-current-v1": (
        "speech-request-v1",
        "relationship-support-seeking-v1",
    ),
    "semantic-sadness-owner-current-v1": ("affect-sadness-v1",),
    "semantic-apology-owner-addressee-current-v1": (
        "speech-apology-v1",
        "relationship-repair-v1",
    ),
    "semantic-affection-owner-addressee-resolved-v1": (
        "affect-affection-v1",
        "relationship-affectionate-v1",
    ),
    "semantic-sleep-departure-owner-self-resolved-v1": ("speech-farewell-v1",),
    "semantic-care-support-addressee-owner-resolved-v1": ("relationship-care-giving-v1",),
    "semantic-sadness-owner-resolved-v1": ("affect-sadness-v1",),
}


def _validate_user_id(user_id: str) -> str:
    if type(user_id) is not str or not user_id or user_id != user_id.strip() or len(user_id) > 120:
        raise StyleValidationError("untrusted_style_user", "style user is not trusted")
    return user_id


def _validate_request_inputs(
    *,
    text: str,
    recent_context: Sequence[str],
    idle_seconds: int | None,
    observed_at: datetime,
) -> tuple[str, tuple[str, ...]]:
    if type(text) is not str:
        raise StyleValidationError("invalid_pragmatic_input", "pragmatic input is invalid")
    normalized_text = text.strip()
    if not normalized_text or len(normalized_text) > _MAX_TEXT_LENGTH:
        raise StyleValidationError("invalid_pragmatic_input", "pragmatic input is invalid")
    if isinstance(recent_context, (str, bytes)):
        raise StyleValidationError("invalid_pragmatic_input", "pragmatic input is invalid")
    try:
        context_items = tuple(recent_context)
    except (TypeError, RuntimeError):
        raise StyleValidationError(
            "invalid_pragmatic_input", "pragmatic input is invalid"
        ) from None
    if len(context_items) > _MAX_RECENT_CONTEXT or any(
        type(item) is not str or not item.strip() or len(item.strip()) > _MAX_TEXT_LENGTH
        for item in context_items
    ):
        raise StyleValidationError("invalid_pragmatic_input", "pragmatic input is invalid")
    if idle_seconds is not None and (type(idle_seconds) is not int or idle_seconds < 0):
        raise StyleValidationError("invalid_pragmatic_input", "pragmatic input is invalid")
    if type(observed_at) is not datetime or observed_at.tzinfo is None:
        raise StyleValidationError("invalid_pragmatic_input", "pragmatic input is invalid")
    return normalized_text, tuple(item.strip() for item in context_items)


def _signal(rule: _Rule, observed_at: datetime) -> PragmaticSignal:
    registration = _RULE_REGISTRY[rule.rule_id]
    calibration = PragmaticCalibrationBasis(
        calibration_version=PUBLIC_PRAGMATIC_CALIBRATION_VERSION,
        correct_count=registration.correct_count,
        evaluated_count=registration.evaluated_count,
        confidence_method=PragmaticConfidenceMethod.WILSON_LOWER_BOUND_95,
        matched_rule_ids=(rule.rule_id,),
    )
    return PragmaticSignal(
        dimension=registration.dimension,
        label=registration.label,
        score=round(registration.correct_count / registration.evaluated_count, 6),
        confidence=_wilson_lower_bound(registration.correct_count, registration.evaluated_count),
        provenance=PragmaticSignalProvenance(
            source_type=PragmaticSourceType.PUBLIC_SYNTHETIC_CALIBRATION,
            evidence_count=1,
            observed_from=rule.observed_from,
            updated_at=observed_at,
            confidence_method=PragmaticConfidenceMethod.WILSON_LOWER_BOUND_95,
        ),
        calibration=calibration,
    )


def _stage_rule(
    label: PragmaticLabel,
    *,
    observed_from: PragmaticObservedFrom,
) -> _Rule:
    return _Rule(
        rule_id=f"stage-{label.value}-v1",
        patterns=(),
        observed_from=observed_from,
    )


def build_pragmatic_context_delta(
    *,
    user_id: str,
    trusted_user_id: str,
    text: str,
    recent_context: Sequence[str],
    idle_seconds: int | None,
    observed_at: datetime,
) -> PragmaticContextDelta:
    """Build an ephemeral calibrated representation for one trusted request."""

    owner = _validate_user_id(trusted_user_id)
    if type(user_id) is not str or user_id != owner:
        raise StyleValidationError("untrusted_style_user", "style user is not trusted")
    normalized_text, context_items = _validate_request_inputs(
        text=text,
        recent_context=recent_context,
        idle_seconds=idle_seconds,
        observed_at=observed_at,
    )

    semantic_definition_ids = _semantic_frame_ids(normalized_text, context_items)
    semantic_frames = tuple(_semantic_frame(item) for item in semantic_definition_ids)
    matched_rules: list[_Rule] = []
    if not semantic_frames:
        matched_rules = [rule for rule in _TEXT_RULES if rule.matches(normalized_text)]
    for semantic_definition_id in semantic_definition_ids:
        for rule_id in _SEMANTIC_SIGNAL_RULE_IDS.get(semantic_definition_id, ()):
            resolved = semantic_definition_id.endswith("-resolved-v1")
            matched_rules.append(
                _Rule(
                    rule_id,
                    (),
                    observed_from=(
                        PragmaticObservedFrom.RECENT_CONTEXT_PRESENCE
                        if resolved
                        else PragmaticObservedFrom.CURRENT_REQUEST
                    ),
                )
            )
    if (
        semantic_frames
        and _has_question_form(normalized_text)
        and not any(
            frame.predicate is SemanticPredicate.TECHNICAL_REQUEST for frame in semantic_frames
        )
    ):
        matched_rules.insert(0, _Rule("speech-question-v1", ()))
    deduplicated_rules = tuple({rule.rule_id: rule for rule in matched_rules}.values())
    matched = [_signal(rule, observed_at) for rule in deduplicated_rules]
    by_dimension = {
        dimension: tuple(signal for signal in matched if signal.dimension is dimension)
        for dimension in PragmaticDimension
    }

    speech_labels = {signal.label for signal in by_dimension[PragmaticDimension.SPEECH_ACT]}
    if PragmaticLabel.FAREWELL in speech_labels:
        stage = _stage_rule(
            PragmaticLabel.CLOSING,
            observed_from=PragmaticObservedFrom.CURRENT_REQUEST,
        )
    elif (
        PragmaticLabel.GREETING in speech_labels
        and idle_seconds is not None
        and (idle_seconds >= _REOPEN_SECONDS)
    ):
        stage = _stage_rule(
            PragmaticLabel.REOPENING,
            observed_from=PragmaticObservedFrom.DETERMINISTIC_REQUEST_STATE,
        )
    elif PragmaticLabel.GREETING in speech_labels and not context_items:
        stage = _stage_rule(
            PragmaticLabel.OPENING,
            observed_from=PragmaticObservedFrom.DETERMINISTIC_REQUEST_STATE,
        )
    elif context_items:
        stage = _stage_rule(
            PragmaticLabel.CONTINUING,
            observed_from=PragmaticObservedFrom.RECENT_CONTEXT_PRESENCE,
        )
    else:
        stage = None

    return PragmaticContextDelta(
        user_id=user_id,
        calibration_version=PUBLIC_PRAGMATIC_CALIBRATION_VERSION,
        semantic_registry_version=PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION,
        created_at=observed_at,
        speech_act=by_dimension[PragmaticDimension.SPEECH_ACT],
        expressed_affect=by_dimension[PragmaticDimension.EXPRESSED_AFFECT],
        relationship_context=by_dimension[PragmaticDimension.RELATIONSHIP_CONTEXT],
        dialogue_stage=() if stage is None else (_signal(stage, observed_at),),
        semantic_frames=semantic_frames,
    )
