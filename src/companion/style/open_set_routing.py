"""Deterministic known, nearby, compound, and open-set style routing."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from companion.contracts import StyleIntent
from companion.contracts.base import ContractModel
from companion.style.errors import StyleValidationError
from companion.style.pragmatic_context import PragmaticContextDelta, PragmaticLabel
from companion.style.user_profile import (
    UserStyleProfile,
    UserStyleProfileStatus,
    UserStyleSnapshot,
)

_MAX_SCENARIOS = 64
_MAX_CANDIDATES = 4
_NEARBY_MIN_COVERAGE = 0.5
PUBLIC_STYLE_ROUTING_VERSION = "public-synthetic-routing-v1"
FORMAL_STYLE_SCENARIO_CATALOG_VERSION = "formal-style-scenario-catalog-v1"
_COARSE_ONLY_SEMANTIC_DEFINITION = "scenario-coarse-labels-only-v1"
_SCENARIO_SEMANTIC_REGISTRY: dict[str, frozenset[str]] = {
    _COARSE_ONLY_SEMANTIC_DEFINITION: frozenset(),
    "scenario-gratitude-v1": frozenset({"semantic-gratitude-owner-addressee-current-v1"}),
    "scenario-sleep-farewell-v1": frozenset(
        {
            "semantic-sleep-departure-owner-self-current-v1",
            "semantic-sleep-departure-owner-self-resolved-v1",
        }
    ),
    "scenario-fatigue-v1": frozenset({"semantic-fatigue-owner-current-v1"}),
    "scenario-sleep-care-v1": frozenset({"semantic-sleep-advice-owner-addressee-current-v1"}),
    "scenario-generic-question-v1": frozenset(
        {
            "semantic-question-use-current-v1",
            "semantic-question-mention-current-v1",
            "semantic-question-reported-current-v1",
            "semantic-affection-addressee-owner-question-current-v1",
        }
    ),
    "scenario-affection-v1": frozenset(
        {
            "semantic-affection-owner-addressee-current-v1",
            "semantic-affection-owner-addressee-resolved-v1",
        }
    ),
    "scenario-care-support-v1": frozenset(
        {
            "semantic-care-support-owner-addressee-current-v1",
            "semantic-care-support-addressee-owner-resolved-v1",
        }
    ),
    "scenario-support-request-v1": frozenset(
        {"semantic-support-request-addressee-owner-current-v1"}
    ),
    "scenario-sad-checkin-v1": frozenset(
        {
            "semantic-sadness-owner-current-v1",
            "semantic-sadness-owner-resolved-v1",
        }
    ),
    "scenario-sad-support-v1": frozenset(
        {
            "semantic-sadness-owner-current-v1",
            "semantic-sadness-owner-resolved-v1",
        }
    ),
    "scenario-apology-v1": frozenset({"semantic-apology-owner-addressee-current-v1"}),
}
_SCENARIO_SEMANTIC_EXCLUSIONS: dict[str, frozenset[str]] = {
    "scenario-sad-checkin-v1": frozenset({"semantic-support-request-addressee-owner-current-v1"}),
}
_SCENARIO_SEMANTIC_LABEL_SIGNATURES: dict[str, frozenset[PragmaticLabel] | None] = {
    _COARSE_ONLY_SEMANTIC_DEFINITION: None,
    "scenario-gratitude-v1": frozenset({PragmaticLabel.GRATITUDE}),
    "scenario-sleep-farewell-v1": frozenset({PragmaticLabel.FAREWELL, PragmaticLabel.CLOSING}),
    "scenario-fatigue-v1": frozenset({PragmaticLabel.FATIGUE}),
    "scenario-sleep-care-v1": frozenset({PragmaticLabel.CARE_GIVING}),
    "scenario-generic-question-v1": frozenset({PragmaticLabel.QUESTION}),
    "scenario-affection-v1": frozenset({PragmaticLabel.AFFECTION, PragmaticLabel.AFFECTIONATE}),
    "scenario-care-support-v1": frozenset({PragmaticLabel.CARE_GIVING}),
    "scenario-support-request-v1": frozenset(
        {PragmaticLabel.REQUEST, PragmaticLabel.SUPPORT_SEEKING}
    ),
    "scenario-sad-checkin-v1": frozenset({PragmaticLabel.SADNESS, PragmaticLabel.QUESTION}),
    "scenario-sad-support-v1": frozenset({PragmaticLabel.SADNESS, PragmaticLabel.SUPPORT_SEEKING}),
    "scenario-apology-v1": frozenset({PragmaticLabel.APOLOGY, PragmaticLabel.REPAIR}),
}

_FORMAL_SCENARIO_IDS = (
    "gratitude",
    "sleep_farewell",
    "fatigue",
    "sleep_care",
    "generic_question",
    "affection",
    "care_support",
    "support_request",
    "sad_checkin",
    "sad_support",
    "apology",
)
_FORMAL_SCENARIO_ROWS = (
    ("gratitude", "scenario-gratitude-v1", (PragmaticLabel.GRATITUDE,)),
    (
        "sleep_farewell",
        "scenario-sleep-farewell-v1",
        (PragmaticLabel.FAREWELL, PragmaticLabel.CLOSING),
    ),
    ("fatigue", "scenario-fatigue-v1", (PragmaticLabel.FATIGUE,)),
    ("sleep_care", "scenario-sleep-care-v1", (PragmaticLabel.CARE_GIVING,)),
    ("generic_question", "scenario-generic-question-v1", (PragmaticLabel.QUESTION,)),
    (
        "affection",
        "scenario-affection-v1",
        (PragmaticLabel.AFFECTION, PragmaticLabel.AFFECTIONATE),
    ),
    ("care_support", "scenario-care-support-v1", (PragmaticLabel.CARE_GIVING,)),
    (
        "support_request",
        "scenario-support-request-v1",
        (PragmaticLabel.REQUEST, PragmaticLabel.SUPPORT_SEEKING),
    ),
    (
        "sad_checkin",
        "scenario-sad-checkin-v1",
        (PragmaticLabel.SADNESS, PragmaticLabel.QUESTION),
    ),
    (
        "sad_support",
        "scenario-sad-support-v1",
        (PragmaticLabel.SADNESS, PragmaticLabel.SUPPORT_SEEKING),
    ),
    ("apology", "scenario-apology-v1", (PragmaticLabel.APOLOGY, PragmaticLabel.REPAIR)),
)


def _semantic_label_signature_is_valid(
    semantic_definition_id: str,
    labels: frozenset[PragmaticLabel],
) -> bool:
    expected = _SCENARIO_SEMANTIC_LABEL_SIGNATURES.get(semantic_definition_id)
    return semantic_definition_id in _SCENARIO_SEMANTIC_LABEL_SIGNATURES and (
        expected is None or labels == expected
    )


UserId = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
OpaqueId = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


class _FrozenContractModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenSetRouteKind(StrEnum):
    KNOWN = "known"
    NEARBY = "nearby"
    OPEN_SET = "open_set"
    COMPOUND = "compound"


class StyleScenarioPrototype(_FrozenContractModel):
    """User-scoped, non-content signature for one known style scenario."""

    user_id: UserId
    scenario_id: OpaqueId
    required_labels: tuple[PragmaticLabel, ...] = Field(min_length=1, max_length=8)
    auxiliary_intent: StyleIntent | None = None
    prototype_version: OpaqueId
    semantic_definition_id: OpaqueId = _COARSE_ONLY_SEMANTIC_DEFINITION

    @field_validator("required_labels")
    @classmethod
    def require_unique_canonical_labels(
        cls, value: tuple[PragmaticLabel, ...]
    ) -> tuple[PragmaticLabel, ...]:
        if len(value) != len(set(value)):
            raise ValueError("scenario prototype labels must be unique")
        return tuple(sorted(value, key=lambda label: label.value))

    @model_validator(mode="after")
    def require_registered_version(self) -> Self:
        if self.prototype_version != PUBLIC_STYLE_ROUTING_VERSION:
            raise ValueError("style routing prototype version is not registered")
        if self.semantic_definition_id not in _SCENARIO_SEMANTIC_REGISTRY:
            raise ValueError("style scenario semantic definition is not registered")
        if not _semantic_label_signature_is_valid(
            self.semantic_definition_id,
            frozenset(self.required_labels),
        ):
            raise ValueError("style scenario semantic definition label signature is invalid")
        return self


class StyleRouteCandidate(_FrozenContractModel):
    user_id: UserId
    scenario_id: OpaqueId
    prototype_version: OpaqueId
    auxiliary_intent: StyleIntent | None = None
    semantic_definition_id: OpaqueId = _COARSE_ONLY_SEMANTIC_DEFINITION
    matched_labels: tuple[PragmaticLabel, ...] = Field(min_length=1, max_length=8)
    missing_labels: tuple[PragmaticLabel, ...] = Field(default=(), max_length=8)
    coverage: UnitInterval
    raw_score: UnitInterval
    weight: UnitInterval

    @model_validator(mode="after")
    def require_explainable_score(self) -> Self:
        if self.prototype_version != PUBLIC_STYLE_ROUTING_VERSION:
            raise ValueError("style routing prototype version is not registered")
        if self.semantic_definition_id not in _SCENARIO_SEMANTIC_REGISTRY:
            raise ValueError("style scenario semantic definition is not registered")
        matched = set(self.matched_labels)
        missing = set(self.missing_labels)
        if len(matched) != len(self.matched_labels) or len(missing) != len(self.missing_labels):
            raise ValueError("route candidate labels must be unique")
        if matched & missing:
            raise ValueError("route candidate matched and missing labels must be disjoint")
        if not _semantic_label_signature_is_valid(
            self.semantic_definition_id,
            frozenset(matched | missing),
        ):
            raise ValueError("style scenario semantic definition label signature is invalid")
        expected_coverage = round(len(matched) / (len(matched) + len(missing)), 6)
        if self.coverage != expected_coverage:
            raise ValueError("route candidate coverage must be reproducible from labels")
        if self.raw_score <= 0.0 or self.weight <= 0.0:
            raise ValueError("route candidate scores and weights must be positive")
        return self


class OpenSetStyleRoute(_FrozenContractModel):
    """Self-verifying Phase 3 result with authorized, non-content routing evidence."""

    user_id: UserId
    routing_version: OpaqueId
    route_kind: OpenSetRouteKind
    abstained: StrictBool
    unknown: StrictBool
    snapshot: UserStyleSnapshot
    context_delta: PragmaticContextDelta
    scenario_catalog: tuple[StyleScenarioPrototype, ...] = Field(max_length=_MAX_SCENARIOS)
    candidates: tuple[StyleRouteCandidate, ...] = Field(default=(), max_length=_MAX_CANDIDATES)

    @property
    def stable_profile(self) -> UserStyleProfile:
        return self.snapshot.profile

    @model_validator(mode="after")
    def require_route_invariants(self) -> Self:
        if self.routing_version != PUBLIC_STYLE_ROUTING_VERSION:
            raise ValueError("style routing version is not registered")
        validated_snapshot = _deep_validated_snapshot(self.snapshot)
        if validated_snapshot is None or not _snapshot_matches_user(
            validated_snapshot, self.user_id
        ):
            raise ValueError("style route requires an authorized snapshot")
        validated_delta = _deep_validated_delta(self.context_delta)
        if validated_delta is None or validated_delta.user_id != self.user_id:
            raise ValueError("style route user scope is inconsistent")
        validated_catalog = _deep_validated_catalog(self.scenario_catalog)
        if validated_catalog is None or tuple(self.scenario_catalog) != validated_catalog:
            raise ValueError("style route catalog must be valid and deterministic")
        if any(prototype.user_id != self.user_id for prototype in validated_catalog):
            raise ValueError("style route catalog user scope is inconsistent")
        if any(candidate.user_id != self.user_id for candidate in self.candidates):
            raise ValueError("style route candidate user scope is inconsistent")
        validated_candidates = _deep_validated_candidates(self.candidates)
        if validated_candidates is None or tuple(self.candidates) != validated_candidates:
            raise ValueError("style route candidates are invalid")

        expected_kind, expected_candidates = _calculate_route(
            user_id=self.user_id,
            context_delta=validated_delta,
            catalog=validated_catalog,
        )
        if self.route_kind is not expected_kind:
            raise ValueError("style route kind is not reproducible from routing evidence")
        if tuple(candidate.scenario_id for candidate in self.candidates) != tuple(
            candidate.scenario_id for candidate in expected_candidates
        ):
            raise ValueError("style route candidate selection or order is not deterministic")
        for actual, expected in zip(self.candidates, expected_candidates, strict=True):
            if actual.matched_labels != expected.matched_labels:
                raise ValueError("style route candidate matched labels do not match the delta")
            if actual.missing_labels != expected.missing_labels:
                raise ValueError("style route candidate missing labels do not match the delta")
            if actual.coverage != expected.coverage:
                raise ValueError("style route candidate coverage is not reproducible")
            if actual.raw_score != expected.raw_score:
                raise ValueError("style route candidate raw score is not reproducible")
            if actual.weight != expected.weight:
                raise ValueError("style route candidate normalized weight is not reproducible")
            if (
                actual.prototype_version != expected.prototype_version
                or actual.auxiliary_intent is not expected.auxiliary_intent
                or actual.semantic_definition_id != expected.semantic_definition_id
            ):
                raise ValueError("style route candidate prototype evidence is inconsistent")
        expected_unknown = expected_kind is OpenSetRouteKind.OPEN_SET
        if self.abstained is not expected_unknown or self.unknown is not expected_unknown:
            raise ValueError("style route abstention state is inconsistent")
        return self


def _validate_user_scope(user_id: str, trusted_user_id: str) -> str:
    if (
        type(trusted_user_id) is not str
        or not trusted_user_id
        or trusted_user_id != trusted_user_id.strip()
        or len(trusted_user_id) > 120
        or type(user_id) is not str
        or user_id != trusted_user_id
    ):
        raise StyleValidationError("untrusted_style_user", "style user is not trusted")
    return user_id


def _deep_validated_snapshot(snapshot: UserStyleSnapshot) -> UserStyleSnapshot | None:
    if type(snapshot) is not UserStyleSnapshot:
        return None
    validated: UserStyleSnapshot | None = None
    with suppress(Exception):
        validated = UserStyleSnapshot.model_validate(snapshot.model_dump(mode="python"))
    return validated


def _snapshot_matches_user(snapshot: UserStyleSnapshot, user_id: str) -> bool:
    consent = snapshot.consent
    profile = snapshot.profile
    derived = snapshot.derived
    identities = {consent.user_id, profile.user_id, derived.user_id}
    authorization_versions = {
        consent.authorization_version,
        profile.authorization_version,
        derived.authorization_version,
    }
    return not (
        identities != {user_id}
        or consent.status is not UserStyleProfileStatus.AUTHORIZED
        or len(authorization_versions) != 1
        or profile.profile_version != derived.profile_version
    )


def _validated_snapshot(user_id: str, snapshot: UserStyleSnapshot) -> UserStyleSnapshot:
    validated = _deep_validated_snapshot(snapshot)
    if validated is None or not _snapshot_matches_user(validated, user_id):
        raise StyleValidationError("style_profile_unavailable", "style profile is unavailable")
    return validated


def _deep_validated_delta(
    context_delta: PragmaticContextDelta,
) -> PragmaticContextDelta | None:
    if type(context_delta) is not PragmaticContextDelta:
        return None
    validated: PragmaticContextDelta | None = None
    with suppress(Exception):
        validated = PragmaticContextDelta.model_validate(context_delta.model_dump(mode="python"))
    return validated


def _deep_validated_catalog(
    scenario_catalog: Sequence[StyleScenarioPrototype],
) -> tuple[StyleScenarioPrototype, ...] | None:
    validated: list[StyleScenarioPrototype] = []
    with suppress(Exception):
        items = tuple(scenario_catalog)
        if len(items) > _MAX_SCENARIOS:
            return None
        for item in items:
            if type(item) is not StyleScenarioPrototype:
                return None
            validated.append(StyleScenarioPrototype.model_validate(item.model_dump(mode="python")))
    if len(validated) != len(tuple(scenario_catalog)):
        return None
    scenario_ids = [item.scenario_id for item in validated]
    signatures = [(item.required_labels, item.semantic_definition_id) for item in validated]
    if len(scenario_ids) != len(set(scenario_ids)) or len(signatures) != len(set(signatures)):
        return None
    return tuple(sorted(validated, key=lambda item: item.scenario_id))


def _deep_validated_candidates(
    candidates: Sequence[StyleRouteCandidate],
) -> tuple[StyleRouteCandidate, ...] | None:
    validated: list[StyleRouteCandidate] = []
    with suppress(Exception):
        for candidate in candidates:
            if type(candidate) is not StyleRouteCandidate:
                return None
            validated.append(
                StyleRouteCandidate.model_validate(candidate.model_dump(mode="python"))
            )
    if len(validated) != len(tuple(candidates)):
        return None
    return tuple(validated)


def _validated_catalog(
    user_id: str,
    scenario_catalog: Sequence[StyleScenarioPrototype],
) -> tuple[StyleScenarioPrototype, ...]:
    if isinstance(scenario_catalog, (str, bytes)):
        raise StyleValidationError("invalid_style_route", "style route input is invalid")
    items: tuple[object, ...] | None = None
    with suppress(Exception):
        items = tuple(scenario_catalog)
    if items is None:
        raise StyleValidationError("invalid_style_route", "style route input is invalid") from None
    if len(items) > _MAX_SCENARIOS:
        raise StyleValidationError("invalid_style_route", "style route input is invalid")
    validated: list[StyleScenarioPrototype] = []
    for item in items:
        if type(item) is not StyleScenarioPrototype:
            raise StyleValidationError("invalid_style_route", "style route input is invalid")
        candidate: StyleScenarioPrototype | None = None
        with suppress(Exception):
            candidate = StyleScenarioPrototype.model_validate(item.model_dump(mode="python"))
        if candidate is None:
            raise StyleValidationError(
                "invalid_style_route", "style route input is invalid"
            ) from None
        if candidate.user_id != user_id:
            raise StyleValidationError(
                "style_route_scope_mismatch", "style route user scope does not match"
            )
        validated.append(candidate)
    scenario_ids = [item.scenario_id for item in validated]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise StyleValidationError("invalid_style_route", "style route input is invalid")
    signatures = [(item.required_labels, item.semantic_definition_id) for item in validated]
    if len(signatures) != len(set(signatures)):
        raise StyleValidationError("invalid_style_route", "style route input is invalid")
    return tuple(sorted(validated, key=lambda item: item.scenario_id))


def build_formal_style_scenario_catalog(
    *,
    user_id: str,
    trusted_user_id: str,
    snapshot: UserStyleSnapshot,
    expected_catalog_version: str,
) -> tuple[StyleScenarioPrototype, ...]:
    """Build the complete fixed, non-content catalog for an authorized owner."""

    owner = _validate_user_scope(user_id, trusted_user_id)
    _validated_snapshot(owner, snapshot)
    if (
        type(expected_catalog_version) is not str
        or expected_catalog_version != FORMAL_STYLE_SCENARIO_CATALOG_VERSION
    ):
        raise StyleValidationError("invalid_style_route", "style route input is invalid")

    rows = _FORMAL_SCENARIO_ROWS
    try:
        if type(rows) is not tuple or len(rows) != len(_FORMAL_SCENARIO_IDS):
            raise ValueError("incomplete formal scenario catalog")
        prototypes = []
        for row, expected_id in zip(rows, _FORMAL_SCENARIO_IDS, strict=True):
            if type(row) is not tuple or len(row) != 3:
                raise ValueError("invalid formal scenario row")
            scenario_id, semantic_id, labels = row
            if (
                scenario_id != expected_id
                or semantic_id != f"scenario-{expected_id.replace('_', '-')}-v1"
                or type(labels) is not tuple
                or not labels
                or frozenset(labels) != _SCENARIO_SEMANTIC_LABEL_SIGNATURES.get(semantic_id)
            ):
                raise ValueError("formal scenario row differs from registered definition")
            prototypes.append(
                StyleScenarioPrototype(
                    user_id=owner,
                    scenario_id=scenario_id,
                    required_labels=labels,
                    auxiliary_intent=None,
                    prototype_version=PUBLIC_STYLE_ROUTING_VERSION,
                    semantic_definition_id=semantic_id,
                )
            )
        if {item.semantic_definition_id for item in prototypes} != (
            set(_SCENARIO_SEMANTIC_REGISTRY) - {_COARSE_ONLY_SEMANTIC_DEFINITION}
        ):
            raise ValueError("formal scenario definitions are incomplete")
        catalog = _validated_catalog(owner, tuple(prototypes))
        if len(catalog) != len(_FORMAL_SCENARIO_IDS):
            raise ValueError("formal scenario catalog is incomplete")
        return catalog
    except Exception as exc:
        raise StyleValidationError("invalid_style_route", "style route input is invalid") from exc


def _candidate_parts(
    prototype: StyleScenarioPrototype,
    signal_quality: dict[PragmaticLabel, float],
    semantic_frame_ids: frozenset[str],
) -> tuple[float, float, tuple[PragmaticLabel, ...], tuple[PragmaticLabel, ...]] | None:
    accepted_frames = _SCENARIO_SEMANTIC_REGISTRY[prototype.semantic_definition_id]
    if accepted_frames and accepted_frames.isdisjoint(semantic_frame_ids):
        return None
    excluded_frames = _SCENARIO_SEMANTIC_EXCLUSIONS.get(
        prototype.semantic_definition_id, frozenset()
    )
    if not excluded_frames.isdisjoint(semantic_frame_ids):
        return None
    matched = tuple(label for label in prototype.required_labels if label in signal_quality)
    if not matched:
        return None
    missing = tuple(label for label in prototype.required_labels if label not in signal_quality)
    coverage = round(len(matched) / len(prototype.required_labels), 6)
    raw_score = round(
        sum(signal_quality[label] for label in matched) / len(prototype.required_labels),
        6,
    )
    return coverage, raw_score, matched, missing


def _weighted_candidates(
    user_id: str,
    selected: Sequence[
        tuple[
            StyleScenarioPrototype,
            float,
            float,
            tuple[PragmaticLabel, ...],
            tuple[PragmaticLabel, ...],
        ]
    ],
) -> tuple[StyleRouteCandidate, ...]:
    ordered = tuple(sorted(selected, key=lambda item: (-item[2], item[0].scenario_id)))[
        :_MAX_CANDIDATES
    ]
    total = sum(item[2] for item in ordered)
    weights: list[float] = []
    for index, item in enumerate(ordered):
        if index == len(ordered) - 1:
            weight = round(1.0 - sum(weights), 6)
        else:
            weight = round(item[2] / total, 6)
        weights.append(weight)
    return tuple(
        StyleRouteCandidate(
            user_id=user_id,
            scenario_id=prototype.scenario_id,
            prototype_version=prototype.prototype_version,
            auxiliary_intent=prototype.auxiliary_intent,
            semantic_definition_id=prototype.semantic_definition_id,
            matched_labels=matched,
            missing_labels=missing,
            coverage=coverage,
            raw_score=raw_score,
            weight=weight,
        )
        for (prototype, coverage, raw_score, matched, missing), weight in zip(
            ordered, weights, strict=True
        )
    )


def _calculate_route(
    *,
    user_id: str,
    context_delta: PragmaticContextDelta,
    catalog: Sequence[StyleScenarioPrototype],
) -> tuple[OpenSetRouteKind, tuple[StyleRouteCandidate, ...]]:
    signal_quality = {
        signal.label: round(signal.score * signal.confidence, 12)
        for signal in context_delta.all_signals
    }
    semantic_frame_ids = frozenset(
        frame.semantic_definition_id for frame in context_delta.semantic_frames
    )
    matches = []
    for prototype in catalog:
        parts = _candidate_parts(prototype, signal_quality, semantic_frame_ids)
        if parts is not None:
            matches.append((prototype, *parts))

    complete = [match for match in matches if match[1] == 1.0]
    complete = [
        match
        for match in complete
        if not any(
            set(match[0].required_labels) < set(other[0].required_labels) for other in complete
        )
    ]
    if len(complete) >= 2:
        route_kind = OpenSetRouteKind.COMPOUND
        selected = complete
    elif len(complete) == 1:
        route_kind = OpenSetRouteKind.KNOWN
        selected = complete
    else:
        selected = [match for match in matches if match[1] >= _NEARBY_MIN_COVERAGE]
        route_kind = OpenSetRouteKind.NEARBY if selected else OpenSetRouteKind.OPEN_SET
    candidates = () if not selected else _weighted_candidates(user_id, selected)
    return route_kind, candidates


def route_open_set_style(
    *,
    user_id: str,
    trusted_user_id: str,
    snapshot: UserStyleSnapshot,
    context_delta: PragmaticContextDelta,
    scenario_catalog: Sequence[StyleScenarioPrototype],
) -> OpenSetStyleRoute:
    """Route one trusted request without inventing an unknown-scenario match."""

    owner = _validate_user_scope(user_id, trusted_user_id)
    validated_snapshot = _validated_snapshot(owner, snapshot)
    if type(context_delta) is not PragmaticContextDelta or context_delta.user_id != owner:
        raise StyleValidationError(
            "style_route_scope_mismatch", "style route user scope does not match"
        )
    validated_delta = _deep_validated_delta(context_delta)
    if validated_delta is None:
        raise StyleValidationError("invalid_style_route", "style route input is invalid") from None
    context_delta = validated_delta
    catalog = _validated_catalog(owner, scenario_catalog)
    route_kind, candidates = _calculate_route(
        user_id=owner,
        context_delta=context_delta,
        catalog=catalog,
    )
    is_unknown = route_kind is OpenSetRouteKind.OPEN_SET
    return OpenSetStyleRoute(
        user_id=owner,
        routing_version=PUBLIC_STYLE_ROUTING_VERSION,
        route_kind=route_kind,
        abstained=is_unknown,
        unknown=is_unknown,
        snapshot=validated_snapshot,
        context_delta=context_delta,
        scenario_catalog=catalog,
        candidates=candidates,
    )
