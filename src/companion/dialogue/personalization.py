"""Compile validated Style routing evidence into Dialogue model controls."""

from __future__ import annotations

import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import ConfigDict, Field, StrictFloat, StrictStr, StringConstraints, model_validator

from companion.contracts import StyleIntent
from companion.contracts.base import ContractModel
from companion.style.evidence_retrieval import (
    StyleEvidenceResult,
)
from companion.style.open_set_routing import (
    PUBLIC_STYLE_ROUTING_VERSION,
    OpenSetRouteKind,
    OpenSetStyleRoute,
)
from companion.style.pragmatic_context import (
    PUBLIC_PRAGMATIC_CALIBRATION_VERSION,
    PragmaticDimension,
    PragmaticLabel,
)
from companion.style.user_profile import StyleFeatureKind

PERSONALIZATION_SPEC_VERSION = "dialogue-personalization-v1"

UserId = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
OpaqueReference = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=180,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9:._-]*$",
    ),
]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


class _FrozenModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PersonalizationCompileError(ValueError):
    """Fixed public failure for invalid or cross-user compiler input."""


class RetrievalStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    AVAILABLE = "available"
    REJECTED = "rejected"
    NO_MATCH = "no_match"
    LOW_CONFIDENCE = "low_confidence"
    CORRUPT = "corrupt"


class HardExclusion(StrEnum):
    PRESERVE_AI_IDENTITY = "preserve_ai_identity"
    PRESERVE_SAFETY_AND_PRIVACY = "preserve_safety_and_privacy"
    PRESERVE_FACTUAL_ACCURACY = "preserve_factual_accuracy"
    PRESERVE_CURRENT_INTENT = "preserve_current_intent"
    DO_NOT_FABRICATE_STYLE_MATCH = "do_not_fabricate_style_match"


_PHASE4_HARD_EXCLUSIONS = tuple(HardExclusion)


class PersonalizationProvenanceSource(StrEnum):
    STABLE_PROFILE = "stable_profile"
    PRAGMATIC_CALIBRATION = "pragmatic_calibration"
    SCENARIO_ROUTE = "scenario_route"
    DIALOGUE_POLICY = "dialogue_policy"
    RETRIEVED_EVIDENCE = "retrieved_evidence"


class PersonalizationObservedFrom(StrEnum):
    OWNER_MESSAGES_ONLY = "owner_messages_only"
    OWNER_DECLARATION = "owner_declaration"
    CURRENT_REQUEST = "current_request"
    RECENT_CONTEXT_PRESENCE = "recent_context_presence"
    DETERMINISTIC_REQUEST_STATE = "deterministic_request_state"
    PHASE3_ROUTE = "phase3_route"
    DIALOGUE_POLICY = "dialogue_policy"


class PersonalizationConfidenceMethod(StrEnum):
    DETERMINISTIC_STATISTIC = "deterministic_statistic"
    CALIBRATED_MODEL = "calibrated_model"
    EXPLICIT_RULE = "explicit_rule"
    USER_DECLARATION = "user_declaration"
    WILSON_LOWER_BOUND_95 = "wilson_lower_bound_95"
    CALIBRATED_ROUTE_SCORE = "calibrated_route_score"
    EXPLICIT_OWNER_APPROVAL = "explicit_owner_approval"


class StablePriorControl(_FrozenModel):
    user_id: UserId
    kind: StyleFeatureKind
    value: StrictFloat | StrictStr
    confidence: UnitInterval
    profile_version: OpaqueReference

    @model_validator(mode="after")
    def require_value_shape(self) -> Self:
        if self.kind is StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE:
            if type(self.value) is not str or not self.value.strip() or len(self.value) > 32:
                raise ValueError("confirmed expression control must be a short string")
        elif type(self.value) is not float:
            raise ValueError("statistical stable controls must use float values")
        return self


class ContextDeltaControl(_FrozenModel):
    user_id: UserId
    dimension: PragmaticDimension
    label: PragmaticLabel
    score: UnitInterval
    confidence: UnitInterval
    rule_id: OpaqueReference
    calibration_version: OpaqueReference

    @model_validator(mode="after")
    def require_registered_calibration(self) -> Self:
        if self.calibration_version != PUBLIC_PRAGMATIC_CALIBRATION_VERSION:
            raise ValueError("context control calibration version is not registered")
        return self


class ScenarioCandidateControl(_FrozenModel):
    user_id: UserId
    scenario_id: OpaqueReference
    matched_labels: tuple[PragmaticLabel, ...] = Field(min_length=1, max_length=8)
    missing_labels: tuple[PragmaticLabel, ...] = Field(default=(), max_length=8)
    coverage: UnitInterval
    raw_score: UnitInterval
    weight: UnitInterval
    auxiliary_intent: StyleIntent | None = None
    prototype_version: OpaqueReference

    @model_validator(mode="after")
    def require_explainable_candidate(self) -> Self:
        matched = set(self.matched_labels)
        missing = set(self.missing_labels)
        if (
            self.prototype_version != PUBLIC_STYLE_ROUTING_VERSION
            or len(matched) != len(self.matched_labels)
            or len(missing) != len(self.missing_labels)
            or matched & missing
        ):
            raise ValueError("scenario personalization candidate is invalid")
        expected_coverage = round(len(matched) / (len(matched) + len(missing)), 6)
        if self.coverage != expected_coverage or self.raw_score <= 0.0 or self.weight <= 0.0:
            raise ValueError("scenario personalization score is not explainable")
        return self


class RetrievedEvidenceControl(_FrozenModel):
    """Bounded approved expression used only inside the current Dialogue request."""

    user_id: UserId
    evidence_ref: OpaqueReference
    scenario_id: OpaqueReference
    content: Annotated[
        StrictStr,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
    ]
    confidence: UnitInterval
    evidence_version: OpaqueReference


class PersonalizationProvenance(_FrozenModel):
    control_ref: OpaqueReference
    source_type: PersonalizationProvenanceSource
    evidence_count: int = Field(strict=True, ge=1, le=1_000_000)
    observed_from: PersonalizationObservedFrom
    updated_at: datetime
    confidence_method: PersonalizationConfidenceMethod
    confidence: UnitInterval
    source_version: OpaqueReference

    @model_validator(mode="after")
    def require_aware_timestamp(self) -> Self:
        if self.updated_at.tzinfo is None:
            raise ValueError("personalization provenance timestamp must include a timezone")
        return self


class PersonalizationSpec(_FrozenModel):
    """Dialogue-internal, user-scoped continuous style control specification."""

    user_id: UserId
    spec_version: OpaqueReference
    authorization_version: OpaqueReference
    profile_version: OpaqueReference
    routing_version: OpaqueReference
    route_kind: OpenSetRouteKind
    source_route: OpenSetStyleRoute
    source_retrieval: StyleEvidenceResult | None = None
    stable_prior: tuple[StablePriorControl, ...] = Field(min_length=1, max_length=32)
    context_delta: tuple[ContextDeltaControl, ...] = Field(max_length=28)
    scenario_candidates: tuple[ScenarioCandidateControl, ...] = Field(max_length=4)
    retrieval_status: RetrievalStatus
    retrieved_evidence: tuple[RetrievedEvidenceControl, ...] = Field(max_length=8)
    hard_exclusions: tuple[HardExclusion, ...] = Field(min_length=1, max_length=8)
    provenance: tuple[PersonalizationProvenance, ...] = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def require_traceable_controls(self) -> Self:
        if self.spec_version != PERSONALIZATION_SPEC_VERSION:
            raise ValueError("personalization spec version is not registered")
        if self.routing_version != PUBLIC_STYLE_ROUTING_VERSION:
            raise ValueError("personalization routing version is not registered")
        controls = (
            *self.stable_prior,
            *self.context_delta,
            *self.scenario_candidates,
            *self.retrieved_evidence,
        )
        if any(control.user_id != self.user_id for control in controls):
            raise ValueError("personalization control user scope is inconsistent")
        if self.retrieval_status is RetrievalStatus.AVAILABLE:
            if not self.retrieved_evidence:
                raise ValueError("available retrieval requires evidence")
        elif self.retrieved_evidence:
            raise ValueError("non-available retrieval cannot contain evidence")
        if self.hard_exclusions != _PHASE4_HARD_EXCLUSIONS:
            raise ValueError("personalization hard exclusions are incomplete")
        if self.route_kind is OpenSetRouteKind.OPEN_SET:
            if self.scenario_candidates:
                raise ValueError("open-set personalization cannot contain scenario candidates")
        elif not self.scenario_candidates:
            raise ValueError("matched personalization requires scenario candidates")
        if (
            self.scenario_candidates
            and abs(sum(control.weight for control in self.scenario_candidates) - 1.0) > 0.000001
        ):
            raise ValueError("scenario personalization weights must be normalized")
        if self.route_kind is OpenSetRouteKind.KNOWN and (
            len(self.scenario_candidates) != 1 or self.scenario_candidates[0].coverage != 1.0
        ):
            raise ValueError("known personalization requires one complete candidate")
        if self.route_kind is OpenSetRouteKind.COMPOUND and (
            len(self.scenario_candidates) < 2
            or any(control.coverage != 1.0 for control in self.scenario_candidates)
        ):
            raise ValueError("compound personalization requires complete candidates")
        if self.route_kind is OpenSetRouteKind.NEARBY and any(
            not 0.5 <= control.coverage < 1.0 for control in self.scenario_candidates
        ):
            raise ValueError("nearby personalization requires partial candidates")
        if len({control.kind for control in self.stable_prior}) != len(self.stable_prior):
            raise ValueError("stable personalization controls must be unique")
        if any(control.profile_version != self.profile_version for control in self.stable_prior):
            raise ValueError("stable personalization profile version is inconsistent")
        context_keys = {(control.dimension, control.label) for control in self.context_delta}
        if len(context_keys) != len(self.context_delta):
            raise ValueError("context personalization controls must be unique")
        if len({control.scenario_id for control in self.scenario_candidates}) != len(
            self.scenario_candidates
        ):
            raise ValueError("scenario personalization controls must be unique")

        expected_refs = {
            *(_stable_ref(control.kind) for control in self.stable_prior),
            *(_context_ref(control.dimension, control.label) for control in self.context_delta),
            *(_scenario_ref(control.scenario_id) for control in self.scenario_candidates),
            *(_retrieved_ref(control.evidence_ref) for control in self.retrieved_evidence),
            *(_hard_ref(exclusion) for exclusion in self.hard_exclusions),
        }
        actual_refs = [item.control_ref for item in self.provenance]
        if len(actual_refs) != len(set(actual_refs)) or set(actual_refs) != expected_refs:
            raise ValueError("personalization provenance does not cover every control exactly once")
        provenance = {item.control_ref: item for item in self.provenance}
        for control in self.stable_prior:
            item = provenance[_stable_ref(control.kind)]
            if (
                item.source_type is not PersonalizationProvenanceSource.STABLE_PROFILE
                or item.confidence != control.confidence
                or item.observed_from
                not in {
                    PersonalizationObservedFrom.OWNER_MESSAGES_ONLY,
                    PersonalizationObservedFrom.OWNER_DECLARATION,
                }
                or (
                    item.observed_from is PersonalizationObservedFrom.OWNER_DECLARATION
                    and item.confidence_method
                    is not PersonalizationConfidenceMethod.USER_DECLARATION
                )
                or (
                    item.observed_from is PersonalizationObservedFrom.OWNER_MESSAGES_ONLY
                    and item.confidence_method is PersonalizationConfidenceMethod.USER_DECLARATION
                )
            ):
                raise ValueError("stable personalization provenance is inconsistent")
        for control in self.context_delta:
            item = provenance[_context_ref(control.dimension, control.label)]
            if (
                item.source_type is not PersonalizationProvenanceSource.PRAGMATIC_CALIBRATION
                or item.confidence != control.confidence
                or item.source_version != PUBLIC_PRAGMATIC_CALIBRATION_VERSION
                or item.confidence_method
                is not PersonalizationConfidenceMethod.WILSON_LOWER_BOUND_95
                or item.observed_from
                not in {
                    PersonalizationObservedFrom.CURRENT_REQUEST,
                    PersonalizationObservedFrom.RECENT_CONTEXT_PRESENCE,
                    PersonalizationObservedFrom.DETERMINISTIC_REQUEST_STATE,
                }
            ):
                raise ValueError("context personalization provenance is inconsistent")
        for control in self.scenario_candidates:
            item = provenance[_scenario_ref(control.scenario_id)]
            if (
                item.source_type is not PersonalizationProvenanceSource.SCENARIO_ROUTE
                or item.confidence != control.raw_score
                or item.source_version != self.routing_version
                or item.confidence_method
                is not PersonalizationConfidenceMethod.CALIBRATED_ROUTE_SCORE
                or item.observed_from is not PersonalizationObservedFrom.PHASE3_ROUTE
            ):
                raise ValueError("scenario personalization provenance is inconsistent")
        for control in self.retrieved_evidence:
            item = provenance[_retrieved_ref(control.evidence_ref)]
            if (
                item.source_type is not PersonalizationProvenanceSource.RETRIEVED_EVIDENCE
                or item.confidence != control.confidence
                or item.source_version != control.evidence_version
                or item.confidence_method
                is not PersonalizationConfidenceMethod.EXPLICIT_OWNER_APPROVAL
                or item.observed_from is not PersonalizationObservedFrom.OWNER_MESSAGES_ONLY
            ):
                raise ValueError("retrieved evidence provenance is inconsistent")
        for exclusion in self.hard_exclusions:
            item = provenance[_hard_ref(exclusion)]
            if (
                item.source_type is not PersonalizationProvenanceSource.DIALOGUE_POLICY
                or item.confidence != 1.0
                or item.source_version != self.spec_version
                or item.confidence_method is not PersonalizationConfidenceMethod.EXPLICIT_RULE
                or item.observed_from is not PersonalizationObservedFrom.DIALOGUE_POLICY
            ):
                raise ValueError("policy personalization provenance is inconsistent")
        validated_route = _validated_source_route(self.source_route)
        if validated_route.user_id != self.user_id:
            raise ValueError("personalization source route user scope is inconsistent")
        validated_retrieval = _validated_source_retrieval(self.source_retrieval)
        if validated_retrieval is not None and validated_retrieval.source_route != validated_route:
            raise ValueError("personalization retrieval route is inconsistent")
        expected = _compile_parts(validated_route, validated_retrieval)
        actual = _CompiledParts(
            authorization_version=self.authorization_version,
            profile_version=self.profile_version,
            routing_version=self.routing_version,
            route_kind=self.route_kind,
            stable_prior=self.stable_prior,
            context_delta=self.context_delta,
            scenario_candidates=self.scenario_candidates,
            retrieval_status=self.retrieval_status,
            retrieved_evidence=self.retrieved_evidence,
            hard_exclusions=self.hard_exclusions,
            provenance=self.provenance,
        )
        if actual != expected:
            raise ValueError("personalization controls are not reproducible from source route")
        return self


class ModelPersonalizationInstructions(_FrozenModel):
    user_id: UserId
    spec_version: OpaqueReference
    directives: tuple[str, ...] = Field(min_length=1, max_length=80)


@dataclass(frozen=True)
class _CompiledParts:
    authorization_version: str
    profile_version: str
    routing_version: str
    route_kind: OpenSetRouteKind
    stable_prior: tuple[StablePriorControl, ...]
    context_delta: tuple[ContextDeltaControl, ...]
    scenario_candidates: tuple[ScenarioCandidateControl, ...]
    retrieval_status: RetrievalStatus
    retrieved_evidence: tuple[RetrievedEvidenceControl, ...]
    hard_exclusions: tuple[HardExclusion, ...]
    provenance: tuple[PersonalizationProvenance, ...]


def _validated_source_route(route: OpenSetStyleRoute) -> OpenSetStyleRoute:
    if type(route) is not OpenSetStyleRoute:
        raise ValueError("personalization source route is invalid")
    return OpenSetStyleRoute.model_validate(route.model_dump(mode="python"))


def _validated_source_retrieval(
    result: StyleEvidenceResult | None,
) -> StyleEvidenceResult | None:
    if result is None:
        return None
    if type(result) is not StyleEvidenceResult:
        raise ValueError("personalization source retrieval is invalid")
    return StyleEvidenceResult.model_validate(result.model_dump(mode="python"))


def _compile_parts(
    route: OpenSetStyleRoute,
    retrieval_result: StyleEvidenceResult | None = None,
) -> _CompiledParts:
    user_id = route.user_id
    profile = route.stable_profile
    stable_prior: list[StablePriorControl] = []
    context_delta: list[ContextDeltaControl] = []
    scenario_candidates: list[ScenarioCandidateControl] = []
    retrieved_evidence: list[RetrievedEvidenceControl] = []
    provenance: list[PersonalizationProvenance] = []

    for feature in profile.features:
        stable_prior.append(
            StablePriorControl(
                user_id=user_id,
                kind=feature.kind,
                value=feature.value,
                confidence=feature.confidence,
                profile_version=profile.profile_version,
            )
        )
        source = feature.provenance
        provenance.append(
            PersonalizationProvenance(
                control_ref=_stable_ref(feature.kind),
                source_type=PersonalizationProvenanceSource.STABLE_PROFILE,
                evidence_count=source.evidence_count,
                observed_from=source.observed_from.value,
                updated_at=source.updated_at,
                confidence_method=source.confidence_method.value,
                confidence=feature.confidence,
                source_version=source.source_version,
            )
        )

    for signal in route.context_delta.all_signals:
        rule_id = signal.calibration.matched_rule_ids[0]
        context_delta.append(
            ContextDeltaControl(
                user_id=user_id,
                dimension=signal.dimension,
                label=signal.label,
                score=signal.score,
                confidence=signal.confidence,
                rule_id=rule_id,
                calibration_version=signal.calibration.calibration_version,
            )
        )
        provenance.append(
            PersonalizationProvenance(
                control_ref=_context_ref(signal.dimension, signal.label),
                source_type=PersonalizationProvenanceSource.PRAGMATIC_CALIBRATION,
                evidence_count=signal.provenance.evidence_count,
                observed_from=signal.provenance.observed_from.value,
                updated_at=signal.provenance.updated_at,
                confidence_method=signal.provenance.confidence_method.value,
                confidence=signal.confidence,
                source_version=signal.calibration.calibration_version,
            )
        )

    for candidate in route.candidates:
        scenario_candidates.append(
            ScenarioCandidateControl(
                user_id=user_id,
                scenario_id=candidate.scenario_id,
                matched_labels=candidate.matched_labels,
                missing_labels=candidate.missing_labels,
                coverage=candidate.coverage,
                raw_score=candidate.raw_score,
                weight=candidate.weight,
                auxiliary_intent=candidate.auxiliary_intent,
                prototype_version=candidate.prototype_version,
            )
        )
        provenance.append(
            PersonalizationProvenance(
                control_ref=_scenario_ref(candidate.scenario_id),
                source_type=PersonalizationProvenanceSource.SCENARIO_ROUTE,
                evidence_count=len(candidate.matched_labels),
                observed_from="phase3_route",
                updated_at=route.context_delta.created_at,
                confidence_method="calibrated_route_score",
                confidence=candidate.raw_score,
                source_version=route.routing_version,
            )
        )

    retrieval_status = RetrievalStatus.NOT_REQUESTED
    if retrieval_result is not None:
        if retrieval_result.user_id != user_id or retrieval_result.source_route != route:
            raise ValueError("personalization retrieval scope is inconsistent")
        retrieval_status = RetrievalStatus(retrieval_result.status.value)
        evidence_version = (
            retrieval_result.versions.evidence_version
            if retrieval_result.versions is not None
            else "not-requested"
        )
        for match in retrieval_result.matches:
            retrieved_evidence.append(
                RetrievedEvidenceControl(
                    user_id=user_id,
                    evidence_ref=match.evidence_ref,
                    scenario_id=match.scenario_id,
                    content=match.content,
                    confidence=match.confidence,
                    evidence_version=evidence_version,
                )
            )
            provenance.append(
                PersonalizationProvenance(
                    control_ref=_retrieved_ref(match.evidence_ref),
                    source_type=PersonalizationProvenanceSource.RETRIEVED_EVIDENCE,
                    evidence_count=1,
                    observed_from=match.observed_from.value,
                    updated_at=match.approved_at,
                    confidence_method=match.confidence_method.value,
                    confidence=match.confidence,
                    source_version=evidence_version,
                )
            )
    for exclusion in _PHASE4_HARD_EXCLUSIONS:
        provenance.append(
            PersonalizationProvenance(
                control_ref=_hard_ref(exclusion),
                source_type=PersonalizationProvenanceSource.DIALOGUE_POLICY,
                evidence_count=1,
                observed_from="dialogue_policy",
                updated_at=route.context_delta.created_at,
                confidence_method="explicit_rule",
                confidence=1.0,
                source_version=PERSONALIZATION_SPEC_VERSION,
            )
        )

    return _CompiledParts(
        authorization_version=profile.authorization_version,
        profile_version=profile.profile_version,
        routing_version=route.routing_version,
        route_kind=route.route_kind,
        stable_prior=tuple(stable_prior),
        context_delta=tuple(context_delta),
        scenario_candidates=tuple(scenario_candidates),
        retrieval_status=retrieval_status,
        retrieved_evidence=tuple(retrieved_evidence),
        hard_exclusions=_PHASE4_HARD_EXCLUSIONS,
        provenance=tuple(provenance),
    )


def compile_personalization_spec(
    *,
    user_id: str,
    route: OpenSetStyleRoute,
    retrieval_result: StyleEvidenceResult | None = None,
) -> PersonalizationSpec:
    """Compile a self-verified Phase 3 route without retaining request text."""
    try:
        if type(user_id) is not str:
            raise ValueError
        validated_route = _validated_source_route(route)
        if validated_route.user_id != user_id:
            raise ValueError
        validated_retrieval = _validated_source_retrieval(retrieval_result)
        if validated_retrieval is not None and validated_retrieval.source_route != validated_route:
            raise ValueError
        parts = _compile_parts(validated_route, validated_retrieval)
        return PersonalizationSpec(
            user_id=user_id,
            spec_version=PERSONALIZATION_SPEC_VERSION,
            source_route=validated_route,
            source_retrieval=validated_retrieval,
            authorization_version=parts.authorization_version,
            profile_version=parts.profile_version,
            routing_version=parts.routing_version,
            route_kind=parts.route_kind,
            stable_prior=parts.stable_prior,
            context_delta=parts.context_delta,
            scenario_candidates=parts.scenario_candidates,
            retrieval_status=parts.retrieval_status,
            retrieved_evidence=parts.retrieved_evidence,
            hard_exclusions=parts.hard_exclusions,
            provenance=parts.provenance,
        )
    except Exception as error:
        _detach_exception(error)
        raise PersonalizationCompileError("personalization input is invalid") from None


def adapt_personalization_spec(
    *,
    user_id: str,
    spec: PersonalizationSpec,
) -> ModelPersonalizationInstructions:
    """Translate a validated spec into bounded model directives for one request."""
    try:
        if type(user_id) is not str or type(spec) is not PersonalizationSpec:
            raise ValueError
        validated = PersonalizationSpec.model_validate(spec.model_dump(mode="python"))
        if validated.user_id != user_id:
            raise ValueError
        provenance = {item.control_ref: item for item in validated.provenance}
        directives = [
            "priority=identity,safety,privacy,factual_accuracy,current_intent,personalization",
            f"route_kind={validated.route_kind.value}",
            f"retrieval_status={validated.retrieval_status.value}",
        ]
        for control in validated.stable_prior:
            source = provenance[_stable_ref(control.kind)]
            directives.append(
                "stable_prior "
                f"kind={control.kind.value} value={_render_value(control.value)} "
                f"confidence={control.confidence:.6f} source={source.source_version}"
            )
        for control in validated.context_delta:
            source = provenance[_context_ref(control.dimension, control.label)]
            directives.append(
                "context_delta "
                f"dimension={control.dimension.value} label={control.label.value} "
                f"score={control.score:.6f} confidence={control.confidence:.6f} "
                f"source={source.source_version}"
            )
        for control in validated.scenario_candidates:
            source = provenance[_scenario_ref(control.scenario_id)]
            directives.append(
                "scenario_candidate "
                f"scenario={_escape_directive_text(control.scenario_id)} "
                f"coverage={control.coverage:.6f} raw_score={control.raw_score:.6f} "
                f"weight={control.weight:.6f} source={source.source_version}"
            )
        for control in validated.retrieved_evidence:
            source = provenance[_retrieved_ref(control.evidence_ref)]
            directives.append(
                "retrieved_evidence "
                f"scenario={_escape_directive_text(control.scenario_id)} "
                f"confidence={control.confidence:.6f} "
                f"expression={_escape_directive_text(control.content)} "
                f"source={source.source_version}"
            )
        directives.extend(
            f"hard_exclusion={exclusion.value}" for exclusion in validated.hard_exclusions
        )
        return ModelPersonalizationInstructions(
            user_id=user_id,
            spec_version=validated.spec_version,
            directives=tuple(directives),
        )
    except Exception as error:
        _detach_exception(error)
        raise PersonalizationCompileError("personalization input is invalid") from None


def _stable_ref(kind: StyleFeatureKind) -> str:
    return f"stable:{kind.value}"


def _context_ref(dimension: PragmaticDimension, label: PragmaticLabel) -> str:
    return f"context:{dimension.value}:{label.value}"


def _scenario_ref(scenario_id: str) -> str:
    return f"scenario:{scenario_id}"


def _retrieved_ref(evidence_ref: str) -> str:
    return f"retrieved:{evidence_ref}"


def _hard_ref(exclusion: HardExclusion) -> str:
    return f"hard:{exclusion.value}"


def _render_value(value: float | str) -> str:
    if type(value) is float:
        return f"{value:.6f}"
    return _escape_directive_text(value)


def _escape_directive_text(value: str) -> str:
    escaped: list[str] = []
    for character in value:
        if character == "\\":
            escaped.append("\\\\")
        elif character == "\r":
            escaped.append("\\r")
        elif character == "\n":
            escaped.append("\\n")
        elif character in "[]<>":
            escaped.append(f"\\{character}")
        elif character in "\u2028\u2029":
            escaped.append(f"\\u{ord(character):04x}")
        elif unicodedata.category(character) == "Cc":
            escaped.append(f"\\x{ord(character):02x}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _detach_exception(error: BaseException) -> None:
    with suppress(Exception):
        error.__traceback__ = None
        error.__context__ = None
        error.__cause__ = None
