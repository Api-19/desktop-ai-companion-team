"""Public synthetic, component-only Style framework evaluation (Phase 8)."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from companion.contracts import StyleReplyGuidance
from companion.style.evidence_retrieval import (
    STYLE_EVIDENCE_RETRIEVAL_VERSION,
    StyleEvidenceApproval,
    StyleEvidenceCatalog,
    StyleEvidenceConfidenceMethod,
    StyleEvidenceObservedFrom,
    StyleEvidenceRecord,
    StyleEvidenceResult,
    StyleEvidenceRetrievalStatus,
    StyleEvidenceSourceType,
    StyleEvidenceVersions,
    retrieve_style_evidence,
)
from companion.style.open_set_routing import (
    PUBLIC_STYLE_ROUTING_VERSION,
    OpenSetRouteKind,
    StyleScenarioPrototype,
    route_open_set_style,
)
from companion.style.pragmatic_context import PragmaticLabel, build_pragmatic_context_delta
from companion.style.user_profile import (
    StyleConfidenceMethod,
    StyleFeatureKind,
    StyleFeatureProvenance,
    StyleFeatureSourceType,
    StyleObservedFrom,
    UserStyleDerivedArtifacts,
    UserStyleFeature,
    UserStyleProfile,
    UserStyleProfileConsent,
    UserStyleProfileStatus,
    UserStyleSnapshot,
)
from companion.style.verifier import DeterministicStyleVerifier

FIXTURE_VERSION = "public-synthetic-style-framework-v1"
INPUT_VERSION = "public-synthetic-input-v1"
PARAMETER_VERSION = "offline-fixed-parameters-v1"
EVALUATOR_VERSION = "offline-style-framework-evaluator-v1"
_DIGEST_VERSION = "sha256-canonical-json-v1"
_NOW = datetime(2026, 9, 23, tzinfo=UTC)
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "synthetic"
    / "style_framework_evaluation_v1.json"
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvaluationLane(StrEnum):
    BASELINE = "baseline"
    LEGACY_HARD_RAG = "legacy_hard_rag"
    PHASE1_7 = "phase1_7"


class EvaluationLayer(StrEnum):
    KNOWN = "known"
    NEARBY = "nearby"
    OPEN_SET = "open_set"
    COMPOUND = "compound"


class MetricStatus(StrEnum):
    MEASURED = "measured"
    NOT_MEASURED = "not_measured"
    NOT_APPLICABLE = "not_applicable"


class FrameworkScenario(_Frozen):
    scenario_id: StrictStr = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9-]+$")
    layer: EvaluationLayer
    user_id: Literal["synthetic-owner-a", "synthetic-owner-b"]
    text: StrictStr = Field(min_length=1, max_length=200)
    recent_context: tuple[StrictStr, ...] = Field(max_length=4)
    idle_seconds: int = Field(ge=0)
    expected_route: OpenSetRouteKind
    retrieval_case: Literal[
        "not_requested", "available", "no_match", "low_confidence", "corrupt", "unavailable"
    ]
    reply: StrictStr | None = Field(default=None, min_length=1, max_length=200)
    baseline_reply: StrictStr | None = Field(min_length=1, max_length=200)
    legacy_reply: StrictStr | None = Field(min_length=1, max_length=200)
    allowed_example: StrictStr = Field(min_length=1, max_length=160)
    privacy_lure: StrictStr = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def check_layer(self) -> Self:
        if self.expected_route.value != self.layer.value:
            raise ValueError("invalid framework scenario layer")
        return self


class FrameworkFixture(_Frozen):
    fixture_version: Literal[FIXTURE_VERSION]
    input_version: Literal[INPUT_VERSION]
    parameter_version: Literal[PARAMETER_VERSION]
    scenarios: tuple[FrameworkScenario, ...] = Field(min_length=16, max_length=64)

    @model_validator(mode="after")
    def check_set(self) -> Self:
        ids = [item.scenario_id for item in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate framework scenario")
        layers = Counter(item.layer for item in self.scenarios)
        if any(layers[layer] < 3 for layer in EvaluationLayer):
            raise ValueError("framework layers are incomplete")
        return self


class LaneObservation(_Frozen):
    lane: EvaluationLane
    scenario_id: StrictStr
    user_id: StrictStr
    input_version: Literal[INPUT_VERSION]
    parameter_version: Literal[PARAMETER_VERSION]
    digest_version: Literal[_DIGEST_VERSION]
    request_digest: StrictStr = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    route_kind: OpenSetRouteKind | None
    abstained: bool | None
    stable_prior_count: int | None = Field(ge=0)
    retrieval_status: StrictStr | None
    evidence_count: int | None = Field(ge=0)
    reply: StrictStr | None
    source_version: StrictStr


class MetricEvidence(_Frozen):
    lane: EvaluationLane
    layer: EvaluationLayer
    metric: StrictStr
    status: MetricStatus
    numerator: int | None = Field(default=None, ge=0)
    denominator: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_count(self) -> Self:
        if self.status is MetricStatus.MEASURED:
            if (
                self.numerator is None
                or self.denominator is None
                or self.numerator > self.denominator
            ):
                raise ValueError("invalid measured metric")
        elif self.numerator is not None or self.denominator is not None:
            raise ValueError("unmeasured metric has a count")
        return self


class FrameworkReport(_Frozen):
    fixture_version: Literal[FIXTURE_VERSION]
    evaluator_version: Literal[EVALUATOR_VERSION]
    parameter_version: Literal[PARAMETER_VERSION]
    scenario_count: int = Field(ge=16)
    layer_counts: dict[EvaluationLayer, int]
    metrics: tuple[MetricEvidence, ...]
    evidence_level: Literal["offline_public_synthetic_component"] = (
        "offline_public_synthetic_component"
    )


class BlindItem(_Frozen):
    scenario_id: StrictStr
    answers: dict[Literal["A", "B", "C"], StrictStr]

    @model_validator(mode="after")
    def check_answers(self) -> Self:
        if set(self.answers) != {"A", "B", "C"}:
            raise ValueError("blind answers are incomplete")
        return self


class BlindBundle(_Frozen):
    fixture_version: Literal[FIXTURE_VERSION]
    bundle_digest: StrictStr = Field(min_length=64, max_length=64)
    items: tuple[BlindItem, ...]


class SeparateAnswerKey(_Frozen):
    fixture_version: Literal[FIXTURE_VERSION]
    bundle_digest: StrictStr = Field(min_length=64, max_length=64)
    seed: int
    labels: dict[StrictStr, dict[Literal["A", "B", "C"], EvaluationLane]]


class BlindRating(_Frozen):
    scenario_id: StrictStr
    preferred: Literal["A", "B", "C", "none"]
    similarity: int = Field(ge=0, le=5)


class CompleteBlindRatings(_Frozen):
    bundle_digest: StrictStr = Field(min_length=64, max_length=64)
    items: tuple[BlindRating, ...]


def _digest(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _request_digest(scenario: FrameworkScenario) -> str:
    return _digest(
        {
            "scenario_id": scenario.scenario_id,
            "user_id": scenario.user_id,
            "text": scenario.text,
            "recent_context": scenario.recent_context,
            "idle_seconds": scenario.idle_seconds,
            "input_version": INPUT_VERSION,
            "parameter_version": PARAMETER_VERSION,
        }
    )


def load_framework_fixture() -> FrameworkFixture:
    """Load the one fixed, tracked public synthetic fixture."""
    try:
        return FrameworkFixture.model_validate_json(_FIXTURE_PATH.read_text(encoding="utf-8"))
    except Exception:
        raise ValueError("invalid public framework fixture") from None


def _snapshot(user_id: str) -> UserStyleSnapshot:
    provenance = StyleFeatureProvenance(
        source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
        evidence_count=8,
        observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
        source_version="public-safe-v1",
        updated_at=_NOW,
        confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
    )
    return UserStyleSnapshot(
        consent=UserStyleProfileConsent(
            user_id=user_id,
            authorization_version="public-consent-v1",
            status=UserStyleProfileStatus.AUTHORIZED,
            updated_at=_NOW,
        ),
        profile=UserStyleProfile(
            user_id=user_id,
            profile_version="public-profile-v1",
            authorization_version="public-consent-v1",
            features=(
                UserStyleFeature(
                    kind=StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
                    value=8.0,
                    confidence=0.9,
                    provenance=provenance,
                ),
            ),
            created_at=_NOW,
            updated_at=_NOW,
        ),
        evidence_count=8,
        derived=UserStyleDerivedArtifacts(
            user_id=user_id,
            profile_version="public-profile-v1",
            authorization_version="public-consent-v1",
            statistics_version="public-stats-v1",
            index_version="public-index-v1",
            cache_version="public-cache-v1",
            updated_at=_NOW,
        ),
    )


_PROTOTYPES: tuple[tuple[str, tuple[PragmaticLabel, ...], str], ...] = (
    ("gratitude", (PragmaticLabel.GRATITUDE,), "scenario-gratitude-v1"),
    (
        "sleep_farewell",
        (PragmaticLabel.FAREWELL, PragmaticLabel.CLOSING),
        "scenario-sleep-farewell-v1",
    ),
    ("affection", (PragmaticLabel.AFFECTION, PragmaticLabel.AFFECTIONATE), "scenario-affection-v1"),
    (
        "support_request",
        (PragmaticLabel.REQUEST, PragmaticLabel.SUPPORT_SEEKING),
        "scenario-support-request-v1",
    ),
    ("care_support", (PragmaticLabel.CARE_GIVING,), "scenario-care-support-v1"),
    ("sad_checkin", (PragmaticLabel.SADNESS, PragmaticLabel.QUESTION), "scenario-sad-checkin-v1"),
    (
        "sad_support",
        (PragmaticLabel.SADNESS, PragmaticLabel.SUPPORT_SEEKING),
        "scenario-sad-support-v1",
    ),
    ("generic_question", (PragmaticLabel.QUESTION,), "scenario-generic-question-v1"),
)


def _catalog(user_id: str) -> tuple[StyleScenarioPrototype, ...]:
    return tuple(
        StyleScenarioPrototype(
            user_id=user_id,
            scenario_id=scenario_id,
            required_labels=labels,
            prototype_version=PUBLIC_STYLE_ROUTING_VERSION,
            semantic_definition_id=semantic_id,
        )
        for scenario_id, labels, semantic_id in _PROTOTYPES
    )


def _retrieval(route: object, scenario: FrameworkScenario) -> StyleEvidenceResult | None:
    from companion.style.open_set_routing import OpenSetStyleRoute

    if type(route) is not OpenSetStyleRoute:
        raise ValueError("invalid framework route")
    if (
        scenario.retrieval_case == "unavailable"
        and route.route_kind is not OpenSetRouteKind.OPEN_SET
    ):
        return None
    if scenario.retrieval_case == "corrupt" and route.route_kind is not OpenSetRouteKind.OPEN_SET:
        return StyleEvidenceResult(
            user_id=scenario.user_id,
            retrieval_version=STYLE_EVIDENCE_RETRIEVAL_VERSION,
            route_kind=route.route_kind,
            source_route=route,
            status=StyleEvidenceRetrievalStatus.CORRUPT,
        )
    records: tuple[StyleEvidenceRecord, ...] = ()
    if scenario.retrieval_case in {"available", "low_confidence"} and route.candidates:
        records = (
            StyleEvidenceRecord(
                user_id=scenario.user_id,
                evidence_ref="public-evidence-v1",
                scenario_id=route.candidates[0].scenario_id,
                content=scenario.allowed_example,
                confidence=0.9 if scenario.retrieval_case == "available" else 0.1,
                approval=StyleEvidenceApproval.APPROVED_FOR_STYLE,
                source_type=StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL,
                observed_from=StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY,
                confidence_method=StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL,
                source_version="public-evidence-v1",
                approved_at=_NOW,
            ),
        )
    catalog = StyleEvidenceCatalog(
        user_id=scenario.user_id,
        authorization_version="public-consent-v1",
        profile_version="public-profile-v1",
        versions=StyleEvidenceVersions(
            evidence_version="public-evidence-v1",
            index_version="public-index-v1",
            cache_version="public-cache-v1",
        ),
        records=records,
        updated_at=_NOW,
    )
    return retrieve_style_evidence(
        user_id=scenario.user_id,
        trusted_user_id=scenario.user_id,
        route=route,
        catalog=catalog,
    )


_LEGACY_EXACT = frozenset({"谢谢", "晚安", "早上好", "我还没吃饭", "天气转冷"})


def _observation(scenario: FrameworkScenario, lane: EvaluationLane) -> LaneObservation:
    from companion.dialogue.personalization import compile_personalization_spec

    common = dict(
        lane=lane,
        scenario_id=scenario.scenario_id,
        user_id=scenario.user_id,
        input_version=INPUT_VERSION,
        parameter_version=PARAMETER_VERSION,
        digest_version=_DIGEST_VERSION,
        request_digest=_request_digest(scenario),
    )
    if lane is EvaluationLane.BASELINE:
        return LaneObservation(
            **common,
            route_kind=None,
            abstained=None,
            stable_prior_count=None,
            retrieval_status=None,
            evidence_count=None,
            reply=scenario.baseline_reply,
            source_version="baseline-recording-fake-v1",
        )
    if lane is EvaluationLane.LEGACY_HARD_RAG:
        hard_match = scenario.text in _LEGACY_EXACT
        return LaneObservation(
            **common,
            route_kind=None,
            abstained=not hard_match,
            stable_prior_count=None,
            retrieval_status=None,
            evidence_count=None,
            reply=scenario.legacy_reply,
            source_version="legacy-hard-rag-recording-fake-v1",
        )
    delta = build_pragmatic_context_delta(
        user_id=scenario.user_id,
        trusted_user_id=scenario.user_id,
        text=scenario.text,
        recent_context=scenario.recent_context,
        idle_seconds=scenario.idle_seconds,
        observed_at=_NOW,
    )
    route = route_open_set_style(
        user_id=scenario.user_id,
        trusted_user_id=scenario.user_id,
        snapshot=_snapshot(scenario.user_id),
        context_delta=delta,
        scenario_catalog=_catalog(scenario.user_id),
    )
    retrieval = _retrieval(route, scenario)
    spec = compile_personalization_spec(
        user_id=scenario.user_id, route=route, retrieval_result=retrieval
    )
    return LaneObservation(
        **common,
        route_kind=route.route_kind,
        abstained=route.abstained,
        stable_prior_count=len(spec.stable_prior),
        retrieval_status=(retrieval.status.value if retrieval else "unavailable"),
        evidence_count=len(spec.retrieved_evidence),
        reply=scenario.reply,
        source_version="phase2-5-production-pure-functions-v1",
    )


def build_framework_observations(fixture: FrameworkFixture) -> tuple[LaneObservation, ...]:
    """Run production Phase 2-5 pure components and fixed baseline/legacy fakes."""
    try:
        fixture = FrameworkFixture.model_validate(fixture.model_dump(mode="python"))
        return tuple(
            _observation(scenario, lane)
            for scenario in fixture.scenarios
            for lane in EvaluationLane
        )
    except Exception:
        raise ValueError("invalid framework evaluation inputs") from None


def _compact(text: str) -> str:
    return "".join(text.split())


def _longest_common(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    longest = 0
    for character in left:
        row = [0]
        for index, other in enumerate(right, start=1):
            value = previous[index - 1] + 1 if character == other else 0
            row.append(value)
            longest = max(longest, value)
        previous = row
    return longest


def mechanical_flags(
    reply: str, examples: tuple[str, ...], repeated_count: int
) -> tuple[bool, bool, bool]:
    """Return exact-copy, long-span-copy, and three-scenario repetition flags."""
    compact = _compact(reply)
    exact = any(" ".join(reply.split()) == " ".join(item.split()) for item in examples)
    span = any(
        bool(compact)
        and _longest_common(compact, _compact(item)) >= 8
        and _longest_common(compact, _compact(item)) / len(compact) >= 0.6
        for item in examples
    )
    return exact, span, repeated_count >= 3


_METRICS = (
    "route_correct",
    "abstain_correct",
    "semantic_constraint",
    "stable_prior_retained",
    "retrieval_status_correct",
    "no_fabricated_evidence",
    "privacy_lure_absent",
    "identity_safe",
    "exact_example_copy_absent",
    "long_span_copy_absent",
    "triple_repeat_absent",
)


def score_framework_observations(
    fixture: FrameworkFixture, observations: tuple[LaneObservation, ...]
) -> FrameworkReport:
    """Deep check actuals against adapters, then aggregate comparable metrics."""
    try:
        fixture = FrameworkFixture.model_validate(fixture.model_dump(mode="python"))
        supplied = tuple(
            LaneObservation.model_validate(item.model_dump(mode="python")) for item in observations
        )
        expected_actual = build_framework_observations(fixture)
        if (
            tuple(
                item.model_copy(update={"reply": actual.reply})
                for item, actual in zip(supplied, expected_actual, strict=True)
            )
            != expected_actual
        ):
            raise ValueError
        indexed = {(item.scenario_id, item.lane): item for item in supplied}
        if len(indexed) != len(supplied):
            raise ValueError
        answer_counts = Counter(
            (item.lane, " ".join(item.reply.split())) for item in supplied if item.reply is not None
        )
        metrics: list[MetricEvidence] = []
        for lane in EvaluationLane:
            for layer in EvaluationLayer:
                selected = [scenario for scenario in fixture.scenarios if scenario.layer is layer]
                for metric in _METRICS:
                    values: list[bool] = []
                    applicable = False
                    for scenario in selected:
                        observation = indexed[(scenario.scenario_id, lane)]
                        if metric == "route_correct":
                            if lane is EvaluationLane.PHASE1_7:
                                applicable = True
                                values.append(observation.route_kind is scenario.expected_route)
                        elif metric == "abstain_correct":
                            if lane is EvaluationLane.PHASE1_7:
                                applicable = True
                                values.append(
                                    observation.abstained == (layer is EvaluationLayer.OPEN_SET)
                                )
                        elif metric == "stable_prior_retained":
                            if lane is EvaluationLane.PHASE1_7:
                                applicable = True
                                values.append(bool(observation.stable_prior_count))
                        elif metric == "retrieval_status_correct":
                            if lane is EvaluationLane.PHASE1_7:
                                applicable = True
                                values.append(
                                    observation.retrieval_status == scenario.retrieval_case
                                )
                        elif metric == "no_fabricated_evidence":
                            if lane is EvaluationLane.PHASE1_7:
                                applicable = True
                                expected_count = 1 if scenario.retrieval_case == "available" else 0
                                values.append(observation.evidence_count == expected_count)
                        elif metric == "semantic_constraint":
                            applicable = True
                            if observation.reply is not None:
                                verification = DeterministicStyleVerifier(
                                    trusted_user_id=scenario.user_id
                                ).verify(
                                    user_id=scenario.user_id,
                                    request_text=scenario.text,
                                    response_text=observation.reply,
                                    guidance=StyleReplyGuidance(
                                        status="no_match", style_version="public-synthetic-v1"
                                    ),
                                    mode=(
                                        "baseline" if lane is EvaluationLane.BASELINE else "styled"
                                    ),
                                    previous_soft_lock=None,
                                )
                                if verification.semantic_correctness.covered:
                                    values.append(verification.semantic_correctness.passed)
                        elif metric in {"privacy_lure_absent", "identity_safe"}:
                            applicable = True
                            if observation.reply is not None:
                                values.append(
                                    scenario.privacy_lure not in observation.reply
                                    if metric == "privacy_lure_absent"
                                    else DeterministicStyleVerifier(
                                        trusted_user_id=scenario.user_id
                                    )
                                    .verify(
                                        user_id=scenario.user_id,
                                        request_text=scenario.text,
                                        response_text=observation.reply,
                                        guidance=StyleReplyGuidance(
                                            status="no_match", style_version="public-synthetic-v1"
                                        ),
                                        mode=(
                                            "baseline"
                                            if lane is EvaluationLane.BASELINE
                                            else "styled"
                                        ),
                                        previous_soft_lock=None,
                                    )
                                    .identity_confusion.passed
                                )
                        else:
                            applicable = True
                            if observation.reply is not None:
                                count = answer_counts[(lane, " ".join(observation.reply.split()))]
                                exact, span, repeat = mechanical_flags(
                                    observation.reply, (scenario.allowed_example,), count
                                )
                                values.append(
                                    not {
                                        "exact_example_copy_absent": exact,
                                        "long_span_copy_absent": span,
                                        "triple_repeat_absent": repeat,
                                    }[metric]
                                )
                    status = (
                        MetricStatus.MEASURED
                        if values
                        else MetricStatus.NOT_MEASURED
                        if applicable
                        else MetricStatus.NOT_APPLICABLE
                    )
                    metrics.append(
                        MetricEvidence(
                            lane=lane,
                            layer=layer,
                            metric=metric,
                            status=status,
                            numerator=sum(values) if values else None,
                            denominator=len(values) if values else None,
                        )
                    )
        return FrameworkReport(
            fixture_version=FIXTURE_VERSION,
            evaluator_version=EVALUATOR_VERSION,
            parameter_version=PARAMETER_VERSION,
            scenario_count=len(fixture.scenarios),
            layer_counts={
                layer: sum(s.layer is layer for s in fixture.scenarios) for layer in EvaluationLayer
            },
            metrics=tuple(metrics),
        )
    except Exception:
        raise ValueError("invalid framework evaluation observations") from None


def create_framework_blind_bundle(
    fixture: FrameworkFixture, observations: tuple[LaneObservation, ...], *, seed: int
) -> tuple[BlindBundle, SeparateAnswerKey]:
    """Create separate public placeholder answers and sealed lane mapping."""
    try:
        if type(seed) is not int:
            raise ValueError
        score_framework_observations(fixture, observations)
        indexed = {(item.scenario_id, item.lane): item for item in observations}
        generator = random.Random(seed)
        items: list[BlindItem] = []
        labels: dict[str, dict[Literal["A", "B", "C"], EvaluationLane]] = {}
        for scenario in fixture.scenarios:
            lanes = list(EvaluationLane)
            generator.shuffle(lanes)
            mapping = dict(zip(("A", "B", "C"), lanes, strict=True))
            labels[scenario.scenario_id] = mapping  # type: ignore[assignment]
            items.append(
                BlindItem(
                    scenario_id=scenario.scenario_id,
                    answers={
                        label: indexed[(scenario.scenario_id, lane)].reply
                        or "合成占位 - 未生成回答"
                        for label, lane in mapping.items()
                    },
                )
            )
        digest = _digest([item.model_dump(mode="json") for item in items])
        return (
            BlindBundle(fixture_version=FIXTURE_VERSION, bundle_digest=digest, items=tuple(items)),
            SeparateAnswerKey(
                fixture_version=FIXTURE_VERSION, bundle_digest=digest, seed=seed, labels=labels
            ),
        )
    except Exception:
        raise ValueError("invalid framework blind inputs") from None


def unblind_complete_framework_ratings(
    bundle: BlindBundle, key: SeparateAnswerKey, ratings: CompleteBlindRatings
) -> dict[str, EvaluationLane | None]:
    """Resolve lane choices only after one complete, valid rating per scenario."""
    try:
        bundle = BlindBundle.model_validate(bundle.model_dump(mode="python"))
        key = SeparateAnswerKey.model_validate(key.model_dump(mode="python"))
        ratings = CompleteBlindRatings.model_validate(ratings.model_dump(mode="python"))
        ids = [item.scenario_id for item in bundle.items]
        rated = [item.scenario_id for item in ratings.items]
        if (
            len(ids) != len(set(ids))
            or len(rated) != len(set(rated))
            or set(rated) != set(ids)
            or set(key.labels) != set(ids)
            or bundle.bundle_digest != key.bundle_digest
            or bundle.bundle_digest != ratings.bundle_digest
            or _digest([item.model_dump(mode="json") for item in bundle.items])
            != bundle.bundle_digest
        ):
            raise ValueError
        if key.bundle_digest != ratings.bundle_digest:
            raise ValueError
        if any(
            set(key.labels[item]) != {"A", "B", "C"}
            or set(key.labels[item].values()) != set(EvaluationLane)
            for item in ids
        ):
            raise ValueError
        generator = random.Random(key.seed)
        for scenario_id in ids:
            lanes = list(EvaluationLane)
            generator.shuffle(lanes)
            if key.labels[scenario_id] != dict(zip(("A", "B", "C"), lanes, strict=True)):
                raise ValueError
        return {
            item.scenario_id: (
                None if item.preferred == "none" else key.labels[item.scenario_id][item.preferred]
            )
            for item in ratings.items
        }
    except Exception:
        raise ValueError("incomplete or invalid framework ratings") from None
