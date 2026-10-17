from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import companion.contracts as shared_contracts
from companion.contracts import StyleIntent, StyleReplyGuidance
from companion.dialogue.context import PreparedDialogueInput, build_prompt
from companion.dialogue.models import DialogueMode
from companion.dialogue.personalization import (
    HardExclusion,
    PersonalizationCompileError,
    PersonalizationProvenanceSource,
    PersonalizationSpec,
    RetrievalStatus,
    adapt_personalization_spec,
    compile_personalization_spec,
)
from companion.style import (
    OpenSetRouteKind,
    OpenSetStyleRoute,
    PragmaticLabel,
    StyleConfidenceMethod,
    StyleFeatureKind,
    StyleFeatureProvenance,
    StyleFeatureSourceType,
    StyleObservedFrom,
    StyleScenarioPrototype,
    UserStyleDerivedArtifacts,
    UserStyleFeature,
    UserStyleProfile,
    UserStyleProfileConsent,
    UserStyleProfileStatus,
    UserStyleSnapshot,
    build_pragmatic_context_delta,
    route_open_set_style,
)

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)


def prepared_input() -> PreparedDialogueInput:
    return PreparedDialogueInput(
        user_id="owner-a",
        text="synthetic current request",
        recent_context=("synthetic prior context",),
    )


def _snapshot(user_id: str = "owner-a") -> UserStyleSnapshot:
    statistical = StyleFeatureProvenance(
        source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
        evidence_count=8,
        observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
        source_version="safe-v1",
        updated_at=NOW,
        confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
    )
    declared = StyleFeatureProvenance(
        source_type=StyleFeatureSourceType.EXPLICIT_OWNER_CONFIRMATION,
        evidence_count=1,
        observed_from=StyleObservedFrom.OWNER_DECLARATION,
        source_version="declaration-v1",
        updated_at=NOW,
        confidence_method=StyleConfidenceMethod.USER_DECLARATION,
    )
    return UserStyleSnapshot(
        consent=UserStyleProfileConsent(
            user_id=user_id,
            authorization_version="consent-v1",
            status=UserStyleProfileStatus.AUTHORIZED,
            updated_at=NOW,
        ),
        profile=UserStyleProfile(
            user_id=user_id,
            profile_version="profile-v1",
            authorization_version="consent-v1",
            features=(
                UserStyleFeature(
                    kind=StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
                    value=8.0,
                    confidence=0.9,
                    provenance=statistical,
                ),
                UserStyleFeature(
                    kind=StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE,
                    value="synthetic-approved-expression",
                    confidence=1.0,
                    provenance=declared,
                ),
            ),
            created_at=NOW,
            updated_at=NOW,
        ),
        evidence_count=9,
        derived=UserStyleDerivedArtifacts(
            user_id=user_id,
            profile_version="profile-v1",
            authorization_version="consent-v1",
            statistics_version="stats-v1",
            index_version="index-v1",
            cache_version="cache-v1",
            updated_at=NOW,
        ),
    )


def _scenario(
    scenario_id: str,
    *labels: PragmaticLabel,
    user_id: str = "owner-a",
    intent: StyleIntent | None = None,
    semantic_definition_id: str = "scenario-coarse-labels-only-v1",
) -> StyleScenarioPrototype:
    return StyleScenarioPrototype(
        user_id=user_id,
        scenario_id=scenario_id,
        required_labels=labels,
        auxiliary_intent=intent,
        prototype_version="public-synthetic-routing-v1",
        semantic_definition_id=semantic_definition_id,
    )


def _route(
    text: str,
    scenarios: tuple[StyleScenarioPrototype, ...],
    *,
    user_id: str = "owner-a",
    recent_context: tuple[str, ...] = ("private-context-sentinel",),
) -> OpenSetStyleRoute:
    delta = build_pragmatic_context_delta(
        user_id=user_id,
        trusted_user_id=user_id,
        text=text,
        recent_context=recent_context,
        idle_seconds=None,
        observed_at=NOW,
    )
    return route_open_set_style(
        user_id=user_id,
        trusted_user_id=user_id,
        snapshot=_snapshot(user_id),
        context_delta=delta,
        scenario_catalog=scenarios,
    )


def test_compiler_separates_prior_delta_scenarios_and_provenance() -> None:
    route = _route(
        "谢谢 current-request-sentinel",
        (
            _scenario(
                "gratitude",
                PragmaticLabel.GRATITUDE,
                intent=StyleIntent.COURTESY_THANKS,
            ),
        ),
    )

    spec = compile_personalization_spec(user_id="owner-a", route=route)

    assert [item.kind for item in spec.stable_prior] == [
        StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
        StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE,
    ]
    assert PragmaticLabel.GRATITUDE in {item.label for item in spec.context_delta}
    assert [item.scenario_id for item in spec.scenario_candidates] == ["gratitude"]
    assert spec.scenario_candidates[0].weight == 1.0
    assert spec.authorization_version == "consent-v1"
    assert spec.profile_version == "profile-v1"
    assert spec.routing_version == "public-synthetic-routing-v1"
    assert spec.source_route == route
    assert spec.retrieval_status is RetrievalStatus.NOT_REQUESTED
    assert spec.retrieved_evidence == ()
    assert spec.hard_exclusions == (
        HardExclusion.PRESERVE_AI_IDENTITY,
        HardExclusion.PRESERVE_SAFETY_AND_PRIVACY,
        HardExclusion.PRESERVE_FACTUAL_ACCURACY,
        HardExclusion.PRESERVE_CURRENT_INTENT,
        HardExclusion.DO_NOT_FABRICATE_STYLE_MATCH,
    )
    assert {item.source_type for item in spec.provenance} == {
        PersonalizationProvenanceSource.STABLE_PROFILE,
        PersonalizationProvenanceSource.PRAGMATIC_CALIBRATION,
        PersonalizationProvenanceSource.SCENARIO_ROUTE,
        PersonalizationProvenanceSource.DIALOGUE_POLICY,
    }
    assert all(0.0 <= item.confidence <= 1.0 for item in spec.provenance)
    serialized = spec.model_dump_json()
    assert "current-request-sentinel" not in serialized
    assert "private-context-sentinel" not in serialized


def test_nearby_candidates_keep_phase3_blend_without_using_intent_as_route_key() -> None:
    route = _route(
        "我今天有点难过",
        (
            _scenario(
                "sad-request",
                PragmaticLabel.SADNESS,
                PragmaticLabel.REQUEST,
                intent=StyleIntent.NEGATIVE_EMOTION_CHECK_IN,
            ),
            _scenario(
                "sad-support",
                PragmaticLabel.SADNESS,
                PragmaticLabel.SUPPORT_SEEKING,
                intent=StyleIntent.WORK_FATIGUE_CARE,
            ),
        ),
    )

    spec = compile_personalization_spec(user_id="owner-a", route=route)

    assert [item.scenario_id for item in spec.scenario_candidates] == [
        "sad-request",
        "sad-support",
    ]
    assert [item.weight for item in spec.scenario_candidates] == [0.5, 0.5]
    assert all(item.coverage == 0.5 for item in spec.scenario_candidates)
    assert spec.route_kind.value == "nearby"


def test_open_set_route_still_builds_controlled_spec_without_optional_evidence() -> None:
    route = _route(
        "current-request-sentinel",
        (_scenario("greeting", PragmaticLabel.GREETING),),
    )

    spec = compile_personalization_spec(user_id="owner-a", route=route)

    assert spec.route_kind.value == "open_set"
    assert spec.scenario_candidates == ()
    assert spec.stable_prior
    assert spec.context_delta
    assert spec.retrieved_evidence == ()
    assert spec.retrieval_status is RetrievalStatus.NOT_REQUESTED


def test_zero_signal_open_set_compiles_without_inventing_context_or_candidate() -> None:
    route = _route(
        "合成蓝色纸船",
        (_scenario("greeting", PragmaticLabel.GREETING),),
        recent_context=(),
    )
    assert route.context_delta.all_signals == ()

    spec = compile_personalization_spec(user_id="owner-a", route=route)

    assert spec.stable_prior
    assert spec.context_delta == ()
    assert spec.scenario_candidates == ()
    assert spec.source_route == route
    serialized = spec.model_dump_json()
    assert "合成蓝色纸船" not in serialized


def test_compound_route_preserves_multiple_complete_candidate_weights() -> None:
    route = _route(
        "谢谢你,晚安",
        (
            _scenario(
                "appreciation",
                PragmaticLabel.GRATITUDE,
                PragmaticLabel.AFFILIATIVE,
            ),
            _scenario(
                "farewell",
                PragmaticLabel.FAREWELL,
                PragmaticLabel.CLOSING,
            ),
        ),
    )

    spec = compile_personalization_spec(user_id="owner-a", route=route)

    assert spec.route_kind.value == "compound"
    assert len(spec.scenario_candidates) == 2
    assert all(item.coverage == 1.0 for item in spec.scenario_candidates)
    assert sum(item.weight for item in spec.scenario_candidates) == 1.0


def test_compiler_fails_closed_for_wrong_user_or_forged_route() -> None:
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))

    with pytest.raises(PersonalizationCompileError, match="personalization input is invalid"):
        compile_personalization_spec(user_id="other-user", route=route)

    forged = route.model_copy(deep=True)
    object.__setattr__(forged.candidates[0], "weight", 0.25)
    with pytest.raises(PersonalizationCompileError, match="personalization input is invalid"):
        compile_personalization_spec(user_id="owner-a", route=forged)


def test_model_adapter_consumes_spec_without_emitting_request_or_context_body() -> None:
    route = _route(
        "谢谢 current-request-sentinel",
        (_scenario("gratitude", PragmaticLabel.GRATITUDE),),
    )
    spec = compile_personalization_spec(user_id="owner-a", route=route)

    instructions = adapt_personalization_spec(user_id="owner-a", spec=spec)

    rendered = "\n".join(instructions.directives)
    assert "message_length_median" in rendered
    assert "gratitude" in rendered
    assert "scenario=gratitude" in rendered
    assert "weight=1.000000" in rendered
    assert "current-request-sentinel" not in rendered
    assert "private-context-sentinel" not in rendered
    assert "safety" in rendered
    assert "current_intent" in rendered


def test_model_adapter_revalidates_scope_and_provenance() -> None:
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))
    spec = compile_personalization_spec(user_id="owner-a", route=route)

    with pytest.raises(PersonalizationCompileError, match="personalization input is invalid"):
        adapt_personalization_spec(user_id="other-user", spec=spec)

    forged = spec.model_copy(deep=True)
    context_index = next(
        index
        for index, item in enumerate(forged.provenance)
        if item.source_type is PersonalizationProvenanceSource.PRAGMATIC_CALIBRATION
    )
    object.__setattr__(forged.provenance[context_index], "confidence", 0.1)
    with pytest.raises(ValidationError):
        PersonalizationSpec.model_validate(forged.model_dump(mode="python"))
    with pytest.raises(PersonalizationCompileError, match="personalization input is invalid"):
        adapt_personalization_spec(user_id="owner-a", spec=forged)


@pytest.mark.parametrize(
    "tamper",
    (
        "stable_value",
        "stable_confidence",
        "stable_profile_version",
        "context_score",
        "scenario_intent",
        "authorization_version",
        "routing_version",
        "route_kind",
        "stable_provenance",
    ),
)
def test_spec_recomputes_all_compiled_fields_from_source_route(tamper: str) -> None:
    route = _route(
        "谢谢",
        (
            _scenario(
                "gratitude",
                PragmaticLabel.GRATITUDE,
                intent=StyleIntent.COURTESY_THANKS,
            ),
        ),
    )
    forged = compile_personalization_spec(user_id="owner-a", route=route).model_copy(deep=True)

    if tamper == "stable_value":
        object.__setattr__(forged.stable_prior[0], "value", 99.0)
    elif tamper == "stable_confidence":
        object.__setattr__(forged.stable_prior[0], "confidence", 0.1)
    elif tamper == "stable_profile_version":
        object.__setattr__(forged.stable_prior[0], "profile_version", "fabricated-v9")
    elif tamper == "context_score":
        object.__setattr__(forged.context_delta[0], "score", 0.1)
    elif tamper == "scenario_intent":
        object.__setattr__(
            forged.scenario_candidates[0],
            "auxiliary_intent",
            StyleIntent.SLEEP_FAREWELL,
        )
    elif tamper == "authorization_version":
        object.__setattr__(forged, "authorization_version", "fabricated-v9")
    elif tamper == "routing_version":
        object.__setattr__(forged, "routing_version", "fabricated-v9")
    elif tamper == "route_kind":
        object.__setattr__(forged, "route_kind", OpenSetRouteKind.NEARBY)
    else:
        stable_index = next(
            index
            for index, item in enumerate(forged.provenance)
            if item.source_type is PersonalizationProvenanceSource.STABLE_PROFILE
        )
        object.__setattr__(forged.provenance[stable_index], "source_version", "fabricated-v9")

    with pytest.raises(ValidationError):
        PersonalizationSpec.model_validate(forged.model_dump(mode="python"))
    with pytest.raises(PersonalizationCompileError, match="personalization input is invalid"):
        adapt_personalization_spec(user_id="owner-a", spec=forged)


@pytest.mark.parametrize("tamper", ("revoked", "cross_user"))
def test_spec_rejects_invalid_source_route(tamper: str) -> None:
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))
    forged = compile_personalization_spec(user_id="owner-a", route=route).model_copy(deep=True)
    source_route = forged.source_route.model_copy(deep=True)
    if tamper == "revoked":
        object.__setattr__(
            source_route.snapshot.consent,
            "status",
            UserStyleProfileStatus.REVOKED,
        )
    else:
        object.__setattr__(source_route, "user_id", "other-user")
    object.__setattr__(forged, "source_route", source_route)

    with pytest.raises(ValidationError):
        PersonalizationSpec.model_validate(forged.model_dump(mode="python"))
    with pytest.raises(PersonalizationCompileError, match="personalization input is invalid"):
        adapt_personalization_spec(user_id="owner-a", spec=forged)


def test_spec_and_adapter_deep_validate_semantic_source_route() -> None:
    route = _route(
        "谢谢",
        (
            _scenario(
                "gratitude",
                PragmaticLabel.GRATITUDE,
                semantic_definition_id="scenario-gratitude-v1",
            ),
        ),
    )
    forged = compile_personalization_spec(user_id="owner-a", route=route).model_copy(deep=True)
    object.__setattr__(
        forged.source_route.scenario_catalog[0],
        "semantic_definition_id",
        "fabricated-semantic-v9",
    )

    with pytest.raises(ValidationError):
        PersonalizationSpec.model_validate(forged.model_dump(mode="python"))
    with pytest.raises(PersonalizationCompileError, match="personalization input is invalid"):
        adapt_personalization_spec(user_id="owner-a", spec=forged)


def test_prompt_adapts_optional_spec_and_old_call_shape_still_works() -> None:
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))
    spec = compile_personalization_spec(
        user_id="owner-a",
        route=route,
    )
    guidance = StyleReplyGuidance(status="unavailable")

    old_bundle = build_prompt(prepared_input(), guidance, DialogueMode.STYLED, None)
    new_bundle = build_prompt(
        prepared_input(),
        guidance,
        DialogueMode.STYLED,
        None,
        personalization=spec,
    )

    assert "PERSONALIZATION CONTROL" not in old_bundle.messages[0].content
    assert "[BEGIN PERSONALIZATION CONTROL]" in new_bundle.messages[0].content
    identity_position = new_bundle.messages[0].content.index("你是独立 AI 助手")
    personalization_position = new_bundle.messages[0].content.index(
        "[BEGIN PERSONALIZATION CONTROL]"
    )
    assert identity_position < personalization_position

    baseline = build_prompt(
        prepared_input(),
        guidance,
        DialogueMode.BASELINE,
        None,
        personalization=spec,
    )
    assert "PERSONALIZATION CONTROL" not in baseline.messages[0].content


def test_personalization_spec_remains_dialogue_internal() -> None:
    assert not hasattr(shared_contracts, "PersonalizationSpec")
