from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from companion.contracts import StyleReplyGuidance
from companion.dialogue.context import PreparedDialogueInput, build_prompt
from companion.dialogue.models import DialogueMode
from companion.dialogue.personalization import (
    PersonalizationCompileError,
    PersonalizationSpec,
    RetrievalStatus,
    adapt_personalization_spec,
    compile_personalization_spec,
)
from companion.style import (
    PragmaticLabel,
    StyleConfidenceMethod,
    StyleEvidenceApproval,
    StyleEvidenceCatalog,
    StyleEvidenceConfidenceMethod,
    StyleEvidenceObservedFrom,
    StyleEvidenceRecord,
    StyleEvidenceResult,
    StyleEvidenceRetrievalStatus,
    StyleEvidenceSourceType,
    StyleEvidenceVersions,
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
    retrieve_style_evidence,
    route_open_set_style,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def _route(text: str = "谢谢"):
    provenance = StyleFeatureProvenance(
        source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
        evidence_count=8,
        observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
        source_version="safe-v1",
        updated_at=NOW,
        confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
    )
    snapshot = UserStyleSnapshot(
        consent=UserStyleProfileConsent(
            user_id="owner-a",
            authorization_version="consent-v1",
            status=UserStyleProfileStatus.AUTHORIZED,
            updated_at=NOW,
        ),
        profile=UserStyleProfile(
            user_id="owner-a",
            profile_version="profile-v1",
            authorization_version="consent-v1",
            features=(
                UserStyleFeature(
                    kind=StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
                    value=8.0,
                    confidence=0.9,
                    provenance=provenance,
                ),
            ),
            created_at=NOW,
            updated_at=NOW,
        ),
        evidence_count=8,
        derived=UserStyleDerivedArtifacts(
            user_id="owner-a",
            profile_version="profile-v1",
            authorization_version="consent-v1",
            statistics_version="stats-v1",
            index_version="profile-index-v1",
            cache_version="profile-cache-v1",
            updated_at=NOW,
        ),
    )
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text=text,
        recent_context=(),
        idle_seconds=None,
        observed_at=NOW,
    )
    return route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=snapshot,
        context_delta=delta,
        scenario_catalog=(
            StyleScenarioPrototype(
                user_id="owner-a",
                scenario_id="gratitude",
                required_labels=(PragmaticLabel.GRATITUDE,),
                prototype_version="public-synthetic-routing-v1",
            ),
        ),
    )


def _catalog(confidence: float = 0.95) -> StyleEvidenceCatalog:
    return StyleEvidenceCatalog(
        user_id="owner-a",
        authorization_version="consent-v1",
        profile_version="profile-v1",
        versions=StyleEvidenceVersions(
            evidence_version="evidence-v1",
            index_version="index-v1",
            cache_version="cache-v1",
        ),
        records=(
            StyleEvidenceRecord(
                user_id="owner-a",
                evidence_ref="ev-1",
                scenario_id="gratitude",
                content="synthetic [approved]\nexpression",
                confidence=confidence,
                approval=StyleEvidenceApproval.APPROVED_FOR_STYLE,
                source_type=StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL,
                observed_from=StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY,
                confidence_method=StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL,
                source_version="approval-v1",
                approved_at=NOW,
            ),
        ),
        updated_at=NOW,
    )


def _retrieval(confidence: float = 0.95):
    return retrieve_style_evidence(
        user_id="owner-a",
        trusted_user_id="owner-a",
        route=_route(),
        catalog=_catalog(confidence),
    )


def test_compiler_adds_only_deeply_validated_retrieved_evidence() -> None:
    route = _route()
    retrieval = retrieve_style_evidence(
        user_id="owner-a",
        trusted_user_id="owner-a",
        route=route,
        catalog=_catalog(),
    )

    spec = compile_personalization_spec(
        user_id="owner-a",
        route=route,
        retrieval_result=retrieval,
    )

    assert spec.retrieval_status is RetrievalStatus.AVAILABLE
    assert [item.evidence_ref for item in spec.retrieved_evidence] == ["ev-1"]
    assert spec.retrieved_evidence[0].content == "synthetic [approved]\nexpression"
    assert spec.stable_prior
    assert spec.context_delta
    assert spec.source_retrieval == retrieval


@pytest.mark.parametrize("tamper", ("content", "confidence", "user", "status"))
def test_spec_and_adapter_reject_forged_retrieval_result(tamper: str) -> None:
    route = _route()
    retrieval = _retrieval().model_copy(deep=True)
    if tamper == "content":
        object.__setattr__(retrieval.matches[0], "content", "fabricated")
    elif tamper == "confidence":
        object.__setattr__(retrieval.matches[0], "confidence", 0.1)
    elif tamper == "user":
        object.__setattr__(retrieval, "user_id", "owner-b")
    else:
        object.__setattr__(retrieval, "status", StyleEvidenceRetrievalStatus.NO_MATCH)

    with pytest.raises(PersonalizationCompileError):
        compile_personalization_spec(
            user_id="owner-a",
            route=route,
            retrieval_result=retrieval,
        )

    valid = compile_personalization_spec(
        user_id="owner-a",
        route=route,
        retrieval_result=_retrieval(),
    ).model_copy(deep=True)
    object.__setattr__(valid, "source_retrieval", retrieval)
    with pytest.raises(ValidationError):
        PersonalizationSpec.model_validate(valid.model_dump(mode="python"))
    with pytest.raises(PersonalizationCompileError):
        adapt_personalization_spec(user_id="owner-a", spec=valid)


def test_no_match_low_confidence_corrupt_and_open_set_keep_prior_and_delta() -> None:
    route = _route()
    corrupt = StyleEvidenceResult(
        user_id="owner-a",
        retrieval_version="style-evidence-retrieval-v1",
        route_kind=route.route_kind,
        source_route=route,
        status=StyleEvidenceRetrievalStatus.CORRUPT,
    )
    open_route = _route("synthetic blue paper boat")
    open_result = retrieve_style_evidence(
        user_id="owner-a",
        trusted_user_id="owner-a",
        route=open_route,
        catalog=None,
    )
    cases = (
        (
            retrieve_style_evidence(
                user_id="owner-a",
                trusted_user_id="owner-a",
                route=route,
                catalog=StyleEvidenceCatalog(
                    **{
                        **_catalog().model_dump(mode="python"),
                        "records": (),
                    }
                ),
            ),
            RetrievalStatus.NO_MATCH,
        ),
        (
            retrieve_style_evidence(
                user_id="owner-a",
                trusted_user_id="owner-a",
                route=route,
                catalog=_catalog(0.4),
            ),
            RetrievalStatus.LOW_CONFIDENCE,
        ),
        (corrupt, RetrievalStatus.CORRUPT),
        (open_result, RetrievalStatus.NOT_REQUESTED),
    )
    for result, status in cases:
        source_route = result.source_route
        spec = compile_personalization_spec(
            user_id="owner-a",
            route=source_route,
            retrieval_result=result,
        )
        assert spec.retrieval_status is status
        assert spec.retrieved_evidence == ()
        assert spec.stable_prior
        assert len(spec.context_delta) == len(source_route.context_delta.all_signals)


def test_adapter_escapes_bounded_evidence_and_baseline_does_not_render_it() -> None:
    route = _route()
    catalog = _catalog().model_copy(
        update={
            "records": (
                *_catalog().records,
                StyleEvidenceRecord(
                    user_id="owner-a",
                    evidence_ref="ev-unmatched",
                    scenario_id="farewell",
                    content="synthetic unmatched expression sentinel",
                    confidence=0.95,
                    approval=StyleEvidenceApproval.APPROVED_FOR_STYLE,
                    source_type=StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL,
                    observed_from=StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY,
                    confidence_method=StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL,
                    source_version="approval-v1",
                    approved_at=NOW,
                ),
            )
        }
    )
    retrieval = retrieve_style_evidence(
        user_id="owner-a",
        trusted_user_id="owner-a",
        route=route,
        catalog=StyleEvidenceCatalog.model_validate(catalog.model_dump(mode="python")),
    )
    spec = compile_personalization_spec(
        user_id="owner-a",
        route=route,
        retrieval_result=retrieval,
    )
    rendered = "\n".join(adapt_personalization_spec(user_id="owner-a", spec=spec).directives)
    result_json = retrieval.model_dump_json()
    spec_json = spec.model_dump_json()
    assert "synthetic \\[approved\\]\\nexpression" in rendered
    assert "retrieved_evidence" in rendered
    assert "synthetic unmatched expression sentinel" not in result_json
    assert "synthetic unmatched expression sentinel" not in spec_json
    assert "synthetic unmatched expression sentinel" not in rendered

    prepared = PreparedDialogueInput(user_id="owner-a", text="synthetic request", recent_context=())
    guidance = StyleReplyGuidance(status="unavailable")
    styled = build_prompt(
        prepared,
        guidance,
        DialogueMode.STYLED,
        None,
        personalization=spec,
    )
    baseline = build_prompt(
        prepared,
        guidance,
        DialogueMode.BASELINE,
        None,
        personalization=spec,
    )
    assert "synthetic \\[approved\\]\\nexpression" in styled.messages[0].content
    assert "synthetic [approved]" not in baseline.messages[0].content
