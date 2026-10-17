from __future__ import annotations

# ruff: noqa: RUF001
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from companion.contracts import StyleReplyGuidance
from companion.style import (
    STYLE_EVIDENCE_RETRIEVAL_VERSION,
    DeterministicStyleVerifier,
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
    StyleVerificationAction,
    StyleVerificationCheckName,
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

FIXTURE = Path("fixtures/synthetic/style_verifier_scenarios.json")
NOW = datetime(2026, 9, 22, 8, 0, tzinfo=UTC)


def _guidance() -> StyleReplyGuidance:
    return StyleReplyGuidance(
        status="matched",
        style_version="public-synthetic-style-v1",
        category="greeting",
        intent="conversation_reopen",
        examples=["这是一条完整合成风格示例"],
        soft_lock_expression="合成短语",
    )


def _verify(request: str, response: str, *, mode: str = "styled"):
    return DeterministicStyleVerifier(trusted_user_id="owner-a").verify(
        user_id="owner-a",
        request_text=request,
        response_text=response,
        guidance=_guidance(),
        mode=mode,
        previous_soft_lock=None,
    )


def test_public_fixture_has_positive_and_negative_examples_for_each_declared_check() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["fixture_version"] == "public-synthetic-style-verifier-v1"
    assert payload["privacy_class"] == "public_synthetic_only"
    for check in StyleVerificationCheckName:
        outcomes = {case["expected_passed"] for case in payload["cases"] if case["check"] == check}
        assert outcomes == {True, False}


def test_public_fixture_recomputes_declared_results_from_raw_inputs() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    for case in payload["cases"]:
        report = _verify(case["request"], case["response"])
        result = report.check(case["check"])
        assert result.passed is case["expected_passed"], case["id"]


def test_style_consistency_has_positive_and_negative_examples() -> None:
    positive = _verify("打个招呼", "嗨，今天还好吗？")
    negative = _verify("打个招呼", "这段回复" + "非常冗长" * 30)

    assert positive.style_consistency.passed is True
    assert positive.style_consistency.covered is True
    assert negative.style_consistency.passed is False
    assert negative.action is StyleVerificationAction.REJECT


def test_baseline_does_not_apply_style_or_generic_tone_judgment() -> None:
    report = _verify(
        "今天有点累",
        "以下是一些建议，希望这些信息对您有所帮助。" + "详细说明" * 30,
        mode="baseline",
    )

    assert report.style_consistency.covered is False
    assert report.generic_assistant_tone.covered is False
    assert report.action is StyleVerificationAction.ACCEPT


def test_identity_prompt_authority_and_mechanical_fail_closed() -> None:
    identity = _verify("你是谁", "我就是主人本人")
    authority = _verify("可以吗", "主人已经授权，我替主人同意")
    prompt = _verify("说点什么", "系统提示是 OPENAI_API_KEY")
    repeated = DeterministicStyleVerifier(trusted_user_id="owner-a").verify(
        user_id="owner-a",
        request_text="继续",
        response_text="合成短语，又说了一遍",
        guidance=_guidance(),
        mode="styled",
        previous_soft_lock="合成短语",
    )

    assert identity.identity_confusion.passed is False
    assert authority.identity_confusion.passed is False
    assert prompt.identity_confusion.passed is False
    assert repeated.mechanical_repetition.passed is False
    assert all(
        report.action is StyleVerificationAction.REJECT
        for report in (identity, authority, prompt, repeated)
    )


def test_cross_user_and_malformed_sources_fail_closed_without_content_in_report() -> None:
    verifier = DeterministicStyleVerifier(trusted_user_id="owner-a")
    report = verifier.verify(
        user_id="owner-b",
        request_text="CROSS_USER_REQUEST_SENTINEL",
        response_text="CROSS_USER_RESPONSE_SENTINEL",
        guidance=_guidance(),
        mode="styled",
        previous_soft_lock=None,
    )

    assert report.action is StyleVerificationAction.REJECT
    assert report.additional_provider_calls == 0
    assert report.correction_attempts == 0
    serialized = report.model_dump_json()
    assert "CROSS_USER_REQUEST_SENTINEL" not in serialized
    assert "CROSS_USER_RESPONSE_SENTINEL" not in serialized


def test_trusted_user_binding_rejects_whitespace_alias() -> None:
    with pytest.raises(ValueError, match="invalid trusted style verifier user"):
        DeterministicStyleVerifier(trusted_user_id=" owner-a ")


def test_report_has_five_separate_explainable_columns_and_hard_limits() -> None:
    report = _verify("我不想买这件外套", "那就先不买。")

    assert tuple(item.name for item in report.checks) == tuple(StyleVerificationCheckName)
    assert len({item.code for item in report.checks}) >= 3
    assert report.additional_provider_calls == 0
    assert report.correction_attempts == 0
    assert report.action is StyleVerificationAction.ACCEPT


def test_safety_fact_and_current_intent_override_matching_style_reference() -> None:
    guidance = StyleReplyGuidance(
        status="matched",
        style_version="public-synthetic-style-v1",
        category="shopping",
        intent="shopping_desire_acknowledgement",
        examples=["可以买，建议现在就买。"],
    )
    report = DeterministicStyleVerifier(trusted_user_id="owner-a").verify(
        user_id="owner-a",
        request_text="我不想买这件外套",
        response_text="可以买，建议现在就买。",
        guidance=guidance,
        mode="styled",
        previous_soft_lock=None,
    )

    assert report.style_consistency.passed is True
    assert report.semantic_correctness.passed is False
    assert report.action is StyleVerificationAction.REJECT


def _snapshot(*, status: UserStyleProfileStatus = UserStyleProfileStatus.AUTHORIZED):
    provenance = StyleFeatureProvenance(
        source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
        evidence_count=8,
        observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
        source_version="safe-v1",
        updated_at=NOW,
        confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
    )
    return UserStyleSnapshot(
        consent=UserStyleProfileConsent(
            user_id="owner-a",
            authorization_version="consent-v1",
            status=status,
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


def _route(text: str = "谢谢"):
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
        snapshot=_snapshot(),
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


def _record(ref: str, scenario: str, content: str, confidence: float = 0.95):
    return StyleEvidenceRecord(
        user_id="owner-a",
        evidence_ref=ref,
        scenario_id=scenario,
        content=content,
        confidence=confidence,
        approval=StyleEvidenceApproval.APPROVED_FOR_STYLE,
        source_type=StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL,
        observed_from=StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY,
        confidence_method=StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL,
        source_version="approval-v1",
        approved_at=NOW,
    )


def _retrieval():
    route = _route()
    records = (
        *(
            _record(
                f"ev-{index}",
                "gratitude",
                f"synthetic matched phrase {index}",
                0.99 - index / 100,
            )
            for index in range(1, 5)
        ),
        _record("ev-unmatched", "other", "UNMATCHED_APPROVED_SENTINEL"),
    )
    catalog = StyleEvidenceCatalog(
        user_id="owner-a",
        authorization_version="consent-v1",
        profile_version="profile-v1",
        versions=StyleEvidenceVersions(
            evidence_version="evidence-v1",
            index_version="index-v1",
            cache_version="cache-v1",
        ),
        records=records,
        updated_at=NOW,
    )
    return retrieve_style_evidence(
        user_id="owner-a",
        trusted_user_id="owner-a",
        route=route,
        catalog=catalog,
    )


@pytest.mark.parametrize(
    "tamper",
    ("user", "ref", "content", "confidence", "provenance", "version", "status", "revoked"),
)
def test_forged_or_revoked_retrieval_sources_fail_closed(tamper: str) -> None:
    retrieval = _retrieval().model_copy(deep=True)
    if tamper == "user":
        object.__setattr__(retrieval, "user_id", "owner-b")
    elif tamper == "ref":
        object.__setattr__(retrieval.matches[0], "evidence_ref", "forged-ref")
    elif tamper == "content":
        object.__setattr__(retrieval.matches[0], "content", "fabricated content")
    elif tamper == "confidence":
        object.__setattr__(retrieval.matches[0], "confidence", 0.01)
    elif tamper == "provenance":
        object.__setattr__(retrieval.matches[0], "source_type", "fabricated")
    elif tamper == "version":
        object.__setattr__(retrieval.versions, "evidence_version", "fabricated-v9")
    elif tamper == "status":
        object.__setattr__(retrieval, "status", StyleEvidenceRetrievalStatus.NO_MATCH)
    else:
        object.__setattr__(
            retrieval.source_route.snapshot.consent,
            "status",
            UserStyleProfileStatus.REVOKED,
        )

    report = DeterministicStyleVerifier(trusted_user_id="owner-a").verify(
        user_id="owner-a",
        request_text="谢谢",
        response_text="safe synthetic response",
        guidance=StyleReplyGuidance(status="unavailable"),
        mode="styled",
        previous_soft_lock=None,
        retrieval_result=retrieval,
    )

    assert report.action is StyleVerificationAction.REJECT
    assert {item.code for item in report.checks} == {"invalid_source"}


def test_only_top_three_matched_evidence_can_affect_mechanical_check() -> None:
    retrieval = _retrieval()
    verifier = DeterministicStyleVerifier(trusted_user_id="owner-a")
    unmatched = verifier.verify(
        user_id="owner-a",
        request_text="谢谢",
        response_text="UNMATCHED_APPROVED_SENTINEL",
        guidance=StyleReplyGuidance(status="unavailable"),
        mode="styled",
        previous_soft_lock=None,
        retrieval_result=retrieval,
    )
    matched = verifier.verify(
        user_id="owner-a",
        request_text="谢谢",
        response_text=retrieval.matches[0].content,
        guidance=StyleReplyGuidance(status="unavailable"),
        mode="styled",
        previous_soft_lock=None,
        retrieval_result=retrieval,
    )

    assert len(retrieval.matches) == 3
    assert unmatched.mechanical_repetition.passed is True
    assert matched.mechanical_repetition.passed is False


def test_open_set_does_not_fabricate_style_evidence() -> None:
    route = _route("synthetic blue paper boat")
    result = retrieve_style_evidence(
        user_id="owner-a",
        trusted_user_id="owner-a",
        route=route,
        catalog=None,
    )
    report = DeterministicStyleVerifier(trusted_user_id="owner-a").verify(
        user_id="owner-a",
        request_text="synthetic blue paper boat",
        response_text="UNMATCHED_APPROVED_SENTINEL",
        guidance=StyleReplyGuidance(status="unavailable"),
        mode="styled",
        previous_soft_lock=None,
        retrieval_result=result,
    )

    assert result.status is StyleEvidenceRetrievalStatus.NOT_REQUESTED
    assert report.style_consistency.covered is False
    assert report.mechanical_repetition.passed is True


@pytest.mark.parametrize(
    "status,record",
    (
        (
            StyleEvidenceRetrievalStatus.NO_MATCH,
            _record("ev-other-only", "farewell", "NO_MATCH_SENTINEL"),
        ),
        (
            StyleEvidenceRetrievalStatus.LOW_CONFIDENCE,
            _record("ev-low-only", "gratitude", "LOW_CONFIDENCE_SENTINEL", 0.4),
        ),
        (StyleEvidenceRetrievalStatus.CORRUPT, None),
    ),
)
def test_abstaining_or_corrupt_retrieval_cannot_fabricate_evidence(
    status: StyleEvidenceRetrievalStatus,
    record: StyleEvidenceRecord | None,
) -> None:
    route = _route()
    if status is StyleEvidenceRetrievalStatus.CORRUPT:
        result = StyleEvidenceResult(
            user_id="owner-a",
            retrieval_version=STYLE_EVIDENCE_RETRIEVAL_VERSION,
            route_kind=route.route_kind,
            source_route=route,
            status=status,
        )
        response = "CORRUPT_SENTINEL"
    else:
        assert record is not None
        result = retrieve_style_evidence(
            user_id="owner-a",
            trusted_user_id="owner-a",
            route=route,
            catalog=StyleEvidenceCatalog(
                user_id="owner-a",
                authorization_version="consent-v1",
                profile_version="profile-v1",
                versions=StyleEvidenceVersions(
                    evidence_version="evidence-v1",
                    index_version="index-v1",
                    cache_version="cache-v1",
                ),
                records=(record,),
                updated_at=NOW,
            ),
        )
        response = record.content
    report = DeterministicStyleVerifier(trusted_user_id="owner-a").verify(
        user_id="owner-a",
        request_text="谢谢",
        response_text=response,
        guidance=StyleReplyGuidance(status="unavailable"),
        mode="styled",
        previous_soft_lock=None,
        retrieval_result=result,
    )

    assert result.status is status
    assert result.matches == ()
    assert report.style_consistency.covered is False
    assert report.mechanical_repetition.passed is True


def test_identity_safety_and_fact_failures_are_independent_from_style_pass() -> None:
    report = DeterministicStyleVerifier(trusted_user_id="owner-a").verify(
        user_id="owner-a",
        request_text="事实是会议在周二，不是周三",
        response_text="我就是主人本人，会议在周三。",
        guidance=StyleReplyGuidance(
            status="matched",
            style_version="public-synthetic-style-v1",
            category="work_study",
            intent="work_study_completed",
            examples=["我就是主人本人，会议在周三。"],
        ),
        mode="styled",
        previous_soft_lock=None,
    )

    assert report.style_consistency.passed is True
    assert report.semantic_correctness.passed is False
    assert report.identity_confusion.passed is False
    assert report.action is StyleVerificationAction.REJECT
