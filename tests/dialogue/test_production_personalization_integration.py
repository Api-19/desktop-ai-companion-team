from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from dialogue_support import (
    FakeProvider,
    FakeStyleProvider,
    sample_request,
    synthetic_dialogue_config,
)
from pydantic import SecretStr

from companion.contracts import DialogueRequest, StyleReplyGuidance
from companion.dialogue.ab import ABStatus, build_real_orchestrator, load_framework20_scenarios
from companion.dialogue.models import DialogueMode, DialogueSecrets, ProductionStyleSources
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.style_personalization import ProductionStylePersonalization
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime.settings import Settings
from companion.style import (
    DeterministicStyleVerifier,
    SingleTrustedUserStyleEvidenceStore,
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
    UserStyleDerivedArtifacts,
    UserStyleFeature,
    UserStyleProfile,
    UserStyleProfileConsent,
    UserStyleProfileStatus,
    UserStyleSnapshot,
    UserStyleSnapshotReadResult,
    UserStyleSnapshotReadStatus,
)

NOW = datetime(2026, 9, 23, 9, 0, tzinfo=UTC)
OWNER = "synthetic-user"


def _snapshot() -> UserStyleSnapshot:
    return UserStyleSnapshot(
        consent=UserStyleProfileConsent(
            user_id=OWNER,
            authorization_version="consent-v1",
            status=UserStyleProfileStatus.AUTHORIZED,
            updated_at=NOW,
        ),
        profile=UserStyleProfile(
            user_id=OWNER,
            profile_version="profile-v1",
            authorization_version="consent-v1",
            features=(
                UserStyleFeature(
                    kind=StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
                    value=8.0,
                    confidence=0.9,
                    provenance=StyleFeatureProvenance(
                        source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
                        evidence_count=8,
                        observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
                        source_version="safe-v1",
                        updated_at=NOW,
                        confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
                    ),
                ),
            ),
            created_at=NOW,
            updated_at=NOW,
        ),
        evidence_count=8,
        derived=UserStyleDerivedArtifacts(
            user_id=OWNER,
            profile_version="profile-v1",
            authorization_version="consent-v1",
            statistics_version="stats-v1",
            index_version="index-v1",
            cache_version="cache-v1",
            updated_at=NOW,
        ),
    )


class RecordingProfile:
    def __init__(self, status: UserStyleSnapshotReadStatus) -> None:
        self.status = status
        self.calls = 0

    def read_snapshot_state(self, user_id: str) -> UserStyleSnapshotReadResult:
        assert user_id == OWNER
        self.calls += 1
        return UserStyleSnapshotReadResult(
            status=self.status,
            snapshot=_snapshot() if self.status is UserStyleSnapshotReadStatus.AUTHORIZED else None,
        )


class RecordingEvidence:
    def __init__(self, status: StyleEvidenceRetrievalStatus | None = None) -> None:
        self.calls = 0
        self.catalog_size = 0
        self.status = status

    def retrieve_evidence(self, *, user_id: str, route: object) -> object:
        from companion.style import retrieve_style_evidence

        self.calls += 1
        self.catalog_size = len(route.scenario_catalog)
        if self.status is not None:
            if self.status is StyleEvidenceRetrievalStatus.CORRUPT:
                return StyleEvidenceResult(
                    user_id=user_id,
                    retrieval_version="style-evidence-retrieval-v1",
                    route_kind=route.route_kind,
                    source_route=route,
                    status=self.status,
                )
            records = ()
            if self.status is StyleEvidenceRetrievalStatus.LOW_CONFIDENCE:
                records = (
                    StyleEvidenceRecord(
                        user_id=user_id,
                        evidence_ref="evidence-one",
                        scenario_id=route.candidates[0].scenario_id,
                        content="synthetic approved expression",
                        confidence=0.1,
                        approval=StyleEvidenceApproval.APPROVED_FOR_STYLE,
                        source_type=StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL,
                        observed_from=StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY,
                        confidence_method=StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL,
                        source_version="approval-v1",
                        approved_at=NOW,
                    ),
                )
            return retrieve_style_evidence(
                user_id=user_id,
                trusted_user_id=OWNER,
                route=route,
                catalog=StyleEvidenceCatalog(
                    user_id=user_id,
                    authorization_version="consent-v1",
                    profile_version="profile-v1",
                    versions=StyleEvidenceVersions(
                        evidence_version="evidence-v1",
                        index_version="index-v1",
                        cache_version="cache-v1",
                    ),
                    records=records,
                    updated_at=NOW,
                ),
            )
        return retrieve_style_evidence(
            user_id=user_id, trusted_user_id=OWNER, route=route, catalog=None
        )


def _orchestrator(
    root: Path,
    profile: RecordingProfile,
    evidence: RecordingEvidence,
    *,
    guidance: StyleReplyGuidance | None = None,
):
    config = synthetic_dialogue_config(root)
    provider = FakeProvider("synthetic safe answer")
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(guidance),
        provider=provider,
        ledger=MonthlyUsageLedger(root / "runtime", config.budget),
        clock=lambda: NOW,
        style_personalization=ProductionStylePersonalization(
            trusted_user_id=OWNER,
            profile_gateway=profile,
            evidence_gateway=evidence,
        ),
        style_verifier=DeterministicStyleVerifier(trusted_user_id=OWNER),
        test_results_dir=root / "results",
    )
    return orchestrator, provider


def test_framework20_preflight_compiles_all_scenarios_without_provider_or_answers(
    tmp_path: Path,
) -> None:
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence(StyleEvidenceRetrievalStatus.NO_MATCH)
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    scenarios = load_framework20_scenarios(Path("config/dialogue-ab-full-scenarios.json"))

    orchestrator.preflight_framework20(
        scenarios, trusted_user_id=OWNER, results_dir=tmp_path / "results"
    )

    assert profile.calls == evidence.calls == 20
    assert provider.call_count == 0
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("failure", ("owner", "results", "consent", "corrupt"))
def test_framework20_preflight_fails_closed_without_provider(tmp_path: Path, failure: str) -> None:
    profile = RecordingProfile(
        UserStyleSnapshotReadStatus.UNAVAILABLE
        if failure == "consent"
        else UserStyleSnapshotReadStatus.AUTHORIZED
    )
    evidence = RecordingEvidence(
        StyleEvidenceRetrievalStatus.CORRUPT if failure == "corrupt" else None
    )
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    scenarios = load_framework20_scenarios(Path("config/dialogue-ab-full-scenarios.json"))
    with pytest.raises(ValueError, match=r"^framework preflight unavailable$"):
        orchestrator.preflight_framework20(
            scenarios,
            trusted_user_id="synthetic-other" if failure == "owner" else OWNER,
            results_dir=tmp_path / ("other" if failure == "results" else "results"),
        )
    assert provider.call_count == 0
    assert not (tmp_path / "results").exists()


def test_styled_uses_formal_catalog_and_open_set_retrieval_without_evidence_read(tmp_path: Path):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)

    run = orchestrator.reply(sample_request(), trusted_user_id=OWNER)

    assert run.response.blocked_reason is None
    # One compile read plus one pre-send recheck; open-set recheck skips Phase 5.
    assert profile.calls == 2
    assert evidence.calls == 1
    assert provider.call_count == 1
    assert "route_kind=open_set" in provider.system_text
    assert evidence.catalog_size == 11


def test_baseline_and_default_constructor_do_not_read_new_style(tmp_path: Path):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    baseline = orchestrator.reply(
        sample_request(), mode=DialogueMode.BASELINE, trusted_user_id=OWNER
    )
    assert baseline.response.blocked_reason is None
    assert profile.calls == evidence.calls == 0
    assert "personalization" not in provider.system_text.lower()

    config = synthetic_dialogue_config(tmp_path)
    legacy = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "legacy-runtime", config.budget),
        clock=lambda: NOW,
    )
    assert legacy.reply(sample_request(), trusted_user_id=OWNER).response.blocked_reason is None
    assert profile.calls == evidence.calls == 0


def test_unavailable_snapshot_uses_plain_reply_without_legacy_matched_style(tmp_path: Path):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.UNAVAILABLE)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    run = orchestrator.reply(sample_request(), trusted_user_id=OWNER)
    assert run.response.blocked_reason is None
    assert run.trace.guidance_status.value == "unavailable"
    assert provider.call_count == 1
    assert evidence.calls == 0
    assert "synthetic-style-example" not in provider.system_text
    assert "personalization" not in provider.system_text.lower()


def test_corrupt_snapshot_closes_before_provider_and_has_no_content_in_trace(tmp_path: Path):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.CORRUPT)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    run = orchestrator.reply(sample_request(), trusted_user_id=OWNER)
    assert run.response.blocked_reason == "configuration_unavailable"
    assert run.trace.provider_call_count == 0
    assert evidence.calls == provider.call_count == 0
    assert "synthetic current request" not in run.model_dump_json()
    assert "synthetic prior context" not in repr(orchestrator._sessions)


@pytest.mark.parametrize("gateway", ("profile", "evidence"))
def test_gateway_exception_is_local_and_does_not_expose_private_text(tmp_path: Path, gateway: str):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()

    def fail(*args, **kwargs):
        raise RuntimeError("private-gateway-sentinel: synthetic current request")

    if gateway == "profile":
        profile.read_snapshot_state = fail
    else:
        evidence.retrieve_evidence = fail
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    run = orchestrator.reply(sample_request(), trusted_user_id=OWNER)
    assert run.response.blocked_reason == "configuration_unavailable"
    assert provider.call_count == 0
    assert "private-gateway-sentinel" not in run.model_dump_json()
    assert "synthetic current request" not in repr(orchestrator._sessions)


def test_untrusted_user_is_rejected_before_new_style_or_provider(tmp_path: Path):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    request = DialogueRequest(user_id="other-user", text="synthetic text", recent_context=[])
    run = orchestrator.reply(request, trusted_user_id=OWNER)
    assert run.response.blocked_reason == "untrusted_user"
    assert profile.calls == evidence.calls == provider.call_count == 0


@pytest.mark.parametrize(
    "stage",
    (
        "build_pragmatic_context_delta",
        "build_formal_style_scenario_catalog",
        "route_open_set_style",
        "compile_personalization_spec",
    ),
)
def test_style_stage_exception_closes_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
):
    def fail(*args: object, **kwargs: object):
        raise RuntimeError("private-stage-sentinel")

    monkeypatch.setattr("companion.dialogue.style_personalization." + stage, fail)
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    run = orchestrator.reply(sample_request(), trusted_user_id=OWNER)
    assert run.response.blocked_reason == "configuration_unavailable"
    assert provider.call_count == 0
    assert "private-stage-sentinel" not in run.model_dump_json()


@pytest.mark.parametrize(
    "status",
    (
        StyleEvidenceRetrievalStatus.NO_MATCH,
        StyleEvidenceRetrievalStatus.LOW_CONFIDENCE,
    ),
)
def test_retrieval_abstain_status_reaches_styled_spec(
    tmp_path: Path, status: StyleEvidenceRetrievalStatus
):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence(status)
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    run = orchestrator.reply(
        DialogueRequest(user_id=OWNER, text="谢谢", recent_context=[]), trusted_user_id=OWNER
    )
    assert run.trace.provider_call_count == 1
    assert f"retrieval_status={status.value}" in provider.system_text
    assert evidence.catalog_size == 11


def test_corrupt_retrieval_refuses_styled_request_before_reservation(tmp_path: Path):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence(StyleEvidenceRetrievalStatus.CORRUPT)
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    request = DialogueRequest(user_id=OWNER, text="谢谢", recent_context=[])

    run = orchestrator.reply(request, trusted_user_id=OWNER)
    pair = orchestrator.reply_ab("corrupt-pair", request, trusted_user_id=OWNER)

    assert run.response.blocked_reason == "configuration_unavailable"
    assert run.trace.provider_call_count == 0
    assert pair.status is ABStatus.FAILED
    assert pair.request_attempt_count == 0
    assert provider.call_count == 0
    assert not (tmp_path / "runtime").exists()


def test_ab_baseline_prompt_has_no_new_controls_and_styled_uses_one_spec(tmp_path: Path):
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    pair = orchestrator.reply_ab("synthetic-scenario", sample_request(), trusted_user_id=OWNER)
    assert pair.status is ABStatus.COMPLETE
    assert (profile.calls, evidence.calls) == (2, 1)
    assert provider.call_count == 2
    systems = [request[0].content for request in provider.requests]
    assert sum("route_kind=open_set" in text for text in systems) == 1
    assert sum("synthetic-style-example" in text for text in systems) == 1


@pytest.mark.parametrize(
    "guidance",
    (
        StyleReplyGuidance(status="no_match", style_version="synthetic-legacy-v1"),
        StyleReplyGuidance(status="unavailable"),
    ),
)
def test_ab_new_spec_pairs_even_when_legacy_guidance_does_not_match(
    tmp_path: Path, guidance: StyleReplyGuidance
) -> None:
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence, guidance=guidance)

    pair = orchestrator.reply_ab("new-route", sample_request(), trusted_user_id=OWNER)

    assert pair.status is ABStatus.COMPLETE
    assert len(pair.answers) == 2
    assert provider.call_count == 2
    assert (profile.calls, evidence.calls) == (2, 1)
    systems = [request[0].content for request in provider.requests]
    assert sum("route_kind=open_set" in text for text in systems) == 1
    assert sum("route_kind=open_set" not in text for text in systems) == 1
    assert all("synthetic-style-example" not in text for text in systems)
    assert orchestrator._sessions == {}


@pytest.mark.parametrize(
    ("guidance", "expected"),
    (
        (StyleReplyGuidance(status="no_match", style_version="legacy-v1"), ABStatus.NOT_MATCHED),
        (StyleReplyGuidance(status="unavailable"), ABStatus.FAILED),
    ),
)
def test_ab_without_opt_in_keeps_legacy_nonmatch_gate(
    tmp_path: Path, guidance: StyleReplyGuidance, expected: ABStatus
) -> None:
    config = synthetic_dialogue_config(tmp_path)
    provider = FakeProvider()
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(guidance),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: NOW,
    )
    pair = orchestrator.reply_ab("legacy-gate", sample_request(), trusted_user_id=OWNER)
    assert pair.status is expected
    assert pair.answers == pair.answer_key == {}
    assert provider.call_count == 0


@pytest.mark.parametrize(
    "status",
    (UserStyleSnapshotReadStatus.UNAVAILABLE, UserStyleSnapshotReadStatus.CORRUPT),
)
def test_ab_phase1_non_authorized_state_never_uses_old_matched_guidance(
    tmp_path: Path, status: UserStyleSnapshotReadStatus
) -> None:
    profile = RecordingProfile(status)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    pair = orchestrator.reply_ab("phase1-gate", sample_request(), trusted_user_id=OWNER)
    assert pair.status is ABStatus.FAILED
    assert profile.calls == 1
    assert evidence.calls == provider.call_count == 0


def test_ab_pipeline_exception_or_forged_spec_never_reaches_provider(tmp_path: Path) -> None:
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(
        tmp_path,
        profile,
        evidence,
        guidance=StyleReplyGuidance(status="no_match", style_version="legacy-v1"),
    )

    def fail(**kwargs):
        raise RuntimeError("synthetic private pipeline failure")

    orchestrator._style_personalization.compile = fail
    failed = orchestrator.reply_ab("pipeline-fail", sample_request(), trusted_user_id=OWNER)
    assert failed.status is ABStatus.FAILED
    assert provider.call_count == 0

    orchestrator._style_personalization.compile = lambda **kwargs: object()
    forged = orchestrator.reply_ab("forged-spec", sample_request(), trusted_user_id=OWNER)
    assert forged.status is ABStatus.FAILED
    assert provider.call_count == 0


def test_ab_new_spec_reserves_two_calls_and_keeps_guard_before_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import companion.dialogue.orchestrator as orchestrator_module

    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(
        tmp_path,
        profile,
        evidence,
        guidance=StyleReplyGuidance(status="no_match", style_version="legacy-v1"),
    )
    events = []
    ledger = orchestrator._ledger
    original_reserve = ledger.reserve
    original_settle = ledger.settle
    original_complete = provider.complete
    original_guard = orchestrator_module.validate_dialogue_output
    original_verify = orchestrator._style_verifier.verify

    def reserve(*, count, now):
        events.append(("reserve", count))
        return original_reserve(count=count, now=now)

    def settle(*args, **kwargs):
        events.append(("settle",))
        return original_settle(*args, **kwargs)

    def complete(messages):
        events.append(("provider",))
        return original_complete(messages)

    def guard(*args, **kwargs):
        events.append(("guard",))
        return original_guard(*args, **kwargs)

    def verify(**kwargs):
        events.append(("verifier", kwargs["mode"], kwargs["retrieval_result"]))
        return original_verify(**kwargs)

    ledger.reserve = reserve
    ledger.settle = settle
    provider.complete = complete
    monkeypatch.setattr(orchestrator_module, "validate_dialogue_output", guard)
    orchestrator._style_verifier.verify = verify

    pair = orchestrator.reply_ab("bounded-pair", sample_request(), trusted_user_id=OWNER)
    assert pair.status is ABStatus.COMPLETE
    assert events[0] == ("reserve", 2)
    assert [event[0] for event in events[1:]] == [
        "provider",
        "guard",
        "verifier",
        "settle",
        "provider",
        "guard",
        "verifier",
        "settle",
    ]
    verification = [event for event in events if event[0] == "verifier"]
    assert {event[1] for event in verification} == {"baseline", "styled"}
    assert next(event[2] for event in verification if event[1] == "baseline") is None
    assert next(event[2] for event in verification if event[1] == "styled") is not None
    assert provider.call_count == 2
    assert (profile.calls, evidence.calls) == (2, 1)
    assert pair.request_attempt_count == 2
    assert orchestrator._sessions == {}
    state = json.loads(
        (tmp_path / "runtime" / "dialogue-usage-2026-09.json").read_text(encoding="utf-8")
    )
    assert state["reservations"] == {}


def test_real_builder_requires_explicit_config_and_enables_verifier_with_opt_in(
    tmp_path: Path,
):
    root = tmp_path / "private"
    root.mkdir()
    settings = Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=root,
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )
    config = synthetic_dialogue_config(root)
    secrets = DialogueSecrets(api_key=SecretStr("synthetic-key"))
    with pytest.raises(ValueError):
        build_real_orchestrator(
            settings,
            config,
            secrets,
            trusted_user_id=OWNER,
            enable_production_personalization=True,
        )
    sources = ProductionStyleSources.model_validate(
        {
            field: root / "framework" / f"{field}.json"
            for field in ProductionStyleSources.model_fields
        }
    )
    configured = config.model_copy(
        update={"style_framework": sources, "style_framework_owner_id": OWNER}
    )
    disabled = build_real_orchestrator(settings, configured, secrets, trusted_user_id=OWNER)
    assert disabled._style_personalization is None
    assert disabled._style_verifier is None
    # Non-Admin paths and a missing Admin config never reach strict Phase 1/5 wiring.
    with pytest.raises(ValueError, match="production personalization configuration unavailable"):
        build_real_orchestrator(
            settings,
            configured,
            secrets,
            trusted_user_id=OWNER,
            enable_production_personalization=True,
        )


def test_verifier_receives_exact_compiled_retrieval_after_output_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import companion.dialogue.style_personalization as pipeline_module

    original_compile = pipeline_module.compile_personalization_spec
    captured = {}

    def recording_compile(**kwargs):
        spec = original_compile(**kwargs)
        captured["retrieval"] = spec.source_retrieval
        return spec

    class RecordingVerifier:
        def __init__(self):
            self.delegate = DeterministicStyleVerifier(trusted_user_id=OWNER)
            self.calls = 0

        def verify(self, **kwargs):
            self.calls += 1
            assert kwargs["retrieval_result"] is captured["retrieval"]
            return self.delegate.verify(**kwargs)

    monkeypatch.setattr(pipeline_module, "compile_personalization_spec", recording_compile)
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    verifier = RecordingVerifier()
    orchestrator._style_verifier = verifier
    run = orchestrator.reply(sample_request(), trusted_user_id=OWNER)
    assert run.trace.verification_action == "accepted"
    assert verifier.calls == 1
    assert provider.call_count == 1
    assert run.trace.additional_provider_calls == run.trace.correction_attempts == 0

    provider.text = "我代表主人承诺这笔交易。"
    blocked = orchestrator.reply(sample_request(), trusted_user_id=OWNER)
    assert blocked.response.blocked_reason == "unsafe_output"
    assert verifier.calls == 1
    assert provider.call_count == 2
    ledger_state = json.loads(
        (tmp_path / "runtime" / "dialogue-usage-2026-09.json").read_text(encoding="utf-8")
    )
    assert ledger_state["reservations"] == {}


def test_styled_stage_order_and_formal_catalog_are_request_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import companion.dialogue.style_personalization as pipeline_module

    events = []
    profile = RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED)
    evidence = RecordingEvidence()
    original_read = profile.read_snapshot_state
    original_retrieve = evidence.retrieve_evidence

    def read(user_id):
        events.append("snapshot")
        return original_read(user_id)

    def retrieve(**kwargs):
        events.append("retrieval")
        return original_retrieve(**kwargs)

    profile.read_snapshot_state = read
    evidence.retrieve_evidence = retrieve
    for name, label in (
        ("build_pragmatic_context_delta", "delta"),
        ("build_formal_style_scenario_catalog", "catalog"),
        ("route_open_set_style", "route"),
        ("compile_personalization_spec", "spec"),
    ):
        original = getattr(pipeline_module, name)

        def wrapped(*args, _original=original, _label=label, **kwargs):
            events.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(pipeline_module, name, wrapped)

    orchestrator, provider = _orchestrator(tmp_path, profile, evidence)
    original_complete = provider.complete

    def complete(messages):
        events.append("provider")
        return original_complete(messages)

    provider.complete = complete
    run = orchestrator.reply(sample_request(), trusted_user_id=OWNER)
    assert run.response.blocked_reason is None
    assert events == [
        "snapshot",
        "delta",
        "catalog",
        "route",
        "retrieval",
        "spec",
        "snapshot",
        "provider",
    ]
    assert evidence.catalog_size == 11
    assert not hasattr(orchestrator._style_personalization, "last_catalog")


def test_open_set_real_store_does_not_load_evidence_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "private"
    root.mkdir()
    store = SingleTrustedUserStyleEvidenceStore(
        trusted_user_id=OWNER,
        private_data_dir=root,
        evidence_source=root / "evidence.json",
        index_source=root / "index.json",
        cache_source=root / "cache.json",
        version_source=root / "version.json",
    )

    def fail_read():
        raise AssertionError("open-set touched evidence state")

    monkeypatch.setattr(store, "_load_state", fail_read)
    config = synthetic_dialogue_config(tmp_path)
    provider = FakeProvider("synthetic safe answer")
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: NOW,
        style_personalization=ProductionStylePersonalization(
            trusted_user_id=OWNER,
            profile_gateway=RecordingProfile(UserStyleSnapshotReadStatus.AUTHORIZED),
            evidence_gateway=store,
        ),
        style_verifier=DeterministicStyleVerifier(trusted_user_id=OWNER),
    )
    run = orchestrator.reply(sample_request(), trusted_user_id=OWNER)
    assert run.response.blocked_reason is None
    assert provider.call_count == 1
    assert "retrieval_status=not_requested" in provider.system_text
