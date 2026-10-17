from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.style import (
    AdaptiveCandidateStatus,
    AdaptiveFeatureCorrection,
    AdaptiveLearningConsentStatus,
    AdaptiveStyleLearningConsent,
    SingleTrustedUserAdaptiveStyleStore,
    StyleEvidenceApproval,
    StyleEvidenceConfidenceMethod,
    StyleEvidenceObservedFrom,
    StyleEvidenceRecord,
    StyleEvidenceSourceType,
    StyleFeatureKind,
    StyleFeatureObservation,
    StyleObservationSource,
    StyleVerificationAction,
    StyleVerificationCheck,
    StyleVerificationCheckName,
    StyleVerificationReport,
)
from companion.style.errors import StyleValidationError

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
FIXTURE = Path("fixtures/synthetic/adaptive_style_learning_scenarios.json")


def _store(tmp_path: Path, user_id: str = "owner-a") -> SingleTrustedUserAdaptiveStyleStore:
    return SingleTrustedUserAdaptiveStyleStore(
        trusted_user_id=user_id,
        private_data_dir=tmp_path,
        consent_source=tmp_path / "learning-consent.json",
        observation_source=tmp_path / "learning-observations.json",
        approved_evidence_source=tmp_path / "learning-approved-evidence.json",
        candidate_source=tmp_path / "learning-candidate.json",
        active_source=tmp_path / "learning-active.json",
        history_source=tmp_path / "learning-history.json",
        statistics_source=tmp_path / "learning-statistics.json",
        index_source=tmp_path / "learning-index.json",
        cache_source=tmp_path / "learning-cache.json",
        version_source=tmp_path / "learning-version.json",
    )


def _grant(store: SingleTrustedUserAdaptiveStyleStore, *, version: str = "learning-v1") -> None:
    store.grant_learning_consent(
        user_id="owner-a",
        consent=AdaptiveStyleLearningConsent(
            user_id="owner-a",
            authorization_version=version,
            status=AdaptiveLearningConsentStatus.AUTHORIZED,
            updated_at=NOW - timedelta(days=100),
        ),
    )


def _observation(
    observation_id: str,
    value: float,
    age_days: int,
    *,
    kind: StyleFeatureKind = StyleFeatureKind.EMOJI_USAGE_RATIO,
    user_id: str = "owner-a",
    authorization_version: str = "learning-v1",
) -> StyleFeatureObservation:
    return StyleFeatureObservation(
        user_id=user_id,
        observation_id=observation_id,
        feature_kind=kind,
        value=value,
        source=StyleObservationSource.OWNER_MESSAGE,
        authorization_version=authorization_version,
        source_version="public-synthetic-observer-v1",
        observed_at=NOW - timedelta(days=age_days),
    )


def _base_snapshot(
    *,
    value: float = 0.2,
    kind: StyleFeatureKind = StyleFeatureKind.EMOJI_USAGE_RATIO,
):
    from tests.style.test_style_evidence_retrieval import _snapshot

    snapshot = _snapshot()
    payload = snapshot.model_dump(mode="python")
    payload["profile"]["features"][0]["kind"] = kind
    payload["profile"]["features"][0]["value"] = value
    return type(snapshot).model_validate(payload)


def _passing_report() -> StyleVerificationReport:
    checks = tuple(
        StyleVerificationCheck(name=name, passed=True, covered=True, code="public_check_passed")
        for name in StyleVerificationCheckName
    )
    return StyleVerificationReport(
        semantic_correctness=checks[0],
        style_consistency=checks[1],
        identity_confusion=checks[2],
        mechanical_repetition=checks[3],
        generic_assistant_tone=checks[4],
        action=StyleVerificationAction.ACCEPT,
    )


def test_public_fixture_declares_threshold_conflict_drift_and_lifecycle_cases() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["fixture_version"] == "public-synthetic-adaptive-style-v1"
    assert payload["privacy_class"] == "public_synthetic_only"
    assert {case["expected_status"] for case in payload["cases"]} >= {
        "ready",
        "conflict",
        "drift_blocked",
        "insufficient_evidence",
    }


def test_consent_is_independent_and_unauthorized_paths_do_not_read_or_write_derivatives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    reads: list[Path] = []
    import companion.style.adaptive_learning as module

    original = module.read_private_model

    def recording_read(model_type, source, private_data_dir):
        reads.append(source)
        return original(model_type, source, private_data_dir)

    monkeypatch.setattr(module, "read_private_model", recording_read)
    with pytest.raises(StyleValidationError, match="adaptive_style_not_authorized"):
        store.record_observation(user_id="owner-a", observation=_observation("obs-1", 0.3, 1))
    assert reads == []
    assert not any(path.exists() for path in store.derived_artifact_paths)

    _grant(store)
    reads.clear()
    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        store.view_learning_state("owner-b")
    assert reads == []


def test_observation_model_excludes_partner_robot_text_and_expression_inference() -> None:
    payload = _observation("obs-1", 0.3, 1).model_dump(mode="python")
    for source in ("partner_message", "robot_output"):
        payload["source"] = source
        with pytest.raises(ValidationError):
            StyleFeatureObservation.model_validate(payload)
    payload = _observation("obs-1", 0.3, 1).model_dump(mode="python")
    payload["feature_kind"] = StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE
    with pytest.raises(ValidationError):
        StyleFeatureObservation.model_validate(payload)
    payload = _observation("obs-1", 0.3, 1).model_dump(mode="python")
    payload["semantic_route_used"] = True
    with pytest.raises(ValidationError):
        StyleFeatureObservation.model_validate(payload)
    assert "text" not in StyleFeatureObservation.model_fields
    assert "content" not in StyleFeatureObservation.model_fields


def test_threshold_day_window_and_weights_are_exact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    for observation in (
        _observation("obs-1", 0.24, 0),
        _observation("obs-2", 0.26, 1),
        _observation("obs-3", 0.28, 2),
        _observation("obs-4", 0.30, 30),
        _observation("obs-5", 0.32, 60),
        _observation("obs-old", 0.99, 91),
    ):
        store.record_observation(user_id="owner-a", observation=observation)

    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    proposal = candidate.feature_proposals[0]
    assert candidate.status is AdaptiveCandidateStatus.READY
    assert proposal.raw_observation_count == 5
    assert proposal.utc_day_count == 5
    assert proposal.weighted_support == 4.75
    assert "obs-old" not in proposal.observation_refs


def test_time_decay_uses_inclusive_30_60_90_day_boundaries(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    for index, age in enumerate((0, 1, 2, 30, 31, 60, 61, 90, 91)):
        store.record_observation(
            user_id="owner-a",
            observation=_observation(f"boundary-{index}", 0.25, age),
        )

    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    proposal = candidate.feature_proposals[0]
    assert proposal.raw_observation_count == 8
    assert proposal.weighted_support == 6.5
    assert "boundary-7" in proposal.observation_refs
    assert "boundary-8" not in proposal.observation_refs


def test_minimum_five_observations_three_days_and_four_weighted_support(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    for index, age in enumerate((60, 61, 62, 63, 64), start=1):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", 0.25, age)
        )
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    assert candidate.status is AdaptiveCandidateStatus.INSUFFICIENT_EVIDENCE

    (tmp_path / "days").mkdir()
    other = _store(tmp_path / "days")
    _grant(other)
    for index in range(5):
        other.record_observation(
            user_id="owner-a", observation=_observation(f"same-{index}", 0.25, index % 2)
        )
    assert (
        other.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW).status
        is AdaptiveCandidateStatus.INSUFFICIENT_EVIDENCE
    )

    (tmp_path / "exact").mkdir()
    exact = _store(tmp_path / "exact")
    _grant(exact)
    for index, age in enumerate((0, 31, 32, 33, 34)):
        exact.record_observation(
            user_id="owner-a", observation=_observation(f"exact-{index}", 0.25, age)
        )
    exact_candidate = exact.propose_candidate(
        user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW
    )
    assert exact_candidate.status is AdaptiveCandidateStatus.READY
    assert exact_candidate.feature_proposals[0].weighted_support == 4.0


def test_bidirectional_twenty_percent_conflict_blocks_activation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot(value=0.5)
    for index in range(5):
        store.record_observation(
            user_id="owner-a",
            observation=_observation(f"base-{index}", 0.5, index),
        )
    baseline = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    store.activate_candidate(
        user_id="owner-a",
        base_snapshot=snapshot,
        candidate_version=baseline.candidate_version,
        owner_confirmation_version="approval-base",
        accepts_drift=False,
        verifier_report=_passing_report(),
        activated_at=NOW + timedelta(seconds=1),
    )
    for index, value in enumerate((0.7, 0.7, 0.7, 0.3, 0.3), start=1):
        store.record_observation(
            user_id="owner-a",
            observation=_observation(f"conflict-{index}", value, index - 1),
        )
    candidate = store.propose_candidate(
        user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW + timedelta(seconds=2)
    )
    assert candidate.status is AdaptiveCandidateStatus.CONFLICT
    assert candidate.feature_proposals[0].downward_weight_share == 0.2
    with pytest.raises(StyleValidationError, match="adaptive_candidate_conflict"):
        store.activate_candidate(
            user_id="owner-a",
            base_snapshot=snapshot,
            candidate_version=candidate.candidate_version,
            owner_confirmation_version="approval-v1",
            accepts_drift=False,
            verifier_report=_passing_report(),
            activated_at=NOW + timedelta(minutes=1),
        )


def test_first_candidate_conflict_uses_phase1_snapshot_as_current_baseline(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot(value=0.5)
    for index, value in enumerate((0.7, 0.7, 0.7, 0.3, 0.3)):
        store.record_observation(
            user_id="owner-a",
            observation=_observation(f"first-conflict-{index}", value, index),
        )

    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    proposal = candidate.feature_proposals[0]
    assert candidate.status is AdaptiveCandidateStatus.CONFLICT
    assert proposal.upward_weight_share == 0.6
    assert proposal.downward_weight_share == 0.4
    with pytest.raises(StyleValidationError, match="adaptive_candidate_conflict"):
        store.activate_candidate(
            user_id="owner-a",
            base_snapshot=snapshot,
            candidate_version=candidate.candidate_version,
            owner_confirmation_version="approval-first-conflict",
            accepts_drift=True,
            verifier_report=_passing_report(),
            activated_at=NOW + timedelta(minutes=1),
        )


@pytest.mark.parametrize(
    ("kind", "base", "values"),
    (
        (StyleFeatureKind.TERMINAL_PUNCTUATION_RATIO, 0.2, (0.31,) * 5),
        (StyleFeatureKind.EMOJI_USAGE_RATIO, 0.2, (0.31,) * 5),
        (StyleFeatureKind.FUNCTION_WORD_RATIO, 0.2, (0.31,) * 5),
        (StyleFeatureKind.FOLLOW_UP_QUESTION_RATIO, 0.2, (0.31,) * 5),
        (StyleFeatureKind.MESSAGE_LENGTH_Q25, 20.0, (29.0,) * 5),
        (StyleFeatureKind.MESSAGE_LENGTH_MEDIAN, 20.0, (29.0,) * 5),
        (StyleFeatureKind.MESSAGE_LENGTH_Q75, 20.0, (29.0,) * 5),
    ),
)
def test_feature_specific_drift_requires_explicit_drift_approval(
    tmp_path: Path,
    kind: StyleFeatureKind,
    base: float,
    values: tuple[float, ...],
) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot(value=base, kind=kind)
    for index, value in enumerate(values):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", value, index, kind=kind)
        )
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    assert candidate.status is AdaptiveCandidateStatus.DRIFT_BLOCKED
    with pytest.raises(StyleValidationError, match="adaptive_drift_approval_required"):
        store.activate_candidate(
            user_id="owner-a",
            base_snapshot=snapshot,
            candidate_version=candidate.candidate_version,
            owner_confirmation_version="approval-v1",
            accepts_drift=False,
            verifier_report=_passing_report(),
            activated_at=NOW + timedelta(minutes=1),
        )


def test_activation_version_chain_correction_and_rollback_are_append_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    for index, value in enumerate((0.22, 0.23, 0.24, 0.25, 0.26)):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", value, index)
        )
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    active_v1 = store.activate_candidate(
        user_id="owner-a",
        base_snapshot=snapshot,
        candidate_version=candidate.candidate_version,
        owner_confirmation_version="approval-v1",
        accepts_drift=False,
        verifier_report=_passing_report(),
        activated_at=NOW + timedelta(minutes=1),
    )
    correction_candidate = store.correct_feature(
        user_id="owner-a",
        base_snapshot=snapshot,
        correction=AdaptiveFeatureCorrection(
            user_id="owner-a",
            feature_kind=StyleFeatureKind.EMOJI_USAGE_RATIO,
            corrected_value=0.4,
            authorization_version="learning-v1",
            correction_version="correction-v1",
            corrected_at=NOW + timedelta(minutes=2),
            explicitly_confirmed=True,
        ),
    )
    active_v2 = store.activate_candidate(
        user_id="owner-a",
        base_snapshot=snapshot,
        candidate_version=correction_candidate.candidate_version,
        owner_confirmation_version="approval-v2",
        accepts_drift=True,
        verifier_report=_passing_report(),
        activated_at=NOW + timedelta(minutes=3),
    )
    rolled_back = store.rollback(
        user_id="owner-a",
        target_active_version=active_v1.active_version,
        owner_confirmation_version="rollback-v1",
        rolled_back_at=NOW + timedelta(minutes=4),
    )
    state = store.view_learning_state("owner-a")
    assert active_v1.active_version != active_v2.active_version != rolled_back.active_version
    assert rolled_back.feature_values == active_v1.feature_values
    assert state is not None
    assert tuple(item.active_version for item in state.history.entries) == (
        active_v1.active_version,
        active_v2.active_version,
    )


def test_revocation_and_delete_clear_only_phase7_derivatives(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    store.record_observation(user_id="owner-a", observation=_observation("obs-1", 0.2, 1))
    unrelated = tmp_path / "phase1-source.json"
    unrelated.write_text("public synthetic sentinel", encoding="utf-8")

    store.revoke_learning_consent(user_id="owner-a", updated_at=NOW)
    consent = store.view_learning_consent("owner-a")
    assert consent is not None and consent.status is AdaptiveLearningConsentStatus.REVOKED
    assert not any(path.exists() for path in store.derived_artifact_paths)
    assert unrelated.exists()

    store.delete_adaptive_learning("owner-a")
    assert not any(path.exists() for path in store.artifact_paths)
    assert unrelated.exists()


def test_second_user_fake_isolated_for_view_rollback_and_delete(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _store(tmp_path / "a", "owner-a")
    second = _store(tmp_path / "b", "owner-b")
    first.grant_learning_consent(
        user_id="owner-a",
        consent=AdaptiveStyleLearningConsent(
            user_id="owner-a",
            authorization_version="learning-a",
            status=AdaptiveLearningConsentStatus.AUTHORIZED,
            updated_at=NOW - timedelta(days=2),
        ),
    )
    second.grant_learning_consent(
        user_id="owner-b",
        consent=AdaptiveStyleLearningConsent(
            user_id="owner-b",
            authorization_version="learning-b",
            status=AdaptiveLearningConsentStatus.AUTHORIZED,
            updated_at=NOW - timedelta(days=2),
        ),
    )
    first.record_observation(
        user_id="owner-a",
        observation=_observation("a-1", 0.2, 1, authorization_version="learning-a"),
    )
    second.record_observation(
        user_id="owner-b",
        observation=_observation(
            "b-1", 0.8, 1, user_id="owner-b", authorization_version="learning-b"
        ),
    )
    first.delete_adaptive_learning("owner-a")
    second_state = second.view_learning_state("owner-b")
    assert second_state is not None
    assert second_state.observations.observations[0].observation_id == "b-1"


def test_wrong_authorization_and_hard_verifier_failure_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    with pytest.raises(StyleValidationError, match="invalid_adaptive_observation"):
        store.record_observation(
            user_id="owner-a",
            observation=_observation("obs-1", 0.2, 1, authorization_version="old-learning"),
        )

    bad = _passing_report().model_dump(mode="python")
    bad["semantic_correctness"]["passed"] = False
    bad["semantic_correctness"]["code"] = "public_constraint_failed"
    bad["action"] = StyleVerificationAction.REJECT
    report = StyleVerificationReport.model_validate(bad)
    for index, value in enumerate((0.22, 0.23, 0.24, 0.25, 0.26)):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"valid-{index}", value, index)
        )
    snapshot = _base_snapshot()
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    with pytest.raises(StyleValidationError, match="adaptive_verifier_rejected"):
        store.activate_candidate(
            user_id="owner-a",
            base_snapshot=snapshot,
            candidate_version=candidate.candidate_version,
            owner_confirmation_version="approval-v1",
            accepts_drift=False,
            verifier_report=report,
            activated_at=NOW + timedelta(minutes=1),
        )


def test_verifier_requires_accept_and_all_five_checks_covered_and_passed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    for index, value in enumerate((0.22, 0.23, 0.24, 0.25, 0.26)):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", value, index)
        )
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    payload = _passing_report().model_dump(mode="python")
    payload["style_consistency"]["covered"] = False
    report = StyleVerificationReport.model_validate(payload)

    with pytest.raises(StyleValidationError, match="adaptive_verifier_rejected"):
        store.activate_candidate(
            user_id="owner-a",
            base_snapshot=snapshot,
            candidate_version=candidate.candidate_version,
            owner_confirmation_version="approval-v1",
            accepts_drift=False,
            verifier_report=report,
            activated_at=NOW + timedelta(minutes=1),
        )


def test_base_snapshot_is_deeply_revalidated_before_proposal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    payload = snapshot.model_dump(mode="python")
    payload["evidence_count"] = 999
    forged = type(snapshot).model_validate(payload)

    with pytest.raises(StyleValidationError, match="invalid_adaptive_base_profile"):
        store.propose_candidate(user_id="owner-a", base_snapshot=forged, proposal_at=NOW)


def test_candidate_binds_complete_snapshot_and_rejects_snapshot_tampering(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    for index, value in enumerate((0.22, 0.23, 0.24, 0.25, 0.26)):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", value, index)
        )
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)

    alternate_payload = snapshot.model_dump(mode="python")
    alternate_payload["derived"]["cache_version"] = "profile-cache-v2"
    alternate = type(snapshot).model_validate(alternate_payload)
    assert alternate.profile.profile_version == snapshot.profile.profile_version
    assert alternate.consent.authorization_version == snapshot.consent.authorization_version
    with pytest.raises(StyleValidationError, match="invalid_adaptive_activation"):
        store.activate_candidate(
            user_id="owner-a",
            base_snapshot=alternate,
            candidate_version=candidate.candidate_version,
            owner_confirmation_version="approval-wrong-snapshot",
            accepts_drift=False,
            verifier_report=_passing_report(),
            activated_at=NOW + timedelta(minutes=1),
        )

    candidate_path = tmp_path / "learning-candidate.json"
    persisted = json.loads(candidate_path.read_text(encoding="utf-8"))
    persisted["base_snapshot"]["derived"]["cache_version"] = "profile-cache-v3"
    candidate_path.write_text(json.dumps(persisted), encoding="utf-8")
    with pytest.raises(StyleValidationError, match="adaptive_style_corrupt"):
        store.view_learning_state("owner-a")


def test_candidate_binds_observation_version_and_all_metadata_artifacts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    for index, value in enumerate((0.22, 0.23, 0.24, 0.25, 0.26)):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", value, index)
        )
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    for name in ("statistics", "index", "cache", "version"):
        assert (tmp_path / f"learning-{name}.json").exists()

    store.record_observation(
        user_id="owner-a", observation=_observation("late-observation", 0.27, 3)
    )
    with pytest.raises(StyleValidationError, match="adaptive_style_corrupt"):
        store.activate_candidate(
            user_id="owner-a",
            base_snapshot=snapshot,
            candidate_version=candidate.candidate_version,
            owner_confirmation_version="approval-v1",
            accepts_drift=False,
            verifier_report=_passing_report(),
            activated_at=NOW + timedelta(minutes=1),
        )


def test_metadata_tampering_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    store.propose_candidate(user_id="owner-a", base_snapshot=_base_snapshot(), proposal_at=NOW)
    index_path = tmp_path / "learning-index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["artifact_kind"] = "cache"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StyleValidationError, match="adaptive_style_corrupt"):
        store.view_learning_state("owner-a")


def test_duplicate_activation_and_broken_history_chain_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    snapshot = _base_snapshot()
    for index, value in enumerate((0.22, 0.23, 0.24, 0.25, 0.26)):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", value, index)
        )
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    store.activate_candidate(
        user_id="owner-a",
        base_snapshot=snapshot,
        candidate_version=candidate.candidate_version,
        owner_confirmation_version="approval-v1",
        accepts_drift=False,
        verifier_report=_passing_report(),
        activated_at=NOW + timedelta(minutes=1),
    )
    with pytest.raises(StyleValidationError, match="duplicate_adaptive_activation"):
        store.activate_candidate(
            user_id="owner-a",
            base_snapshot=snapshot,
            candidate_version=candidate.candidate_version,
            owner_confirmation_version="approval-duplicate",
            accepts_drift=False,
            verifier_report=_passing_report(),
            activated_at=NOW + timedelta(minutes=2),
        )

    active_path = tmp_path / "learning-active.json"
    active_payload = json.loads(active_path.read_text(encoding="utf-8"))
    forged_active = json.loads(json.dumps(active_payload))
    forged_active["features"][0]["value"] = 0.99
    active_path.write_text(json.dumps(forged_active), encoding="utf-8")
    with pytest.raises(StyleValidationError, match="adaptive_style_corrupt"):
        store.view_learning_state("owner-a")
    active_path.write_text(json.dumps(active_payload), encoding="utf-8")

    history_path = tmp_path / "learning-history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["updated_at"] = (NOW + timedelta(minutes=3)).isoformat()
    history_path.write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(StyleValidationError, match="adaptive_style_corrupt"):
        store.view_learning_state("owner-a")


def test_numeric_learning_artifacts_do_not_persist_message_content_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    for index, value in enumerate((0.22, 0.23, 0.24, 0.25, 0.26)):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", value, index)
        )
    store.propose_candidate(user_id="owner-a", base_snapshot=_base_snapshot(), proposal_at=NOW)

    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in store.artifact_paths if path.exists()
    )
    assert "private-message-sentinel" not in serialized
    for path in store.artifact_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        pending = [payload]
        keys: set[str] = set()
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                keys.update(item)
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        assert not {"text", "content", "message", "raw_chat"}.intersection(keys)


def test_phase5_owner_approved_expression_is_the_only_expression_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    record = StyleEvidenceRecord(
        user_id="owner-a",
        evidence_ref="phase5-ref-1",
        scenario_id="public-scenario-1",
        content="public synthetic phrase",
        confidence=0.9,
        approval=StyleEvidenceApproval.APPROVED_FOR_STYLE,
        source_type=StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL,
        observed_from=StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY,
        confidence_method=StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL,
        source_version="phase5-approval-v1",
        approved_at=NOW - timedelta(days=1),
    )
    candidate = store.approve_phase5_expression(
        user_id="owner-a",
        base_snapshot=_base_snapshot(),
        evidence_record=record,
        evidence_version="phase7-evidence-v1",
        approved_at=NOW,
    )
    assert candidate.status is AdaptiveCandidateStatus.READY
    assert candidate.feature_proposals[0].feature_kind is (
        StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE
    )
    state = store.view_learning_state("owner-a")
    assert state is not None and state.approved_evidence is not None
    assert state.approved_evidence.records[0].source_ref == "phase5-ref-1"


def test_forged_candidate_version_and_status_are_rejected_on_read(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _grant(store)
    for index, value in enumerate((0.22, 0.23, 0.24, 0.25, 0.26)):
        store.record_observation(
            user_id="owner-a", observation=_observation(f"obs-{index}", value, index)
        )
    store.propose_candidate(user_id="owner-a", base_snapshot=_base_snapshot(), proposal_at=NOW)
    candidate_path = tmp_path / "learning-candidate.json"
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["candidate_version"] = "forged-candidate-v9"
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StyleValidationError, match="adaptive_style_corrupt"):
        store.view_learning_state("owner-a")


def test_quantile_inversion_is_a_non_activatable_conflict(tmp_path: Path) -> None:
    from tests.style.test_style_evidence_retrieval import _snapshot

    store = _store(tmp_path)
    _grant(store)
    snapshot = _snapshot()
    payload = snapshot.model_dump(mode="python")
    feature = payload["profile"]["features"][0]
    feature["kind"] = StyleFeatureKind.MESSAGE_LENGTH_Q25
    feature["value"] = 8.0
    q75 = dict(feature)
    q75["kind"] = StyleFeatureKind.MESSAGE_LENGTH_Q75
    q75["value"] = 12.0
    payload["profile"]["features"] = (feature, q75)
    payload["evidence_count"] = 16
    snapshot = type(snapshot).model_validate(payload)
    for index in range(5):
        store.record_observation(
            user_id="owner-a",
            observation=_observation(
                f"q25-{index}",
                13.0,
                index,
                kind=StyleFeatureKind.MESSAGE_LENGTH_Q25,
            ),
        )
    candidate = store.propose_candidate(user_id="owner-a", base_snapshot=snapshot, proposal_at=NOW)
    assert candidate.status is AdaptiveCandidateStatus.CONFLICT


def test_symlink_escape_and_alias_paths_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    inside = tmp_path / "inside"
    outside.mkdir()
    inside.mkdir()
    (inside / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="adaptive style paths"):
        SingleTrustedUserAdaptiveStyleStore(
            trusted_user_id="owner-a",
            private_data_dir=inside,
            consent_source=inside / "consent.json",
            observation_source=inside / "alias" / "observations.json",
            approved_evidence_source=inside / "approved.json",
            candidate_source=inside / "candidate.json",
            active_source=inside / "active.json",
            history_source=inside / "history.json",
            statistics_source=inside / "statistics.json",
            index_source=inside / "index.json",
            cache_source=inside / "cache.json",
            version_source=inside / "version.json",
        )


def test_delete_rejects_symlink_drift_without_touching_outside_file(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    store = _store(inside)
    _grant(store)
    outside_file = outside / "keep.json"
    outside_file.write_text("public synthetic sentinel", encoding="utf-8")
    (inside / "learning-observations.json").symlink_to(outside_file)

    with pytest.raises(StyleValidationError, match="unsafe_adaptive_style_path"):
        store.delete_adaptive_learning("owner-a")
    assert outside_file.read_text(encoding="utf-8") == "public synthetic sentinel"
