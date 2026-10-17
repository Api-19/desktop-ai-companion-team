from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from companion.privacy import (
    CALIBRATION_CANDIDATE_RELATIVE_PATH,
    CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH,
    PRECISE_INTENT_CALIBRATION_HISTORY_RELATIVE_PATH,
    PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH,
    CalibrationCandidate,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CalibrationReviewDecision,
    CalibrationReviewDecisionValue,
    CalibrationReviewSubmission,
    CalibrationRuntimeHandoff,
    CandidateCategory,
    PrivacyValidationError,
    calibration_candidate_id,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
    calibration_review_package_fingerprint,
    calibration_submission_id,
    consume_precise_intent_calibration_handoff,
    load_calibration_candidate_package,
    publish_calibration_candidate_package,
    publish_precise_intent_calibration_batch,
    write_precise_intent_calibration_handoff,
)

NOW = datetime(2026, 8, 1, 10, tzinfo=UTC)
PUBLISHED_AT = NOW + timedelta(minutes=5)
NAMES = ("合成姓名甲",)
NICKNAMES = ("小团子",)
MAPPING = (
    ("greeting-1", CandidateCategory.GREETING, "morning_greeting"),
    ("meals-1", CandidateCategory.MEALS, "missed_meal_care"),
    (
        "household-2",
        CandidateCategory.HOUSEHOLD,
        "unfinished_household_reassurance",
    ),
    ("shopping-1", CandidateCategory.SHOPPING, "shopping_comparison_request"),
    ("work_study-2", CandidateCategory.WORK_STUDY, "work_fatigue_care"),
    (
        "general_emotion-1",
        CandidateCategory.GENERAL_EMOTION,
        "positive_emotion_share",
    ),
    (
        "general_emotion-2",
        CandidateCategory.GENERAL_EMOTION,
        "negative_emotion_check_in",
    ),
    (
        "gentle_affection-1",
        CandidateCategory.GENTLE_AFFECTION,
        "missed_contact_affection",
    ),
)


def _submissions() -> list[CalibrationReviewSubmission]:
    return [
        CalibrationReviewSubmission(
            submission_id=calibration_submission_id(
                scenario_id=scenario_id,
                category=category,
                intent=intent,
            ),
            scenario_id=scenario_id,
            category=category,
            intent=intent,
            text=f"合成安全回复{index}",
            source="explicit_owner_ideal_response",
        )
        for index, (scenario_id, category, intent) in enumerate(MAPPING, start=1)
    ]


def _old_package() -> CalibrationCandidatePackage:
    old_pairs = (
        (CandidateCategory.COURTESY, "old_courtesy", "谢谢你呀"),
        (CandidateCategory.SLEEP, "old_sleep", "早点休息呀"),
        (CandidateCategory.WEATHER, "old_weather", "今天有点凉"),
        (CandidateCategory.SHORT_REPLY, "old_short", "好呀"),
        (CandidateCategory.EMOJI, "old_emoji", "哈哈"),
    )
    candidates = sorted(
        [
            CalibrationCandidate(
                candidate_id=calibration_candidate_id(
                    category=category,
                    intent=intent,
                    text=text,
                ),
                category=category,
                intent=intent,
                text=text,
            )
            for category, intent, text in old_pairs
        ],
        key=lambda item: item.candidate_id,
    )
    audit = CalibrationCandidateAudit(
        reviewed_at=NOW - timedelta(days=1),
        submission_count=5,
        approved_count=5,
        rejected_count=0,
        privacy_rejected_count=0,
        revoked_count=0,
        output_count=5,
    )
    return CalibrationCandidatePackage(
        created_at=audit.reviewed_at,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=calibration_identity_config_fingerprint(NAMES, NICKNAMES),
        candidate_count=5,
        candidates=candidates,
        audit=audit,
    )


def _prepare(
    private_root: Path,
) -> tuple[CalibrationCandidatePackage, list[CalibrationReviewDecision]]:
    old = _old_package()
    publish_calibration_candidate_package(
        old,
        private_data_dir=private_root,
        true_name_terms=NAMES,
        approved_nicknames=NICKNAMES,
    )
    submissions = _submissions()
    write_precise_intent_calibration_handoff(
        submissions,
        created_at=NOW,
        private_data_dir=private_root,
    )
    pending = consume_precise_intent_calibration_handoff(private_data_dir=private_root)
    decisions = [
        CalibrationReviewDecision(
            submission_id=item.submission_id,
            submission_fingerprint=item.content_fingerprint,
            review_package_fingerprint=calibration_review_package_fingerprint(
                submissions,
                submitted_at=pending.submitted_at,
            ),
            decided_at=NOW + timedelta(minutes=1),
            decision=CalibrationReviewDecisionValue.APPROVE,
        )
        for item in submissions
    ]
    return old, decisions


def _publish(private_root: Path, decisions: list[CalibrationReviewDecision]):
    old = _old_package()
    return publish_precise_intent_calibration_batch(
        decisions,
        published_at=PUBLISHED_AT,
        expected_current_fingerprint=old.content_fingerprint,
        true_name_terms=NAMES,
        approved_nicknames=NICKNAMES,
        private_data_dir=private_root,
    )


def test_approved_batch_merges_old_five_and_new_eight_without_overwrite(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private_data"
    old, decisions = _prepare(private_root)

    merged = _publish(private_root, decisions)

    assert merged.candidate_count == 13
    assert merged.audit.submission_count == 13
    assert merged.audit.approved_count == 13
    assert merged.audit.output_count == 13
    by_id = {item.candidate_id: item for item in merged.candidates}
    assert all(by_id[item.candidate_id] == item for item in old.candidates)
    assert {(item.category, item.intent) for item in merged.candidates} == {
        (item.category, item.intent) for item in old.candidates
    } | {(category, intent) for _, category, intent in MAPPING}
    assert len(by_id) == 13

    loaded = load_calibration_candidate_package(
        private_data_dir=private_root,
        expected_fingerprint=merged.content_fingerprint,
        true_name_terms=NAMES,
        approved_nicknames=NICKNAMES,
    )
    assert loaded == merged
    assert (private_root / PRECISE_INTENT_CALIBRATION_HISTORY_RELATIVE_PATH).exists()
    assert (private_root / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH).exists()


def test_rejected_or_incomplete_batch_cannot_change_formal_five(tmp_path: Path) -> None:
    private_root = tmp_path / "private_data"
    _, decisions = _prepare(private_root)
    formal = private_root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    original = formal.read_bytes()
    decisions[0] = decisions[0].model_copy(
        update={"decision": CalibrationReviewDecisionValue.REJECT}
    )

    with pytest.raises(PrivacyValidationError) as captured:
        _publish(private_root, decisions)

    assert captured.value.code == "precise_intent_publish_failed"
    assert formal.read_bytes() == original


def test_category_intent_collision_fails_before_formal_replacement(tmp_path: Path) -> None:
    private_root = tmp_path / "private_data"
    old, decisions = _prepare(private_root)
    colliding = old.candidates[0].model_copy(
        update={
            "candidate_id": calibration_candidate_id(
                category=CandidateCategory.GREETING,
                intent="morning_greeting",
                text=old.candidates[0].text,
            ),
            "category": CandidateCategory.GREETING,
            "intent": "morning_greeting",
        }
    )
    candidates = sorted([colliding, *old.candidates[1:]], key=lambda item: item.candidate_id)
    altered = CalibrationCandidatePackage(
        created_at=old.created_at,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=old.identity_config_fingerprint,
        candidate_count=5,
        candidates=candidates,
        audit=old.audit,
    )
    publish_calibration_candidate_package(
        altered,
        private_data_dir=private_root,
        true_name_terms=NAMES,
        approved_nicknames=NICKNAMES,
    )
    formal = private_root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    original = formal.read_bytes()

    with pytest.raises(PrivacyValidationError):
        publish_precise_intent_calibration_batch(
            decisions,
            published_at=PUBLISHED_AT,
            expected_current_fingerprint=altered.content_fingerprint,
            true_name_terms=NAMES,
            approved_nicknames=NICKNAMES,
            private_data_dir=private_root,
        )

    assert formal.read_bytes() == original


def test_runtime_handoff_failure_rolls_back_formal_and_preserves_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private_data"
    _, decisions = _prepare(private_root)
    formal = private_root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    original = formal.read_bytes()
    original_atomic = __import__(
        "companion.privacy.precise_intent_io",
        fromlist=["_atomic_write_model"],
    )._atomic_write_model

    def fail_runtime_handoff(model: object, *args: object, **kwargs: object):
        if type(model).__name__ == "CalibrationRuntimeHandoff":
            raise RuntimeError("SYNTHETIC_RUNTIME_HANDOFF_SENTINEL")
        return original_atomic(model, *args, **kwargs)

    monkeypatch.setattr(
        "companion.privacy.precise_intent_io._atomic_write_model",
        fail_runtime_handoff,
    )

    with pytest.raises(PrivacyValidationError) as captured:
        _publish(private_root, decisions)

    assert captured.value.code == "precise_intent_publish_failed"
    assert "SYNTHETIC_RUNTIME_HANDOFF_SENTINEL" not in str(captured.value)
    assert formal.read_bytes() == original
    assert (private_root / PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH).exists()


def test_runtime_handoff_failure_preserves_existing_trusted_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private_data"
    old, decisions = _prepare(private_root)
    runtime_path = private_root / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    runtime_path.parent.mkdir(parents=True)
    existing = CalibrationRuntimeHandoff(
        purpose="calibration_runtime",
        version=1,
        created_at=old.created_at,
        expected_calibration_fingerprint=old.content_fingerprint,
        identity_config_fingerprint=old.identity_config_fingerprint,
        true_name_terms=list(NAMES),
        approved_nicknames=list(NICKNAMES),
    )
    runtime_path.write_text(existing.model_dump_json(indent=2), encoding="utf-8")
    original_runtime = runtime_path.read_bytes()
    original_atomic = __import__(
        "companion.privacy.precise_intent_io",
        fromlist=["_atomic_write_model"],
    )._atomic_write_model

    def fail_runtime_handoff(model: object, *args: object, **kwargs: object):
        if type(model).__name__ == "CalibrationRuntimeHandoff":
            raise RuntimeError("SYNTHETIC_RUNTIME_HANDOFF_SENTINEL")
        return original_atomic(model, *args, **kwargs)

    monkeypatch.setattr(
        "companion.privacy.precise_intent_io._atomic_write_model",
        fail_runtime_handoff,
    )

    with pytest.raises(PrivacyValidationError):
        _publish(private_root, decisions)

    assert runtime_path.read_bytes() == original_runtime
