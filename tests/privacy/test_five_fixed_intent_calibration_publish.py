from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from companion.privacy import (
    CALIBRATION_CANDIDATE_RELATIVE_PATH,
    CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH,
    FIVE_FIXED_INTENT_CALIBRATION_HISTORY_RELATIVE_PATH,
    FIVE_FIXED_INTENT_CALIBRATION_MAPPING,
    FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH,
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
    consume_five_fixed_intent_calibration_handoff,
    publish_calibration_candidate_package,
    publish_five_fixed_intent_calibration_batch,
    write_five_fixed_intent_calibration_handoff,
)

NOW = datetime(2026, 8, 2, 10, tzinfo=UTC)
PUBLISHED = NOW + timedelta(minutes=5)
NAMES = ("合成姓名甲",)
NICKNAMES = ("小团子",)


def _submissions() -> list[CalibrationReviewSubmission]:
    return [
        CalibrationReviewSubmission(
            submission_id=calibration_submission_id(
                scenario_id=scenario, category=category, intent=intent
            ),
            scenario_id=scenario,
            category=category,
            intent=intent,
            text=f"合成新增回复{index}",
            source="explicit_owner_ideal_response",
        )
        for index, (scenario, category, intent) in enumerate(
            FIVE_FIXED_INTENT_CALIBRATION_MAPPING, start=1
        )
    ]


def _current() -> CalibrationCandidatePackage:
    pairs = [(CandidateCategory.WEATHER, "weather_cold_reminder", "合成降温提醒")]
    pairs.extend(
        (CandidateCategory.COURTESY, f"existing_{index}", f"合成既有回复{index}")
        for index in range(12)
    )
    candidates = sorted(
        [
            CalibrationCandidate(
                candidate_id=calibration_candidate_id(category=category, intent=intent, text=text),
                category=category,
                intent=intent,
                text=text,
            )
            for category, intent, text in pairs
        ],
        key=lambda item: item.candidate_id,
    )
    audit = CalibrationCandidateAudit(
        reviewed_at=NOW - timedelta(days=1),
        submission_count=13,
        approved_count=13,
        rejected_count=0,
        privacy_rejected_count=0,
        output_count=13,
    )
    return CalibrationCandidatePackage(
        created_at=audit.reviewed_at,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=calibration_identity_config_fingerprint(NAMES, NICKNAMES),
        candidate_count=13,
        candidates=candidates,
        audit=audit,
    )


def _prepare(root: Path) -> tuple[CalibrationCandidatePackage, list[CalibrationReviewDecision]]:
    current = _current()
    publish_calibration_candidate_package(
        current,
        private_data_dir=root,
        true_name_terms=NAMES,
        approved_nicknames=NICKNAMES,
    )
    runtime = CalibrationRuntimeHandoff(
        purpose="calibration_runtime",
        version=1,
        created_at=current.created_at,
        expected_calibration_fingerprint=current.content_fingerprint,
        identity_config_fingerprint=current.identity_config_fingerprint,
        true_name_terms=list(NAMES),
        approved_nicknames=list(NICKNAMES),
    )
    runtime_path = root / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text(runtime.model_dump_json(), encoding="utf-8")
    submissions = _submissions()
    write_five_fixed_intent_calibration_handoff(submissions, created_at=NOW, private_data_dir=root)
    pending = consume_five_fixed_intent_calibration_handoff(private_data_dir=root)
    fingerprint = calibration_review_package_fingerprint(
        submissions, submitted_at=pending.submitted_at
    )
    decisions = [
        CalibrationReviewDecision(
            submission_id=item.submission_id,
            submission_fingerprint=item.content_fingerprint,
            review_package_fingerprint=fingerprint,
            decided_at=NOW + timedelta(minutes=1),
            decision=CalibrationReviewDecisionValue.APPROVE,
        )
        for item in submissions
    ]
    return current, decisions


def _publish(root: Path, decisions: list[CalibrationReviewDecision]):
    return publish_five_fixed_intent_calibration_batch(
        decisions,
        published_at=PUBLISHED,
        expected_current_fingerprint=_current().content_fingerprint,
        true_name_terms=NAMES,
        approved_nicknames=NICKNAMES,
        private_data_dir=root,
    )


def test_publish_preserves_thirteen_adds_five_and_records_one_reuse(tmp_path: Path) -> None:
    root = tmp_path / "private_data"
    current, decisions = _prepare(root)

    result = _publish(root, decisions)

    assert result.package.candidate_count == 18
    assert result.added_count == 5
    assert result.reused_existing == 1
    by_id = {candidate.candidate_id: candidate for candidate in result.package.candidates}
    assert all(by_id[candidate.candidate_id] == candidate for candidate in current.candidates)
    assert len({(item.category, item.intent) for item in result.package.candidates}) == 18
    assert (root / FIVE_FIXED_INTENT_CALIBRATION_HISTORY_RELATIVE_PATH).exists()
    assert not (root / FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH).exists()


def test_runtime_failure_restores_formal_runtime_and_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private_data"
    _, decisions = _prepare(root)
    formal = root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    runtime = root / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    pending = root / FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH
    before = (formal.read_bytes(), runtime.read_bytes(), pending.read_bytes())
    original = __import__(
        "companion.privacy.precise_intent_io", fromlist=["_atomic_write_model"]
    )._atomic_write_model

    def fail_runtime(model: object, *args: object, **kwargs: object):
        if type(model).__name__ == "CalibrationRuntimeHandoff":
            raise RuntimeError("SYNTHETIC_PRIVATE_SENTINEL")
        return original(model, *args, **kwargs)

    monkeypatch.setattr("companion.privacy.precise_intent_io._atomic_write_model", fail_runtime)
    with pytest.raises(PrivacyValidationError) as captured:
        _publish(root, decisions)

    assert captured.value.code == "five_fixed_intent_publish_failed"
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in str(captured.value)
    assert (formal.read_bytes(), runtime.read_bytes(), pending.read_bytes()) == before
