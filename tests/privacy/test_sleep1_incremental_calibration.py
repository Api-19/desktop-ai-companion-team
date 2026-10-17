from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.privacy import (
    CALIBRATION_CANDIDATE_RELATIVE_PATH,
    CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH,
    SLEEP1_INCREMENTAL_CALIBRATION_HISTORY_RELATIVE_PATH,
    SLEEP1_INCREMENTAL_CALIBRATION_INTAKE_RELATIVE_PATH,
    SLEEP1_INCREMENTAL_CALIBRATION_MAPPING,
    SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH,
    SLEEP1_INCREMENTAL_CALIBRATION_PURPOSE,
    CalibrationCandidate,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CalibrationReviewDecision,
    CalibrationReviewDecisionValue,
    CalibrationReviewSubmission,
    CalibrationRuntimeHandoff,
    CandidateCategory,
    PrivacyValidationError,
    Sleep1IncrementalCalibrationHandoff,
    calibration_candidate_id,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
    calibration_review_package_fingerprint,
    calibration_submission_id,
    consume_sleep1_incremental_calibration_handoff,
    publish_calibration_candidate_package,
    publish_sleep1_incremental_calibration_batch,
    write_sleep1_incremental_calibration_handoff,
)

NOW = datetime(2026, 9, 13, 9, tzinfo=UTC)
PUBLISHED = NOW + timedelta(minutes=5)
NAMES = ("合成姓名甲",)
NICKNAMES = ("合成爱称甲",)
SENTINEL = "SYNTHETIC_SLEEP1_PRIVATE_SENTINEL"


class _HostileSequence(Sequence[CalibrationReviewSubmission]):
    def __len__(self) -> int:
        raise RuntimeError(SENTINEL)

    def __getitem__(self, index: int) -> CalibrationReviewSubmission:
        raise RuntimeError(SENTINEL)

    def __iter__(self):
        raise RuntimeError(SENTINEL)


def _submission() -> CalibrationReviewSubmission:
    scenario, category, intent = SLEEP1_INCREMENTAL_CALIBRATION_MAPPING[0]
    return CalibrationReviewSubmission(
        submission_id=calibration_submission_id(
            scenario_id=scenario, category=category, intent=intent
        ),
        scenario_id=scenario,
        category=category,
        intent=intent,
        text="合成睡前回复",
        source="explicit_owner_ideal_response",
    )


def _current(*, include_target: bool = False) -> CalibrationCandidatePackage:
    pairs = [
        (CandidateCategory.COURTESY, f"existing_{index}", f"合成既有回复{index}")
        for index in range(18)
    ]
    if include_target:
        pairs[-1] = (CandidateCategory.SLEEP, "sleep_farewell", "合成既有睡前回复")
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
    audited_at = NOW - timedelta(days=1)
    return CalibrationCandidatePackage(
        created_at=audited_at,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=calibration_identity_config_fingerprint(NAMES, NICKNAMES),
        candidate_count=18,
        candidates=candidates,
        audit=CalibrationCandidateAudit(
            reviewed_at=audited_at,
            submission_count=18,
            approved_count=18,
            rejected_count=0,
            privacy_rejected_count=0,
            output_count=18,
        ),
    )


def _prepare(root: Path, *, include_target: bool = False):
    current = _current(include_target=include_target)
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
    submission = _submission()
    write_sleep1_incremental_calibration_handoff(
        [submission], created_at=NOW, private_data_dir=root
    )
    pending = consume_sleep1_incremental_calibration_handoff(private_data_dir=root)
    decision = CalibrationReviewDecision(
        submission_id=submission.submission_id,
        submission_fingerprint=submission.content_fingerprint,
        review_package_fingerprint=calibration_review_package_fingerprint(
            [submission], submitted_at=pending.submitted_at
        ),
        decided_at=NOW + timedelta(minutes=1),
        decision=CalibrationReviewDecisionValue.APPROVE,
    )
    return current, [decision]


def _publish(root: Path, current: CalibrationCandidatePackage, decisions):
    return publish_sleep1_incremental_calibration_batch(
        decisions,
        published_at=PUBLISHED,
        expected_current_fingerprint=current.content_fingerprint,
        true_name_terms=NAMES,
        approved_nicknames=NICKNAMES,
        private_data_dir=root,
    )


def test_handoff_requires_exact_single_mapping() -> None:
    handoff = Sleep1IncrementalCalibrationHandoff(
        purpose=SLEEP1_INCREMENTAL_CALIBRATION_PURPOSE,
        version=1,
        created_at=NOW,
        item_count=1,
        submissions=[_submission()],
    )
    assert handoff.item_count == 1
    bad = _submission().model_copy(update={"intent": "sleep_reminder"})
    with pytest.raises(ValidationError):
        Sleep1IncrementalCalibrationHandoff(
            purpose=SLEEP1_INCREMENTAL_CALIBRATION_PURPOSE,
            version=1,
            created_at=NOW,
            item_count=1,
            submissions=[bad],
        )


def test_write_sanitizes_hostile_sequence(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(PrivacyValidationError) as captured:
        write_sleep1_incremental_calibration_handoff(
            _HostileSequence(), created_at=NOW, private_data_dir=tmp_path / "private_data"
        )

    assert captured.value.code == "invalid_sleep1_incremental_handoff"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    traceback = captured.value.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename:
            assert SENTINEL not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    captured_io = capsys.readouterr()
    assert SENTINEL not in captured_io.out
    assert SENTINEL not in captured_io.err
    assert SENTINEL not in caplog.text


def test_consume_failure_preserves_intake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "private_data"
    intake = write_sleep1_incremental_calibration_handoff(
        [_submission()], created_at=NOW, private_data_dir=root
    )

    def fail_write(*args: object, **kwargs: object) -> Path:
        raise OSError(SENTINEL)

    monkeypatch.setattr("companion.privacy.precise_intent_io._atomic_write_model", fail_write)
    with pytest.raises(PrivacyValidationError) as captured:
        consume_sleep1_incremental_calibration_handoff(private_data_dir=root)

    assert captured.value.code == "sleep1_incremental_pending_failed"
    assert SENTINEL not in str(captured.value)
    assert intake.exists()
    assert not (root / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH).exists()


def test_write_consume_and_publish_preserve_eighteen_and_add_one(tmp_path: Path) -> None:
    root = tmp_path / "private_data"
    current, decisions = _prepare(root)
    assert not (root / SLEEP1_INCREMENTAL_CALIBRATION_INTAKE_RELATIVE_PATH).exists()
    assert (root / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH).exists()

    result = _publish(root, current, decisions)

    assert result.added_count == 1
    assert result.package.candidate_count == 19
    by_id = {item.candidate_id: item for item in result.package.candidates}
    assert all(by_id[item.candidate_id] == item for item in current.candidates)
    assert len({(item.category, item.intent) for item in result.package.candidates}) == 19
    assert (root / SLEEP1_INCREMENTAL_CALIBRATION_HISTORY_RELATIVE_PATH).exists()
    assert not (root / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH).exists()


def test_existing_target_pair_fails_closed_without_changing_files(tmp_path: Path) -> None:
    root = tmp_path / "private_data"
    current, decisions = _prepare(root, include_target=True)
    formal = root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    runtime = root / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    pending = root / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH
    before = formal.read_bytes(), runtime.read_bytes(), pending.read_bytes()

    with pytest.raises(PrivacyValidationError) as captured:
        _publish(root, current, decisions)

    assert captured.value.code == "sleep1_incremental_publish_failed"
    assert (formal.read_bytes(), runtime.read_bytes(), pending.read_bytes()) == before


def test_runtime_write_failure_restores_old_bytes_and_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private_data"
    current, decisions = _prepare(root)
    formal = root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    runtime = root / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    pending = root / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH
    before = formal.read_bytes(), runtime.read_bytes(), pending.read_bytes()
    module = __import__("companion.privacy.precise_intent_io", fromlist=["_atomic_write_model"])
    original = module._atomic_write_model

    def fail_runtime(model: object, *args: object, **kwargs: object):
        if type(model).__name__ == "CalibrationRuntimeHandoff":
            raise RuntimeError("SYNTHETIC_SLEEP1_PRIVATE_SENTINEL")
        return original(model, *args, **kwargs)

    monkeypatch.setattr(module, "_atomic_write_model", fail_runtime)
    with pytest.raises(PrivacyValidationError) as captured:
        _publish(root, current, decisions)

    assert captured.value.code == "sleep1_incremental_publish_failed"
    assert "SYNTHETIC_SLEEP1_PRIVATE_SENTINEL" not in str(captured.value)
    assert (formal.read_bytes(), runtime.read_bytes(), pending.read_bytes()) == before


def test_runtime_post_write_failure_restores_old_bytes_and_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private_data"
    current, decisions = _prepare(root)
    formal = root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    runtime = root / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    pending = root / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH
    before = formal.read_bytes(), runtime.read_bytes(), pending.read_bytes()
    module = __import__("companion.privacy.precise_intent_io", fromlist=["_atomic_write_model"])
    original = module._atomic_write_model

    def fail_after_runtime_write(model: object, *args: object, **kwargs: object):
        result = original(model, *args, **kwargs)
        if type(model).__name__ == "CalibrationRuntimeHandoff":
            raise RuntimeError(SENTINEL)
        return result

    monkeypatch.setattr(module, "_atomic_write_model", fail_after_runtime_write)
    with pytest.raises(PrivacyValidationError):
        _publish(root, current, decisions)

    assert (formal.read_bytes(), runtime.read_bytes(), pending.read_bytes()) == before
