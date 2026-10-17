from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.privacy import (
    PRECISE_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH,
    PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH,
    PRECISE_INTENT_CALIBRATION_PURPOSE,
    CalibrationReviewPackage,
    CalibrationReviewSubmission,
    CandidateCategory,
    PreciseIntentCalibrationHandoff,
    PrivacyValidationError,
    calibration_submission_id,
    consume_precise_intent_calibration_handoff,
    write_precise_intent_calibration_handoff,
)

NOW = datetime(2026, 8, 1, 9, 30, tzinfo=UTC)
HOSTILE_SENTINEL = "SYNTHETIC_PRECISE_INTENT_PRIVATE_SENTINEL"
MAPPING = (
    ("greeting-1", CandidateCategory.GREETING, "morning_greeting"),
    ("meals-1", CandidateCategory.MEALS, "missed_meal_care"),
    (
        "household-2",
        CandidateCategory.HOUSEHOLD,
        "unfinished_household_reassurance",
    ),
    (
        "shopping-1",
        CandidateCategory.SHOPPING,
        "shopping_comparison_request",
    ),
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


class _HostileSequence(Sequence[CalibrationReviewSubmission]):
    def __iter__(self) -> Iterator[CalibrationReviewSubmission]:
        raise RuntimeError(HOSTILE_SENTINEL)

    def __len__(self) -> int:
        raise RuntimeError(HOSTILE_SENTINEL)

    def __getitem__(self, index: int) -> CalibrationReviewSubmission:
        raise RuntimeError(HOSTILE_SENTINEL)

    def __repr__(self) -> str:
        return HOSTILE_SENTINEL


class _HostileSubmission(CalibrationReviewSubmission):
    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError(HOSTILE_SENTINEL)


class _HostilePrivateRoot:
    def __truediv__(self, value: object) -> object:
        raise RuntimeError(HOSTILE_SENTINEL)

    def __repr__(self) -> str:
        return HOSTILE_SENTINEL


def _assert_sanitized(
    error: PrivacyValidationError,
    *,
    expected_code: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert error.code == expected_code
    assert HOSTILE_SENTINEL not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename:
            assert HOSTILE_SENTINEL not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    captured = capsys.readouterr()
    assert HOSTILE_SENTINEL not in captured.out
    assert HOSTILE_SENTINEL not in captured.err
    assert HOSTILE_SENTINEL not in caplog.text


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
            text=f"合成理想回复{index}",
            source="explicit_owner_ideal_response",
        )
        for index, (scenario_id, category, intent) in enumerate(MAPPING, start=1)
    ]


def _handoff_payload() -> dict[str, object]:
    return {
        "purpose": PRECISE_INTENT_CALIBRATION_PURPOSE,
        "version": 1,
        "created_at": NOW,
        "item_count": 8,
        "submissions": [item.model_dump(mode="json") for item in _submissions()],
    }


def test_handoff_requires_the_complete_exact_public_mapping() -> None:
    handoff = PreciseIntentCalibrationHandoff.model_validate(_handoff_payload())
    assert handoff.item_count == 8
    assert {(item.scenario_id, item.category, item.intent) for item in handoff.submissions} == set(
        MAPPING
    )

    missing = _handoff_payload()
    missing["submissions"] = missing["submissions"][:-1]  # type: ignore[index]
    missing["item_count"] = 7
    with pytest.raises(ValidationError):
        PreciseIntentCalibrationHandoff.model_validate(missing)

    wrong = _handoff_payload()
    wrong["submissions"][0]["intent"] = "nearby_but_wrong"  # type: ignore[index]
    with pytest.raises(ValidationError):
        PreciseIntentCalibrationHandoff.model_validate(wrong)


def test_handoff_rejects_unknown_fields_naive_timestamps_and_wrong_source() -> None:
    extra = _handoff_payload() | {"private_note": "synthetic-private-sentinel"}
    with pytest.raises(ValidationError):
        PreciseIntentCalibrationHandoff.model_validate(extra)

    naive = _handoff_payload() | {"created_at": datetime(2026, 8, 1, 9, 30)}
    with pytest.raises(ValidationError):
        PreciseIntentCalibrationHandoff.model_validate(naive)

    wrong_source = _handoff_payload()
    wrong_source["submissions"][0]["source"] = "model_response"  # type: ignore[index]
    with pytest.raises(ValidationError):
        PreciseIntentCalibrationHandoff.model_validate(wrong_source)


def test_write_and_consume_use_fixed_private_paths_and_retire_intake(tmp_path: Path) -> None:
    private_root = tmp_path / "private_data"
    written = write_precise_intent_calibration_handoff(
        _submissions(),
        created_at=NOW,
        private_data_dir=private_root,
    )
    assert written == private_root / PRECISE_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH
    assert written.exists()

    pending = consume_precise_intent_calibration_handoff(private_data_dir=private_root)

    assert isinstance(pending, CalibrationReviewPackage)
    assert pending.submission_count == 8
    assert not written.exists()
    assert (private_root / PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH).exists()

    with pytest.raises(PrivacyValidationError):
        consume_precise_intent_calibration_handoff(private_data_dir=private_root)


def test_failed_pending_generation_keeps_intake_and_produces_no_half_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private_data"
    intake = write_precise_intent_calibration_handoff(
        _submissions(),
        created_at=NOW,
        private_data_dir=private_root,
    )

    def fail_write(*args: object, **kwargs: object) -> Path:
        raise OSError("synthetic-private-sentinel")

    monkeypatch.setattr(
        "companion.privacy.precise_intent_io._atomic_write_model",
        fail_write,
    )
    with pytest.raises(PrivacyValidationError) as captured:
        consume_precise_intent_calibration_handoff(private_data_dir=private_root)

    assert captured.value.code == "precise_intent_pending_failed"
    assert "synthetic-private-sentinel" not in str(captured.value)
    assert intake.exists()
    assert not (private_root / PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH).exists()


@pytest.mark.parametrize("attack", ["sequence", "model_dump", "path"])
def test_write_sanitizes_hostile_deep_inputs(
    tmp_path: Path,
    attack: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions: object = _submissions()
    private_root: object = tmp_path / "private_data"
    if attack == "sequence":
        submissions = _HostileSequence()
    elif attack == "model_dump":
        submissions = [
            _HostileSubmission.model_validate(_submissions()[0].model_dump()),
            *_submissions()[1:],
        ]
    else:
        private_root = _HostilePrivateRoot()

    with pytest.raises(PrivacyValidationError) as captured:
        write_precise_intent_calibration_handoff(
            submissions,  # type: ignore[arg-type]
            created_at=NOW,
            private_data_dir=private_root,  # type: ignore[arg-type]
        )

    _assert_sanitized(
        captured.value,
        expected_code="invalid_precise_intent_handoff",
        caplog=caplog,
        capsys=capsys,
    )
