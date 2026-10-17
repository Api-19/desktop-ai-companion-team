from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.privacy import (
    FIVE_FIXED_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH,
    FIVE_FIXED_INTENT_CALIBRATION_MAPPING,
    FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH,
    FIVE_FIXED_INTENT_CALIBRATION_PURPOSE,
    CalibrationReviewSubmission,
    CandidateCategory,
    FiveFixedIntentCalibrationHandoff,
    PrivacyValidationError,
    calibration_submission_id,
    consume_five_fixed_intent_calibration_handoff,
    write_five_fixed_intent_calibration_handoff,
)

NOW = datetime(2026, 8, 2, 9, 30, tzinfo=UTC)
SENTINEL = "SYNTHETIC_FIVE_FIXED_PRIVATE_SENTINEL"


class _HostileSequence(Sequence[CalibrationReviewSubmission]):
    def __len__(self) -> int:
        raise RuntimeError(SENTINEL)

    def __getitem__(self, index: int) -> CalibrationReviewSubmission:
        raise RuntimeError(SENTINEL)

    def __iter__(self):
        raise RuntimeError(SENTINEL)


def _submission(
    scenario_id: str, category: CandidateCategory, intent: str, index: int
) -> CalibrationReviewSubmission:
    return CalibrationReviewSubmission(
        submission_id=calibration_submission_id(
            scenario_id=scenario_id, category=category, intent=intent
        ),
        scenario_id=scenario_id,
        category=category,
        intent=intent,
        text=f"合成五项回复{index}",
        source="explicit_owner_ideal_response",
    )


def _submissions() -> list[CalibrationReviewSubmission]:
    return [
        _submission(scenario, category, intent, index)
        for index, (scenario, category, intent) in enumerate(
            FIVE_FIXED_INTENT_CALIBRATION_MAPPING, start=1
        )
    ]


def test_model_requires_exact_five_mapping_and_excludes_non_private_scenarios() -> None:
    handoff = FiveFixedIntentCalibrationHandoff(
        purpose=FIVE_FIXED_INTENT_CALIBRATION_PURPOSE,
        version=1,
        created_at=NOW,
        item_count=5,
        submissions=list(reversed(_submissions())),
    )
    assert handoff.item_count == 5
    assert {(item.scenario_id, item.category, item.intent) for item in handoff.submissions} == set(
        FIVE_FIXED_INTENT_CALIBRATION_MAPPING
    )
    assert all(item.scenario_id not in {"weather-1", "shopping-2"} for item in handoff.submissions)

    for lure_index in range(50):
        lure = _submission(
            f"lure-{lure_index}", CandidateCategory.WEATHER, f"lure_{lure_index}", lure_index
        )
        with pytest.raises(ValidationError):
            FiveFixedIntentCalibrationHandoff(
                purpose=FIVE_FIXED_INTENT_CALIBRATION_PURPOSE,
                version=1,
                created_at=NOW,
                item_count=5,
                submissions=[lure, *_submissions()[1:]],
            )


def test_write_and_consume_use_fixed_paths_and_retire_intake(tmp_path: Path) -> None:
    private_root = tmp_path / "private_data"
    intake = write_five_fixed_intent_calibration_handoff(
        _submissions(), created_at=NOW, private_data_dir=private_root
    )
    assert intake == private_root / FIVE_FIXED_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH

    pending = consume_five_fixed_intent_calibration_handoff(private_data_dir=private_root)

    assert pending.submission_count == 5
    assert not intake.exists()
    assert (private_root / FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH).exists()


def test_failed_consume_preserves_intake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    private_root = tmp_path / "private_data"
    intake = write_five_fixed_intent_calibration_handoff(
        _submissions(), created_at=NOW, private_data_dir=private_root
    )

    def fail_write(*args: object, **kwargs: object) -> Path:
        raise OSError("SYNTHETIC_PRIVATE_SENTINEL")

    monkeypatch.setattr("companion.privacy.precise_intent_io._atomic_write_model", fail_write)
    with pytest.raises(PrivacyValidationError) as captured:
        consume_five_fixed_intent_calibration_handoff(private_data_dir=private_root)

    assert captured.value.code == "five_fixed_intent_pending_failed"
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in str(captured.value)
    assert intake.exists()
    assert not (private_root / FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH).exists()


def test_write_sanitizes_hostile_sequence(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(PrivacyValidationError) as captured:
        write_five_fixed_intent_calibration_handoff(
            _HostileSequence(), created_at=NOW, private_data_dir=tmp_path / "private_data"
        )

    error = captured.value
    assert error.code == "invalid_five_fixed_intent_handoff"
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename:
            assert SENTINEL not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    captured_io = capsys.readouterr()
    assert SENTINEL not in captured_io.out
    assert SENTINEL not in captured_io.err
    assert SENTINEL not in caplog.text
