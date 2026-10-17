from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from companion.privacy.calibration_models import (
    CalibrationCandidate,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CalibrationReviewDecision,
    CalibrationReviewDecisionValue,
    CalibrationReviewPackage,
    CalibrationReviewSubmission,
    calibration_candidate_id,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
    calibration_review_package_fingerprint,
    calibration_submission_id,
)
from companion.privacy.candidate_models import CandidateCategory
from companion.privacy.errors import PrivacyValidationError

NOW = datetime(2026, 7, 22, tzinfo=UTC)
IDENTITY_FINGERPRINT = "a" * 64
MALICIOUS_SENTINEL = "SYNTHETIC_MALICIOUS_IDENTITY_SENTINEL"


class _IterBomb(Sequence[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError(MALICIOUS_SENTINEL)

    def __getitem__(self, index: int) -> str:
        return "synthetic-name"

    def __len__(self) -> int:
        return 1


class _LenBomb(Sequence[str], Iterator[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        raise StopIteration

    def __getitem__(self, index: int) -> str:
        raise IndexError

    def __len__(self) -> int:
        raise RuntimeError(MALICIOUS_SENTINEL)


class _StripBomb(str):
    def strip(self, chars: str | None = None) -> str:
        raise RuntimeError(MALICIOUS_SENTINEL)


class _SafePrivacyFailureSequence(Sequence[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        raise PrivacyValidationError(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )

    def __getitem__(self, index: int) -> str:
        return "synthetic-name"

    def __len__(self) -> int:
        return 1


class _UnsafePrivacyFailureSequence(Sequence[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        raise PrivacyValidationError(
            "invalid_candidate_terms",
            MALICIOUS_SENTINEL,
        )

    def __getitem__(self, index: int) -> str:
        return "synthetic-name"

    def __len__(self) -> int:
        return 1


class _MutatedPrivacyFailureSequence(Sequence[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        error = PrivacyValidationError(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )
        error.code = []  # type: ignore[assignment]
        error.args = (_StripBomb(MALICIOUS_SENTINEL),)
        raise error

    def __getitem__(self, index: int) -> str:
        raise IndexError(index)

    def __len__(self) -> int:
        return 1


def _submission(
    *,
    scenario_id: str = "shopping-status",
    intent: str = "shopping_status",
    text: str = "先看看也行",
) -> CalibrationReviewSubmission:
    return CalibrationReviewSubmission(
        submission_id=calibration_submission_id(
            scenario_id=scenario_id,
            category=CandidateCategory.SHOPPING,
            intent=intent,
        ),
        scenario_id=scenario_id,
        category=CandidateCategory.SHOPPING,
        intent=intent,
        text=text,
        source="explicit_owner_ideal_response",
    )


def test_submission_is_strict_and_normalizes_only_outer_whitespace() -> None:
    submission = _submission(text="  先  看看也行  ")

    assert submission.text == "先  看看也行"
    with pytest.raises(ValidationError):
        CalibrationReviewSubmission.model_validate({**submission.model_dump(), "unknown": True})


@pytest.mark.parametrize("text", ["", "   ", "甲" * 161])
def test_submission_rejects_blank_or_overlong_text(text: str) -> None:
    with pytest.raises(ValidationError):
        _submission(text=text)


def test_submission_rejects_forged_source_and_unstable_id() -> None:
    valid = _submission()
    with pytest.raises(ValidationError):
        CalibrationReviewSubmission.model_validate(
            {**valid.model_dump(), "source": "model_response"}
        )
    with pytest.raises(ValidationError):
        CalibrationReviewSubmission.model_validate(
            {**valid.model_dump(), "submission_id": "0" * 64}
        )


def test_review_package_requires_unique_ids_consistent_count_and_fingerprint() -> None:
    submission = _submission()
    fingerprint = calibration_review_fingerprint([submission])
    package = CalibrationReviewPackage(
        submitted_at=NOW,
        content_fingerprint=fingerprint,
        submission_count=1,
        submissions=[submission],
    )
    assert package.submission_count == 1

    with pytest.raises(ValidationError):
        CalibrationReviewPackage.model_validate({**package.model_dump(), "submission_count": 2})
    with pytest.raises(ValidationError):
        CalibrationReviewPackage(
            submitted_at=NOW,
            content_fingerprint=fingerprint,
            submission_count=2,
            submissions=[submission, submission],
        )
    with pytest.raises(ValidationError):
        CalibrationReviewPackage(
            submitted_at=NOW,
            content_fingerprint="f" * 64,
            submission_count=1,
            submissions=[submission],
        )


def test_review_models_require_timezone_aware_timestamps() -> None:
    submission = _submission()
    with pytest.raises(ValidationError):
        CalibrationReviewPackage(
            submitted_at=datetime(2026, 7, 22),
            content_fingerprint=calibration_review_fingerprint([submission]),
            submission_count=1,
            submissions=[submission],
        )
    with pytest.raises(ValidationError):
        CalibrationCandidateAudit(
            reviewed_at=datetime(2026, 7, 22),
            submission_count=0,
            approved_count=0,
            rejected_count=0,
            privacy_rejected_count=0,
            output_count=0,
        )


def test_decision_contract_is_strict_and_uses_two_values() -> None:
    submission = _submission()
    decision = CalibrationReviewDecision(
        submission_id=submission.submission_id,
        submission_fingerprint=submission.content_fingerprint,
        review_package_fingerprint=calibration_review_package_fingerprint(
            [submission], submitted_at=NOW
        ),
        decided_at=NOW,
        decision=CalibrationReviewDecisionValue.APPROVE,
    )
    assert decision.decision is CalibrationReviewDecisionValue.APPROVE
    with pytest.raises(ValidationError):
        CalibrationReviewDecision.model_validate({**decision.model_dump(), "decision": "uncertain"})
    with pytest.raises(ValidationError):
        CalibrationReviewDecision.model_validate(
            {**decision.model_dump(), "decided_at": datetime(2026, 7, 22)}
        )


def test_identity_config_fingerprint_is_order_duplicate_and_whitespace_invariant() -> None:
    expected = calibration_identity_config_fingerprint(
        ("示例姓名甲", "示例姓名乙"),
        ("小团子", "小宝"),
    )

    assert len(expected) == 64
    assert expected == expected.lower()
    assert set(expected) <= set("0123456789abcdef")
    assert (
        calibration_identity_config_fingerprint(
            (" 示例姓名乙 ", "示例姓名甲", "示例姓名甲"),
            (" 小宝 ", "小团子", "小团子"),
        )
        == expected
    )


def test_identity_config_rejects_cross_domain_collision_after_normalization() -> None:
    with pytest.raises(PrivacyValidationError) as error:
        calibration_identity_config_fingerprint(
            (" 示例姓名甲 ",),
            ("示例姓名甲",),
        )

    assert error.value.code == "invalid_candidate_terms"


@pytest.mark.parametrize(
    ("true_name_terms", "approved_nicknames"),
    [
        (_IterBomb(), ()),
        (_LenBomb(), ()),
        ((_StripBomb(MALICIOUS_SENTINEL),), ()),
        (_SafePrivacyFailureSequence(), ()),
        (_UnsafePrivacyFailureSequence(), ()),
        (_MutatedPrivacyFailureSequence(), ()),
        (("synthetic-name",), _IterBomb()),
    ],
)
def test_identity_fingerprint_sanitizes_hostile_deep_validation_failures(
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(PrivacyValidationError) as captured:
        calibration_identity_config_fingerprint(
            true_name_terms,
            approved_nicknames,
        )

    error = captured.value
    assert error.code == "invalid_candidate_terms"
    assert str(error) == "invalid_candidate_terms: candidate terms are invalid"
    assert MALICIOUS_SENTINEL not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if "companion\\privacy" in traceback.tb_frame.f_code.co_filename.replace("/", "\\"):
            assert MALICIOUS_SENTINEL not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    output = capsys.readouterr()
    assert MALICIOUS_SENTINEL not in output.out
    assert MALICIOUS_SENTINEL not in output.err
    assert MALICIOUS_SENTINEL not in caplog.text


@pytest.mark.parametrize(
    ("true_names", "nicknames"),
    [
        (("示例姓名甲",), ("小团子", "小宝")),
        (("示例姓名甲", "示例姓名乙"), ("小团子",)),
        (("示例姓名甲", "示例姓名丙"), ("小团子", "小宝")),
        (("示例姓名甲", "小宝"), ("小团子", "示例姓名乙")),
    ],
)
def test_identity_config_fingerprint_changes_for_set_or_domain_changes(
    true_names: tuple[str, ...],
    nicknames: tuple[str, ...],
) -> None:
    baseline = calibration_identity_config_fingerprint(
        ("示例姓名甲", "示例姓名乙"),
        ("小团子", "小宝"),
    )

    assert calibration_identity_config_fingerprint(true_names, nicknames) != baseline


def test_candidate_package_requires_unique_ids_and_consistent_aggregates() -> None:
    candidate = CalibrationCandidate(
        candidate_id=calibration_candidate_id(
            category=CandidateCategory.SHOPPING,
            intent="shopping_status",
            text="先看看也行",
        ),
        category=CandidateCategory.SHOPPING,
        intent="shopping_status",
        text="先看看也行",
    )
    audit = CalibrationCandidateAudit(
        reviewed_at=NOW,
        submission_count=1,
        approved_count=1,
        rejected_count=0,
        privacy_rejected_count=0,
        output_count=1,
    )
    package = CalibrationCandidatePackage(
        created_at=NOW,
        content_fingerprint=calibration_review_fingerprint([candidate]),
        identity_config_fingerprint=IDENTITY_FINGERPRINT,
        candidate_count=1,
        candidates=[candidate],
        audit=audit,
    )
    assert package.candidate_count == 1

    with pytest.raises(ValidationError):
        CalibrationCandidatePackage(
            created_at=NOW,
            content_fingerprint=package.content_fingerprint,
            identity_config_fingerprint=IDENTITY_FINGERPRINT,
            candidate_count=2,
            candidates=[candidate, candidate],
            audit=audit.model_copy(
                update={"submission_count": 2, "approved_count": 2, "output_count": 2}
            ),
        )


@pytest.mark.parametrize(
    "identity_fingerprint",
    ["A" * 64, "g" * 64, "a" * 63, "a" * 65],
)
def test_candidate_package_rejects_invalid_identity_fingerprint(
    identity_fingerprint: str,
) -> None:
    candidate = CalibrationCandidate(
        candidate_id=calibration_candidate_id(
            category=CandidateCategory.SHOPPING,
            intent="shopping_status",
            text="先看看也行",
        ),
        category=CandidateCategory.SHOPPING,
        intent="shopping_status",
        text="先看看也行",
    )
    audit = CalibrationCandidateAudit(
        reviewed_at=NOW,
        submission_count=1,
        approved_count=1,
        rejected_count=0,
        privacy_rejected_count=0,
        output_count=1,
    )

    with pytest.raises(ValidationError):
        CalibrationCandidatePackage(
            created_at=NOW,
            content_fingerprint=calibration_review_fingerprint([candidate]),
            identity_config_fingerprint=identity_fingerprint,
            candidate_count=1,
            candidates=[candidate],
            audit=audit,
        )


def test_legacy_candidate_package_without_identity_fingerprint_fails_closed() -> None:
    candidate = CalibrationCandidate(
        candidate_id=calibration_candidate_id(
            category=CandidateCategory.SHOPPING,
            intent="shopping_status",
            text="先看看也行",
        ),
        category=CandidateCategory.SHOPPING,
        intent="shopping_status",
        text="先看看也行",
    )
    audit = CalibrationCandidateAudit(
        reviewed_at=NOW,
        submission_count=1,
        approved_count=1,
        rejected_count=0,
        privacy_rejected_count=0,
        output_count=1,
    )

    with pytest.raises(ValidationError):
        CalibrationCandidatePackage.model_validate(
            {
                "created_at": NOW,
                "content_fingerprint": calibration_review_fingerprint([candidate]),
                "candidate_count": 1,
                "candidates": [candidate.model_dump()],
                "audit": audit.model_dump(),
            }
        )
