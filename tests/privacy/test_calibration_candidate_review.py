from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from companion.privacy import PrivacyValidationError
from companion.privacy.calibration_models import (
    CalibrationReviewDecision,
    CalibrationReviewDecisionValue,
    CalibrationReviewSubmission,
    calibration_identity_config_fingerprint,
    calibration_review_package_fingerprint,
    calibration_submission_id,
)
from companion.privacy.calibration_review import build_calibration_candidate_package
from companion.privacy.candidate_models import CandidateCategory

NOW = datetime(2026, 7, 22, tzinfo=UTC)
HOSTILE_SENTINEL = "SYNTHETIC_HOSTILE_BUILD_SENTINEL"


class _HostileTerms(Sequence[str]):
    def __init__(self, attack: str) -> None:
        self.attack = attack

    def __iter__(self) -> Iterator[str]:
        if self.attack == "privacy":
            raise PrivacyValidationError("invalid_calibration_submission", HOSTILE_SENTINEL)
        if self.attack == "iter":
            raise RuntimeError(HOSTILE_SENTINEL)
        return super().__iter__()

    def __len__(self) -> int:
        if self.attack == "len":
            raise RuntimeError(HOSTILE_SENTINEL)
        return 1

    def __getitem__(self, index: int) -> str:
        if index:
            raise IndexError(index)
        if self.attack == "strip":
            return _HostileStrip(HOSTILE_SENTINEL)
        return "synthetic-name"

    def __repr__(self) -> str:
        return HOSTILE_SENTINEL


class _HostileStrip(str):
    def strip(self, *args: object, **kwargs: object) -> str:
        raise RuntimeError(HOSTILE_SENTINEL)


def _assert_public_failure_is_sanitized(
    error: PrivacyValidationError,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert error.code == "invalid_calibration_submission"
    assert str(error) == ("invalid_calibration_submission: calibration submissions are invalid")
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


def _submission(
    *,
    scenario_id: str,
    text: str,
    source: str = "explicit_owner_ideal_response",
) -> CalibrationReviewSubmission:
    category = CandidateCategory.SHOPPING
    intent = f"intent_{scenario_id}"
    return CalibrationReviewSubmission.model_validate(
        {
            "submission_id": calibration_submission_id(
                scenario_id=scenario_id,
                category=category,
                intent=intent,
            ),
            "scenario_id": scenario_id,
            "category": category,
            "intent": intent,
            "text": text,
            "source": source,
        }
    )


def _decision(
    submission: CalibrationReviewSubmission,
    value: CalibrationReviewDecisionValue,
    *,
    review_submissions: list[CalibrationReviewSubmission] | None = None,
    submitted_at: datetime = NOW,
    decided_at: datetime = NOW,
) -> CalibrationReviewDecision:
    return CalibrationReviewDecision(
        submission_id=submission.submission_id,
        submission_fingerprint=submission.content_fingerprint,
        review_package_fingerprint=calibration_review_package_fingerprint(
            review_submissions or [submission], submitted_at=submitted_at
        ),
        decided_at=decided_at,
        decision=value,
    )


def _build(
    submissions: list[CalibrationReviewSubmission],
    decisions: list[CalibrationReviewDecision],
):
    return build_calibration_candidate_package(
        submissions,
        decisions,
        submitted_at=NOW,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=("小团子",),
    )


@pytest.mark.parametrize("attack", ["iter", "len", "strip", "privacy"])
def test_build_sanitizes_all_hostile_deep_validation_failures(
    attack: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission = _submission(scenario_id="hostile", text="先看看")
    decision = _decision(submission, CalibrationReviewDecisionValue.APPROVE)

    with pytest.raises(PrivacyValidationError) as captured:
        build_calibration_candidate_package(
            [submission],
            [decision],
            submitted_at=NOW,
            true_name_terms=_HostileTerms(attack),
            approved_nicknames=(),
        )

    _assert_public_failure_is_sanitized(captured.value, caplog, capsys)


def test_decisions_bind_the_exact_review_batch_and_cannot_precede_it() -> None:
    submission = _submission(scenario_id="one", text="先看看也行")
    decision = _decision(submission, CalibrationReviewDecisionValue.APPROVE)

    with pytest.raises(PrivacyValidationError) as replay_error:
        build_calibration_candidate_package(
            [submission],
            [decision],
            submitted_at=NOW.replace(minute=1),
            true_name_terms=(),
            approved_nicknames=(),
        )
    assert replay_error.value.code == "stale_calibration_decision"

    early = _decision(
        submission,
        CalibrationReviewDecisionValue.APPROVE,
        decided_at=NOW - timedelta(seconds=1),
    )
    with pytest.raises(PrivacyValidationError) as early_error:
        _build([submission], [early])
    assert early_error.value.code == "stale_calibration_decision"


def test_builds_only_explicitly_approved_safe_owner_ideal_responses() -> None:
    approved = _submission(scenario_id="one", text="先看看也行")
    rejected = _submission(scenario_id="two", text="我觉得可以再等等")

    package = _build(
        [approved, rejected],
        [
            _decision(
                approved,
                CalibrationReviewDecisionValue.APPROVE,
                review_submissions=[approved, rejected],
            ),
            _decision(
                rejected,
                CalibrationReviewDecisionValue.REJECT,
                review_submissions=[approved, rejected],
            ),
        ],
    )

    assert package.candidate_count == 1
    assert package.candidates[0].text == "先看看也行"
    assert package.candidates[0].intent == approved.intent
    assert package.audit.submission_count == 2
    assert package.audit.approved_count == 1
    assert package.audit.rejected_count == 1
    assert package.audit.privacy_rejected_count == 0
    assert package.identity_config_fingerprint == calibration_identity_config_fingerprint(
        ("示例姓名甲",),
        ("小团子",),
    )


@pytest.mark.parametrize("mode", ["missing", "duplicate", "stale", "extra"])
def test_decisions_must_be_complete_unique_and_fresh(mode: str) -> None:
    submission = _submission(scenario_id="one", text="先看看也行")
    decision = _decision(submission, CalibrationReviewDecisionValue.APPROVE)
    decisions = [decision]
    if mode == "missing":
        decisions = []
    elif mode == "duplicate":
        decisions = [decision, decision]
    elif mode == "stale":
        decisions = [decision.model_copy(update={"submission_fingerprint": "f" * 64})]
    else:
        other = _submission(scenario_id="two", text="等等也行")
        decisions = [decision, _decision(other, CalibrationReviewDecisionValue.REJECT)]

    with pytest.raises(PrivacyValidationError) as error:
        _build([submission], decisions)

    assert error.value.code in {
        "incomplete_calibration_decisions",
        "invalid_calibration_decisions",
        "stale_calibration_decision",
    }


@pytest.mark.parametrize(
    "text",
    [
        "密码是合成值",
        "联系示例姓名甲",
        "今晚九点见",
        "这是私人秘密",
    ],
)
def test_manual_approval_cannot_override_privacy_gates(text: str) -> None:
    submission = _submission(scenario_id="risk", text=text)

    with pytest.raises(PrivacyValidationError) as error:
        _build(
            [submission],
            [_decision(submission, CalibrationReviewDecisionValue.APPROVE)],
        )

    assert error.value.code == "calibration_privacy_rejected"


def test_whitespace_normalized_true_name_still_rejects_approved_submission() -> None:
    sensitive_name = "仅用于合成规范化门禁测试的姓名"
    submission = _submission(
        scenario_id="normalized-name",
        text=f"联系{sensitive_name}",
    )

    with pytest.raises(PrivacyValidationError) as error:
        build_calibration_candidate_package(
            [submission],
            [_decision(submission, CalibrationReviewDecisionValue.APPROVE)],
            submitted_at=NOW,
            true_name_terms=(f"  {sensitive_name}  ",),
            approved_nicknames=(),
        )

    assert error.value.code == "calibration_privacy_rejected"


def test_one_privacy_rejected_approval_fails_the_entire_batch_without_leakage() -> None:
    safe = _submission(scenario_id="safe", text="先看看也行")
    sensitive_text = "联系仅用于合成整批失败测试的姓名"
    risky = _submission(scenario_id="risk", text=sensitive_text)
    submissions = [safe, risky]
    decisions = [
        _decision(
            submission,
            CalibrationReviewDecisionValue.APPROVE,
            review_submissions=submissions,
        )
        for submission in submissions
    ]

    try:
        build_calibration_candidate_package(
            submissions,
            decisions,
            submitted_at=NOW,
            true_name_terms=("仅用于合成整批失败测试的姓名",),
            approved_nicknames=(),
        )
    except PrivacyValidationError as error:
        assert error.code == "calibration_privacy_rejected"
        assert sensitive_text not in str(error)
        traceback = error.__traceback__
        while traceback is not None:
            filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
            if "companion\\privacy" in filename:
                assert sensitive_text not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next
    else:
        raise AssertionError("expected whole-batch privacy failure")


def test_deep_revalidation_rejects_forged_source_objects() -> None:
    valid = _submission(scenario_id="one", text="先看看也行")
    forged = CalibrationReviewSubmission.model_construct(
        **{**valid.model_dump(), "source": "model_response"}
    )

    with pytest.raises(PrivacyValidationError) as error:
        _build(
            [forged],
            [_decision(valid, CalibrationReviewDecisionValue.APPROVE)],
        )

    assert error.value.code == "invalid_calibration_submission"


def test_public_failure_does_not_echo_or_retain_sensitive_payloads() -> None:
    sensitive_text = "仅用于合成泄漏测试的回复"
    sensitive_name = "仅用于合成泄漏测试的姓名"
    submission = _submission(scenario_id="one", text=sensitive_text)
    stale = _decision(submission, CalibrationReviewDecisionValue.APPROVE).model_copy(
        update={"submission_fingerprint": "f" * 64}
    )

    try:
        build_calibration_candidate_package(
            [submission],
            [stale],
            submitted_at=NOW,
            true_name_terms=(sensitive_name,),
            approved_nicknames=(),
        )
    except PrivacyValidationError as error:
        assert sensitive_text not in str(error)
        assert sensitive_name not in str(error)
        traceback = error.__traceback__
        while traceback is not None:
            filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
            if "companion\\privacy" in filename:
                frame_values = repr(traceback.tb_frame.f_locals)
                assert sensitive_text not in frame_values
                assert sensitive_name not in frame_values
            traceback = traceback.tb_next
    else:
        raise AssertionError("expected privacy validation failure")


def test_naive_submitted_at_fails_without_retaining_sensitive_payloads() -> None:
    sensitive_text = "仅用于合成无时区泄漏测试的回复"
    submission = _submission(scenario_id="one", text=sensitive_text)
    naive_submitted_at = NOW.replace(tzinfo=None)
    decision = _decision(
        submission,
        CalibrationReviewDecisionValue.APPROVE,
        submitted_at=naive_submitted_at,
    )

    try:
        build_calibration_candidate_package(
            [submission],
            [decision],
            submitted_at=naive_submitted_at,
            true_name_terms=(),
            approved_nicknames=(),
        )
    except PrivacyValidationError as error:
        assert sensitive_text not in str(error)
        traceback = error.__traceback__
        while traceback is not None:
            filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
            if "companion\\privacy" in filename:
                assert sensitive_text not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next
    else:
        raise AssertionError("expected privacy validation failure")
