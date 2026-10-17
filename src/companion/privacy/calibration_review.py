from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import ValidationError

from companion.contracts.base import TimezoneRequiredModel
from companion.privacy.calibration_models import (
    CalibrationCandidate,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CalibrationReviewDecision,
    CalibrationReviewDecisionValue,
    CalibrationReviewSubmission,
    calibration_candidate_id,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
    calibration_review_package_fingerprint,
)
from companion.privacy.candidate_io import _CandidateIoFailure, _validated_runtime_terms
from companion.privacy.candidate_rules import hard_rejection_reason
from companion.privacy.errors import (
    PrivacyValidationError,
    _PrivacyFailure,
    _sanitized_public_privacy_failure,
)

_CALIBRATION_REVIEW_FAILURE_DETAILS = {
    "calibration_privacy_rejected": ("calibration review batch failed privacy verification",),
    "incomplete_calibration_decisions": (
        "calibration decisions must cover every submission exactly once",
    ),
    "invalid_calibration_decisions": ("calibration decisions are invalid",),
    "invalid_calibration_submission": ("calibration submissions are invalid",),
    "invalid_candidate_terms": ("candidate terms are invalid",),
    "stale_calibration_decision": ("calibration decision does not match its submission",),
}
_BUILD_FALLBACK = (
    "invalid_calibration_submission",
    "calibration submissions are invalid",
)


class _CalibrationSubmittedAt(TimezoneRequiredModel):
    submitted_at: datetime


def _validated_submissions(
    submissions: Sequence[CalibrationReviewSubmission],
) -> list[CalibrationReviewSubmission]:
    if isinstance(submissions, (str, bytes)) or not isinstance(submissions, Sequence):
        raise _PrivacyFailure(
            "invalid_calibration_submission",
            "calibration submissions are invalid",
        )
    try:
        values = [
            CalibrationReviewSubmission.model_validate(submission.model_dump())
            for submission in submissions
        ]
    except (AttributeError, ValidationError, ValueError):
        raise _PrivacyFailure(
            "invalid_calibration_submission",
            "calibration submissions are invalid",
        ) from None
    submission_ids = [submission.submission_id for submission in values]
    if not values or len(submission_ids) != len(set(submission_ids)):
        raise _PrivacyFailure(
            "invalid_calibration_submission",
            "calibration submissions are invalid",
        )
    return values


def _validated_decisions(
    decisions: Sequence[CalibrationReviewDecision],
) -> list[CalibrationReviewDecision]:
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise _PrivacyFailure(
            "invalid_calibration_decisions",
            "calibration decisions are invalid",
        )
    try:
        values = [
            CalibrationReviewDecision.model_validate(decision.model_dump())
            for decision in decisions
        ]
    except (AttributeError, ValidationError, ValueError):
        raise _PrivacyFailure(
            "invalid_calibration_decisions",
            "calibration decisions are invalid",
        ) from None
    decision_ids = [decision.submission_id for decision in values]
    if len(decision_ids) != len(set(decision_ids)):
        raise _PrivacyFailure(
            "invalid_calibration_decisions",
            "calibration decisions are invalid",
        )
    return values


def _validate_decision_set(
    submissions: list[CalibrationReviewSubmission],
    decisions: list[CalibrationReviewDecision],
    *,
    submitted_at: datetime,
) -> dict[str, CalibrationReviewDecision]:
    expected_ids = {submission.submission_id for submission in submissions}
    actual_ids = {decision.submission_id for decision in decisions}
    if actual_ids != expected_ids:
        raise _PrivacyFailure(
            "incomplete_calibration_decisions",
            "calibration decisions must cover every submission exactly once",
        )
    by_id = {decision.submission_id: decision for decision in decisions}
    expected_package_fingerprint = calibration_review_package_fingerprint(
        submissions, submitted_at=submitted_at
    )
    for submission in submissions:
        decision = by_id[submission.submission_id]
        if (
            decision.submission_fingerprint != submission.content_fingerprint
            or decision.review_package_fingerprint != expected_package_fingerprint
            or decision.decided_at < submitted_at
        ):
            raise _PrivacyFailure(
                "stale_calibration_decision",
                "calibration decision does not match its submission",
            )
    return by_id


def _build_calibration_candidate_package(
    submissions: Sequence[CalibrationReviewSubmission],
    decisions: Sequence[CalibrationReviewDecision],
    *,
    submitted_at: datetime,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    submitted_at = _CalibrationSubmittedAt.model_validate(
        {"submitted_at": submitted_at}
    ).submitted_at
    validated_submissions = _validated_submissions(submissions)
    validated_decisions = _validated_decisions(decisions)
    decisions_by_id = _validate_decision_set(
        validated_submissions,
        validated_decisions,
        submitted_at=submitted_at,
    )
    try:
        names, nicknames = _validated_runtime_terms(
            true_name_terms,
            approved_nicknames,
        )
    except _CandidateIoFailure as error:
        raise _PrivacyFailure(error.code, error.detail) from None

    candidates: list[CalibrationCandidate] = []
    rejected_count = 0
    for submission in validated_submissions:
        decision = decisions_by_id[submission.submission_id]
        if decision.decision is CalibrationReviewDecisionValue.REJECT:
            rejected_count += 1
            continue
        if hard_rejection_reason(submission.text) is not None or any(
            name in submission.text for name in names
        ):
            raise _PrivacyFailure(
                "calibration_privacy_rejected",
                "calibration review batch failed privacy verification",
            )
        candidates.append(
            CalibrationCandidate(
                candidate_id=calibration_candidate_id(
                    category=submission.category,
                    intent=submission.intent,
                    text=submission.text,
                ),
                category=submission.category,
                intent=submission.intent,
                text=submission.text,
            )
        )

    candidates.sort(key=lambda candidate: candidate.candidate_id)
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise _PrivacyFailure(
            "invalid_calibration_submission",
            "calibration submissions are invalid",
        )
    approved_count = len(candidates)
    audit = CalibrationCandidateAudit(
        reviewed_at=submitted_at,
        submission_count=len(validated_submissions),
        approved_count=approved_count,
        rejected_count=rejected_count,
        privacy_rejected_count=0,
        revoked_count=0,
        output_count=approved_count,
    )
    return CalibrationCandidatePackage(
        created_at=submitted_at,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=calibration_identity_config_fingerprint(
            names,
            nicknames,
        ),
        candidate_count=approved_count,
        candidates=candidates,
        audit=audit,
    )


def build_calibration_candidate_package(
    submissions: Sequence[CalibrationReviewSubmission],
    decisions: Sequence[CalibrationReviewDecision],
    *,
    submitted_at: datetime,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    try:
        return _build_calibration_candidate_package(
            submissions,
            decisions,
            submitted_at=submitted_at,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PrivacyFailure,
            allowed_details=_CALIBRATION_REVIEW_FAILURE_DETAILS,
            fallback=_BUILD_FALLBACK,
        )
    del submissions, decisions, submitted_at, true_name_terms, approved_nicknames
    raise PrivacyValidationError(failure[0], failure[1]) from None
