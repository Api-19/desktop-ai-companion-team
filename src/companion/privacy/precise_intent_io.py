from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from companion.privacy.calibration_io import (
    _atomic_write_model,
    _safe_private_path,
    load_calibration_candidate_package,
    publish_calibration_candidate_package,
)
from companion.privacy.calibration_models import (
    FIVE_FIXED_INTENT_CALIBRATION_PURPOSE,
    PRECISE_INTENT_CALIBRATION_PURPOSE,
    SLEEP1_INCREMENTAL_CALIBRATION_PURPOSE,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CalibrationReviewDecision,
    CalibrationReviewPackage,
    CalibrationReviewSubmission,
    CalibrationRuntimeHandoff,
    FiveFixedIntentCalibrationHandoff,
    FiveFixedIntentCalibrationPublishResult,
    PreciseIntentCalibrationHandoff,
    Sleep1IncrementalCalibrationHandoff,
    Sleep1IncrementalCalibrationPublishResult,
    calibration_review_fingerprint,
)
from companion.privacy.calibration_review import build_calibration_candidate_package
from companion.privacy.candidate_models import CandidateCategory
from companion.privacy.errors import PrivacyValidationError, _sanitized_public_privacy_failure

PRECISE_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH = Path(
    "style_candidates/calibration/intake/precise-intent-submissions.json"
)
PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH = Path(
    "style_candidates/calibration/pending/precise-intent-review-package.json"
)
PRECISE_INTENT_CALIBRATION_HISTORY_RELATIVE_PATH = Path(
    "style_candidates/calibration/history/calibration-candidates-before-precise-intent.json"
)
CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH = Path("handoff/calibration-runtime.json")
FIVE_FIXED_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH = Path(
    "style_candidates/calibration/intake/five-fixed-intent-submissions.json"
)
FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH = Path(
    "style_candidates/calibration/pending/five-fixed-intent-review-package.json"
)
FIVE_FIXED_INTENT_CALIBRATION_HISTORY_RELATIVE_PATH = Path(
    "style_candidates/calibration/history/calibration-candidates-before-five-fixed-intent.json"
)
SLEEP1_INCREMENTAL_CALIBRATION_INTAKE_RELATIVE_PATH = Path(
    "style_candidates/calibration/intake/sleep1-incremental-submission.json"
)
SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH = Path(
    "style_candidates/calibration/pending/sleep1-incremental-review-package.json"
)
SLEEP1_INCREMENTAL_CALIBRATION_HISTORY_RELATIVE_PATH = Path(
    "style_candidates/calibration/history/calibration-candidates-before-sleep1-incremental.json"
)


class _PreciseIntentFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


_FAILURE_DETAILS = {
    "invalid_precise_intent_handoff": ("precise intent calibration handoff is invalid",),
    "precise_intent_handoff_missing": ("precise intent calibration handoff is missing",),
    "precise_intent_pending_failed": ("precise intent pending review could not be created",),
    "precise_intent_publish_failed": ("precise intent calibration batch could not be published",),
}
_WRITE_FALLBACK = (
    "invalid_precise_intent_handoff",
    "precise intent calibration handoff is invalid",
)
_CONSUME_FALLBACK = (
    "precise_intent_pending_failed",
    "precise intent pending review could not be created",
)
_PUBLISH_FALLBACK = (
    "precise_intent_publish_failed",
    "precise intent calibration batch could not be published",
)
_FIVE_FAILURE_DETAILS = {
    "invalid_five_fixed_intent_handoff": ("five fixed intent calibration handoff is invalid",),
    "five_fixed_intent_handoff_missing": ("five fixed intent calibration handoff is missing",),
    "five_fixed_intent_pending_failed": ("five fixed intent pending review could not be created",),
    "five_fixed_intent_publish_failed": (
        "five fixed intent calibration batch could not be published",
    ),
}
_FIVE_WRITE_FALLBACK = (
    "invalid_five_fixed_intent_handoff",
    "five fixed intent calibration handoff is invalid",
)
_FIVE_CONSUME_FALLBACK = (
    "five_fixed_intent_pending_failed",
    "five fixed intent pending review could not be created",
)
_FIVE_PUBLISH_FALLBACK = (
    "five_fixed_intent_publish_failed",
    "five fixed intent calibration batch could not be published",
)
_SLEEP1_FAILURE_DETAILS = {
    "invalid_sleep1_incremental_handoff": ("sleep-1 incremental handoff is invalid",),
    "sleep1_incremental_handoff_missing": ("sleep-1 incremental handoff is missing",),
    "sleep1_incremental_pending_failed": ("sleep-1 pending review could not be created",),
    "sleep1_incremental_publish_failed": ("sleep-1 calibration could not be published",),
}
_SLEEP1_WRITE_FALLBACK = (
    "invalid_sleep1_incremental_handoff",
    "sleep-1 incremental handoff is invalid",
)
_SLEEP1_CONSUME_FALLBACK = (
    "sleep1_incremental_pending_failed",
    "sleep-1 pending review could not be created",
)
_SLEEP1_PUBLISH_FALLBACK = (
    "sleep1_incremental_publish_failed",
    "sleep-1 calibration could not be published",
)


def _deep_submissions(
    submissions: Sequence[CalibrationReviewSubmission],
) -> list[CalibrationReviewSubmission]:
    if isinstance(submissions, (str, bytes)) or not isinstance(submissions, Sequence):
        raise _PreciseIntentFailure(*_WRITE_FALLBACK)
    try:
        return [
            CalibrationReviewSubmission.model_validate(item.model_dump(mode="json"))
            for item in submissions
        ]
    except Exception:
        raise _PreciseIntentFailure(*_WRITE_FALLBACK) from None


def _write_precise_intent_calibration_handoff(
    submissions: Sequence[CalibrationReviewSubmission],
    created_at: datetime,
    private_data_dir: Path,
) -> Path:
    handoff = PreciseIntentCalibrationHandoff(
        purpose=PRECISE_INTENT_CALIBRATION_PURPOSE,
        version=1,
        created_at=created_at,
        item_count=8,
        submissions=_deep_submissions(submissions),
    )
    return _atomic_write_model(
        handoff,
        private_data_dir / PRECISE_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH,
        private_data_dir,
        PreciseIntentCalibrationHandoff,
    )


def write_precise_intent_calibration_handoff(
    submissions: Sequence[CalibrationReviewSubmission],
    *,
    created_at: datetime,
    private_data_dir: Path,
) -> Path:
    try:
        return _write_precise_intent_calibration_handoff(
            submissions,
            created_at,
            private_data_dir,
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_FAILURE_DETAILS,
            fallback=_WRITE_FALLBACK,
        )
    del submissions, created_at, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _read_handoff(path: Path, private_data_dir: Path) -> PreciseIntentCalibrationHandoff:
    resolved = _safe_private_path(path, private_data_dir, include_leaf=True)
    try:
        return PreciseIntentCalibrationHandoff.model_validate_json(resolved.read_bytes())
    except FileNotFoundError:
        raise _PreciseIntentFailure(
            "precise_intent_handoff_missing",
            "precise intent calibration handoff is missing",
        ) from None
    except (OSError, UnicodeError, ValidationError, ValueError, json.JSONDecodeError):
        raise _PreciseIntentFailure(*_WRITE_FALLBACK) from None


def _consume_precise_intent_calibration_handoff(
    private_data_dir: Path,
) -> CalibrationReviewPackage:
    intake = private_data_dir / PRECISE_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH
    pending_path = private_data_dir / PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH
    resolved_pending = _safe_private_path(
        pending_path,
        private_data_dir,
        include_leaf=pending_path.exists(),
    )
    if resolved_pending.exists():
        raise _PreciseIntentFailure(*_CONSUME_FALLBACK)
    handoff = _read_handoff(intake, private_data_dir)
    package = CalibrationReviewPackage(
        submitted_at=handoff.created_at,
        content_fingerprint=calibration_review_fingerprint(handoff.submissions),
        submission_count=handoff.item_count,
        submissions=handoff.submissions,
    )
    try:
        _atomic_write_model(
            package,
            pending_path,
            private_data_dir,
            CalibrationReviewPackage,
        )
        verified = CalibrationReviewPackage.model_validate_json(resolved_pending.read_bytes())
        if verified != package:
            raise _PreciseIntentFailure(*_CONSUME_FALLBACK)
        resolved_intake = _safe_private_path(intake, private_data_dir, include_leaf=True)
        resolved_intake.unlink()
        return verified
    except Exception:
        with suppress(OSError):
            resolved_pending.unlink(missing_ok=True)
        raise _PreciseIntentFailure(*_CONSUME_FALLBACK) from None


def consume_precise_intent_calibration_handoff(
    *, private_data_dir: Path
) -> CalibrationReviewPackage:
    try:
        return _consume_precise_intent_calibration_handoff(private_data_dir)
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_FAILURE_DETAILS,
            fallback=_CONSUME_FALLBACK,
        )
    del private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _load_pending(private_data_dir: Path) -> CalibrationReviewPackage:
    path = private_data_dir / PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH
    resolved = _safe_private_path(path, private_data_dir, include_leaf=True)
    try:
        package = CalibrationReviewPackage.model_validate_json(resolved.read_bytes())
        PreciseIntentCalibrationHandoff(
            purpose=PRECISE_INTENT_CALIBRATION_PURPOSE,
            version=1,
            created_at=package.submitted_at,
            item_count=8,
            submissions=package.submissions,
        )
        return package
    except Exception:
        raise _PreciseIntentFailure(*_PUBLISH_FALLBACK) from None


def _merged_package(
    current: CalibrationCandidatePackage,
    additions: CalibrationCandidatePackage,
    *,
    published_at: datetime,
) -> CalibrationCandidatePackage:
    if (
        current.candidate_count != 5
        or additions.candidate_count != 8
        or additions.audit.approved_count != 8
        or additions.audit.rejected_count != 0
        or additions.audit.privacy_rejected_count != 0
        or additions.identity_config_fingerprint != current.identity_config_fingerprint
    ):
        raise _PreciseIntentFailure(*_PUBLISH_FALLBACK)
    candidates = sorted(
        [*current.candidates, *additions.candidates],
        key=lambda candidate: candidate.candidate_id,
    )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    category_intents = [(candidate.category, candidate.intent) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)) or len(category_intents) != len(
        set(category_intents)
    ):
        raise _PreciseIntentFailure(*_PUBLISH_FALLBACK)
    audit = CalibrationCandidateAudit(
        reviewed_at=published_at,
        submission_count=current.audit.submission_count + additions.audit.submission_count,
        approved_count=current.audit.approved_count + additions.audit.approved_count,
        rejected_count=current.audit.rejected_count + additions.audit.rejected_count,
        privacy_rejected_count=(
            current.audit.privacy_rejected_count + additions.audit.privacy_rejected_count
        ),
        revoked_count=current.audit.revoked_count + additions.audit.revoked_count,
        output_count=len(candidates),
    )
    return CalibrationCandidatePackage(
        created_at=published_at,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=current.identity_config_fingerprint,
        candidate_count=len(candidates),
        candidates=candidates,
        audit=audit,
    )


def _publish_precise_intent_calibration_batch(
    decisions: Sequence[CalibrationReviewDecision],
    *,
    published_at: datetime,
    expected_current_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    private_data_dir: Path,
) -> CalibrationCandidatePackage:
    pending = _load_pending(private_data_dir)
    additions = build_calibration_candidate_package(
        pending.submissions,
        decisions,
        submitted_at=pending.submitted_at,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    current = load_calibration_candidate_package(
        private_data_dir=private_data_dir,
        expected_fingerprint=expected_current_fingerprint,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    merged = _merged_package(current, additions, published_at=published_at)

    backup_path = private_data_dir / PRECISE_INTENT_CALIBRATION_HISTORY_RELATIVE_PATH
    _atomic_write_model(
        current,
        backup_path,
        private_data_dir,
        CalibrationCandidatePackage,
    )
    backup = CalibrationCandidatePackage.model_validate_json(
        _safe_private_path(backup_path, private_data_dir, include_leaf=True).read_bytes()
    )
    if backup != current:
        raise _PreciseIntentFailure(*_PUBLISH_FALLBACK)

    formal_published = False
    runtime_path = private_data_dir / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    existing_runtime: CalibrationRuntimeHandoff | None = None
    if runtime_path.exists():
        try:
            existing_runtime = CalibrationRuntimeHandoff.model_validate_json(
                _safe_private_path(
                    runtime_path,
                    private_data_dir,
                    include_leaf=True,
                ).read_bytes()
            )
        except Exception:
            raise _PreciseIntentFailure(*_PUBLISH_FALLBACK) from None
    runtime_published = False
    try:
        publish_calibration_candidate_package(
            merged,
            private_data_dir=private_data_dir,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
        formal_published = True
        formal = load_calibration_candidate_package(
            private_data_dir=private_data_dir,
            expected_fingerprint=merged.content_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
        if formal != merged:
            raise _PreciseIntentFailure(*_PUBLISH_FALLBACK)
        runtime = CalibrationRuntimeHandoff(
            purpose="calibration_runtime",
            version=1,
            created_at=published_at,
            expected_calibration_fingerprint=merged.content_fingerprint,
            identity_config_fingerprint=merged.identity_config_fingerprint,
            true_name_terms=list(true_name_terms),
            approved_nicknames=list(approved_nicknames),
        )
        _atomic_write_model(
            runtime,
            runtime_path,
            private_data_dir,
            CalibrationRuntimeHandoff,
        )
        runtime_published = True
        verified_runtime = CalibrationRuntimeHandoff.model_validate_json(
            _safe_private_path(runtime_path, private_data_dir, include_leaf=True).read_bytes()
        )
        if verified_runtime != runtime:
            raise _PreciseIntentFailure(*_PUBLISH_FALLBACK)
        pending_path = private_data_dir / PRECISE_INTENT_CALIBRATION_PENDING_RELATIVE_PATH
        _safe_private_path(pending_path, private_data_dir, include_leaf=True).unlink()
        return formal
    except Exception:
        if formal_published:
            try:
                publish_calibration_candidate_package(
                    current,
                    private_data_dir=private_data_dir,
                    true_name_terms=true_name_terms,
                    approved_nicknames=approved_nicknames,
                )
                if runtime_published:
                    if existing_runtime is None:
                        _safe_private_path(
                            runtime_path,
                            private_data_dir,
                            include_leaf=runtime_path.exists(),
                        ).unlink(missing_ok=True)
                    else:
                        _atomic_write_model(
                            existing_runtime,
                            runtime_path,
                            private_data_dir,
                            CalibrationRuntimeHandoff,
                        )
            except Exception:
                pass
        raise _PreciseIntentFailure(*_PUBLISH_FALLBACK) from None


def publish_precise_intent_calibration_batch(
    decisions: Sequence[CalibrationReviewDecision],
    *,
    published_at: datetime,
    expected_current_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    private_data_dir: Path,
) -> CalibrationCandidatePackage:
    try:
        return _publish_precise_intent_calibration_batch(
            decisions,
            published_at=published_at,
            expected_current_fingerprint=expected_current_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
            private_data_dir=private_data_dir,
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_FAILURE_DETAILS,
            fallback=_PUBLISH_FALLBACK,
        )
    del (
        decisions,
        published_at,
        expected_current_fingerprint,
        true_name_terms,
        approved_nicknames,
        private_data_dir,
    )
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _write_five_fixed_intent_calibration_handoff(
    submissions: Sequence[CalibrationReviewSubmission],
    created_at: datetime,
    private_data_dir: Path,
) -> Path:
    try:
        copied = [
            CalibrationReviewSubmission.model_validate(item.model_dump(mode="json"))
            for item in submissions
        ]
        handoff = FiveFixedIntentCalibrationHandoff(
            purpose=FIVE_FIXED_INTENT_CALIBRATION_PURPOSE,
            version=1,
            created_at=created_at,
            item_count=5,
            submissions=copied,
        )
        return _atomic_write_model(
            handoff,
            private_data_dir / FIVE_FIXED_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH,
            private_data_dir,
            FiveFixedIntentCalibrationHandoff,
        )
    except Exception:
        raise _PreciseIntentFailure(*_FIVE_WRITE_FALLBACK) from None


def write_five_fixed_intent_calibration_handoff(
    submissions: Sequence[CalibrationReviewSubmission],
    *,
    created_at: datetime,
    private_data_dir: Path,
) -> Path:
    try:
        return _write_five_fixed_intent_calibration_handoff(
            submissions, created_at, private_data_dir
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_FIVE_FAILURE_DETAILS,
            fallback=_FIVE_WRITE_FALLBACK,
        )
    del submissions, created_at, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _consume_five_fixed_intent_calibration_handoff(
    private_data_dir: Path,
) -> CalibrationReviewPackage:
    intake_path = private_data_dir / FIVE_FIXED_INTENT_CALIBRATION_INTAKE_RELATIVE_PATH
    pending_path = private_data_dir / FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH
    resolved_pending = _safe_private_path(
        pending_path, private_data_dir, include_leaf=pending_path.exists()
    )
    if resolved_pending.exists():
        raise _PreciseIntentFailure(*_FIVE_CONSUME_FALLBACK)
    try:
        resolved_intake = _safe_private_path(intake_path, private_data_dir, include_leaf=True)
        handoff = FiveFixedIntentCalibrationHandoff.model_validate_json(
            resolved_intake.read_bytes()
        )
        package = CalibrationReviewPackage(
            submitted_at=handoff.created_at,
            content_fingerprint=calibration_review_fingerprint(handoff.submissions),
            submission_count=5,
            submissions=handoff.submissions,
        )
        _atomic_write_model(package, pending_path, private_data_dir, CalibrationReviewPackage)
        verified = CalibrationReviewPackage.model_validate_json(resolved_pending.read_bytes())
        if verified != package:
            raise _PreciseIntentFailure(*_FIVE_CONSUME_FALLBACK)
        resolved_intake.unlink()
        return verified
    except Exception:
        with suppress(OSError):
            resolved_pending.unlink(missing_ok=True)
        raise _PreciseIntentFailure(*_FIVE_CONSUME_FALLBACK) from None


def consume_five_fixed_intent_calibration_handoff(
    *, private_data_dir: Path
) -> CalibrationReviewPackage:
    try:
        return _consume_five_fixed_intent_calibration_handoff(private_data_dir)
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_FIVE_FAILURE_DETAILS,
            fallback=_FIVE_CONSUME_FALLBACK,
        )
    del private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _load_five_fixed_pending(private_data_dir: Path) -> CalibrationReviewPackage:
    path = private_data_dir / FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH
    try:
        resolved = _safe_private_path(path, private_data_dir, include_leaf=True)
        package = CalibrationReviewPackage.model_validate_json(resolved.read_bytes())
        FiveFixedIntentCalibrationHandoff(
            purpose=FIVE_FIXED_INTENT_CALIBRATION_PURPOSE,
            version=1,
            created_at=package.submitted_at,
            item_count=5,
            submissions=package.submissions,
        )
        return package
    except Exception:
        raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK) from None


def _merge_five_fixed_package(
    current: CalibrationCandidatePackage,
    additions: CalibrationCandidatePackage,
    *,
    published_at: datetime,
) -> CalibrationCandidatePackage:
    reused = [
        candidate
        for candidate in current.candidates
        if candidate.category.value == "weather" and candidate.intent == "weather_cold_reminder"
    ]
    addition_pairs = {(candidate.category, candidate.intent) for candidate in additions.candidates}
    if (
        current.candidate_count != 13
        or additions.candidate_count != 5
        or additions.audit.approved_count != 5
        or additions.audit.rejected_count != 0
        or additions.audit.privacy_rejected_count != 0
        or additions.identity_config_fingerprint != current.identity_config_fingerprint
        or len(reused) != 1
        or (CandidateCategory.WEATHER, "weather_cold_reminder") in addition_pairs
    ):
        raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK)
    candidates = sorted(
        [*current.candidates, *additions.candidates], key=lambda item: item.candidate_id
    )
    ids = [candidate.candidate_id for candidate in candidates]
    pairs = [(candidate.category, candidate.intent) for candidate in candidates]
    if len(candidates) != 18 or len(ids) != len(set(ids)) or len(pairs) != len(set(pairs)):
        raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK)
    audit = CalibrationCandidateAudit(
        reviewed_at=published_at,
        submission_count=current.audit.submission_count + additions.audit.submission_count,
        approved_count=current.audit.approved_count + additions.audit.approved_count,
        rejected_count=current.audit.rejected_count + additions.audit.rejected_count,
        privacy_rejected_count=(
            current.audit.privacy_rejected_count + additions.audit.privacy_rejected_count
        ),
        revoked_count=current.audit.revoked_count + additions.audit.revoked_count,
        output_count=18,
    )
    return CalibrationCandidatePackage(
        created_at=published_at,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=current.identity_config_fingerprint,
        candidate_count=18,
        candidates=candidates,
        audit=audit,
    )


def _publish_five_fixed_intent_calibration_batch(
    decisions: Sequence[CalibrationReviewDecision],
    *,
    published_at: datetime,
    expected_current_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    private_data_dir: Path,
) -> FiveFixedIntentCalibrationPublishResult:
    pending = _load_five_fixed_pending(private_data_dir)
    additions = build_calibration_candidate_package(
        pending.submissions,
        decisions,
        submitted_at=pending.submitted_at,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    current = load_calibration_candidate_package(
        private_data_dir=private_data_dir,
        expected_fingerprint=expected_current_fingerprint,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    merged = _merge_five_fixed_package(current, additions, published_at=published_at)
    backup_path = private_data_dir / FIVE_FIXED_INTENT_CALIBRATION_HISTORY_RELATIVE_PATH
    _atomic_write_model(current, backup_path, private_data_dir, CalibrationCandidatePackage)
    backup = CalibrationCandidatePackage.model_validate_json(
        _safe_private_path(backup_path, private_data_dir, include_leaf=True).read_bytes()
    )
    if backup != current:
        raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK)

    runtime_path = private_data_dir / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    try:
        existing_runtime = CalibrationRuntimeHandoff.model_validate_json(
            _safe_private_path(runtime_path, private_data_dir, include_leaf=True).read_bytes()
        )
        if (
            existing_runtime.expected_calibration_fingerprint != current.content_fingerprint
            or existing_runtime.identity_config_fingerprint != current.identity_config_fingerprint
        ):
            raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK)
    except Exception:
        raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK) from None

    formal_published = False
    runtime_published = False
    try:
        publish_calibration_candidate_package(
            merged,
            private_data_dir=private_data_dir,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
        formal_published = True
        formal = load_calibration_candidate_package(
            private_data_dir=private_data_dir,
            expected_fingerprint=merged.content_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
        if formal != merged:
            raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK)
        runtime = CalibrationRuntimeHandoff(
            purpose="calibration_runtime",
            version=1,
            created_at=published_at,
            expected_calibration_fingerprint=merged.content_fingerprint,
            identity_config_fingerprint=merged.identity_config_fingerprint,
            true_name_terms=list(true_name_terms),
            approved_nicknames=list(approved_nicknames),
        )
        _atomic_write_model(runtime, runtime_path, private_data_dir, CalibrationRuntimeHandoff)
        runtime_published = True
        verified_runtime = CalibrationRuntimeHandoff.model_validate_json(
            _safe_private_path(runtime_path, private_data_dir, include_leaf=True).read_bytes()
        )
        if verified_runtime != runtime:
            raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK)
        pending_path = private_data_dir / FIVE_FIXED_INTENT_CALIBRATION_PENDING_RELATIVE_PATH
        _safe_private_path(pending_path, private_data_dir, include_leaf=True).unlink()
        return FiveFixedIntentCalibrationPublishResult(
            package=formal, added_count=5, reused_existing=1
        )
    except Exception:
        if formal_published:
            try:
                publish_calibration_candidate_package(
                    current,
                    private_data_dir=private_data_dir,
                    true_name_terms=true_name_terms,
                    approved_nicknames=approved_nicknames,
                )
                if runtime_published:
                    _atomic_write_model(
                        existing_runtime,
                        runtime_path,
                        private_data_dir,
                        CalibrationRuntimeHandoff,
                    )
            except Exception:
                pass
        raise _PreciseIntentFailure(*_FIVE_PUBLISH_FALLBACK) from None


def publish_five_fixed_intent_calibration_batch(
    decisions: Sequence[CalibrationReviewDecision],
    *,
    published_at: datetime,
    expected_current_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    private_data_dir: Path,
) -> FiveFixedIntentCalibrationPublishResult:
    try:
        return _publish_five_fixed_intent_calibration_batch(
            decisions,
            published_at=published_at,
            expected_current_fingerprint=expected_current_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
            private_data_dir=private_data_dir,
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_FIVE_FAILURE_DETAILS,
            fallback=_FIVE_PUBLISH_FALLBACK,
        )
    del (
        decisions,
        published_at,
        expected_current_fingerprint,
        true_name_terms,
        approved_nicknames,
        private_data_dir,
    )
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _write_sleep1_incremental_calibration_handoff(
    submissions: Sequence[CalibrationReviewSubmission],
    created_at: datetime,
    private_data_dir: Path,
) -> Path:
    try:
        copied = [
            CalibrationReviewSubmission.model_validate(item.model_dump(mode="json"))
            for item in submissions
        ]
        handoff = Sleep1IncrementalCalibrationHandoff(
            purpose=SLEEP1_INCREMENTAL_CALIBRATION_PURPOSE,
            version=1,
            created_at=created_at,
            item_count=1,
            submissions=copied,
        )
        return _atomic_write_model(
            handoff,
            private_data_dir / SLEEP1_INCREMENTAL_CALIBRATION_INTAKE_RELATIVE_PATH,
            private_data_dir,
            Sleep1IncrementalCalibrationHandoff,
        )
    except Exception:
        raise _PreciseIntentFailure(*_SLEEP1_WRITE_FALLBACK) from None


def write_sleep1_incremental_calibration_handoff(
    submissions: Sequence[CalibrationReviewSubmission],
    *,
    created_at: datetime,
    private_data_dir: Path,
) -> Path:
    try:
        return _write_sleep1_incremental_calibration_handoff(
            submissions, created_at, private_data_dir
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_SLEEP1_FAILURE_DETAILS,
            fallback=_SLEEP1_WRITE_FALLBACK,
        )
    del submissions, created_at, private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _consume_sleep1_incremental_calibration_handoff(
    private_data_dir: Path,
) -> CalibrationReviewPackage:
    intake_path = private_data_dir / SLEEP1_INCREMENTAL_CALIBRATION_INTAKE_RELATIVE_PATH
    pending_path = private_data_dir / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH
    resolved_pending = _safe_private_path(
        pending_path, private_data_dir, include_leaf=pending_path.exists()
    )
    if resolved_pending.exists():
        raise _PreciseIntentFailure(*_SLEEP1_CONSUME_FALLBACK)
    try:
        resolved_intake = _safe_private_path(intake_path, private_data_dir, include_leaf=True)
        handoff = Sleep1IncrementalCalibrationHandoff.model_validate_json(
            resolved_intake.read_bytes()
        )
        package = CalibrationReviewPackage(
            submitted_at=handoff.created_at,
            content_fingerprint=calibration_review_fingerprint(handoff.submissions),
            submission_count=1,
            submissions=handoff.submissions,
        )
        _atomic_write_model(package, pending_path, private_data_dir, CalibrationReviewPackage)
        verified = CalibrationReviewPackage.model_validate_json(resolved_pending.read_bytes())
        if verified != package:
            raise _PreciseIntentFailure(*_SLEEP1_CONSUME_FALLBACK)
        resolved_intake.unlink()
        return verified
    except Exception:
        with suppress(OSError):
            resolved_pending.unlink(missing_ok=True)
        raise _PreciseIntentFailure(*_SLEEP1_CONSUME_FALLBACK) from None


def consume_sleep1_incremental_calibration_handoff(
    *, private_data_dir: Path
) -> CalibrationReviewPackage:
    try:
        return _consume_sleep1_incremental_calibration_handoff(private_data_dir)
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_SLEEP1_FAILURE_DETAILS,
            fallback=_SLEEP1_CONSUME_FALLBACK,
        )
    del private_data_dir
    raise PrivacyValidationError(failure[0], failure[1]) from None


def _load_sleep1_pending(private_data_dir: Path) -> CalibrationReviewPackage:
    path = private_data_dir / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH
    try:
        resolved = _safe_private_path(path, private_data_dir, include_leaf=True)
        package = CalibrationReviewPackage.model_validate_json(resolved.read_bytes())
        Sleep1IncrementalCalibrationHandoff(
            purpose=SLEEP1_INCREMENTAL_CALIBRATION_PURPOSE,
            version=1,
            created_at=package.submitted_at,
            item_count=1,
            submissions=package.submissions,
        )
        return package
    except Exception:
        raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK) from None


def _restore_sleep1_runtime_bytes(
    payload: bytes,
    destination: Path,
    private_data_dir: Path,
) -> None:
    resolved = _safe_private_path(destination, private_data_dir, include_leaf=True)
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{resolved.name}.", suffix=".restore", dir=resolved.parent
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "w+b") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        _safe_private_path(destination, private_data_dir, include_leaf=True)
        os.replace(temporary, resolved)
        temporary = None
        if resolved.read_bytes() != payload:
            raise OSError
    except Exception:
        raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK) from None
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _merge_sleep1_incremental_package(
    current: CalibrationCandidatePackage,
    addition: CalibrationCandidatePackage,
    *,
    published_at: datetime,
) -> CalibrationCandidatePackage:
    target = (CandidateCategory.SLEEP, "sleep_farewell")
    current_pairs = {(item.category, item.intent) for item in current.candidates}
    addition_pairs = {(item.category, item.intent) for item in addition.candidates}
    if (
        current.candidate_count != 18
        or addition.candidate_count != 1
        or addition.audit.approved_count != 1
        or addition.audit.rejected_count != 0
        or addition.audit.privacy_rejected_count != 0
        or addition.identity_config_fingerprint != current.identity_config_fingerprint
        or target in current_pairs
        or addition_pairs != {target}
    ):
        raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK)
    candidates = sorted(
        [*current.candidates, *addition.candidates], key=lambda item: item.candidate_id
    )
    ids = [item.candidate_id for item in candidates]
    pairs = [(item.category, item.intent) for item in candidates]
    if len(candidates) != 19 or len(ids) != len(set(ids)) or len(pairs) != len(set(pairs)):
        raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK)
    audit = CalibrationCandidateAudit(
        reviewed_at=published_at,
        submission_count=current.audit.submission_count + addition.audit.submission_count,
        approved_count=current.audit.approved_count + addition.audit.approved_count,
        rejected_count=current.audit.rejected_count + addition.audit.rejected_count,
        privacy_rejected_count=(
            current.audit.privacy_rejected_count + addition.audit.privacy_rejected_count
        ),
        revoked_count=current.audit.revoked_count + addition.audit.revoked_count,
        output_count=19,
    )
    return CalibrationCandidatePackage(
        created_at=published_at,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=current.identity_config_fingerprint,
        candidate_count=19,
        candidates=candidates,
        audit=audit,
    )


def _publish_sleep1_incremental_calibration_batch(
    decisions: Sequence[CalibrationReviewDecision],
    *,
    published_at: datetime,
    expected_current_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    private_data_dir: Path,
) -> Sleep1IncrementalCalibrationPublishResult:
    pending = _load_sleep1_pending(private_data_dir)
    addition = build_calibration_candidate_package(
        pending.submissions,
        decisions,
        submitted_at=pending.submitted_at,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    current = load_calibration_candidate_package(
        private_data_dir=private_data_dir,
        expected_fingerprint=expected_current_fingerprint,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    merged = _merge_sleep1_incremental_package(current, addition, published_at=published_at)
    backup_path = private_data_dir / SLEEP1_INCREMENTAL_CALIBRATION_HISTORY_RELATIVE_PATH
    _atomic_write_model(current, backup_path, private_data_dir, CalibrationCandidatePackage)
    backup = CalibrationCandidatePackage.model_validate_json(
        _safe_private_path(backup_path, private_data_dir, include_leaf=True).read_bytes()
    )
    if backup != current:
        raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK)

    runtime_path = private_data_dir / CALIBRATION_RUNTIME_HANDOFF_RELATIVE_PATH
    try:
        resolved_runtime = _safe_private_path(runtime_path, private_data_dir, include_leaf=True)
        existing_runtime_bytes = resolved_runtime.read_bytes()
        existing_runtime = CalibrationRuntimeHandoff.model_validate_json(existing_runtime_bytes)
        if (
            existing_runtime.expected_calibration_fingerprint != current.content_fingerprint
            or existing_runtime.identity_config_fingerprint != current.identity_config_fingerprint
        ):
            raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK)
    except Exception:
        raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK) from None

    formal_published = False
    runtime_published = False
    try:
        publish_calibration_candidate_package(
            merged,
            private_data_dir=private_data_dir,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
        formal_published = True
        formal = load_calibration_candidate_package(
            private_data_dir=private_data_dir,
            expected_fingerprint=merged.content_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
        if formal != merged:
            raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK)
        runtime = CalibrationRuntimeHandoff(
            purpose="calibration_runtime",
            version=1,
            created_at=published_at,
            expected_calibration_fingerprint=merged.content_fingerprint,
            identity_config_fingerprint=merged.identity_config_fingerprint,
            true_name_terms=list(true_name_terms),
            approved_nicknames=list(approved_nicknames),
        )
        runtime_published = True
        _atomic_write_model(runtime, runtime_path, private_data_dir, CalibrationRuntimeHandoff)
        verified_runtime = CalibrationRuntimeHandoff.model_validate_json(
            _safe_private_path(runtime_path, private_data_dir, include_leaf=True).read_bytes()
        )
        if verified_runtime != runtime:
            raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK)
        pending_path = private_data_dir / SLEEP1_INCREMENTAL_CALIBRATION_PENDING_RELATIVE_PATH
        _safe_private_path(pending_path, private_data_dir, include_leaf=True).unlink()
        return Sleep1IncrementalCalibrationPublishResult(package=formal, added_count=1)
    except Exception:
        if formal_published:
            try:
                publish_calibration_candidate_package(
                    current,
                    private_data_dir=private_data_dir,
                    true_name_terms=true_name_terms,
                    approved_nicknames=approved_nicknames,
                )
                if runtime_published:
                    _restore_sleep1_runtime_bytes(
                        existing_runtime_bytes,
                        runtime_path,
                        private_data_dir,
                    )
            except Exception:
                pass
        raise _PreciseIntentFailure(*_SLEEP1_PUBLISH_FALLBACK) from None


def publish_sleep1_incremental_calibration_batch(
    decisions: Sequence[CalibrationReviewDecision],
    *,
    published_at: datetime,
    expected_current_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    private_data_dir: Path,
) -> Sleep1IncrementalCalibrationPublishResult:
    try:
        return _publish_sleep1_incremental_calibration_batch(
            decisions,
            published_at=published_at,
            expected_current_fingerprint=expected_current_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
            private_data_dir=private_data_dir,
        )
    except Exception as error:
        failure = _sanitized_public_privacy_failure(
            error,
            internal_failure_type=_PreciseIntentFailure,
            allowed_details=_SLEEP1_FAILURE_DETAILS,
            fallback=_SLEEP1_PUBLISH_FALLBACK,
        )
    del (
        decisions,
        published_at,
        expected_current_fingerprint,
        true_name_terms,
        approved_nicknames,
        private_data_dir,
    )
    raise PrivacyValidationError(failure[0], failure[1]) from None
