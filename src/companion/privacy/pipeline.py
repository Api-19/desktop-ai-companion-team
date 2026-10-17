from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime

from companion.contracts.chat import ChatRecord, SpeakerRole
from companion.contracts.privacy import RedactedRecord
from companion.privacy.errors import PrivacyValidationError, _PrivacyFailure
from companion.privacy.models import (
    PrivacyAuditEvent,
    PrivacyBatchResult,
    PrivacyReviewPackage,
    ReviewDecision,
    ReviewDecisionValue,
)
from companion.privacy.redaction import redact_text
from companion.privacy.review import (
    PrivateTerms,
    dataset_fingerprint,
    scan_for_review,
    segment_records,
)


def _decision_map(
    review_package: PrivacyReviewPackage,
    decisions: list[ReviewDecision],
) -> dict[str, ReviewDecisionValue]:
    required = {segment.segment_id for segment in review_package.risk_segments}
    supplied_ids = [decision.segment_id for decision in decisions]
    supplied = set(supplied_ids)

    if len(supplied_ids) != len(supplied) or supplied - required:
        raise _PrivacyFailure(
            "invalid_review_decisions",
            "review decisions contain duplicate or unexpected segment IDs",
        )
    if required - supplied:
        raise _PrivacyFailure(
            "incomplete_review",
            "every risk segment requires an explicit decision",
        )
    return {decision.segment_id: decision.decision for decision in decisions}


def _finalize_privacy_batch(
    records: list[ChatRecord],
    review_package: PrivacyReviewPackage,
    decisions: list[ReviewDecision],
    *,
    processed_at: datetime,
    risk_terms: PrivateTerms | None = None,
    redaction_terms: Mapping[str, Sequence[str]] | None = None,
) -> PrivacyBatchResult:
    if processed_at.tzinfo is None or processed_at.utcoffset() is None:
        raise _PrivacyFailure(
            "invalid_timestamp",
            "audit timestamps must include a timezone",
        )
    if dataset_fingerprint(records) != review_package.dataset_fingerprint:
        raise _PrivacyFailure(
            "stale_review_package",
            "record batch no longer matches the review package",
        )

    try:
        expected_package = scan_for_review(
            records,
            created_at=review_package.created_at,
            private_terms=risk_terms,
        )
    except PrivacyValidationError as error:
        failure = (error.code, "risk terms are invalid")
    else:
        failure = None
    if failure is not None:
        raise _PrivacyFailure(failure[0], failure[1])
    if expected_package.risk_segments != review_package.risk_segments:
        raise _PrivacyFailure(
            "stale_review_package",
            "risk scan no longer matches the review package",
        )

    decision_by_segment = _decision_map(review_package, decisions)
    risk_segment_ids = {segment.segment_id for segment in review_package.risk_segments}
    decision_by_message_id = {
        record.message_id: decision_by_segment[risk_segment.segment_id]
        for risk_segment in review_package.risk_segments
        for record in risk_segment.records
    }
    risk_source_segment_ids = {
        risk_segment.source_segment_id for risk_segment in review_package.risk_segments
    }
    segments = segment_records(records)
    if len(segments) != review_package.scanned_segment_count:
        raise _PrivacyFailure(
            "stale_review_package",
            "record batch no longer matches the review package",
        )

    redacted_records: list[RedactedRecord] = []
    redaction_counts: Counter[str] = Counter()
    approved_risk_count = sum(
        decision is ReviewDecisionValue.APPROVE for decision in decision_by_segment.values()
    )
    excluded_risk_count = sum(
        decision is ReviewDecisionValue.EXCLUDE for decision in decision_by_segment.values()
    )

    for _, segment in segments:
        for record in segment:
            if record.speaker_role is not SpeakerRole.OWNER:
                continue
            if decision_by_message_id.get(record.message_id) is ReviewDecisionValue.EXCLUDE:
                continue
            try:
                redaction = redact_text(record.text, private_terms=redaction_terms)
            except PrivacyValidationError as error:
                failure = (error.code, "redaction terms are invalid")
            else:
                failure = None
            if failure is not None:
                raise _PrivacyFailure(failure[0], failure[1])
            redaction_counts.update(redaction.counts)
            redacted_records.append(
                RedactedRecord(
                    record=record.model_copy(update={"text": None}),
                    redacted_text=redaction.text,
                    redaction_labels=list(redaction.labels),
                )
            )

    audit = PrivacyAuditEvent(
        event_type="privacy_batch_finalized",
        processed_at=processed_at,
        input_count=len(records),
        scanned_segment_count=len(segments),
        risk_segment_count=len(risk_segment_ids),
        approved_risk_segment_count=approved_risk_count,
        excluded_risk_segment_count=excluded_risk_count,
        auto_cleared_segment_count=len(segments) - len(risk_source_segment_ids),
        owner_output_count=len(redacted_records),
        dropped_other_count=sum(record.speaker_role is SpeakerRole.OTHER for record in records),
        redaction_counts=dict(sorted(redaction_counts.items())),
        succeeded=True,
    )
    return PrivacyBatchResult(records=redacted_records, audit=audit)


def finalize_privacy_batch(
    records: list[ChatRecord],
    review_package: PrivacyReviewPackage,
    decisions: list[ReviewDecision],
    *,
    processed_at: datetime,
    risk_terms: PrivateTerms | None = None,
    redaction_terms: Mapping[str, Sequence[str]] | None = None,
) -> PrivacyBatchResult:
    try:
        return _finalize_privacy_batch(
            records,
            review_package,
            decisions,
            processed_at=processed_at,
            risk_terms=risk_terms,
            redaction_terms=redaction_terms,
        )
    except _PrivacyFailure as error:
        failure = (error.code, error.detail)
    del records, review_package, decisions, processed_at, risk_terms, redaction_terms
    raise PrivacyValidationError(failure[0], failure[1]) from None
