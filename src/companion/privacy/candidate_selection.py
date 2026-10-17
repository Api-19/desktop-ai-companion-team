from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from companion.contracts.chat import ChatRecord, SpeakerRole
from companion.privacy.ambiguous_models import CalibrationProfile
from companion.privacy.ambiguous_sampling import text_similarity
from companion.privacy.candidate_models import (
    CandidateCategory,
    CandidateRejectReason,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.privacy.candidate_rules import (
    allowed_category,
    classification_text,
    hard_rejection_reason,
    normalize_text,
    replace_private_names,
    short_repeat_key,
)
from companion.privacy.errors import PrivacyValidationError, _PrivacyFailure
from companion.privacy.review import segment_records

_CONTEXT_RADIUS = 5
_SHORT_REPEAT_LIMIT = 3


def _validate_records(records: list[ChatRecord], processed_at: datetime) -> None:
    if processed_at.tzinfo is None or processed_at.utcoffset() is None:
        raise _PrivacyFailure(
            "invalid_timestamp",
            "candidate timestamps must include a timezone",
        )
    if not records:
        raise _PrivacyFailure("empty_records", "record batch must not be empty")

    message_ids: set[str] = set()
    for record in records:
        if record.timestamp.tzinfo is None or record.timestamp.utcoffset() is None:
            raise _PrivacyFailure(
                "invalid_timestamp",
                "candidate timestamps must include a timezone",
            )
        if record.speaker_role not in (SpeakerRole.OWNER, SpeakerRole.OTHER):
            raise _PrivacyFailure(
                "unconfirmed_speaker_role",
                "record batch contains an unconfirmed speaker role",
            )
        if record.message_id in message_ids:
            raise _PrivacyFailure(
                "duplicate_message_id",
                "record batch contains a duplicate message ID",
            )
        message_ids.add(record.message_id)


def _normalise_terms(
    terms: Sequence[str],
    *,
    required: bool,
) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise _PrivacyFailure(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )
    values = tuple(terms)
    if (required and not values) or any(
        not isinstance(term, str) or not term.strip() or hard_rejection_reason(term) is not None
        for term in values
    ):
        raise _PrivacyFailure(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )
    return values


def _context_records(segment: list[ChatRecord], index: int) -> list[ChatRecord]:
    start = max(0, index - _CONTEXT_RADIUS)
    end = min(len(segment), index + _CONTEXT_RADIUS + 1)
    return [*segment[start:index], *segment[index + 1 : end]]


def _context_is_positively_safe(
    context: list[ChatRecord],
    true_name_terms: tuple[str, ...],
    approved_nicknames: tuple[str, ...],
) -> bool:
    if not context:
        return False
    for item in context:
        if item.content_type != "text" or item.text is None:
            return False
        if hard_rejection_reason(item.text) is not None:
            return False
        redacted = replace_private_names(item.text, true_name_terms)
        classifier_input = classification_text(redacted, approved_nicknames)
        if allowed_category(classifier_input) is None:
            return False
    return True


def _feature_group(text: str) -> str:
    length = len(normalize_text(text))
    bucket = "xs" if length <= 4 else "s" if length <= 10 else "m" if length <= 20 else "l"
    punctuation = (
        "bang"
        if "!" in text or "\uff01" in text
        else "question"
        if "?" in text or "\uff1f" in text
        else "plain"
    )
    emoji = "emoji" if any(ord(char) > 0xFFFF for char in text) else "text"
    return f"{bucket}:{punctuation}:{emoji}"


def _calibrated_category(
    text: str,
    profile: CalibrationProfile | None,
) -> CandidateCategory | None:
    if profile is None or not profile.enabled:
        return None
    normalized = normalize_text(text)
    if normalized in {normalize_text(value) for value in profile.manual_approved_texts}:
        return CandidateCategory.CALIBRATED_DAILY
    group = _feature_group(normalized)
    for rule in profile.rules:
        if (
            rule.enabled
            and group == rule.similarity_group
            and text_similarity(normalized, rule.normalized_template) >= rule.similarity_threshold
        ):
            return CandidateCategory.CALIBRATED_DAILY
    return None


def _select_safe_candidates(
    records: list[ChatRecord],
    *,
    processed_at: datetime,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    calibration_profile: CalibrationProfile | None,
) -> SafeCandidatePackage:
    _validate_records(records, processed_at)
    names = _normalise_terms(true_name_terms, required=True)
    nicknames = _normalise_terms(approved_nicknames, required=False)
    if set(names) & set(nicknames):
        raise _PrivacyFailure(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )

    candidates: list[SafeStyleCandidate] = []
    allowed_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    repeat_counts: Counter[tuple[str, str]] = Counter()
    unknown_count = 0
    deduplicated_count = 0

    for _, segment in segment_records(records):
        index_by_message_id = {item.message_id: index for index, item in enumerate(segment)}
        for record in segment:
            if record.speaker_role is SpeakerRole.OTHER:
                continue
            if record.text is None or record.content_type != "text":
                rejection_counts[CandidateRejectReason.NON_TEXT.value] += 1
                continue

            reason = hard_rejection_reason(record.text)
            if reason is not None:
                rejection_counts[reason.value] += 1
                continue

            context = _context_records(segment, index_by_message_id[record.message_id])
            if any(hard_rejection_reason(item.text or "") is not None for item in context):
                rejection_counts[CandidateRejectReason.SENSITIVE_CONTEXT.value] += 1
                continue

            redacted = replace_private_names(record.text, names)
            redacted_reason = hard_rejection_reason(redacted)
            if redacted_reason is not None:
                rejection_counts[redacted_reason.value] += 1
                continue
            classifier_input = classification_text(redacted, nicknames)
            category = allowed_category(classifier_input)
            if category is None:
                category = _calibrated_category(redacted, calibration_profile)
                if category is None:
                    unknown_count += 1
                    continue

            if category in {CandidateCategory.SHORT_REPLY, CandidateCategory.EMOJI} and not (
                _context_is_positively_safe(context, names, nicknames)
            ):
                rejection_counts[CandidateRejectReason.SENSITIVE_CONTEXT.value] += 1
                continue

            repeat_key = short_repeat_key(redacted, category)
            repeat_limit = _SHORT_REPEAT_LIMIT
            if category is CandidateCategory.CALIBRATED_DAILY:
                repeat_key = normalize_text(redacted)
                repeat_limit = 1
            if repeat_key is not None:
                counter_key = (category.value, repeat_key)
                if repeat_counts[counter_key] >= repeat_limit:
                    deduplicated_count += 1
                    continue
                repeat_counts[counter_key] += 1

            candidates.append(
                SafeStyleCandidate(
                    text=redacted,
                    category=category,
                    scene_tags=[category.value],
                )
            )
            allowed_counts[category.value] += 1

    audit = CandidateSelectionAudit(
        processed_at=processed_at,
        input_count=len(records),
        owner_input_count=sum(record.speaker_role is SpeakerRole.OWNER for record in records),
        dropped_other_count=sum(record.speaker_role is SpeakerRole.OTHER for record in records),
        allowed_category_counts=dict(sorted(allowed_counts.items())),
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
        unknown_count=unknown_count,
        deduplicated_count=deduplicated_count,
        output_count=len(candidates),
        succeeded=True,
    )
    assert audit.output_count == len(candidates)
    return SafeCandidatePackage(
        created_at=processed_at,
        candidates=candidates,
        audit=audit,
    )


def select_safe_candidates(
    records: list[ChatRecord],
    *,
    processed_at: datetime,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    calibration_profile: CalibrationProfile | None = None,
) -> SafeCandidatePackage:
    try:
        return _select_safe_candidates(
            records,
            processed_at=processed_at,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
            calibration_profile=calibration_profile,
        )
    except _PrivacyFailure as error:
        failure = (error.code, error.detail)
    del records, processed_at, true_name_terms, approved_nicknames, calibration_profile
    raise PrivacyValidationError(failure[0], failure[1]) from None
