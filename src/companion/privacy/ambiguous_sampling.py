from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from companion.contracts.chat import ChatRecord, SpeakerRole
from companion.privacy.ambiguous_models import (
    AmbiguousReviewItem,
    AmbiguousReviewPackage,
    CalibrationProposal,
    CalibrationRule,
    ReviewContextItem,
    ReviewRound,
    SecondRoundRuleLink,
    SecondRoundState,
)
from companion.privacy.candidate_rules import (
    allowed_category,
    classification_text,
    hard_rejection_reason,
    normalize_text,
    replace_private_names,
)
from companion.privacy.errors import PrivacyValidationError, _PrivacyFailure
from companion.privacy.redaction import PrivateRedactionTerms, redact_text
from companion.privacy.review import segment_records

_CONTEXT_RADIUS = 2
_FIRST_ROUND_COUNT = 100
_SECOND_ROUND_COUNT = 50
_NON_TEXT_CONTENT_TYPES = {
    "audio",
    "file",
    "image",
    "quote",
    "sticker",
    "system",
    "video",
    "voice",
}


@dataclass(frozen=True)
class _PoolEntry:
    text: str
    context_before: tuple[ReviewContextItem, ...]
    context_after: tuple[ReviewContextItem, ...]
    similarity_group: str
    occurrence_count: int


@dataclass(frozen=True)
class _PoolOccurrence:
    sort_key: tuple[datetime, str, str]
    normalized_text: str
    entry: _PoolEntry


@dataclass(frozen=True)
class _RuleLinkedEntry:
    entry: _PoolEntry
    rule_id: str
    is_match: bool

    @property
    def text(self) -> str:
        return self.entry.text

    @property
    def similarity_group(self) -> str:
        return self.entry.similarity_group


def _ngrams(text: str) -> frozenset[str]:
    value = normalize_text(text)
    grams = {
        value[index : index + width]
        for width in (2, 3)
        for index in range(max(0, len(value) - width + 1))
    }
    return frozenset(grams or {value})


def text_similarity(left: str, right: str) -> float:
    left_grams, right_grams = _ngrams(left), _ngrams(right)
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 1.0


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


def _validate_records(records: list[ChatRecord]) -> None:
    message_ids: set[str] = set()
    for record in records:
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
        if record.timestamp.tzinfo is None or record.timestamp.utcoffset() is None:
            raise _PrivacyFailure(
                "invalid_timestamp",
                "record batch timestamps must include a timezone",
            )
        message_ids.add(record.message_id)


def _normalise_terms(terms: Sequence[str], *, required: bool) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise _PrivacyFailure("invalid_candidate_terms", "candidate terms are invalid")
    values = tuple(terms)
    if (required and not values) or any(
        not isinstance(term, str) or not term.strip() or hard_rejection_reason(term) is not None
        for term in values
    ):
        raise _PrivacyFailure("invalid_candidate_terms", "candidate terms are invalid")
    return values


def _normalise_existing_candidate_texts(texts: Sequence[str]) -> frozenset[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise _PrivacyFailure("invalid_existing_candidates", "existing candidates are invalid")
    values = tuple(texts)
    if any(not isinstance(text, str) or not normalize_text(text) for text in values):
        raise _PrivacyFailure("invalid_existing_candidates", "existing candidates are invalid")
    return frozenset(normalize_text(text) for text in values)


def _validate_package_inputs(created_at: datetime, package_token: str) -> None:
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise _PrivacyFailure("invalid_timestamp", "review timestamps must include a timezone")
    if not isinstance(package_token, str) or len(package_token) < 32:
        raise _PrivacyFailure("invalid_review_package", "review package token is invalid")


def _redacted_text(
    text: str,
    *,
    true_name_terms: tuple[str, ...],
    private_terms: PrivateRedactionTerms | None,
) -> str:
    replaced = replace_private_names(text, true_name_terms)
    try:
        redacted = redact_text(replaced, private_terms=private_terms).text
    except PrivacyValidationError as error:
        raise _PrivacyFailure(error.code, "redaction terms are invalid") from None
    if redacted is None or not redacted:
        raise _PrivacyFailure("invalid_redaction", "redaction result is invalid")
    return redacted


def _non_text_label(content_type: str) -> str:
    safe_type = content_type if content_type in _NON_TEXT_CONTENT_TYPES else "other"
    return f"[NON_TEXT:{safe_type}]"


def _context_item(
    record: ChatRecord,
    *,
    true_name_terms: tuple[str, ...],
    private_terms: PrivateRedactionTerms | None,
) -> ReviewContextItem:
    if record.content_type != "text" or record.text is None:
        return ReviewContextItem(
            role=record.speaker_role, text=_non_text_label(record.content_type)
        )
    return ReviewContextItem(
        role=record.speaker_role,
        text=_redacted_text(
            record.text,
            true_name_terms=true_name_terms,
            private_terms=private_terms,
        ),
    )


def _has_hard_risk_context(segment: list[ChatRecord], index: int) -> bool:
    start = max(0, index - _CONTEXT_RADIUS)
    end = min(len(segment), index + _CONTEXT_RADIUS + 1)
    return any(
        hard_rejection_reason(item.text or "") is not None
        for item in [*segment[start:index], *segment[index + 1 : end]]
    )


def _build_pool(
    records: list[ChatRecord],
    *,
    true_name_terms: tuple[str, ...],
    approved_nicknames: tuple[str, ...],
    private_terms: PrivateRedactionTerms | None,
    excluded_texts: frozenset[str],
) -> list[_PoolEntry]:
    occurrences: list[_PoolOccurrence] = []
    for _, segment in segment_records(records):
        for index, record in enumerate(segment):
            if record.speaker_role is not SpeakerRole.OWNER:
                continue
            if record.content_type != "text" or record.text is None:
                continue
            if hard_rejection_reason(record.text) is not None or _has_hard_risk_context(
                segment, index
            ):
                continue

            replaced = replace_private_names(record.text, true_name_terms)
            if allowed_category(classification_text(replaced, approved_nicknames)) is not None:
                continue
            candidate_text = _redacted_text(
                record.text,
                true_name_terms=true_name_terms,
                private_terms=private_terms,
            )
            if hard_rejection_reason(candidate_text) is not None:
                continue
            normalized_candidate = normalize_text(candidate_text)
            if not normalized_candidate or normalized_candidate in excluded_texts:
                continue

            context_before = tuple(
                _context_item(
                    item,
                    true_name_terms=true_name_terms,
                    private_terms=private_terms,
                )
                for item in segment[max(0, index - _CONTEXT_RADIUS) : index]
            )
            context_after = tuple(
                _context_item(
                    item,
                    true_name_terms=true_name_terms,
                    private_terms=private_terms,
                )
                for item in segment[index + 1 : index + _CONTEXT_RADIUS + 1]
            )
            occurrences.append(
                _PoolOccurrence(
                    sort_key=(record.timestamp, record.conversation_id, record.message_id),
                    normalized_text=normalized_candidate,
                    entry=_PoolEntry(
                        text=candidate_text,
                        context_before=context_before,
                        context_after=context_after,
                        similarity_group=_feature_group(candidate_text),
                        occurrence_count=1,
                    ),
                )
            )

    collapsed: dict[str, _PoolEntry] = {}
    for occurrence in sorted(occurrences, key=lambda item: item.sort_key):
        current = collapsed.get(occurrence.normalized_text)
        if current is None:
            collapsed[occurrence.normalized_text] = occurrence.entry
        else:
            collapsed[occurrence.normalized_text] = replace(
                current,
                occurrence_count=current.occurrence_count + 1,
            )
    return list(collapsed.values())


def _sample_round_robin(
    entries: Sequence[_PoolEntry] | Sequence[_RuleLinkedEntry],
    requested_count: int,
    *,
    prior: Sequence[_PoolEntry] | Sequence[_RuleLinkedEntry] = (),
) -> list[_PoolEntry] | list[_RuleLinkedEntry]:
    groups: dict[str, list[_PoolEntry] | list[_RuleLinkedEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.similarity_group, []).append(entry)
    for candidates in groups.values():
        candidates.sort(key=lambda item: normalize_text(item.text))

    selected: list[_PoolEntry] | list[_RuleLinkedEntry] = list(prior)
    added: list[_PoolEntry] | list[_RuleLinkedEntry] = []
    while groups and len(added) < requested_count:
        for group_key in sorted(tuple(groups)):
            candidates = groups[group_key]
            choice = min(
                candidates,
                key=lambda item: (
                    max(
                        (text_similarity(item.text, prior_item.text) for prior_item in selected),
                        default=0.0,
                    ),
                    normalize_text(item.text),
                ),
            )
            selected.append(choice)
            added.append(choice)
            candidates.remove(choice)
            if not candidates:
                del groups[group_key]
            if len(added) == requested_count:
                break
    return added


def _sample_rule_covered_matches(
    matches: Sequence[_RuleLinkedEntry], requested_count: int
) -> list[_RuleLinkedEntry]:
    by_rule: dict[str, list[_RuleLinkedEntry]] = {}
    for match in matches:
        by_rule.setdefault(match.rule_id, []).append(match)

    selected: list[_RuleLinkedEntry] = []
    for rule_id in sorted(by_rule):
        if len(selected) == requested_count:
            break
        candidates = by_rule[rule_id]
        choice = min(
            candidates,
            key=lambda item: (
                max(
                    (text_similarity(item.text, selected_item.text) for selected_item in selected),
                    default=0.0,
                ),
                normalize_text(item.text),
            ),
        )
        selected.append(choice)

    remaining = [match for match in matches if match not in selected]
    selected.extend(
        _sample_round_robin(
            remaining,
            requested_count - len(selected),
            prior=selected,
        )
    )
    return selected


def _review_item(entry: _PoolEntry, used_ids: set[str]) -> AmbiguousReviewItem:
    review_item_id = secrets.token_urlsafe(18)
    while review_item_id in used_ids:
        review_item_id = secrets.token_urlsafe(18)
    used_ids.add(review_item_id)
    return AmbiguousReviewItem(
        review_item_id=review_item_id,
        text=entry.text,
        context_before=list(entry.context_before),
        context_after=list(entry.context_after),
        similarity_group=entry.similarity_group,
        occurrence_count=entry.occurrence_count,
    )


def _content_fingerprint(items: Sequence[AmbiguousReviewItem]) -> str:
    minimized_items = [
        {
            "text": item.text,
            "context_before": [context.model_dump(mode="json") for context in item.context_before],
            "context_after": [context.model_dump(mode="json") for context in item.context_after],
            "similarity_group": item.similarity_group,
            "occurrence_count": item.occurrence_count,
        }
        for item in items
    ]
    canonical = json.dumps(
        minimized_items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_package(
    entries: Sequence[_PoolEntry],
    *,
    review_round: ReviewRound,
    created_at: datetime,
    package_token: str,
) -> AmbiguousReviewPackage:
    used_ids: set[str] = set()
    items = [_review_item(entry, used_ids) for entry in entries]
    return AmbiguousReviewPackage(
        package_token=package_token,
        review_round=review_round,
        created_at=created_at,
        content_fingerprint=_content_fingerprint(items),
        requested_count=_FIRST_ROUND_COUNT
        if review_round is ReviewRound.FIRST
        else _SECOND_ROUND_COUNT,
        item_count=len(items),
        items=items,
    )


def build_first_review_package(
    records: list[ChatRecord],
    *,
    created_at: datetime,
    package_token: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    private_terms: PrivateRedactionTerms | None,
    existing_candidate_texts: Sequence[str],
) -> AmbiguousReviewPackage:
    try:
        _validate_records(records)
        _validate_package_inputs(created_at, package_token)
        names = _normalise_terms(true_name_terms, required=True)
        nicknames = _normalise_terms(approved_nicknames, required=False)
        if set(names) & set(nicknames):
            raise _PrivacyFailure("invalid_candidate_terms", "candidate terms are invalid")
        pool = _build_pool(
            records,
            true_name_terms=names,
            approved_nicknames=nicknames,
            private_terms=private_terms,
            excluded_texts=_normalise_existing_candidate_texts(existing_candidate_texts),
        )
        selected = _sample_round_robin(pool, _FIRST_ROUND_COUNT)
        return _build_package(
            selected,
            review_round=ReviewRound.FIRST,
            created_at=created_at,
            package_token=package_token,
        )
    except _PrivacyFailure as error:
        failure_code, failure_detail = error.code, error.detail
    del records, created_at, package_token, true_name_terms, approved_nicknames, private_terms
    del existing_candidate_texts
    raise PrivacyValidationError(failure_code, failure_detail) from None


def _rule_link(entry: _PoolEntry, rules: Sequence[CalibrationRule]) -> _RuleLinkedEntry | None:
    links: list[tuple[bool, float, str]] = []
    for rule in rules:
        if entry.similarity_group != rule.similarity_group:
            continue
        similarity = text_similarity(entry.text, rule.normalized_template)
        links.append((similarity >= rule.similarity_threshold, similarity, rule.rule_id))
    if not links:
        return None
    is_match, _, rule_id = min(
        links,
        key=lambda value: (not value[0], -value[1], value[2]),
    )
    if not is_match and max(link[1] for link in links) == 0:
        return None
    return _RuleLinkedEntry(entry=entry, rule_id=rule_id, is_match=is_match)


def build_second_review_package(
    records: list[ChatRecord],
    *,
    proposal: CalibrationProposal,
    first_package: AmbiguousReviewPackage,
    created_at: datetime,
    package_token: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    private_terms: PrivateRedactionTerms | None,
    existing_candidate_texts: Sequence[str],
) -> tuple[AmbiguousReviewPackage, SecondRoundState]:
    try:
        _validate_records(records)
        _validate_package_inputs(created_at, package_token)
        if (
            first_package.review_round is not ReviewRound.FIRST
            or proposal.first_package_token != first_package.package_token
            or proposal.first_content_fingerprint != first_package.content_fingerprint
        ):
            raise _PrivacyFailure("stale_review_package", "first review package is invalid")

        names = _normalise_terms(true_name_terms, required=True)
        nicknames = _normalise_terms(approved_nicknames, required=False)
        if set(names) & set(nicknames):
            raise _PrivacyFailure("invalid_candidate_terms", "candidate terms are invalid")
        excluded_texts = _normalise_existing_candidate_texts(existing_candidate_texts) | frozenset(
            normalize_text(item.text) for item in first_package.items
        )
        pool = _build_pool(
            records,
            true_name_terms=names,
            approved_nicknames=nicknames,
            private_terms=private_terms,
            excluded_texts=excluded_texts,
        )
        linked = [link for entry in pool if (link := _rule_link(entry, proposal.rules)) is not None]
        matches = [link for link in linked if link.is_match]
        boundaries = [link for link in linked if not link.is_match]
        selected_matches = _sample_rule_covered_matches(matches, _SECOND_ROUND_COUNT)
        selected_boundaries = _sample_round_robin(
            boundaries,
            _SECOND_ROUND_COUNT - len(selected_matches),
            prior=selected_matches,
        )
        selected_links = [*selected_matches, *selected_boundaries]
        selected_entries = [link.entry for link in selected_links]
        package = _build_package(
            selected_entries,
            review_round=ReviewRound.SECOND,
            created_at=created_at,
            package_token=package_token,
        )
        state = SecondRoundState(
            first_package_token=first_package.package_token,
            first_content_fingerprint=first_package.content_fingerprint,
            second_package_token=package.package_token,
            second_content_fingerprint=package.content_fingerprint,
            item_count=package.item_count,
            links=[
                SecondRoundRuleLink(
                    review_item_id=item.review_item_id,
                    rule_id=link.rule_id,
                    is_match=link.is_match,
                )
                for item, link in zip(package.items, selected_links, strict=True)
            ],
        )
        return package, state
    except _PrivacyFailure as error:
        failure_code, failure_detail = error.code, error.detail
    del records, proposal, first_package, created_at, package_token, true_name_terms
    del approved_nicknames, private_terms, existing_candidate_texts
    raise PrivacyValidationError(failure_code, failure_detail) from None
