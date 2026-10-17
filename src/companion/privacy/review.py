from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from companion.contracts.chat import ChatRecord, SpeakerRole
from companion.privacy.errors import PrivacyValidationError, _PrivacyFailure
from companion.privacy.models import (
    PrivacyReviewPackage,
    ReviewDecisionValue,
    ReviewSegment,
    RiskCategory,
)

PrivateTerms = Mapping[RiskCategory, Sequence[str]]

_SEGMENT_GAP = timedelta(minutes=30)
_REVIEW_CONTEXT_MESSAGES = 5
_CATEGORY_ORDER = (
    RiskCategory.CREDENTIAL,
    RiskCategory.FINANCIAL,
    RiskCategory.INTIMATE,
    RiskCategory.SEXUAL,
    RiskCategory.HEALTH,
    RiskCategory.FAMILY,
    RiskCategory.CONFLICT,
    RiskCategory.LOCATION,
    RiskCategory.PRIVATE_EVENT,
)
_TOPIC_KEYWORDS: dict[RiskCategory, tuple[str, ...]] = {
    RiskCategory.INTIMATE: ("亲密关系", "亲密", "恋爱", "爱情", "想你", "爱你", "情侣", "纪念日"),
    RiskCategory.SEXUAL: ("性生活", "性行为", "做爱", "避孕", "怀孕"),
    RiskCategory.HEALTH: ("医院", "生病", "体检", "复查", "手术", "诊断", "症状", "吃药"),
    RiskCategory.FAMILY: ("家庭", "父母", "妈妈", "爸爸", "家人", "婆婆", "岳父"),
    RiskCategory.FINANCIAL: ("银行卡", "余额", "工资", "借钱", "欠款", "转账", "存款", "贷款"),
    RiskCategory.CREDENTIAL: ("密码", "验证码", "登录", "账号", "密钥", "令牌"),
    RiskCategory.CONFLICT: ("吵架", "争执", "冲突", "分手", "生气", "道歉", "矛盾"),
    RiskCategory.LOCATION: ("详细地址", "地址", "住址", "位置", "定位", "行程", "酒店"),
    RiskCategory.PRIVATE_EVENT: ("私人事件", "秘密", "隐私", "不要告诉", "保密"),
}
_STRUCTURED_PATTERNS: tuple[tuple[RiskCategory, str, re.Pattern[str]], ...] = (
    (
        RiskCategory.CREDENTIAL,
        "email_pattern",
        re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    ),
    (
        RiskCategory.CREDENTIAL,
        "phone_pattern",
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    ),
    (
        RiskCategory.CREDENTIAL,
        "national_id_pattern",
        re.compile(r"(?<!\d)\d{17}[\dXx](?!\w)"),
    ),
    (
        RiskCategory.FINANCIAL,
        "bank_card_pattern",
        re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)"),
    ),
    (
        RiskCategory.LOCATION,
        "ip_address_pattern",
        re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
        ),
    ),
)


def _validate_records(records: list[ChatRecord]) -> None:
    if not records:
        raise _PrivacyFailure("empty_records", "record batch must not be empty")

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
        message_ids.add(record.message_id)


def _normalise_private_terms(
    private_terms: PrivateTerms | None,
) -> dict[RiskCategory, tuple[str, ...]]:
    if private_terms is None:
        return {}

    normalised: dict[RiskCategory, tuple[str, ...]] = {}
    for category, terms in private_terms.items():
        if not isinstance(category, RiskCategory) or isinstance(terms, (str, bytes)):
            raise _PrivacyFailure(
                "invalid_private_terms",
                "private-term categories or values are invalid",
            )
        values = tuple(terms)
        if not values or any(not isinstance(term, str) or not term.strip() for term in values):
            raise _PrivacyFailure(
                "invalid_private_terms",
                "private-term categories or values are invalid",
            )
        normalised[category] = values
    return normalised


def segment_records(records: list[ChatRecord]) -> list[tuple[str, list[ChatRecord]]]:
    ordered = sorted(
        records,
        key=lambda record: (record.conversation_id, record.timestamp, record.message_id),
    )
    segments: list[list[ChatRecord]] = []
    current: list[ChatRecord] = []

    for record in ordered:
        if current and (
            record.conversation_id != current[-1].conversation_id
            or record.timestamp - current[-1].timestamp > _SEGMENT_GAP
        ):
            segments.append(current)
            current = []
        current.append(record)
    if current:
        segments.append(current)

    return [(f"segment-{index:06d}", segment) for index, segment in enumerate(segments, start=1)]


def dataset_fingerprint(records: list[ChatRecord]) -> str:
    canonical = json.dumps(
        [record.model_dump(mode="json") for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _detect_record_risks(
    record: ChatRecord,
    private_terms: dict[RiskCategory, tuple[str, ...]],
) -> tuple[list[RiskCategory], list[str]]:
    found_categories: set[RiskCategory] = set()
    found_reasons: set[str] = set()

    if record.text is not None:
        text = record.text
        for category in _CATEGORY_ORDER:
            if any(keyword in text for keyword in _TOPIC_KEYWORDS[category]):
                found_categories.add(category)
                found_reasons.add(f"{category.value}_keyword")
            if any(term in text for term in private_terms.get(category, ())):
                found_categories.add(category)
                found_reasons.add("private_term_match")
        for category, reason, pattern in _STRUCTURED_PATTERNS:
            if pattern.search(text):
                found_categories.add(category)
                found_reasons.add(reason)

    categories = [category for category in _CATEGORY_ORDER if category in found_categories]
    reason_order = [
        f"{category.value}_keyword"
        for category in _CATEGORY_ORDER
        if f"{category.value}_keyword" in found_reasons
    ]
    reason_order.extend(reason for _, reason, _ in _STRUCTURED_PATTERNS if reason in found_reasons)
    if "private_term_match" in found_reasons:
        reason_order.append("private_term_match")
    return categories, reason_order


def _risk_windows(
    records: list[ChatRecord],
    private_terms: dict[RiskCategory, tuple[str, ...]],
) -> list[tuple[int, int, list[RiskCategory], list[str]]]:
    detections = [_detect_record_risks(record, private_terms) for record in records]
    intervals = [
        (
            max(0, index - _REVIEW_CONTEXT_MESSAGES),
            min(len(records), index + _REVIEW_CONTEXT_MESSAGES + 1),
        )
        for index, (categories, _) in enumerate(detections)
        if categories
    ]
    if not intervals:
        return []

    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    windows: list[tuple[int, int, list[RiskCategory], list[str]]] = []
    for start, end in merged:
        found_categories = {
            category for categories, _ in detections[start:end] for category in categories
        }
        found_reasons = {reason for _, reasons in detections[start:end] for reason in reasons}
        categories = [category for category in _CATEGORY_ORDER if category in found_categories]
        reasons = [
            f"{category.value}_keyword"
            for category in _CATEGORY_ORDER
            if f"{category.value}_keyword" in found_reasons
        ]
        reasons.extend(reason for _, reason, _ in _STRUCTURED_PATTERNS if reason in found_reasons)
        if "private_term_match" in found_reasons:
            reasons.append("private_term_match")
        windows.append((start, end, categories, reasons))
    return windows


def _scan_for_review(
    records: list[ChatRecord],
    *,
    created_at: datetime,
    private_terms: PrivateTerms | None = None,
) -> PrivacyReviewPackage:
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise _PrivacyFailure(
            "invalid_timestamp",
            "review timestamps must include a timezone",
        )
    _validate_records(records)
    terms = _normalise_private_terms(private_terms)
    segments = segment_records(records)

    risk_segments: list[ReviewSegment] = []
    for source_segment_id, segment in segments:
        for window_index, (start, end, categories, reasons) in enumerate(
            _risk_windows(segment, terms),
            start=1,
        ):
            risk_segments.append(
                ReviewSegment(
                    segment_id=f"{source_segment_id}-risk-{window_index:03d}",
                    source_segment_id=source_segment_id,
                    records=segment[start:end],
                    risk_categories=categories,
                    reason_labels=reasons,
                    recommended_decision=ReviewDecisionValue.EXCLUDE,
                )
            )

    return PrivacyReviewPackage(
        dataset_fingerprint=dataset_fingerprint(records),
        scanned_segment_count=len(segments),
        risk_segments=risk_segments,
        created_at=created_at,
    )


def scan_for_review(
    records: list[ChatRecord],
    *,
    created_at: datetime,
    private_terms: PrivateTerms | None = None,
) -> PrivacyReviewPackage:
    try:
        return _scan_for_review(
            records,
            created_at=created_at,
            private_terms=private_terms,
        )
    except _PrivacyFailure as error:
        failure = (error.code, error.detail)
    del records, created_at, private_terms
    raise PrivacyValidationError(failure[0], failure[1]) from None
