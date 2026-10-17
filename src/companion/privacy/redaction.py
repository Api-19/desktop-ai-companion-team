from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from companion.privacy.errors import PrivacyValidationError, _PrivacyFailure

PrivateRedactionTerms = Mapping[str, Sequence[str]]

_LABEL_ORDER = (
    "EMAIL",
    "PHONE",
    "NATIONAL_ID",
    "BANK_CARD",
    "IP_ADDRESS",
    "NAME",
    "ADDRESS",
    "ACCOUNT",
    "PRIVATE_EVENT",
    "URL",
    "LONG_NUMBER",
    "PLATE",
)
_PRIVATE_LABELS = {
    "name": "NAME",
    "address": "ADDRESS",
    "account": "ACCOUNT",
    "private_event": "PRIVATE_EVENT",
}
_STRUCTURED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL",
        re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    ),
    ("PHONE", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("NATIONAL_ID", re.compile(r"(?<!\d)\d{17}[\dXx](?!\w)")),
    ("BANK_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)")),
    (
        "IP_ADDRESS",
        re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
        ),
    ),
    ("URL", re.compile(r"(?i)\b(?:https?://|www\.)[^\s\uff0c。\uff01\uff1f、]+")),
    ("LONG_NUMBER", re.compile(r"(?<![\d.])\d{7,}(?![\d.])")),
    (
        "PLATE",
        re.compile(
            r"(?:尾号|车牌|车号)\s*[A-Za-z0-9]{3,6}"
            r"|[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5,6}"
        ),
    ),
)


@dataclass(frozen=True)
class RedactionResult:
    text: str | None
    labels: tuple[str, ...]
    counts: dict[str, int]


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    label: str
    priority: int


def _normalise_private_terms(
    private_terms: PrivateRedactionTerms | None,
) -> dict[str, tuple[str, ...]]:
    if private_terms is None:
        return {}

    normalised: dict[str, tuple[str, ...]] = {}
    for category, terms in private_terms.items():
        if category not in _PRIVATE_LABELS or isinstance(terms, (str, bytes)):
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


def _collect_candidates(
    text: str,
    private_terms: dict[str, tuple[str, ...]],
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    priorities = {label: index for index, label in enumerate(_LABEL_ORDER)}

    for label, pattern in _STRUCTURED_PATTERNS:
        candidates.extend(
            _Candidate(match.start(), match.end(), label, priorities[label])
            for match in pattern.finditer(text)
        )

    for category, terms in private_terms.items():
        label = _PRIVATE_LABELS[category]
        for term in sorted(set(terms), key=lambda item: (-len(item), item)):
            candidates.extend(
                _Candidate(match.start(), match.end(), label, priorities[label])
                for match in re.finditer(re.escape(term), text)
            )
    return candidates


def _select_non_overlapping(candidates: list[_Candidate]) -> list[_Candidate]:
    selected: list[_Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.label in {"URL", "LONG_NUMBER", "PLATE"},
            item.start,
            -(item.end - item.start),
            item.priority,
        ),
    ):
        if any(candidate.start < item.end and item.start < candidate.end for item in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.start)


def _redact_text(
    text: str | None,
    *,
    private_terms: PrivateRedactionTerms | None = None,
) -> RedactionResult:
    terms = _normalise_private_terms(private_terms)
    if text is None:
        return RedactionResult(text=None, labels=(), counts={})

    selected = _select_non_overlapping(_collect_candidates(text, terms))
    if not selected:
        return RedactionResult(text=text, labels=(), counts={})

    parts: list[str] = []
    cursor = 0
    counts: Counter[str] = Counter()
    for candidate in selected:
        parts.append(text[cursor : candidate.start])
        parts.append(f"[{candidate.label}]")
        cursor = candidate.end
        counts[candidate.label] += 1
    parts.append(text[cursor:])

    ordered_counts = {label: counts[label] for label in _LABEL_ORDER if counts[label]}
    return RedactionResult(
        text="".join(parts),
        labels=tuple(ordered_counts),
        counts=ordered_counts,
    )


def redact_text(
    text: str | None,
    *,
    private_terms: PrivateRedactionTerms | None = None,
) -> RedactionResult:
    try:
        return _redact_text(text, private_terms=private_terms)
    except _PrivacyFailure as error:
        failure = (error.code, error.detail)
    del text, private_terms
    raise PrivacyValidationError(failure[0], failure[1]) from None
