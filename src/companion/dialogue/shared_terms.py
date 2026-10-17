"""Owner-approved shared-terms dictionary: load and match terms the partner just used."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, model_validator

MAX_MATCHED_TERMS = 8


class SharedTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    term: StrictStr = Field(min_length=1)
    meaning: StrictStr = Field(min_length=1)
    occasion: StrictStr
    speaker: Literal["owner", "partner", "both"]
    addressee: StrictStr
    permission: Literal["allowed", "understand_only", "disabled"]
    exclude_within: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def validate_exclude_within(self) -> Self:
        if any(not word or self.term not in word for word in self.exclude_within):
            raise ValueError("exclude_within items must be nonempty and contain term")
        return self


class _SharedTermsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    entries: tuple[SharedTerm, ...]


def load_shared_terms(path: Path) -> tuple[SharedTerm, ...]:
    try:
        return _SharedTermsFile.model_validate_json(path.read_bytes()).entries
    except (OSError, ValidationError):
        raise ValueError("shared terms configuration unavailable") from None


def shared_term_query_texts(text: str, recent_context: Sequence[str]) -> list[str]:
    partner_context = [
        item.removeprefix("对方：") for item in recent_context if not item.startswith("你：")  # noqa: RUF001
    ]
    return [text, *partner_context[-2:]]


def match_shared_terms(
    terms: Sequence[SharedTerm], texts: Sequence[str]
) -> tuple[SharedTerm, ...]:
    matched = tuple(
        entry
        for entry in terms
        if entry.permission != "disabled" and any(_matches_text(entry, text) for text in texts)
    )
    return matched[:MAX_MATCHED_TERMS]


def _matches_text(entry: SharedTerm, text: str) -> bool:
    if not entry.exclude_within:
        return entry.term in text
    excluded_spans = [
        (start, start + len(word))
        for word in entry.exclude_within
        for start in _occurrences(text, word)
    ]
    return any(
        not any(
            left <= start and start + len(entry.term) <= right for left, right in excluded_spans
        )
        for start in _occurrences(text, entry.term)
    )


def _occurrences(text: str, term: str) -> Iterator[int]:
    start = text.find(term)
    while start != -1:
        yield start
        start = text.find(term, start + 1)
