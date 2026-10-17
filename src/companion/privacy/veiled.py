"""Load and scan the local, two-tier veiled-expression lexicon."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from companion.contracts.base import ContractModel

_Terms = dict[str, list[Annotated[str, Field(min_length=1)]]]
_SEPARATORS = r"[\s.*\-_~·]*"


def _normalize(text: str) -> str:
    return "".join(
        character for character in unicodedata.normalize("NFKC", text).lower()
        if unicodedata.category(character) != "Cf"
    )


class _Variants(ContractModel):
    allow_separators: bool = False
    pinyin_initials: _Terms = Field(default_factory=dict)


class _LexiconFile(ContractModel):
    version: Literal[1]
    note: str = ""
    high: _Terms
    suspect: _Terms
    variants: _Variants = Field(default_factory=_Variants)

    @model_validator(mode="after")
    def validate_terms(self) -> _LexiconFile:
        terms = {term for tier in (self.high, self.suspect) for group in tier.values()
                 for term in group}
        if self.variants.pinyin_initials.keys() - terms:
            raise ValueError("aliases must refer to a configured term")
        for group in (*self.high.values(), *self.suspect.values(),
                      *self.variants.pinyin_initials.values()):
            if not group or any(not _normalize(term).strip() for term in group):
                raise ValueError("terms must not be blank")
        return self


@dataclass(frozen=True)
class VeiledHits:
    high: tuple[str, ...]
    suspect: tuple[str, ...]


def _compile(term: str, allow_separators: bool) -> re.Pattern[str]:
    normalized = _normalize(term)
    emoji = any(0x1F000 <= ord(char) <= 0x1FAFF or 0x2600 <= ord(char) <= 0x27BF
                for char in normalized)
    pattern = (_SEPARATORS.join(re.escape(char) for char in normalized)
               if allow_separators and len(normalized) >= 2 and not emoji
               else re.escape(normalized))
    return re.compile(pattern)


@dataclass(frozen=True)
class VeiledLexicon:
    _high: tuple[tuple[str, re.Pattern[str]], ...]
    _suspect: tuple[tuple[str, re.Pattern[str]], ...]

    def scan(self, text: str) -> VeiledHits:
        """Return sorted canonical category:term hits after normalization."""
        normalized = _normalize(text)
        return VeiledHits(
            high=tuple(sorted({label for label, pattern in self._high
                               if pattern.search(normalized)})),
            suspect=tuple(sorted({label for label, pattern in self._suspect
                                  if pattern.search(normalized)})),
        )


def load_veiled_terms(path: Path) -> VeiledLexicon:
    """Validate the versioned JSON file and compile terms and their aliases."""
    payload = _LexiconFile.model_validate_json(path.read_text(encoding="utf-8-sig"))
    tiers = []
    for tier in (payload.high, payload.suspect):
        patterns = []
        for category, terms in tier.items():
            for term in terms:
                for spelling in (term, *payload.variants.pinyin_initials.get(term, [])):
                    patterns.append((f"{category}:{term}",
                                     _compile(spelling, payload.variants.allow_separators)))
        tiers.append(tuple(patterns))
    return VeiledLexicon(tiers[0], tiers[1])
