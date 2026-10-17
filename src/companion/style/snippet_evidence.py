"""Local BM25 retrieval and length- or line-stratified paired-snippet evidence."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, deque
from collections.abc import Callable, Collection, Sequence
from datetime import datetime
from typing import Literal

from companion.contracts.style import AddresseeRegister, StyleEvidence, StyleSnippet


def _tokens(text: str) -> list[str]:
    compact = "".join(text.split())
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[index : index + 2] for index in range(len(compact) - 1)]


class SnippetIndex:
    """Index partner turns using character bigrams and BM25 (k1=1.5, b=0.75)."""

    def __init__(self, snippets: Sequence[StyleSnippet]) -> None:
        self._documents = [Counter(_tokens("\n".join(item.partner_turns))) for item in snippets]
        self._lengths = [sum(document.values()) for document in self._documents]
        count = len(self._documents)
        self._average_length = sum(self._lengths) / count if count else 0.0
        frequencies = Counter(term for document in self._documents for term in document)
        self._idf = {
            term: math.log((count - frequency + 0.5) / (frequency + 0.5) + 1)
            for term, frequency in frequencies.items()
        }

    def search(self, query: str, *, k: int, exclude: Collection[int] = ()) -> list[int]:
        """Return positive-scoring indices, breaking score ties by input order."""
        if k <= 0 or not self._average_length:
            return []
        terms = set(_tokens(query))
        excluded = set(exclude)
        scored = []
        for index, document in enumerate(self._documents):
            if index in excluded:
                continue
            normalization = 1.5 * (1 - 0.75 + 0.75 * self._lengths[index] / self._average_length)
            score = sum(
                self._idf[term] * document[term] * 2.5 / (document[term] + normalization)
                for term in sorted(terms & document.keys())
            )
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [index for _, index in scored[:k]]


class SnippetStyleAdapter:
    """Provide partner-only evidence with a three-request reuse exclusion window."""

    def __init__(
        self,
        *,
        snippets: Sequence[StyleSnippet],
        partner_user_id: str,
        style_version: str,
        clock: Callable[[], datetime],
        general_strata: Literal["length", "lines"] = "length",
    ) -> None:
        if general_strata not in ("length", "lines"):
            raise ValueError("general_strata must be 'length' or 'lines'")
        if type(partner_user_id) is not str:
            raise TypeError("partner_user_id must be exact builtin str")
        if not partner_user_id.strip() or len(partner_user_id) > 120:
            raise ValueError("partner_user_id must contain 1 to 120 non-whitespace characters")
        unique: dict[tuple[tuple[str, ...], tuple[str, ...]], StyleSnippet] = {}
        for snippet in snippets:
            key = (tuple(snippet.partner_turns), tuple(snippet.owner_reply))
            unique.setdefault(key, snippet)
        self._snippets = tuple(unique.values())
        self._index = SnippetIndex(self._snippets)
        self._partner_user_id = partner_user_id
        self._style_version = style_version
        self._clock = clock
        self._general_strata = general_strata
        self._recent: dict[str, deque[set[int]]] = {}

    def accepts_user(self, user_id: str) -> bool:
        """Accept every exact builtin string ID valid under the adapter ID rules."""
        return type(user_id) is str and bool(user_id.strip()) and len(user_id) <= 120

    def evidence_for(self, user_id: str, text: str, recent_context: Sequence[str]) -> StyleEvidence:
        """Return local evidence; never expose partner snippets to other users."""
        if user_id != self._partner_user_id:
            return StyleEvidence(
                style_version=self._style_version, addressee_register=AddresseeRegister.NEUTRAL
            )

        history = self._recent.setdefault(user_id, deque(maxlen=3))
        excluded = set().union(*history)
        query = "\n".join([*recent_context[-2:], text])
        similar = self._index.search(query, k=6, exclude=excluded)
        unavailable = excluded | set(similar)
        if self._general_strata == "length":
            candidates = sorted(
                (index for index in range(len(self._snippets)) if index not in unavailable),
                key=lambda index: len("".join(self._snippets[index].owner_reply)),
            )
            width = len(candidates) // 3
            strata = [candidates[:width], candidates[width : 2 * width], candidates[2 * width :]]
        else:
            strata = [[], [], []]
            for index, snippet in enumerate(self._snippets):
                if index not in unavailable:
                    strata[min(len(snippet.owner_reply), 3) - 1].append(index)
        seed_text = f"{self._clock().date().isoformat()}\x1f{text}"
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        general = []
        for group in strata:
            general.extend(rng.sample(group, 2) if len(group) >= 2 else group)
        evidence = StyleEvidence(
            style_version=self._style_version,
            addressee_register=AddresseeRegister.INTIMATE,
            similar=[self._snippets[index] for index in similar],
            general=[self._snippets[index] for index in general],
        )
        history.append(set(similar + general))
        return evidence
