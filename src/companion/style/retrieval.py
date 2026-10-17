from collections import defaultdict

from companion.privacy import CandidateCategory, SafeCandidatePackage
from companion.style.models import CoverageStatus, RetrievalResult
from companion.style.semantic_routing import derive_calibrated_category


def _character_bigrams(text: str) -> set[str]:
    compact = "".join(character for character in text if not character.isspace())
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _coverage_status(count: int) -> CoverageStatus:
    if count >= 3:
        return CoverageStatus.TESTABLE
    if count == 2:
        return CoverageStatus.SPARSE
    if count == 1:
        return CoverageStatus.MECHANICAL_RISK
    return CoverageStatus.MISSING


class StyleExampleIndex:
    @classmethod
    def from_package(cls, package: SafeCandidatePackage) -> "StyleExampleIndex":
        return cls(package)

    def __init__(self, package: SafeCandidatePackage) -> None:
        by_category: dict[CandidateCategory, list[tuple[int, str, set[str]]]] = defaultdict(list)
        for position, candidate in enumerate(package.candidates):
            item = (position, candidate.text, _character_bigrams(candidate.text))
            by_category[candidate.category].append(item)
            if candidate.category is CandidateCategory.CALIBRATED_DAILY:
                derived = derive_calibrated_category(candidate.text)
                if derived is not None:
                    by_category[derived].append(item)
        self._by_category = dict(by_category)

    def retrieve(
        self,
        query: str,
        category: CandidateCategory,
        *,
        limit: int = 2,
    ) -> RetrievalResult:
        if limit not in (1, 2):
            raise ValueError("retrieval limit must be one or two")
        candidates = self._by_category.get(category, [])
        query_bigrams = _character_bigrams(query)
        ranked = sorted(
            candidates,
            key=lambda item: (-_jaccard(query_bigrams, item[2]), item[0]),
        )
        examples: list[str] = []
        seen: set[str] = set()
        for _, text, _ in ranked:
            if text in seen:
                continue
            seen.add(text)
            examples.append(text)
            if len(examples) == limit:
                break
        distinct_count = len({text for _, text, _ in candidates})
        return RetrievalResult(
            examples=examples,
            coverage_status=_coverage_status(distinct_count),
            category_count=distinct_count,
        )
