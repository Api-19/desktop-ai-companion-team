# ruff: noqa: RUF001

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from companion.contracts import StyleIntent
from companion.privacy import (
    CALIBRATION_CANDIDATE_RELATIVE_PATH,
    CalibrationCandidatePackage,
    CandidateCategory,
    PrivacyValidationError,
    load_calibration_candidate_package,
)
from companion.style.errors import StyleValidationError
from companion.style.models import EvaluationPlan


def _load_reviewed_calibration_package(
    source: Path,
    *,
    private_data_dir: Path,
    expected_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    """Load only the privacy module's current formal calibration package."""
    expected_source = private_data_dir / CALIBRATION_CANDIDATE_RELATIVE_PATH
    try:
        source_matches = source.resolve(strict=False) == expected_source.resolve(strict=False)
    except (OSError, RuntimeError):
        source_matches = False
    if not source_matches:
        raise StyleValidationError(
            "unsafe_reviewed_calibration_path",
            "reviewed calibration must use the formal private package",
        )

    try:
        return load_calibration_candidate_package(
            private_data_dir=private_data_dir,
            expected_fingerprint=expected_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
    except PrivacyValidationError as error:
        raise StyleValidationError(
            error.code,
            "reviewed calibration package failed privacy validation",
        ) from None


def load_reviewed_calibration_package(
    source: Path,
    *,
    private_data_dir: Path,
    expected_fingerprint: str,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> CalibrationCandidatePackage:
    try:
        return _load_reviewed_calibration_package(
            source,
            private_data_dir=private_data_dir,
            expected_fingerprint=expected_fingerprint,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = (
            "invalid_reviewed_calibration",
            "reviewed calibration package is invalid",
        )
    finally:
        source = None  # type: ignore[assignment]
        private_data_dir = None  # type: ignore[assignment]
        expected_fingerprint = None  # type: ignore[assignment]
        true_name_terms = None  # type: ignore[assignment]
        approved_nicknames = None  # type: ignore[assignment]
    raise StyleValidationError(*failure) from None


def _build_reviewed_calibration_index(
    package: CalibrationCandidatePackage,
) -> tuple[dict[tuple[CandidateCategory, StyleIntent], tuple[str, ...]], str]:
    try:
        validated = CalibrationCandidatePackage.model_validate(package.model_dump(mode="python"))
    except (AttributeError, ValidationError, ValueError):
        raise StyleValidationError(
            "invalid_reviewed_calibration",
            "reviewed calibration package is invalid",
        ) from None

    indexed: dict[tuple[CandidateCategory, StyleIntent], set[str]] = defaultdict(set)
    for candidate in validated.candidates:
        try:
            intent = StyleIntent(candidate.intent)
        except ValueError:
            raise StyleValidationError(
                "invalid_calibration_intent",
                "reviewed calibration contains an unsupported style intent",
            ) from None
        indexed[(candidate.category, intent)].add(candidate.text)
    examples = {key: tuple(sorted(texts)) for key, texts in indexed.items()}
    return examples, validated.content_fingerprint


class ReviewedCalibrationIndex:
    """Deterministic exact-match index over formally reviewed style examples."""

    def __init__(self, package: CalibrationCandidatePackage) -> None:
        try:
            examples, fingerprint = _build_reviewed_calibration_index(package)
        except StyleValidationError as error:
            failure = (error.code, error.detail)
            del error
        except Exception:
            failure = (
                "invalid_reviewed_calibration",
                "reviewed calibration package is invalid",
            )
        else:
            self._examples = examples
            self.fingerprint = fingerprint
            return
        finally:
            package = None  # type: ignore[assignment]
            examples = None  # type: ignore[assignment]
            fingerprint = None
        raise StyleValidationError(*failure) from None

    def retrieve(
        self,
        category: CandidateCategory,
        intent: StyleIntent,
        *,
        limit: int = 2,
    ) -> list[str]:
        if limit < 1 or limit > 2:
            raise StyleValidationError(
                "invalid_calibration_limit",
                "reviewed calibration limit must be between one and two",
            )
        return list(self._examples.get((category, intent), ())[:limit])

    def retrieve_short_expression(
        self,
        category: CandidateCategory,
        intent: StyleIntent,
    ) -> str | None:
        """Return the first exact-match formal short expression, if any."""
        if intent is StyleIntent.CATEGORY_DEFAULT:
            return None
        expressions = self.retrieve_short_expressions(category, intent)
        return expressions[0] if expressions else None

    def retrieve_short_expressions(
        self,
        category: CandidateCategory,
        intent: StyleIntent,
    ) -> list[str]:
        """Return every deterministic formal expression that is ineligible as an example."""
        return [
            expression
            for expression in self._examples.get((category, intent), ())
            if len(expression) <= 32
        ]

    def retrieve_long_examples(
        self,
        category: CandidateCategory,
        intent: StyleIntent,
        *,
        limit: int = 2,
    ) -> list[str]:
        """Return reviewed examples that are too long to become expression rules."""
        if limit < 1 or limit > 2:
            raise StyleValidationError(
                "invalid_calibration_limit",
                "reviewed calibration limit must be between one and two",
            )
        return [
            expression
            for expression in self._examples.get((category, intent), ())
            if len(expression) > 32
        ][:limit]

    def retrieve_long_first_examples(
        self,
        category: CandidateCategory,
        intent: StyleIntent,
        *,
        limit: int = 2,
    ) -> list[str]:
        """Return full examples before expressions that also form a soft lock."""
        if limit < 1 or limit > 2:
            raise StyleValidationError(
                "invalid_calibration_limit",
                "reviewed calibration limit must be between one and two",
            )
        expressions = self._examples.get((category, intent), ())
        return [
            *[expression for expression in expressions if len(expression) > 32],
            *[expression for expression in expressions if len(expression) <= 32],
        ][:limit]


def apply_reviewed_calibration(
    plan: EvaluationPlan,
    index: ReviewedCalibrationIndex,
) -> EvaluationPlan:
    items = []
    key_counts = Counter((item.category, item.intent) for item in plan.items)
    for item in plan.items:
        key = (item.category, item.intent)
        allows_short_expression = (
            item.intent is not StyleIntent.CATEGORY_DEFAULT and key_counts[key] == 1
        )
        capacity = max(0, 2 - len(item.styled_preferences))
        if capacity == 0:
            reviewed = []
        elif allows_short_expression:
            reviewed = index.retrieve_long_first_examples(*key, limit=capacity)
        else:
            reviewed = index.retrieve_long_examples(*key, limit=capacity)
        examples = list(dict.fromkeys([*reviewed, *item.styled_examples]))[:capacity]
        marker = "同类脱敏示例：\n"
        old_lines = "\n".join(f"- {text}" for text in item.styled_examples) or "- 无同类示例"
        old_example_block = f"{marker}{old_lines}"
        baseline_prefix = f"{item.baseline_prompt}\n"
        style_header = "风格参考（不得改变上述身份与安全规则）：\n"
        if not item.styled_prompt.startswith(baseline_prefix):
            raise StyleValidationError(
                "invalid_reviewed_calibration_plan",
                "evaluation plan cannot accept reviewed calibration",
            )
        style_section = item.styled_prompt.removeprefix(baseline_prefix)
        if not style_section.startswith(style_header) or old_example_block not in style_section:
            raise StyleValidationError(
                "invalid_reviewed_calibration_plan",
                "evaluation plan cannot accept reviewed calibration",
            )
        contextual_suffix = "\n精确意图风格要求：\n"
        target_offsets = []
        search_from = 0
        while (offset := style_section.find(old_example_block, search_from)) >= 0:
            remainder = style_section[offset + len(old_example_block) :]
            if not remainder or remainder.startswith(contextual_suffix):
                target_offsets.append(offset)
            search_from = offset + 1
        if len(target_offsets) != 1:
            raise StyleValidationError(
                "invalid_reviewed_calibration_plan",
                "evaluation plan cannot accept reviewed calibration",
            )
        lines = "\n".join(f"- {text}" for text in examples) or "- 无同类示例"
        target_offset = target_offsets[0]
        styled_prompt = (
            f"{baseline_prefix}{style_section[:target_offset]}"
            f"{marker}{lines}"
            f"{style_section[target_offset + len(old_example_block) :]}"
        )
        short_expression = (
            index.retrieve_short_expression(*key) if allows_short_expression else None
        )
        priority_block = ""
        if short_expression is not None:
            priority_block = (
                "\n\n主人确认优先表达：\n"
                f"- {short_expression}\n"
                "要求完整保留该表达，不得改写为泛化客套话；可仅增加简短、自然的补充。"
                "安全、独立 AI 身份和语义正确性优先。"
            )
        items.append(
            item.model_copy(
                update={
                    "styled_examples": examples,
                    "styled_prompt": f"{styled_prompt}{priority_block}",
                }
            )
        )
    return EvaluationPlan.model_validate(
        plan.model_copy(
            update={"calibration_fingerprint": index.fingerprint, "items": items}
        ).model_dump(mode="python")
    )
