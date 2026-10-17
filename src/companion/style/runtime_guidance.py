from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from companion.contracts import (
    StyleGuidanceStatus,
    StyleIntent,
    StyleReplyGuidance,
)
from companion.privacy import (
    CandidateCategory,
    SafeCandidatePackage,
    validate_safe_candidate_runtime_material,
)
from companion.style.adjustments import candidate_package_fingerprint
from companion.style.calibration import (
    ReviewedCalibrationIndex,
    load_reviewed_calibration_package,
)
from companion.style.errors import StyleValidationError
from companion.style.iteration_pipeline import load_verified_active_style_iteration
from companion.style.loader import load_safe_candidate_package
from companion.style.retrieval import StyleExampleIndex
from companion.style.runtime_classification import classify_style_reply

SAFE_CANDIDATE_RELATIVE_PATH = Path("style_candidates/current/safe-candidates.json")


def _valid_runtime_input(
    text: object,
    recent_context: object,
    idle_seconds: object,
) -> bool:
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2_000:
        return False
    if (
        not isinstance(recent_context, Sequence)
        or isinstance(recent_context, (str, bytes))
        or len(recent_context) > 4
    ):
        return False
    if any(
        not isinstance(item, str) or not 1 <= len(item.strip()) <= 2_000 for item in recent_context
    ):
        return False
    return idle_seconds is None or (
        isinstance(idle_seconds, int) and not isinstance(idle_seconds, bool) and idle_seconds >= 0
    )


def _uses_formal_candidate_path(candidate_source: Path, private_data_dir: Path) -> bool:
    try:
        expected_source = private_data_dir / SAFE_CANDIDATE_RELATIVE_PATH
        return candidate_source.resolve(strict=False) == expected_source.resolve(strict=False)
    except (AttributeError, OSError, RuntimeError, TypeError):
        return False


def _runtime_version(
    dataset_fingerprint: str,
    manifest_digest: str,
    style_version: str,
    calibration_fingerprint: str,
) -> str:
    payload = json.dumps(
        [
            "companion-style-runtime-v1",
            dataset_fingerprint,
            manifest_digest,
            style_version,
            calibration_fingerprint,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"dev-style-runtime-{digest}"


def _matching_preference(
    style_pack: object,
    category: CandidateCategory,
    intent: StyleIntent,
) -> str | None:
    if intent in {
        StyleIntent.CATEGORY_DEFAULT,
        StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
    }:
        return None
    preferences = sorted(style_pack.preferences, key=lambda preference: preference.rule_id)
    for preference in preferences:
        if (
            preference.category == category.value
            and intent in preference.intent_tags
            and preference.max_consecutive_uses > 0
        ):
            return preference.expression
    return None


def _normalized_text(text: str) -> str | None:
    normalized = text.strip()
    return normalized or None


def _normalized_unique_texts(
    texts: Sequence[str],
    *,
    excluded: set[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    normalized_texts: list[str] = []
    seen = set(excluded or ())
    for text in texts:
        normalized = _normalized_text(text)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        normalized_texts.append(normalized)
        if limit is not None and len(normalized_texts) == limit:
            break
    return normalized_texts


def _normalized_candidate_view(
    package: SafeCandidatePackage,
    *,
    excluded: set[str],
) -> SafeCandidatePackage:
    candidates = []
    for candidate in package.candidates:
        normalized = _normalized_text(candidate.text)
        if normalized is None or normalized in excluded:
            continue
        candidates.append(candidate.model_copy(update={"text": normalized}))
    return package.model_copy(update={"candidates": candidates})


def _build_style_reply_guidance(
    text: str,
    recent_context: Sequence[str],
    idle_seconds: int | None,
    *,
    user_id: str,
    candidate_source: Path,
    expected_count: int,
    active_style_source: Path,
    active_manifest_source: Path,
    expected_active_manifest_digest: str,
    calibration_source: Path,
    expected_calibration_fingerprint: str,
    private_data_dir: Path,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> StyleReplyGuidance:
    if not _uses_formal_candidate_path(candidate_source, private_data_dir):
        raise StyleValidationError(
            "unsafe_runtime_candidate_path",
            "runtime style candidates must use the formal private package",
        )
    package = load_safe_candidate_package(
        candidate_source,
        private_data_dir=private_data_dir,
        expected_count=expected_count,
    )
    validate_safe_candidate_runtime_material(
        package,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    dataset_fingerprint = candidate_package_fingerprint(package)
    style_pack, verified_manifest_digest = load_verified_active_style_iteration(
        manifest_source=active_manifest_source,
        candidate_source=active_style_source,
        private_data_dir=private_data_dir,
        dataset_fingerprint=dataset_fingerprint,
        expected_manifest_digest=expected_active_manifest_digest,
    )
    reviewed_package = load_reviewed_calibration_package(
        calibration_source,
        private_data_dir=private_data_dir,
        expected_fingerprint=expected_calibration_fingerprint,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    reviewed_index = ReviewedCalibrationIndex(reviewed_package)
    version = _runtime_version(
        dataset_fingerprint,
        verified_manifest_digest,
        style_pack.version,
        reviewed_package.content_fingerprint,
    )

    match = classify_style_reply(text, recent_context, idle_seconds)
    if match is None:
        return StyleReplyGuidance(
            status=StyleGuidanceStatus.NO_MATCH,
            style_version=version,
        )

    formal_examples = _normalized_unique_texts(
        reviewed_index.retrieve_long_examples(
            match.category,
            match.intent,
            limit=2,
        ),
    )
    formal_short_expressions = set(
        _normalized_unique_texts(
            reviewed_index.retrieve_short_expressions(match.category, match.intent)
        )
    )
    soft_lock = None
    if match.intent is not StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT:
        soft_lock = reviewed_index.retrieve_short_expression(match.category, match.intent)
        if soft_lock is None:
            soft_lock = _matching_preference(style_pack, match.category, match.intent)
    soft_lock = _normalized_text(soft_lock) if soft_lock is not None else None
    excluded_examples = formal_short_expressions | ({soft_lock} if soft_lock else set())
    excluded_base_examples = excluded_examples | set(formal_examples)
    eligible_package = _normalized_candidate_view(
        package,
        excluded=excluded_base_examples,
    )
    base_examples = (
        StyleExampleIndex.from_package(eligible_package)
        .retrieve(
            text,
            match.category,
            limit=2,
        )
        .examples
    )

    examples = _normalized_unique_texts(
        (*formal_examples, *base_examples),
        excluded=excluded_examples,
        limit=2,
    )
    if not examples and soft_lock is None:
        return StyleReplyGuidance(
            status=StyleGuidanceStatus.NO_MATCH,
            style_version=version,
        )
    return StyleReplyGuidance(
        status=StyleGuidanceStatus.MATCHED,
        style_version=version,
        category=match.category.value,
        intent=match.intent,
        examples=examples,
        soft_lock_expression=soft_lock,
    )


def build_style_reply_guidance(
    text: str,
    recent_context: Sequence[str],
    idle_seconds: int | None,
    *,
    user_id: str,
    candidate_source: Path,
    expected_count: int,
    active_style_source: Path,
    active_manifest_source: Path,
    expected_active_manifest_digest: str,
    calibration_source: Path,
    expected_calibration_fingerprint: str,
    private_data_dir: Path,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
) -> StyleReplyGuidance:
    """Build validated, fail-closed runtime style guidance from formal resources."""
    try:
        input_is_valid = _valid_runtime_input(text, recent_context, idle_seconds)
    except Exception:
        input_is_valid = False
    if not input_is_valid:
        text = None  # type: ignore[assignment]
        recent_context = None  # type: ignore[assignment]
        idle_seconds = None
        user_id = None  # type: ignore[assignment]
        candidate_source = None  # type: ignore[assignment]
        active_style_source = None  # type: ignore[assignment]
        active_manifest_source = None  # type: ignore[assignment]
        expected_active_manifest_digest = None  # type: ignore[assignment]
        calibration_source = None  # type: ignore[assignment]
        expected_calibration_fingerprint = None  # type: ignore[assignment]
        private_data_dir = None  # type: ignore[assignment]
        true_name_terms = None  # type: ignore[assignment]
        approved_nicknames = None  # type: ignore[assignment]
        raise StyleValidationError(
            "invalid_style_guidance_input",
            "style guidance input is invalid",
        ) from None

    if type(user_id) is not str or not user_id.strip() or len(user_id) > 120:
        user_id = None  # type: ignore[assignment]
        text = None  # type: ignore[assignment]
        recent_context = None  # type: ignore[assignment]
        idle_seconds = None
        candidate_source = None  # type: ignore[assignment]
        active_style_source = None  # type: ignore[assignment]
        active_manifest_source = None  # type: ignore[assignment]
        expected_active_manifest_digest = None  # type: ignore[assignment]
        calibration_source = None  # type: ignore[assignment]
        expected_calibration_fingerprint = None  # type: ignore[assignment]
        private_data_dir = None  # type: ignore[assignment]
        true_name_terms = None  # type: ignore[assignment]
        approved_nicknames = None  # type: ignore[assignment]
        raise StyleValidationError(
            "invalid_user_identity",
            "user_id is invalid",
        ) from None

    try:
        return _build_style_reply_guidance(
            text,
            recent_context,
            idle_seconds,
            user_id=user_id,
            candidate_source=candidate_source,
            expected_count=expected_count,
            active_style_source=active_style_source,
            active_manifest_source=active_manifest_source,
            expected_active_manifest_digest=expected_active_manifest_digest,
            calibration_source=calibration_source,
            expected_calibration_fingerprint=expected_calibration_fingerprint,
            private_data_dir=private_data_dir,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )
    except Exception:
        return StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)
