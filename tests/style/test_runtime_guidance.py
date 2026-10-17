import hashlib
import json
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import companion.style.iteration_pipeline as iteration_pipeline
import companion.style.runtime_guidance as runtime_guidance
from companion.contracts import (
    StyleGuidanceStatus,
    StyleIntent,
    StylePack,
    StyleReplyGuidance,
)
from companion.privacy import (
    CalibrationCandidate,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CandidateCategory,
    CandidateSelectionAudit,
    PrivacyValidationError,
    SafeCandidatePackage,
    SafeStyleCandidate,
    calibration_candidate_id,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
    publish_calibration_candidate_package,
)
from companion.style import (
    StyleValidationError,
    activate_style_iteration,
    build_style_reply_guidance,
    invalidate_style_iteration,
    prepare_style_iteration,
)
from companion.style.adjustments import (
    candidate_package_fingerprint,
    propose_style_adjustments,
)
from companion.style.iteration_models import (
    AdjustmentDecision,
    AdjustmentDecisionPackage,
    StyleAdjustmentEvidence,
    StyleIterationManifest,
)
from companion.style.private_io import read_private_model, write_private_model

NOW = datetime(2026, 7, 27, tzinfo=UTC)
TRUE_NAMES = ["合成真实姓名"]
APPROVED_NICKNAMES = ["合成爱称"]


def _model_digest(model: object) -> str:
    payload = json.dumps(
        model.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _safe_package() -> SafeCandidatePackage:
    examples = {
        CandidateCategory.WEATHER: ("今天天气真好", "今天天气不错", "今天天气晴朗"),
        CandidateCategory.SLEEP: ("我要睡觉啦", "我去睡觉了", "记得早点睡"),
        CandidateCategory.MEALS: ("我准备吃饭啦", "我要吃晚饭了", "我想吃面"),
    }
    candidates = [
        SafeStyleCandidate(
            text=text,
            category=category,
            scene_tags=[category.value],
        )
        for category, texts in examples.items()
        for text in texts
    ]
    counts = {category.value: len(texts) for category, texts in examples.items()}
    return SafeCandidatePackage(
        created_at=NOW,
        candidates=candidates,
        audit=CandidateSelectionAudit(
            processed_at=NOW,
            input_count=len(candidates),
            owner_input_count=len(candidates),
            dropped_other_count=0,
            allowed_category_counts=counts,
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=len(candidates),
            succeeded=True,
        ),
    )


def _calibration_package(
    *items: tuple[CandidateCategory, StyleIntent, str],
    true_name_terms: list[str] = TRUE_NAMES,
    approved_nicknames: list[str] = APPROVED_NICKNAMES,
) -> CalibrationCandidatePackage:
    candidates = [
        CalibrationCandidate(
            candidate_id=calibration_candidate_id(
                category=category,
                intent=intent.value,
                text=text,
            ),
            category=category,
            intent=intent.value,
            text=text,
        )
        for category, intent, text in items
    ]
    return CalibrationCandidatePackage(
        created_at=NOW,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=calibration_identity_config_fingerprint(
            true_name_terms,
            approved_nicknames,
        ),
        candidate_count=len(candidates),
        candidates=candidates,
        audit=CalibrationCandidateAudit(
            reviewed_at=NOW,
            submission_count=len(candidates),
            approved_count=len(candidates),
            rejected_count=0,
            privacy_rejected_count=0,
            revoked_count=0,
            output_count=len(candidates),
        ),
    )


def _activate_style_pack(
    root: Path,
    package: SafeCandidatePackage,
    *,
    suffix: str,
    base_version: str,
    expression: str = "记得早点睡",
) -> tuple[Path, Path]:
    base_pack = StylePack(
        version=base_version,
        statistics={"candidate_count": len(package.candidates)},
        scene_tags=[CandidateCategory.SLEEP.value],
        forbidden_traits=["do_not_mechanically_repeat"],
        examples=["合成基础示例"],
    )
    proposal = propose_style_adjustments(
        [
            StyleAdjustmentEvidence(
                category=CandidateCategory.SLEEP,
                expression=expression,
                trigger_tags=[CandidateCategory.SLEEP.value],
                intent_tags=[StyleIntent.SLEEP_FAREWELL],
                aggregate_count=1,
                source="explicit_owner_feedback",
            )
        ],
        candidate_package=package,
        proposal_id=f"synthetic-proposal-{suffix}",
        base_style_version=base_pack.version,
    )
    decisions = AdjustmentDecisionPackage(
        proposal_id=proposal.proposal_id,
        decisions=[
            AdjustmentDecision(
                rule_id=proposal.rules[0].rule_id,
                decision="approve",
            )
        ],
    )
    base_source = root / f"base-{suffix}.json"
    proposal_source = root / f"proposal-{suffix}.json"
    decisions_source = root / f"decisions-{suffix}.json"
    active_style_source = root / f"active-{suffix}.json"
    active_manifest_source = root / f"manifest-{suffix}.json"
    for model, destination in (
        (base_pack, base_source),
        (proposal, proposal_source),
        (decisions, decisions_source),
    ):
        write_private_model(model, destination, root)
    prepare_style_iteration(
        base_pack_source=base_source,
        proposal_source=proposal_source,
        decisions_source=decisions_source,
        private_data_dir=root,
        iteration_round=1,
        created_at=NOW,
        candidate_destination=active_style_source,
        manifest_destination=active_manifest_source,
    )
    activate_style_iteration(
        manifest_source=active_manifest_source,
        private_data_dir=root,
        user_confirmed=True,
        activated_at=NOW,
    )
    return active_style_source, active_manifest_source


def _resources(
    root: Path,
    *,
    package: SafeCandidatePackage | None = None,
    active_preference_expression: str = "记得早点睡",
    true_name_terms: list[str] = TRUE_NAMES,
    approved_nicknames: list[str] = APPROVED_NICKNAMES,
) -> tuple[dict[str, object], CalibrationCandidatePackage]:
    root.mkdir(parents=True, exist_ok=True)
    package = package or _safe_package()
    candidate_source = root / "style_candidates/current/safe-candidates.json"
    write_private_model(package, candidate_source, root)
    active_style_source, active_manifest_source = _activate_style_pack(
        root,
        package,
        suffix="one",
        base_version="synthetic-base-one",
        expression=active_preference_expression,
    )
    active_manifest = read_private_model(
        StyleIterationManifest,
        active_manifest_source,
        root,
    )
    long_example = "合成长示例" + "呀" * 28
    assert len(long_example) == 33
    calibration = _calibration_package(
        (
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
            "合成记得添衣",
        ),
        (
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
            long_example,
        ),
        (
            CandidateCategory.SLEEP,
            StyleIntent.SLEEP_REMINDER,
            "合成别熬夜",
        ),
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    calibration_source = publish_calibration_candidate_package(
        calibration,
        private_data_dir=root,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )
    sources: dict[str, object] = {
        "candidate_source": candidate_source,
        "expected_count": len(package.candidates),
        "active_style_source": active_style_source,
        "active_manifest_source": active_manifest_source,
        "expected_active_manifest_digest": _model_digest(active_manifest),
        "calibration_source": calibration_source,
        "expected_calibration_fingerprint": calibration.content_fingerprint,
        "private_data_dir": root,
        "true_name_terms": true_name_terms,
        "approved_nicknames": approved_nicknames,
    }
    return sources, calibration


def test_matched_guidance_prefers_formal_long_example_and_short_soft_lock(
    tmp_path: Path,
) -> None:
    sources, _ = _resources(tmp_path)

    first = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )
    second = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert first == second
    assert first.status is StyleGuidanceStatus.MATCHED
    assert first.category == CandidateCategory.WEATHER.value
    assert first.intent is StyleIntent.WEATHER_COLD_REMINDER
    assert first.soft_lock_expression == "合成记得添衣"
    assert first.examples[0] == "合成长示例" + "呀" * 28
    assert first.soft_lock_expression not in first.examples
    assert len(first.examples) <= 2


def test_preference_is_used_when_formal_short_expression_is_absent(
    tmp_path: Path,
) -> None:
    sources, _ = _resources(tmp_path)

    guidance = build_style_reply_guidance(
        "我要睡觉了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.intent is StyleIntent.SLEEP_FAREWELL
    assert guidance.soft_lock_expression == "记得早点睡"
    assert guidance.soft_lock_expression not in guidance.examples


def test_active_preference_is_filtered_and_base_examples_are_backfilled(
    tmp_path: Path,
) -> None:
    package = _safe_package()
    candidates = [
        candidate.model_copy(update={"text": "我要睡觉了"})
        if candidate.text == "记得早点睡"
        else candidate
        for candidate in package.candidates
    ]
    package = SafeCandidatePackage.model_validate(
        package.model_copy(update={"candidates": candidates}).model_dump()
    )
    sources, _ = _resources(
        tmp_path,
        package=package,
        active_preference_expression="我要睡觉了",
    )

    guidance = build_style_reply_guidance(
        "我要睡觉了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.intent is StyleIntent.SLEEP_FAREWELL
    assert guidance.soft_lock_expression == "我要睡觉了"
    assert guidance.examples == ["我要睡觉啦", "我去睡觉了"]
    assert guidance.soft_lock_expression not in guidance.examples
    assert len(guidance.examples) == len(set(guidance.examples)) == 2


def test_active_preference_whitespace_variant_is_filtered_and_backfilled(
    tmp_path: Path,
) -> None:
    package = _replace_category_candidates(
        _safe_package(),
        CandidateCategory.SLEEP,
        "我要睡觉了",
        " 我要睡觉了 ",
        "我要睡觉啦",
        "我去睡觉了",
    )
    sources, _ = _resources(
        tmp_path,
        package=package,
        active_preference_expression="我要睡觉了",
    )

    guidance = build_style_reply_guidance(
        "我要睡觉了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.soft_lock_expression == "我要睡觉了"
    assert guidance.examples == ["我要睡觉啦", "我去睡觉了"]


def test_formal_and_base_whitespace_variants_are_normalized_and_deduplicated(
    tmp_path: Path,
) -> None:
    long_example = "合成长示例" + "呀" * 28
    package = _replace_category_candidates(
        _safe_package(),
        CandidateCategory.WEATHER,
        " 今天天气不错 ",
        "今天天气不错",
        " 今天天气真好 ",
        " 今天天气晴朗 ",
    )
    sources, _ = _resources(tmp_path, package=package)

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.soft_lock_expression == "合成记得添衣"
    assert guidance.examples == [long_example, "今天天气不错"]


def test_category_default_normalizes_deduplicates_and_backfills_base_examples(
    tmp_path: Path,
) -> None:
    package = _replace_category_candidates(
        _safe_package(),
        CandidateCategory.MEALS,
        " 我准备吃饭啦 ",
        "我准备吃饭啦",
        " 我要吃晚饭了 ",
    )
    sources, _ = _resources(tmp_path, package=package)

    guidance = build_style_reply_guidance(
        "准备吃饭", [], None, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.intent is StyleIntent.CATEGORY_DEFAULT
    assert guidance.soft_lock_expression is None
    assert guidance.examples == ["我准备吃饭啦", "我要吃晚饭了"]


def test_category_default_returns_only_deduplicated_examples(tmp_path: Path) -> None:
    sources, _ = _resources(tmp_path)

    first = build_style_reply_guidance(
        "准备吃饭", [], None, user_id="synthetic-test-user", **sources
    )
    second = build_style_reply_guidance(
        "准备吃饭", [], None, user_id="synthetic-test-user", **sources
    )

    assert first == second
    assert first.status is StyleGuidanceStatus.MATCHED
    assert first.category == CandidateCategory.MEALS.value
    assert first.intent is StyleIntent.CATEGORY_DEFAULT
    assert len(first.examples) == 2
    assert len(first.examples) == len(set(first.examples))
    assert first.soft_lock_expression is None


def test_identity_config_drift_returns_exact_empty_unavailable(tmp_path: Path) -> None:
    stale_sources, stale_calibration = _resources(
        tmp_path / "stale",
        approved_nicknames=["合成旧批准爱称"],
    )
    migrated_sources, migrated_calibration = _resources(
        tmp_path / "migrated",
        approved_nicknames=["合成当前批准爱称"],
    )
    stale_sources["approved_nicknames"] = ["合成当前批准爱称"]

    stale = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **stale_sources
    )
    migrated = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **migrated_sources
    )

    assert stale_calibration.content_fingerprint == migrated_calibration.content_fingerprint
    assert (
        stale_sources["expected_calibration_fingerprint"]
        == migrated_sources["expected_calibration_fingerprint"]
    )
    assert stale == StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)
    assert migrated.status is StyleGuidanceStatus.MATCHED


def test_identity_config_normalized_equivalence_remains_matched(tmp_path: Path) -> None:
    sources, _ = _resources(
        tmp_path,
        true_name_terms=["合成身份甲", "合成身份乙"],
        approved_nicknames=["合成爱称甲", "合成爱称乙"],
    )
    sources["true_name_terms"] = [
        " 合成身份乙 ",
        "合成身份甲",
        "合成身份甲",
    ]
    sources["approved_nicknames"] = [
        " 合成爱称乙 ",
        "合成爱称甲",
        "合成爱称甲",
    ]

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED


def test_runtime_version_is_deterministic_and_hides_source_fingerprints(
    tmp_path: Path,
) -> None:
    sources, first_calibration = _resources(tmp_path)
    first = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    second_calibration = _calibration_package(
        (
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
            "合成换件厚衣服",
        ),
    )
    publish_calibration_candidate_package(
        second_calibration,
        private_data_dir=tmp_path,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=APPROVED_NICKNAMES,
    )
    sources["expected_calibration_fingerprint"] = second_calibration.content_fingerprint
    second = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    package = _safe_package()
    active_style_source, active_manifest_source = _activate_style_pack(
        tmp_path,
        package,
        suffix="two",
        base_version="synthetic-base-two",
    )
    sources["active_style_source"] = active_style_source
    sources["active_manifest_source"] = active_manifest_source
    sources["expected_active_manifest_digest"] = _model_digest(
        read_private_model(
            StyleIterationManifest,
            active_manifest_source,
            tmp_path,
        )
    )
    third = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert first.style_version != second.style_version
    assert second.style_version != third.style_version
    for fingerprint in (
        first_calibration.content_fingerprint,
        second_calibration.content_fingerprint,
    ):
        assert first.style_version is not None
        assert second.style_version is not None
        assert third.style_version is not None
        assert fingerprint not in first.style_version
        assert fingerprint not in second.style_version
        assert fingerprint not in third.style_version


def test_runtime_version_changes_with_self_consistent_candidate_dataset(
    tmp_path: Path,
) -> None:
    sources, _ = _resources(tmp_path)
    original_package = _safe_package()
    original_fingerprint = candidate_package_fingerprint(original_package)
    first = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    changed_package = _replace_first_candidate(
        original_package,
        text="今天天气暖和",
    )
    changed_fingerprint = candidate_package_fingerprint(changed_package)
    write_private_model(changed_package, sources["candidate_source"], tmp_path)
    active_style_source, active_manifest_source = _activate_style_pack(
        tmp_path,
        changed_package,
        suffix="dataset-two",
        base_version="synthetic-base-one",
    )
    active_manifest = read_private_model(
        StyleIterationManifest,
        active_manifest_source,
        tmp_path,
    )
    sources["active_style_source"] = active_style_source
    sources["active_manifest_source"] = active_manifest_source
    sources["expected_active_manifest_digest"] = _model_digest(active_manifest)
    second = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert first.status is StyleGuidanceStatus.MATCHED
    assert second.status is StyleGuidanceStatus.MATCHED
    assert first.examples != second.examples
    assert first.style_version != second.style_version
    assert first.style_version is not None
    assert second.style_version is not None
    assert first.style_version.startswith("dev-style-runtime-")
    assert len(first.style_version.removeprefix("dev-style-runtime-")) == 12
    assert original_fingerprint not in first.style_version
    assert changed_fingerprint not in second.style_version


@pytest.mark.parametrize(
    "invalid_digest",
    ("", "a" * 63, "A" * 64, "g" * 64),
)
def test_invalid_expected_manifest_digest_returns_unavailable(
    tmp_path: Path,
    invalid_digest: str,
) -> None:
    sources, _ = _resources(tmp_path)
    sources["expected_active_manifest_digest"] = invalid_digest

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance == StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)


def test_trusted_manifest_digest_binds_self_consistent_manifest_changes(
    tmp_path: Path,
) -> None:
    sources, _ = _resources(tmp_path)
    first = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )
    manifest = read_private_model(
        StyleIterationManifest,
        sources["active_manifest_source"],
        tmp_path,
    )
    changed = StyleIterationManifest.model_validate(
        manifest.model_copy(update={"lineage_id": "d" * 64}).model_dump()
    )
    write_private_model(changed, sources["active_manifest_source"], tmp_path)

    stale = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )
    sources["expected_active_manifest_digest"] = _model_digest(changed)
    refreshed = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert stale == StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)
    assert refreshed.status is StyleGuidanceStatus.MATCHED
    assert first.style_version != refreshed.style_version
    assert refreshed.style_version is not None
    assert _model_digest(changed) not in refreshed.style_version


def test_runtime_validates_and_versions_one_manifest_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, _ = _resources(tmp_path)
    real_read = iteration_pipeline.read_private_model
    manifest_reads = 0

    def count_manifest_reads(model_type, source: Path, private_data_dir: Path):
        nonlocal manifest_reads
        if model_type is StyleIterationManifest:
            manifest_reads += 1
        return real_read(model_type, source, private_data_dir)

    monkeypatch.setattr(
        iteration_pipeline,
        "read_private_model",
        count_manifest_reads,
    )

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert manifest_reads == 1


def test_formal_short_expression_takes_priority_over_active_preference(
    tmp_path: Path,
) -> None:
    sources, _ = _resources(tmp_path)
    calibration = _calibration_package(
        (
            CandidateCategory.SLEEP,
            StyleIntent.SLEEP_FAREWELL,
            "合成正式晚安",
        ),
    )
    publish_calibration_candidate_package(
        calibration,
        private_data_dir=tmp_path,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=APPROVED_NICKNAMES,
    )
    sources["expected_calibration_fingerprint"] = calibration.content_fingerprint

    guidance = build_style_reply_guidance(
        "我要睡觉了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.soft_lock_expression == "合成正式晚安"


def test_all_formal_short_candidates_are_excluded_from_examples(
    tmp_path: Path,
) -> None:
    package = _safe_package()
    sources, _ = _resources(tmp_path, package=package)
    long_example = "合成长示例" + "呀" * 28
    calibration = _calibration_package(
        (
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
            "今天天气真好",
        ),
        (
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
            "今天天气不错",
        ),
        (
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
            long_example,
        ),
    )
    publish_calibration_candidate_package(
        calibration,
        private_data_dir=tmp_path,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=APPROVED_NICKNAMES,
    )
    sources["expected_calibration_fingerprint"] = calibration.content_fingerprint

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.soft_lock_expression == "今天天气不错"
    assert guidance.examples[0] == long_example
    assert "今天天气真好" not in guidance.examples
    assert "今天天气不错" not in guidance.examples


def _replace_first_candidate(
    package: SafeCandidatePackage,
    *,
    text: str,
) -> SafeCandidatePackage:
    candidates = [
        package.candidates[0].model_copy(update={"text": text}),
        *package.candidates[1:],
    ]
    return SafeCandidatePackage.model_validate(
        package.model_copy(update={"candidates": candidates}).model_dump()
    )


def _replace_category_candidates(
    package: SafeCandidatePackage,
    category: CandidateCategory,
    *texts: str,
) -> SafeCandidatePackage:
    candidates = [
        candidate for candidate in package.candidates if candidate.category is not category
    ]
    candidates.extend(
        SafeStyleCandidate(
            text=text,
            category=category,
            scene_tags=[category.value],
        )
        for text in texts
    )
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.category.value] = counts.get(candidate.category.value, 0) + 1
    audit = package.audit.model_copy(
        update={
            "input_count": len(candidates),
            "owner_input_count": len(candidates),
            "allowed_category_counts": counts,
            "output_count": len(candidates),
        }
    )
    return SafeCandidatePackage.model_validate(
        package.model_copy(
            update={
                "candidates": candidates,
                "audit": audit,
            }
        ).model_dump()
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_expected_count",
        "non_formal_candidate_path",
        "inactive_manifest",
        "active_candidate_digest_mismatch",
        "dataset_fingerprint_mismatch",
        "missing_calibration",
        "wrong_calibration_fingerprint",
        "changed_true_name_terms",
        "changed_approved_nicknames",
        "unsupported_calibration_intent",
        "malformed_candidate",
        "malformed_active_pack",
        "malformed_manifest",
        "malformed_calibration",
    ),
)
def test_resource_failure_returns_exact_empty_unavailable(
    tmp_path: Path,
    mutation: str,
) -> None:
    sources, _ = _resources(tmp_path)
    if mutation == "wrong_expected_count":
        sources["expected_count"] = int(sources["expected_count"]) + 1
    elif mutation == "non_formal_candidate_path":
        sources["candidate_source"] = tmp_path / "other/safe-candidates.json"
    elif mutation == "inactive_manifest":
        invalidate_style_iteration(
            manifest_source=sources["active_manifest_source"],
            private_data_dir=tmp_path,
            invalidated_at=NOW + timedelta(seconds=1),
        )
    elif mutation == "active_candidate_digest_mismatch":
        source = sources["active_style_source"]
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["examples"].append("合成篡改示例")
        source.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "dataset_fingerprint_mismatch":
        changed = _replace_first_candidate(_safe_package(), text="合成天气丁")
        write_private_model(changed, sources["candidate_source"], tmp_path)
    elif mutation == "missing_calibration":
        sources["calibration_source"].unlink()
    elif mutation == "wrong_calibration_fingerprint":
        sources["expected_calibration_fingerprint"] = "0" * 64
    elif mutation == "changed_true_name_terms":
        sources["true_name_terms"] = ["合成记得"]
    elif mutation == "changed_approved_nicknames":
        sources["approved_nicknames"] = TRUE_NAMES
    elif mutation == "unsupported_calibration_intent":
        unsupported = CalibrationCandidate(
            candidate_id=calibration_candidate_id(
                category=CandidateCategory.WEATHER,
                intent="synthetic_unknown_intent",
                text="合成正常文本",
            ),
            category=CandidateCategory.WEATHER,
            intent="synthetic_unknown_intent",
            text="合成正常文本",
        )
        calibration = CalibrationCandidatePackage(
            created_at=NOW,
            content_fingerprint=calibration_review_fingerprint([unsupported]),
            identity_config_fingerprint=calibration_identity_config_fingerprint(
                TRUE_NAMES,
                APPROVED_NICKNAMES,
            ),
            candidate_count=1,
            candidates=[unsupported],
            audit=CalibrationCandidateAudit(
                reviewed_at=NOW,
                submission_count=1,
                approved_count=1,
                rejected_count=0,
                privacy_rejected_count=0,
                revoked_count=0,
                output_count=1,
            ),
        )
        publish_calibration_candidate_package(
            calibration,
            private_data_dir=tmp_path,
            true_name_terms=TRUE_NAMES,
            approved_nicknames=APPROVED_NICKNAMES,
        )
        sources["expected_calibration_fingerprint"] = calibration.content_fingerprint
    else:
        malformed_sources = {
            "malformed_candidate": "candidate_source",
            "malformed_active_pack": "active_style_source",
            "malformed_manifest": "active_manifest_source",
            "malformed_calibration": "calibration_source",
        }
        sources[malformed_sources[mutation]].write_text("{", encoding="utf-8")

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance == StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)


def test_valid_resources_without_reliable_match_return_no_match(tmp_path: Path) -> None:
    sources, _ = _resources(tmp_path)

    guidance = build_style_reply_guidance(
        "合成无关输入", [], None, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.NO_MATCH
    assert guidance.style_version is not None


def test_invalid_input_traceback_does_not_retain_runtime_secrets(tmp_path: Path) -> None:
    sentinel = "SYNTHETIC_RUNTIME_SECRET_SENTINEL"
    sources, _ = _resources(tmp_path / sentinel)
    sources["expected_calibration_fingerprint"] = sentinel
    sources["true_name_terms"] = [sentinel]
    sources["approved_nicknames"] = [sentinel]

    with pytest.raises(StyleValidationError) as caught:
        build_style_reply_guidance(
            f" {sentinel}",
            [sentinel],
            True,
            user_id="synthetic-test-user",
            **sources,
        )

    assert sentinel not in str(caught.value)
    for frame, _line_number in traceback.walk_tb(caught.value.__traceback__):
        if "/companion/style/" in frame.f_code.co_filename.replace("\\", "/"):
            assert sentinel not in repr(frame.f_locals)


@pytest.mark.parametrize("user_id", [None, "", "   ", "x" * 121])
def test_invalid_user_identity_fails_before_reading_runtime_sources(
    tmp_path: Path,
    user_id: object,
) -> None:
    sources, _ = _resources(tmp_path)
    sources["candidate_source"].unlink()

    with pytest.raises(StyleValidationError) as caught:
        build_style_reply_guidance(
            "今天降温了",
            [],
            0,
            user_id=user_id,  # type: ignore[arg-type]
            **sources,
        )

    assert caught.value.code == "invalid_user_identity"
    assert caught.value.detail == "user_id is invalid"


@pytest.mark.parametrize("failure_point", ("strip", "length", "iteration"))
def test_input_validation_exceptions_are_sanitized_without_side_channels(
    tmp_path: Path,
    failure_point: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_sentinel = "SYNTHETIC_VALIDATION_EXCEPTION_SENTINEL"
    path_sentinel = "SYNTHETIC_VALIDATION_PATH_SENTINEL"
    identity_sentinel = "SYNTHETIC_VALIDATION_IDENTITY_SENTINEL"
    nickname_sentinel = "SYNTHETIC_VALIDATION_NICKNAME_SENTINEL"
    calibration_fingerprint = "1" * 64
    manifest_digest = "2" * 64
    sources, _ = _resources(tmp_path / path_sentinel)
    sources["expected_calibration_fingerprint"] = calibration_fingerprint
    sources["expected_active_manifest_digest"] = manifest_digest
    sources["true_name_terms"] = [identity_sentinel]
    sources["approved_nicknames"] = [nickname_sentinel]

    class ExplodingText(str):
        def strip(self, *args: object, **kwargs: object) -> str:
            del args, kwargs
            raise RuntimeError(payload_sentinel)

    class ExplodingLength(list[str]):
        def __len__(self) -> int:
            raise RuntimeError(payload_sentinel)

    class ExplodingIteration(list[str]):
        def __iter__(self):
            raise RuntimeError(payload_sentinel)

    text: str = "今天降温了"
    context: list[str] = []
    if failure_point == "strip":
        text = ExplodingText(payload_sentinel)
    elif failure_point == "length":
        context = ExplodingLength([payload_sentinel])
    else:
        context = ExplodingIteration([payload_sentinel])

    with pytest.raises(StyleValidationError) as caught:
        build_style_reply_guidance(text, context, 0, user_id="synthetic-test-user", **sources)

    assert caught.value.code == "invalid_style_guidance_input"
    assert caught.value.detail == "style guidance input is invalid"
    secrets = (
        payload_sentinel,
        path_sentinel,
        identity_sentinel,
        nickname_sentinel,
        calibration_fingerprint,
        manifest_digest,
    )
    retained = any(
        "/companion/style/" in frame.f_code.co_filename.replace("\\", "/")
        and any(secret in repr(frame.f_locals) for secret in secrets)
        for frame, _line_number in traceback.walk_tb(caught.value.__traceback__)
    )
    assert not retained
    assert not any(secret in caplog.text for secret in secrets)
    output = capsys.readouterr()
    assert not any(secret in output.out for secret in secrets)
    assert not any(secret in output.err for secret in secrets)


def test_unexpected_resource_failure_is_swallowed_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "SYNTHETIC_RESOURCE_EXCEPTION_SENTINEL"
    sources, _ = _resources(tmp_path)

    def fail_loader(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError(sentinel)

    monkeypatch.setattr(runtime_guidance, "load_safe_candidate_package", fail_loader)

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance == StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)
    assert sentinel not in caplog.text
    output = capsys.readouterr()
    assert sentinel not in output.out
    assert sentinel not in output.err


@pytest.mark.parametrize(
    "bait",
    [
        *[f"合成真实姓名诱饵{index}" for index in range(25)],
        *[f"1380000{index:04d}" for index in range(25)],
    ],
    ids=[f"bait-{index}" for index in range(50)],
)
def test_candidate_privacy_bait_cannot_enter_guidance(
    tmp_path: Path,
    bait: str,
) -> None:
    package = _replace_first_candidate(_safe_package(), text=bait)
    sources, _ = _resources(tmp_path, package=package)

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance == StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)


def test_normalized_true_name_in_safe_candidate_fails_closed_without_side_channels(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _replace_first_candidate(_safe_package(), text="今天天气不错")
    sources, _ = _resources(
        tmp_path,
        package=package,
        true_name_terms=[" 天气 "],
    )
    captured_errors: list[PrivacyValidationError] = []
    validate_runtime_material = runtime_guidance.validate_safe_candidate_runtime_material

    def capture_validation_error(
        package: SafeCandidatePackage,
        *,
        true_name_terms: list[str],
        approved_nicknames: list[str],
    ) -> None:
        try:
            validate_runtime_material(
                package,
                true_name_terms=true_name_terms,
                approved_nicknames=approved_nicknames,
            )
        except PrivacyValidationError as error:
            captured_errors.append(error)
            raise

    monkeypatch.setattr(
        runtime_guidance,
        "validate_safe_candidate_runtime_material",
        capture_validation_error,
    )

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance == StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)
    sentinels = (package.candidates[0].text, sources["true_name_terms"][0].strip())
    assert len(captured_errors) == 1
    error = captured_errors[0]
    assert not any(sentinel in str(error) for sentinel in sentinels)
    error_traceback = error.__traceback__
    while error_traceback is not None:
        filename = error_traceback.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename:
            frame_values = repr(error_traceback.tb_frame.f_locals)
            assert not any(sentinel in frame_values for sentinel in sentinels)
        error_traceback = error_traceback.tb_next
    assert not any(sentinel in repr(guidance) for sentinel in sentinels)
    assert not any(sentinel in caplog.text for sentinel in sentinels)
    output = capsys.readouterr()
    assert not any(sentinel in output.out for sentinel in sentinels)
    assert not any(sentinel in output.err for sentinel in sentinels)


def test_runtime_accepts_valid_calibrated_daily_candidate(tmp_path: Path) -> None:
    package = _replace_category_candidates(
        _safe_package(),
        CandidateCategory.CALIBRATED_DAILY,
        "今天整理了一下桌面",
    )
    sources, _ = _resources(tmp_path, package=package)

    guidance = build_style_reply_guidance(
        "今天降温了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
