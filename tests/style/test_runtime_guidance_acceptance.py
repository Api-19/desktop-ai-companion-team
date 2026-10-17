import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from companion.contracts import StyleGuidanceStatus, StyleIntent, StylePack, StyleReplyGuidance
from companion.privacy import (
    CalibrationCandidate,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
    calibration_candidate_id,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
    publish_calibration_candidate_package,
)
from companion.style import (
    activate_style_iteration,
    build_style_reply_guidance,
    prepare_style_iteration,
)
from companion.style.adjustments import candidate_package_fingerprint, propose_style_adjustments
from companion.style.iteration_models import (
    AdjustmentDecision,
    AdjustmentDecisionPackage,
    StyleAdjustmentEvidence,
    StyleIterationManifest,
)
from companion.style.private_io import read_private_model, write_private_model

NOW = datetime(2026, 7, 27, tzinfo=UTC)
TRUE_NAME_TERMS = ["synthetic-runtime-name"]
APPROVED_NICKNAMES = ["synthetic-runtime-nickname"]
SCENARIO_SOURCE = Path(__file__).parents[2] / "fixtures/synthetic/style_evaluation_scenarios.json"
RUNTIME_SCENARIO_SOURCE = (
    Path(__file__).parents[2] / "fixtures/synthetic/style_runtime_guidance_scenarios.json"
)


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
        CandidateCategory.GREETING: ("你好",),
        CandidateCategory.COURTESY: ("谢谢",),
        CandidateCategory.WEATHER: ("今天天气不错",),
        CandidateCategory.WORK_STUDY: ("我去学习",),
        CandidateCategory.SLEEP: ("我去睡觉了", "记得早点睡"),
        CandidateCategory.MEALS: ("我准备吃饭啦",),
        CandidateCategory.HOUSEHOLD: ("我去收拾房间啦",),
        CandidateCategory.SHOPPING: ("我去买日用品",),
        CandidateCategory.GENERAL_EMOTION: ("今天好开心呀",),
        CandidateCategory.GENTLE_AFFECTION: ("我很想你呀",),
    }
    candidates = [
        SafeStyleCandidate(text=text, category=category, scene_tags=[category.value])
        for category, texts in examples.items()
        for text in texts
    ]
    return SafeCandidatePackage(
        created_at=NOW,
        candidates=candidates,
        audit=CandidateSelectionAudit(
            processed_at=NOW,
            input_count=len(candidates),
            owner_input_count=len(candidates),
            dropped_other_count=0,
            allowed_category_counts={
                category.value: len(texts) for category, texts in examples.items()
            },
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=len(candidates),
            succeeded=True,
        ),
    )


def _calibration_package(
    *,
    shopping_candidate: bool = False,
) -> CalibrationCandidatePackage:
    reviewed_locks = (
        (CandidateCategory.GREETING, StyleIntent.MORNING_GREETING, "合成早安锁"),
        (CandidateCategory.GREETING, StyleIntent.CONVERSATION_REOPEN, "合成开场锁"),
        (CandidateCategory.COURTESY, StyleIntent.COURTESY_THANKS, "合成感谢锁"),
        (CandidateCategory.COURTESY, StyleIntent.COURTESY_APOLOGY, "合成道歉锁"),
        (CandidateCategory.WEATHER, StyleIntent.WEATHER_COLD_REMINDER, "合成降温锁"),
        (CandidateCategory.WORK_STUDY, StyleIntent.WORK_STUDY_COMPLETED, "合成完成锁"),
        (CandidateCategory.MEALS, StyleIntent.MISSED_MEAL_CARE, "合成吃饭锁"),
        (
            CandidateCategory.HOUSEHOLD,
            StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE,
            "合成整理锁",
        ),
        (
            CandidateCategory.SHOPPING,
            StyleIntent.SHOPPING_COMPARISON_REQUEST,
            "合成选择锁",
        ),
        (CandidateCategory.WORK_STUDY, StyleIntent.WORK_FATIGUE_CARE, "合成疲惫锁"),
        (
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.POSITIVE_EMOTION_SHARE,
            "合成开心锁",
        ),
        (
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.NEGATIVE_EMOTION_CHECK_IN,
            "合成失落锁",
        ),
        (
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.MISSED_CONTACT_AFFECTION,
            "合成想念锁",
        ),
        (CandidateCategory.MEALS, StyleIntent.NOT_EATEN_YET_CARE, "合成未吃饭锁"),
        (CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER, "合成熟睡提醒锁"),
        (CandidateCategory.WEATHER, StyleIntent.RAIN_REMINDER, "合成带伞锁"),
        (
            CandidateCategory.HOUSEHOLD,
            StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
            "合成家务完成锁",
        ),
        (CandidateCategory.GENTLE_AFFECTION, StyleIntent.GOING_OUT_CARE, "合成出门锁"),
    )
    if shopping_candidate:
        reviewed_locks += (
            (
                CandidateCategory.SHOPPING,
                StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
                "合成购物私人锁",
            ),
        )
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
        for category, intent, text in reviewed_locks
    ]
    return CalibrationCandidatePackage(
        created_at=NOW,
        content_fingerprint=calibration_review_fingerprint(candidates),
        identity_config_fingerprint=calibration_identity_config_fingerprint(
            TRUE_NAME_TERMS,
            APPROVED_NICKNAMES,
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


def _active_style_sources(
    root: Path,
    package: SafeCandidatePackage,
    *,
    shopping_preference: bool = False,
) -> tuple[Path, Path]:
    base_pack = StylePack(
        version="synthetic-runtime-base",
        statistics={"candidate_count": len(package.candidates)},
        scene_tags=[CandidateCategory.SLEEP.value],
        forbidden_traits=["do_not_mechanically_repeat"],
        examples=["synthetic base style example"],
    )
    evidence = [
        StyleAdjustmentEvidence(
            category=CandidateCategory.SLEEP,
            expression="记得早点睡",
            trigger_tags=[CandidateCategory.SLEEP.value],
            intent_tags=[StyleIntent.SLEEP_FAREWELL],
            aggregate_count=1,
            source="explicit_owner_feedback",
        )
    ]
    if shopping_preference:
        evidence.append(
            StyleAdjustmentEvidence(
                category=CandidateCategory.SHOPPING,
                expression="我去买日用品",
                trigger_tags=[CandidateCategory.SHOPPING.value],
                intent_tags=[StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT],
                aggregate_count=1,
                source="explicit_owner_feedback",
            )
        )
    proposal = propose_style_adjustments(
        evidence,
        candidate_package=package,
        proposal_id="synthetic-runtime-proposal",
        base_style_version=base_pack.version,
    )
    decisions = AdjustmentDecisionPackage(
        proposal_id=proposal.proposal_id,
        decisions=[
            AdjustmentDecision(rule_id=rule.rule_id, decision="approve") for rule in proposal.rules
        ],
    )
    base_source = root / "base.json"
    proposal_source = root / "proposal.json"
    decisions_source = root / "decisions.json"
    active_style_source = root / "active.json"
    active_manifest_source = root / "manifest.json"
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


def _formal_runtime_sources(
    root: Path,
    *,
    shopping_preference: bool = False,
    shopping_candidate: bool = False,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    package = _safe_package()
    candidate_source = root / "style_candidates/current/safe-candidates.json"
    write_private_model(package, candidate_source, root)
    active_style_source, active_manifest_source = _active_style_sources(
        root,
        package,
        shopping_preference=shopping_preference,
    )
    manifest = read_private_model(StyleIterationManifest, active_manifest_source, root)
    calibration = _calibration_package(shopping_candidate=shopping_candidate)
    calibration_source = publish_calibration_candidate_package(
        calibration,
        private_data_dir=root,
        true_name_terms=TRUE_NAME_TERMS,
        approved_nicknames=APPROVED_NICKNAMES,
    )
    return {
        "candidate_source": candidate_source,
        "expected_count": len(package.candidates),
        "active_style_source": active_style_source,
        "active_manifest_source": active_manifest_source,
        "expected_active_manifest_digest": _model_digest(manifest),
        "calibration_source": calibration_source,
        "expected_calibration_fingerprint": calibration.content_fingerprint,
        "private_data_dir": root,
        "true_name_terms": TRUE_NAME_TERMS,
        "approved_nicknames": APPROVED_NICKNAMES,
    }


def _formal_output_forbidden_values(
    sources: dict[str, object],
    text: str,
    context: list[str],
) -> tuple[str, ...]:
    private_data_dir = cast(Path, sources["private_data_dir"])
    candidate_source = cast(Path, sources["candidate_source"])
    active_style_source = cast(Path, sources["active_style_source"])
    active_manifest_source = cast(Path, sources["active_manifest_source"])
    calibration_source = cast(Path, sources["calibration_source"])
    candidate_package = read_private_model(
        SafeCandidatePackage,
        candidate_source,
        private_data_dir,
    )
    calibration_package = read_private_model(
        CalibrationCandidatePackage,
        calibration_source,
        private_data_dir,
    )
    return (
        str(private_data_dir),
        str(candidate_source),
        str(active_style_source),
        str(active_manifest_source),
        str(calibration_source),
        cast(str, sources["expected_active_manifest_digest"]),
        cast(str, sources["expected_calibration_fingerprint"]),
        calibration_package.identity_config_fingerprint,
        candidate_package_fingerprint(candidate_package),
        *(candidate.candidate_id for candidate in calibration_package.candidates),
        *TRUE_NAME_TERMS,
        *APPROVED_NICKNAMES,
        text,
        *context,
        "scenario_id",
        "evaluation_response",
        "prompt",
        "api_parameter",
    )


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for child in value for item in _string_values(child))
    if isinstance(value, dict):
        return tuple(item for child in value.values() for item in _string_values(child))
    return ()


def _assert_safe_serialized_guidance(
    guidance: StyleReplyGuidance,
    forbidden_values: tuple[str, ...],
) -> None:
    serialized = guidance.model_dump_json()
    restored = StyleReplyGuidance.model_validate_json(serialized)
    assert restored == guidance
    payload = json.loads(serialized)
    assert set(payload) == {
        "status",
        "style_version",
        "category",
        "intent",
        "examples",
        "soft_lock_expression",
    }
    string_values = _string_values(payload)
    for forbidden in forbidden_values:
        assert forbidden not in string_values
        assert not any(forbidden in value for value in string_values)


@pytest.mark.parametrize("leak_kind", ("windows_path", "manifest_digest", "candidate_id"))
def test_serialized_guidance_guard_rejects_formal_identifiers(
    tmp_path: Path,
    leak_kind: str,
) -> None:
    sources = _formal_runtime_sources(tmp_path)
    forbidden = _formal_output_forbidden_values(sources, "hello", [])
    leak_by_kind = {
        "windows_path": forbidden[0],
        "manifest_digest": cast(str, sources["expected_active_manifest_digest"]),
        "candidate_id": forbidden[9],
    }
    guidance = StyleReplyGuidance(
        status=StyleGuidanceStatus.MATCHED,
        style_version="synthetic-runtime-version",
        category=CandidateCategory.GREETING.value,
        intent=StyleIntent.CONVERSATION_REOPEN,
        examples=["safe synthetic example", leak_by_kind[leak_kind]],
    )

    with pytest.raises(AssertionError):
        _assert_safe_serialized_guidance(guidance, forbidden)


@pytest.mark.parametrize(
    ("text", "context", "idle_seconds", "status", "category", "intent", "soft_lock"),
    [
        (
            "hello",
            [],
            21_600,
            StyleGuidanceStatus.MATCHED,
            CandidateCategory.GREETING,
            StyleIntent.CONVERSATION_REOPEN,
            "合成开场锁",
        ),
        (
            "弄好了",
            ["可以帮我整理一下吗"],
            None,
            StyleGuidanceStatus.MATCHED,
            CandidateCategory.COURTESY,
            StyleIntent.COURTESY_THANKS,
            "合成感谢锁",
        ),
        (
            "你刚才打断我了",
            [],
            0,
            StyleGuidanceStatus.MATCHED,
            CandidateCategory.COURTESY,
            StyleIntent.COURTESY_APOLOGY,
            "合成道歉锁",
        ),
        (
            "今天降温了",
            [],
            0,
            StyleGuidanceStatus.MATCHED,
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
            "合成降温锁",
        ),
        (
            "我下课了",
            [],
            0,
            StyleGuidanceStatus.MATCHED,
            CandidateCategory.WORK_STUDY,
            StyleIntent.WORK_STUDY_COMPLETED,
            "合成完成锁",
        ),
        (
            "我要睡觉了",
            [],
            0,
            StyleGuidanceStatus.MATCHED,
            CandidateCategory.SLEEP,
            StyleIntent.SLEEP_FAREWELL,
            "记得早点睡",
        ),
        (
            "我还没吃饭",
            [],
            None,
            StyleGuidanceStatus.MATCHED,
            CandidateCategory.MEALS,
            StyleIntent.NOT_EATEN_YET_CARE,
            "合成未吃饭锁",
        ),
        (
            "下班了但是外面好冷",
            [],
            0,
            StyleGuidanceStatus.NO_MATCH,
            None,
            None,
            None,
        ),
    ],
    ids=(
        "reopen",
        "contextual-thanks",
        "apology",
        "cold-weather",
        "study-complete",
        "sleep-preference",
        "not-eaten-yet",
        "conflict",
    ),
)
def test_public_runtime_guidance_accepts_only_formal_synthetic_material(
    tmp_path: Path,
    text: str,
    context: list[str],
    idle_seconds: int | None,
    status: StyleGuidanceStatus,
    category: CandidateCategory | None,
    intent: StyleIntent | None,
    soft_lock: str | None,
) -> None:
    sources = _formal_runtime_sources(tmp_path)

    guidance = build_style_reply_guidance(
        text, context, idle_seconds, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is status
    assert guidance.category == (category.value if category is not None else None)
    assert guidance.intent is intent
    assert guidance.soft_lock_expression == soft_lock
    assert len(guidance.examples) <= 2
    _assert_safe_serialized_guidance(
        guidance,
        _formal_output_forbidden_values(sources, text, context),
    )


@pytest.mark.parametrize(
    ("text", "category", "intent", "expected_example"),
    [
        (
            "早上好",
            CandidateCategory.GREETING,
            StyleIntent.MORNING_GREETING,
            "你好",
        ),
        (
            "中午忙得忘记吃饭了",
            CandidateCategory.MEALS,
            StyleIntent.MISSED_MEAL_CARE,
            "我准备吃饭啦",
        ),
        (
            "我准备休息了",
            CandidateCategory.SLEEP,
            StyleIntent.SLEEP_FAREWELL,
            "我去睡觉了",
        ),
        (
            "今晚又熬夜了",
            CandidateCategory.SLEEP,
            StyleIntent.SLEEP_REMINDER,
            "我去睡觉了",
        ),
        (
            "外面好像要下雨",
            CandidateCategory.WEATHER,
            StyleIntent.RAIN_REMINDER,
            "今天天气不错",
        ),
        (
            "家务刚做完",
            CandidateCategory.HOUSEHOLD,
            StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
            "我去收拾房间啦",
        ),
        (
            "房间还没整理",
            CandidateCategory.HOUSEHOLD,
            StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE,
            "我去收拾房间啦",
        ),
        (
            "这两件我不知道选哪个",
            CandidateCategory.SHOPPING,
            StyleIntent.SHOPPING_COMPARISON_REQUEST,
            "我去买日用品",
        ),
        (
            "我又想买东西了",
            CandidateCategory.SHOPPING,
            StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
            "我去买日用品",
        ),
        (
            "今天工作有点累",
            CandidateCategory.WORK_STUDY,
            StyleIntent.WORK_FATIGUE_CARE,
            "我去学习",
        ),
        (
            "我今天心情挺好的",
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.POSITIVE_EMOTION_SHARE,
            "今天好开心呀",
        ),
        (
            "我有点失落",
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.NEGATIVE_EMOTION_CHECK_IN,
            "今天好开心呀",
        ),
        (
            "今天都没怎么见到你",
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.MISSED_CONTACT_AFFECTION,
            "我很想你呀",
        ),
        (
            "我准备出门了",
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.GOING_OUT_CARE,
            "我很想你呀",
        ),
    ],
)
def test_fixed_runtime_guidance_uses_only_same_category_formal_material(
    tmp_path: Path,
    text: str,
    category: CandidateCategory,
    intent: StyleIntent,
    expected_example: str,
) -> None:
    sources = _formal_runtime_sources(tmp_path)

    guidance = build_style_reply_guidance(text, [], 0, user_id="synthetic-test-user", **sources)

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.category == category.value
    assert guidance.intent is intent
    assert 1 <= len(guidance.examples) <= 2
    assert expected_example in guidance.examples
    if intent is StyleIntent.CATEGORY_DEFAULT:
        assert guidance.soft_lock_expression is None


def test_formal_eighteen_has_unique_keys_and_reuses_cold_weather_once() -> None:
    package = _calibration_package()
    keys = [(candidate.category, candidate.intent) for candidate in package.candidates]

    assert package.candidate_count == 18
    assert len(keys) == len(set(keys))
    assert keys.count((CandidateCategory.WEATHER, StyleIntent.WEATHER_COLD_REMINDER.value)) == 1


@pytest.mark.parametrize(
    ("text", "category", "intent", "expected_lock"),
    [
        ("早上好", CandidateCategory.GREETING, StyleIntent.MORNING_GREETING, "合成早安锁"),
        (
            "中午忙得忘记吃饭了",
            CandidateCategory.MEALS,
            StyleIntent.MISSED_MEAL_CARE,
            "合成吃饭锁",
        ),
        (
            "房间还没整理",
            CandidateCategory.HOUSEHOLD,
            StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE,
            "合成整理锁",
        ),
        (
            "这两件我不知道选哪个",
            CandidateCategory.SHOPPING,
            StyleIntent.SHOPPING_COMPARISON_REQUEST,
            "合成选择锁",
        ),
        (
            "今天工作有点累",
            CandidateCategory.WORK_STUDY,
            StyleIntent.WORK_FATIGUE_CARE,
            "合成疲惫锁",
        ),
        (
            "我今天心情挺好的",
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.POSITIVE_EMOTION_SHARE,
            "合成开心锁",
        ),
        (
            "我有点失落",
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.NEGATIVE_EMOTION_CHECK_IN,
            "合成失落锁",
        ),
        (
            "今天都没怎么见到你",
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.MISSED_CONTACT_AFFECTION,
            "合成想念锁",
        ),
        (
            "我还没吃饭",
            CandidateCategory.MEALS,
            StyleIntent.NOT_EATEN_YET_CARE,
            "合成未吃饭锁",
        ),
        (
            "今晚又熬夜了",
            CandidateCategory.SLEEP,
            StyleIntent.SLEEP_REMINDER,
            "合成熟睡提醒锁",
        ),
        (
            "外面好像要下雨",
            CandidateCategory.WEATHER,
            StyleIntent.RAIN_REMINDER,
            "合成带伞锁",
        ),
        (
            "家务刚做完",
            CandidateCategory.HOUSEHOLD,
            StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
            "合成家务完成锁",
        ),
        (
            "我准备出门了",
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.GOING_OUT_CARE,
            "合成出门锁",
        ),
    ],
)
def test_new_precise_runtime_guidance_uses_only_exact_formal_lock(
    tmp_path: Path,
    text: str,
    category: CandidateCategory,
    intent: StyleIntent,
    expected_lock: str,
) -> None:
    sources = _formal_runtime_sources(tmp_path)

    guidance = build_style_reply_guidance(text, [], 0, user_id="synthetic-test-user", **sources)

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.category == category.value
    assert guidance.intent is intent
    assert guidance.soft_lock_expression == expected_lock
    assert expected_lock not in guidance.examples
    assert len(guidance.examples) <= 2


def test_shopping_desire_uses_safe_examples_without_private_soft_lock(
    tmp_path: Path,
) -> None:
    sources = _formal_runtime_sources(
        tmp_path,
        shopping_preference=True,
        shopping_candidate=True,
    )

    guidance = build_style_reply_guidance(
        "我又想买东西了", [], 0, user_id="synthetic-test-user", **sources
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.intent is StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT
    assert guidance.examples == ["我去买日用品"]
    assert guidance.soft_lock_expression is None


def test_reopen_threshold_selects_reopen_material_before_morning_material(
    tmp_path: Path,
) -> None:
    sources = _formal_runtime_sources(tmp_path)

    reopened = build_style_reply_guidance(
        "早上好", [], 21_600, user_id="synthetic-test-user", **sources
    )
    morning = build_style_reply_guidance(
        "早上好", [], 21_599, user_id="synthetic-test-user", **sources
    )

    assert reopened.status is StyleGuidanceStatus.MATCHED
    assert reopened.intent is StyleIntent.CONVERSATION_REOPEN
    assert reopened.soft_lock_expression == "合成开场锁"
    assert morning.status is StyleGuidanceStatus.MATCHED
    assert morning.intent is StyleIntent.MORNING_GREETING
    assert morning.soft_lock_expression == "合成早安锁"


def test_public_fixed_twenty_runtime_inputs_are_structurally_matched(
    tmp_path: Path,
) -> None:
    scenario_ids = {
        item["scenario_id"] for item in json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
    }
    runtime_scenarios = json.loads(RUNTIME_SCENARIO_SOURCE.read_text(encoding="utf-8"))
    assert len(runtime_scenarios) == 20
    assert all(
        set(item) == {"scenario_id", "text", "recent_context", "idle_seconds"}
        for item in runtime_scenarios
    )
    runtime_scenario_ids = [item["scenario_id"] for item in runtime_scenarios]
    assert len(runtime_scenario_ids) == len(set(runtime_scenario_ids))
    assert set(runtime_scenario_ids) == scenario_ids
    sources = _formal_runtime_sources(tmp_path)

    expected_intents = {
        item["scenario_id"]: StyleIntent(item["intent"])
        for item in json.loads(SCENARIO_SOURCE.read_text(encoding="utf-8"))
    }
    guidance_items = [
        build_style_reply_guidance(
            item["text"],
            item["recent_context"],
            item["idle_seconds"],
            user_id="synthetic-test-user",
            **sources,
        )
        for item in runtime_scenarios
    ]
    repeated_guidance_items = [
        build_style_reply_guidance(
            item["text"],
            item["recent_context"],
            item["idle_seconds"],
            user_id="synthetic-test-user",
            **sources,
        )
        for item in runtime_scenarios
    ]
    statuses = [guidance.status for guidance in guidance_items]

    assert statuses.count(StyleGuidanceStatus.MATCHED) == 20
    assert statuses.count(StyleGuidanceStatus.NO_MATCH) == 0
    assert statuses.count(StyleGuidanceStatus.UNAVAILABLE) == 0
    assert repeated_guidance_items == guidance_items
    assert {
        item["scenario_id"]: guidance.intent
        for item, guidance in zip(runtime_scenarios, guidance_items, strict=True)
    } == expected_intents


@pytest.mark.parametrize("broken_source", ("count", "manifest"))
def test_fixed_runtime_resource_failure_is_atomic_unavailable(
    tmp_path: Path,
    broken_source: str,
) -> None:
    sources = _formal_runtime_sources(tmp_path)
    if broken_source == "count":
        sources["expected_count"] = cast(int, sources["expected_count"]) + 1
    else:
        sources["expected_active_manifest_digest"] = "f" * 64

    guidance = build_style_reply_guidance(
        "房间还没整理",
        [],
        0,
        user_id="synthetic-test-user",
        **sources,
    )

    assert guidance == StyleReplyGuidance(
        status=StyleGuidanceStatus.UNAVAILABLE,
    )
