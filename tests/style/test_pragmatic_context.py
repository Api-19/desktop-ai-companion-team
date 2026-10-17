from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.style import (
    PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION,
    PragmaticCalibrationBasis,
    PragmaticConfidenceMethod,
    PragmaticContextDelta,
    PragmaticDimension,
    PragmaticLabel,
    PragmaticObservedFrom,
    PragmaticSignal,
    PragmaticSignalProvenance,
    SemanticContextResolution,
    SemanticDiscourseMode,
    SemanticFrame,
    SemanticParticipant,
    SemanticPolarity,
    SemanticPredicate,
    build_pragmatic_context_delta,
)
from companion.style.errors import StyleValidationError

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
FIXTURE = Path(__file__).parents[2] / "fixtures" / "synthetic" / "pragmatic_context_scenarios.json"
SEMANTIC_FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "synthetic" / "chinese_semantic_routing_scenarios.json"
)


def _labels(delta: object, field: str) -> list[str]:
    return [signal.label.value for signal in getattr(delta, field)]


def _rule_ids(delta: PragmaticContextDelta) -> set[str]:
    return {
        rule_id for signal in delta.all_signals for rule_id in signal.calibration.matched_rule_ids
    }


def test_public_synthetic_set_covers_all_four_pragmatic_dimensions() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    covered: set[str] = set()

    for scenario in payload["scenarios"]:
        delta = build_pragmatic_context_delta(
            user_id="owner-a",
            trusted_user_id="owner-a",
            text=scenario["text"],
            recent_context=scenario["recent_context"],
            idle_seconds=scenario["idle_seconds"],
            observed_at=NOW,
        )

        assert delta.calibration_version == payload["calibration_version"]
        for field, expected in scenario["expected"].items():
            assert _labels(delta, field) == expected
            if expected:
                covered.add(field)
            for signal in getattr(delta, field):
                assert 0.0 <= signal.confidence <= signal.score <= 1.0
                assert signal.provenance.evidence_count >= 1
                assert signal.provenance.updated_at == NOW
                assert signal.calibration.calibration_version == payload["calibration_version"]
                assert signal.calibration.correct_count <= signal.calibration.evaluated_count
                assert signal.calibration.confidence_method is (
                    PragmaticConfidenceMethod.WILSON_LOWER_BOUND_95
                )
                assert signal.provenance.confidence_method is (
                    PragmaticConfidenceMethod.WILSON_LOWER_BOUND_95
                )
                assert signal.score == round(
                    signal.calibration.correct_count / signal.calibration.evaluated_count,
                    6,
                )

    assert covered == {
        "speech_act",
        "expressed_affect",
        "relationship_context",
        "dialogue_stage",
    }


def test_public_calibration_cases_recompute_every_registered_rule() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    aggregates: dict[str, list[int]] = {}
    observed_rule_ids: set[str] = set()

    for case in payload["calibration_cases"]:
        delta = build_pragmatic_context_delta(
            user_id="owner-a",
            trusted_user_id="owner-a",
            text=case["text"],
            recent_context=case["recent_context"],
            idle_seconds=case["idle_seconds"],
            observed_at=NOW,
        )
        emitted_rule_ids = _rule_ids(delta)
        observed_rule_ids.update(emitted_rule_ids)
        passed = (case["rule_id"] in emitted_rule_ids) is case["expected_emitted"]
        aggregate = aggregates.setdefault(case["rule_id"], [0, 0])
        aggregate[0] += int(passed)
        aggregate[1] += 1

    fixture_rule_ids = set(aggregates)
    assert observed_rule_ids == fixture_rule_ids
    assert all(evaluated == 5 for _, evaluated in aggregates.values())

    for rule_id, (correct, evaluated) in aggregates.items():
        calibration = PragmaticCalibrationBasis(
            calibration_version=payload["calibration_version"],
            correct_count=correct,
            evaluated_count=evaluated,
            confidence_method=PragmaticConfidenceMethod.WILSON_LOWER_BOUND_95,
            matched_rule_ids=(rule_id,),
        )
        matching_signal = next(
            signal
            for case in payload["calibration_cases"]
            if case["rule_id"] == rule_id
            for signal in build_pragmatic_context_delta(
                user_id="owner-a",
                trusted_user_id="owner-a",
                text=case["text"],
                recent_context=case["recent_context"],
                idle_seconds=case["idle_seconds"],
                observed_at=NOW,
            ).all_signals
            if rule_id in signal.calibration.matched_rule_ids
        )
        assert matching_signal.calibration == calibration


@pytest.mark.parametrize(
    ("text", "forbidden_labels"),
    [
        ("我今天不开心", {PragmaticLabel.JOY}),
        ("我不想你", {PragmaticLabel.AFFECTION, PragmaticLabel.AFFECTIONATE}),
        ("不用陪陪我", {PragmaticLabel.REQUEST, PragmaticLabel.SUPPORT_SEEKING}),
    ],
)
def test_locally_negated_phrases_do_not_emit_opposite_signals(
    text: str,
    forbidden_labels: set[PragmaticLabel],
) -> None:
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text=text,
        recent_context=[],
        idle_seconds=None,
        observed_at=NOW,
    )

    assert forbidden_labels.isdisjoint(signal.label for signal in delta.all_signals)


def test_context_delta_is_multi_signal_and_not_a_style_intent_hard_route() -> None:
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text="今天有点难过,陪陪我好吗?",
        recent_context=["合成上文"],
        idle_seconds=10,
        observed_at=NOW,
    )

    assert _labels(delta, "speech_act") == ["question", "request"]
    assert _labels(delta, "expressed_affect") == ["sadness"]
    assert delta.model_dump().get("intent") is None
    assert all(signal.score not in {0.0, 0.5} for signal in delta.all_signals)


def test_scores_and_confidence_are_reproducible_from_public_counts() -> None:
    first = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text="你好",
        recent_context=[],
        idle_seconds=None,
        observed_at=NOW,
    )
    second = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text="你好",
        recent_context=[],
        idle_seconds=None,
        observed_at=NOW,
    )

    assert first == second
    greeting = first.speech_act[0]
    assert greeting.score == 0.8
    assert greeting.calibration.correct_count == 4
    legacy_field_name = "positive" + "_count"
    assert legacy_field_name not in greeting.calibration.model_dump()
    assert greeting.calibration.evaluated_count == 5
    assert greeting.confidence == 0.375535
    assert greeting.provenance.observed_from is PragmaticObservedFrom.CURRENT_REQUEST


def test_reopening_uses_deterministic_idle_state_without_replacing_other_dimensions() -> None:
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text="你好,我好开心",
        recent_context=["上次的合成上文"],
        idle_seconds=21_600,
        observed_at=NOW,
    )

    assert _labels(delta, "speech_act") == ["greeting"]
    assert _labels(delta, "expressed_affect") == ["joy"]
    assert _labels(delta, "dialogue_stage") == ["reopening"]
    assert delta.dialogue_stage[0].provenance.observed_from is (
        PragmaticObservedFrom.DETERMINISTIC_REQUEST_STATE
    )


def test_non_owner_fails_closed_before_request_text_is_inspected() -> None:
    class ExplodingText(str):
        def strip(self, *args: object, **kwargs: object) -> str:
            raise AssertionError("non-owner text was inspected")

    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        build_pragmatic_context_delta(
            user_id="owner-b",
            trusted_user_id="owner-a",
            text=ExplodingText("synthetic"),
            recent_context=[],
            idle_seconds=None,
            observed_at=NOW,
        )


@pytest.mark.parametrize("user_id", ["", " owner-a", True, 7])
def test_user_scope_requires_exact_builtin_user_id(user_id: object) -> None:
    with pytest.raises((StyleValidationError, ValueError)):
        build_pragmatic_context_delta(
            user_id=user_id,  # type: ignore[arg-type]
            trusted_user_id="owner-a",
            text="你好",
            recent_context=[],
            idle_seconds=None,
            observed_at=NOW,
        )


def test_pragmatic_models_reject_cross_dimension_labels_and_private_provenance() -> None:
    calibration = PragmaticCalibrationBasis(
        calibration_version="public-synthetic-pragmatics-v1",
        correct_count=4,
        evaluated_count=5,
        confidence_method=PragmaticConfidenceMethod.WILSON_LOWER_BOUND_95,
        matched_rule_ids=("speech-greeting-v1",),
    )
    provenance = PragmaticSignalProvenance(
        source_type="public_synthetic_calibration",
        evidence_count=1,
        observed_from=PragmaticObservedFrom.CURRENT_REQUEST,
        updated_at=NOW,
        confidence_method=PragmaticConfidenceMethod.WILSON_LOWER_BOUND_95,
    )

    with pytest.raises(ValidationError, match="label does not belong"):
        PragmaticSignal(
            dimension=PragmaticDimension.EXPRESSED_AFFECT,
            label=PragmaticLabel.GREETING,
            score=0.8,
            confidence=0.375535,
            provenance=provenance,
            calibration=calibration,
        )
    with pytest.raises(ValidationError, match="registered dimension and label"):
        PragmaticSignal(
            dimension=PragmaticDimension.SPEECH_ACT,
            label=PragmaticLabel.QUESTION,
            score=0.8,
            confidence=0.375535,
            provenance=provenance,
            calibration=calibration,
        )
    with pytest.raises(ValidationError, match="empirical calibration accuracy"):
        PragmaticSignal(
            dimension=PragmaticDimension.SPEECH_ACT,
            label=PragmaticLabel.GREETING,
            score=0.7,
            confidence=0.375535,
            provenance=provenance,
            calibration=calibration,
        )
    with pytest.raises(ValidationError, match="calibrated Wilson bound"):
        PragmaticSignal(
            dimension=PragmaticDimension.SPEECH_ACT,
            label=PragmaticLabel.GREETING,
            score=0.8,
            confidence=0.8,
            provenance=provenance,
            calibration=calibration,
        )
    with pytest.raises(ValidationError):
        PragmaticSignalProvenance.model_validate(
            {
                **provenance.model_dump(),
                "raw_text": "synthetic private text",
            }
        )


def test_public_contracts_reject_fabricated_calibration_evidence() -> None:
    valid = {
        "calibration_version": "public-synthetic-pragmatics-v1",
        "correct_count": 4,
        "evaluated_count": 5,
        "confidence_method": PragmaticConfidenceMethod.WILSON_LOWER_BOUND_95,
        "matched_rule_ids": ("speech-greeting-v1",),
    }

    with pytest.raises(ValidationError, match="calibration version"):
        PragmaticCalibrationBasis(**{**valid, "calibration_version": "fabricated-v9"})
    with pytest.raises(ValidationError, match="registered"):
        PragmaticCalibrationBasis(**{**valid, "matched_rule_ids": ("fabricated-rule",)})
    with pytest.raises(ValidationError, match="registered calibration counts"):
        PragmaticCalibrationBasis(**{**valid, "correct_count": 999, "evaluated_count": 1000})
    with pytest.raises(ValidationError, match="calibration version"):
        PragmaticContextDelta(
            user_id="owner-a",
            calibration_version="fabricated-v9",
            created_at=NOW,
        )


def test_context_delta_does_not_expose_personality_or_long_term_state_fields() -> None:
    forbidden = {
        "personality",
        "mental_health",
        "long_term_emotion",
        "values",
        "profile",
        "style_intent",
    }

    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text="我有点累",
        recent_context=[],
        idle_seconds=None,
        observed_at=NOW,
    )

    assert forbidden.isdisjoint(type(delta).model_fields)
    serialized = json.dumps(delta.model_dump(mode="json"), ensure_ascii=False)
    assert "我有点累" not in serialized
    assert "raw_text" not in serialized


def test_semantic_frames_distinguish_role_swaps_and_are_reproducible() -> None:
    texts = (
        "我困了\uff0c先睡了",
        "你困了就早点睡",
        "我来陪你",
        "你陪我一会儿",
        "我想你",
        "你想我吗\uff1f",
    )
    deltas = tuple(
        build_pragmatic_context_delta(
            user_id="owner-a",
            trusted_user_id="owner-a",
            text=text,
            recent_context=[],
            idle_seconds=None,
            observed_at=NOW,
        )
        for text in texts
    )

    frame_signatures = {
        tuple(frame.semantic_definition_id for frame in delta.semantic_frames) for delta in deltas
    }
    assert len(frame_signatures) == 6
    assert deltas[0].semantic_frames[0].actor is SemanticParticipant.OWNER
    assert deltas[1].semantic_frames[0].target is SemanticParticipant.ADDRESSEE
    assert deltas[2].semantic_frames[0].actor is SemanticParticipant.OWNER
    assert deltas[3].semantic_frames[0].actor is SemanticParticipant.ADDRESSEE
    assert deltas[4].semantic_frames[0].experiencer is SemanticParticipant.OWNER
    assert deltas[5].semantic_frames[0].experiencer is SemanticParticipant.ADDRESSEE
    assert all(
        frame.registry_version == PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION
        for delta in deltas
        for frame in delta.semantic_frames
    )


@pytest.mark.parametrize(
    ("text", "expected_mode", "expected_polarity"),
    [
        (
            "他说晚安了吗\uff1f",
            SemanticDiscourseMode.REPORTED,
            SemanticPolarity.AFFIRMED,
        ),
        (
            "晚安是什么意思\uff1f",
            SemanticDiscourseMode.MENTION,
            SemanticPolarity.AFFIRMED,
        ),
        ("我没说晚安", SemanticDiscourseMode.REPORTED, SemanticPolarity.NEGATED),
    ],
)
def test_semantic_frames_express_use_mention_reported_and_negation(
    text: str,
    expected_mode: SemanticDiscourseMode,
    expected_polarity: SemanticPolarity,
) -> None:
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text=text,
        recent_context=[],
        idle_seconds=None,
        observed_at=NOW,
    )

    assert delta.semantic_frames
    assert all(frame.discourse_mode is expected_mode for frame in delta.semantic_frames)
    assert any(frame.polarity is expected_polarity for frame in delta.semantic_frames)


@pytest.mark.parametrize(
    ("text", "predicate", "mode"),
    [
        (
            "小王说谢谢了",
            SemanticPredicate.REPORTED_CONTENT,
            SemanticDiscourseMode.REPORTED,
        ),
        (
            "她有点累",
            SemanticPredicate.THIRD_PARTY_CONTENT,
            SemanticDiscourseMode.USE,
        ),
    ],
)
def test_non_owner_content_emits_auditable_neutral_frame_without_style_signals(
    text: str,
    predicate: SemanticPredicate,
    mode: SemanticDiscourseMode,
) -> None:
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text=text,
        recent_context=[],
        idle_seconds=None,
        observed_at=NOW,
    )

    assert delta.all_signals == ()
    assert len(delta.semantic_frames) == 1
    assert delta.semantic_frames[0].predicate is predicate
    assert delta.semantic_frames[0].actor is SemanticParticipant.THIRD_PARTY
    assert delta.semantic_frames[0].discourse_mode is mode
    assert text not in delta.model_dump_json()


def test_semantic_frame_can_resolve_bounded_recent_context_without_serializing_it() -> None:
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text="我也是",
        recent_context=["我想你"],
        idle_seconds=None,
        observed_at=NOW,
    )

    assert delta.semantic_frames[0].context_resolution is SemanticContextResolution.RESOLVED_CONTEXT
    serialized = delta.model_dump_json()
    assert "我也是" not in serialized
    assert "我想你" not in serialized
    assert "path" not in serialized.lower()


def test_semantic_frame_registry_rejects_fabricated_or_role_tampered_definitions() -> None:
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text="我想你",
        recent_context=[],
        idle_seconds=None,
        observed_at=NOW,
    )
    payload = delta.semantic_frames[0].model_dump(mode="python")

    with pytest.raises(ValidationError, match="definition is not registered"):
        SemanticFrame.model_validate(
            {**payload, "semantic_definition_id": "fabricated-semantic-v9"}
        )
    with pytest.raises(ValidationError, match="registered definition"):
        SemanticFrame.model_validate({**payload, "target": SemanticParticipant.OWNER})


def test_semantic_fixture_has_the_frozen_40_case_eight_category_shape() -> None:
    payload = json.loads(SEMANTIC_FIXTURE.read_text(encoding="utf-8"))
    categories = {case["category"] for case in payload["cases"]}

    assert payload["semantic_registry_version"] == PUBLIC_SEMANTIC_FRAME_REGISTRY_VERSION
    assert len(payload["cases"]) == 40
    assert categories == {
        "low_overlap_same",
        "high_overlap_different",
        "negation",
        "role_swap",
        "needs_context",
        "compound",
        "true_open_set",
        "controls",
    }
