from collections import Counter
from datetime import UTC, datetime

import pytest

from companion.style.errors import StyleValidationError
from companion.style.framework_evaluation import (
    BlindRating,
    CompleteBlindRatings,
    EvaluationLane,
    EvaluationLayer,
    FrameworkFixture,
    MetricStatus,
    _catalog,
    _snapshot,
    build_framework_observations,
    create_framework_blind_bundle,
    load_framework_fixture,
    mechanical_flags,
    score_framework_observations,
    unblind_complete_framework_ratings,
)
from companion.style.open_set_routing import OpenSetRouteKind, route_open_set_style
from companion.style.pragmatic_context import build_pragmatic_context_delta
from companion.style.user_profile import UserStyleProfileStatus


def test_phase8_public_lane_set_is_frozen() -> None:
    assert tuple(lane.value for lane in EvaluationLane) == (
        "baseline",
        "legacy_hard_rag",
        "phase1_7",
    )


@pytest.fixture(scope="module")
def evaluation():
    fixture = load_framework_fixture()
    observations = build_framework_observations(fixture)
    return fixture, observations


def test_fixture_has_four_balanced_layers_and_unique_public_ids(evaluation) -> None:
    fixture, observations = evaluation
    assert fixture.fixture_version == "public-synthetic-style-framework-v1"
    assert Counter(item.layer for item in fixture.scenarios) == dict.fromkeys(EvaluationLayer, 4)
    assert len({item.scenario_id for item in fixture.scenarios}) == 16
    assert len(observations) == 48
    assert {item.user_id for item in fixture.scenarios} == {
        "synthetic-owner-a",
        "synthetic-owner-b",
    }


def test_all_three_lanes_bind_identical_requests_and_parameters(evaluation) -> None:
    fixture, observations = evaluation
    for scenario in fixture.scenarios:
        group = [item for item in observations if item.scenario_id == scenario.scenario_id]
        assert {item.lane for item in group} == set(EvaluationLane)
        assert len({item.request_digest for item in group}) == 1
        assert len({item.user_id for item in group}) == 1
        assert len({item.input_version for item in group}) == 1
        assert len({item.parameter_version for item in group}) == 1
        assert len({item.digest_version for item in group}) == 1


def test_phase_lane_uses_actual_routes_and_retains_prior_without_rag(evaluation) -> None:
    fixture, observations = evaluation
    phase = {
        item.scenario_id: item for item in observations if item.lane is EvaluationLane.PHASE1_7
    }
    for scenario in fixture.scenarios:
        actual = phase[scenario.scenario_id]
        assert actual.route_kind is scenario.expected_route
        assert actual.stable_prior_count == 1
        assert actual.retrieval_status == scenario.retrieval_case
        assert actual.evidence_count == (1 if scenario.retrieval_case == "available" else 0)
        assert actual.abstained is (scenario.layer is EvaluationLayer.OPEN_SET)
    assert {item.retrieval_case for item in fixture.scenarios} == {
        "not_requested",
        "available",
        "no_match",
        "low_confidence",
        "corrupt",
        "unavailable",
    }


@pytest.mark.parametrize(
    "changed",
    [
        {"request_digest": "0" * 64},
        {"parameter_version": "offline-fixed-parameters-v2"},
        {"user_id": "synthetic-owner-b"},
        {"route_kind": OpenSetRouteKind.OPEN_SET},
        {"retrieval_status": "no_match"},
        {"source_version": "forged-v1"},
    ],
)
def test_observation_tampering_fails_closed_without_body(evaluation, changed) -> None:
    fixture, observations = evaluation
    first = observations[2]
    replacement = first.model_copy(update=changed)
    with pytest.raises(ValueError, match=r"^invalid framework evaluation observations$") as error:
        score_framework_observations(
            fixture, (observations[0], observations[1], replacement, *observations[3:])
        )
    assert "感谢你" not in str(error.value)


def test_fixture_expected_cannot_be_used_as_actual_and_digest_binds_body(evaluation) -> None:
    fixture, observations = evaluation
    changed = fixture.model_dump(mode="python")
    changed["scenarios"][0]["text"] = "合成修改请求"
    tampered_fixture = FrameworkFixture.model_validate(changed)
    with pytest.raises(ValueError, match=r"^invalid framework evaluation observations$"):
        score_framework_observations(tampered_fixture, observations)
    changed = fixture.model_dump(mode="python")
    changed["scenarios"][0]["scenario_id"] = changed["scenarios"][1]["scenario_id"]
    with pytest.raises(ValueError):
        FrameworkFixture.model_validate(changed)


def test_report_is_deterministic_content_free_and_has_not_measured_denominators(evaluation) -> None:
    fixture, observations = evaluation
    first = score_framework_observations(fixture, observations)
    second = score_framework_observations(fixture, build_framework_observations(fixture))
    assert first.model_dump_json() == second.model_dump_json()
    assert first.layer_counts == dict.fromkeys(EvaluationLayer, 4)
    serialized = first.model_dump_json()
    for scenario in fixture.scenarios:
        assert scenario.text not in serialized
        assert scenario.privacy_lure not in serialized
        assert scenario.user_id not in serialized
    route_baseline = next(
        item
        for item in first.metrics
        if item.lane is EvaluationLane.BASELINE
        and item.layer is EvaluationLayer.KNOWN
        and item.metric == "route_correct"
    )
    assert route_baseline.status is MetricStatus.NOT_APPLICABLE
    open_replies = next(
        item
        for item in first.metrics
        if item.lane is EvaluationLane.PHASE1_7
        and item.layer is EvaluationLayer.OPEN_SET
        and item.metric == "identity_safe"
    )
    assert (open_replies.numerator, open_replies.denominator) == (3, 3)
    assert [
        (item.status, item.numerator, item.denominator)
        for item in first.metrics
        if item.lane is EvaluationLane.PHASE1_7 and item.metric == "semantic_constraint"
    ] == [
        (MetricStatus.NOT_MEASURED, None, None),
        (MetricStatus.NOT_MEASURED, None, None),
        (MetricStatus.MEASURED, 1, 1),
        (MetricStatus.NOT_MEASURED, None, None),
    ]
    assert "winner" not in serialized


def test_mechanical_rules_are_independent_and_use_nonwhitespace_coverage() -> None:
    example = "合成示例连续八字符以上"
    assert mechanical_flags(example, (example,), 1)[:2] == (True, True)
    assert mechanical_flags("示例连续八字符以上", (example,), 1) == (False, True, False)
    assert mechanical_flags("完全不同", (example,), 3) == (False, False, True)
    assert mechanical_flags("完全不同", (example,), 2) == (False, False, False)


def test_blind_bundle_is_deterministic_separate_and_requires_complete_ratings(evaluation) -> None:
    fixture, observations = evaluation
    bundle, key = create_framework_blind_bundle(fixture, observations, seed=23)
    again, again_key = create_framework_blind_bundle(fixture, observations, seed=23)
    assert bundle.model_dump_json() == again.model_dump_json()
    assert key.model_dump_json() == again_key.model_dump_json()
    assert "baseline" not in bundle.model_dump_json()
    assert "legacy_hard_rag" not in bundle.model_dump_json()
    assert "phase1_7" not in bundle.model_dump_json()
    assert set(bundle.items[0].answers) == {"A", "B", "C"}
    assert len(set(bundle.items[0].answers.values())) == 3
    complete = CompleteBlindRatings(
        bundle_digest=bundle.bundle_digest,
        items=tuple(
            BlindRating(scenario_id=item.scenario_id, preferred="none", similarity=0)
            for item in bundle.items
        ),
    )
    with pytest.raises(ValueError, match=r"^incomplete or invalid framework ratings$"):
        unblind_complete_framework_ratings(
            bundle, key, complete.model_copy(update={"items": complete.items[:-1]})
        )
    with pytest.raises(ValueError):
        unblind_complete_framework_ratings(
            bundle,
            key,
            complete.model_copy(update={"items": (*complete.items[:-1], complete.items[0])}),
        )
    assert set(unblind_complete_framework_ratings(bundle, key, complete).values()) == {None}
    altered = dict(key.labels)
    altered[bundle.items[0].scenario_id] = {
        "A": EvaluationLane.BASELINE,
        "B": EvaluationLane.PHASE1_7,
        "C": EvaluationLane.LEGACY_HARD_RAG,
    }
    if altered[bundle.items[0].scenario_id] == key.labels[bundle.items[0].scenario_id]:
        altered[bundle.items[0].scenario_id] = {
            "A": EvaluationLane.PHASE1_7,
            "B": EvaluationLane.BASELINE,
            "C": EvaluationLane.LEGACY_HARD_RAG,
        }
    with pytest.raises(ValueError):
        unblind_complete_framework_ratings(
            bundle, key.model_copy(update={"labels": altered}), complete
        )


def test_scorer_accepts_independently_recorded_reply_but_not_forged_structure(evaluation) -> None:
    fixture, observations = evaluation
    changed = observations[0].model_copy(update={"reply": "新的公开合成回答"})
    report = score_framework_observations(fixture, (changed, *observations[1:]))
    assert report.scenario_count == 16


def test_phase_public_functions_fail_closed_on_second_user_and_revocation() -> None:
    now = datetime(2026, 9, 23, tzinfo=UTC)
    with pytest.raises(StyleValidationError):
        build_pragmatic_context_delta(
            user_id="synthetic-owner-b",
            trusted_user_id="synthetic-owner-a",
            text="谢谢",
            recent_context=(),
            idle_seconds=0,
            observed_at=now,
        )
    delta = build_pragmatic_context_delta(
        user_id="synthetic-owner-a",
        trusted_user_id="synthetic-owner-a",
        text="谢谢",
        recent_context=(),
        idle_seconds=0,
        observed_at=now,
    )
    with pytest.raises(StyleValidationError):
        route_open_set_style(
            user_id="synthetic-owner-b",
            trusted_user_id="synthetic-owner-b",
            snapshot=_snapshot("synthetic-owner-a"),
            context_delta=delta,
            scenario_catalog=_catalog("synthetic-owner-b"),
        )
    snapshot = _snapshot("synthetic-owner-a")
    revoked = snapshot.model_copy(
        update={
            "consent": snapshot.consent.model_copy(
                update={"status": UserStyleProfileStatus.REVOKED}
            )
        }
    )
    with pytest.raises(StyleValidationError):
        route_open_set_style(
            user_id="synthetic-owner-a",
            trusted_user_id="synthetic-owner-a",
            snapshot=revoked,
            context_delta=delta,
            scenario_catalog=_catalog("synthetic-owner-a"),
        )
