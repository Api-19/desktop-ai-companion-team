from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from companion.contracts import StyleIntent
from companion.style import (
    PUBLIC_STYLE_ROUTING_VERSION,
    OpenSetRouteKind,
    OpenSetStyleRoute,
    PragmaticLabel,
    StyleConfidenceMethod,
    StyleFeatureKind,
    StyleFeatureProvenance,
    StyleFeatureSourceType,
    StyleObservedFrom,
    StyleScenarioPrototype,
    UserStyleDerivedArtifacts,
    UserStyleFeature,
    UserStyleProfile,
    UserStyleProfileConsent,
    UserStyleProfileStatus,
    UserStyleSnapshot,
    build_pragmatic_context_delta,
    route_open_set_style,
)
from companion.style.errors import StyleValidationError

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
SEMANTIC_FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "synthetic" / "chinese_semantic_routing_scenarios.json"
)
GENERALIZATION_FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "synthetic" / "chinese_semantic_generalization_v1.json"
)


def _snapshot(user_id: str = "owner-a") -> UserStyleSnapshot:
    provenance = StyleFeatureProvenance(
        source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
        evidence_count=8,
        observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
        source_version="safe-v1",
        updated_at=NOW,
        confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
    )
    return UserStyleSnapshot(
        consent=UserStyleProfileConsent(
            user_id=user_id,
            authorization_version="consent-v1",
            status=UserStyleProfileStatus.AUTHORIZED,
            updated_at=NOW,
        ),
        profile=UserStyleProfile(
            user_id=user_id,
            profile_version="profile-v1",
            authorization_version="consent-v1",
            features=(
                UserStyleFeature(
                    kind=StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
                    value=8.0,
                    confidence=0.9,
                    provenance=provenance,
                ),
            ),
            created_at=NOW,
            updated_at=NOW,
        ),
        evidence_count=8,
        derived=UserStyleDerivedArtifacts(
            user_id=user_id,
            profile_version="profile-v1",
            authorization_version="consent-v1",
            statistics_version="stats-v1",
            index_version="index-v1",
            cache_version="cache-v1",
            updated_at=NOW,
        ),
    )


def _delta(
    text: str,
    *,
    user_id: str = "owner-a",
    recent_context: tuple[str, ...] = (),
    idle_seconds: int | None = None,
):
    return build_pragmatic_context_delta(
        user_id=user_id,
        trusted_user_id=user_id,
        text=text,
        recent_context=recent_context,
        idle_seconds=idle_seconds,
        observed_at=NOW,
    )


def _scenario(
    scenario_id: str,
    *labels: PragmaticLabel,
    user_id: str = "owner-a",
    intent: StyleIntent | None = None,
    semantic_definition_id: str = "scenario-coarse-labels-only-v1",
) -> StyleScenarioPrototype:
    return StyleScenarioPrototype(
        user_id=user_id,
        scenario_id=scenario_id,
        required_labels=labels,
        auxiliary_intent=intent,
        prototype_version="public-synthetic-routing-v1",
        semantic_definition_id=semantic_definition_id,
    )


def test_known_route_uses_one_complete_candidate() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢"),
        scenario_catalog=(
            _scenario(
                "gratitude",
                PragmaticLabel.GRATITUDE,
                intent=StyleIntent.COURTESY_THANKS,
            ),
            _scenario("support", PragmaticLabel.SUPPORT_SEEKING),
        ),
    )

    assert result.route_kind is OpenSetRouteKind.KNOWN
    assert result.abstained is False
    assert result.unknown is False
    assert [candidate.scenario_id for candidate in result.candidates] == ["gratitude"]
    assert result.candidates[0].weight == 1.0
    assert result.candidates[0].matched_labels == (PragmaticLabel.GRATITUDE,)
    assert result.candidates[0].missing_labels == ()
    assert result.stable_profile == _snapshot().profile
    assert result.context_delta.user_id == "owner-a"


def test_nearby_route_blends_multiple_candidates_with_explainable_weights() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("我今天有点难过"),
        scenario_catalog=(
            _scenario(
                "sad-support",
                PragmaticLabel.SADNESS,
                PragmaticLabel.SUPPORT_SEEKING,
            ),
            _scenario(
                "sad-request",
                PragmaticLabel.SADNESS,
                PragmaticLabel.REQUEST,
            ),
            _scenario("greeting", PragmaticLabel.GREETING),
        ),
    )

    assert result.route_kind is OpenSetRouteKind.NEARBY
    assert [candidate.scenario_id for candidate in result.candidates] == [
        "sad-request",
        "sad-support",
    ]
    assert [candidate.weight for candidate in result.candidates] == [0.5, 0.5]
    assert sum(candidate.weight for candidate in result.candidates) == 1.0
    for candidate in result.candidates:
        assert candidate.coverage == 0.5
        assert candidate.matched_labels == (PragmaticLabel.SADNESS,)
        assert len(candidate.missing_labels) == 1
        assert candidate.raw_score > 0.0


def test_open_set_abstains_without_fabricating_a_match() -> None:
    delta = _delta("合成蓝色纸船慢慢转圈")
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=delta,
        scenario_catalog=(
            _scenario("greeting", PragmaticLabel.GREETING),
            _scenario("farewell", PragmaticLabel.FAREWELL, PragmaticLabel.CLOSING),
        ),
    )

    assert result.route_kind is OpenSetRouteKind.OPEN_SET
    assert result.abstained is True
    assert result.unknown is True
    assert result.candidates == ()
    assert result.stable_profile.profile_version == "profile-v1"
    assert result.context_delta == delta
    serialized = result.model_dump_json()
    assert "合成蓝色纸船" not in serialized
    assert "StyleIntent" not in serialized


def test_weak_partial_overlap_abstains_below_nearby_threshold() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("我今天有点难过"),
        scenario_catalog=(
            _scenario(
                "weak-overlap",
                PragmaticLabel.SADNESS,
                PragmaticLabel.REQUEST,
                PragmaticLabel.SUPPORT_SEEKING,
            ),
        ),
    )

    assert result.route_kind is OpenSetRouteKind.OPEN_SET
    assert result.candidates == ()


def test_compound_route_keeps_multiple_complete_scenarios() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢你,晚安"),
        scenario_catalog=(
            _scenario(
                "appreciation",
                PragmaticLabel.GRATITUDE,
                PragmaticLabel.AFFILIATIVE,
            ),
            _scenario(
                "farewell",
                PragmaticLabel.FAREWELL,
                PragmaticLabel.CLOSING,
            ),
            _scenario("question", PragmaticLabel.QUESTION),
        ),
    )

    assert result.route_kind is OpenSetRouteKind.COMPOUND
    assert [candidate.scenario_id for candidate in result.candidates] == [
        "farewell",
        "appreciation",
    ]
    assert result.candidates[0].raw_score > result.candidates[1].raw_score
    assert all(candidate.coverage == 1.0 for candidate in result.candidates)
    assert sum(candidate.weight for candidate in result.candidates) == 1.0
    assert all(candidate.weight > 0.0 for candidate in result.candidates)


def test_style_intent_is_auxiliary_and_does_not_change_routing_weight() -> None:
    without_intent = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢"),
        scenario_catalog=(_scenario("gratitude", PragmaticLabel.GRATITUDE),),
    )
    with_intent = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢"),
        scenario_catalog=(
            _scenario(
                "gratitude",
                PragmaticLabel.GRATITUDE,
                intent=StyleIntent.WEATHER_COLD_REMINDER,
            ),
        ),
    )

    assert without_intent.route_kind is with_intent.route_kind is OpenSetRouteKind.KNOWN
    assert without_intent.candidates[0].weight == with_intent.candidates[0].weight
    assert without_intent.candidates[0].raw_score == with_intent.candidates[0].raw_score
    assert with_intent.candidates[0].auxiliary_intent is StyleIntent.WEATHER_COLD_REMINDER


def test_catalog_and_results_are_deterministic_regardless_of_input_order() -> None:
    scenarios = (
        _scenario("z-support", PragmaticLabel.SADNESS, PragmaticLabel.SUPPORT_SEEKING),
        _scenario("a-request", PragmaticLabel.SADNESS, PragmaticLabel.REQUEST),
    )
    kwargs: dict[str, Any] = {
        "user_id": "owner-a",
        "trusted_user_id": "owner-a",
        "snapshot": _snapshot(),
        "context_delta": _delta("我今天有点难过"),
    }

    first = route_open_set_style(**kwargs, scenario_catalog=scenarios)
    second = route_open_set_style(**kwargs, scenario_catalog=tuple(reversed(scenarios)))

    assert first == second
    assert first.routing_version == PUBLIC_STYLE_ROUTING_VERSION


def test_more_specific_complete_signature_dominates_its_subset() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢你"),
        scenario_catalog=(
            _scenario("gratitude-only", PragmaticLabel.GRATITUDE),
            _scenario(
                "appreciation",
                PragmaticLabel.GRATITUDE,
                PragmaticLabel.AFFILIATIVE,
            ),
        ),
    )

    assert result.route_kind is OpenSetRouteKind.KNOWN
    assert [candidate.scenario_id for candidate in result.candidates] == ["appreciation"]


def test_cross_user_profile_delta_and_candidates_fail_closed() -> None:
    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        route_open_set_style(
            user_id="owner-b",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            context_delta=_delta("谢谢"),
            scenario_catalog=(),
        )
    with pytest.raises(StyleValidationError, match="style_route_scope_mismatch"):
        route_open_set_style(
            user_id="owner-a",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            context_delta=_delta("谢谢", user_id="owner-b"),
            scenario_catalog=(),
        )
    with pytest.raises(StyleValidationError, match="style_route_scope_mismatch"):
        route_open_set_style(
            user_id="owner-a",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            context_delta=_delta("谢谢"),
            scenario_catalog=(_scenario("foreign", PragmaticLabel.GRATITUDE, user_id="owner-b"),),
        )


def test_non_owner_is_rejected_before_route_inputs_are_inspected() -> None:
    class ExplodingCatalog:
        def __iter__(self):
            raise AssertionError("non-owner catalog was inspected")

    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        route_open_set_style(
            user_id="owner-b",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            context_delta=_delta("谢谢"),
            scenario_catalog=ExplodingCatalog(),  # type: ignore[arg-type]
        )


def test_profile_must_be_authorized_complete_and_same_user() -> None:
    snapshot = _snapshot().model_copy(
        update={
            "consent": _snapshot().consent.model_copy(
                update={"status": UserStyleProfileStatus.REVOKED}
            )
        }
    )

    with pytest.raises(StyleValidationError, match="style_profile_unavailable"):
        route_open_set_style(
            user_id="owner-a",
            trusted_user_id="owner-a",
            snapshot=snapshot,
            context_delta=_delta("谢谢"),
            scenario_catalog=(),
        )


def test_route_models_reject_duplicate_labels_and_private_extension_fields() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _scenario(
            "duplicate",
            PragmaticLabel.GRATITUDE,
            PragmaticLabel.GRATITUDE,
        )
    with pytest.raises(ValidationError):
        StyleScenarioPrototype.model_validate(
            {
                **_scenario("safe", PragmaticLabel.GRATITUDE).model_dump(),
                "raw_chat": "synthetic private text",
            }
        )
    with pytest.raises(ValidationError, match="version is not registered"):
        StyleScenarioPrototype(
            user_id="owner-a",
            scenario_id="fabricated",
            required_labels=(PragmaticLabel.GRATITUDE,),
            prototype_version="fabricated-v9",
        )


def test_catalog_rejects_duplicate_semantic_signatures() -> None:
    with pytest.raises(StyleValidationError, match="invalid_style_route"):
        route_open_set_style(
            user_id="owner-a",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            context_delta=_delta("谢谢"),
            scenario_catalog=(
                _scenario("first", PragmaticLabel.GRATITUDE),
                _scenario(
                    "second",
                    PragmaticLabel.GRATITUDE,
                    intent=StyleIntent.COURTESY_THANKS,
                ),
            ),
        )


def test_forged_catalog_model_is_sanitized_at_the_route_boundary() -> None:
    forged = _scenario("safe", PragmaticLabel.GRATITUDE).model_copy(
        update={"prototype_version": "synthetic/private/path"}
    )

    with pytest.raises(StyleValidationError, match="invalid_style_route") as caught:
        route_open_set_style(
            user_id="owner-a",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            context_delta=_delta("谢谢"),
            scenario_catalog=(forged,),
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "synthetic/private/path" not in str(caught.value)


def _known_result() -> OpenSetStyleRoute:
    return route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢"),
        scenario_catalog=(_scenario("gratitude", PragmaticLabel.GRATITUDE),),
    )


def _nearby_result() -> OpenSetStyleRoute:
    return route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("我今天有点难过"),
        scenario_catalog=(
            _scenario("sad-request", PragmaticLabel.SADNESS, PragmaticLabel.REQUEST),
            _scenario(
                "sad-support",
                PragmaticLabel.SADNESS,
                PragmaticLabel.SUPPORT_SEEKING,
            ),
        ),
    )


@pytest.mark.parametrize("candidate_count", (0, 2))
def test_public_route_rejects_missing_or_duplicate_candidates(candidate_count: int) -> None:
    payload = _known_result().model_dump(mode="python")
    payload["candidates"] = payload["candidates"] * candidate_count

    with pytest.raises(ValidationError):
        OpenSetStyleRoute.model_validate(payload)


@pytest.mark.parametrize("evidence_count", (True, 0, -1, "1"))
def test_public_route_rejects_invalid_snapshot_evidence_count(evidence_count: Any) -> None:
    payload = _known_result().model_dump(mode="python")
    payload["snapshot"]["evidence_count"] = evidence_count

    with pytest.raises(ValidationError):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_candidate_label_absent_from_delta() -> None:
    payload = _known_result().model_dump(mode="python")
    payload["candidates"][0]["matched_labels"] = (PragmaticLabel.GREETING,)

    with pytest.raises(ValidationError, match="matched labels"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_missing_label_that_delta_actually_contains() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢你"),
        scenario_catalog=(
            _scenario(
                "partial",
                PragmaticLabel.GRATITUDE,
                PragmaticLabel.QUESTION,
            ),
        ),
    )
    payload = result.model_dump(mode="python")
    payload["candidates"][0]["missing_labels"] = (PragmaticLabel.AFFILIATIVE,)

    with pytest.raises(ValidationError, match="missing labels"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_arbitrary_raw_score() -> None:
    payload = _known_result().model_dump(mode="python")
    payload["candidates"][0]["raw_score"] = 0.999

    with pytest.raises(ValidationError, match="raw score"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_weights_that_only_sum_to_one() -> None:
    payload = _nearby_result().model_dump(mode="python")
    payload["candidates"][0]["weight"] = 0.75
    payload["candidates"][1]["weight"] = 0.25

    with pytest.raises(ValidationError, match="normalized weight"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_candidate_from_another_user_scope() -> None:
    payload = _known_result().model_dump(mode="python")
    assert "user_id" in payload["candidates"][0]
    payload["candidates"][0]["user_id"] = "owner-b"

    with pytest.raises(ValidationError, match="user scope"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_catalog_from_another_user_scope() -> None:
    payload = _known_result().model_dump(mode="python")
    payload["scenario_catalog"][0]["user_id"] = "owner-b"

    with pytest.raises(ValidationError, match="catalog user scope"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_revoked_snapshot() -> None:
    payload = _known_result().model_dump(mode="python")
    assert "snapshot" in payload
    payload["snapshot"]["consent"]["status"] = UserStyleProfileStatus.REVOKED

    with pytest.raises(ValidationError, match="authorized snapshot"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_non_deterministic_candidate_order() -> None:
    payload = _nearby_result().model_dump(mode="python")
    payload["candidates"] = list(reversed(payload["candidates"]))

    with pytest.raises(ValidationError, match="deterministic"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_subset_candidate_removed_by_dominance() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢你"),
        scenario_catalog=(
            _scenario("gratitude-only", PragmaticLabel.GRATITUDE),
            _scenario(
                "appreciation",
                PragmaticLabel.GRATITUDE,
                PragmaticLabel.AFFILIATIVE,
            ),
        ),
    )
    subset = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢你"),
        scenario_catalog=(_scenario("gratitude-only", PragmaticLabel.GRATITUDE),),
    ).candidates[0]
    payload = result.model_dump(mode="python")
    payload["candidates"] = [
        *payload["candidates"],
        subset.model_dump(mode="python"),
    ]

    with pytest.raises(ValidationError, match="selection or order"):
        OpenSetStyleRoute.model_validate(payload)


def test_public_route_rejects_collapsing_a_compound_route_to_known() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢你,晚安"),
        scenario_catalog=(
            _scenario(
                "appreciation",
                PragmaticLabel.GRATITUDE,
                PragmaticLabel.AFFILIATIVE,
            ),
            _scenario("farewell", PragmaticLabel.FAREWELL, PragmaticLabel.CLOSING),
        ),
    )
    payload = result.model_dump(mode="python")
    payload["route_kind"] = OpenSetRouteKind.KNOWN
    payload["candidates"] = payload["candidates"][:1]
    payload["candidates"][0]["weight"] = 1.0

    with pytest.raises(ValidationError, match="route kind"):
        OpenSetStyleRoute.model_validate(payload)


def _semantic_catalog(payload: dict[str, Any]) -> tuple[StyleScenarioPrototype, ...]:
    return tuple(
        _scenario(
            item["scenario_id"],
            *(PragmaticLabel(label) for label in item["required_labels"]),
            semantic_definition_id=item["semantic_definition_id"],
        )
        for item in payload["scenario_catalog"]
    )


def test_frozen_40_case_semantic_matrix_routes_exactly_by_category() -> None:
    payload = json.loads(SEMANTIC_FIXTURE.read_text(encoding="utf-8"))
    catalog = _semantic_catalog(payload)
    category_results: Counter[str] = Counter()

    for case in payload["cases"]:
        route = route_open_set_style(
            user_id="owner-a",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            context_delta=_delta(case["text"], recent_context=tuple(case["recent_context"])),
            scenario_catalog=catalog,
        )
        assert route.route_kind.value == case["expected_kind"], case["id"]
        assert {candidate.scenario_id for candidate in route.candidates} == set(case["expected"]), (
            case["id"]
        )
        category_results[case["category"]] += 1

    assert category_results == Counter(
        {
            "low_overlap_same": 5,
            "high_overlap_different": 5,
            "negation": 5,
            "role_swap": 6,
            "needs_context": 5,
            "compound": 5,
            "true_open_set": 5,
            "controls": 4,
        }
    )


def test_reconstructed_v1_33_case_generalization_matrix_routes_exactly_by_category() -> None:
    payload = json.loads(GENERALIZATION_FIXTURE.read_text(encoding="utf-8"))
    catalog = _semantic_catalog(payload)
    category_results: Counter[str] = Counter()

    assert payload["origin"] == "reconstructed-new-version-not-original-temporary-set"
    assert payload["case_count"] == len(payload["cases"]) == 33
    for case in payload["cases"]:
        route = route_open_set_style(
            user_id="owner-a",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            context_delta=_delta(case["text"], recent_context=tuple(case["recent_context"])),
            scenario_catalog=catalog,
        )
        actual = {candidate.scenario_id for candidate in route.candidates}
        assert route.route_kind.value == case["expected_kind"], case["id"]
        assert actual == set(case["expected"]), case["id"]
        category_results[case["category"]] += 1

    assert category_results == Counter(
        {
            "synonym_colloquial": 6,
            "direct_question": 3,
            "reported": 4,
            "mention": 4,
            "negation_scope": 4,
            "role_swap": 3,
            "resolved_context": 3,
            "compound": 3,
            "true_open_set": 3,
        }
    )


@pytest.mark.parametrize(
    ("text", "recent_context", "expected_kind", "expected"),
    [
        ("感激你啦", (), "known", {"gratitude"}),
        ("我去歇会儿", (), "known", {"sleep_farewell"}),
        ("我挺惦记你的", (), "known", {"affection"}),
        ("能陪我说说话不", (), "known", {"support_request"}),
        ("我陪你说会儿话", (), "known", {"care_support"}),
        ("我情绪有点低落", (), "nearby", {"sad_checkin", "sad_support"}),
        ("感谢是什么意思\uff1f", (), "known", {"generic_question"}),
        ("她问晚安怎么用\uff1f", (), "known", {"generic_question"}),
        ("我才不困", (), "open_set", set()),
        ("线性代数矩阵怎么求逆\uff1f", (), "open_set", set()),
        ("也是呢", ("我挺惦记你的",), "known", {"affection"}),
        ("她有点累", (), "open_set", set()),
        ("他已经睡了", (), "open_set", set()),
        ("小王说谢谢了", (), "open_set", set()),
        ("她说她晚安了", (), "open_set", set()),
        ("他说自己有点累", (), "open_set", set()),
        ("请问你累吗", (), "known", {"generic_question"}),
        ("我不难过只是有点累", (), "known", {"fatigue"}),
        ("她感谢你", (), "open_set", set()),
        ("朋友陪我聊聊", (), "open_set", set()),
        (
            "感激你\uff0c我去歇会儿啦",
            (),
            "compound",
            {"gratitude", "sleep_farewell"},
        ),
    ],
)
def test_semantic_family_isolation_examples_not_copied_from_the_33_case_fixture(
    text: str,
    recent_context: tuple[str, ...],
    expected_kind: str,
    expected: set[str],
) -> None:
    payload = json.loads(GENERALIZATION_FIXTURE.read_text(encoding="utf-8"))
    assert text not in {case["text"] for case in payload["cases"]}
    route = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta(text, recent_context=recent_context),
        scenario_catalog=_semantic_catalog(payload),
    )

    assert route.route_kind.value == expected_kind
    assert {candidate.scenario_id for candidate in route.candidates} == expected


@pytest.mark.parametrize(
    "text",
    ["他说晚安了吗\uff1f", "晚安是什么意思\uff1f", "我没说晚安"],
)
def test_mention_reported_and_negated_farewell_never_routes_as_sleep_farewell(
    text: str,
) -> None:
    payload = json.loads(SEMANTIC_FIXTURE.read_text(encoding="utf-8"))
    route = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta(text),
        scenario_catalog=_semantic_catalog(payload),
    )

    assert "sleep_farewell" not in {candidate.scenario_id for candidate in route.candidates}


@pytest.mark.parametrize("text", ["帮我算一下十七乘二十三", "这个量子公式怎么推\uff1f"])
def test_true_open_set_math_and_technical_questions_abstain(text: str) -> None:
    payload = json.loads(SEMANTIC_FIXTURE.read_text(encoding="utf-8"))
    route = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta(text),
        scenario_catalog=_semantic_catalog(payload),
    )

    assert route.route_kind is OpenSetRouteKind.OPEN_SET
    assert route.abstained is route.unknown is True
    assert route.candidates == ()


def test_semantic_definition_registry_rejects_fabricated_prototype_and_route() -> None:
    with pytest.raises(ValidationError, match="semantic definition"):
        _scenario(
            "forged",
            PragmaticLabel.GRATITUDE,
            semantic_definition_id="fabricated-semantic-v9",
        )

    payload = _known_result().model_dump(mode="python")
    payload["scenario_catalog"][0]["semantic_definition_id"] = "fabricated-semantic-v9"
    with pytest.raises(ValidationError, match="catalog"):
        OpenSetStyleRoute.model_validate(payload)


@pytest.mark.parametrize(
    ("labels", "semantic_definition_id"),
    [
        ((PragmaticLabel.GRATITUDE,), "scenario-fatigue-v1"),
        (
            (PragmaticLabel.FAREWELL, PragmaticLabel.CLOSING),
            "scenario-gratitude-v1",
        ),
    ],
)
def test_registered_semantic_definition_must_match_prototype_label_signature(
    labels: tuple[PragmaticLabel, ...],
    semantic_definition_id: str,
) -> None:
    with pytest.raises(ValidationError, match="semantic definition label signature"):
        _scenario(
            "mismatched",
            *labels,
            semantic_definition_id=semantic_definition_id,
        )


def test_registered_semantic_definition_cannot_be_swapped_on_route_candidate() -> None:
    payload = _known_result().model_dump(mode="python")
    payload["candidates"][0]["semantic_definition_id"] = "scenario-fatigue-v1"

    with pytest.raises(ValidationError, match="semantic definition label signature"):
        OpenSetStyleRoute.model_validate(payload)


def test_registered_semantic_definition_cannot_be_swapped_in_route_catalog() -> None:
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta("谢谢"),
        scenario_catalog=(
            _scenario(
                "gratitude",
                PragmaticLabel.GRATITUDE,
                semantic_definition_id="scenario-gratitude-v1",
            ),
        ),
    )
    payload = result.model_dump(mode="python")
    payload["scenario_catalog"][0]["semantic_definition_id"] = "scenario-fatigue-v1"

    with pytest.raises(ValidationError, match="catalog"):
        OpenSetStyleRoute.model_validate(payload)


def test_semantic_route_serialization_contains_no_request_or_reversible_material() -> None:
    payload = json.loads(SEMANTIC_FIXTURE.read_text(encoding="utf-8"))
    request_text = "我困了\uff0c先睡了"
    route = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=_snapshot(),
        context_delta=_delta(request_text),
        scenario_catalog=_semantic_catalog(payload),
    )
    serialized = route.model_dump_json()

    assert request_text not in serialized
    for forbidden in ("raw_text", "example", "negative_example", "file_path", "sha256"):
        assert forbidden not in serialized
