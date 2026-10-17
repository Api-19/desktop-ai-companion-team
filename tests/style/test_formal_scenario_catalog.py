"""Public synthetic acceptance tests for the formal Phase 3 scenario catalog."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import companion.style.open_set_routing as routing
from companion.style import (
    FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
    OpenSetRouteKind,
    PragmaticLabel,
    StyleConfidenceMethod,
    StyleFeatureKind,
    StyleFeatureProvenance,
    StyleFeatureSourceType,
    StyleObservedFrom,
    UserStyleDerivedArtifacts,
    UserStyleFeature,
    UserStyleProfile,
    UserStyleProfileConsent,
    UserStyleProfileStatus,
    UserStyleSnapshot,
    build_formal_style_scenario_catalog,
    build_pragmatic_context_delta,
    route_open_set_style,
)
from companion.style.errors import StyleValidationError

NOW = datetime(2026, 9, 23, tzinfo=UTC)
EXPECTED = {
    "affection": ("scenario-affection-v1", {"affection", "affectionate"}),
    "apology": ("scenario-apology-v1", {"apology", "repair"}),
    "care_support": ("scenario-care-support-v1", {"care_giving"}),
    "fatigue": ("scenario-fatigue-v1", {"fatigue"}),
    "generic_question": ("scenario-generic-question-v1", {"question"}),
    "gratitude": ("scenario-gratitude-v1", {"gratitude"}),
    "sad_checkin": ("scenario-sad-checkin-v1", {"sadness", "question"}),
    "sad_support": ("scenario-sad-support-v1", {"sadness", "support_seeking"}),
    "sleep_care": ("scenario-sleep-care-v1", {"care_giving"}),
    "sleep_farewell": ("scenario-sleep-farewell-v1", {"farewell", "closing"}),
    "support_request": ("scenario-support-request-v1", {"request", "support_seeking"}),
}


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


def _catalog(snapshot: UserStyleSnapshot | None = None):
    return build_formal_style_scenario_catalog(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=snapshot or _snapshot(),
        expected_catalog_version=FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
    )


def test_catalog_is_exact_complete_sorted_and_non_content() -> None:
    catalog = _catalog()
    assert type(catalog) is tuple
    assert len(catalog) == 11
    assert [item.scenario_id for item in catalog] == sorted(EXPECTED)
    assert {
        item.scenario_id: (
            item.semantic_definition_id,
            {label.value for label in item.required_labels},
        )
        for item in catalog
    } == EXPECTED
    assert all(item.user_id == "owner-a" and item.auxiliary_intent is None for item in catalog)
    assert catalog == _catalog()
    assert {item.scenario_id for item in catalog} != {
        item.semantic_definition_id for item in catalog
    }
    care = [item for item in catalog if item.required_labels == (PragmaticLabel.CARE_GIVING,)]
    assert {item.scenario_id for item in care} == {"care_support", "sleep_care"}


def test_non_owner_rejected_before_snapshot_or_table_inspection(monkeypatch) -> None:
    class ExplodingSnapshot:
        def __getattribute__(self, name):
            raise AssertionError(name)

    monkeypatch.setattr(routing, "_FORMAL_SCENARIO_ROWS", None, raising=False)
    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        build_formal_style_scenario_catalog(
            user_id="intruder",
            trusted_user_id="owner-a",
            snapshot=ExplodingSnapshot(),
            expected_catalog_version=FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
        )


@pytest.mark.parametrize("user_id,trusted_user_id", [("owner-a", 1), (1, "owner-a")])
def test_identity_requires_exact_builtin_strings(user_id, trusted_user_id) -> None:
    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        build_formal_style_scenario_catalog(
            user_id=user_id,
            trusted_user_id=trusted_user_id,
            snapshot=_snapshot(),
            expected_catalog_version=FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
        )


def test_revoked_cross_user_and_forged_snapshots_fail_closed() -> None:
    revoked = _snapshot().model_copy(
        update={
            "consent": _snapshot().consent.model_copy(
                update={"status": UserStyleProfileStatus.REVOKED}
            )
        }
    )
    forged = _snapshot().model_copy(update={"evidence_count": 0})
    for snapshot in (revoked, _snapshot("owner-b"), forged, object()):
        with pytest.raises(StyleValidationError, match="style_profile_unavailable"):
            _catalog(snapshot)


@pytest.mark.parametrize("version", ["wrong-v1", "", None])
def test_wrong_expected_version_fails_closed(version) -> None:
    with pytest.raises(StyleValidationError):
        build_formal_style_scenario_catalog(
            user_id="owner-a",
            trusted_user_id="owner-a",
            snapshot=_snapshot(),
            expected_catalog_version=version,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[:-1],
        lambda rows: (*rows, ("extra", *rows[0][1:])),
        lambda rows: (("synthetic", *rows[0][1:]), *rows[1:]),
        lambda rows: tuple((f"synthetic_{row[0]}", *row[1:]) for row in rows),
        lambda rows: (rows[1], rows[1], *rows[2:]),
        lambda rows: (*rows[:-1], (rows[-1][0], rows[0][1], rows[0][2])),
        lambda rows: (*rows[:-1], (rows[-1][0], "scenario-coarse-labels-only-v1", rows[-1][2])),
        lambda rows: (*rows[:-1], (rows[-1][0], rows[-1][1], rows[0][2])),
    ],
)
def test_incomplete_extra_substituted_duplicate_or_synthetic_table_fails(
    monkeypatch, mutation
) -> None:
    monkeypatch.setattr(routing, "_FORMAL_SCENARIO_ROWS", mutation(routing._FORMAL_SCENARIO_ROWS))
    with pytest.raises(StyleValidationError, match="invalid_style_route"):
        _catalog()


def test_registered_label_signature_corruption_fails(monkeypatch) -> None:
    signatures = dict(routing._SCENARIO_SEMANTIC_LABEL_SIGNATURES)
    signatures["scenario-gratitude-v1"] = frozenset({PragmaticLabel.FATIGUE})
    monkeypatch.setattr(routing, "_SCENARIO_SEMANTIC_LABEL_SIGNATURES", signatures)
    with pytest.raises(StyleValidationError, match="invalid_style_route"):
        _catalog()


@pytest.mark.parametrize(
    "text,kind",
    [
        ("谢谢", OpenSetRouteKind.KNOWN),
        ("我今天有点难过", OpenSetRouteKind.NEARBY),
        ("合成蓝色纸船慢慢转圈", OpenSetRouteKind.OPEN_SET),
        ("谢谢你,晚安", OpenSetRouteKind.COMPOUND),
    ],
)
def test_formal_catalog_preserves_existing_route_kinds(text, kind) -> None:
    snapshot = _snapshot()
    delta = build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text=text,
        recent_context=(),
        idle_seconds=None,
        observed_at=NOW,
    )
    result = route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=snapshot,
        context_delta=delta,
        scenario_catalog=_catalog(snapshot),
    )
    assert result.route_kind is kind
    assert len(result.scenario_catalog) == 11
