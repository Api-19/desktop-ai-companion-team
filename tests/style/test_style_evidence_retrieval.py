from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.style import (
    OpenSetRouteKind,
    PragmaticLabel,
    SingleTrustedUserStyleEvidenceStore,
    StyleConfidenceMethod,
    StyleEvidenceApproval,
    StyleEvidenceCatalog,
    StyleEvidenceConfidenceMethod,
    StyleEvidenceGateway,
    StyleEvidenceObservedFrom,
    StyleEvidenceRecord,
    StyleEvidenceRetrievalStatus,
    StyleEvidenceSourceType,
    StyleEvidenceVersions,
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

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
FIXTURE = Path("fixtures/synthetic/style_evidence_retrieval_scenarios.json")


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
            index_version="profile-index-v1",
            cache_version="profile-cache-v1",
            updated_at=NOW,
        ),
    )


def _scenario(scenario_id: str, *labels: PragmaticLabel, user_id: str = "owner-a"):
    return StyleScenarioPrototype(
        user_id=user_id,
        scenario_id=scenario_id,
        required_labels=labels,
        prototype_version="public-synthetic-routing-v1",
    )


def _route(
    text: str,
    scenarios: tuple[StyleScenarioPrototype, ...],
    *,
    user_id: str = "owner-a",
):
    delta = build_pragmatic_context_delta(
        user_id=user_id,
        trusted_user_id=user_id,
        text=text,
        recent_context=(),
        idle_seconds=None,
        observed_at=NOW,
    )
    return route_open_set_style(
        user_id=user_id,
        trusted_user_id=user_id,
        snapshot=_snapshot(user_id),
        context_delta=delta,
        scenario_catalog=scenarios,
    )


def _record(
    ref: str,
    scenario_id: str,
    content: str,
    *,
    confidence: float = 0.95,
    user_id: str = "owner-a",
) -> StyleEvidenceRecord:
    return StyleEvidenceRecord(
        user_id=user_id,
        evidence_ref=ref,
        scenario_id=scenario_id,
        content=content,
        confidence=confidence,
        approval=StyleEvidenceApproval.APPROVED_FOR_STYLE,
        source_type=StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL,
        observed_from=StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY,
        confidence_method=StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL,
        source_version="approval-v1",
        approved_at=NOW,
    )


def _catalog(*records: StyleEvidenceRecord, user_id: str = "owner-a") -> StyleEvidenceCatalog:
    return StyleEvidenceCatalog(
        user_id=user_id,
        authorization_version="consent-v1",
        profile_version="profile-v1",
        versions=StyleEvidenceVersions(
            evidence_version="evidence-v1",
            index_version="index-v1",
            cache_version="cache-v1",
        ),
        records=records,
        updated_at=NOW,
    )


def _store(tmp_path: Path) -> SingleTrustedUserStyleEvidenceStore:
    return SingleTrustedUserStyleEvidenceStore(
        trusted_user_id="owner-a",
        private_data_dir=tmp_path,
        evidence_source=tmp_path / "style-evidence.json",
        index_source=tmp_path / "style-evidence-index.json",
        cache_source=tmp_path / "style-evidence-cache.json",
        version_source=tmp_path / "style-evidence-version.json",
    )


def test_evidence_model_only_accepts_owner_approved_style_material() -> None:
    payload = _record("ev-1", "gratitude", "synthetic approved phrase").model_dump()
    payload["observed_from"] = "counterparty_messages"
    with pytest.raises(ValidationError):
        StyleEvidenceRecord.model_validate(payload)

    payload = _record("ev-1", "gratitude", "synthetic approved phrase").model_dump()
    payload["approval"] = "pending"
    with pytest.raises(ValidationError):
        StyleEvidenceRecord.model_validate(payload)


def test_public_synthetic_fixture_covers_phase5_status_matrix() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["fixture_version"] == "public-synthetic-style-evidence-v1"
    assert payload["privacy_class"] == "public_synthetic_only"
    assert {case["route_kind"] for case in payload["cases"]} == {
        "known",
        "nearby",
        "compound",
        "open_set",
    }
    assert {case["expected_status"] for case in payload["cases"]} == {
        "available",
        "no_match",
        "low_confidence",
        "corrupt",
        "not_requested",
    }


@pytest.mark.parametrize(
    ("text", "scenarios", "expected_kind", "expected_refs"),
    (
        (
            "谢谢",
            (_scenario("gratitude", PragmaticLabel.GRATITUDE),),
            OpenSetRouteKind.KNOWN,
            ("ev-gratitude",),
        ),
        (
            "我今天有点难过",
            (
                _scenario("sad-request", PragmaticLabel.SADNESS, PragmaticLabel.REQUEST),
                _scenario("sad-support", PragmaticLabel.SADNESS, PragmaticLabel.SUPPORT_SEEKING),
            ),
            OpenSetRouteKind.NEARBY,
            ("ev-sad-request", "ev-sad-support"),
        ),
        (
            "谢谢你,晚安",
            (
                _scenario("appreciation", PragmaticLabel.GRATITUDE, PragmaticLabel.AFFILIATIVE),
                _scenario("farewell", PragmaticLabel.FAREWELL, PragmaticLabel.CLOSING),
            ),
            OpenSetRouteKind.COMPOUND,
            ("ev-farewell", "ev-appreciation"),
        ),
    ),
)
def test_store_retrieves_approved_evidence_for_known_nearby_and_compound_routes(
    tmp_path: Path,
    text: str,
    scenarios: tuple[StyleScenarioPrototype, ...],
    expected_kind: OpenSetRouteKind,
    expected_refs: tuple[str, ...],
) -> None:
    store = _store(tmp_path)
    catalog = _catalog(
        _record("ev-gratitude", "gratitude", "synthetic gratitude phrase"),
        _record("ev-sad-request", "sad-request", "synthetic caring question"),
        _record("ev-sad-support", "sad-support", "synthetic support phrase"),
        _record("ev-appreciation", "appreciation", "synthetic appreciation phrase"),
        _record("ev-farewell", "farewell", "synthetic farewell phrase"),
    )
    store.publish_approved_evidence(user_id="owner-a", snapshot=_snapshot(), catalog=catalog)
    route = _route(text, scenarios)
    assert route.route_kind is expected_kind

    result = store.retrieve_evidence(user_id="owner-a", route=route)

    assert result.status is StyleEvidenceRetrievalStatus.AVAILABLE
    assert tuple(item.evidence_ref for item in result.matches) == expected_refs
    assert all(item.user_id == "owner-a" for item in result.matches)


def test_open_set_does_not_read_evidence_index_or_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    reads = 0

    def forbidden_read(*args: object, **kwargs: object) -> bytes:
        nonlocal reads
        reads += 1
        raise AssertionError("open-set route reached evidence storage")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    route = _route(
        "synthetic blue paper boat",
        (_scenario("gratitude", PragmaticLabel.GRATITUDE),),
    )
    assert route.route_kind is OpenSetRouteKind.OPEN_SET

    result = store.retrieve_evidence(user_id="owner-a", route=route)

    assert result.status is StyleEvidenceRetrievalStatus.NOT_REQUESTED
    assert result.matches == ()
    assert reads == 0


def test_no_match_low_confidence_and_corrupt_are_distinct(tmp_path: Path) -> None:
    store = _store(tmp_path)
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))

    store.publish_approved_evidence(
        user_id="owner-a",
        snapshot=_snapshot(),
        catalog=_catalog(_record("ev-other", "farewell", "synthetic farewell phrase")),
    )
    assert (
        store.retrieve_evidence(user_id="owner-a", route=route).status
        is StyleEvidenceRetrievalStatus.NO_MATCH
    )

    store.publish_approved_evidence(
        user_id="owner-a",
        snapshot=_snapshot(),
        catalog=_catalog(
            _record(
                "ev-low",
                "gratitude",
                "synthetic low confidence phrase",
                confidence=0.4,
            )
        ),
    )
    assert (
        store.retrieve_evidence(user_id="owner-a", route=route).status
        is StyleEvidenceRetrievalStatus.LOW_CONFIDENCE
    )

    store.evidence_source.write_text("{}", encoding="utf-8")
    assert (
        store.retrieve_evidence(user_id="owner-a", route=route).status
        is StyleEvidenceRetrievalStatus.CORRUPT
    )


@pytest.mark.parametrize(
    "artifact_name",
    ("evidence_source", "index_source", "cache_source", "version_source"),
)
def test_each_damaged_private_artifact_fails_closed(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    store = _store(tmp_path)
    store.publish_approved_evidence(
        user_id="owner-a",
        snapshot=_snapshot(),
        catalog=_catalog(_record("ev-1", "gratitude", "synthetic approved phrase")),
    )
    getattr(store, artifact_name).write_text("{}", encoding="utf-8")
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))

    result = store.retrieve_evidence(user_id="owner-a", route=route)

    assert result.status is StyleEvidenceRetrievalStatus.CORRUPT
    assert result.matches == ()


def test_non_owner_fails_before_any_evidence_artifact_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    reads = 0

    def forbidden_read(*args: object, **kwargs: object) -> bytes:
        nonlocal reads
        reads += 1
        raise AssertionError("non-owner reached evidence storage")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))

    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        store.retrieve_evidence(user_id="owner-b", route=route)
    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        store.view_approved_evidence("owner-b")
    assert reads == 0


def test_cross_user_catalog_and_route_fail_without_owner_fallback(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(StyleValidationError, match="invalid_style_evidence_catalog"):
        store.publish_approved_evidence(
            user_id="owner-a",
            snapshot=_snapshot(),
            catalog=_catalog(
                _record(
                    "ev-other",
                    "gratitude",
                    "synthetic other-user phrase",
                    user_id="owner-b",
                ),
                user_id="owner-b",
            ),
        )
    assert not any(path.exists() for path in store.artifact_paths)

    other_route = _route(
        "谢谢",
        (_scenario("gratitude", PragmaticLabel.GRATITUDE, user_id="owner-b"),),
        user_id="owner-b",
    )
    with pytest.raises(StyleValidationError, match="invalid_style_evidence_query"):
        store.retrieve_evidence(user_id="owner-a", route=other_route)
    assert not any(path.exists() for path in store.artifact_paths)


def test_view_withdraw_and_delete_cover_evidence_index_cache_and_version(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog(_record("ev-1", "gratitude", "synthetic approved phrase"))
    store.publish_approved_evidence(user_id="owner-a", snapshot=_snapshot(), catalog=catalog)
    assert store.view_approved_evidence("owner-a") == catalog.records
    assert all(path.exists() for path in store.artifact_paths)

    store.withdraw_evidence(
        user_id="owner-a",
        evidence_ref="ev-1",
        versions=StyleEvidenceVersions(
            evidence_version="evidence-v2",
            index_version="index-v2",
            cache_version="cache-v2",
        ),
        updated_at=NOW,
    )
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))
    assert store.view_approved_evidence("owner-a") == ()
    assert (
        store.retrieve_evidence(user_id="owner-a", route=route).status
        is StyleEvidenceRetrievalStatus.NO_MATCH
    )

    store.delete_user_evidence("owner-a")
    assert not any(path.exists() for path in store.artifact_paths)


def test_invalid_withdrawal_fails_with_fixed_public_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sentinel = "synthetic approved phrase sentinel"
    store.publish_approved_evidence(
        user_id="owner-a",
        snapshot=_snapshot(),
        catalog=_catalog(_record("ev-1", "gratitude", sentinel)),
    )

    with pytest.raises(
        StyleValidationError,
        match="invalid_style_evidence_withdrawal",
    ) as captured:
        store.withdraw_evidence(
            user_id="owner-a",
            evidence_ref="ev-1",
            versions=StyleEvidenceVersions(
                evidence_version="evidence-v2",
                index_version="index-v2",
                cache_version="cache-v2",
            ),
            updated_at=datetime(2026, 9, 21, 12, 0),
        )

    assert sentinel not in str(captured.value)


@pytest.mark.parametrize(
    "tamper",
    (
        "ref",
        "content",
        "confidence",
        "provenance",
        "source_content",
        "manifest_ref",
        "manifest_confidence",
        "manifest_provenance",
        "manifest_user",
        "manifest_version",
        "evidence_version",
        "user",
        "version",
        "status",
    ),
)
def test_result_recomputes_matches_and_rejects_forgery(tmp_path: Path, tamper: str) -> None:
    store = _store(tmp_path)
    store.publish_approved_evidence(
        user_id="owner-a",
        snapshot=_snapshot(),
        catalog=_catalog(_record("ev-1", "gratitude", "synthetic approved phrase")),
    )
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))
    forged = store.retrieve_evidence(user_id="owner-a", route=route).model_copy(deep=True)

    if tamper == "ref":
        object.__setattr__(forged.matches[0], "evidence_ref", "fabricated-ref")
    elif tamper == "content":
        object.__setattr__(forged.matches[0], "content", "fabricated content")
    elif tamper == "confidence":
        object.__setattr__(forged.matches[0], "confidence", 0.1)
    elif tamper == "provenance":
        object.__setattr__(forged.matches[0], "source_version", "fabricated-v9")
    elif tamper == "source_content":
        object.__setattr__(forged.matched_sources[0], "content", "fabricated content")
    elif tamper == "manifest_ref":
        object.__setattr__(forged.source_manifest.entries[0], "evidence_ref", "fabricated-ref")
    elif tamper == "manifest_confidence":
        object.__setattr__(forged.source_manifest.entries[0], "confidence", 0.1)
    elif tamper == "manifest_provenance":
        object.__setattr__(forged.source_manifest.entries[0], "source_version", "fabricated-v9")
    elif tamper == "manifest_user":
        object.__setattr__(forged.source_manifest.entries[0], "user_id", "owner-b")
    elif tamper == "manifest_version":
        object.__setattr__(forged.source_manifest.versions, "evidence_version", "fabricated-v9")
    elif tamper == "evidence_version":
        object.__setattr__(forged.versions, "evidence_version", "fabricated-v9")
    elif tamper == "user":
        object.__setattr__(forged, "user_id", "owner-b")
    elif tamper == "version":
        object.__setattr__(forged, "retrieval_version", "fabricated-v9")
    else:
        object.__setattr__(forged, "status", StyleEvidenceRetrievalStatus.NO_MATCH)

    with pytest.raises(ValidationError):
        type(forged).model_validate(forged.model_dump(mode="python"))


def test_result_serialization_contains_only_bounded_matched_content(tmp_path: Path) -> None:
    store = _store(tmp_path)
    matched = "synthetic matched approved phrase"
    matched_two = "synthetic second matched approved phrase"
    matched_three = "synthetic third matched approved phrase"
    overflow = "MATCHED_BUT_OUTSIDE_TOP_THREE_SENTINEL"
    unmatched = "UNMATCHED_PRIVATE_APPROVED_SENTINEL"
    store.publish_approved_evidence(
        user_id="owner-a",
        snapshot=_snapshot(),
        catalog=_catalog(
            _record("ev-1", "gratitude", matched),
            _record("ev-2", "gratitude", matched_two),
            _record("ev-3", "gratitude", matched_three),
            _record("ev-4", "gratitude", overflow),
            _record("ev-5", "farewell", unmatched),
        ),
    )
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))

    result = store.retrieve_evidence(user_id="owner-a", route=route)
    serialized = result.model_dump_json()

    assert matched in serialized
    assert matched_two in serialized
    assert matched_three in serialized
    assert overflow not in serialized
    assert unmatched not in serialized
    assert len(result.matched_sources) == len(result.matches) == 3
    assert all(not hasattr(entry, "content") for entry in result.source_manifest.entries)


def test_index_and_cache_never_persist_approved_expression_content(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sentinel = "synthetic expression body sentinel"
    store.publish_approved_evidence(
        user_id="owner-a",
        snapshot=_snapshot(),
        catalog=_catalog(_record("ev-1", "gratitude", sentinel)),
    )
    route = _route("谢谢", (_scenario("gratitude", PragmaticLabel.GRATITUDE),))
    store.retrieve_evidence(user_id="owner-a", route=route)

    assert sentinel not in store.index_source.read_text(encoding="utf-8")
    assert sentinel not in store.cache_source.read_text(encoding="utf-8")
    assert sentinel not in store.version_source.read_text(encoding="utf-8")


def test_gateway_can_be_replaced_by_multi_user_fake_without_cross_user_delete() -> None:
    class RecordingMultiUserFake:
        def __init__(self) -> None:
            self.rows = {"user-a": (), "user-b": ()}

        def retrieve_evidence(self, *, user_id: str, route: object) -> object:
            return self.rows[user_id]

        def view_approved_evidence(self, user_id: str) -> tuple[()]:
            return self.rows[user_id]

        def publish_approved_evidence(
            self, *, user_id: str, snapshot: object, catalog: object
        ) -> None:
            self.rows[user_id] = ()

        def withdraw_evidence(
            self,
            *,
            user_id: str,
            evidence_ref: str,
            versions: object,
            updated_at: datetime,
        ) -> None:
            self.rows[user_id] = ()

        def delete_user_evidence(self, user_id: str) -> None:
            self.rows.pop(user_id, None)

    fake: StyleEvidenceGateway = RecordingMultiUserFake()  # type: ignore[assignment]
    fake.delete_user_evidence("user-a")

    assert "user-a" not in fake.rows
    assert "user-b" in fake.rows


def test_delete_revalidates_paths_after_parent_symlink_swap(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = private_root / "evidence-artifacts"
    store = SingleTrustedUserStyleEvidenceStore(
        trusted_user_id="owner-a",
        private_data_dir=private_root,
        evidence_source=linked_parent / "evidence.json",
        index_source=linked_parent / "index.json",
        cache_source=linked_parent / "cache.json",
        version_source=linked_parent / "version.json",
    )
    linked_parent.symlink_to(outside, target_is_directory=True)
    sentinel = outside / "evidence.json"
    sentinel.write_text("synthetic sentinel", encoding="utf-8")

    with pytest.raises(StyleValidationError, match="style_evidence_delete_failed"):
        store.delete_user_evidence("owner-a")

    assert sentinel.read_text(encoding="utf-8") == "synthetic sentinel"
