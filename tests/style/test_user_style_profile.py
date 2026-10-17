from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from companion.style import (
    SingleTrustedUserStyleProfileStore,
    SourceApprovalState,
    StyleConfidenceMethod,
    StyleFeatureKind,
    StyleFeatureProvenance,
    StyleFeatureSourceType,
    StyleObservedFrom,
    UserStyleDerivedArtifacts,
    UserStyleEvidenceLedger,
    UserStyleFeature,
    UserStyleProfile,
    UserStyleProfileConsent,
    UserStyleProfileGateway,
    UserStyleProfileStatus,
    UserStyleSnapshotReadResult,
    UserStyleSnapshotReadStatus,
)
from companion.style.errors import StyleValidationError

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _consent(user_id: str = "owner-a") -> UserStyleProfileConsent:
    return UserStyleProfileConsent(
        user_id=user_id,
        authorization_version="consent-v1",
        status=UserStyleProfileStatus.AUTHORIZED,
        updated_at=NOW,
    )


def _provenance(*, evidence_count: int = 8) -> StyleFeatureProvenance:
    return StyleFeatureProvenance(
        source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
        evidence_count=evidence_count,
        observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
        source_version="safe-v1",
        updated_at=NOW,
        confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
    )


def _profile(user_id: str = "owner-a") -> UserStyleProfile:
    return UserStyleProfile(
        user_id=user_id,
        profile_version="profile-v1",
        authorization_version="consent-v1",
        features=(
            UserStyleFeature(
                kind=StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
                value=9.0,
                confidence=0.9,
                provenance=_provenance(),
            ),
            UserStyleFeature(
                kind=StyleFeatureKind.EMOJI_USAGE_RATIO,
                value=0.25,
                confidence=0.8,
                provenance=_provenance(evidence_count=6),
            ),
        ),
        created_at=NOW,
        updated_at=NOW,
    )


def _evidence(user_id: str = "owner-a") -> UserStyleEvidenceLedger:
    return UserStyleEvidenceLedger(
        user_id=user_id,
        profile_version="profile-v1",
        authorization_version="consent-v1",
        feature_evidence={
            StyleFeatureKind.MESSAGE_LENGTH_MEDIAN: 8,
            StyleFeatureKind.EMOJI_USAGE_RATIO: 6,
        },
        updated_at=NOW,
    )


def _derived(user_id: str = "owner-a") -> UserStyleDerivedArtifacts:
    return UserStyleDerivedArtifacts(
        user_id=user_id,
        profile_version="profile-v1",
        authorization_version="consent-v1",
        statistics_version="stats-v1",
        index_version="index-v1",
        cache_version="cache-v1",
        updated_at=NOW,
    )


def _store(tmp_path: Path, trusted_user_id: str = "owner-a") -> SingleTrustedUserStyleProfileStore:
    return SingleTrustedUserStyleProfileStore(
        trusted_user_id=trusted_user_id,
        private_data_dir=tmp_path,
        consent_source=tmp_path / "style-profile-consent.json",
        profile_source=tmp_path / "style-profile.json",
        evidence_source=tmp_path / "style-profile-evidence.json",
        statistics_source=tmp_path / "style-profile-statistics.json",
        index_source=tmp_path / "style-profile-index.json",
        cache_source=tmp_path / "style-profile-cache.json",
    )


class _SourceGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.state: SourceApprovalState | None = SourceApprovalState(
            owner_id="owner-a",
            batch_digest="a" * 64,
            package_digest="b" * 64,
            approved_indices=(0, 2),
            revision="v1-" + "c" * 64,
        )
        self.failure = False

    def current_state(self, user_id: str) -> SourceApprovalState | None:
        self.calls.append(user_id)
        if self.failure:
            raise RuntimeError("synthetic source unavailable")
        return self.state


def _bound_store(tmp_path: Path, gateway: _SourceGateway) -> SingleTrustedUserStyleProfileStore:
    return SingleTrustedUserStyleProfileStore(
        trusted_user_id="owner-a",
        private_data_dir=tmp_path,
        consent_source=tmp_path / "style-profile-consent.json",
        profile_source=tmp_path / "style-profile.json",
        evidence_source=tmp_path / "style-profile-evidence.json",
        statistics_source=tmp_path / "style-profile-statistics.json",
        index_source=tmp_path / "style-profile-index.json",
        cache_source=tmp_path / "style-profile-cache.json",
        strict_source_binding=True,
        source_gateway=gateway,
    )


def _create_bound_profile(store: SingleTrustedUserStyleProfileStore, version: str) -> None:
    store.create_profile(
        user_id="owner-a",
        profile=_profile().model_copy(update={"authorization_version": version}),
        evidence=_evidence().model_copy(update={"authorization_version": version}),
        derived=_derived().model_copy(update={"authorization_version": version}),
    )


def test_strict_binding_grants_reads_across_store_rebuild_and_rejects_legacy(
    tmp_path: Path,
) -> None:
    legacy = _store(tmp_path)
    legacy.grant_consent(user_id="owner-a", consent=_consent())
    legacy.create_profile(
        user_id="owner-a", profile=_profile(), evidence=_evidence(), derived=_derived()
    )
    gateway = _SourceGateway()
    strict = _bound_store(tmp_path, gateway)
    assert (
        strict.read_snapshot_state("owner-a").status is not UserStyleSnapshotReadStatus.AUTHORIZED
    )
    with pytest.raises(StyleValidationError):
        strict.grant_consent(user_id="owner-a", consent=_consent())
    with pytest.raises(StyleValidationError):
        strict.create_profile(
            user_id="owner-a", profile=_profile(), evidence=_evidence(), derived=_derived()
        )
    consent = strict.grant_bound_consent(user_id="owner-a", updated_at=NOW)
    assert consent.authorization_version != "consent-v1"
    assert consent.source_binding == gateway.state
    assert not any(path.exists() for path in strict.derived_artifact_paths)
    _create_bound_profile(strict, consent.authorization_version)
    assert (
        _bound_store(tmp_path, gateway).read_snapshot_state("owner-a").status
        is UserStyleSnapshotReadStatus.AUTHORIZED
    )


@pytest.mark.parametrize(
    "changed",
    [
        "owner_id",
        "batch_digest",
        "package_digest",
        "approved_indices",
        "revision",
        "missing",
        "empty",
        "failure",
    ],
)
def test_strict_binding_invalidates_each_source_change(tmp_path: Path, changed: str) -> None:
    gateway = _SourceGateway()
    store = _bound_store(tmp_path, gateway)
    consent = store.grant_bound_consent(user_id="owner-a", updated_at=NOW)
    _create_bound_profile(store, consent.authorization_version)
    if changed == "missing":
        gateway.state = None
    elif changed == "empty":
        gateway.state = gateway.state.model_copy(update={"approved_indices": ()})
    elif changed == "failure":
        gateway.failure = True
    else:
        value = {
            "owner_id": "owner-b",
            "batch_digest": "d" * 64,
            "package_digest": "e" * 64,
            "approved_indices": (0,),
            "revision": "v1-" + "f" * 64,
        }[changed]
        gateway.state = gateway.state.model_copy(update={changed: value})
    assert store.read_snapshot_state("owner-a").status is not UserStyleSnapshotReadStatus.AUTHORIZED
    assert store.load_snapshot("owner-a") is None
    assert store.view_profile("owner-a") is None
    with pytest.raises(StyleValidationError):
        _create_bound_profile(store, consent.authorization_version)


def test_strict_binding_regrant_revoke_delete_and_wrong_owner(tmp_path: Path) -> None:
    gateway = _SourceGateway()
    store = _bound_store(tmp_path, gateway)
    with pytest.raises(StyleValidationError):
        store.grant_bound_consent(user_id="owner-b", updated_at=NOW)
    with pytest.raises(StyleValidationError):
        store.read_snapshot_state("owner-b")
    assert gateway.calls == []
    gateway.state = None
    with pytest.raises(StyleValidationError):
        store.grant_bound_consent(user_id="owner-a", updated_at=NOW)
    assert not store.consent_source.exists()
    gateway.state = _SourceGateway().state
    first = store.grant_bound_consent(user_id="owner-a", updated_at=NOW)
    _create_bound_profile(store, first.authorization_version)
    second = store.grant_bound_consent(user_id="owner-a", updated_at=NOW)
    assert second.authorization_version != first.authorization_version
    assert not any(path.exists() for path in store.derived_artifact_paths)
    _create_bound_profile(store, second.authorization_version)
    store.revoke_consent(user_id="owner-a", updated_at=NOW)
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.UNAVAILABLE
    assert not any(path.exists() for path in store.derived_artifact_paths)
    store.delete_user_style("owner-a")
    assert not any(path.exists() for path in store.artifact_paths)


@pytest.mark.parametrize(
    "change",
    [
        {"batch_digest": "bad"},
        {"package_digest": "bad"},
        {"revision": "v1-bad"},
        {"approved_indices": (1, 1)},
        {"approved_indices": (2, 1)},
        {"approved_indices": (True,)},
    ],
)
def test_strict_binding_rejects_malformed_gateway_state(tmp_path: Path, change: dict) -> None:
    gateway = _SourceGateway()
    gateway.state = gateway.state.model_copy(update=change)
    store = _bound_store(tmp_path, gateway)
    with pytest.raises(StyleValidationError, match="style_source_unavailable"):
        store.grant_bound_consent(user_id="owner-a", updated_at=NOW)
    assert not store.consent_source.exists()


def test_strict_delete_failure_leaves_revoked_consent(tmp_path: Path, monkeypatch) -> None:
    gateway = _SourceGateway()
    store = _bound_store(tmp_path, gateway)
    consent = store.grant_bound_consent(user_id="owner-a", updated_at=NOW)
    _create_bound_profile(store, consent.authorization_version)

    def failed_delete(_paths) -> None:
        raise StyleValidationError("style_profile_delete_failed", "synthetic deletion failed")

    monkeypatch.setattr(store, "_delete_paths", failed_delete)
    with pytest.raises(StyleValidationError, match="style_profile_delete_failed"):
        store.delete_user_style("owner-a")
    assert store.view_consent("owner-a").status is UserStyleProfileStatus.REVOKED
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.UNAVAILABLE


def test_profile_models_reject_private_content_and_invalid_provenance() -> None:
    with pytest.raises(ValidationError):
        UserStyleFeature.model_validate(
            {
                "kind": "favorite_place",
                "value": "synthetic-secret-place",
                "confidence": 0.9,
                "provenance": _provenance().model_dump(),
            }
        )

    with pytest.raises(ValidationError):
        StyleFeatureProvenance(
            source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
            evidence_count=0,
            observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
            source_version="safe-v1",
            updated_at=NOW,
            confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
        )

    with pytest.raises(ValidationError):
        UserStyleFeature(
            kind=StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE,
            value="synthetic phrase",
            confidence=1.0,
            provenance=_provenance(),
        )

    with pytest.raises(ValidationError):
        StyleFeatureProvenance(
            source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
            evidence_count=1,
            observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
            source_version="/private/chat/export.json",
            updated_at=NOW,
            confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
        )


def test_profile_requires_unique_stable_features_and_consistent_timestamps() -> None:
    feature = UserStyleFeature(
        kind=StyleFeatureKind.FOLLOW_UP_QUESTION_RATIO,
        value=0.2,
        confidence=0.7,
        provenance=_provenance(),
    )
    with pytest.raises(ValidationError):
        UserStyleProfile(
            user_id="owner-a",
            profile_version="profile-v1",
            authorization_version="consent-v1",
            features=(feature, feature),
            created_at=NOW,
            updated_at=NOW,
        )

    with pytest.raises(ValidationError):
        UserStyleProfile(
            user_id="owner-a",
            profile_version="profile-v1",
            authorization_version="consent-v1",
            features=(feature,),
            created_at=NOW,
            updated_at=datetime(2026, 9, 20, 11, 59, tzinfo=UTC),
        )


def _length_feature(kind: StyleFeatureKind, value: float) -> UserStyleFeature:
    return UserStyleFeature(
        kind=kind,
        value=value,
        confidence=0.9,
        provenance=_provenance(),
    )


@pytest.mark.parametrize(
    "features",
    [
        (
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_Q25, 10.0),
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_MEDIAN, 9.0),
        ),
        (
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_MEDIAN, 10.0),
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_Q75, 9.0),
        ),
        (
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_Q25, 10.0),
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_Q75, 9.0),
        ),
    ],
)
def test_profile_rejects_inverted_message_length_quantiles(
    features: tuple[UserStyleFeature, ...],
) -> None:
    with pytest.raises(ValidationError, match="message length quantiles"):
        UserStyleProfile(
            user_id="owner-a",
            profile_version="profile-v1",
            authorization_version="consent-v1",
            features=features,
            created_at=NOW,
            updated_at=NOW,
        )


@pytest.mark.parametrize(
    "values",
    [
        (9.0, 9.0, 9.0),
        (4.0, 9.0, 14.0),
    ],
)
def test_profile_accepts_equal_or_increasing_message_length_quantiles(
    values: tuple[float, float, float],
) -> None:
    q25, median, q75 = values

    profile = UserStyleProfile(
        user_id="owner-a",
        profile_version="profile-v1",
        authorization_version="consent-v1",
        features=(
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_Q25, q25),
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_MEDIAN, median),
            _length_feature(StyleFeatureKind.MESSAGE_LENGTH_Q75, q75),
        ),
        created_at=NOW,
        updated_at=NOW,
    )

    assert tuple(feature.value for feature in profile.features) == values


def test_trusted_user_can_authorize_create_read_and_view_profile(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a",
        profile=_profile(),
        evidence=_evidence(),
        derived=_derived(),
    )

    snapshot = store.load_snapshot("owner-a")

    assert snapshot is not None
    assert snapshot.consent.status is UserStyleProfileStatus.AUTHORIZED
    assert snapshot.profile == _profile()
    assert store.view_profile("owner-a") == _profile()
    assert snapshot.evidence_count == 14
    assert all(path.exists() for path in store.artifact_paths)


def test_snapshot_read_result_requires_exact_status_snapshot_pair(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a", profile=_profile(), evidence=_evidence(), derived=_derived()
    )
    snapshot = store.load_snapshot("owner-a")
    assert snapshot is not None
    for status in (UserStyleSnapshotReadStatus.UNAVAILABLE, UserStyleSnapshotReadStatus.CORRUPT):
        with pytest.raises(ValidationError):
            UserStyleSnapshotReadResult(status=status, snapshot=snapshot)
    with pytest.raises(ValidationError):
        UserStyleSnapshotReadResult(status=UserStyleSnapshotReadStatus.AUTHORIZED, snapshot=None)
    revoked = snapshot.model_copy(
        update={
            "consent": snapshot.consent.model_copy(
                update={"status": UserStyleProfileStatus.REVOKED}
            )
        }
    )
    with pytest.raises(ValidationError):
        UserStyleSnapshotReadResult(status=UserStyleSnapshotReadStatus.AUTHORIZED, snapshot=revoked)
    with pytest.raises(ValidationError):
        UserStyleSnapshotReadResult.model_validate(
            {"status": "unavailable", "snapshot": None, "source_path": "private"}
        )
    with pytest.raises(ValidationError):
        UserStyleSnapshotReadResult(
            status=UserStyleSnapshotReadStatus.AUTHORIZED,
            snapshot=snapshot,
        ).status = UserStyleSnapshotReadStatus.CORRUPT


def test_snapshot_read_state_authorized_and_legacy_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a", profile=_profile(), evidence=_evidence(), derived=_derived()
    )
    result = store.read_snapshot_state("owner-a")
    assert result.status is UserStyleSnapshotReadStatus.AUTHORIZED
    assert result.snapshot == store.load_snapshot("owner-a")
    assert set(result.model_dump()) == {"status", "snapshot"}


def test_snapshot_read_state_unavailable_for_no_consent_revocation_and_all_missing(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.UNAVAILABLE
    store.grant_consent(user_id="owner-a", consent=_consent())
    all_missing = store.read_snapshot_state("owner-a")
    assert all_missing.status is UserStyleSnapshotReadStatus.UNAVAILABLE
    assert all_missing.snapshot is None
    assert store.load_snapshot("owner-a") is None
    store.revoke_consent(user_id="owner-a", updated_at=NOW)
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.UNAVAILABLE


@pytest.mark.parametrize("missing_index", range(5))
def test_snapshot_read_state_corrupt_for_each_single_missing_artifact(
    tmp_path: Path, missing_index: int
) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a", profile=_profile(), evidence=_evidence(), derived=_derived()
    )
    store.derived_artifact_paths[missing_index].unlink()
    result = store.read_snapshot_state("owner-a")
    assert result.status is UserStyleSnapshotReadStatus.CORRUPT
    assert result.snapshot is None
    assert store.load_snapshot("owner-a") is None


def test_snapshot_read_state_corrupt_for_malformed_consent_and_wrong_owner(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.consent_source.write_text("{}", encoding="utf-8")
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.CORRUPT
    store.consent_source.write_bytes(_consent("owner-b").model_dump_json().encode())
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.CORRUPT


def test_snapshot_read_state_corrupt_for_malformed_and_mismatched_artifacts(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a", profile=_profile(), evidence=_evidence(), derived=_derived()
    )
    profile_path = store.derived_artifact_paths[0]
    profile_path.write_text("{}", encoding="utf-8")
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.CORRUPT
    profile_path.write_bytes(_profile("owner-b").model_dump_json().encode())
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.CORRUPT
    profile_path.write_bytes(
        _profile()
        .model_copy(update={"authorization_version": "consent-v2"})
        .model_dump_json()
        .encode()
    )
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.CORRUPT


def test_snapshot_read_state_corrupt_for_derived_version_and_evidence_mismatch(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a", profile=_profile(), evidence=_evidence(), derived=_derived()
    )
    statistics_path = store.derived_artifact_paths[2]
    statistics_path.write_bytes(
        _derived().model_copy(update={"statistics_version": "stats-v2"}).model_dump_json().encode()
    )
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.CORRUPT
    statistics_path.write_bytes(_derived().model_dump_json().encode())
    evidence_path = store.derived_artifact_paths[1]
    evidence_path.write_bytes(
        _evidence()
        .model_copy(update={"feature_evidence": {StyleFeatureKind.MESSAGE_LENGTH_MEDIAN: 7}})
        .model_dump_json()
        .encode()
    )
    assert store.read_snapshot_state("owner-a").status is UserStyleSnapshotReadStatus.CORRUPT


def test_snapshot_read_state_rejects_non_owner_before_path_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)

    def forbidden_probe(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-owner reached path probe")

    monkeypatch.setattr(store, "_validate_fixed_paths", forbidden_probe)
    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        store.read_snapshot_state("owner-b")
    assert store.load_snapshot("owner-b") is None


def test_snapshot_read_state_rejects_path_escape_with_redacted_error(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = private_root / "artifacts"
    store = SingleTrustedUserStyleProfileStore(
        trusted_user_id="owner-a",
        private_data_dir=private_root,
        consent_source=linked / "consent.json",
        profile_source=linked / "profile.json",
        evidence_source=linked / "evidence.json",
        statistics_source=linked / "statistics.json",
        index_source=linked / "index.json",
        cache_source=linked / "cache.json",
    )
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(StyleValidationError) as caught:
        store.read_snapshot_state("owner-a")
    assert caught.value.code == "style_snapshot_read_failed"
    assert str(outside) not in str(caught.value)
    assert store.load_snapshot("owner-a") is None


def test_snapshot_read_state_io_failure_is_redacted_and_legacy_api_stays_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    original_read = Path.read_bytes

    def fail_consent_read(path: Path) -> bytes:
        if path == store.consent_source:
            raise OSError("synthetic-private-path-and-detail")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", fail_consent_read)
    with pytest.raises(StyleValidationError) as caught:
        store.read_snapshot_state("owner-a")
    assert "synthetic-private-path-and-detail" not in str(caught.value)
    assert str(tmp_path) not in str(caught.value)
    assert store.load_snapshot("owner-a") is None


def test_profile_create_requires_prior_matching_authorization(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(StyleValidationError, match="style_profile_not_authorized"):
        store.create_profile(
            user_id="owner-a",
            profile=_profile(),
            evidence=_evidence(),
            derived=_derived(),
        )

    assert not any(path.exists() for path in store.artifact_paths)


def test_non_owner_fails_before_any_profile_artifact_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    reads = 0

    def forbidden_read(*args: object, **kwargs: object) -> bytes:
        nonlocal reads
        reads += 1
        raise AssertionError("non-owner reached profile storage")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)

    assert store.load_snapshot("owner-b") is None
    assert store.view_profile("owner-b") is None
    assert reads == 0


def test_non_owner_cannot_mutate_profile_storage(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        store.grant_consent(user_id="owner-b", consent=_consent(user_id="owner-b"))
    with pytest.raises(StyleValidationError, match="untrusted_style_user"):
        store.delete_user_style("owner-b")

    assert not any(path.exists() for path in store.artifact_paths)


def test_corrupt_profile_fails_closed_without_partial_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a",
        profile=_profile(),
        evidence=_evidence(),
        derived=_derived(),
    )
    store.derived_artifact_paths[0].write_text("{}", encoding="utf-8")

    assert store.load_snapshot("owner-a") is None
    assert store.view_profile("owner-a") is None


def test_revocation_stops_reads_and_cleans_all_profile_derivatives(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a",
        profile=_profile(),
        evidence=_evidence(),
        derived=_derived(),
    )

    store.revoke_consent(user_id="owner-a", updated_at=NOW)

    assert store.load_snapshot("owner-a") is None
    assert store.view_profile("owner-a") is None
    assert store.consent_source.exists()
    assert not any(path.exists() for path in store.derived_artifact_paths)
    consent = store.view_consent("owner-a")
    assert consent is not None
    assert consent.status is UserStyleProfileStatus.REVOKED
    with pytest.raises(StyleValidationError, match="style_profile_not_authorized"):
        store.create_profile(
            user_id="owner-a",
            profile=_profile(),
            evidence=_evidence(),
            derived=_derived(),
        )
    assert not any(path.exists() for path in store.derived_artifact_paths)


def test_delete_removes_consent_profile_evidence_statistics_index_and_cache(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a",
        profile=_profile(),
        evidence=_evidence(),
        derived=_derived(),
    )

    store.delete_user_style("owner-a")

    assert store.load_snapshot("owner-a") is None
    assert store.view_consent("owner-a") is None
    assert not any(path.exists() for path in store.artifact_paths)


def test_new_authorization_version_invalidates_old_profile_derivatives(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a",
        profile=_profile(),
        evidence=_evidence(),
        derived=_derived(),
    )
    replacement = UserStyleProfileConsent(
        user_id="owner-a",
        authorization_version="consent-v2",
        status=UserStyleProfileStatus.AUTHORIZED,
        updated_at=NOW,
    )

    store.grant_consent(user_id="owner-a", consent=replacement)

    assert store.view_consent("owner-a") == replacement
    assert store.load_snapshot("owner-a") is None
    assert not any(path.exists() for path in store.derived_artifact_paths)


def test_delete_revalidates_fixed_paths_after_parent_symlink_swap(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = private_root / "profile-artifacts"
    store = SingleTrustedUserStyleProfileStore(
        trusted_user_id="owner-a",
        private_data_dir=private_root,
        consent_source=linked_parent / "consent.json",
        profile_source=linked_parent / "profile.json",
        evidence_source=linked_parent / "evidence.json",
        statistics_source=linked_parent / "statistics.json",
        index_source=linked_parent / "index.json",
        cache_source=linked_parent / "cache.json",
    )
    linked_parent.symlink_to(outside, target_is_directory=True)
    sentinel = outside / "consent.json"
    sentinel.write_text("synthetic sentinel", encoding="utf-8")

    with pytest.raises(StyleValidationError, match="style_profile_delete_failed"):
        store.delete_user_style("owner-a")

    assert sentinel.read_text(encoding="utf-8") == "synthetic sentinel"


def test_constructor_rejects_distinct_paths_resolving_to_same_file(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="distinct"):
        SingleTrustedUserStyleProfileStore(
            trusted_user_id="owner-a",
            private_data_dir=tmp_path,
            consent_source=real_parent / "shared.json",
            profile_source=alias_parent / "shared.json",
            evidence_source=tmp_path / "evidence.json",
            statistics_source=tmp_path / "statistics.json",
            index_source=tmp_path / "index.json",
            cache_source=tmp_path / "cache.json",
        )


def test_single_user_store_is_replaceable_by_recording_multi_user_fake() -> None:
    class RecordingMultiUserFake:
        def __init__(self) -> None:
            self.loaded: list[str] = []
            self.deleted: list[str] = []
            self.rows = {"user-a": object(), "user-b": object()}

        def load_snapshot(self, user_id: str) -> None:
            self.loaded.append(user_id)
            return None

        def read_snapshot_state(self, user_id: str) -> UserStyleSnapshotReadResult:
            self.loaded.append(user_id)
            return UserStyleSnapshotReadResult(
                status=UserStyleSnapshotReadStatus.UNAVAILABLE, snapshot=None
            )

        def view_profile(self, user_id: str) -> None:
            self.loaded.append(user_id)
            return None

        def view_consent(self, user_id: str) -> None:
            self.loaded.append(user_id)
            return None

        def grant_consent(self, *, user_id: str, consent: UserStyleProfileConsent) -> None:
            self.loaded.append(user_id)

        def create_profile(
            self,
            *,
            user_id: str,
            profile: UserStyleProfile,
            evidence: UserStyleEvidenceLedger,
            derived: UserStyleDerivedArtifacts,
        ) -> None:
            self.loaded.append(user_id)

        def revoke_consent(self, *, user_id: str, updated_at: datetime) -> None:
            self.deleted.append(user_id)

        def delete_user_style(self, user_id: str) -> None:
            self.deleted.append(user_id)
            self.rows.pop(user_id, None)

    fake: UserStyleProfileGateway = RecordingMultiUserFake()
    fake.load_snapshot("user-a")
    fake.load_snapshot("user-b")
    assert fake.read_snapshot_state("user-b").status is UserStyleSnapshotReadStatus.UNAVAILABLE
    fake.delete_user_style("user-a")

    assert fake.loaded == ["user-a", "user-b", "user-b"]
    assert fake.deleted == ["user-a"]
    assert "user-a" not in fake.rows
    assert "user-b" in fake.rows


def test_serialized_profile_contains_no_message_or_private_content_fields() -> None:
    payload = _profile().model_dump(mode="json")

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    for forbidden in (
        "message",
        "text",
        "topic",
        "name",
        "location",
        "girlfriend",
        "other_messages",
    ):
        assert forbidden not in keys(payload)


def test_persisted_profile_artifacts_contain_zero_counterparty_or_content_records(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.grant_consent(user_id="owner-a", consent=_consent())
    store.create_profile(
        user_id="owner-a",
        profile=_profile(),
        evidence=_evidence(),
        derived=_derived(),
    )

    serialized = b"\n".join(path.read_bytes() for path in store.artifact_paths).lower()
    for forbidden in (
        b"girlfriend",
        b"counterparty",
        b"other_messages",
        b"message_text",
        b"topic_noun",
        b"personality",
        b"mental_health",
        b"values",
    ):
        assert forbidden not in serialized
