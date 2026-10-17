import hashlib
import json
from datetime import UTC, datetime

import pytest

from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.style import (
    SingleTrustedUserStyleProfileStore,
    SourceApprovalState,
    StyleFeatureKind,
    UserStyleProfileConsent,
    UserStyleProfileStatus,
    UserStyleSnapshotReadStatus,
    generate_user_style_profile,
)
from companion.style.errors import StyleValidationError

NOW = datetime(2026, 9, 28, tzinfo=UTC)


def package(texts):
    return SafeCandidatePackage(
        created_at=NOW,
        candidates=[
            SafeStyleCandidate(
                text=text, category=CandidateCategory.GREETING, scene_tags=["greeting"]
            )
            for text in texts
        ],
        audit=CandidateSelectionAudit(
            processed_at=NOW,
            input_count=len(texts),
            owner_input_count=len(texts),
            dropped_other_count=0,
            allowed_category_counts={"greeting": len(texts)},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=len(texts),
            succeeded=True,
        ),
    )


def state_for(current, indices=None):
    digest = hashlib.sha256(
        json.dumps(
            current.model_dump(mode="json", warnings=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return SourceApprovalState(
        owner_id="synthetic-owner",
        batch_digest="a" * 64,
        package_digest=digest,
        approved_indices=tuple(range(len(current.candidates))) if indices is None else indices,
        revision="v1-" + "b" * 64,
    )


def consent_for(state):
    return UserStyleProfileConsent(
        user_id="synthetic-owner",
        authorization_version="bound-synthetic-authorization",
        status=UserStyleProfileStatus.AUTHORIZED,
        updated_at=NOW,
        scope="non_text_style_features_local_persistence",
        source_binding=state,
    )


def generate(current, consent=None, state=None, **kwargs):
    return generate_user_style_profile(
        user_id=kwargs.pop("user_id", "synthetic-owner"),
        package=current,
        consent=consent,
        source_state=state,
        observed_at=kwargs.pop("observed_at", NOW),
        **kwargs,
    )


@pytest.mark.parametrize("n,expected", [(20, (5.0, 10.5, 15.0)), (21, (6.0, 11.0, 16.0))])
def test_approved_subset_exact_quantiles_and_non_text_bundle(n, expected):
    current = package(["Z" * i for i in range(1, n + 1)] + ["UNAPPROVED_SYNTHETIC_SENTINEL"])
    state = state_for(current, tuple(range(n)))
    bundle = generate(current, consent_for(state), state)
    assert tuple(feature.value for feature in bundle.profile.features) == expected
    assert tuple(feature.kind for feature in bundle.profile.features) == (
        StyleFeatureKind.MESSAGE_LENGTH_Q25,
        StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
        StyleFeatureKind.MESSAGE_LENGTH_Q75,
    )
    assert all(feature.confidence == n / 100 for feature in bundle.profile.features)
    assert set(bundle.evidence.feature_evidence.values()) == {n}
    assert bundle.evidence.total_evidence_count == 3 * n
    assert bundle.profile.authorization_version == "bound-synthetic-authorization"
    assert (
        bundle.profile.profile_version
        == bundle.evidence.profile_version
        == bundle.derived.profile_version
    )
    assert all(
        feature.provenance.source_version == bundle.profile.profile_version
        for feature in bundle.profile.features
    )
    serialized = bundle.model_dump_json()
    assert "UNAPPROVED_SYNTHETIC_SENTINEL" not in serialized
    assert "ZZZZ" not in serialized
    for private_metadata in (state.package_digest, state.batch_digest, state.revision):
        assert private_metadata not in serialized
    assert set(bundle.model_dump()) == {"profile", "evidence", "derived"}


def test_unicode_whitespace_codepoints_and_repeated_candidates_count():
    current = package(["\t👩‍💻\u3000e\u0301\u00a0\n"] * 100)
    state = state_for(current)
    bundle = generate(current, consent_for(state), state)
    assert tuple(feature.value for feature in bundle.profile.features) == (5.0, 5.0, 5.0)
    assert all(feature.confidence == 1.0 for feature in bundle.profile.features)
    assert set(bundle.evidence.feature_evidence.values()) == {100}
    assert "👩‍💻" not in bundle.model_dump_json()
    assert (
        bundle.profile.updated_at == bundle.evidence.updated_at == bundle.derived.updated_at == NOW
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "revoked",
        "legacy",
        "scope",
        "state",
        "owner",
        "package",
        "unapproved",
        "bounds",
        "indices",
        "few",
        "zero",
        "empty",
        "long",
        "naive",
        "bad_package",
    ],
)
def test_generation_rejects_invalid_authorization_or_source_with_fixed_error(failure):
    current = package(["SYNTHETIC_APPROVED_SENTINEL"] * 20 + ["UNAPPROVED_SYNTHETIC_SENTINEL"])
    state = state_for(current, tuple(range(20)))
    consent = consent_for(state)
    kwargs = {}
    if failure == "missing":
        consent = None
    elif failure == "revoked":
        consent = consent.model_copy(update={"status": UserStyleProfileStatus.REVOKED})
    elif failure == "legacy":
        consent = consent.model_copy(update={"scope": None, "source_binding": None})
    elif failure == "scope":
        consent = consent.model_copy(update={"scope": "other-purpose"})
    elif failure == "state":
        state = state.model_copy(update={"revision": "v1-" + "c" * 64})
    elif failure == "owner":
        kwargs["user_id"] = "other-synthetic-owner"
    elif failure in {"package", "unapproved"}:
        texts = [candidate.text for candidate in current.candidates]
        texts[0 if failure == "package" else 20] = "TAMPERED_SYNTHETIC_SENTINEL"
        current = package(texts)
    elif failure in {"bounds", "indices", "few", "zero"}:
        indices = {
            "bounds": (*range(19), 200),
            "indices": (0,) * 20,
            "few": tuple(range(19)),
            "zero": (),
        }[failure]
        state = state.model_copy(update={"approved_indices": indices})
        consent = consent.model_copy(update={"source_binding": state})
    elif failure in {"empty", "long"}:
        current = package([" \t\n" if failure == "empty" else "Z" * 2001] * 20)
        state = state_for(current)
        consent = consent_for(state)
    elif failure == "naive":
        kwargs["observed_at"] = datetime(2026, 9, 28)
    elif failure == "bad_package":
        current = current.model_copy(
            update={"audit": current.audit.model_copy(update={"output_count": 0})}
        )
    with pytest.raises(StyleValidationError) as captured:
        generate(current, consent, state, **kwargs)
    assert (
        str(captured.value)
        == "style_profile_generation_failed: style profile could not be generated"
    )


def test_generated_bundle_persists_in_strict_store_and_fails_after_source_change(tmp_path):
    current = package(["SYNTHETIC_APPROVED_SENTINEL"] * 20)
    state = state_for(current)

    class Gateway:
        def current_state(self, user_id):
            assert user_id == "synthetic-owner"
            return state

    store = SingleTrustedUserStyleProfileStore(
        trusted_user_id="synthetic-owner",
        private_data_dir=tmp_path,
        consent_source=tmp_path / "consent.json",
        profile_source=tmp_path / "profile.json",
        evidence_source=tmp_path / "evidence.json",
        statistics_source=tmp_path / "statistics.json",
        index_source=tmp_path / "index.json",
        cache_source=tmp_path / "cache.json",
        strict_source_binding=True,
        source_gateway=Gateway(),
    )
    consent = store.grant_bound_consent(user_id="synthetic-owner", updated_at=NOW)
    bundle = generate(current, consent, state)
    store.create_profile(
        user_id="synthetic-owner",
        profile=bundle.profile,
        evidence=bundle.evidence,
        derived=bundle.derived,
    )
    assert (
        store.read_snapshot_state("synthetic-owner").status
        is UserStyleSnapshotReadStatus.AUTHORIZED
    )
    assert all(
        "SYNTHETIC_APPROVED_SENTINEL" not in path.read_text() for path in store.artifact_paths
    )
    state = state.model_copy(update={"revision": "v1-" + "d" * 64})
    assert (
        store.read_snapshot_state("synthetic-owner").status is UserStyleSnapshotReadStatus.CORRUPT
    )
