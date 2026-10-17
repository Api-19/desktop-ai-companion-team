from datetime import UTC, datetime

import pytest
from test_local_entry import call
from test_mapping_confirmation import unlock
from test_retain_intent import approved, form, post, ready
from test_style_decisions import decide, review

import companion.admin.__main__ as admin_main
import companion.admin.app as admin_module
from companion.admin.app import build_admin_app
from companion.privacy.style_candidate_approval import ApprovalWitness
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
    UserStyleProfileStatus,
    UserStyleSnapshotReadStatus,
)

OWNER = "synthetic-owner"
NOW = datetime(2026, 9, 28, tzinfo=UTC)


def sealed_ready(tmp_path):
    app, cookie, package, source, settings, config = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    return app, cookie, package, source, settings, config


def grant(app, cookie):
    status, page, values = form(app, cookie, "/admin/style/consent", "consent")
    assert status == 200 and values is not None
    assert "非正文" in page and "长期本地保存" in page and "Phase 5" in page
    assert "云端" in page and "撤权" in page and "删除" in page
    assert OWNER not in page and "synthetic one" not in page
    assert post(app, cookie, "/admin/style/consent", values)[0] == 200
    return app._phase1_store.view_consent(OWNER)


def create_synthetic_profile(store, consent):
    shared = dict(
        user_id=OWNER,
        profile_version="synthetic-v1",
        authorization_version=consent.authorization_version,
    )
    kind = StyleFeatureKind.MESSAGE_LENGTH_MEDIAN
    store.create_profile(
        user_id=OWNER,
        profile=UserStyleProfile(
            **shared,
            features=(
                UserStyleFeature(
                    kind=kind,
                    value=9.0,
                    confidence=0.9,
                    provenance=StyleFeatureProvenance(
                        source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
                        evidence_count=8,
                        observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
                        source_version="synthetic-safe-v1",
                        updated_at=NOW,
                        confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
                    ),
                ),
            ),
            created_at=NOW,
            updated_at=NOW,
        ),
        evidence=UserStyleEvidenceLedger(**shared, feature_evidence={kind: 8}, updated_at=NOW),
        derived=UserStyleDerivedArtifacts(
            **shared,
            statistics_version="s-v1",
            index_version="i-v1",
            cache_version="c-v1",
            updated_at=NOW,
        ),
    )


def test_product_gateway_and_bound_consent_across_rebuild(tmp_path):
    app, cookie, package, _, settings, config = sealed_ready(tmp_path)
    assert isinstance(app._phase1_store, SingleTrustedUserStyleProfileStore)
    state = app._source_gateway.current_state(OWNER)
    witness = app._approval_store.approval_witness(OWNER, package)
    assert type(state) is SourceApprovalState
    assert state.package_digest == witness.package_digest and state.approved_indices == (0,)
    assert state.revision == witness.revision
    assert state.batch_digest == app._sessions[cookie.split("=", 1)[1]].confirmed_digest
    assert not any(path.exists() for path in app._phase1_store.artifact_paths)
    consent = grant(app, cookie)
    assert consent.source_binding == state and consent.status is UserStyleProfileStatus.AUTHORIZED
    assert not any(path.exists() for path in app._phase1_store.derived_artifact_paths)
    rebuilt = build_admin_app(settings, config, port=12345)
    assert rebuilt._source_gateway.current_state(OWNER) == state
    assert rebuilt._phase1_store.view_consent(OWNER) == consent
    assert tuple(path.name for path in rebuilt._phase1_store.artifact_paths) == (
        "consent.json",
        "profile.json",
        "evidence.json",
        "statistics.json",
        "index.json",
        "cache.json",
    )
    assert all(
        path.parent == settings.private_data_dir / "style-framework/phase1"
        for path in rebuilt._phase1_store.artifact_paths
    )
    create_synthetic_profile(rebuilt._phase1_store, consent)
    assert (
        rebuilt._phase1_store.read_snapshot_state(OWNER).status
        is UserStyleSnapshotReadStatus.AUTHORIZED
    )
    rebuilt._approval_store.set_approved(OWNER, package, index=0, approved=True)
    assert (
        rebuilt._phase1_store.read_snapshot_state(OWNER).status
        is not UserStyleSnapshotReadStatus.AUTHORIZED
    )


def test_gateway_wrong_owner_has_zero_downstream_calls(tmp_path, monkeypatch):
    app, _, _, _, _, _ = sealed_ready(tmp_path)
    calls = []
    monkeypatch.setattr(app._source_seal, "read", lambda *_: calls.append("seal"))
    monkeypatch.setattr(app._approval_store, "approval_witness", lambda *_: calls.append("privacy"))
    assert app._source_gateway.current_state("other-owner") is None
    assert app._source_gateway.current_state(True) is None
    assert calls == []


@pytest.mark.parametrize(
    "fault", ["none", "empty", "bool", "out_of_range", "revision", "digest", "wrong_type"]
)
def test_gateway_rejects_bad_witness(tmp_path, monkeypatch, fault):
    app, _, package, _, _, _ = sealed_ready(tmp_path)
    witness = app._approval_store.approval_witness(OWNER, package)
    changes = {
        "empty": {"approved_indices": ()},
        "bool": {"approved_indices": (True,)},
        "out_of_range": {"approved_indices": (len(package.candidates),)},
        "revision": {"revision": "invalid"},
        "digest": {"package_digest": "f" * 64},
    }
    value = (
        None
        if fault == "none"
        else object()
        if fault == "wrong_type"
        else ApprovalWitness(**{**vars(witness), **changes[fault]})
    )
    monkeypatch.setattr(app._approval_store, "approval_witness", lambda *_: value)
    with pytest.raises(Exception, match="unavailable"):
        app._source_gateway.current_state(OWNER)


@pytest.mark.parametrize(
    "fault", ["unsealed", "unapproved", "source_drift", "missing_source", "no_mapping"]
)
def test_consent_requires_current_sealed_source_and_approved_session(tmp_path, fault):
    app, cookie, package, source, _, _ = sealed_ready(tmp_path)
    if fault == "unsealed":
        app._source_seal.delete(OWNER)
    elif fault == "unapproved":
        app._approval_store.set_approved(OWNER, package, index=0, approved=False)
    elif fault == "source_drift":
        source.write_bytes(source.read_bytes() + b" ")
    elif fault == "missing_source":
        source.unlink()
    else:
        app._clear_confirmation(app._sessions[cookie.split("=", 1)[1]])
    status, page, values = form(app, cookie, "/admin/style/consent", "consent")
    assert status == 503 and values is None and OWNER not in page
    assert app._phase1_store.view_consent(OWNER) is None


def test_consent_explicit_csrf_session_origin_and_replay(tmp_path):
    app, cookie, _, _, _, _ = sealed_ready(tmp_path)
    _, _, values = form(app, cookie, "/admin/style/consent", "consent")
    assert post(app, unlock(app), "/admin/style/consent", values)[0] == 403
    assert post(app, cookie, "/admin/style/consent", values, origin="http://elsewhere")[0] == 403
    assert post(app, cookie, "/admin/style/consent", {**values, "csrf": "bad"})[0] == 403
    assert post(app, cookie, "/admin/style/consent", {**values, "confirm": "no"})[0] == 400
    assert app._phase1_store.view_consent(OWNER) is None
    _, _, values = form(app, cookie, "/admin/style/consent", "consent")
    assert post(app, cookie, "/admin/style/consent", {**values, "owner_id": "forged"})[0] == 400
    _, _, values = form(app, cookie, "/admin/style/consent", "consent")
    status, _, page = post(app, cookie, "/admin/style/consent", values)
    assert status == 200 and "同意已记录。画像尚未生成" in page
    assert post(app, cookie, "/admin/style/consent", values)[0] == 403


@pytest.mark.parametrize("action", ["revoke", "delete", "delete-source"])
def test_source_missing_does_not_block_cascade_delete(tmp_path, action):
    app, cookie, package, source, _, _ = sealed_ready(tmp_path)
    consent = grant(app, cookie)
    create_synthetic_profile(app._phase1_store, consent)
    source.unlink()
    assert review(app, cookie)[0] == 503
    token_action = "delete_source" if action == "delete-source" else action
    status, page, values = form(app, cookie, f"/admin/style/{action}", token_action)
    assert status == 200 and values is not None and "Phase 5" in page
    assert post(app, cookie, f"/admin/style/{action}", values)[0] == 200
    assert not any(path.exists() for path in app._phase1_store.artifact_paths)
    assert app._source_seal.exists_for_owner(OWNER) is False
    assert app._approval_store.approved_indices(OWNER, package) == (0,)
    assert (
        app._phase1_store.read_snapshot_state(OWNER).status
        is not UserStyleSnapshotReadStatus.AUTHORIZED
    )


@pytest.mark.parametrize("action", ["decide", "clear"])
def test_privacy_mutation_clears_bound_style_first(tmp_path, action, monkeypatch):
    app, cookie, _, _, _, _ = sealed_ready(tmp_path)
    create_synthetic_profile(app._phase1_store, grant(app, cookie))
    name = "set_approved" if action == "decide" else "delete_stale"
    original = getattr(app._approval_store, name)
    calls = []

    def mutate(*args, **kwargs):
        assert not any(path.exists() for path in app._phase1_store.artifact_paths)
        assert app._source_seal.exists_for_owner(OWNER) is False
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(app._approval_store, name, mutate)
    if action == "decide":
        _, _, tokens = review(app, cookie)
        result = decide(app, cookie, *tokens, index="0", decision="approve")
    else:
        _, _, values = form(app, cookie, "/admin/style/clear", "clear")
        result = post(app, cookie, "/admin/style/clear", values)
    assert result[0] == (303 if action == "decide" else 200) and calls == [True]
    assert (
        app._phase1_store.read_snapshot_state(OWNER).status
        is not UserStyleSnapshotReadStatus.AUTHORIZED
    )


@pytest.mark.parametrize("action", ["decide", "clear", "delete-source", "revoke", "delete"])
def test_cleanup_failure_stops_privacy_and_source_delete(tmp_path, action, monkeypatch):
    app, cookie, package, _, _, _ = sealed_ready(tmp_path)
    grant(app, cookie)
    witness = app._approval_store.approval_witness(OWNER, package)
    monkeypatch.setattr(
        app._phase1_store,
        "delete_user_style",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic private detail")),
    )
    if action == "decide":
        _, _, tokens = review(app, cookie)
        result = decide(app, cookie, *tokens, index="0", decision="approve")
    else:
        token_action = "delete_source" if action == "delete-source" else action
        _, _, values = form(app, cookie, f"/admin/style/{action}", token_action)
        result = post(app, cookie, f"/admin/style/{action}", values)
    assert result[0] == 503 and "synthetic private detail" not in result[2]
    assert app._approval_store.approval_witness(OWNER, package) == witness
    assert app._source_seal.exists_for_owner(OWNER) is True


@pytest.mark.parametrize("change", ["source", "revision", "expired", "owner"])
def test_consent_rechecks_at_submit_and_never_grants_stale_page(tmp_path, change):
    app, cookie, package, source, _, _ = sealed_ready(tmp_path)
    _, _, values = form(app, cookie, "/admin/style/consent", "consent")
    if change == "source":
        source.write_bytes(source.read_bytes() + b" ")
    elif change == "revision":
        approved(app, package)
    elif change == "expired":
        app._sessions[cookie.split("=", 1)[1]].expires = 0
    else:
        app._sessions[cookie.split("=", 1)[1]].owner_id = "forged-owner"
    assert post(app, cookie, "/admin/style/consent", values)[0] in {403, 503}
    assert app._phase1_store.view_consent(OWNER) is None


def test_consent_rejects_gateway_binding_inconsistent_with_current_seal(tmp_path, monkeypatch):
    app, cookie, _, _, _, _ = sealed_ready(tmp_path)
    state = app._source_gateway.current_state(OWNER)
    monkeypatch.setattr(
        app._source_gateway,
        "current_state",
        lambda *_: state.model_copy(update={"batch_digest": "f" * 64}),
    )
    assert form(app, cookie, "/admin/style/consent", "consent")[0] == 503
    assert app._phase1_store.view_consent(OWNER) is None


def test_revision_change_during_grant_is_rolled_back(tmp_path, monkeypatch):
    app, cookie, package, _, _, _ = sealed_ready(tmp_path)
    _, _, values = form(app, cookie, "/admin/style/consent", "consent")
    original = app._phase1_store.grant_bound_consent

    def changing(**kwargs):
        result = original(**kwargs)
        approved(app, package)
        return result

    monkeypatch.setattr(app._phase1_store, "grant_bound_consent", changing)
    status, _, page = post(app, cookie, "/admin/style/consent", values)
    assert status == 503 and OWNER not in page and "synthetic one" not in page
    assert app._phase1_store.view_consent(OWNER) is None


@pytest.mark.parametrize("action", ["revoke", "delete"])
def test_management_explicit_session_csrf_origin_and_replay(tmp_path, action):
    app, cookie, _, _, _, _ = sealed_ready(tmp_path)
    grant(app, cookie)
    path = f"/admin/style/{action}"
    assert call(app, path)[0] == 403
    _, _, values = form(app, cookie, path, action)
    assert post(app, unlock(app), path, values)[0] == 403
    assert post(app, cookie, path, values, origin="http://elsewhere")[0] == 403
    assert post(app, cookie, path, {**values, "csrf": "wrong"})[0] == 403
    assert post(app, cookie, path, {**values, "confirm": "no"})[0] == 400
    assert app._phase1_store.view_consent(OWNER) is not None
    _, _, values = form(app, cookie, path, action)
    assert post(app, cookie, path, {**values, "owner_id": "forged"})[0] == 400
    _, _, values = form(app, cookie, path, action)
    assert post(app, cookie, path, values)[0] == 200
    assert post(app, cookie, path, values)[0] == 403


def test_real_partial_phase1_cleanup_failure_leaves_revoked_and_skips_privacy(
    tmp_path, monkeypatch
):
    app, cookie, package, _, _, _ = sealed_ready(tmp_path)
    consent = grant(app, cookie)
    create_synthetic_profile(app._phase1_store, consent)
    witness = app._approval_store.approval_witness(OWNER, package)
    original = app._phase1_store._delete_paths

    def fail_derived(paths):
        if paths == app._phase1_store.derived_artifact_paths:
            raise OSError("synthetic cleanup failure")
        return original(paths)

    monkeypatch.setattr(app._phase1_store, "_delete_paths", fail_derived)
    _, _, tokens = review(app, cookie)
    assert decide(app, cookie, *tokens, index="0", decision="revoke")[0] == 503
    assert app._phase1_store.view_consent(OWNER).status is UserStyleProfileStatus.REVOKED
    assert (
        app._phase1_store.read_snapshot_state(OWNER).status
        is not UserStyleSnapshotReadStatus.AUTHORIZED
    )
    assert app._approval_store.approval_witness(OWNER, package) == witness
    assert app._source_seal.exists_for_owner(OWNER) is True


def test_source_delete_failure_after_phase1_cleanup_does_not_mutate_privacy(tmp_path, monkeypatch):
    app, cookie, package, _, _, _ = sealed_ready(tmp_path)
    create_synthetic_profile(app._phase1_store, grant(app, cookie))
    witness = app._approval_store.approval_witness(OWNER, package)
    monkeypatch.setattr(
        app._source_seal,
        "delete",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic cleanup failure")),
    )
    _, _, tokens = review(app, cookie)
    assert decide(app, cookie, *tokens, index="0", decision="approve")[0] == 503
    assert not any(path.exists() for path in app._phase1_store.artifact_paths)
    assert app._approval_store.approval_witness(OWNER, package) == witness
    assert app._source_seal.exists_for_owner(OWNER) is True


def test_product_construction_does_not_create_phase1_directory_or_consent(tmp_path):
    app, _, _, _, settings, config = ready(tmp_path)
    rebuilt = build_admin_app(settings, config, port=12345)
    assert not (settings.private_data_dir / "style-framework").exists()
    assert not any(path.exists() for path in rebuilt._phase1_store.artifact_paths)
    assert app._phase1_store.view_consent(OWNER) is None


def test_source_changes_during_gateway_witness_read_fail_closed(tmp_path, monkeypatch):
    app, _, _, source, _, _ = sealed_ready(tmp_path)
    original = app._approval_store.approval_witness

    def changing(*args):
        result = original(*args)
        source.write_bytes(source.read_bytes() + b" ")
        return result

    monkeypatch.setattr(app._approval_store, "approval_witness", changing)
    with pytest.raises(RuntimeError, match=r"^style source unavailable$"):
        app._source_gateway.current_state(OWNER)


def test_phase1_private_directory_failure_does_not_grant(tmp_path):
    app, cookie, _, _, settings, _ = sealed_ready(tmp_path)
    _, _, values = form(app, cookie, "/admin/style/consent", "consent")
    (settings.private_data_dir / "style-framework").mkdir(mode=0o755)
    assert post(app, cookie, "/admin/style/consent", values)[0] == 503
    assert app._phase1_store.view_consent(OWNER) is None


def test_main_phase1_assembly_failure_is_fixed_without_listener(tmp_path, monkeypatch, capsys):
    _, _, _, _, settings, _ = ready(tmp_path)

    class FakeServer:
        server_port = 12345

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def set_app(self, _app):
            pytest.fail("failed assembly must not start the app")

    monkeypatch.setattr(admin_main, "load_settings", lambda _: settings)
    monkeypatch.setattr(admin_main, "make_server", lambda *_args, **_kwargs: FakeServer())
    monkeypatch.setattr(
        admin_module,
        "SingleTrustedUserStyleProfileStore",
        lambda **_: (_ for _ in ()).throw(RuntimeError("synthetic private detail")),
    )
    assert admin_main.main() == 2
    assert capsys.readouterr().out == "Admin unavailable\n"
