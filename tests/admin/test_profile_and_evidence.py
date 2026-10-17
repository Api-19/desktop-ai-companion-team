import os
import re
import stat
from datetime import UTC, datetime
from math import ceil
from statistics import median

import pytest
from test_bound_consent import grant
from test_fixed_import_source import private_source
from test_local_entry import call, settings
from test_mapping_confirmation import unlock
from test_retain_intent import form, post, ready
from test_style_decisions import decide, review

import companion.admin.__main__ as admin_main
import companion.admin.app as admin_module
from companion.admin.app import build_admin_app, load_admin_config
from companion.admin.phase1 import (
    phase1_paths,
    phase5_paths,
    validate_phase1_paths,
    validate_phase5_paths,
)
from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.style import (
    SingleTrustedUserStyleEvidenceStore,
    SingleTrustedUserStyleProfileStore,
    StyleFeatureKind,
    UserStyleSnapshotReadStatus,
)

OWNER = "synthetic-owner"
SHORT = 66
LONG_INDEX = 66
UNAPPROVED_INDEX = 67


def many_candidates():
    now = datetime(2026, 9, 24, tzinfo=UTC)
    texts = [f"合成表达<{index}>" for index in range(SHORT)] + ["长" * 161, "合成未批准"]
    count = len(texts)
    return SafeCandidatePackage(
        created_at=now,
        candidates=[
            SafeStyleCandidate(
                text=text,
                category=CandidateCategory.GREETING,
                scene_tags=[CandidateCategory.GREETING.value],
            )
            for text in texts
        ],
        audit=CandidateSelectionAudit(
            processed_at=now,
            input_count=count,
            owner_input_count=count,
            dropped_other_count=0,
            allowed_category_counts={"greeting": count},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=count,
            succeeded=True,
        ),
    )


def consented(tmp_path, approved_count=UNAPPROVED_INDEX):
    app, cookie, package, source, trusted, config = ready(tmp_path, many_candidates())
    for index in range(approved_count):
        app._approval_store.set_approved(OWNER, package, index=index, approved=True)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    consent = grant(app, cookie)
    assert consent is not None
    return app, cookie, package, source, trusted, config


def generate(app, cookie):
    status, page, values = form(app, cookie, "/admin/style/profile", "profile")
    assert status == 200 and values is not None, page
    return post(app, cookie, "/admin/style/profile", values)


def generated(tmp_path):
    app, cookie, package, source, trusted, config = consented(tmp_path)
    assert generate(app, cookie)[0] == 200
    return app, cookie, package, source, trusted, config


def evidence_page(app, cookie):
    status, _, page = call(app, "/admin/style/evidence", cookie=cookie)
    token = re.search(r'name="evidence_token" value="([^"]+)"', page)
    csrfs = dict(
        re.findall(
            r'action="/admin/style/(evidence-[a-z]+)">\s*<input type="hidden" '
            r'name="csrf" value="([^"]+)"',
            page,
        )
    )
    return status, page, (token.group(1) if token else None), csrfs


def evidence_post(app, cookie, action, fields):
    status, page, token, csrfs = evidence_page(app, cookie)
    assert status == 200 and token is not None, page
    values = {"csrf": csrfs[action], "evidence_token": token, **fields}
    return post(app, cookie, f"/admin/style/{action}", values)


def add(app, cookie, index, scenario="gratitude"):
    return evidence_post(app, cookie, "evidence-add", {"index": str(index), "scenario": scenario})


def publish(app, cookie):
    return evidence_post(app, cookie, "evidence-publish", {"confirm_local": "yes"})


def published(tmp_path):
    app, cookie, package, source, trusted, config = generated(tmp_path)
    assert add(app, cookie, 0, "gratitude")[0] == 200
    assert add(app, cookie, 1, "apology")[0] == 200
    assert publish(app, cookie)[0] == 200
    assert len(app._phase5_store.view_approved_evidence(OWNER)) == 2
    return app, cookie, package, source, trusted, config


def phase1_absent(app):
    return not any(path.exists() for path in app._phase1_store.artifact_paths)


def phase5_absent(app):
    return not any(path.exists() for path in app._phase5_store.artifact_paths)


def secrets_absent(app, page, trusted):
    state = app._source_gateway.current_state(OWNER)
    assert OWNER not in page and str(trusted.private_data_dir) not in page
    assert state is None or (
        state.package_digest not in page
        and state.batch_digest not in page
        and state.revision not in page
    )
    assert ".json" not in page and "Synthetic Owner" not in page


def test_phase5_paths_are_fixed_disjoint_and_created_only_when_requested(tmp_path):
    root, _ = private_source(tmp_path)
    paths = phase5_paths(root)
    assert tuple(path.name for path in paths) == (
        "evidence.json",
        "index.json",
        "cache.json",
        "version.json",
    )
    assert all(path.parent == root / "style-framework/phase5" for path in paths)
    assert not set(paths) & set(phase1_paths(root))
    validate_phase5_paths(root)
    assert not (root / "style-framework").exists()
    validate_phase5_paths(root, create=True)
    for folder in (root / "style-framework", root / "style-framework/phase5"):
        assert stat.S_IMODE(folder.lstat().st_mode) == 0o700
    assert not any(path.exists() for path in paths)
    paths[0].write_text("{}")
    paths[0].chmod(0o644)
    with pytest.raises(RuntimeError, match=r"^style management unavailable$"):
        validate_phase5_paths(root)
    paths[0].unlink()
    paths[1].symlink_to(root / "admin/config.toml")
    with pytest.raises(RuntimeError, match=r"^style management unavailable$"):
        validate_phase5_paths(root)
    paths[1].unlink()
    (root / "style-framework/phase5").chmod(0o755)
    with pytest.raises(RuntimeError, match=r"^style management unavailable$"):
        validate_phase5_paths(root)
    validate_phase1_paths(root)


def test_generate_profile_from_approved_only_readable_after_restart(tmp_path):
    app, cookie, package, _, trusted, config = consented(tmp_path)
    status, page, values = form(app, cookie, "/admin/style/profile", "profile")
    assert status == 200 and values is not None
    assert "非正文" in page and "67" in page and "20" in page
    secrets_absent(app, page, trusted)
    assert "合成表达" not in page
    status, _, page = post(app, cookie, "/admin/style/profile", values)
    assert status == 200, page
    snapshot = app._phase1_store.read_snapshot_state(OWNER).snapshot
    assert snapshot is not None and snapshot.consent == app._phase1_store.view_consent(OWNER)
    lengths = sorted(
        len("".join(c for c in package.candidates[i].text if not c.isspace()))
        for i in range(UNAPPROVED_INDEX)
    )
    values_by_kind = {feature.kind: feature.value for feature in snapshot.profile.features}
    assert values_by_kind == {
        StyleFeatureKind.MESSAGE_LENGTH_Q25: float(lengths[ceil(67 * 0.25) - 1]),
        StyleFeatureKind.MESSAGE_LENGTH_MEDIAN: float(median(lengths)),
        StyleFeatureKind.MESSAGE_LENGTH_Q75: float(lengths[ceil(67 * 0.75) - 1]),
    }
    assert all(f.provenance.evidence_count == 67 for f in snapshot.profile.features)
    assert "样本数 67" in page and "合成表达" not in page
    secrets_absent(app, page, trusted)
    assert phase5_absent(app)
    rebuilt = build_admin_app(trusted, config, port=12345)
    assert rebuilt._phase1_store.read_snapshot_state(OWNER).snapshot == snapshot
    assert post(app, cookie, "/admin/style/profile", values)[0] == 403


def test_profile_requires_twenty_approved_samples(tmp_path):
    app, cookie, _, _, _, _ = consented(tmp_path, approved_count=19)
    status, _, page = generate(app, cookie)
    assert status == 503 and "合成表达" not in page
    assert not any(path.exists() for path in app._phase1_store.derived_artifact_paths)
    assert app._phase1_store.view_consent(OWNER) is not None


def test_profile_without_consent_is_unavailable(tmp_path):
    app, cookie, package, _, _, _ = ready(tmp_path, many_candidates())
    for index in range(UNAPPROVED_INDEX):
        app._approval_store.set_approved(OWNER, package, index=index, approved=True)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    assert form(app, cookie, "/admin/style/profile", "profile")[0] == 503
    assert phase1_absent(app)


@pytest.mark.parametrize(
    "fault", ["stale_consent", "source_drift", "owner", "csrf", "confirm", "extra", "origin"]
)
def test_profile_rejects_stale_forged_or_unconfirmed_submission(tmp_path, fault):
    app, cookie, package, source, _, _ = consented(tmp_path)
    status, _, values = form(app, cookie, "/admin/style/profile", "profile")
    assert status == 200
    origin = "http://127.0.0.1:12345"
    if fault == "stale_consent":
        app._approval_store.set_approved(OWNER, package, index=0, approved=False)
    elif fault == "source_drift":
        source.write_bytes(source.read_bytes() + b" ")
    elif fault == "owner":
        app._sessions[cookie.split("=", 1)[1]].owner_id = "forged-owner"
    elif fault == "csrf":
        values = {**values, "csrf": "bad"}
    elif fault == "confirm":
        values = {**values, "confirm": "no"}
    elif fault == "extra":
        values = {**values, "owner_id": OWNER}
    else:
        origin = "http://elsewhere"
    assert post(app, cookie, "/admin/style/profile", values, origin=origin)[0] in {400, 403, 503}
    assert not any(path.exists() for path in app._phase1_store.derived_artifact_paths)


def test_stale_consent_blocks_profile_page(tmp_path):
    app, cookie, package, _, _, _ = consented(tmp_path)
    app._approval_store.set_approved(OWNER, package, index=0, approved=False)
    assert form(app, cookie, "/admin/style/profile", "profile")[0] == 503


def test_profile_persist_failure_invalidates_phase1_and_phase5_but_keeps_seal(
    tmp_path, monkeypatch
):
    app, cookie, package, _, _, _ = published(tmp_path)
    original = app._phase1_store.create_profile

    def drifting(**kwargs):
        original(**kwargs)
        app._approval_store.set_approved(OWNER, package, index=0, approved=False)

    monkeypatch.setattr(app._phase1_store, "create_profile", drifting)
    status, _, page = generate(app, cookie)
    assert status == 503 and "合成表达" not in page
    assert phase1_absent(app) and phase5_absent(app)
    assert app._source_seal.exists_for_owner(OWNER) is True


def test_profile_readback_mismatch_fails_closed(tmp_path, monkeypatch):
    app, cookie, _, _, _, _ = consented(tmp_path)
    monkeypatch.setattr(
        app._phase1_store,
        "read_snapshot_state",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic private detail")),
    )
    status, _, page = generate(app, cookie)
    assert status == 503 and "synthetic private detail" not in page
    assert phase1_absent(app) and app._source_seal.exists_for_owner(OWNER) is True


def test_regenerate_clears_old_phase5_before_writing(tmp_path, monkeypatch):
    app, cookie, _, _, _, _ = published(tmp_path)
    old = app._phase1_store.read_snapshot_state(OWNER).snapshot.profile.profile_version
    original = app._phase1_store.create_profile

    def checked(**kwargs):
        assert phase5_absent(app)
        return original(**kwargs)

    monkeypatch.setattr(app._phase1_store, "create_profile", checked)
    assert generate(app, cookie)[0] == 200
    new = app._phase1_store.read_snapshot_state(OWNER).snapshot.profile.profile_version
    assert new != old and phase5_absent(app)


def test_phase5_clear_failure_blocks_generation(tmp_path, monkeypatch):
    app, cookie, _, _, _, _ = published(tmp_path)
    before = app._phase1_store.read_snapshot_state(OWNER).snapshot
    monkeypatch.setattr(
        app._phase5_store,
        "delete_user_evidence",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic private detail")),
    )
    status, _, page = generate(app, cookie)
    assert status == 503 and "synthetic private detail" not in page
    assert app._phase1_store.read_snapshot_state(OWNER).snapshot == before


def test_phase5_page_requires_authorized_profile_and_explains_local_purpose(tmp_path):
    app, cookie, _, _, trusted, _ = consented(tmp_path)
    assert evidence_page(app, cookie)[0] == 503
    assert generate(app, cookie)[0] == 200
    status, page, token, csrfs = evidence_page(app, cookie)
    assert status == 200 and token is not None
    assert set(csrfs) == {"evidence-add", "evidence-publish"}
    assert "正文" in page and "本地长期保存" in page and "检索" in page
    assert "云端" in page and "费用" in page and "Phase 1" in page
    for scenario in ("gratitude", "apology", "sad_support", "generic_question"):
        assert f'value="{scenario}"' in page
    assert "合成表达" not in page
    secrets_absent(app, page, trusted)


def test_phase5_explicit_basket_preview_and_local_publish(tmp_path):
    app, cookie, package, _, trusted, config = generated(tmp_path)
    assert add(app, cookie, 0, "gratitude")[0] == 200
    assert add(app, cookie, 1, "apology")[0] == 200
    assert not any(path.exists() for path in phase5_paths(trusted.private_data_dir))
    status, page, _, csrfs = evidence_page(app, cookie)
    assert status == 200 and "evidence-remove" in csrfs
    assert "合成表达&lt;0&gt;" in page and "合成表达&lt;1&gt;" in page
    assert "合成表达<0>" not in page and "合成表达&lt;2&gt;" not in page
    assert "gratitude" in page and "apology" in page
    secrets_absent(app, page, trusted)
    status, _, page = evidence_post(app, cookie, "evidence-publish", {})
    assert status == 400 and not any(path.exists() for path in app._phase5_store.artifact_paths)
    status, _, page = publish(app, cookie)
    assert status == 200 and "2" in page
    records = app._phase5_store.view_approved_evidence(OWNER)
    assert [(r.candidate_index, r.scenario_id, r.content) for r in records] == [
        (0, "gratitude", package.candidates[0].text),
        (1, "apology", package.candidates[1].text),
    ]
    folder = trusted.private_data_dir / "style-framework/phase5"
    assert stat.S_IMODE(folder.lstat().st_mode) == 0o700
    for path in app._phase5_store.artifact_paths:
        info = path.lstat()
        assert stat.S_IMODE(info.st_mode) == 0o600 and info.st_uid == os.geteuid()
    assert evidence_page(app, cookie)[1].count("合成表达&lt;") == 0
    rebuilt = build_admin_app(trusted, config, port=12345)
    assert rebuilt._phase5_store.view_approved_evidence(OWNER) == records


@pytest.mark.parametrize(
    ("index", "scenario"),
    [
        (UNAPPROVED_INDEX, "gratitude"),
        (LONG_INDEX, "gratitude"),
        (0, "unknown_scenario"),
        (0, ""),
        (999, "gratitude"),
        ("01", "gratitude"),
        ("-1", "gratitude"),
        ("", "gratitude"),
    ],
)
def test_phase5_add_rejects_unapproved_long_or_invalid_choice(tmp_path, index, scenario):
    app, cookie, _, _, _, _ = generated(tmp_path)
    assert add(app, cookie, index, scenario)[0] == 400
    page = evidence_page(app, cookie)[1]
    assert "合成表达&lt;" not in page and "长长长" not in page and "合成未批准" not in page


def test_phase5_add_rejects_duplicates_forged_fields_and_overflow(tmp_path):
    app, cookie, _, _, _, _ = generated(tmp_path)
    assert add(app, cookie, 0)[0] == 200
    assert add(app, cookie, 0, "apology")[0] == 400
    _, _, token, csrfs = evidence_page(app, cookie)
    forged = {
        "csrf": csrfs["evidence-add"],
        "evidence_token": token,
        "index": "2",
        "scenario": "gratitude",
        "text": "forged",
    }
    assert post(app, cookie, "/admin/style/evidence-add", forged)[0] == 400
    for index in range(1, 64):
        assert add(app, cookie, index)[0] == 200
    assert add(app, cookie, 64)[0] == 400
    assert len(app._sessions[cookie.split("=", 1)[1]].evidence_basket) == 64


def test_phase5_remove_and_token_replay_csrf_origin_session(tmp_path):
    app, cookie, _, _, _, _ = generated(tmp_path)
    assert add(app, cookie, 0)[0] == 200
    _, _, token, csrfs = evidence_page(app, cookie)
    values = {"csrf": csrfs["evidence-add"], "evidence_token": token, "index": "1"}
    values["scenario"] = "apology"
    assert post(app, unlock(app), "/admin/style/evidence-add", values)[0] == 403
    assert post(app, cookie, "/admin/style/evidence-add", values, origin="http://x")[0] == 403
    bad = {**values, "csrf": csrfs["evidence-publish"]}
    assert post(app, cookie, "/admin/style/evidence-add", bad)[0] == 403
    assert post(app, cookie, "/admin/style/evidence-add", values)[0] == 200
    assert post(app, cookie, "/admin/style/evidence-add", values)[0] == 403
    assert evidence_post(app, cookie, "evidence-remove", {"index": "0"})[0] == 200
    page = evidence_page(app, cookie)[1]
    assert "合成表达&lt;0&gt;" not in page and "合成表达&lt;1&gt;" in page
    assert evidence_post(app, cookie, "evidence-remove", {"index": "0"})[0] == 400


def test_phase5_publish_requires_real_selection(tmp_path):
    app, cookie, _, _, _, _ = generated(tmp_path)
    assert publish(app, cookie)[0] == 400
    assert phase5_absent(app)


@pytest.mark.parametrize("change", ["approval", "regenerate", "consent", "source", "preview"])
def test_basket_is_dropped_when_binding_changes(tmp_path, change):
    app, cookie, package, source, _, _ = generated(tmp_path)
    assert add(app, cookie, 0)[0] == 200
    _, _, token, csrfs = evidence_page(app, cookie)
    stale = {"csrf": csrfs["evidence-publish"], "evidence_token": token, "confirm_local": "yes"}
    if change == "approval":
        app._approval_store.set_approved(OWNER, package, index=0, approved=False)
    elif change == "regenerate":
        assert generate(app, cookie)[0] == 200
    elif change == "consent":
        grant(app, cookie)
    elif change == "source":
        source.write_bytes(source.read_bytes() + b" ")
    else:
        assert call(app, "/admin/import/preview", cookie=cookie)[0] == 200
    assert post(app, cookie, "/admin/style/evidence-publish", stale)[0] in {403, 409, 503}
    assert phase5_absent(app)
    assert not app._sessions[cookie.split("=", 1)[1]].evidence_basket


@pytest.mark.parametrize("fault", ["drift", "readback", "raise"])
def test_publish_post_verification_failure_removes_phase5(tmp_path, monkeypatch, fault):
    app, cookie, package, _, _, _ = generated(tmp_path)
    assert add(app, cookie, 0)[0] == 200
    store = app._phase5_store
    original = store.publish_approved_evidence

    def publishing(**kwargs):
        original(**kwargs)
        if fault == "drift":
            app._approval_store.set_approved(OWNER, package, index=0, approved=True)
            app._approval_store.set_approved(OWNER, package, index=5, approved=False)
        elif fault == "raise":
            raise OSError("synthetic private detail")

    monkeypatch.setattr(store, "publish_approved_evidence", publishing)
    if fault == "readback":
        monkeypatch.setattr(store, "view_approved_evidence", lambda *_: ())
    status, _, page = publish(app, cookie)
    assert status == 503 and "synthetic private detail" not in page and "合成" not in page
    assert phase5_absent(app)


def test_publish_cleanup_failure_is_still_fixed_failure(tmp_path, monkeypatch):
    app, cookie, _, _, _, _ = generated(tmp_path)
    assert add(app, cookie, 0)[0] == 200
    monkeypatch.setattr(app._phase5_store, "view_approved_evidence", lambda *_: None)
    monkeypatch.setattr(
        app._phase5_store,
        "delete_user_evidence",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic private detail")),
    )
    status, _, page = publish(app, cookie)
    assert status == 503 and "synthetic private detail" not in page


@pytest.mark.parametrize("action", ["decide", "clear", "revoke", "delete", "delete-source"])
def test_unified_cleanup_removes_ten_and_seal_before_privacy(tmp_path, monkeypatch, action):
    app, cookie, _, _, _, _ = published(tmp_path)
    calls = []
    for name in ("set_approved", "delete_stale"):
        original = getattr(app._approval_store, name)

        def mutate(*args, _original=original, **kwargs):
            assert phase1_absent(app) and phase5_absent(app)
            assert app._source_seal.exists_for_owner(OWNER) is False
            calls.append(True)
            return _original(*args, **kwargs)

        monkeypatch.setattr(app._approval_store, name, mutate)
    if action == "decide":
        _, _, tokens = review(app, cookie)
        result = decide(app, cookie, *tokens, index="3", decision="approve")
    else:
        token_action = "delete_source" if action == "delete-source" else action
        status, page, values = form(app, cookie, f"/admin/style/{action}", token_action)
        assert status == 200 and "Phase 5" in page
        result = post(app, cookie, f"/admin/style/{action}", values)
    assert result[0] == (303 if action == "decide" else 200)
    assert calls == ([True] if action in {"decide", "clear"} else [])
    assert phase1_absent(app) and phase5_absent(app)
    assert app._source_seal.exists_for_owner(OWNER) is False


@pytest.mark.parametrize("action", ["revoke", "delete", "delete-source"])
def test_cleanup_works_after_source_disappears(tmp_path, action):
    app, cookie, _, source, _, _ = published(tmp_path)
    source.unlink()
    token_action = "delete_source" if action == "delete-source" else action
    _, _, values = form(app, cookie, f"/admin/style/{action}", token_action)
    assert post(app, cookie, f"/admin/style/{action}", values)[0] == 200
    assert phase1_absent(app) and phase5_absent(app)
    assert app._source_seal.exists_for_owner(OWNER) is False


@pytest.mark.parametrize("action", ["decide", "clear", "revoke", "delete", "delete-source"])
def test_phase5_cleanup_failure_leaves_phase1_invalid_and_privacy_untouched(
    tmp_path, monkeypatch, action
):
    app, cookie, package, _, _, _ = published(tmp_path)
    witness = app._approval_store.approval_witness(OWNER, package)
    monkeypatch.setattr(
        app._phase5_store,
        "delete_user_evidence",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic private detail")),
    )
    if action == "decide":
        _, _, tokens = review(app, cookie)
        result = decide(app, cookie, *tokens, index="3", decision="revoke")
    else:
        token_action = "delete_source" if action == "delete-source" else action
        _, _, values = form(app, cookie, f"/admin/style/{action}", token_action)
        result = post(app, cookie, f"/admin/style/{action}", values)
    assert result[0] == 503 and "synthetic private detail" not in result[2]
    assert phase1_absent(app)
    assert (
        app._phase1_store.read_snapshot_state(OWNER).status
        is not UserStyleSnapshotReadStatus.AUTHORIZED
    )
    assert app._approval_store.approval_witness(OWNER, package) == witness
    assert app._source_seal.exists_for_owner(OWNER) is True


def test_consent_regrant_clears_phase5_and_failure_does_not_grant(tmp_path, monkeypatch):
    app, cookie, _, _, _, _ = published(tmp_path)
    before = app._phase1_store.view_consent(OWNER)
    monkeypatch.setattr(
        app._phase5_store,
        "delete_user_evidence",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic private detail")),
    )
    _, _, values = form(app, cookie, "/admin/style/consent", "consent")
    status, _, page = post(app, cookie, "/admin/style/consent", values)
    assert status == 503 and "synthetic private detail" not in page
    assert app._phase1_store.view_consent(OWNER) == before
    monkeypatch.undo()
    after = grant(app, cookie)
    assert after.authorization_version != before.authorization_version
    assert phase5_absent(app)


def test_home_links_new_pages(tmp_path):
    app, cookie, _, _, _, _ = consented(tmp_path)
    page = call(app, "/admin/home", cookie=cookie)[2]
    assert "/admin/style/profile" in page and "/admin/style/evidence" in page


class _FakeServer:
    server_port = 12345

    def __init__(self, captured):
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def set_app(self, app):
        self._captured.append(app)

    def serve_forever(self):
        return None


def test_main_assembles_two_strict_stores_with_same_gateway(tmp_path, monkeypatch):
    root, _ = private_source(tmp_path)
    trusted = settings(root)
    captured = []
    monkeypatch.setattr(admin_main, "load_settings", lambda _path: trusted)
    monkeypatch.setattr(admin_main, "make_server", lambda *_a, **_k: _FakeServer(captured))
    assert admin_main.main() == 0
    app = captured[0]
    assert isinstance(app._phase1_store, SingleTrustedUserStyleProfileStore)
    assert isinstance(app._phase5_store, SingleTrustedUserStyleEvidenceStore)
    assert app._phase1_store._strict_source_binding is True
    assert app._phase5_store._strict_source_binding is True
    assert app._phase1_store._source_gateway is app._source_gateway
    assert app._phase5_store._source_gateway is app._source_gateway
    assert app._phase5_store.artifact_paths == phase5_paths(root)
    assert not (root / "style-framework").exists()


def test_main_without_import_format_creates_no_style_files(tmp_path, monkeypatch):
    from test_local_entry import config

    root = tmp_path / "private"
    config(root)
    trusted = settings(root)
    captured = []
    monkeypatch.setattr(admin_main, "load_settings", lambda _path: trusted)
    monkeypatch.setattr(admin_main, "make_server", lambda *_a, **_k: _FakeServer(captured))
    assert admin_main.main() == 0
    app = captured[0]
    assert app._phase1_store is None and app._phase5_store is None
    assert not (root / "style-framework").exists()
    assert load_admin_config(trusted).import_format is None


def test_main_phase5_assembly_failure_is_fixed(tmp_path, monkeypatch, capsys):
    root, _ = private_source(tmp_path)
    trusted = settings(root)
    captured = []
    monkeypatch.setattr(admin_main, "load_settings", lambda _path: trusted)
    monkeypatch.setattr(admin_main, "make_server", lambda *_a, **_k: _FakeServer(captured))
    monkeypatch.setattr(
        admin_module,
        "SingleTrustedUserStyleEvidenceStore",
        lambda **_: (_ for _ in ()).throw(RuntimeError("synthetic private detail")),
    )
    assert admin_main.main() == 2
    assert capsys.readouterr().out == "Admin unavailable\n" and captured == []
