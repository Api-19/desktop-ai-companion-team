import os
import re

import pytest
from test_candidate_review_handoff import candidate_package
from test_fixed_import_source import private_source
from test_local_entry import call, settings
from test_mapping_confirmation import confirm, preview_form, unlock
from test_style_decisions import review, two_candidates

import companion.admin.__main__ as admin_main
from companion.admin.app import build_admin_app, load_admin_config
from companion.admin.source_seal import StyleSourceSealError, StyleSourceSealStore

ORIGIN = "http://127.0.0.1:12345"


def ready(tmp_path, package=None):
    root, source = private_source(tmp_path)
    trusted = settings(root)
    config = load_admin_config(trusted)
    package = package if package is not None else two_candidates()
    app = build_admin_app(
        trusted, config, port=12345, candidate_reader=lambda _owner, _records: package
    )
    cookie = unlock(app)
    assert confirm(app, cookie, preview_form(app, cookie))[0] == 200
    assert review(app, cookie)[0] == 200
    return app, cookie, package, source, trusted, config


def form(app, cookie, path, action):
    status, _, page = call(app, path, cookie=cookie)
    token = re.search(rf'name="{action}_token" value="([^"]+)"', page)
    csrf = re.search(r'name="csrf" value="([^"]+)"', page)
    values = (
        {"csrf": csrf.group(1), f"{action}_token": token.group(1), "confirm": "yes"}
        if token and csrf
        else None
    )
    return status, page, values


def post(app, cookie, path, values, *, origin=ORIGIN, host="127.0.0.1:12345"):
    return call(app, path, method="POST", cookie=cookie, origin=origin, host=host, form=values)


def approved(app, package):
    app._approval_store.set_approved("synthetic-owner", package, index=0, approved=True)


def test_unlock_redirects_to_owner_home(tmp_path):
    root, _ = private_source(tmp_path)
    trusted = settings(root)
    app = build_admin_app(trusted, load_admin_config(trusted), port=12345)
    code = app.issue_start_code()
    _, headers, page = call(app, "/admin")
    pre_cookie = headers["Set-Cookie"].split(";", 1)[0]
    csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
    status, headers, _ = call(
        app,
        "/admin/unlock",
        method="POST",
        cookie=pre_cookie,
        origin=ORIGIN,
        form={"code": code, "csrf": csrf},
    )
    assert status == 303 and headers["Location"] == "/admin/home"


def test_product_seal_intent_and_explicit_delete(tmp_path):
    app, cookie, package, _, trusted, config = ready(tmp_path)
    approved(app, package)
    assert isinstance(app._source_seal, StyleSourceSealStore)
    status, page, values = form(app, cookie, "/admin/style/retain", "retain")
    assert status == 200 and values is not None
    assert "完整安全候选包正文" in page
    assert "非正文风格特征" in page and "逐条批准" in page and "删除" in page
    assert "synthetic one" not in page and "synthetic-owner" not in page
    status, _, page = post(app, cookie, "/admin/style/retain", values)
    assert status == 200 and "Phase 1" in page and "synthetic one" not in page
    sealed = StyleSourceSealStore(trusted, config).read("synthetic-owner")
    assert sealed is not None and sealed.package == package
    assert sealed.batch_digest == app._sessions[cookie.split("=", 1)[1]].confirmed_digest
    assert post(app, cookie, "/admin/style/retain", values)[0] == 403

    status, page, delete_values = form(app, cookie, "/admin/style/delete-source", "delete_source")
    assert status == 200 and delete_values is not None and "synthetic one" not in page
    assert post(app, cookie, "/admin/style/delete-source", delete_values)[0] == 200
    assert StyleSourceSealStore(trusted, config).read("synthetic-owner") is None
    assert app._approval_store.approved_indices("synthetic-owner", package) == (0,)


@pytest.mark.parametrize("package", [two_candidates(), candidate_package(empty=True)])
def test_unapproved_or_zero_candidates_cannot_seal(tmp_path, package):
    app, cookie, _, _, trusted, _ = ready(tmp_path, package)
    assert form(app, cookie, "/admin/style/retain", "retain")[0] == 503
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()


def test_retain_requires_explicit_confirmation_and_rejects_form_identity(tmp_path):
    app, cookie, package, _, trusted, _ = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", {**values, "confirm": "no"})[0] == 400
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", {**values, "owner_id": "forged"})[0] == 400
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()


def test_cross_session_origin_csrf_replay_and_expiry_fail_closed(tmp_path):
    app, cookie, package, _, trusted, _ = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    other = unlock(app)
    assert post(app, other, "/admin/style/retain", values)[0] == 403
    assert post(app, cookie, "/admin/style/retain", values, origin="http://elsewhere")[0] == 403
    assert post(app, cookie, "/admin/style/retain", {**values, "csrf": "wrong"})[0] == 403
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    assert post(app, cookie, "/admin/style/retain", values)[0] == 403
    _, _, values = form(app, cookie, "/admin/style/delete-source", "delete_source")
    app._sessions[cookie.split("=", 1)[1]].expires = 0
    assert post(app, cookie, "/admin/style/delete-source", values)[0] == 403
    assert (trusted.private_data_dir / "admin/style-source/current.json").exists()


def test_source_drift_and_wrong_owner_block_retain_and_delete(tmp_path):
    app, cookie, package, source, trusted, _ = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    source.write_bytes(source.read_bytes() + b" ")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 503
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()

    second = tmp_path / "second"
    second.mkdir()
    app, cookie, package, _, trusted, _ = ready(second)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    app._sessions[cookie.split("=", 1)[1]].owner_id = "forged-owner"
    assert post(app, cookie, "/admin/style/retain", values)[0] == 403
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()


def test_approval_revision_change_blocks_seal(tmp_path):
    app, cookie, package, _, trusted, _ = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    approved(app, package)
    assert post(app, cookie, "/admin/style/retain", values)[0] == 503
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()


def test_revision_change_during_seal_never_claims_success(tmp_path, monkeypatch):
    app, cookie, package, _, _, _ = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    original = app._source_seal.seal

    def change_revision(owner, digest, candidate):
        result = original(owner, digest, candidate)
        approved(app, package)
        return result

    monkeypatch.setattr(app._source_seal, "seal", change_revision)
    status, _, page = post(app, cookie, "/admin/style/retain", values)
    assert status == 503 and "synthetic one" not in page and "synthetic-owner" not in page


def test_store_and_witness_failure_are_fixed_and_no_seal(tmp_path, monkeypatch):
    app, cookie, package, _, trusted, _ = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    monkeypatch.setattr(
        app._source_seal,
        "seal",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic private detail")),
    )
    status, _, page = post(app, cookie, "/admin/style/retain", values)
    assert status == 503 and "synthetic private detail" not in page
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()
    monkeypatch.setattr(
        app._approval_store,
        "approval_witness",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic private detail")),
    )
    status, page, _ = form(app, cookie, "/admin/style/retain", "retain")
    assert status == 503 and "synthetic private detail" not in page


@pytest.mark.parametrize("source_change", ["missing", "drift"])
def test_delete_survives_source_loss_and_invalidated_mapping(tmp_path, source_change):
    app, cookie, package, source, trusted, config = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    _, _, delete_values = form(app, cookie, "/admin/style/delete-source", "delete_source")
    assert (
        post(app, cookie, "/admin/style/delete-source", {**delete_values, "confirm": "no"})[0]
        == 400
    )
    _, _, delete_values = form(app, cookie, "/admin/style/delete-source", "delete_source")
    if source_change == "missing":
        source.unlink()
    else:
        source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(StyleSourceSealError):
        StyleSourceSealStore(trusted, config).read("synthetic-owner")
    assert review(app, cookie)[0] == 503
    status, _, home = call(app, "/admin/home", cookie=cookie)
    assert status == 200 and "/admin/style/delete-source" in home
    status, page, delete_values = form(app, cookie, "/admin/style/delete-source", "delete_source")
    assert status == 200 and delete_values is not None and "synthetic one" not in page
    assert post(app, cookie, "/admin/style/delete-source", delete_values)[0] == 200
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()
    assert post(app, cookie, "/admin/style/delete-source", delete_values)[0] == 403


def test_get_pages_recheck_source_and_frozen_package(tmp_path):
    app, cookie, package, source, trusted, _ = ready(tmp_path)
    approved(app, package)
    source.write_bytes(source.read_bytes() + b" ")
    assert form(app, cookie, "/admin/style/retain", "retain")[0] == 503
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()

    other = tmp_path / "other"
    other.mkdir()
    app, cookie, package, source, trusted, _ = ready(other)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    source.write_bytes(source.read_bytes() + b" ")
    assert form(app, cookie, "/admin/style/delete-source", "delete_source")[0] == 200
    assert (trusted.private_data_dir / "admin/style-source/current.json").exists()


def test_delete_origin_csrf_session_and_replay_fail_closed(tmp_path):
    app, cookie, package, _, trusted, config = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    _, _, delete_values = form(app, cookie, "/admin/style/delete-source", "delete_source")
    other = unlock(app)
    assert post(app, other, "/admin/style/delete-source", delete_values)[0] == 403
    assert (
        post(app, cookie, "/admin/style/delete-source", delete_values, origin="http://elsewhere")[0]
        == 403
    )
    assert (
        post(app, cookie, "/admin/style/delete-source", {**delete_values, "csrf": "bad"})[0] == 403
    )
    assert StyleSourceSealStore(trusted, config).read("synthetic-owner") is not None
    assert post(app, cookie, "/admin/style/delete-source", delete_values)[0] == 200
    assert post(app, cookie, "/admin/style/delete-source", delete_values)[0] == 403


@pytest.mark.parametrize("fault", ["bad_json", "permissions", "symlink", "hardlink"])
def test_delete_rejects_unsafe_seal_without_source(tmp_path, fault):
    app, cookie, package, source, trusted, _ = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    current = trusted.private_data_dir / "admin/style-source/current.json"
    source.unlink()
    if fault == "bad_json":
        current.write_bytes(b"not-json")
    elif fault == "permissions":
        current.chmod(0o644)
    elif fault == "symlink":
        target = tmp_path / "synthetic-target.json"
        current.rename(target)
        current.symlink_to(target)
    else:
        os.link(current, tmp_path / "synthetic-hardlink.json")
    status, page, values = form(app, cookie, "/admin/style/delete-source", "delete_source")
    assert status == 503 and values is None and "synthetic one" not in page
    assert current.is_symlink() if fault == "symlink" else current.exists()


def test_delete_home_requires_live_owner_session(tmp_path):
    app, cookie, package, _, trusted, _ = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    assert call(app, "/admin/home")[0] == 403
    session = app._sessions[cookie.split("=", 1)[1]]
    session.owner_id = "forged-owner"
    assert call(app, "/admin/home", cookie=cookie)[0] == 403
    assert form(app, cookie, "/admin/style/delete-source", "delete_source")[0] == 403
    assert (trusted.private_data_dir / "admin/style-source/current.json").exists()


def test_delete_store_failure_is_fixed_and_keeps_seal(tmp_path, monkeypatch):
    app, cookie, package, _, trusted, config = ready(tmp_path)
    approved(app, package)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    assert post(app, cookie, "/admin/style/retain", values)[0] == 200
    _, _, delete_values = form(app, cookie, "/admin/style/delete-source", "delete_source")
    monkeypatch.setattr(
        app._source_seal,
        "delete",
        lambda _owner: (_ for _ in ()).throw(RuntimeError("synthetic private detail")),
    )
    status, _, page = post(app, cookie, "/admin/style/delete-source", delete_values)
    assert status == 503 and "synthetic private detail" not in page
    assert StyleSourceSealStore(trusted, config).read("synthetic-owner") is not None


def test_main_assembles_real_source_seal_without_listener(tmp_path, monkeypatch):
    root, _ = private_source(tmp_path)
    trusted = settings(root)
    captured = []

    class FakeServer:
        server_port = 12345

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def set_app(self, app):
            captured.append(app)

        def serve_forever(self):
            return None

    monkeypatch.setattr(admin_main, "load_settings", lambda _path: trusted)
    monkeypatch.setattr(admin_main, "make_server", lambda *_args, **_kwargs: FakeServer())
    assert admin_main.main() == 0
    assert len(captured) == 1 and isinstance(captured[0]._source_seal, StyleSourceSealStore)
    assert captured[0]._phase1_store._strict_source_binding is True
    assert captured[0]._phase1_store._source_gateway is captured[0]._source_gateway
    assert captured[0]._source_gateway._source_seal is captured[0]._source_seal
    assert captured[0]._source_gateway._approval_store is captured[0]._approval_store
    assert not any(path.exists() for path in captured[0]._phase1_store.artifact_paths)
    assert not (root / "admin/style-source/current.json").exists()


def test_main_store_setup_failure_has_fixed_output_without_listener(tmp_path, monkeypatch, capsys):
    root, _ = private_source(tmp_path)
    (root / "admin/style-source").symlink_to(tmp_path, target_is_directory=True)
    trusted = settings(root)

    class FakeServer:
        server_port = 12345

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(admin_main, "load_settings", lambda _path: trusted)
    monkeypatch.setattr(admin_main, "make_server", lambda *_args, **_kwargs: FakeServer())
    assert admin_main.main() == 2
    assert capsys.readouterr().out == "Admin unavailable\n"
