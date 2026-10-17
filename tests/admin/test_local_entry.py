from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import pytest

from companion.admin.app import AdminApp, AdminConfigError, load_admin_config
from companion.importer import ImportBatch, MessageDirection, SourceChatRecord
from companion.runtime.settings import Settings


def settings(root: Path) -> Settings:
    return Settings(
        environment="test",
        log_level="ERROR",
        private_data_dir=root,
        runtime_data_dir=root / "runtime",
        test_results_dir=root / "results",
    )


def config(root: Path, content: str = 'enabled = true\nowner_id = "synthetic-owner"\n') -> None:
    root.mkdir(mode=0o700)
    folder = root / "admin"
    folder.mkdir(mode=0o700)
    path = folder / "config.toml"
    path.write_text(content)
    path.chmod(0o600)


def call(
    app: AdminApp,
    path: str,
    *,
    method: str = "GET",
    cookie: str = "",
    origin: str = "",
    host: str = "127.0.0.1:12345",
    form: dict[str, str] | None = None,
):
    body = urlencode(form or {}).encode()
    headers = {}

    def start(status, values):
        headers.update(values)
        headers["status"] = status

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "HTTP_HOST": host,
        "HTTP_ORIGIN": origin,
        "HTTP_COOKIE": cookie,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": BytesIO(body),
    }
    payload = b"".join(app(environ, start)).decode()
    return int(headers["status"][:3]), headers, payload


def batch() -> ImportBatch:
    return ImportBatch(
        records=[
            SourceChatRecord(
                conversation_id="synthetic-chat",
                message_id="m1",
                timestamp=datetime(2026, 9, 24, tzinfo=UTC),
                source_speaker_id="synthetic-owner",
                source_speaker_name="Synthetic Owner",
                direction=MessageDirection.OUTBOUND,
                content_type="text",
                text="synthetic sample",
            )
        ],
        issues=[],
    )


def test_config_disabled_and_permissions(tmp_path):
    root = tmp_path / "private"
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(root))
    config(root, 'owner_id = "synthetic-owner"\n')
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(root))
    config_path = root / "admin" / "config.toml"
    config_path.chmod(0o644)
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(root))
    config_path.chmod(0o600)
    config_path.write_text('enabled = true\nowner_id = "synthetic-owner"\n')
    assert load_admin_config(settings(root)).owner_id == "synthetic-owner"
    alias = tmp_path / "linked-private"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(alias))
    config_path.unlink()
    config_path.symlink_to(root / "outside.toml")
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(root))


def test_config_unknown_fields_and_non_loopback(tmp_path):
    root = tmp_path / "private"
    config(root, 'enabled = true\nowner_id = "synthetic-owner"\nother = 1\n')
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(root))
    with pytest.raises(ValueError):
        AdminApp("synthetic-owner", host="0.0.0.0", port=12345)


def test_preauth_expires_and_has_bounded_capacity():
    now = [100.0]
    app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345, clock=lambda: now[0])
    for _ in range(65):
        assert call(app, "/admin")[0] == 200
    assert len(app._preauth) == 64
    old_tokens = set(app._preauth)
    now[0] += 300
    assert call(app, "/admin")[0] == 200
    assert len(app._preauth) == 1
    assert old_tokens.isdisjoint(app._preauth)


def test_parent_symlink_cannot_reach_private_config(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    root = real_parent / "private"
    config(root)
    assert load_admin_config(settings(root)).owner_id == "synthetic-owner"
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(linked_parent / "private"))


@pytest.mark.parametrize("owner_id", ["x" * 121, "synthetic\x01owner"])
def test_owner_id_length_and_control_characters_fail_at_both_entries(tmp_path, owner_id):
    root = tmp_path / "private"
    escaped_owner_id = owner_id.replace(chr(1), r"\u0001")
    config(root, f'enabled = true\nowner_id = "{escaped_owner_id}"\n')
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(root))
    with pytest.raises(ValueError):
        AdminApp(owner_id, host="127.0.0.1", port=12345)


def test_unlock_preview_and_fail_closed_actions():
    now = [100.0]
    app = AdminApp(
        "synthetic-owner", host="127.0.0.1", port=12345, clock=lambda: now[0], batch_loader=batch
    )
    code = app.issue_start_code()
    assert code not in repr(app)
    status, headers, page = call(app, "/admin")
    assert status == 200
    pre_cookie = headers["Set-Cookie"].split(";", 1)[0]
    csrf = page.split('name="csrf" value="')[1].split('"', 1)[0]
    assert (
        call(
            app,
            "/admin/unlock",
            method="POST",
            cookie=pre_cookie,
            origin="http://evil.test",
            form={"code": code, "csrf": csrf},
        )[0]
        == 403
    )
    assert (
        call(
            app,
            "/admin/unlock",
            method="POST",
            cookie=pre_cookie,
            origin="http://127.0.0.1:12345",
            form={"code": code, "csrf": "bad"},
        )[0]
        == 403
    )
    status, headers, _ = call(
        app,
        "/admin/unlock",
        method="POST",
        cookie=pre_cookie,
        origin="http://127.0.0.1:12345",
        form={"code": code, "csrf": csrf, "owner_id": "attacker"},
    )
    assert status == 303
    session = headers["Set-Cookie"].split(";", 1)[0]
    assert "synthetic-owner" not in session
    assert (
        call(
            app,
            "/admin/unlock",
            method="POST",
            cookie=pre_cookie,
            origin="http://127.0.0.1:12345",
            form={"code": code, "csrf": csrf},
        )[0]
        == 403
    )
    assert call(app, "/admin/import/preview", cookie="admin_session=fake")[0] == 403
    status, _, preview = call(app, "/admin/import/preview", cookie=session)
    assert status == 200 and "synthetic sample" in preview
    assert call(app, "/admin/import/preview", cookie=session, host="evil.test")[0] == 403
    assert (
        call(
            app,
            "/admin/import/confirm",
            method="POST",
            cookie=session,
            origin="http://127.0.0.1:12345",
            form={"csrf": "wrong"},
        )[0]
        == 403
    )
    assert (
        call(
            app,
            "/admin/style/decide",
            method="POST",
            cookie=session,
            origin="http://127.0.0.1:12345",
            form={"csrf": "wrong"},
        )[0]
        == 403
    )
    assert "未就绪" in call(app, "/admin/style/review", cookie=session)[2]
    now[0] += 901
    assert call(app, "/admin/import/preview", cookie=session)[0] == 403


def test_code_expires_and_batch_change_never_confirms():
    now = [100.0]
    source = [batch()]
    app = AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        clock=lambda: now[0],
        batch_loader=lambda: source[0],
    )
    code = app.issue_start_code()
    status, headers, page = call(app, "/admin")
    assert status == 200
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    csrf = page.split('name="csrf" value="')[1].split('"', 1)[0]
    now[0] += 301
    assert (
        call(
            app,
            "/admin/unlock",
            method="POST",
            cookie=cookie,
            origin="http://127.0.0.1:12345",
            form={"code": code, "csrf": csrf},
        )[0]
        == 403
    )
    assert (
        call(
            app,
            "/admin/import/confirm",
            method="POST",
            origin="http://127.0.0.1:12345",
            form={"csrf": csrf},
        )[0]
        == 403
    )


def test_valid_session_batch_drift_and_privacy_absence():
    source = [batch()]
    app = AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        batch_loader=lambda: source[0],
    )
    code = app.issue_start_code()
    _, headers, page = call(app, "/admin")
    pre_cookie = headers["Set-Cookie"].split(";", 1)[0]
    csrf = page.split('name="csrf" value="')[1].split('"', 1)[0]
    status, headers, _ = call(
        app,
        "/admin/unlock",
        method="POST",
        cookie=pre_cookie,
        origin="http://127.0.0.1:12345",
        form={"code": code, "csrf": csrf},
    )
    assert status == 303
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    status, _, preview_page = call(app, "/admin/import/preview", cookie=cookie)
    assert status == 200
    token = cookie.split("=", 1)[1]
    session = app._sessions[token]
    confirm_csrf = app._token(session, "confirm")
    decide_csrf = app._token(session, "decide")
    assert (
        call(
            app,
            "/admin/style/decide",
            method="POST",
            cookie=cookie,
            origin="http://127.0.0.1:12345",
            form={"csrf": decide_csrf, "owner_id": "attacker", "decision": "approve"},
        )[0]
        == 503
    )
    source[0] = source[0].model_copy(
        update={"records": [source[0].records[0].model_copy(update={"text": "changed"})]}
    )
    confirm_token = preview_page.split('name="confirm_token" value="')[1].split('"', 1)[0]
    assert (
        call(
            app,
            "/admin/import/confirm",
            method="POST",
            cookie=cookie,
            origin="http://127.0.0.1:12345",
            form={
                "csrf": confirm_csrf,
                "confirm_token": confirm_token,
            },
        )[0]
        == 409
    )
    app.close()
    assert call(app, "/admin/import/preview", cookie=cookie)[0] == 403


def test_start_code_rotation_and_close_invalidate_access():
    app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345)
    old_code = app.issue_start_code()
    new_code = app.issue_start_code()
    _, headers, page = call(app, "/admin")
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    csrf = page.split('name="csrf" value="')[1].split('"', 1)[0]
    assert (
        call(
            app,
            "/admin/unlock",
            method="POST",
            cookie=cookie,
            origin="http://127.0.0.1:12345",
            form={"code": old_code, "csrf": csrf},
        )[0]
        == 403
    )
    app.close()
    assert (
        call(
            app,
            "/admin/unlock",
            method="POST",
            cookie=cookie,
            origin="http://127.0.0.1:12345",
            form={"code": new_code, "csrf": csrf},
        )[0]
        == 403
    )
    with pytest.raises(RuntimeError):
        app.issue_start_code()
