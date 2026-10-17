# ruff: noqa: RUF001

"""Owner character settings, using disposable synthetic configuration only."""

import html
import json
import re
from io import BytesIO
from urllib.parse import urlencode

import pytest
from test_local_entry import config, settings
from test_mapping_confirmation import ORIGIN, unlock

import companion.admin.app as admin
from companion.admin.app import AdminApp, build_admin_app, load_admin_config

ROUTE = "/admin/character"
FIELDS = {"csrf", "character_token", "name", "confirm"}


def request(app, cookie="", *, method="GET", form=None, raw=None, query="", **headers):
    body = urlencode(form or {}).encode() if raw is None else raw
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": ROUTE,
        "QUERY_STRING": query,
        "HTTP_HOST": "127.0.0.1:12345",
        "HTTP_ORIGIN": ORIGIN if method == "POST" else "",
        "HTTP_COOKIE": cookie,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": BytesIO(body),
        **headers,
    }
    captured = []
    page = b"".join(app(environ, lambda status, values: captured.append((status, values))))
    status, response_headers = captured[0]
    return int(status[:3]), dict(response_headers), page.decode()


def product(tmp_path, *, name="kiki", **kwargs):
    root = tmp_path / "synthetic-private"
    config(root)
    trusted = settings(root)
    path = root / "dialogue" / "runtime-config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"character_name": name}), encoding="utf-8")
    app = build_admin_app(trusted, load_admin_config(trusted), port=12345, **kwargs)
    return app, trusted, path


def character_form(app, cookie):
    status, _, page = request(app, cookie)
    assert status == 200
    values = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', page))
    return {key: html.unescape(value) for key, value in values.items()}


def test_character_form_and_save(tmp_path):
    app, _, path = product(tmp_path)
    cookie = unlock(app)
    original = path.read_bytes()
    status, headers, page = request(app, cookie)
    assert status == 200
    assert "<h1>角色名称</h1>" in page
    assert '<form method="post" action="/admin/character">' in page
    assert "保存后需重启对话才生效。" in page
    assert set(re.findall(r'name="([^"]+)"', page)) == FIELDS
    assert '<input type="checkbox" name="confirm" value="yes"' in page
    assert path.read_bytes() == original
    assert headers["Cache-Control"] == "no-store"
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert int(headers["Content-Length"]) == len(page.encode())
    values = character_form(app, cookie)
    values["name"] = " 合成新角色 "
    status, headers, page = request(app, cookie, method="POST", form=values)
    assert status == 200
    assert "<p>角色名称已保存，重启对话后生效。</p>" in page
    assert '<a href="/admin/character">返回角色名称</a>' in page
    assert '<nav aria-label="管理导航">' in page
    assert int(headers["Content-Length"]) == len(page.encode())
    assert json.loads(path.read_bytes()) == {"character_name": "合成新角色"}


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("gate", ["locked", "expired", "owner", "host", "origin", "closed"])
def test_character_requires_owner_csrf_and_one_time_token(tmp_path, method, gate):
    app, _, path = product(tmp_path)
    now = [0.0]
    app._clock = lambda: now[0]
    cookie = unlock(app)
    values = character_form(app, cookie)
    original = path.read_bytes()
    headers = {}
    if gate == "locked":
        cookie = ""
    elif gate == "expired":
        now[0] = 900
    elif gate == "owner":
        app._sessions[cookie.split("=", 1)[1]].owner_id = "other-owner"
    elif gate == "host":
        headers["HTTP_HOST"] = "invalid:12345"
    elif gate == "origin":
        headers["HTTP_ORIGIN"] = "http://invalid"
    else:
        app.close()
    values["name"] = "合成新角色"
    assert request(app, cookie, method=method, form=values, **headers)[0] == 403
    assert path.read_bytes() == original


@pytest.mark.parametrize("bad_csrf", ["bad", "合成错误"])
def test_character_csrf_failure_preserves_action_token(tmp_path, bad_csrf):
    app, _, _ = product(tmp_path)
    cookie = unlock(app)
    values = character_form(app, cookie)
    assert request(app, cookie, method="POST", form={**values, "csrf": bad_csrf})[0] == 403
    assert request(app, cookie, method="POST", form=values)[0] == 200
    assert request(app, cookie, method="POST", form=values)[0] == 403


@pytest.mark.parametrize("fault", ["missing", "mismatch", "unicode", "old", "cross_session"])
def test_character_invalid_action_token_is_consumed(tmp_path, fault):
    app, _, _ = product(tmp_path)
    cookie = unlock(app)
    values = character_form(app, cookie)
    current = values
    if fault == "old":
        current = character_form(app, cookie)
    elif fault == "cross_session":
        other = unlock(app)
        current = character_form(app, other)
        values = {**current, "character_token": values["character_token"]}
        cookie = other
    else:
        values = {
            **values,
            "character_token": {"missing": "", "mismatch": "bad", "unicode": "合成错误"}[fault],
        }
    assert request(app, cookie, method="POST", form=values)[0] == 403
    assert request(app, cookie, method="POST", form=current)[0] == 403


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_character_rejects_duplicate_fields(tmp_path, field):
    app, _, path = product(tmp_path)
    cookie = unlock(app)
    values = character_form(app, cookie)
    original = path.read_bytes()
    raw = urlencode([*values.items(), (field, values[field])]).encode()
    status, _, page = request(app, cookie, method="POST", raw=raw)
    assert status == 400 and "角色名称请求无效" in page
    assert path.read_bytes() == original
    assert request(app, cookie, method="POST", form=values)[0] == 200


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_character_query_is_rejected_and_get_clears_old_form(tmp_path, method):
    app, _, _ = product(tmp_path)
    cookie = unlock(app)
    values = character_form(app, cookie)
    assert request(app, cookie, method=method, query="name=bad", form=values)[0] == 400
    expected = 403 if method == "GET" else 200
    assert request(app, cookie, method="POST", form=values)[0] == expected


@pytest.mark.parametrize("fault", ["extra", "missing", "confirmation", "invalid_name"])
def test_character_invalid_requests(tmp_path, fault):
    app, _, path = product(tmp_path)
    cookie = unlock(app)
    values = character_form(app, cookie)
    invalid = dict(values)
    if fault == "extra":
        invalid["expected_digest"] = "a" * 64
    elif fault == "missing":
        invalid.pop("confirm")
    elif fault == "confirmation":
        invalid["confirm"] = "no"
    else:
        invalid["name"] = ""
    original = path.read_bytes()
    status, _, page = request(app, cookie, method="POST", form=invalid)
    assert status == 400
    assert ("角色名称无效" if fault == "invalid_name" else "角色名称请求无效") in page
    assert path.read_bytes() == original
    expected = 200 if fault in {"extra", "missing"} else 403
    assert request(app, cookie, method="POST", form=values)[0] == expected


def test_character_stale_form_conflicts(tmp_path):
    app, _, path = product(tmp_path)
    cookie = unlock(app)
    values = character_form(app, cookie)
    changed = path.read_bytes() + b"\n"
    path.write_bytes(changed)
    status, _, page = request(app, cookie, method="POST", form=values)
    assert status == 409 and "角色名称配置已变化，请重新查看" in page
    assert path.read_bytes() == changed
    assert request(app, cookie, method="POST", form=values)[0] == 403


@pytest.mark.parametrize("fault", ["unconfigured", "missing", "invalid"])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_character_missing_config_is_unavailable(tmp_path, fault, method):
    app, _, path = product(tmp_path)
    cookie = unlock(app)
    values = character_form(app, cookie)
    if fault == "unconfigured":
        app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345)
        cookie = unlock(app)
    elif fault == "missing":
        path.unlink()
    else:
        path.write_text("{")
    status, _, page = request(app, cookie, method=method, form=values)
    assert status == 503 and "角色名称配置不可用" in page


def test_character_get_exposes_only_escaped_name(tmp_path, monkeypatch):
    name = '<合成 "名字">&'
    app, _, path = product(tmp_path, name=name)
    path.write_text(
        json.dumps(
            {
                "character_name": name,
                "provider": {"api_key": "SYNTHETIC_SECRET_CANARY"},
                "budget": "SYNTHETIC_BUDGET_CANARY",
                "style": {"path": str(path.parent / "SYNTHETIC_STYLE_CANARY")},
                "unknown": "SYNTHETIC_UNKNOWN_CANARY",
            }
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("character GET must not load dialogue or call provider")

    monkeypatch.setattr("companion.dialogue.config.load_dialogue_config", forbidden)
    monkeypatch.setattr("companion.dialogue.provider.OpenAICompatibleProvider.complete", forbidden)
    cookie = unlock(app)
    status, _, page = request(app, cookie)
    assert status == 200 and html.escape(name, quote=True) in page
    assert name not in page
    assert str(path) not in page and str(path.parent) not in page
    for hidden in (
        "SYNTHETIC_SECRET_CANARY",
        "SYNTHETIC_BUDGET_CANARY",
        "SYNTHETIC_STYLE_CANARY",
        "SYNTHETIC_UNKNOWN_CANARY",
        "provider",
        "budget",
        "style",
        "unknown",
        "expected_digest",
        "user_id",
    ):
        assert hidden not in page


@pytest.mark.parametrize(
    "gate",
    [
        "locked",
        "expired",
        "owner",
        "host",
        "origin_get",
        "origin_post",
        "closed",
        "csrf",
        "token",
        "replay",
        "confirm",
        "duplicate",
        "fields",
        "query_get",
        "query_post",
    ],
)
def test_character_gates_before_config_access(tmp_path, monkeypatch, gate):
    app, _, _ = product(tmp_path)
    now = [0.0]
    app._clock = lambda: now[0]
    cookie = unlock(app)
    values = character_form(app, cookie)
    if gate == "replay":
        assert request(app, cookie, method="POST", form=values)[0] == 200
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("gate must precede configuration access")

    monkeypatch.setattr(admin, "read_character_name", forbidden)
    monkeypatch.setattr(admin, "write_character_name", forbidden)
    kwargs = {"method": "POST", "form": values}
    if gate == "locked":
        cookie = ""
    elif gate == "expired":
        now[0] = 900
    elif gate == "owner":
        app._sessions[cookie.split("=", 1)[1]].owner_id = "other-owner"
    elif gate == "host":
        kwargs["HTTP_HOST"] = "invalid:12345"
    elif gate.startswith("origin_"):
        kwargs.update(
            method="GET" if gate == "origin_get" else "POST", HTTP_ORIGIN="http://invalid"
        )
    elif gate == "closed":
        app.close()
    elif gate in {"csrf", "token", "confirm"}:
        field = "character_token" if gate == "token" else gate
        kwargs["form"] = {**values, field: "bad"}
    elif gate == "duplicate":
        kwargs["raw"] = urlencode([*values.items(), ("name", "bad")]).encode()
    elif gate == "fields":
        kwargs["form"] = {**values, "user_id": "other-owner"}
    elif gate.startswith("query_"):
        kwargs.update(method="GET" if gate == "query_get" else "POST", query="bad=1")
    assert request(app, cookie, **kwargs)[0] in {400, 403}
    assert calls == []


def test_character_default_construction_and_other_pages_do_not_read(tmp_path, monkeypatch):
    from test_local_entry import call

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration access outside character page")

    monkeypatch.setattr(admin, "read_character_name", forbidden)
    monkeypatch.setattr(admin, "write_character_name", forbidden)
    app, _, _ = product(tmp_path)
    default = AdminApp("synthetic-owner", host="127.0.0.1", port=12345)
    for entry in (app, default):
        cookie = unlock(entry)
        assert call(entry, "/admin/home", cookie=cookie)[0] == 200
        assert call(entry, "/admin/usage", cookie=cookie)[0] in {200, 503}
        assert call(entry, "/admin/style/review", cookie=cookie)[0] == 503
        assert request(entry, cookie, method="PUT")[0] == 404
