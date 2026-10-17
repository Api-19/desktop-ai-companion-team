import re
from datetime import UTC, datetime
from io import BytesIO
from urllib.parse import urlencode, urlsplit

import pytest
from test_local_entry import call, config, settings
from test_mapping_confirmation import ORIGIN, unlock

from companion.admin.app import AdminApp, build_admin_app, load_admin_config
from companion.contracts import ConsentRecord
from companion.memory import CreateMemoryRequest, MemoryField, build_private_memory_service

NOW = datetime(2026, 10, 3, tzinfo=UTC)
OWNER = "synthetic-owner"
FIELDS = [field.value for field in MemoryField]


def request(app, route, **kwargs):
    parsed = urlsplit(route)

    def wsgi(environ, start):
        environ["QUERY_STRING"] = parsed.query
        return app(environ, start)

    return call(wsgi, parsed.path, **kwargs)


def factory_app(tmp_path, seed=False):
    root = tmp_path / "private"
    config(root)
    trusted = settings(root)
    created = []

    def factory():
        service = build_private_memory_service(trusted, clock=lambda: NOW)
        created.append(service)
        return service

    app = AdminApp(
        OWNER, host="127.0.0.1", port=12345, memory_factory=factory, wall_clock=lambda: NOW
    )
    cookie = unlock(app)
    if seed:
        request(app, "/admin/memory", cookie=cookie)
        service = created[0]
        service.grant_consent(
            ConsentRecord(user_id=OWNER, granted=True, granted_at=NOW, allowed_fields=FIELDS)
        )
        entry = remember(service, OWNER, "old <script>synthetic</script>")
        return app, cookie, created, trusted, entry
    return app, cookie, created, trusted, None


def remember(service, owner, value, field=MemoryField.EXPLICIT_NOTE):
    return service.create_memory(
        CreateMemoryRequest(
            user_id=owner, field=field, value=value, source_timestamp=NOW, explicitly_requested=True
        )
    )


def get_form(app, cookie, action, memory_id=None):
    route = "/admin/memory/" + action
    if memory_id:
        route += "?" + urlencode({"memory_id": memory_id})
    status, headers, page = request(app, route, cookie=cookie)
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    values = dict(re.findall(r'name="(csrf|memory_token)" value="([^"]+)"', page))
    assert set(values) == {"csrf", "memory_token"}
    values["confirm"] = "yes"
    return values, page


def submit(app, cookie, action, values):
    return request(
        app, "/admin/memory/" + action, method="POST", cookie=cookie, origin=ORIGIN, form=values
    )


def grant(app, cookie, fields):
    values, _ = get_form(app, cookie, "consent")
    values.update({field: "yes" for field in fields})
    return submit(app, cookie, "consent", values)


@pytest.mark.parametrize("fault", ["locked", "expired", "owner", "host", "origin", "closed"])
@pytest.mark.parametrize(
    "route", ["/admin/memory", "/admin/memory/consent", "/admin/memory/revoke"]
)
def test_memory_gate_does_not_construct(fault, route):
    called = []
    now = [0]
    app = AdminApp(
        OWNER,
        host="127.0.0.1",
        port=12345,
        clock=lambda: now[0],
        memory_factory=lambda: called.append(True),
    )
    cookie = "" if fault == "locked" else unlock(app)
    args = {}
    if fault == "expired":
        now[0] = 900
    if fault == "owner":
        app._sessions[cookie.split("=", 1)[1]].owner_id = "other"
    if fault == "host":
        args["host"] = "invalid"
    if fault == "origin":
        args["origin"] = "http://invalid"
    if fault == "closed":
        app.close()
    assert request(app, route, cookie=cookie, **args)[0] == 403
    assert called == []


def test_lazy_product_and_main_home_other_pages_do_not_open_memory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "private"
    config(root)
    trusted = settings(root)
    app = build_admin_app(trusted, load_admin_config(trusted), port=12345, wall_clock=lambda: NOW)
    cookie = unlock(app)
    for route in ["/admin/home", "/admin/usage", "/admin/import/preview"]:
        request(app, route, cookie=cookie)
    assert not (root / "memory").exists()
    assert "/admin/memory" in request(app, "/admin/home", cookie=cookie)[2]
    assert request(app, "/admin/memory", cookie=cookie)[0] == 200
    assert (root / "memory/memory.sqlite3").stat().st_mode & 0o777 == 0o600
    import companion.admin.__main__ as main

    monkeypatch.setattr(main, "load_settings", lambda *_: trusted)

    class Server:
        server_port = 12345

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_app(self, value):
            self.app = value

        def serve_forever(self):
            cookie = unlock(self.app)
            assert request(self.app, "/admin/memory", cookie=cookie)[0] == 200
            assert grant(self.app, cookie, ["preference"])[0] == 200
            raise KeyboardInterrupt

    monkeypatch.setattr(main, "make_server", lambda *a, **k: Server())
    assert main.main() == 0
    capsys.readouterr()


def test_real_memory_lifecycle_and_cross_user_isolation(tmp_path):
    app, cookie, created, trusted, _ = factory_app(tmp_path)
    page = request(app, "/admin/memory", cookie=cookie)[2]
    assert "未授权" in page
    service = created[0]
    assert service.list_memories(OWNER) == ()
    values, page = get_form(app, cookie, "consent")
    assert all(field in page for field in FIELDS)
    assert "跨会话" in page and "云端" in page and "风格" in page
    assert grant(app, cookie, FIELDS)[0] == 200
    entry = remember(service, OWNER, "old <script>synthetic</script>")
    service.grant_consent(
        ConsentRecord(
            user_id="synthetic-other", granted=True, granted_at=NOW, allowed_fields=FIELDS
        )
    )
    other = remember(service, "synthetic-other", "OTHER_PRIVATE_VALUE")
    page = request(app, "/admin/memory", cookie=cookie)[2]
    assert "old &lt;script&gt;synthetic&lt;/script&gt;" in page
    assert "<script>" not in page and "OTHER_PRIVATE_VALUE" not in page
    assert OWNER not in page and str(trusted.private_data_dir) not in page
    values, _ = get_form(app, cookie, "correct", entry.memory_id)
    values["value"] = "new <b>note</b>"
    assert submit(app, cookie, "correct", values)[0] == 200
    assert service.retrieve_memories(OWNER, "old") == ()
    assert service.list_memories(OWNER)[0].value == "new <b>note</b>"
    values, _ = get_form(app, cookie, "delete", entry.memory_id)
    assert submit(app, cookie, "delete", values)[0] == 200
    assert service.list_memories(OWNER) == () and service.retrieve_memories(OWNER, "new") == ()
    remember(service, OWNER, "delete on revoke")
    values, _ = get_form(app, cookie, "revoke")
    assert submit(app, cookie, "revoke", values)[0] == 200
    assert service.get_consent(OWNER).granted is False
    assert service.list_memories(OWNER) == () and service.retrieve_memories(OWNER, "delete") == ()
    assert service.list_memories("synthetic-other") == (other,)
    assert len(created) == 1


def test_narrowing_removes_old_fields_and_preserves_other_owner(tmp_path):
    app, cookie, created, _, entry = factory_app(tmp_path, True)
    service = created[0]
    keep = remember(service, OWNER, "keep", MemoryField.PREFERENCE)
    assert grant(app, cookie, ["preference"])[0] == 200
    assert service.list_memories(OWNER) == (keep,)
    assert service.retrieve_memories(OWNER, "old") == ()
    assert grant(app, cookie, FIELDS)[0] == 200
    assert all(x.memory_id != entry.memory_id for x in service.list_memories(OWNER))


@pytest.mark.parametrize(
    "fault",
    [
        "csrf",
        "extra",
        "owner",
        "path",
        "confirm",
        "empty_fields",
        "unknown_field",
        "wrong_checkbox",
        "wrong_token",
        "cross_session",
    ],
)
def test_invalid_consent_has_no_mutation_and_token_is_consumed(tmp_path, fault):
    app, cookie, created, _, _ = factory_app(tmp_path)
    values, _ = get_form(app, cookie, "consent")
    values["preference"] = "yes"
    original = values.copy()
    if fault == "csrf":
        values["csrf"] = "wrong"
    elif fault in {"extra", "owner", "path", "unknown_field"}:
        values[fault] = "forged"
    elif fault == "confirm":
        values.pop("confirm")
    elif fault == "empty_fields":
        values.pop("preference")
    elif fault == "wrong_checkbox":
        values["preference"] = "no"
    elif fault == "wrong_token":
        values["memory_token"] = "wrong"
    other_cookie = unlock(app) if fault == "cross_session" else cookie
    assert submit(app, other_cookie, "consent", values)[0] in {400, 403}
    assert created[0].get_consent(OWNER) is None
    if fault != "cross_session":
        assert submit(app, cookie, "consent", original)[0] == 403


@pytest.mark.parametrize("action", ["consent", "correct", "delete", "revoke"])
def test_replay_and_binding_drift_rejected(tmp_path, action):
    app, cookie, created, _, entry = factory_app(tmp_path, True)
    service = created[0]
    values, _ = get_form(
        app, cookie, action, entry.memory_id if action in {"correct", "delete"} else None
    )
    if action == "consent":
        values["preference"] = "yes"
    if action == "correct":
        values["value"] = "corrected"
    service.grant_consent(
        ConsentRecord(
            user_id=OWNER,
            granted=True,
            granted_at=datetime(2026, 10, 4, tzinfo=UTC),
            allowed_fields=FIELDS,
        )
    )
    assert submit(app, cookie, action, values)[0] == 409
    assert submit(app, cookie, action, values)[0] == 403
    assert service.list_memories(OWNER) == (entry,)


@pytest.mark.parametrize("action", ["correct", "delete"])
@pytest.mark.parametrize(
    "query",
    [
        "",
        "memory_id=bad",
        "memory_id=" + "a" * 32 + "&owner=x",
        "memory_id=" + "a" * 32 + "&memory_id=" + "a" * 32,
        "memory_id=" + "a" * 32,
    ],
)
def test_target_query_strict_and_unknown_targets_not_disclosed(tmp_path, action, query):
    app, cookie, created, _, _ = factory_app(tmp_path)
    status, _, page = request(app, "/admin/memory/" + action + "?" + query, cookie=cookie)
    assert status in {400, 403} and "a" * 32 not in page
    if not query or "owner=" in query or query == "memory_id=bad" or query.count("memory_id=") > 1:
        assert not created


@pytest.mark.parametrize("action", ["correct", "delete"])
def test_cross_owner_target_refused(tmp_path, action):
    app, cookie, created, _, _ = factory_app(tmp_path)
    request(app, "/admin/memory", cookie=cookie)
    service = created[0]
    service.grant_consent(
        ConsentRecord(user_id="other", granted=True, granted_at=NOW, allowed_fields=FIELDS)
    )
    entry = remember(service, "other", "CROSS_OWNER_PRIVATE")
    code, _, page = request(
        app,
        "/admin/memory/" + action + "?" + urlencode({"memory_id": entry.memory_id}),
        cookie=cookie,
    )
    assert code == 403 and "CROSS_OWNER_PRIVATE" not in page


@pytest.mark.parametrize("value", ["", " " * 3, "x" * 501])
def test_bad_correction_value_does_not_change(tmp_path, value):
    app, cookie, created, _, entry = factory_app(tmp_path, True)
    values, _ = get_form(app, cookie, "correct", entry.memory_id)
    values["value"] = value
    assert submit(app, cookie, "correct", values)[0] == 400
    assert created[0].list_memories(OWNER) == (entry,)
    assert submit(app, cookie, "correct", values)[0] == 403


def test_duplicate_form_refused(tmp_path):
    app, cookie, created, _, _ = factory_app(tmp_path)
    values, _ = get_form(app, cookie, "consent")
    values["preference"] = "yes"
    raw = (urlencode(values) + "&confirm=yes").encode()
    response = []
    app(
        {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/admin/memory/consent",
            "HTTP_HOST": "127.0.0.1:12345",
            "HTTP_ORIGIN": ORIGIN,
            "HTTP_COOKIE": cookie,
            "CONTENT_LENGTH": str(len(raw)),
            "wsgi.input": BytesIO(raw),
        },
        lambda status, headers: response.append(status),
    )
    assert response[0].startswith(("400", "403"))
    assert created[0].get_consent(OWNER) is None
    assert submit(app, cookie, "consent", values)[0] == 403


@pytest.mark.parametrize("action", ["consent", "correct", "delete", "revoke"])
def test_after_write_readback_failure_is_unconfirmed_not_rollback(tmp_path, monkeypatch, action):
    app, cookie, created, _, entry = factory_app(tmp_path, True)
    service = created[0]
    values, _ = get_form(
        app, cookie, action, entry.memory_id if action in {"correct", "delete"} else None
    )
    if action == "consent":
        values["preference"] = "yes"
    if action == "correct":
        values["value"] = "written"
    name = {
        "consent": "grant_consent",
        "correct": "correct_memory",
        "delete": "delete_memory",
        "revoke": "revoke_consent",
    }[action]
    original = getattr(service, name)

    def write(*a, **k):
        result = original(*a, **k)
        monkeypatch.setattr(
            service,
            "get_consent",
            lambda *_: (_ for _ in ()).throw(OSError("SYNTHETIC_PRIVATE_FAILURE")),
        )
        return result

    monkeypatch.setattr(service, name, write)
    code, _, page = submit(app, cookie, action, values)
    assert code == 503 and "结果未确认" in page
    assert "SYNTHETIC_PRIVATE_FAILURE" not in page and entry.value not in page
    assert submit(app, cookie, action, values)[0] == 403
    if action in {"delete", "revoke", "consent"}:
        assert service.list_memories(OWNER) == ()
    else:
        assert service.list_memories(OWNER)[0].value == "written"


@pytest.mark.parametrize(
    "fault",
    [
        "consent_owner",
        "consent_fields",
        "consent_forged",
        "entry_owner",
        "entry_field",
        "entry_forged",
        "factory",
        "root_permission",
    ],
)
def test_invalid_public_results_and_private_boundary_fail_closed(tmp_path, monkeypatch, fault):
    app, cookie, created, trusted, entry = factory_app(tmp_path, True)
    service = created[0]
    consent = service.get_consent(OWNER)
    if fault == "consent_owner":
        monkeypatch.setattr(
            service, "get_consent", lambda *_: consent.model_copy(update={"user_id": "other"})
        )
    elif fault == "consent_fields":
        monkeypatch.setattr(
            service,
            "get_consent",
            lambda *_: consent.model_copy(update={"allowed_fields": ["unknown"]}),
        )
    elif fault == "consent_forged":
        monkeypatch.setattr(
            service, "get_consent", lambda *_: consent.model_copy(update={"granted": "yes"})
        )
    elif fault == "entry_owner":
        monkeypatch.setattr(
            service, "list_memories", lambda *_: (entry.model_copy(update={"user_id": "other"}),)
        )
    elif fault == "entry_field":
        monkeypatch.setattr(
            service, "list_memories", lambda *_: (entry.model_copy(update={"field": "unknown"}),)
        )
    elif fault == "entry_forged":
        monkeypatch.setattr(
            service, "list_memories", lambda *_: (entry.model_copy(update={"value": 42}),)
        )
    elif fault == "root_permission":
        trusted.private_data_dir.chmod(0o755)
    else:
        app = AdminApp(OWNER, host="127.0.0.1", port=12345, memory_factory=lambda: object())
        cookie = unlock(app)
    code, _, page = request(app, "/admin/memory", cookie=cookie)
    assert code == 503 and "记忆管理不可用" in page
    assert entry.value not in page and str(trusted.private_data_dir) not in page


def test_cached_private_service_detects_inode_replacement(tmp_path):
    app, cookie, created, trusted, _ = factory_app(tmp_path)
    assert request(app, "/admin/memory", cookie=cookie)[0] == 200
    path = trusted.private_data_dir / "memory/memory.sqlite3"
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o600)
    replacement.replace(path)
    assert request(app, "/admin/memory", cookie=cookie)[0] == 503
    assert len(created) == 1


def test_full_500_multibyte_correction_is_supported(tmp_path):
    app, cookie, created, _, entry = factory_app(tmp_path, True)
    values, _ = get_form(app, cookie, "correct", entry.memory_id)
    values["value"] = "🙂" * 500
    assert submit(app, cookie, "correct", values)[0] == 200
    assert created[0].list_memories(OWNER)[0].value == values["value"]


@pytest.mark.parametrize("action", ["correct", "delete"])
def test_target_change_invalidates_form_even_with_same_consent(tmp_path, action):
    app, cookie, created, _, entry = factory_app(tmp_path, True)
    service = created[0]
    values, _ = get_form(app, cookie, action, entry.memory_id)
    if action == "correct":
        values["value"] = "new"
    from companion.memory import CorrectMemoryRequest

    service.correct_memory(
        CorrectMemoryRequest(
            user_id=OWNER, memory_id=entry.memory_id, value="external change", source_timestamp=NOW
        )
    )
    assert submit(app, cookie, action, values)[0] == 409
    assert service.list_memories(OWNER)[0].value == "external change"


@pytest.mark.parametrize("action", ["consent", "correct", "delete", "revoke"])
def test_mutation_return_model_is_deep_validated(tmp_path, monkeypatch, action):
    app, cookie, created, _, entry = factory_app(tmp_path, True)
    service = created[0]
    values, _ = get_form(
        app, cookie, action, entry.memory_id if action in {"correct", "delete"} else None
    )
    if action == "consent":
        values["preference"] = "yes"
    if action == "correct":
        values["value"] = "changed"
    name = {
        "consent": "grant_consent",
        "correct": "correct_memory",
        "delete": "delete_memory",
        "revoke": "revoke_consent",
    }[action]
    original = getattr(service, name)

    def invalid(*args, **kwargs):
        result = original(*args, **kwargs)
        return result.model_copy(update={"user_id": "forged-owner"})

    monkeypatch.setattr(service, name, invalid)
    code, _, page = submit(app, cookie, action, values)
    assert code == 503 and "结果未确认" in page and "forged-owner" not in page


@pytest.mark.parametrize("action", ["consent", "correct", "delete", "revoke"])
def test_direct_post_without_issued_token_does_not_build(tmp_path, action):
    reads = []
    app = AdminApp(OWNER, host="127.0.0.1", port=12345, memory_factory=lambda: reads.append(True))
    cookie = unlock(app)
    assert (
        submit(app, cookie, action, {"csrf": "fake", "memory_token": "fake", "confirm": "yes"})[0]
        == 403
    )
    assert reads == []


def test_existing_empty_field_consent_can_still_be_revoked(tmp_path):
    app, cookie, created, _, _ = factory_app(tmp_path)
    request(app, "/admin/memory", cookie=cookie)
    service = created[0]
    service.grant_consent(
        ConsentRecord(user_id=OWNER, granted=True, granted_at=NOW, allowed_fields=[])
    )
    values, _ = get_form(app, cookie, "revoke")
    assert submit(app, cookie, "revoke", values)[0] == 200
    assert service.get_consent(OWNER).granted is False


def test_memory_consent_preserves_independent_profile_permission(tmp_path):
    app, cookie, created, _, _ = factory_app(tmp_path)
    assert request(app, "/admin/memory", cookie=cookie)[0] == 200
    service = created[0]
    service.grant_consent(ConsentRecord(user_id=OWNER, granted=True, granted_at=NOW,
                                       allowed_fields=["profile", "preference"]))
    item = service.add_profile_candidate(
        user_id=OWNER, category="habit", topic="饮水", text="合成少喝水",
        source="owner_added", evidence_date=NOW.date(), confidence="low",
    )
    assert request(app, "/admin/memory", cookie=cookie)[0] == 200
    values, page = get_form(app, cookie, "consent")
    assert 'name="profile"' not in page
    values.update({field: "yes" for field in FIELDS})
    assert submit(app, cookie, "consent", values)[0] == 200
    assert set(service.get_consent(OWNER).allowed_fields) == {*FIELDS, "profile"}
    assert service.list_profile(OWNER) == (item,)


def test_memory_revoke_accepts_receipt_including_profiles(tmp_path):
    app, cookie, created, _, _ = factory_app(tmp_path)
    assert request(app, "/admin/memory", cookie=cookie)[0] == 200
    service = created[0]
    service.grant_consent(ConsentRecord(user_id=OWNER, granted=True, granted_at=NOW,
                                       allowed_fields=["profile", "preference"]))
    service.add_profile_candidate(user_id=OWNER, category="habit", topic="饮水",
                                  text="合成少喝水", source="owner_added",
                                  evidence_date=NOW.date(), confidence="low")
    remember(service, OWNER, "合成偏好", field=MemoryField.PREFERENCE)
    values, _ = get_form(app, cookie, "revoke")
    assert submit(app, cookie, "revoke", values)[0] == 200
    assert service.list_profile(OWNER) == ()
    assert service.list_memories(OWNER) == ()
