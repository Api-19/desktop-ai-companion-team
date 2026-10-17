import html
import re

import pytest
from test_mapping_confirmation import ORIGIN, unlock
from test_memory_management import NOW, OWNER, factory_app, remember, request

from companion.contracts import ConsentRecord

PARTNER = "synthetic-partner"
ROOT = "/admin/partner"
SELF = ROOT + "/self"
ACTIONS = [
    "consent",
    "add",
    "approve",
    "edit",
    "disable",
    "delete",
    "self/confirm",
    "self/correct",
    "self/delete",
    "self/revoke",
]


def form(app, cookie, action, profile_id=None):
    page_path = SELF if action.startswith("self/") else ROOT
    status, headers, page = request(app, page_path, cookie=cookie)
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    for path, body in re.findall(r'<form method="post" action="([^"]+)">(.*?)</form>', page, re.S):
        values = dict(
            (name, html.unescape(value))
            for name, value in re.findall(
                r'<input type="hidden" name="([^"]+)" value="([^"]*)">', body
            )
        )
        if path == ROOT + "/" + action and values.get("profile_id") == profile_id:
            return values
    raise AssertionError(f"missing form {action}")


def post(app, cookie, action, values):
    return request(
        app, ROOT + "/" + action, method="POST", cookie=cookie, origin=ORIGIN, form=values
    )


def ready(tmp_path):
    app, cookie, created, _, _ = factory_app(tmp_path)
    assert request(app, ROOT, cookie=cookie)[0] == 200
    values = form(app, cookie, "consent") | {"mode": "grant"}
    assert post(app, cookie, "consent", values)[0] == 200
    return app, cookie, created[0]


def add(app, cookie, service, **changes):
    before = {item.profile_id for item in service.list_profile(PARTNER)}
    values = (
        form(app, cookie, "add")
        | {
            "topic": "饮水",
            "text": "合成少喝水",
            "category": "habit",
            "evidence_date": NOW.date().isoformat(),
            "evidence_quote": "<script>合成</script>",
            "confidence": "med",
        }
        | changes
    )
    assert post(app, cookie, "add", values)[0] == 200
    return next(item for item in service.list_profile(PARTNER) if item.profile_id not in before)


def approved(app, cookie, service, **changes):
    item = add(app, cookie, service, **changes)
    assert post(app, cookie, "approve", form(app, cookie, "approve", item.profile_id))[0] == 200
    return next(e for e in service.list_profile(PARTNER) if e.profile_id == item.profile_id)


@pytest.mark.parametrize("route", [ROOT, SELF, *(ROOT + "/" + a for a in ACTIONS)])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_all_routes_require_login(tmp_path, route, method):
    app, _, created, _, _ = factory_app(tmp_path)
    assert request(app, route, method=method, origin=ORIGIN)[0] == 403
    assert created == []


def test_owner_lifecycle_and_escaped_evidence(tmp_path):
    app, cookie, service = ready(tmp_path)
    item = add(app, cookie, service)
    assert item.status == "candidate" and item.source == "owner_added"
    page = request(app, ROOT, cookie=cookie)[2]
    assert "&lt;script&gt;合成&lt;/script&gt;" in page and "<script>" not in page
    assert "模拟授权" in page and "owner_added" in page and "candidate" in page
    assert "/self/confirm" not in page and "/self/correct" not in page
    assert not any("确认" in b for b in re.findall(r"<button[^>]*>(.*?)</button>", page))
    for action, expected in [
        ("approve", "owner_approved"),
        ("edit", "owner_approved"),
        ("disable", "disabled"),
    ]:
        values = form(app, cookie, action, item.profile_id)
        if action == "edit":
            values |= {"text": "合成定时喝水", "topic": "饮水", "category": "habit"}
        code, _, result = post(app, cookie, action, values)
        assert code == 200 and "读回确认" in result
        item = service.list_profile(PARTNER)[0]
        assert item.status == expected
    assert item.version == 2 and item.text == "合成定时喝水" and item.disabled_reason == "owner"
    assert "owner" in request(app, ROOT, cookie=cookie)[2]
    assert post(app, cookie, "delete", form(app, cookie, "delete", item.profile_id))[0] == 200
    assert service.list_profile(PARTNER) == ()


def test_simulated_confirmation_correction_and_version_guard(tmp_path):
    app, cookie, service = ready(tmp_path)
    candidate = add(app, cookie, service, topic="作息")
    assert candidate.text not in request(app, SELF, cookie=cookie)[2]
    item = approved(app, cookie, service)
    page = request(app, SELF, cookie=cookie)[2]
    assert "模拟她本人" in page
    assert "确认后可能用于聊天（会发到已配置的云端模型）" in page  # noqa: RUF001
    assert all("模拟" in b for b in re.findall(r"<button[^>]*>(.*?)</button>", page))
    stale = form(app, cookie, "self/confirm", item.profile_id)
    service.edit_profile(user_id=PARTNER, profile_id=item.profile_id, text="合成外部修改")
    code, _, page = post(app, cookie, "self/confirm", stale)
    assert code == 409 and "刷新" in page
    values = form(app, cookie, "self/confirm", item.profile_id)
    assert post(app, cookie, "self/confirm", values | {"version": "1"})[0] == 409
    values = form(app, cookie, "self/confirm", item.profile_id) | {"use_in_chat": "yes"}
    assert post(app, cookie, "self/confirm", values)[0] == 200
    item = next(e for e in service.list_profile(PARTNER) if e.profile_id == item.profile_id)
    assert item.status == "confirmed" and item.simulated and item.use_in_chat
    assert item.confirmed_by == PARTNER and item.evidence_quote is None
    assert item.valid_until.isoformat() in request(app, SELF, cookie=cookie)[2]
    version = item.version
    values = form(app, cookie, "self/correct", item.profile_id) | {"text": "合成纠正内容"}
    assert post(app, cookie, "self/correct", values)[0] == 200
    item = next(e for e in service.list_profile(PARTNER) if e.profile_id == item.profile_id)
    assert item.text == "合成纠正内容" and item.version == version + 1
    assert item.confirmed_version == item.version and item.simulated and not item.use_in_chat
    assert (
        post(app, cookie, "self/delete", form(app, cookie, "self/delete", item.profile_id))[0]
        == 200
    )
    assert all(e.profile_id != item.profile_id for e in service.list_profile(PARTNER))


@pytest.mark.parametrize("action", ["consent", "self/revoke"])
def test_missing_consent_and_profile_only_revocation(tmp_path, action):
    app, cookie, created, _, _ = factory_app(tmp_path)
    values = form(app, cookie, "add") | {
        "topic": "饮水",
        "text": "合成内容",
        "category": "habit",
        "confidence": "low",
        "evidence_date": NOW.date().isoformat(),
        "evidence_quote": "",
    }
    code, _, page = post(app, cookie, "add", values)
    assert code == 403 and "未授予画像授权" in page
    service = created[0]
    service.grant_consent(
        ConsentRecord(
            user_id=PARTNER,
            granted=True,
            granted_at=NOW,
            allowed_fields=["profile", "explicit_note"],
        )
    )
    memory = remember(service, PARTNER, "合成普通记忆")
    item = approved(app, cookie, service)
    values = form(app, cookie, action)
    if action == "consent":
        values["mode"] = "revoke"
    assert post(app, cookie, action, values)[0] == 200
    assert service.list_profile(PARTNER) == ()
    assert service.list_memories(PARTNER) == (memory,)
    assert service.get_consent(PARTNER).allowed_fields == ["explicit_note"]
    assert item.profile_id not in request(app, ROOT, cookie=cookie)[2]


@pytest.mark.parametrize("fault", ["user_id", "other_target", "csrf", "cross_session", "replay"])
def test_owner_target_and_session_binding(tmp_path, fault):
    app, cookie, service = ready(tmp_path)
    item = add(app, cookie, service)
    service.grant_consent(
        ConsentRecord(user_id=OWNER, granted=True, granted_at=NOW, allowed_fields=["profile"])
    )
    other = service.add_profile_candidate(
        user_id=OWNER,
        category="habit",
        topic="合成",
        text="合成其他用户",
        source="owner_added",
        evidence_date=NOW.date(),
        confidence="low",
    )
    values = form(app, cookie, "approve", item.profile_id)
    submitted_cookie = cookie
    if fault == "user_id":
        values["user_id"] = OWNER
    elif fault == "other_target":
        values["profile_id"] = other.profile_id
    elif fault == "csrf":
        values["csrf"] = "forged"
    elif fault == "cross_session":
        submitted_cookie = unlock(app)
    else:
        assert post(app, cookie, "approve", values)[0] == 200
    assert post(app, submitted_cookie, "approve", values)[0] in {400, 403}
    assert service.list_profile(OWNER) == (other,)
    if fault not in {"replay", "cross_session"}:
        assert service.list_profile(PARTNER) == (item,)
        assert post(app, cookie, "approve", values)[0] == 403


def test_conflict_is_visible_and_home_link(tmp_path):
    app, cookie, service = ready(tmp_path)
    first = approved(app, cookie, service)
    assert (
        post(app, cookie, "self/confirm", form(app, cookie, "self/confirm", first.profile_id))[0]
        == 200
    )
    second = approved(app, cookie, service, text="合成爱喝水")
    assert (
        post(app, cookie, "self/confirm", form(app, cookie, "self/confirm", second.profile_id))[0]
        == 200
    )
    assert all(e.disabled_reason == "conflict" for e in service.list_profile(PARTNER))
    assert "冲突停用" in request(app, ROOT, cookie=cookie)[2]
    assert (
        '<a href="/admin/partner">对方画像（合成演示）</a>'  # noqa: RUF001
        in request(app, "/admin/home", cookie=cookie)[2]
    )


def test_readback_failure_is_unconfirmed_and_token_consumed(tmp_path, monkeypatch):
    app, cookie, service = ready(tmp_path)
    item = add(app, cookie, service)
    values = form(app, cookie, "approve", item.profile_id)
    original = service.approve_profile

    def write(**kwargs):
        result = original(**kwargs)
        monkeypatch.setattr(
            service, "list_profile", lambda *_: (_ for _ in ()).throw(OSError("SYNTHETIC_ERROR"))
        )
        return result

    monkeypatch.setattr(service, "approve_profile", write)
    code, _, page = post(app, cookie, "approve", values)
    assert code == 503 and "结果未确认" in page and "SYNTHETIC_ERROR" not in page
    assert post(app, cookie, "approve", values)[0] == 403


@pytest.mark.parametrize("action", ["consent", "self/revoke"])
@pytest.mark.parametrize("seed", [True, False])
def test_revoke_without_other_memory_fields(tmp_path, action, seed):
    app, cookie, created, _, _ = factory_app(tmp_path)
    if seed:
        values = form(app, cookie, "consent") | {"mode": "grant"}
        assert post(app, cookie, "consent", values)[0] == 200
        service = created[0]
        add(app, cookie, service)
    values = form(app, cookie, action)
    if action == "consent":
        values["mode"] = "revoke"
    assert post(app, cookie, action, values)[0] == 200
    if not seed:
        service = created[0]
    assert service.list_profile(PARTNER) == ()
    assert service.get_consent(PARTNER).granted is False


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("fault", ["csrf", "cross_session", "replay"])
def test_all_mutations_use_session_bound_one_time_tokens(tmp_path, action, fault):
    app, cookie, service = ready(tmp_path)
    item = add(app, cookie, service) if action == "approve" else approved(app, cookie, service)
    target = None if action in {"consent", "add", "self/revoke"} else item.profile_id
    values = form(app, cookie, action, target)
    if action == "consent":
        values["mode"] = "grant"
    elif action == "add":
        values |= {
            "topic": "合成",
            "text": "合成新增",
            "category": "preference",
            "evidence_date": NOW.date().isoformat(),
            "evidence_quote": "",
            "confidence": "low",
        }
    elif action == "edit":
        values |= {"topic": item.topic, "text": "合成编辑", "category": item.category}
    elif action == "self/correct":
        values["text"] = "合成纠正"
    before = service.list_profile(PARTNER)
    if fault == "csrf":
        original = dict(values)
        values["csrf"] = "wrong"
        assert post(app, cookie, action, values)[0] == 403
        assert post(app, cookie, action, original)[0] == 403
    elif fault == "cross_session":
        assert post(app, unlock(app), action, values)[0] == 403
    else:
        assert post(app, cookie, action, values)[0] == 200
        assert post(app, cookie, action, values)[0] == 403
    if fault != "replay":
        assert service.list_profile(PARTNER) == before


def test_configured_product_app_routes_are_lazy_and_share_memory_service(tmp_path):
    from companion.admin.app import build_admin_app, load_admin_config

    _, _, _, trusted, _ = factory_app(tmp_path)
    app = build_admin_app(trusted, load_admin_config(trusted), port=12345, wall_clock=lambda: NOW)
    cookie = unlock(app)
    assert request(app, "/admin/home", cookie=cookie)[0] == 200
    assert not (trusted.private_data_dir / "memory").exists()
    values = form(app, cookie, "consent") | {"mode": "grant"}
    assert post(app, cookie, "consent", values)[0] == 200
    item = approved(app, cookie, app._memory.service())
    values = form(app, cookie, "self/confirm", item.profile_id)
    assert post(app, cookie, "self/confirm", values)[0] == 200
    assert app._memory.service().list_profile(PARTNER)[0].simulated


def test_public_service_is_lazy_and_constructed_once_for_both_admin_pages(tmp_path):
    app, cookie, created, _, _ = factory_app(tmp_path)
    assert created == []
    service = app._memory.service()
    assert app._memory.service() is service
    assert request(app, ROOT, cookie=cookie)[0] == 200
    assert request(app, "/admin/memory", cookie=cookie)[0] == 200
    assert request(app, SELF, cookie=cookie)[0] == 200
    assert created == [service]


@pytest.mark.parametrize(
    "action", ["add", "approve", "edit", "disable", "self/confirm", "self/correct"]
)
def test_readback_does_not_predict_other_profile_changes(tmp_path, monkeypatch, action):
    app, cookie, service = ready(tmp_path)
    item = add(app, cookie, service) if action == "approve" else approved(app, cookie, service)
    sibling = approved(app, cookie, service, topic="作息")
    values = form(app, cookie, action, None if action == "add" else item.profile_id)
    if action == "add":
        values |= {
            "topic": "合成",
            "text": "合成新增",
            "category": "preference",
            "evidence_date": NOW.date().isoformat(),
            "evidence_quote": "",
            "confidence": "low",
        }
    elif action == "edit":
        values |= {"topic": item.topic, "text": "合成编辑", "category": item.category}
    elif action == "self/correct":
        values["text"] = "合成纠正"
    name = {
        "add": "add_profile_candidate",
        "approve": "approve_profile",
        "edit": "edit_profile",
        "disable": "disable_profile",
        "self/confirm": "confirm_profile",
        "self/correct": "confirm_profile",
    }[action]
    original = getattr(service, name)
    disable_sibling = service.disable_profile

    def write(**kwargs):
        result = original(**kwargs)
        disable_sibling(user_id=PARTNER, profile_id=sibling.profile_id, reason="owner")
        return result

    monkeypatch.setattr(service, name, write)
    assert post(app, cookie, action, values)[0] == 200
    assert (
        next(i for i in service.list_profile(PARTNER) if i.profile_id == sibling.profile_id).status
        == "disabled"
    )


def test_edit_readback_accepts_service_return_without_predicting_status(tmp_path, monkeypatch):
    app, cookie, service = ready(tmp_path)
    item = approved(app, cookie, service)
    values = form(app, cookie, "edit", item.profile_id) | {
        "topic": item.topic,
        "text": "合成编辑",
        "category": item.category,
    }
    original = service.edit_profile

    def write(**kwargs):
        original(**kwargs)
        return service.disable_profile(user_id=PARTNER, profile_id=item.profile_id, reason="owner")

    monkeypatch.setattr(service, "edit_profile", write)
    assert post(app, cookie, "edit", values)[0] == 200
    after = service.list_profile(PARTNER)[0]
    assert after.status == "disabled" and after.text == "合成编辑"


def test_return_matching_readback_still_requires_requested_fields(tmp_path, monkeypatch):
    app, cookie, service = ready(tmp_path)
    item = add(app, cookie, service)
    values = form(app, cookie, "approve", item.profile_id)
    monkeypatch.setattr(service, "approve_profile", lambda **_: item)
    code, _, page = post(app, cookie, "approve", values)
    assert code == 503 and "结果未确认" in page
    assert service.list_profile(PARTNER) == (item,)
    assert post(app, cookie, "approve", values)[0] == 403


def test_requested_fields_matching_still_requires_full_readback_equality(tmp_path, monkeypatch):
    app, cookie, service = ready(tmp_path)
    item = approved(app, cookie, service)
    values = form(app, cookie, "edit", item.profile_id) | {
        "topic": item.topic,
        "text": "合成编辑",
        "category": item.category,
    }
    original = service.edit_profile

    def write(**kwargs):
        result = original(**kwargs)
        return result.model_copy(update={"evidence_quote": "合成伪造返回"})

    monkeypatch.setattr(service, "edit_profile", write)
    code, _, page = post(app, cookie, "edit", values)
    assert code == 503 and "结果未确认" in page
    assert service.list_profile(PARTNER)[0].text == "合成编辑"
    assert post(app, cookie, "edit", values)[0] == 403
