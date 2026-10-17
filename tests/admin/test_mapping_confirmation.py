import re
from datetime import UTC, datetime, timedelta

import pytest
from test_local_entry import call

from companion.admin.app import AdminApp
from companion.contracts import SpeakerRole
from companion.importer import ImportBatch, MessageDirection, SourceChatRecord

ORIGIN = "http://127.0.0.1:12345"


def synthetic_batch() -> ImportBatch:
    return ImportBatch(
        records=[
            SourceChatRecord(
                conversation_id="synthetic-chat",
                message_id="m1",
                timestamp=datetime(2026, 9, 24, tzinfo=UTC),
                source_speaker_id="synthetic-owner",
                source_speaker_name="Owner",
                direction=MessageDirection.OUTBOUND,
                content_type="text",
                text="SYNTHETIC_PRIVATE_SENTINEL",
            ),
            SourceChatRecord(
                conversation_id="synthetic-chat",
                message_id="m2",
                timestamp=datetime(2026, 9, 24, tzinfo=UTC) + timedelta(minutes=1),
                source_speaker_id="synthetic-other",
                source_speaker_name="Other",
                direction=MessageDirection.INBOUND,
                content_type="text",
                text="synthetic reply",
            ),
        ],
        issues=[],
    )


def unlock(app: AdminApp) -> str:
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
    assert status == 303
    return headers["Set-Cookie"].split(";", 1)[0]


def preview_form(app: AdminApp, cookie: str) -> dict[str, str]:
    status, _, page = call(app, "/admin/import/preview", cookie=cookie)
    assert status == 200
    return {
        "csrf": re.search(r'name="csrf" value="([^"]+)"', page).group(1),
        "confirm_token": re.search(r'name="confirm_token" value="([^"]+)"', page).group(1),
        "confirm": "yes",
        "role_0": "other",
        "role_1": "owner",
    }


def confirm(app: AdminApp, cookie: str, form: dict[str, str]):
    return call(
        app, "/admin/import/confirm", method="POST", cookie=cookie, origin=ORIGIN, form=form
    )


def test_same_batch_explicit_mapping_succeeds_only_in_owner_session():
    current = [synthetic_batch()]
    app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345, batch_loader=lambda: current[0])
    cookie = unlock(app)
    form = preview_form(app, cookie)
    form["owner_id"] = "attacker"
    status, _, page = confirm(app, cookie, form)
    assert status == 200 and "映射已确认" in page and "风格逐条审批未就绪" in page
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in page
    session = app._sessions[cookie.split("=", 1)[1]]
    assert session.owner_id == "synthetic-owner"
    assert [record.speaker_role for record in session.confirmed_records] == [
        SpeakerRole.OWNER,
        SpeakerRole.OTHER,
    ]
    assert confirm(app, cookie, form)[0] == 403
    assert len(session.confirmed_records) == 2


@pytest.mark.parametrize("change", ["add", "delete", "text", "direction", "metadata", "order"])
def test_any_batch_change_invalidates_preview(change):
    current = [synthetic_batch()]
    app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345, batch_loader=lambda: current[0])
    cookie = unlock(app)
    form = preview_form(app, cookie)
    records = list(current[0].records)
    if change == "add":
        records.append(records[0].model_copy(update={"message_id": "m3"}))
    elif change == "delete":
        records.pop()
    elif change == "text":
        records[0] = records[0].model_copy(update={"text": "changed"})
    elif change == "direction":
        records[0] = records[0].model_copy(update={"direction": MessageDirection.INBOUND})
    elif change == "metadata":
        records[0] = records[0].model_copy(update={"source_speaker_name": "Changed"})
    else:
        records.reverse()
    current[0] = current[0].model_copy(update={"records": records})
    status, _, page = confirm(app, cookie, form)
    assert status == 409 and "SYNTHETIC_PRIVATE_SENTINEL" not in page
    assert app._sessions[cookie.split("=", 1)[1]].confirmed_records is None


@pytest.mark.parametrize(
    "changes",
    [
        {"role_0": "owner"},
        {"role_1": "other"},
        {"role_0": "unknown"},
        {"role_0": ""},
        {"role_1": None},
        {"role_2": "owner"},
        {"confirm": ""},
    ],
)
def test_incomplete_or_conflicting_roles_never_partially_confirm(changes):
    app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345, batch_loader=synthetic_batch)
    cookie = unlock(app)
    form = preview_form(app, cookie)
    for key, value in changes.items():
        if value is None:
            form.pop(key)
        else:
            form[key] = value
    status, _, page = confirm(app, cookie, form)
    assert status in {400, 403}
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in page
    assert app._sessions[cookie.split("=", 1)[1]].confirmed_records is None


def test_cross_session_token_and_loader_error_fail_closed():
    current = [synthetic_batch()]
    app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345, batch_loader=lambda: current[0])
    first = unlock(app)
    first_form = preview_form(app, first)
    second = unlock(app)
    second_form = preview_form(app, second)
    own_second_token = second_form["confirm_token"]
    second_form["confirm_token"] = first_form["confirm_token"]
    assert confirm(app, second, second_form)[0] == 403
    assert app._sessions[second.split("=", 1)[1]].confirmed_records is None
    second_form["confirm_token"] = own_second_token
    assert confirm(app, second, second_form)[0] == 403
    current[0] = None
    status, _, page = confirm(app, first, first_form)
    assert status == 503 and "SYNTHETIC_PRIVATE_SENTINEL" not in page
    assert app._sessions[first.split("=", 1)[1]].confirmed_records is None


def test_non_owner_expired_and_closed_sessions_cannot_confirm():
    now = [100.0]
    app = AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        clock=lambda: now[0],
        batch_loader=synthetic_batch,
    )
    cookie = unlock(app)
    form = preview_form(app, cookie)
    session = app._sessions[cookie.split("=", 1)[1]]
    session.owner_id = "synthetic-other"
    assert confirm(app, cookie, form)[0] == 403
    assert session.confirmed_records is None
    session.owner_id = "synthetic-owner"
    now[0] += 901
    assert confirm(app, cookie, form)[0] == 403
    assert session.confirmed_records is None
    app.close()
    assert confirm(app, cookie, form)[0] == 403


def test_consent_remains_unready_after_mapping_confirmation():
    app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345, batch_loader=synthetic_batch)
    cookie = unlock(app)
    form = preview_form(app, cookie)
    assert confirm(app, cookie, form)[0] == 200
    session = app._sessions[cookie.split("=", 1)[1]]
    consent_csrf = app._token(session, "consent")
    assert (
        call(
            app,
            "/admin/style/consent",
            method="POST",
            cookie=cookie,
            origin=ORIGIN,
            form={"csrf": consent_csrf, "owner_id": "attacker"},
        )[0]
        == 503
    )
