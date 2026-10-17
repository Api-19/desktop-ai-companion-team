import pytest
from test_local_entry import call
from test_mapping_confirmation import ORIGIN, unlock

from companion.admin.app import AdminApp


def logged_in():
    now = [100.0]
    app = AdminApp("synthetic-owner", host="127.0.0.1", port=12345, clock=lambda: now[0])
    return app, unlock(app), now


def test_activity_every_ten_minutes_keeps_session_for_one_hour():
    app, cookie, now = logged_in()
    for _ in range(6):
        now[0] += 600
        assert call(app, "/admin/home", cookie=cookie)[0] == 200


@pytest.mark.parametrize("idle", [899, 900, 901])
def test_idle_expiration_boundary(idle):
    app, cookie, now = logged_in()
    now[0] += idle
    assert call(app, "/admin/home", cookie=cookie)[0] == (200 if idle < 900 else 403)


@pytest.mark.parametrize("elapsed", [7200, 7201])
def test_activity_cannot_extend_absolute_expiration(elapsed):
    app, cookie, now = logged_in()
    for offset in range(600, 7200, 600):
        now[0] = 100 + offset
        assert call(app, "/admin/home", cookie=cookie)[0] == 200
    now[0] = 100 + 7199
    assert call(app, "/admin/home", cookie=cookie)[0] == 200
    now[0] = 100 + elapsed
    assert call(app, "/admin/home", cookie=cookie)[0] == 403


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_authenticated_unknown_path_renews_session(method):
    app, cookie, now = logged_in()
    now[0] += 600
    assert call(app, "/admin/unknown", method=method, cookie=cookie, origin=ORIGIN)[0] == 404
    now[0] += 600
    assert call(app, "/admin/home", cookie=cookie)[0] == 200


def test_request_rejected_before_authentication_does_not_renew_session():
    app, cookie, now = logged_in()
    now[0] += 600
    assert call(app, "/admin/home", cookie=cookie, host="evil.test")[0] == 403
    now[0] += 300
    assert call(app, "/admin/home", cookie=cookie)[0] == 403
