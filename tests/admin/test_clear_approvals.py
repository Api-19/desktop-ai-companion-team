import re
from datetime import UTC, datetime, timedelta

import pytest
from test_candidate_review_handoff import candidate_package
from test_local_entry import call, settings
from test_mapping_confirmation import synthetic_batch, unlock
from test_runtime_terms import ready as runtime_ready
from test_runtime_terms import submit, terms_form
from test_style_decisions import decide, ready_app, review, two_candidates

from companion.privacy.style_candidate_approval import (
    APPROVAL_RELATIVE_PATH,
    StyleCandidateApprovalStore,
)

ORIGIN = "http://127.0.0.1:12345"


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    return StyleCandidateApprovalStore(settings(root))


def clear_form(app, cookie):
    status, _, page = call(app, "/admin/style/clear", cookie=cookie)
    token = re.search(r'name="clear_token" value="([^"]+)"', page)
    csrf = re.search(r'name="csrf" value="([^"]+)"', page)
    return (
        status,
        page,
        (
            {"clear_token": token.group(1), "csrf": csrf.group(1), "confirm": "yes"}
            if token and csrf
            else None
        ),
    )


def clear(app, cookie, form, *, origin=ORIGIN, host="127.0.0.1:12345"):
    return call(
        app,
        "/admin/style/clear",
        method="POST",
        cookie=cookie,
        origin=origin,
        host=host,
        form=form,
    )


def test_stale_sidecar_clear_then_pending_then_reapprove(store):
    old = two_candidates(datetime(2026, 9, 24, tzinfo=UTC))
    current = two_candidates(old.created_at + timedelta(seconds=1))
    store.set_approved("synthetic-owner", old, index=0, approved=True)
    app, cookie, _ = ready_app(lambda _owner, _records: current, store)
    status, page, _ = review(app, cookie)
    assert status == 503 and 'href="/admin/style/clear"' in page
    assert "privacy/style" not in page and "package_digest" not in page
    status, page, form = clear_form(app, cookie)
    assert status == 200 and form is not None
    assert "所有" in page and "不会自动批准" in page
    assert "privacy/style" not in page and "package_digest" not in page
    assert clear(app, cookie, {**form, "owner_id": "forged-owner"})[0] == 400
    _, _, form = clear_form(app, cookie)
    status, _, page = clear(app, cookie, form)
    assert status == 200 and "已清除" in page
    assert clear(app, cookie, form)[0] == 403
    assert store.decisions("synthetic-owner", current) == ((0, "pending"), (1, "pending"))
    status, page, tokens = review(app, cookie)
    assert status == 200 and page.count("pending") == 2
    assert decide(app, cookie, *tokens, index="0")[0] == 303
    assert store.decisions("synthetic-owner", current) == ((0, "approved"), (1, "pending"))


def test_current_sidecar_requires_explicit_confirm_and_clears_all(store):
    package = two_candidates()
    store.set_approved("synthetic-owner", package, index=0, approved=True)
    store.set_approved("synthetic-owner", package, index=1, approved=True)
    app, cookie, _ = ready_app(lambda _owner, _records: package, store)
    assert review(app, cookie)[0] == 200
    status, _, form = clear_form(app, cookie)
    assert status == 200
    assert clear(app, cookie, {**form, "confirm": "no"})[0] == 400
    assert store.decisions("synthetic-owner", package) == ((0, "approved"), (1, "approved"))
    _, _, form = clear_form(app, cookie)
    assert clear(app, cookie, form)[0] == 200
    assert store.decisions("synthetic-owner", package) == ((0, "pending"), (1, "pending"))


@pytest.mark.parametrize("token_timing", ["before_clear_page", "after_clear_page"])
def test_clear_revokes_prior_decide_token_until_fresh_review(store, token_timing):
    package = two_candidates()
    store.set_approved("synthetic-owner", package, index=0, approved=True)
    app, cookie, _ = ready_app(lambda _owner, _records: package, store)
    if token_timing == "before_clear_page":
        _, _, old_decide = review(app, cookie)
        _, _, form = clear_form(app, cookie)
    else:
        assert review(app, cookie)[0] == 200
        _, _, form = clear_form(app, cookie)
        _, _, old_decide = review(app, cookie)
    assert clear(app, cookie, form)[0] == 200
    assert decide(app, cookie, *old_decide, index="0")[0] == 403
    assert store.decisions("synthetic-owner", package) == ((0, "pending"), (1, "pending"))
    _, _, fresh_decide = review(app, cookie)
    assert decide(app, cookie, *fresh_decide, index="0")[0] == 303
    assert store.decisions("synthetic-owner", package) == ((0, "approved"), (1, "pending"))


def test_cross_session_origin_host_csrf_and_replay_cannot_delete(store):
    package = two_candidates()
    store.set_approved("synthetic-owner", package, index=0, approved=True)
    app, cookie, _ = ready_app(lambda _owner, _records: package, store)
    assert review(app, cookie)[0] == 200
    _, _, form = clear_form(app, cookie)
    other = unlock(app)
    assert clear(app, other, form)[0] == 403
    assert clear(app, cookie, form, origin="http://elsewhere")[0] == 403
    assert clear(app, cookie, form, host="localhost:12345")[0] == 403
    assert clear(app, cookie, {**form, "csrf": "wrong"})[0] == 403
    assert store.decisions("synthetic-owner", package)[0] == (0, "approved")
    assert clear(app, cookie, form)[0] == 200
    assert clear(app, cookie, form)[0] == 403


@pytest.mark.parametrize("change", ["missing_package", "source", "expired", "close", "owner"])
def test_invalid_context_never_deletes(store, change):
    package = two_candidates()
    store.set_approved("synthetic-owner", package, index=0, approved=True)
    source = [synthetic_batch()]
    app, cookie, _ = ready_app(lambda _owner, _records: package, store, source=source)
    if change == "missing_package":
        assert clear_form(app, cookie)[0] == 503
        assert store.decisions("synthetic-owner", package)[0] == (0, "approved")
        return
    assert review(app, cookie)[0] == 200
    _, _, form = clear_form(app, cookie)
    if change == "source":
        source[0] = source[0].model_copy(deep=True)
        source[0].records[0].text = "changed"
    elif change == "expired":
        app._sessions[cookie.split("=", 1)[1]].expires = 0
    elif change == "close":
        app.close()
    else:
        app._sessions[cookie.split("=", 1)[1]].owner_id = "forged-owner"
    assert clear(app, cookie, form)[0] in {403, 503}
    assert store.decisions("synthetic-owner", package)[0] == (0, "approved")


def test_zero_candidates_and_store_failures_fail_closed(store, monkeypatch):
    old = two_candidates()
    store.set_approved("synthetic-owner", old, index=0, approved=True)
    empty = candidate_package(empty=True)
    app, cookie, _ = ready_app(lambda _owner, _records: empty, store)
    assert review(app, cookie)[0] == 503
    _, _, form = clear_form(app, cookie)
    assert clear(app, cookie, form)[0] == 200
    assert store.decisions("synthetic-owner", empty) == ()
    assert review(app, cookie)[0] == 200

    _, _, form = clear_form(app, cookie)
    original_delete = store.delete_stale
    monkeypatch.setattr(
        store,
        "delete_stale",
        lambda _owner: (_ for _ in ()).throw(RuntimeError("synthetic private failure")),
    )
    status, _, page = clear(app, cookie, form)
    assert status == 503 and "synthetic private failure" not in page
    monkeypatch.setattr(store, "delete_stale", original_delete)
    _, _, form = clear_form(app, cookie)
    monkeypatch.setattr(store, "decisions", lambda _owner, _package: ((0, "pending"),))
    assert clear(app, cookie, form)[0] == 503


def test_owner_mismatch_sidecar_stays_and_repreview_invalidates_clear_token(store):
    package = two_candidates()
    store.set_approved("forged-owner", package, index=0, approved=True)
    app, cookie, _ = ready_app(lambda _owner, _records: package, store)
    assert review(app, cookie)[0] == 503
    _, _, form = clear_form(app, cookie)
    assert clear(app, cookie, form)[0] == 503
    assert store.decisions("forged-owner", package)[0] == (0, "approved")
    _, _, form = clear_form(app, cookie)
    assert call(app, "/admin/import/preview", cookie=cookie)[0] == 200
    assert clear(app, cookie, form)[0] in {403, 503}
    assert (store.private_data_dir / APPROVAL_RELATIVE_PATH).exists()


def test_resubmitted_terms_invalidate_prior_clear_token(tmp_path):
    app, cookie, _, _, _ = runtime_ready(tmp_path)
    assert submit(app, cookie, terms_form(app, cookie))[0] == 200
    assert review(app, cookie)[0] == 200
    _, _, old_form = clear_form(app, cookie)
    assert submit(app, cookie, terms_form(app, cookie))[0] == 200
    assert review(app, cookie)[0] == 200
    assert clear(app, cookie, old_form)[0] == 403


def test_clear_page_rechecks_source_and_requires_store(store):
    package = two_candidates()
    source = [synthetic_batch()]
    app, cookie, _ = ready_app(lambda _owner, _records: package, store, source=source)
    assert review(app, cookie)[0] == 200
    source[0] = source[0].model_copy(deep=True)
    source[0].records[0].text = "changed"
    assert clear_form(app, cookie)[0] == 503
    assert app._sessions[cookie.split("=", 1)[1]].candidate_package_bytes is None

    no_store, other, _ = ready_app(lambda _owner, _records: package, None)
    assert review(no_store, other)[0] == 200
    assert clear_form(no_store, other)[0] == 503
