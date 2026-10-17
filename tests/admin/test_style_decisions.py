import re
from datetime import UTC, datetime, timedelta

import pytest
from test_candidate_review_handoff import candidate_package
from test_local_entry import call, settings
from test_mapping_confirmation import confirm, preview_form, synthetic_batch, unlock

from companion.admin.app import AdminApp
from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.privacy.style_candidate_approval import StyleCandidateApprovalStore

ORIGIN = "http://127.0.0.1:12345"


def two_candidates(created_at=None):
    now = created_at or datetime(2026, 9, 24, tzinfo=UTC)
    return SafeCandidatePackage(
        created_at=now,
        candidates=[
            SafeStyleCandidate(
                text="<b>synthetic one</b>",
                category=CandidateCategory.GREETING,
                scene_tags=["greeting"],
            ),
            SafeStyleCandidate(
                text="synthetic two",
                category=CandidateCategory.GREETING,
                scene_tags=["greeting"],
            ),
        ],
        audit=CandidateSelectionAudit(
            processed_at=now,
            input_count=2,
            owner_input_count=2,
            dropped_other_count=0,
            allowed_category_counts={"greeting": 2},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=2,
            succeeded=True,
        ),
    )


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    return StyleCandidateApprovalStore(settings(root))


def ready_app(reader, store, *, source=None):
    source = source if source is not None else [synthetic_batch()]
    app = AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        batch_loader=lambda: source[0],
        candidate_reader=reader,
        approval_store=store,
    )
    cookie = unlock(app)
    assert confirm(app, cookie, preview_form(app, cookie))[0] == 200
    return app, cookie, source


def review(app, cookie):
    status, _, page = call(app, "/admin/style/review", cookie=cookie)
    token = re.search(r'name="decide_token" value="([^"]+)"', page)
    csrf = re.search(r'name="csrf" value="([^"]+)"', page)
    return status, page, (token.group(1), csrf.group(1)) if token and csrf else None


def decide(app, cookie, token, csrf, *, index="1", decision="approve", **extra):
    form = {"decide_token": token, "csrf": csrf, "index": index, "decision": decision}
    form.update(extra)
    return call(app, "/admin/style/decide", method="POST", cookie=cookie, origin=ORIGIN, form=form)


def test_success_returns_to_same_candidate_with_fresh_tokens(store):
    package = two_candidates()
    app, cookie, _ = ready_app(lambda _owner, _records: package, store)
    _, _, tokens = review(app, cookie)
    status, headers, _ = decide(app, cookie, *tokens, index="0")
    assert status == 303
    assert headers["Location"] == "/admin/style/review#candidate-0"
    status, page, fresh = review(app, cookie)
    assert status == 200 and '<li id="candidate-0">' in page
    assert '<li id="candidate-1">' in page and "approved" in page
    assert fresh != tokens
    assert decide(app, cookie, *tokens, index="1")[0] == 403
    _, _, fresh = review(app, cookie)
    assert decide(app, cookie, *fresh, index="1")[0] == 303
    assert store.decisions("synthetic-owner", package) == ((0, "approved"), (1, "approved"))


def test_real_store_pending_approve_revoke_and_frozen_package(store):
    calls = []
    first = two_candidates()

    def reader(owner, records):
        calls.append((owner, records))
        return first if len(calls) == 1 else two_candidates(first.created_at + timedelta(seconds=1))

    app, cookie, _ = ready_app(reader, store)
    status, page, tokens = review(app, cookie)
    assert status == 200 and page.count("pending") == 2
    assert "&lt;b&gt;synthetic one&lt;/b&gt;" in page and "<b>" not in page
    assert len(calls) == 1
    token, csrf = tokens
    status, _, page = decide(
        app,
        cookie,
        token,
        csrf,
        owner_id="forged-owner",
        candidate_text="forged text",
        package_digest="forged-digest",
    )
    assert status == 303 and "approved" in page
    assert store.decisions("synthetic-owner", first) == ((0, "pending"), (1, "approved"))
    status, page, tokens = review(app, cookie)
    assert status == 200 and "approved" in page and len(calls) == 1
    token, csrf = tokens
    status, _, page = decide(app, cookie, token, csrf, decision="revoke")
    assert status == 303 and "pending" in page
    assert store.decisions("synthetic-owner", first) == ((0, "pending"), (1, "pending"))


def test_cross_session_replay_and_wrong_action_cannot_write(store):
    package = two_candidates()
    app, first, _ = ready_app(lambda _owner, _records: package, store)
    assert review(app, first)[0] == 200
    _, _, (first_token, first_csrf) = review(app, first)
    second = unlock(app)
    assert confirm(app, second, preview_form(app, second))[0] == 200
    _, _, (_, second_csrf) = review(app, second)
    assert decide(app, second, first_token, second_csrf)[0] == 403
    assert store.decisions("synthetic-owner", package) == ((0, "pending"), (1, "pending"))
    assert (
        call(
            app,
            "/admin/style/consent",
            method="POST",
            cookie=first,
            origin=ORIGIN,
            form={"csrf": first_csrf, "decide_token": first_token},
        )[0]
        == 403
    )
    assert decide(app, first, first_token, first_csrf)[0] == 303
    assert decide(app, first, first_token, first_csrf)[0] == 403


@pytest.mark.parametrize("change", ["source", "repreview", "close", "expired", "package"])
def test_stale_or_tampered_session_cannot_write(store, change):
    package = two_candidates()
    source = [synthetic_batch()]
    app, cookie, _ = ready_app(lambda _owner, _records: package, store, source=source)
    _, _, (token, csrf) = review(app, cookie)
    if change == "source":
        source[0] = source[0].model_copy(
            update={
                "records": [
                    source[0].records[0].model_copy(update={"text": "changed"}),
                    source[0].records[1],
                ]
            }
        )
    elif change == "repreview":
        assert call(app, "/admin/import/preview", cookie=cookie)[0] == 200
    elif change == "close":
        app.close()
    elif change == "expired":
        app._sessions[cookie.split("=", 1)[1]].expires = 0
    else:
        session = app._sessions[cookie.split("=", 1)[1]]
        session.candidate_package_bytes = b"tampered"
    status, _, page = decide(app, cookie, token, csrf)
    assert status in {403, 503} and "approved" not in page
    assert store.decisions("synthetic-owner", package) == ((0, "pending"), (1, "pending"))


@pytest.mark.parametrize(
    "index,decision", [("01", "approve"), ("-1", "approve"), ("2", "approve"), ("0", "deny")]
)
def test_bad_index_or_action_cannot_write(store, index, decision):
    package = two_candidates()
    app, cookie, _ = ready_app(lambda _owner, _records: package, store)
    _, _, (token, csrf) = review(app, cookie)
    assert decide(app, cookie, token, csrf, index=index, decision=decision)[0] in {400, 403}
    assert store.decisions("synthetic-owner", package) == ((0, "pending"), (1, "pending"))


def test_store_errors_and_readback_fail_closed(store, monkeypatch):
    package = two_candidates()
    app, cookie, _ = ready_app(lambda _owner, _records: package, store)
    _, _, (token, csrf) = review(app, cookie)
    monkeypatch.setattr(
        store,
        "set_approved",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("PRIVATE_SENTINEL")),
    )
    status, _, page = decide(app, cookie, token, csrf)
    assert status == 503 and "PRIVATE_SENTINEL" not in page and "approved" not in page
    assert store.decisions("synthetic-owner", package) == ((0, "pending"), (1, "pending"))


def test_store_readback_error_does_not_render_success(store, monkeypatch):
    package = two_candidates()
    app, cookie, _ = ready_app(lambda _owner, _records: package, store)
    _, _, (token, csrf) = review(app, cookie)

    def readback_error(*args, **kwargs):
        raise RuntimeError("PRIVATE_SENTINEL")

    monkeypatch.setattr(store, "decisions", readback_error)
    status, _, page = decide(app, cookie, token, csrf)
    assert status == 503
    assert "PRIVATE_SENTINEL" not in page and "approved" not in page


def test_zero_candidates_and_missing_store_or_reader(store):
    empty = candidate_package(empty=True)
    app, cookie, _ = ready_app(lambda _owner, _records: empty, store)
    status, page, tokens = review(app, cookie)
    assert status == 200 and "<ul></ul>" in page
    token, csrf = tokens
    assert decide(app, cookie, token, csrf, index="0")[0] in {400, 403}
    no_store, cookie, _ = ready_app(lambda _owner, _records: two_candidates(), None)
    no_store_session = no_store._sessions[cookie.split("=", 1)[1]]
    assert (
        call(
            no_store,
            "/admin/style/decide",
            method="POST",
            cookie=cookie,
            origin=ORIGIN,
            form={"csrf": no_store._token(no_store_session, "decide")},
        )[0]
        == 503
    )
    no_reader, cookie, _ = ready_app(None, store)
    no_reader_session = no_reader._sessions[cookie.split("=", 1)[1]]
    assert (
        call(
            no_reader,
            "/admin/style/decide",
            method="POST",
            cookie=cookie,
            origin=ORIGIN,
            form={"csrf": no_reader._token(no_reader_session, "decide")},
        )[0]
        == 503
    )
