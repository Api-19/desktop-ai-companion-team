import re
from datetime import UTC, datetime

import pytest
from test_fixed_import_source import private_source
from test_local_entry import call, settings
from test_mapping_confirmation import confirm, preview_form, unlock
from test_style_decisions import decide, review

import companion.admin.__main__ as admin_main
from companion.admin.app import AdminApp, build_admin_app, load_admin_config
from companion.contracts import SpeakerRole
from companion.importer import ImportBatch, MessageDirection, SourceChatRecord
from companion.privacy import select_safe_candidates
from companion.privacy.style_candidate_approval import (
    APPROVAL_RELATIVE_PATH,
    StyleCandidateApprovalStore,
)

ORIGIN = "http://127.0.0.1:12345"


def batch():
    return ImportBatch(
        records=[
            SourceChatRecord(
                conversation_id="synthetic-chat",
                message_id="m1",
                timestamp=datetime(2026, 9, 24, tzinfo=UTC),
                source_speaker_id="owner",
                source_speaker_name="Owner",
                direction=MessageDirection.OUTBOUND,
                content_type="text",
                text="早安",
            ),
            SourceChatRecord(
                conversation_id="synthetic-chat",
                message_id="m2",
                timestamp=datetime(2026, 9, 24, tzinfo=UTC),
                source_speaker_id="other",
                source_speaker_name="Other",
                direction=MessageDirection.INBOUND,
                content_type="text",
                text="辛苦了",
            ),
        ],
        issues=[],
    )


def ready(tmp_path, *, selector=select_safe_candidates):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    source = [batch()]
    calls = []

    def counted(records, **kwargs):
        calls.append((records, kwargs))
        return selector(records, **kwargs)

    store = StyleCandidateApprovalStore(settings(root))
    app = AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        batch_loader=lambda: source[0],
        runtime_selector=counted,
        approval_store=store,
    )
    cookie = unlock(app)
    assert confirm(app, cookie, preview_form(app, cookie))[0] == 200
    return app, cookie, source, calls, store


def terms_form(app, cookie):
    status, _, page = call(app, "/admin/style/terms", cookie=cookie)
    assert status == 200
    return {
        "csrf": re.search(r'name="csrf" value="([^"]+)"', page).group(1),
        "terms_token": re.search(r'name="terms_token" value="([^"]+)"', page).group(1),
        "true_name_terms": "示例姓名甲",
        "approved_nicknames": "",
    }


def submit(app, cookie, form, *, origin=ORIGIN):
    return call(app, "/admin/style/terms", method="POST", cookie=cookie, origin=origin, form=form)


def test_runtime_terms_real_selector_owner_only_pending_and_explicit_decision(tmp_path):
    app, cookie, _, calls, store = ready(tmp_path)
    assert review(app, cookie)[0] == 503
    form = terms_form(app, cookie)
    form["approved_nicknames"] = "小团子"
    form["confirm_nicknames"] = "yes"
    assert submit(app, cookie, form)[0] == 200
    assert submit(app, cookie, form)[0] == 403
    status, page, _ = review(app, cookie)
    assert status == 200 and "早安" in page and "辛苦了" not in page
    assert "pending" in page and "示例姓名甲" not in page and "小团子" not in page
    assert len(calls) == 1
    assert calls[0][1]["true_name_terms"] == ("示例姓名甲",)
    assert calls[0][1]["approved_nicknames"] == ("小团子",)
    assert calls[0][1]["processed_at"].tzinfo is not None
    assert calls[0][0][0].speaker_role is SpeakerRole.OWNER
    assert calls[0][0][1].speaker_role is SpeakerRole.OTHER
    session = app._sessions[cookie.split("=", 1)[1]]
    assert session.runtime_terms is None
    assert review(app, cookie)[0] == 200 and len(calls) == 1
    assert decide(app, cookie, *review(app, cookie)[2], index="0")[0] == 303
    package = select_safe_candidates(
        calls[0][0],
        processed_at=calls[0][1]["processed_at"],
        true_name_terms=("示例姓名甲",),
        approved_nicknames=("小团子",),
    )
    assert store.decisions("synthetic-owner", package) == ((0, "approved"),)
    sidecar = (store.private_data_dir / APPROVAL_RELATIVE_PATH).read_bytes()
    assert "示例姓名甲".encode() not in sidecar
    assert "小团子".encode() not in sidecar


@pytest.mark.parametrize(
    "change",
    [
        {"true_name_terms": ""},
        {"true_name_terms": "a" * 65},
        {"true_name_terms": "甲\n甲"},
        {"true_name_terms": "甲\x00乙"},
        {"true_name_terms": "甲\ufffd"},
        {"true_name_terms": "甲\n" * 9},
        {"approved_nicknames": "小团子"},
        {"approved_nicknames": "小团子", "confirm_nicknames": "no"},
        {"unexpected": "value"},
    ],
)
def test_invalid_terms_never_select_or_write(tmp_path, change):
    app, cookie, _, calls, _ = ready(tmp_path)
    form = terms_form(app, cookie)
    form.update(change)
    assert submit(app, cookie, form)[0] == 400
    assert review(app, cookie)[0] == 503 and not calls


def test_cross_session_origin_csrf_source_drift_and_repreview(tmp_path):
    app, cookie, source, calls, _ = ready(tmp_path)
    form = terms_form(app, cookie)
    second = unlock(app)
    assert submit(app, second, form)[0] == 403
    assert submit(app, cookie, form, origin="http://elsewhere")[0] == 403
    assert submit(app, cookie, {**form, "csrf": "bad"})[0] == 403
    assert submit(app, cookie, form)[0] == 200
    source[0] = batch().model_copy(deep=True)
    source[0].records[0].text = "晚安"
    assert review(app, cookie)[0] == 503 and not calls
    assert submit(app, cookie, form)[0] == 403
    assert confirm(app, cookie, preview_form(app, cookie))[0] == 200
    form = terms_form(app, cookie)
    assert submit(app, cookie, form)[0] == 200
    assert call(app, "/admin/import/preview", cookie=cookie)[0] == 200
    assert review(app, cookie)[0] == 503 and not calls


def test_review_without_terms_does_not_read_batch_source(tmp_path):
    app, cookie, _, calls, _ = ready(tmp_path)
    reads = []

    def unexpected():
        reads.append(True)
        raise ValueError("synthetic source should stay unread")

    app._batch_loader = unexpected
    assert review(app, cookie)[0] == 503
    assert not reads and not calls


def test_selector_failure_clears_terms_and_does_not_echo(tmp_path):
    def failure(*_args, **_kwargs):
        raise ValueError("示例姓名甲 synthetic selector failure")

    app, cookie, _, calls, _ = ready(tmp_path, selector=failure)
    form = terms_form(app, cookie)
    assert submit(app, cookie, form)[0] == 200
    status, page, _ = review(app, cookie)
    assert status == 503 and "示例姓名甲" not in page and "synthetic selector" not in page
    assert len(calls) == 1
    assert app._sessions[cookie.split("=", 1)[1]].runtime_terms is None
    assert review(app, cookie)[0] == 503 and len(calls) == 1


def test_resubmitted_terms_invalidate_old_package_and_sidecar_fails_closed(tmp_path):
    app, cookie, _, calls, store = ready(tmp_path)
    form = terms_form(app, cookie)
    assert submit(app, cookie, form)[0] == 200
    assert review(app, cookie)[0] == 200
    assert decide(app, cookie, *review(app, cookie)[2], index="0")[0] == 303
    first = select_safe_candidates(
        calls[0][0],
        processed_at=calls[0][1]["processed_at"],
        true_name_terms=("示例姓名甲",),
        approved_nicknames=(),
    )
    assert store.decisions("synthetic-owner", first) == ((0, "approved"),)
    form = terms_form(app, cookie)
    form["true_name_terms"] = "示例姓名乙"
    assert submit(app, cookie, form)[0] == 200
    assert review(app, cookie)[0] == 503
    assert len(calls) == 2


def test_expired_or_closed_session_cannot_submit_terms(tmp_path):
    app, cookie, _, calls, _ = ready(tmp_path)
    form = terms_form(app, cookie)
    session = app._sessions[cookie.split("=", 1)[1]]
    session.expires = app._clock() - 1
    assert submit(app, cookie, form)[0] == 403
    assert not calls
    app.close()
    assert submit(app, cookie, form)[0] == 403


def test_product_builder_wires_real_selector_and_store_with_synthetic_file(tmp_path):
    root, _ = private_source(tmp_path)
    trusted = settings(root)
    app = build_admin_app(trusted, load_admin_config(trusted), port=12345)
    cookie = unlock(app)
    assert confirm(app, cookie, preview_form(app, cookie))[0] == 200
    assert review(app, cookie)[0] == 503
    form = terms_form(app, cookie)
    assert submit(app, cookie, form)[0] == 200
    assert review(app, cookie)[0] == 200


def test_main_assembles_selector_and_store_without_real_listener(tmp_path, monkeypatch):
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
    assert len(captured) == 1
    assert captured[0]._runtime_selector is select_safe_candidates
    assert isinstance(captured[0]._approval_store, StyleCandidateApprovalStore)
    assert captured[0]._closed is True
