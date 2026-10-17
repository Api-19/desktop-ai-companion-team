from datetime import UTC, datetime

import pytest
from test_local_entry import call
from test_mapping_confirmation import confirm, preview_form, synthetic_batch, unlock

from companion.admin.app import AdminApp
from companion.contracts import SpeakerRole
from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)


def candidate_package(*, text="<script>synthetic candidate</script>", empty=False):
    now = datetime(2026, 9, 24, tzinfo=UTC)
    candidates = (
        []
        if empty
        else [
            SafeStyleCandidate(
                text=text,
                category=CandidateCategory.GREETING,
                scene_tags=[CandidateCategory.GREETING.value],
            )
        ]
    )
    count = len(candidates)
    return SafeCandidatePackage(
        created_at=now,
        candidates=candidates,
        audit=CandidateSelectionAudit(
            processed_at=now,
            input_count=count,
            owner_input_count=count,
            dropped_other_count=0,
            allowed_category_counts={"greeting": count} if count else {},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=count,
            succeeded=True,
        ),
    )


def confirmed_app(reader, *, source=None, clock=None):
    source = source if source is not None else [synthetic_batch()]
    kwargs = {} if clock is None else {"clock": clock}
    app = AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        batch_loader=lambda: source[0],
        candidate_reader=reader,
        **kwargs,
    )
    cookie = unlock(app)
    form = preview_form(app, cookie)
    assert confirm(app, cookie, form)[0] == 200
    return app, cookie, source


def test_confirmed_owner_preview_calls_reader_once_and_escapes_candidate():
    calls = []

    def reader(owner_id, records):
        calls.append((owner_id, records))
        return candidate_package()

    app, cookie, _ = confirmed_app(reader)
    status, _, page = call(app, "/admin/style/review", cookie=cookie)
    assert status == 200
    assert len(calls) == 1
    assert calls[0][0] == "synthetic-owner"
    assert type(calls[0][1]) is tuple
    assert [record.speaker_role for record in calls[0][1]] == [SpeakerRole.OWNER, SpeakerRole.OTHER]
    assert "&lt;script&gt;synthetic candidate&lt;/script&gt;" in page
    assert "<script>" not in page
    assert "greeting" in page and "0" in page
    assert "synthetic-chat" not in page


def test_missing_confirmation_and_missing_reader_stay_unready():
    calls = []

    def reader(owner_id, records):
        calls.append((owner_id, records))
        return candidate_package()

    app = AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        batch_loader=synthetic_batch,
        candidate_reader=reader,
    )
    cookie = unlock(app)
    assert call(app, "/admin/style/review", cookie=cookie)[0] == 503
    assert calls == []
    no_reader, cookie, _ = confirmed_app(None)
    assert call(no_reader, "/admin/style/review", cookie=cookie)[0] == 503


@pytest.mark.parametrize("change", ["text", "source_error"])
def test_source_change_or_error_clears_confirmation_without_reader_call(change):
    calls = []

    def reader(owner_id, records):
        calls.append((owner_id, records))
        return candidate_package()

    app, cookie, source = confirmed_app(reader)
    if change == "text":
        records = list(source[0].records)
        records[0] = records[0].model_copy(update={"text": "changed"})
        source[0] = source[0].model_copy(update={"records": records})
    else:
        source[0] = None
    status, _, page = call(app, "/admin/style/review", cookie=cookie)
    assert status == 503 and "synthetic candidate" not in page
    assert calls == []
    session = app._sessions[cookie.split("=", 1)[1]]
    assert session.confirmed_records is None
    assert session.confirmed_digest is None


def test_repreview_and_close_invalidate_confirmation():
    calls = []

    def reader(owner_id, records):
        calls.append((owner_id, records))
        return candidate_package()

    app, cookie, _ = confirmed_app(reader)
    assert call(app, "/admin/import/preview", cookie=cookie)[0] == 200
    assert call(app, "/admin/style/review", cookie=cookie)[0] == 503
    assert calls == []
    app.close()
    assert call(app, "/admin/style/review", cookie=cookie)[0] == 403


def test_bad_records_do_not_reach_reader():
    calls = []

    def reader(owner_id, records):
        calls.append((owner_id, records))
        return candidate_package()

    app, cookie, _ = confirmed_app(reader)
    session = app._sessions[cookie.split("=", 1)[1]]
    bad_record = session.confirmed_records[0].model_copy(update={"text": "tampered"})
    session.confirmed_records = (bad_record,)
    assert call(app, "/admin/style/review", cookie=cookie)[0] == 503
    assert calls == []


def test_missing_loader_after_confirmation_clears_state_before_callback():
    calls = []

    def reader(owner_id, records):
        calls.append((owner_id, records))
        return candidate_package()

    app, cookie, _ = confirmed_app(reader)
    app._batch_loader = None
    assert call(app, "/admin/style/review", cookie=cookie)[0] == 503
    assert calls == []
    session = app._sessions[cookie.split("=", 1)[1]]
    assert session.confirmed_records is None and session.confirmed_digest is None


def test_reader_receives_independent_record_copies():
    def reader(owner_id, records):
        records[0].text = "callback changed this copy"
        return candidate_package(empty=True)

    app, cookie, _ = confirmed_app(reader)
    session = app._sessions[cookie.split("=", 1)[1]]
    original = session.confirmed_records[0].text
    assert call(app, "/admin/style/review", cookie=cookie)[0] == 200
    assert session.confirmed_records[0].text == original


@pytest.mark.parametrize("mode", ["exception", "wrong_type", "bad_package", "empty"])
def test_reader_failure_bad_package_or_empty_package(mode):
    calls = []

    def reader(owner_id, records):
        calls.append((owner_id, records))
        if mode == "exception":
            raise RuntimeError("SYNTHETIC_SECRET_SENTINEL")
        if mode == "wrong_type":
            return {"candidates": ["SYNTHETIC_SECRET_SENTINEL"]}
        if mode == "bad_package":
            return candidate_package().model_copy(update={"candidates": [object()]})
        return candidate_package(empty=True)

    app, cookie, _ = confirmed_app(reader)
    status, _, page = call(app, "/admin/style/review", cookie=cookie)
    assert len(calls) == 1
    if mode == "empty":
        assert status == 200 and "<ul></ul>" in page
    else:
        assert status == 503 and "SYNTHETIC_SECRET_SENTINEL" not in page
        assert "synthetic candidate" not in page


def test_wrong_or_expired_session_does_not_call_reader():
    now = [100.0]
    calls = []

    def reader(owner_id, records):
        calls.append((owner_id, records))
        return candidate_package()

    app, cookie, _ = confirmed_app(reader, clock=lambda: now[0])
    assert call(app, "/admin/style/review", cookie="admin_session=wrong")[0] == 403
    now[0] += 901
    assert call(app, "/admin/style/review", cookie=cookie)[0] == 403
    assert calls == []
