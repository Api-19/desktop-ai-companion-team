"""Persistent content revisions over synthetic, per-user SQLite data."""

import sqlite3
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from companion.contracts import ConsentRecord
from companion.memory import (
    CorrectMemoryRequest,
    CreateMemoryRequest,
    MemoryAuthorizationError,
    MemoryService,
    MemoryStoreError,
    MemoryValidationError,
)
from companion.memory.profile import ProfileItem, ProfileStatus

A = "synthetic-revision-a"
B = "synthetic-revision-b"
FIELDS = ["preference", "explicit_note", "profile"]


def _grant(service: MemoryService, now: datetime, user: str = A, fields=None):
    return service.grant_consent(
        ConsentRecord(
            user_id=user,
            granted=True,
            granted_at=now,
            allowed_fields=FIELDS if fields is None else fields,
        )
    )


def _memory(service: MemoryService, now: datetime, user: str = A):
    return service.create_memory(
        CreateMemoryRequest(
            user_id=user,
            field="preference",
            value="合成偏好",
            source_timestamp=now,
            explicitly_requested=True,
        )
    )


def _candidate(service: MemoryService, now: datetime, text="合成少喝水") -> ProfileItem:
    return service.add_profile_candidate(
        user_id=A,
        category="habit",
        topic="喝水",
        text=text,
        source="self_stated",
        evidence_date=now.date(),
        confidence="high",
    )


def _confirm(service: MemoryService, item: ProfileItem) -> ProfileItem:
    return service.confirm_profile(
        user_id=A,
        profile_id=item.profile_id,
        version=item.version,
        subject_id=A,
        use_in_chat=True,
        simulated=True,
    )


def test_unknown_user_revision_does_not_create_user_data(
    service: MemoryService, database_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect = sqlite3.connect
    opened = []

    def synthetic_database_only(database, *args, **kwargs):
        # Check at the actual file access, including profile cleanup connections.
        assert database == database_path
        opened.append(database)
        return connect(database, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", synthetic_database_only)
        assert service.content_revision(A) == 0
        assert service.content_revision(A) == 0
    assert opened
    with connect(database_path) as connection:
        for table in ("consents", "memories", "profiles", "content_revisions"):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (A,)
            ).fetchone() == (0,)


def test_revision_database_failure_uses_existing_private_error_boundary(
    service: MemoryService, database_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = "SYNTHETIC_REVISION_SQL_CANARY"
    connect = sqlite3.connect
    attempted = []

    class FailingRevisionConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "SELECT revision" in sql:
                attempted.append(sql)
                raise sqlite3.OperationalError(f"{canary}: {database_path}: {sql}")
            return super().execute(sql, parameters)

    def synthetic_database_only(database, *args, **kwargs):
        assert database == database_path
        return connect(database, *args, factory=FailingRevisionConnection, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", synthetic_database_only)
        with pytest.raises(MemoryStoreError) as captured:
            service.content_revision(A)
    assert attempted
    error = captured.value
    assert str(error) == "memory store unavailable"
    assert error.__cause__ is None
    assert error.__context__ is None
    material = "".join(traceback.format_exception(error))
    frame = error.__traceback__
    while frame is not None:
        if "/companion/memory/" in frame.tb_frame.f_code.co_filename:
            material += repr(frame.tb_frame.f_locals)
        frame = frame.tb_next
    for secret in (canary, str(database_path), "SELECT revision", A):
        assert secret not in material


def test_revision_survives_restart_and_is_visible_to_existing_service(
    service: MemoryService, database_path: Path, now: datetime
) -> None:
    other = MemoryService(database_path, clock=lambda: now)
    assert service.content_revision(A) == 0
    _grant(other, now)
    first = service.content_revision(A)
    assert first > 0
    _memory(other, now)
    current = service.content_revision(A)
    assert current > first
    assert current == other.content_revision(A)
    assert current == MemoryService(database_path, clock=lambda: now).content_revision(A)
    assert service.content_revision(B) == 0


@pytest.mark.parametrize(
    "mutation,delta",
    [
        ("grant", 1),
        ("create", 1),
        ("correct", 1),
        ("delete", 1),
        ("revoke", 3),
        ("narrow", 3),
        ("profile_add", 1),
        ("profile_approve", 1),
        ("profile_edit", 1),
        ("profile_confirm", 1),
        ("profile_disable", 1),
        ("profile_delete", 1),
        ("conflict", 3),
        ("expiry_disable", 1),
        ("candidate_cleanup", 1),
        ("disabled_cleanup", 1),
        ("expired_cleanup", 1),
        ("expired_direct_cleanup", 1),
    ],
)
def test_revision_tracks_all_content_mutations(
    database_path: Path, now: datetime, mutation: str, delta: int
) -> None:
    clock = [now]
    service = MemoryService(database_path, clock=lambda: clock[0])
    _grant(service, now)
    _grant(service, now, B)
    memory = _memory(service, now)
    item = _candidate(service, now)
    target = dict(user_id=A, profile_id=item.profile_id)
    if mutation in {
        "profile_confirm",
        "conflict",
        "expiry_disable",
        "expired_cleanup",
        "expired_direct_cleanup",
    }:
        service.approve_profile(**target)
    if mutation in {"conflict", "expiry_disable", "expired_cleanup", "expired_direct_cleanup"}:
        item = _confirm(service, item)
    if mutation == "conflict":
        item = _candidate(service, now, text="合成多喝水")
        service.approve_profile(user_id=A, profile_id=item.profile_id)
    if mutation == "disabled_cleanup":
        service.disable_profile(**target, reason="owner")
    if mutation == "expired_cleanup":
        clock[0] = datetime.combine(item.valid_until, datetime.min.time(), UTC)
        service.list_profile(A)
    before = service.content_revision(A)
    other_before = service.content_revision(B)

    if mutation == "grant":
        _grant(service, now + timedelta(seconds=1))
    elif mutation == "create":
        _memory(service, now)
    elif mutation == "correct":
        corrected = service.correct_memory(
            CorrectMemoryRequest(
                user_id=A, memory_id=memory.memory_id, value="合成新偏好", source_timestamp=now
            )
        )
        assert service.list_memories(A) == (corrected,)
    elif mutation == "delete":
        assert service.delete_memory(A, memory.memory_id).deleted_count == 1
    elif mutation == "revoke":
        assert service.revoke_consent(A).deleted_count == 2
        assert service.list_memories(A) == service.list_profile(A) == ()
    elif mutation == "narrow":
        _grant(service, now, fields=["explicit_note"])
        assert service.list_memories(A) == service.list_profile(A) == ()
    elif mutation == "profile_add":
        _candidate(service, now)
    elif mutation == "profile_approve":
        approved = service.approve_profile(**target)
        assert approved.status == ProfileStatus.OWNER_APPROVED
    elif mutation == "profile_edit":
        edited = service.edit_profile(**target, text="合成新画像")
        assert service.list_profile(A) == (edited,)
    elif mutation == "profile_confirm":
        assert _confirm(service, item).status == ProfileStatus.CONFIRMED
    elif mutation == "profile_disable":
        assert service.disable_profile(**target, reason="owner").status == ProfileStatus.DISABLED
    elif mutation == "profile_delete":
        assert service.delete_profile(**target).deleted_count == 1
    elif mutation == "conflict":
        assert _confirm(service, item).disabled_reason == "conflict"
        assert all(p.disabled_reason == "conflict" for p in service.list_profile(A))
    else:
        if mutation in {"candidate_cleanup", "disabled_cleanup"}:
            clock[0] = now + timedelta(days=30)
        else:
            clock[0] = datetime.combine(item.valid_until, datetime.min.time(), UTC)
            if mutation != "expiry_disable":
                clock[0] += timedelta(days=30)
        # Revision reads retain the service's existing profile cleanup side effect.
        assert service.content_revision(A) == before + delta
        profiles = service.list_profile(A)
        if mutation == "expiry_disable":
            assert profiles[0].disabled_reason == "expired"
        else:
            assert profiles == ()
    assert service.content_revision(A) == before + delta
    assert service.content_revision(B) == other_before


@pytest.mark.parametrize(
    "mutation", ["consent", "memory", "profile", "confirm", "disable", "delete", "revoke"]
)
def test_unchanged_mutations_do_not_advance_revision(
    service: MemoryService, now: datetime, mutation: str
) -> None:
    consent = _grant(service, now)
    memory = _memory(service, now)
    item = _candidate(service, now)
    target = dict(user_id=A, profile_id=item.profile_id)
    if mutation == "confirm":
        service.approve_profile(**target)
        item = _confirm(service, item)
    if mutation == "disable":
        service.disable_profile(**target, reason="owner")
    if mutation == "delete":
        service.delete_memory(A, memory.memory_id)
    if mutation == "revoke":
        service.revoke_consent(A)
    before = service.content_revision(A)
    if mutation == "consent":
        service.grant_consent(consent)
    elif mutation == "memory":
        service.correct_memory(
            CorrectMemoryRequest(
                user_id=A,
                memory_id=memory.memory_id,
                value=memory.value,
                source_timestamp=memory.source_timestamp,
            )
        )
    elif mutation == "profile":
        assert service.edit_profile(**target) == item
    elif mutation == "confirm":
        assert _confirm(service, item) == item
    elif mutation == "disable":
        service.disable_profile(**target, reason="owner")
    elif mutation == "delete":
        assert service.delete_memory(A, memory.memory_id).deleted_count == 0
    elif mutation == "revoke":
        assert service.revoke_consent(A).deleted_count == 0
    assert service.content_revision(A) == before


@pytest.mark.parametrize("mutation", ["narrow", "revoke", "conflict", "cleanup"])
def test_revision_rolls_back_with_content_transaction(
    database_path: Path, now: datetime, mutation: str
) -> None:
    clock = [now]
    service = MemoryService(database_path, clock=lambda: clock[0])
    consent = _grant(service, now)
    memory = _memory(service, now)
    first = _candidate(service, now)
    if mutation == "conflict":
        service.approve_profile(user_id=A, profile_id=first.profile_id)
        _confirm(service, first)
        second = _candidate(service, now, text="合成多喝水")
        service.approve_profile(user_id=A, profile_id=second.profile_id)
    elif mutation == "cleanup":
        clock[0] += timedelta(days=1)
        _candidate(service, now, text="合成后续画像")
        clock[0] = now + timedelta(days=31)
    # Snapshot directly: a service read would itself run the cleanup under test.
    with sqlite3.connect(database_path) as connection:
        profiles_before = connection.execute(
            "SELECT * FROM profiles ORDER BY profile_id"
        ).fetchall()
        revision_before = connection.execute("SELECT * FROM content_revisions").fetchall()
        if mutation in {"narrow", "revoke"}:
            connection.execute("""CREATE TRIGGER reject_memory_delete BEFORE DELETE ON memories
                BEGIN SELECT RAISE(ABORT, 'synthetic rollback'); END""")
        elif mutation == "conflict":
            connection.execute("""CREATE TRIGGER reject_conflict BEFORE UPDATE ON profiles
                WHEN json_extract(NEW.payload, '$.disabled_reason') = 'conflict'
                BEGIN SELECT RAISE(ABORT, 'synthetic rollback'); END""")
        else:
            connection.execute(f"""CREATE TRIGGER reject_cleanup BEFORE DELETE ON profiles
                WHEN OLD.profile_id != '{first.profile_id}'
                BEGIN SELECT RAISE(ABORT, 'synthetic rollback'); END""")
    with pytest.raises(MemoryStoreError, match=r"^memory store unavailable$"):
        if mutation == "narrow":
            _grant(service, now, fields=[])
        elif mutation == "revoke":
            service.revoke_consent(A)
        elif mutation == "conflict":
            _confirm(service, second)
        else:
            service.content_revision(A)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT * FROM content_revisions").fetchall() == revision_before
        assert (
            connection.execute("SELECT * FROM profiles ORDER BY profile_id").fetchall()
            == profiles_before
        )
    clock[0] = now
    assert service.get_consent(A) == consent
    assert service.list_memories(A) == (memory,)


def test_first_unknown_revocation_advances_once_and_missing_profile_still_raises(
    service: MemoryService,
) -> None:
    assert service.content_revision(A) == 0
    service.revoke_consent(A)
    assert service.content_revision(A) == 1
    service.revoke_consent(A)
    assert service.content_revision(A) == 1
    with pytest.raises(MemoryAuthorizationError, match=r"^memory entry unavailable$"):
        service.delete_profile(user_id=A, profile_id="0" * 32)
    assert service.content_revision(A) == 1


def test_invalid_revision_user_is_rejected_before_cleanup(database_path: Path) -> None:
    def unexpected_clock():
        pytest.fail("invalid user must be rejected before profile cleanup")

    service = MemoryService(database_path, clock=unexpected_clock)
    with pytest.raises(MemoryValidationError, match=r"^invalid memory user$"):
        service.content_revision(" ")


def test_legacy_database_upgrade_is_idempotent_without_backfilling(
    service: MemoryService, database_path: Path, now: datetime
) -> None:
    _grant(service, now)
    memory = _memory(service, now)
    with sqlite3.connect(database_path) as connection:
        # Keep business rows while reverting only the new DDL to a historical database.
        for table in ("consents", "memories", "profiles"):
            for event in ("insert", "update", "delete"):
                connection.execute(f"DROP TRIGGER IF EXISTS content_revision_{table}_{event}")
        connection.execute("DROP TABLE IF EXISTS content_revisions")
        version = connection.execute("PRAGMA user_version").fetchone()
    for _ in range(2):
        restarted = MemoryService(database_path, clock=lambda: now)
        assert restarted.content_revision(A) == 0
        assert restarted.list_memories(A) == (memory,)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == version
        assert connection.execute("SELECT * FROM content_revisions").fetchall() == []
    restarted.delete_memory(A, memory.memory_id)
    assert restarted.content_revision(A) == 1
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM consents WHERE user_id = ?", (A,))
    assert restarted.content_revision(A) == 2


@pytest.mark.parametrize(
    "table,assignment",
    [
        ("consents", "granted = 0"),
        ("consents", "granted_at = NULL"),
        ("consents", "allowed_fields = '[\"preference\"]'"),
        ("memories", "field = 'explicit_note'"),
        ("memories", "value = 'synthetic-new-value'"),
        ("memories", "source_timestamp = '2026-08-11T12:00:00+00:00'"),
        ("profiles", "created_at = '2026-08-11T12:00:00+00:00'"),
        ("profiles", "payload = json_set(payload, '$.text', '合成新画像')"),
        ("profiles", "disabled_at = '2026-08-11T12:00:00+00:00'"),
        ("profiles", "simulated_only = 1"),
    ],
)
def test_each_persistent_field_advances_only_when_changed(
    service: MemoryService, database_path: Path, now: datetime, table: str, assignment: str
) -> None:
    _grant(service, now)
    _memory(service, now)
    _candidate(service, now)
    before = service.content_revision(A)
    for expected in (before + 1, before + 1):
        with sqlite3.connect(database_path) as connection:
            connection.execute(f"UPDATE {table} SET {assignment} WHERE user_id = ?", (A,))
        assert service.content_revision(A) == expected
