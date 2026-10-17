import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from companion.contracts import ConsentRecord, MemoryEntry
from companion.memory import (
    CorrectMemoryRequest,
    CreateMemoryRequest,
    MemoryAuthorizationError,
    MemoryField,
    MemoryService,
    MemoryStoreError,
    MemoryValidationError,
)


def test_identical_queries_never_cross_users(
    service: MemoryService,
    grant_all: Callable[[str], ConsentRecord],
    create: Callable[[str, str], MemoryEntry],
) -> None:
    grant_all("synthetic-user-a")
    grant_all("synthetic-user-b")
    entry_a = create("synthetic-user-a", "shared synthetic phrase")
    entry_b = create("synthetic-user-b", "shared synthetic phrase")

    assert service.retrieve_memories("synthetic-user-a", "shared") == (entry_a,)
    assert service.retrieve_memories("synthetic-user-b", "shared") == (entry_b,)


def test_other_user_cannot_correct_or_delete_memory(
    service: MemoryService,
    grant_all: Callable[[str], ConsentRecord],
    create: Callable[[str, str], MemoryEntry],
    now: datetime,
) -> None:
    grant_all("synthetic-user-a")
    grant_all("synthetic-user-b")
    entry = create("synthetic-user-a", "synthetic old")

    with pytest.raises(MemoryAuthorizationError, match="memory entry unavailable"):
        service.correct_memory(
            CorrectMemoryRequest(
                user_id="synthetic-user-b",
                memory_id=entry.memory_id,
                value="synthetic stolen",
                source_timestamp=now,
            )
        )
    receipt = service.delete_memory("synthetic-user-b", entry.memory_id)
    assert receipt.deleted_count == 0
    assert service.list_memories("synthetic-user-a") == (entry,)


def test_correction_replaces_old_value(
    service: MemoryService,
    grant_all: Callable[[str], ConsentRecord],
    create: Callable[[str, str], MemoryEntry],
    later: datetime,
) -> None:
    grant_all("synthetic-user-a")
    entry = create("synthetic-user-a", "synthetic old")
    corrected = service.correct_memory(
        CorrectMemoryRequest(
            user_id="synthetic-user-a",
            memory_id=entry.memory_id,
            value="synthetic corrected",
            source_timestamp=later,
        )
    )
    assert corrected.memory_id == entry.memory_id
    assert service.retrieve_memories("synthetic-user-a", "old") == ()
    assert service.retrieve_memories("synthetic-user-a", "corrected") == (corrected,)


def test_delete_physically_removes_row_and_results(
    service: MemoryService,
    grant_all: Callable[[str], ConsentRecord],
    create: Callable[[str, str], MemoryEntry],
    count_rows: Callable[[str], int],
) -> None:
    grant_all("synthetic-user-a")
    entry = create("synthetic-user-a", "synthetic delete bait")
    receipt = service.delete_memory("synthetic-user-a", entry.memory_id)
    assert receipt.deleted_count == 1
    assert count_rows("synthetic-user-a") == 0
    assert service.list_memories("synthetic-user-a") == ()
    assert service.retrieve_memories("synthetic-user-a", "delete bait") == ()


def test_revoke_deletes_all_and_blocks_future_writes(
    service: MemoryService,
    grant_all: Callable[[str], ConsentRecord],
    create: Callable[[str, str], MemoryEntry],
    now: datetime,
) -> None:
    grant_all("synthetic-user-a")
    create("synthetic-user-a", "synthetic first")
    create("synthetic-user-a", "synthetic second")
    receipt = service.revoke_consent("synthetic-user-a")
    assert receipt.deleted_count == 2
    consent = service.get_consent("synthetic-user-a")
    assert consent is not None
    assert consent.granted is False
    assert consent.granted_at is None
    assert consent.allowed_fields == []
    assert service.list_memories("synthetic-user-a") == ()

    request = CreateMemoryRequest(
        user_id="synthetic-user-a",
        field=MemoryField.PREFERENCE,
        value="synthetic third",
        source_timestamp=now,
        explicitly_requested=True,
    )
    with pytest.raises(MemoryAuthorizationError, match="memory write not authorized"):
        service.create_memory(request)


def test_delete_erases_value_from_database_artifacts(
    service: MemoryService,
    database_path: Path,
    grant_all: Callable[[str], ConsentRecord],
    create: Callable[[str, str], MemoryEntry],
) -> None:
    marker = "SYNTHETIC_SINGLE_DELETE_SENTINEL_" * 12
    grant_all("synthetic-user-a")
    entry = create("synthetic-user-a", marker)
    assert marker.encode() in database_path.read_bytes()

    service.delete_memory("synthetic-user-a", entry.memory_id)

    _assert_marker_absent_from_database_artifacts(database_path, marker)


def test_revoke_erases_values_from_database_artifacts(
    service: MemoryService,
    database_path: Path,
    grant_all: Callable[[str], ConsentRecord],
    create: Callable[[str, str], MemoryEntry],
) -> None:
    marker = "SYNTHETIC_REVOKE_SENTINEL_" * 15
    grant_all("synthetic-user-a")
    create("synthetic-user-a", marker)
    assert marker.encode() in database_path.read_bytes()

    service.revoke_consent("synthetic-user-a")

    _assert_marker_absent_from_database_artifacts(database_path, marker)


def test_legacy_deleted_values_are_removed_on_initialization(tmp_path: Path, now: datetime) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    marker = "SYNTHETIC_LEGACY_DELETED_SENTINEL_" * 12
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA secure_delete = OFF")
        connection.execute("CREATE TABLE legacy_private(value TEXT NOT NULL)")
        connection.execute("INSERT INTO legacy_private(value) VALUES (?)", (marker,))
        connection.execute("DELETE FROM legacy_private")
    assert marker.encode() in database_path.read_bytes()

    MemoryService(database_path, clock=lambda: now)

    _assert_marker_absent_from_database_artifacts(database_path, marker)


def test_service_releases_database_file_handles(tmp_path: Path, now: datetime) -> None:
    database_path = tmp_path / "disposable.sqlite3"
    service = MemoryService(database_path, clock=lambda: now)
    service.get_consent("synthetic-user-a")

    database_path.unlink()

    assert not database_path.exists()


def test_memories_are_ordered_by_actual_instant(
    service: MemoryService, grant_all: Callable[[str], ConsentRecord]
) -> None:
    grant_all("synthetic-user-a")
    older = service.create_memory(
        CreateMemoryRequest(
            user_id="synthetic-user-a",
            field=MemoryField.PREFERENCE,
            value="synthetic older",
            source_timestamp=datetime(2026, 8, 10, 12, tzinfo=timezone(timedelta(hours=9))),
            explicitly_requested=True,
        )
    )
    newer = service.create_memory(
        CreateMemoryRequest(
            user_id="synthetic-user-a",
            field=MemoryField.PREFERENCE,
            value="synthetic newer",
            source_timestamp=datetime(2026, 8, 10, 4, tzinfo=UTC),
            explicitly_requested=True,
        )
    )

    assert service.list_memories("synthetic-user-a") == (newer, older)
    assert newer.source_timestamp.tzinfo is UTC
    assert older.source_timestamp.tzinfo is UTC


def test_retrieval_preserves_store_error_type(
    service: MemoryService, database_path: Path, grant_all: Callable[[str], ConsentRecord]
) -> None:
    grant_all("synthetic-user-a")
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE memories")

    with pytest.raises(MemoryStoreError, match="memory store unavailable"):
        service.retrieve_memories("synthetic-user-a", "synthetic")


def test_invalid_query_is_rejected_before_store_access(
    service: MemoryService, database_path: Path
) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE consents")

    with pytest.raises(MemoryValidationError, match="invalid memory query") as captured:
        service.retrieve_memories("synthetic-user-a", "   ")

    assert not isinstance(captured.value, MemoryStoreError)


def _assert_marker_absent_from_database_artifacts(database_path: Path, marker: str) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        artifact = Path(f"{database_path}{suffix}")
        if artifact.exists():
            assert marker.encode() not in artifact.read_bytes()
