import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from companion.contracts import ConsentRecord
from companion.memory import CreateMemoryRequest, MemoryField, MemoryService
from companion.memory.errors import MemoryStoreError


def _grant(service: MemoryService, now: datetime, fields: list[str], user: str = "synthetic-a"):
    return service.grant_consent(
        ConsentRecord(user_id=user, granted=True, granted_at=now, allowed_fields=fields)
    )


@pytest.mark.parametrize("fields", [[], ["preference"], ["preference", "explicit_note"]])
def test_consent_update_deletes_only_removed_fields_and_never_resurrects(
    service: MemoryService, database_path: Path, now: datetime, fields: list[str]
) -> None:
    all_fields = ["preference", "explicit_note"]
    for user in ("synthetic-a", "synthetic-b"):
        _grant(service, now, all_fields, user)
        for field in all_fields:
            service.create_memory(
                CreateMemoryRequest(
                    user_id=user,
                    field=MemoryField(field),
                    value=f"SYNTHETIC_WITHDRAWAL_{field}",
                    source_timestamp=now,
                    explicitly_requested=True,
                )
            )

    _grant(service, now, fields)
    restarted = MemoryService(database_path, clock=lambda: now)
    assert {entry.field for entry in restarted.list_memories("synthetic-a")} == set(fields)
    assert len(restarted.retrieve_memories("synthetic-a", "SYNTHETIC_WITHDRAWAL")) == len(fields)
    assert len(restarted.list_memories("synthetic-b")) == 2
    with sqlite3.connect(database_path) as connection:
        assert set(
            connection.execute(
                "SELECT field FROM memories WHERE user_id = 'synthetic-a'"
            ).fetchall()
        ) == {(field,) for field in fields}
    _grant(restarted, now, all_fields)
    assert {entry.field for entry in restarted.list_memories("synthetic-a")} == set(fields)


def test_consent_and_removed_rows_roll_back_together_on_delete_failure(
    service: MemoryService, database_path: Path, now: datetime
) -> None:
    original = _grant(service, now, ["preference"])
    service.create_memory(
        CreateMemoryRequest(
            user_id="synthetic-a",
            field=MemoryField.PREFERENCE,
            value="SYNTHETIC_ATOMIC_WITHDRAWAL",
            source_timestamp=now,
            explicitly_requested=True,
        )
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """CREATE TRIGGER reject_delete BEFORE DELETE ON memories
               BEGIN SELECT RAISE(ABORT, 'synthetic delete failure'); END"""
        )
    with pytest.raises(MemoryStoreError, match=r"^memory store unavailable$"):
        _grant(service, now, [])
    assert service.get_consent("synthetic-a") == original
    assert len(service.list_memories("synthetic-a")) == 1
