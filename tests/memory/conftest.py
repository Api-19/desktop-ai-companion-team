import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from companion.contracts import ConsentRecord, MemoryEntry
from companion.memory import CreateMemoryRequest, MemoryField, MemoryService


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.sqlite3"


@pytest.fixture
def service(database_path: Path, now: datetime) -> MemoryService:
    return MemoryService(database_path, clock=lambda: now)


@pytest.fixture
def count_rows(database_path: Path) -> Callable[[str], int]:
    def count(user_id: str) -> int:
        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
            ).fetchone()
        assert row is not None
        return int(row[0])

    return count


@pytest.fixture
def later() -> datetime:
    return datetime(2026, 8, 10, 12, 5, tzinfo=UTC)


@pytest.fixture
def grant_all(service: MemoryService, now: datetime) -> Callable[[str], ConsentRecord]:
    def grant(user_id: str) -> ConsentRecord:
        return service.grant_consent(
            ConsentRecord(
                user_id=user_id,
                granted=True,
                granted_at=now,
                allowed_fields=[field.value for field in MemoryField],
            )
        )

    return grant


@pytest.fixture
def create(service: MemoryService, now: datetime) -> Callable[[str, str], MemoryEntry]:
    def create_memory(user_id: str, value: str) -> MemoryEntry:
        return service.create_memory(
            CreateMemoryRequest(
                user_id=user_id,
                field=MemoryField.PREFERENCE,
                value=value,
                source_timestamp=now,
                explicitly_requested=True,
            )
        )

    return create_memory
