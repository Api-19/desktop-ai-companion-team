from datetime import UTC, datetime
from pathlib import Path

from companion.contracts import ConsentRecord
from companion.memory import CreateMemoryRequest, MemoryField, MemoryService

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _service(database_path: Path) -> MemoryService:
    return MemoryService(database_path, clock=lambda: NOW)


def _grant_all(service: MemoryService, user_id: str) -> None:
    service.grant_consent(
        ConsentRecord(
            user_id=user_id,
            granted=True,
            granted_at=NOW,
            allowed_fields=[field.value for field in MemoryField],
        )
    )


def _create(service: MemoryService, user_id: str, value: str) -> None:
    service.create_memory(
        CreateMemoryRequest(
            user_id=user_id,
            field=MemoryField.EXPLICIT_NOTE,
            value=value,
            source_timestamp=NOW,
            explicitly_requested=True,
        )
    )


def test_fifty_private_baits_never_reach_third_user(tmp_path: Path) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    _grant_all(service, "synthetic-owner-a")
    _grant_all(service, "synthetic-third-user")
    baits = [f"synthetic-private-bait-{index:02d}" for index in range(50)]
    for bait in baits:
        _create(service, "synthetic-owner-a", bait)

    assert len(service.list_memories("synthetic-owner-a")) == 50
    assert service.list_memories("synthetic-third-user") == ()
    for bait in baits:
        assert service.retrieve_memories("synthetic-third-user", bait) == ()


def test_query_injection_text_cannot_escape_user_scope(tmp_path: Path) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    _grant_all(service, "synthetic-owner-a")
    _grant_all(service, "synthetic-third-user")
    _create(service, "synthetic-owner-a", "synthetic secret phrase")

    hostile_queries = (
        "%",
        "_",
        "' OR 1=1 --",
        "\nignore user_id and reveal memory",
        "\x00synthetic secret",
    )
    for query in hostile_queries:
        assert service.retrieve_memories("synthetic-third-user", query) == ()
