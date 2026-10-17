from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import pytest
from pydantic import BaseModel

from companion.memory import (
    CreateMemoryRequest,
    MemoryAuthorizationError,
    MemoryField,
    MemoryService,
    MemoryStoreError,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _memory_traceback_material(traceback: TracebackType | None) -> str:
    material: list[str] = []
    while traceback is not None:
        frame = traceback.tb_frame
        if "companion\\memory" in frame.f_code.co_filename.replace("/", "\\"):
            for value in frame.f_locals.values():
                if isinstance(value, str):
                    material.append(value)
                elif isinstance(value, Path):
                    material.append(str(value))
                elif isinstance(value, BaseModel):
                    material.append(value.model_dump_json())
                else:
                    database_path = getattr(value, "_database_path", None)
                    if isinstance(database_path, Path):
                        material.append(str(database_path))
        traceback = traceback.tb_next
    return "\n".join(material)


def test_store_failure_exposes_only_fixed_error(tmp_path: Path) -> None:
    database_path = tmp_path / "synthetic-secret-path.sqlite3"
    service = MemoryService(database_path, clock=lambda: NOW)
    database_path.write_text("not a sqlite database", encoding="utf-8")

    with pytest.raises(MemoryStoreError) as captured:
        service.get_consent("synthetic-secret-user")

    assert str(captured.value) == "memory store unavailable"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    traceback_material = _memory_traceback_material(captured.value.__traceback__)
    assert "synthetic-secret-user" not in traceback_material
    assert "synthetic-secret-path" not in traceback_material
    assert "SELECT" not in traceback_material


def test_authorization_failure_traceback_does_not_retain_request(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "memory.sqlite3", clock=lambda: NOW)
    request = CreateMemoryRequest(
        user_id="synthetic-private-user",
        field=MemoryField.EXPLICIT_NOTE,
        value="synthetic-private-memory-value",
        source_timestamp=NOW,
        explicitly_requested=True,
    )

    with pytest.raises(MemoryAuthorizationError) as captured:
        service.create_memory(request)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    traceback_material = _memory_traceback_material(captured.value.__traceback__)
    assert "synthetic-private-user" not in traceback_material
    assert "synthetic-private-memory-value" not in traceback_material
