import json
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest

from companion.contracts import ConsentRecord
from companion.memory import (
    CorrectMemoryRequest,
    CreateMemoryRequest,
    MemoryAuthorizationError,
    MemoryField,
    MemoryService,
    MemoryValidationError,
)


def test_unconsented_request_creates_zero_rows(
    service: MemoryService, count_rows: Callable[[str], int], now: datetime
) -> None:
    request = CreateMemoryRequest(
        user_id="synthetic-user-a",
        field=MemoryField.PREFERENCE,
        value="synthetic preference",
        source_timestamp=now,
        explicitly_requested=True,
    )
    with pytest.raises(MemoryAuthorizationError, match="memory write not authorized"):
        service.create_memory(request)
    assert count_rows("synthetic-user-a") == 0


def test_grant_normalizes_fields_and_allows_explicit_write(
    service: MemoryService, now: datetime
) -> None:
    consent = service.grant_consent(
        ConsentRecord(
            user_id="synthetic-user-a",
            granted=True,
            granted_at=now,
            allowed_fields=["preference", "preference", "preferred_name"],
        )
    )
    assert consent.allowed_fields == ["preference", "preferred_name"]

    entry = service.create_memory(
        CreateMemoryRequest(
            user_id="synthetic-user-a",
            field=MemoryField.PREFERENCE,
            value="synthetic preference",
            source_timestamp=now,
            explicitly_requested=True,
        )
    )
    assert entry.user_id == "synthetic-user-a"
    assert entry.deleted_at is None
    assert service.list_memories("synthetic-user-a") == (entry,)


@pytest.mark.parametrize(
    ("allowed", "requested"),
    [(["preferred_name"], "preference"), ([], "explicit_note")],
)
def test_disallowed_field_creates_zero_rows(
    service: MemoryService,
    count_rows: Callable[[str], int],
    now: datetime,
    allowed: list[str],
    requested: str,
) -> None:
    service.grant_consent(
        ConsentRecord(
            user_id="synthetic-user-a",
            granted=True,
            granted_at=now,
            allowed_fields=allowed,
        )
    )
    request = CreateMemoryRequest(
        user_id="synthetic-user-a",
        field=requested,
        value="synthetic value",
        source_timestamp=now,
        explicitly_requested=True,
    )
    with pytest.raises(MemoryAuthorizationError, match="memory field not authorized"):
        service.create_memory(request)
    assert count_rows("synthetic-user-a") == 0


@pytest.mark.parametrize(
    "record",
    [
        ConsentRecord(
            user_id="synthetic-user-a",
            granted=False,
            granted_at=None,
            allowed_fields=[],
        ),
        ConsentRecord(
            user_id="synthetic-user-a",
            granted=True,
            granted_at=datetime(2026, 8, 10).astimezone(),
            allowed_fields=["unknown"],
        ),
    ],
)
def test_invalid_grant_is_rejected_before_write(
    service: MemoryService, record: ConsentRecord
) -> None:
    with pytest.raises(MemoryValidationError, match="invalid memory consent"):
        service.grant_consent(record)
    assert service.get_consent("synthetic-user-a") is None


def test_model_constructed_naive_grant_is_rejected_before_write(
    service: MemoryService,
) -> None:
    record = ConsentRecord.model_construct(
        user_id="synthetic-user-a",
        granted=True,
        granted_at=datetime(2026, 8, 10),
        allowed_fields=["preference"],
    )

    with pytest.raises(MemoryValidationError, match="invalid memory consent"):
        service.grant_consent(record)

    assert service.get_consent("synthetic-user-a") is None


@pytest.mark.parametrize("explicitly_requested", [1, "true", False])
def test_model_constructed_non_boolean_explicit_request_creates_zero_rows(
    service: MemoryService,
    count_rows: Callable[[str], int],
    explicitly_requested: object,
) -> None:
    service.grant_consent(
        ConsentRecord(
            user_id="synthetic-user-a",
            granted=True,
            granted_at=datetime(2026, 8, 10, tzinfo=UTC),
            allowed_fields=["preference"],
        )
    )
    request = CreateMemoryRequest.model_construct(
        user_id="synthetic-user-a",
        field=MemoryField.PREFERENCE,
        value="synthetic private value",
        source_timestamp=datetime(2026, 8, 10, tzinfo=UTC),
        explicitly_requested=explicitly_requested,
    )

    with pytest.raises(MemoryValidationError, match="invalid memory request"):
        service.create_memory(request)

    assert count_rows("synthetic-user-a") == 0


@pytest.mark.parametrize(
    ("allowed_fields", "expected_message"),
    [([], "memory write not authorized"), (["preferred_name"], "memory field not authorized")],
)
def test_create_waits_for_concurrent_consent_change_and_observes_committed_policy(
    service: MemoryService,
    database_path: Path,
    now: datetime,
    allowed_fields: list[str],
    expected_message: str,
) -> None:
    service.grant_consent(
        ConsentRecord(
            user_id="synthetic-user-a",
            granted=True,
            granted_at=now,
            allowed_fields=["preference"],
        )
    )
    request = CreateMemoryRequest(
        user_id="synthetic-user-a",
        field=MemoryField.PREFERENCE,
        value="synthetic concurrent value",
        source_timestamp=now,
        explicitly_requested=True,
    )
    started = Event()
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE consents SET granted = ?, allowed_fields = ? WHERE user_id = ?",
            (bool(allowed_fields), json.dumps(allowed_fields), "synthetic-user-a"),
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_create_after_signal, service, request, started)
            assert started.wait(timeout=1)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
            connection.commit()
            with pytest.raises(MemoryAuthorizationError, match=expected_message):
                future.result(timeout=2)
    finally:
        connection.close()

    assert service.list_memories("synthetic-user-a") == ()


def _create_after_signal(
    service: MemoryService, request: CreateMemoryRequest, started: Event
) -> None:
    started.set()
    service.create_memory(request)


@pytest.mark.parametrize(
    ("allowed_fields", "expected_message"),
    [([], "memory write not authorized"), (["preferred_name"], "memory field not authorized")],
)
def test_correction_waits_for_concurrent_consent_change_and_preserves_old_value(
    service: MemoryService,
    database_path: Path,
    now: datetime,
    allowed_fields: list[str],
    expected_message: str,
) -> None:
    service.grant_consent(
        ConsentRecord(
            user_id="synthetic-user-a",
            granted=True,
            granted_at=now,
            allowed_fields=["preference"],
        )
    )
    original = service.create_memory(
        CreateMemoryRequest(
            user_id="synthetic-user-a",
            field=MemoryField.PREFERENCE,
            value="synthetic original value",
            source_timestamp=now,
            explicitly_requested=True,
        )
    )
    request = CorrectMemoryRequest(
        user_id="synthetic-user-a",
        memory_id=original.memory_id,
        value="synthetic unauthorized correction",
        source_timestamp=now,
    )
    started = Event()
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE consents SET granted = ?, allowed_fields = ? WHERE user_id = ?",
            (bool(allowed_fields), json.dumps(allowed_fields), "synthetic-user-a"),
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_correct_after_signal, service, request, started)
            assert started.wait(timeout=1)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
            connection.commit()
            with pytest.raises(MemoryAuthorizationError, match=expected_message):
                future.result(timeout=2)
    finally:
        connection.close()

    assert service.list_memories("synthetic-user-a") == (original,)


def _correct_after_signal(
    service: MemoryService, request: CorrectMemoryRequest, started: Event
) -> None:
    started.set()
    service.correct_memory(request)
