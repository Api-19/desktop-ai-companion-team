from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from companion.memory import (
    CorrectMemoryRequest,
    CreateMemoryRequest,
    MemoryField,
)

NOW = datetime(2026, 8, 10, tzinfo=UTC)


def test_create_request_requires_explicit_true_and_fixed_field() -> None:
    request = CreateMemoryRequest(
        user_id="synthetic-user",
        field=MemoryField.PREFERENCE,
        value="synthetic preference",
        source_timestamp=NOW,
        explicitly_requested=True,
    )
    assert request.field is MemoryField.PREFERENCE

    with pytest.raises(ValidationError):
        CreateMemoryRequest(
            user_id="synthetic-user",
            field="unknown",
            value="synthetic preference",
            source_timestamp=NOW,
            explicitly_requested=True,
        )

    with pytest.raises(ValidationError):
        CreateMemoryRequest(
            user_id="synthetic-user",
            field=MemoryField.PREFERENCE,
            value="synthetic preference",
            source_timestamp=NOW,
            explicitly_requested=False,
        )

    for non_boolean_true in (1, "true"):
        with pytest.raises(ValidationError):
            CreateMemoryRequest(
                user_id="synthetic-user",
                field=MemoryField.PREFERENCE,
                value="synthetic preference",
                source_timestamp=NOW,
                explicitly_requested=non_boolean_true,
            )


def test_correct_request_rejects_naive_timestamp_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CorrectMemoryRequest(
            user_id="synthetic-user",
            memory_id="a" * 32,
            value="synthetic corrected",
            source_timestamp=datetime(2026, 8, 10),
            unexpected="blocked",
        )


@pytest.mark.parametrize("memory_id", ["a" * 31, "a" * 33, "g" * 32])
def test_correct_request_requires_exact_lowercase_hex_memory_id(memory_id: str) -> None:
    with pytest.raises(ValidationError):
        CorrectMemoryRequest(
            user_id="synthetic-user",
            memory_id=memory_id,
            value="synthetic corrected",
            source_timestamp=NOW,
        )
