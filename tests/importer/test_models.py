from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from companion.importer import (
    ImportBatch,
    ImportIssue,
    ImportSeverity,
    MessageDirection,
    SourceChatRecord,
)


def test_source_record_round_trips_without_raw_content() -> None:
    record = SourceChatRecord(
        conversation_id="wechat:synthetic-contact",
        message_id="synthetic-message-1",
        timestamp=datetime(2026, 7, 17, tzinfo=UTC),
        source_speaker_id="synthetic-owner",
        source_speaker_name="Synthetic Owner",
        direction=MessageDirection.OUTBOUND,
        content_type="text",
        text="synthetic hello",
    )

    restored = SourceChatRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert "rawContent" not in record.model_dump_json()


def test_source_record_requires_timezone() -> None:
    with pytest.raises(ValidationError):
        SourceChatRecord(
            conversation_id="wechat:synthetic-contact",
            message_id="synthetic-message-1",
            timestamp=datetime(2026, 7, 17),
            source_speaker_id="synthetic-owner",
            source_speaker_name=None,
            direction=MessageDirection.OUTBOUND,
            content_type="text",
            text="synthetic",
        )


def test_import_batch_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ImportBatch(records=[], issues=[], unknown=True)


def test_issue_contains_only_safe_diagnostic_fields() -> None:
    issue = ImportIssue(
        code="sender_direction_conflict",
        severity=ImportSeverity.WARNING,
        message_index=3,
        field_path="messages[3].senderUsername",
        detail="outbound sender disagrees with owner metadata",
    )

    assert set(issue.model_dump()) == {
        "code",
        "severity",
        "message_index",
        "field_path",
        "detail",
    }
