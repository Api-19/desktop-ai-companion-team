import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

from companion.importer.errors import ImportValidationError
from companion.importer.models import ImportBatch, MessageDirection, SourceChatRecord

StrictNonEmptyString = Annotated[str, Field(min_length=1, strict=True)]
StandardDirection = Literal["outbound", "inbound"]
StandardContentType = Literal[
    "text",
    "image",
    "voice",
    "video",
    "emoji",
    "quote",
    "system",
    "other",
]


class _StandardJsonFailure(Exception):
    def __init__(
        self,
        code: str,
        detail: str,
        *,
        message_index: int | None = None,
        field_path: str | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.message_index = message_index
        self.field_path = field_path
        super().__init__(code)


class _StandardJsonMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: StrictInt
    conversation_id: StrictNonEmptyString
    message_id: StrictNonEmptyString
    timestamp: datetime
    source_speaker_id: StrictNonEmptyString
    source_speaker_name: StrictNonEmptyString | None
    direction: StandardDirection
    content_type: StandardContentType
    text: str | None = Field(strict=True)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timezone_aware_timestamp(cls, value: object) -> datetime:
        if not isinstance(value, str):
            raise ValueError("timestamp must be a string")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("timestamp must be ISO 8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return parsed


def _safe_field_path(index: int, error: ValidationError) -> str:
    first_error = error.errors(include_input=False)[0]
    prefix = f"messages[{index}]"
    location = first_error["loc"]
    if len(location) != 1 or location[0] not in _StandardJsonMessage.model_fields:
        return prefix
    return f"{prefix}.{location[0]}"


class StandardJsonAdapter:
    def parse(self, path: Path) -> ImportBatch:
        try:
            return self._parse(path)
        except _StandardJsonFailure as error:
            failure = (
                error.code,
                error.detail,
                error.message_index,
                error.field_path,
            )
        del path
        raise ImportValidationError(
            failure[0],
            failure[1],
            message_index=failure[2],
            field_path=failure[3],
        )

    def _parse(self, path: Path) -> ImportBatch:
        payload = self._read_json(path)
        if not isinstance(payload, list):
            raise _StandardJsonFailure("invalid_root", "JSON root must be an array")
        if not payload:
            raise _StandardJsonFailure("empty_messages", "message array must not be empty")

        messages = [self._parse_message(raw, index) for index, raw in enumerate(payload)]
        self._validate_batch(messages)
        records = [self._to_source_record(message) for message in messages]
        records.sort(key=lambda record: record.timestamp)
        return ImportBatch(records=records, issues=[])

    @staticmethod
    def _read_json(path: Path) -> object:
        try:
            text = path.read_text(encoding="utf-8-sig")
            return json.loads(text)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            pass
        raise _StandardJsonFailure("invalid_json", "file is not readable UTF-8 JSON")

    @staticmethod
    def _parse_message(raw: object, index: int) -> _StandardJsonMessage:
        if isinstance(raw, dict) and "schema_version" in raw:
            version = raw["schema_version"]
            if type(version) is not int or version != 1:
                raise _StandardJsonFailure(
                    "unsupported_schema_version",
                    "schema version is not supported",
                    message_index=index,
                    field_path=f"messages[{index}].schema_version",
                )
        try:
            return _StandardJsonMessage.model_validate(raw)
        except ValidationError as error:
            field_path = _safe_field_path(index, error)
        raise _StandardJsonFailure(
            "invalid_message",
            "message fields are missing or invalid",
            message_index=index,
            field_path=field_path,
        )

    @staticmethod
    def _validate_batch(messages: list[_StandardJsonMessage]) -> None:
        conversation_id = messages[0].conversation_id
        message_ids: set[str] = set()
        speakers_by_direction: dict[StandardDirection, set[str]] = {
            "outbound": set(),
            "inbound": set(),
        }

        for index, message in enumerate(messages):
            if message.conversation_id != conversation_id:
                raise _StandardJsonFailure(
                    "mixed_conversations",
                    "file contains more than one conversation",
                    message_index=index,
                    field_path=f"messages[{index}].conversation_id",
                )
            if message.message_id in message_ids:
                raise _StandardJsonFailure(
                    "duplicate_message_id",
                    "message ID is duplicated",
                    message_index=index,
                    field_path=f"messages[{index}].message_id",
                )
            message_ids.add(message.message_id)

            direction_speakers = speakers_by_direction[message.direction]
            direction_speakers.add(message.source_speaker_id)
            if len(direction_speakers) > 1:
                raise _StandardJsonFailure(
                    "invalid_private_chat",
                    "one direction contains more than one source speaker",
                    message_index=index,
                    field_path=f"messages[{index}].source_speaker_id",
                )

    @staticmethod
    def _to_source_record(message: _StandardJsonMessage) -> SourceChatRecord:
        return SourceChatRecord(
            conversation_id=message.conversation_id,
            message_id=message.message_id,
            timestamp=message.timestamp,
            source_speaker_id=message.source_speaker_id,
            source_speaker_name=message.source_speaker_name,
            direction=MessageDirection(message.direction),
            content_type=message.content_type,
            text=message.text,
        )
