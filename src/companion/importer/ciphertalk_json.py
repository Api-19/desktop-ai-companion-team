import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from companion.importer.errors import ImportValidationError
from companion.importer.models import (
    ImportBatch,
    ImportIssue,
    ImportSeverity,
    MessageDirection,
    SourceChatRecord,
)

CONTENT_TYPES = {
    "文本消息": "text",
    "图片消息": "image",
    "语音消息": "voice",
    "视频消息": "video",
    "动画表情": "emoji",
    "引用消息": "quote",
    "系统消息": "system",
    "其他消息": "other",
}


class _CipherTalkSession(BaseModel):
    model_config = ConfigDict(extra="ignore")

    wxid: str = Field(min_length=1)
    type: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    isGroup: bool
    ownerId: str = Field(min_length=1)


class _CipherTalkMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    localId: str | int | None = None
    platformMessageId: str | int | None = None
    createTime: int | float = Field(gt=0)
    type: str = Field(min_length=1)
    content: str | None = None
    isSend: Literal[0, 1]
    senderUsername: str | None = None
    senderDisplayName: str | None = None


def _safe_field_path(prefix: str, error: ValidationError) -> str:
    first_error = error.errors(include_input=False)[0]
    location = ".".join(str(part) for part in first_error["loc"])
    return f"{prefix}.{location}" if location else prefix


class CipherTalkJsonAdapter:
    def parse(self, path: Path) -> ImportBatch:
        payload = self._read_json(path)
        if not isinstance(payload, dict):
            raise ImportValidationError("invalid_root", "JSON root must be an object")
        if not {"exportInfo", "session", "messages"}.issubset(payload):
            raise ImportValidationError(
                "missing_envelope",
                "required envelope fields are missing",
            )
        if not isinstance(payload["exportInfo"], dict) or not isinstance(payload["messages"], list):
            raise ImportValidationError(
                "missing_envelope",
                "envelope fields have invalid container types",
            )

        session = self._parse_session(payload["session"])
        records: list[SourceChatRecord] = []
        issues: list[ImportIssue] = []
        message_ids: set[str] = set()

        for index, raw_message in enumerate(payload["messages"]):
            message = self._parse_message(raw_message, index)
            message_id = self._message_id(message, index)
            if message_id in message_ids:
                raise ImportValidationError(
                    "duplicate_message_id",
                    "message ID is duplicated",
                    message_index=index,
                    field_path=f"messages[{index}]",
                )
            message_ids.add(message_id)

            direction = (
                MessageDirection.OUTBOUND if message.isSend == 1 else MessageDirection.INBOUND
            )
            if direction is MessageDirection.OUTBOUND:
                source_speaker_id = session.ownerId
                if message.senderUsername and message.senderUsername != session.ownerId:
                    issues.append(
                        ImportIssue(
                            code="sender_direction_conflict",
                            severity=ImportSeverity.WARNING,
                            message_index=index,
                            field_path=f"messages[{index}].senderUsername",
                            detail="outbound sender disagrees with owner metadata",
                        )
                    )
            else:
                source_speaker_id = session.wxid
                if message.senderUsername and message.senderUsername != session.wxid:
                    issues.append(
                        ImportIssue(
                            code="sender_direction_conflict",
                            severity=ImportSeverity.WARNING,
                            message_index=index,
                            field_path=f"messages[{index}].senderUsername",
                            detail="inbound sender disagrees with session metadata",
                        )
                    )

            content_type = CONTENT_TYPES.get(message.type, "other")
            if message.type not in CONTENT_TYPES:
                issues.append(
                    ImportIssue(
                        code="unknown_content_type",
                        severity=ImportSeverity.WARNING,
                        message_index=index,
                        field_path=f"messages[{index}].type",
                        detail="message type was normalized to other",
                    )
                )

            try:
                timestamp = datetime.fromtimestamp(message.createTime, tz=UTC)
            except (OSError, OverflowError, ValueError) as error:
                raise ImportValidationError(
                    "invalid_message",
                    "message timestamp is outside the supported range",
                    message_index=index,
                    field_path=f"messages[{index}].createTime",
                ) from error

            records.append(
                SourceChatRecord(
                    conversation_id=f"wechat:{session.wxid}",
                    message_id=message_id,
                    timestamp=timestamp,
                    source_speaker_id=source_speaker_id,
                    source_speaker_name=message.senderDisplayName,
                    direction=direction,
                    content_type=content_type,
                    text=message.content,
                )
            )

        records.sort(key=lambda record: (record.timestamp, record.message_id))
        return ImportBatch(records=records, issues=issues)

    @staticmethod
    def _read_json(path: Path) -> object:
        try:
            with path.open("r", encoding="utf-8-sig") as source:
                return json.load(source)
        except json.JSONDecodeError as error:
            raise ImportValidationError("invalid_json", "file is not valid JSON") from error
        except OSError as error:
            raise ImportValidationError("file_read_error", "file could not be read") from error

    @staticmethod
    def _parse_session(raw_session: object) -> _CipherTalkSession:
        try:
            session = _CipherTalkSession.model_validate(raw_session)
        except ValidationError as error:
            raise ImportValidationError(
                "invalid_session",
                "session fields are missing or invalid",
                field_path=_safe_field_path("session", error),
            ) from error
        if session.platform != "wechat":
            raise ImportValidationError(
                "unsupported_platform",
                "only the wechat platform is supported",
                field_path="session.platform",
            )
        if session.isGroup or session.type != "私聊":
            raise ImportValidationError(
                "group_chat_unsupported",
                "only private chats are supported",
                field_path="session.isGroup",
            )
        return session

    @staticmethod
    def _parse_message(raw_message: object, index: int) -> _CipherTalkMessage:
        try:
            return _CipherTalkMessage.model_validate(raw_message)
        except ValidationError as error:
            raise ImportValidationError(
                "invalid_message",
                "message fields are missing or invalid",
                message_index=index,
                field_path=_safe_field_path(f"messages[{index}]", error),
            ) from error

    @staticmethod
    def _message_id(message: _CipherTalkMessage, index: int) -> str:
        raw_id = (
            message.platformMessageId if message.platformMessageId is not None else message.localId
        )
        message_id = str(raw_id).strip() if raw_id is not None else ""
        if not message_id:
            raise ImportValidationError(
                "invalid_message",
                "message ID is missing",
                message_index=index,
                field_path=f"messages[{index}].platformMessageId",
            )
        return message_id
