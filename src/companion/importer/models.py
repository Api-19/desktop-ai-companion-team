from datetime import datetime
from enum import StrEnum

from pydantic import Field

from companion.contracts.base import ContractModel, TimezoneRequiredModel
from companion.contracts.chat import SpeakerRole


class MessageDirection(StrEnum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"


class ImportSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class SourceChatRecord(TimezoneRequiredModel):
    conversation_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    timestamp: datetime
    source_speaker_id: str = Field(min_length=1)
    source_speaker_name: str | None
    direction: MessageDirection
    content_type: str = Field(min_length=1)
    text: str | None


class ImportIssue(ContractModel):
    code: str = Field(min_length=1)
    severity: ImportSeverity
    message_index: int | None = Field(default=None, ge=0)
    field_path: str | None = None
    detail: str = Field(min_length=1)


class ImportBatch(ContractModel):
    records: list[SourceChatRecord]
    issues: list[ImportIssue]


class SpeakerPreview(ContractModel):
    source_speaker_id: str = Field(min_length=1)
    source_speaker_name: str | None
    message_count: int = Field(ge=0)
    content_type_counts: dict[str, int]
    sample_texts: list[str]
    outbound_count: int = Field(ge=0)
    inbound_count: int = Field(ge=0)


class SpeakerMappingPreview(ContractModel):
    speakers: list[SpeakerPreview]


class ConfirmedSpeakerMapping(ContractModel):
    roles: dict[str, SpeakerRole]
