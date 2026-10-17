from datetime import datetime
from enum import StrEnum

from companion.contracts.base import TimezoneRequiredModel


class SpeakerRole(StrEnum):
    OWNER = "owner"
    OTHER = "other"
    UNKNOWN = "unknown"


class ChatRecord(TimezoneRequiredModel):
    conversation_id: str
    message_id: str
    timestamp: datetime
    speaker_role: SpeakerRole
    content_type: str
    text: str | None
