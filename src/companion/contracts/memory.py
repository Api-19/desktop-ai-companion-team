from datetime import datetime

from companion.contracts.base import TimezoneRequiredModel


class MemoryEntry(TimezoneRequiredModel):
    memory_id: str
    user_id: str
    field: str
    value: str
    source_timestamp: datetime
    deleted_at: datetime | None
