from datetime import datetime

from companion.contracts.base import TimezoneRequiredModel


class ConsentRecord(TimezoneRequiredModel):
    user_id: str
    granted: bool
    granted_at: datetime | None
    allowed_fields: list[str]
