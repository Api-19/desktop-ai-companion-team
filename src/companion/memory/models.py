from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from companion.contracts.base import TimezoneRequiredModel


class MemoryField(StrEnum):
    PREFERRED_NAME = "preferred_name"
    PREFERENCE = "preference"
    EXPLICIT_NOTE = "explicit_note"


class CreateMemoryRequest(TimezoneRequiredModel):
    user_id: str = Field(min_length=1, max_length=120)
    field: MemoryField
    value: str = Field(min_length=1, max_length=500)
    source_timestamp: datetime
    explicitly_requested: Literal[True]

    @field_validator("explicitly_requested", mode="before")
    @classmethod
    def require_boolean_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("explicit request required")
        return value


class CorrectMemoryRequest(TimezoneRequiredModel):
    user_id: str = Field(min_length=1, max_length=120)
    memory_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    value: str = Field(min_length=1, max_length=500)
    source_timestamp: datetime


class DeletionReceipt(TimezoneRequiredModel):
    user_id: str = Field(min_length=1, max_length=120)
    deleted_count: int = Field(ge=0)
    completed_at: datetime
