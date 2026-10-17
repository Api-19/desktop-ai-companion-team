"""Local partner profile records and the minimal dialogue projection."""

from calendar import monthrange
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from companion.contracts.base import ContractModel, TimezoneRequiredModel

PROFILE_PERMISSION = "profile"


class ProfileCategory(StrEnum):
    HABIT = "habit"
    PREFERENCE = "preference"
    REMINDER_PREF = "reminder_pref"


class ProfileSource(StrEnum):
    SELF_STATED = "self_stated"
    OWNER_ADDED = "owner_added"


class ProfileStatus(StrEnum):
    CANDIDATE = "candidate"
    OWNER_APPROVED = "owner_approved"
    CONFIRMED = "confirmed"
    DISABLED = "disabled"


class ProfileView(ContractModel):
    topic: str = Field(min_length=1, max_length=12)
    text: str = Field(min_length=1, max_length=60)
    category: ProfileCategory

    @field_validator("topic", "text")
    @classmethod
    def require_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("profile content required")
        return value


class ProfileItem(TimezoneRequiredModel, ProfileView):
    profile_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    user_id: str = Field(min_length=1, max_length=120)
    source: ProfileSource
    evidence_date: date
    evidence_quote: str | None = Field(default=None, max_length=30)
    confidence: Literal["low", "med", "high"]
    status: ProfileStatus = ProfileStatus.CANDIDATE
    version: int = Field(default=1, ge=1, strict=True)
    confirmed_at: datetime | None = None
    confirmed_by: str | None = None
    confirmed_version: int | None = Field(default=None, ge=1, strict=True)
    use_in_chat: bool = Field(default=False, strict=True)
    simulated: bool = Field(default=False, strict=True)
    disabled_reason: Literal["owner", "self", "conflict", "expired"] | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def valid_until(self) -> date | None:
        if self.confirmed_at is None:
            return None
        start = max(self.evidence_date, self.confirmed_at.date())
        month_index = start.year * 12 + start.month - 1 + 4
        year, month = divmod(month_index, 12)
        month += 1
        return date(year, month, min(start.day, monthrange(year, month)[1]))


def profile_overlap(item: ProfileItem, query_bigrams: set[str]) -> int:
    return len((bigrams(item.topic) | bigrams(item.text)) & query_bigrams)


def bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}
