from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimezoneRequiredModel(ContractModel):
    @field_validator("*", mode="after")
    @classmethod
    def require_timezone_for_datetimes(cls, value: object) -> object:
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError("datetime values must include a timezone")
        return value
