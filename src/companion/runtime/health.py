import platform
from typing import Literal

from pydantic import BaseModel, ConfigDict

from companion.runtime.settings import Settings


class HealthStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy"]
    environment: str
    python_version: str


def check_health(settings: Settings) -> HealthStatus:
    return HealthStatus(
        status="healthy",
        environment=settings.environment,
        python_version=platform.python_version(),
    )
