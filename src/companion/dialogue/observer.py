"""Optional request-local observation for local dialogue debugging."""

from typing import Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class TurnObservation(BaseModel):
    """Final turn status, counts and provider diagnostics without content or identity.

    Invalid input is represented by zero input counts. The timestamp is the UTC
    wall clock at observation, independent of the dialogue accounting clock.
    Observations are passed to the callback only; dialogue does not persist them.
    Provider diagnostics cover this turn's complete call only: elapsed time is
    rounded monotonic milliseconds. The hash covers the system prompt sent to
    the provider; all provider diagnostics are None when no call occurs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    blocked_reason: str | None
    input_chars: int = Field(ge=0)
    context_count: int = Field(ge=0)
    context_chars: int = Field(ge=0)
    reply_chars: int = Field(ge=0)
    similar_count: int = Field(ge=0)
    general_count: int = Field(ge=0)
    profile_count: int = Field(ge=0)
    provider_attempted: bool
    provider_failure: str | None = None
    provider_elapsed_ms: int | None = None
    system_prompt_sha256: str | None = None
    observed_at: AwareDatetime


class TurnObserver(Protocol):
    def __call__(self, observation: TurnObservation) -> None: ...
