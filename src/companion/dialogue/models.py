from __future__ import annotations

import unicodedata
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from companion.contracts import DialogueResponse, StyleGuidanceStatus
from companion.dialogue.output_guard import OutputBlockReason

MAX_PROMPT_UTF8_BYTES = 48_000
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "[::1]"})


class DialogueMode(StrEnum):
    BASELINE = "baseline"
    STYLED = "styled"


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_total(self) -> Self:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("provider usage total is inconsistent")
        return self


class ProviderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4_000)
    usage: ProviderUsage | None = None


class DialogueTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guidance_status: StyleGuidanceStatus
    style_version: str | None = None
    mode: DialogueMode = DialogueMode.STYLED
    provider_attempted: bool = False
    usage_reliable: bool = False
    used_soft_lock: bool = False
    verifier_enabled: bool = False
    verification_action: Literal["not_run", "accepted", "rejected"] = "not_run"
    provider_call_count: int = Field(default=0, ge=0, le=2)
    style_evidence_count: int | None = Field(default=None, ge=0, le=12)
    output_block_reason: OutputBlockReason | None = None
    correction_attempts: Literal[0] = 0
    additional_provider_calls: Literal[0] = 0


class DialogueRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: DialogueResponse
    trace: DialogueTrace


class DialogueSessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_completed_at: datetime | None = None
    identity_disclosed: bool = False
    previous_soft_lock: str | None = None


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    base_url: AnyHttpUrl
    model: str = Field(min_length=1, max_length=120)
    timeout_seconds: int = Field(ge=1, le=120)
    api_key_env: Literal["OPENAI_API_KEY", "DEEPSEEK_API_KEY"] = "OPENAI_API_KEY"

    @field_validator("base_url", mode="after")
    @classmethod
    def require_secure_base_url(cls, url: AnyHttpUrl) -> AnyHttpUrl:
        if url.username or url.password:
            raise ValueError("provider base_url 不能包含用户信息")
        if url.scheme == "http" and url.host not in LOOPBACK_HOSTS:
            raise ValueError("远端 provider 必须使用 https")
        return url


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    monthly_cap_cny: Decimal
    worst_case_request_cny: Decimal
    input_cny_per_million: Decimal
    output_cny_per_million: Decimal
    max_output_tokens: int = Field(ge=1, le=4_096)

    @model_validator(mode="after")
    def require_safe_reservation(self) -> Self:
        monetary_values = (
            self.monthly_cap_cny,
            self.worst_case_request_cny,
            self.input_cny_per_million,
            self.output_cny_per_million,
        )
        if any(
            not value.is_finite() or value <= 0 for value in monetary_values
        ) or self.monthly_cap_cny != Decimal("100.00"):
            raise ValueError("invalid worst-case reservation")
        minimum_reservation = (
            Decimal(MAX_PROMPT_UTF8_BYTES) * self.input_cny_per_million
            + Decimal(self.max_output_tokens) * self.output_cny_per_million
        ) / Decimal(1_000_000)
        if not minimum_reservation <= self.worst_case_request_cny <= Decimal("100.00"):
            raise ValueError("invalid worst-case reservation")
        return self


class StyleRuntimeSources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_source: Path
    expected_count: int = Field(ge=1)
    active_style_source: Path
    active_manifest_source: Path
    expected_active_manifest_digest: str
    calibration_source: Path
    expected_calibration_fingerprint: str = Field(pattern=r"[0-9a-f]{64}")
    private_data_dir: Path
    true_name_terms: tuple[str, ...]
    approved_nicknames: tuple[str, ...]

    def as_call_kwargs(self) -> dict[str, object]:
        return self.model_dump()


class SnippetStyleSources(BaseModel):
    """Private paired-snippet library, optional habit card and shared-terms dictionary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    library_source: Path
    habit_card_source: Path | None = None
    shared_terms_source: Path | None = None
    session_check: StrictBool = False
    partner_user_id: StrictStr = Field(min_length=1, max_length=120)
    style_version: StrictStr = Field(default="snippet-runtime-v1", min_length=1, max_length=120)

    @field_validator("partner_user_id")
    @classmethod
    def reject_padded_partner(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("invalid partner user id")
        return value


class ProductionStyleSources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    consent_source: Path
    profile_source: Path
    profile_evidence_source: Path
    statistics_source: Path
    profile_index_source: Path
    profile_cache_source: Path
    retrieval_evidence_source: Path
    retrieval_index_source: Path
    retrieval_cache_source: Path
    retrieval_version_source: Path


DEFAULT_CHARACTER_NAME = "kiki"


def validate_character_name(value: object) -> str:
    if not isinstance(value, str) or any(
        unicodedata.category(character) in ("Cc", "Cf") for character in value
    ):
        raise ValueError("invalid character name")
    name = value.strip()
    if not 1 <= len(name) <= 32:
        raise ValueError("invalid character name")
    return name


class DialogueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderConfig
    budget: BudgetConfig
    character_name: str = DEFAULT_CHARACTER_NAME
    style: StyleRuntimeSources | None = None
    style_framework: ProductionStyleSources | None = None
    style_framework_owner_id: StrictStr | None = Field(default=None, repr=False)
    snippet_style: SnippetStyleSources | None = None

    @field_validator("character_name", mode="before")
    @classmethod
    def require_character_name(cls, value: object) -> str:
        return validate_character_name(value)

    @field_validator("style_framework_owner_id", mode="before")
    @classmethod
    def require_exact_framework_owner(cls, value: object) -> object:
        if value is not None and (
            type(value) is not str or not value or value != value.strip() or len(value) > 120
        ):
            raise ValueError("invalid framework owner")
        return value


class DialogueSecrets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr
