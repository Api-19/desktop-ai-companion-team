from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from companion.contracts import (
    DialogueRequest,
    StyleGuidanceStatus,
    StyleIntent,
    StyleReplyGuidance,
)
from companion.dialogue.ab import load_ab_scenarios, load_style_refresh_scenarios
from companion.dialogue.context import build_prompt, prepare_dialogue_input
from companion.dialogue.models import DialogueConfig, DialogueMode
from companion.dialogue.orchestrator import StyleAdapter
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime.settings import Settings
from companion.style import SingleUserStyleAdapter

_EXPECTED_REFRESH_INTENTS = {
    "meals-2": StyleIntent.NOT_EATEN_YET_CARE,
    "sleep-1": StyleIntent.SLEEP_FAREWELL,
    "sleep-2": StyleIntent.SLEEP_REMINDER,
    "weather-2": StyleIntent.RAIN_REMINDER,
    "household-1": StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
    "shopping-2": StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
    "gentle_affection-2": StyleIntent.GOING_OUT_CARE,
}


class StyleRefreshPreflightSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready", "not_ready"]
    fixed_scenario_count: int = Field(ge=0, le=20)
    matched_count: int = Field(ge=0, le=20)
    no_match_count: int = Field(ge=0, le=20)
    unavailable_count: int = Field(ge=0, le=20)
    refresh_scenario_count: int = Field(ge=0, le=7)
    refresh_precise_count: int = Field(ge=0, le=7)
    prompt_pair_count: int = Field(ge=0, le=7)
    request_capacity: Literal[15] = 15
    budget_capacity: bool
    provider_requests: Literal[0] = 0


class _PreflightStyleAdapter(StyleAdapter, Protocol):
    """Style adapter contract requiring an explicit bound owner for preflight."""

    @property
    def trusted_user_id(self) -> str: ...


def preflight_style_refresh(
    settings: Settings,
    config: DialogueConfig,
    *,
    style_adapter: _PreflightStyleAdapter | None = None,
) -> StyleRefreshPreflightSummary:
    """
    Validate Style refresh scenarios for production readiness.

    Args:
        settings: Runtime settings.
        config: Dialogue configuration.
        style_adapter: Optional Style adapter. If None, creates a SingleUserStyleAdapter
                      with a synthetic owner ID for validation.
    """
    fixed_count = matched = no_match = unavailable = 0
    refresh_count = refresh_precise = prompt_pairs = 0
    budget_capacity = False
    try:
        # Create default adapter if not provided
        if style_adapter is None:
            style_adapter = SingleUserStyleAdapter(
                trusted_user_id="synthetic-preflight-owner",
                **config.style.as_call_kwargs(),
            )
        trusted_user_id = style_adapter.trusted_user_id
        if (
            type(trusted_user_id) is not str
            or not trusted_user_id.strip()
            or len(trusted_user_id) > 120
            or not style_adapter.accepts_user(trusted_user_id)
        ):
            raise ValueError("style adapter owner is invalid")
        full = load_ab_scenarios(config_path("dialogue-ab-full-scenarios.json"))
        refresh = load_style_refresh_scenarios(
            config_path("dialogue-ab-full-scenarios.json"),
            config_path("dialogue-ab-style-refresh-scenarios.json"),
        )
        fixed_count = len(full)
        refresh_count = len(refresh)
        guidance_by_id: dict[str, StyleReplyGuidance] = {}
        for scenario in full:
            guidance = style_adapter.guidance_for(
                trusted_user_id,
                scenario.text,
                scenario.recent_context,
                scenario.idle_seconds,
            )
            guidance_by_id[scenario.scenario_id] = guidance
            if guidance.status is StyleGuidanceStatus.MATCHED:
                matched += 1
            elif guidance.status is StyleGuidanceStatus.NO_MATCH:
                no_match += 1
            else:
                unavailable += 1

        for scenario in refresh:
            guidance = guidance_by_id[scenario.scenario_id]
            if guidance.intent is _EXPECTED_REFRESH_INTENTS[scenario.scenario_id]:
                refresh_precise += 1
            prepared = prepare_dialogue_input(
                DialogueRequest(
                    user_id=trusted_user_id,
                    text=scenario.text,
                    recent_context=list(scenario.recent_context),
                )
            )
            baseline = build_prompt(prepared, guidance, DialogueMode.BASELINE, None)
            styled = build_prompt(prepared, guidance, DialogueMode.STYLED, None)
            if len(baseline.messages) != 2 or len(styled.messages) != 2:
                raise ValueError
            prompt_pairs += 1

        MonthlyUsageLedger(
            settings.runtime_data_dir / "dialogue",
            config.budget,
        ).ensure_available(15, datetime.now(UTC))
        budget_capacity = True
        ready = (
            fixed_count == matched == 20
            and no_match == unavailable == 0
            and refresh_count == refresh_precise == prompt_pairs == 7
        )
        return _summary(
            status="ready" if ready else "not_ready",
            fixed_count=fixed_count,
            matched=matched,
            no_match=no_match,
            unavailable=unavailable,
            refresh_count=refresh_count,
            refresh_precise=refresh_precise,
            prompt_pairs=prompt_pairs,
            budget_capacity=budget_capacity,
        )
    except Exception:
        return _summary(
            status="not_ready",
            fixed_count=fixed_count,
            matched=matched,
            no_match=no_match,
            unavailable=unavailable,
            refresh_count=refresh_count,
            refresh_precise=refresh_precise,
            prompt_pairs=prompt_pairs,
            budget_capacity=budget_capacity,
        )


def config_path(filename: str) -> Path:
    return Path("config") / filename


def _summary(
    *,
    status: Literal["ready", "not_ready"],
    fixed_count: int,
    matched: int,
    no_match: int,
    unavailable: int,
    refresh_count: int,
    refresh_precise: int,
    prompt_pairs: int,
    budget_capacity: bool,
) -> StyleRefreshPreflightSummary:
    return StyleRefreshPreflightSummary(
        status=status,
        fixed_scenario_count=fixed_count,
        matched_count=matched,
        no_match_count=no_match,
        unavailable_count=unavailable,
        refresh_scenario_count=refresh_count,
        refresh_precise_count=refresh_precise,
        prompt_pair_count=prompt_pairs,
        budget_capacity=budget_capacity,
    )
