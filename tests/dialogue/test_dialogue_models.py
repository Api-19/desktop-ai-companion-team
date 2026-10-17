from decimal import Decimal

import pytest
from pydantic import ValidationError

from companion.dialogue.models import BudgetConfig, DialogueTrace, ProviderUsage


def test_provider_usage_requires_consistent_total() -> None:
    with pytest.raises(ValidationError, match="provider usage total is inconsistent"):
        ProviderUsage(prompt_tokens=3, completion_tokens=5, total_tokens=7)


def test_budget_config_rejects_reservation_below_bounded_prompt_cost() -> None:
    with pytest.raises(ValidationError, match="invalid worst-case reservation"):
        BudgetConfig(
            monthly_cap_cny=Decimal("100.00"),
            worst_case_request_cny=Decimal("0.01"),
            input_cny_per_million=Decimal("1.00"),
            output_cny_per_million=Decimal("2.00"),
            max_output_tokens=256,
        )


def test_dialogue_trace_preserves_no_match_and_unavailable() -> None:
    assert DialogueTrace(guidance_status="no_match").guidance_status.value == "no_match"
    assert DialogueTrace(guidance_status="unavailable").guidance_status.value == "unavailable"
