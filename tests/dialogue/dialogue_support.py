from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from companion.contracts import DialogueRequest, StyleReplyGuidance
from companion.dialogue.config import DialogueConfig
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import (
    BudgetConfig,
    ProviderConfig,
    ProviderResult,
    ProviderUsage,
    StyleRuntimeSources,
)
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.style import SingleUserStyleAdapter

FIXED_NOW = datetime(2026, 7, 27, 4, 0, tzinfo=UTC)


def matched_guidance() -> StyleReplyGuidance:
    return StyleReplyGuidance(
        status="matched",
        style_version="synthetic-style-v1",
        category="greeting",
        intent="conversation_reopen",
        examples=["synthetic-style-example"],
        soft_lock_expression="synthetic-soft-lock",
    )


class FakeStyleProvider:
    def __init__(self, guidance: StyleReplyGuidance | None = None) -> None:
        self.guidance = guidance or matched_guidance()
        self.call_count = 0
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def guidance_for(
        self,
        user_id: str,
        text: str,
        recent_context: Sequence[str],
        idle_seconds: int | None,
    ) -> StyleReplyGuidance:
        self.call_count += 1
        self.calls.append(((user_id, text, recent_context, idle_seconds), {}))
        return self.guidance


class FakeProvider:
    def __init__(self, text: str = "synthetic safe answer") -> None:
        self.text = text
        self.call_count = 0
        self.requests: list[tuple[PromptMessage, ...]] = []

    @property
    def system_text(self) -> str:
        return self.requests[-1][0].content

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.call_count += 1
        self.requests.append(tuple(messages))
        return ProviderResult(
            text=self.text,
            usage=ProviderUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class RaisingProvider:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.call_count = 0

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.call_count += 1
        raise self.error


def sample_request() -> DialogueRequest:
    return DialogueRequest(
        user_id="synthetic-user",
        text="synthetic current request",
        recent_context=["synthetic prior context"],
    )


def synthetic_dialogue_config(root: Path) -> DialogueConfig:
    return DialogueConfig(
        provider=ProviderConfig(
            base_url="https://jojocode.com/v1",
            model="gpt-5.6-terra",
            timeout_seconds=120,
        ),
        budget=BudgetConfig(
            monthly_cap_cny=Decimal("100.00"),
            worst_case_request_cny=Decimal("1.00"),
            input_cny_per_million=Decimal("1.00"),
            output_cny_per_million=Decimal("2.00"),
            max_output_tokens=256,
        ),
        style=StyleRuntimeSources(
            candidate_source=root / "style_candidates/current/safe-candidates.json",
            expected_count=7,
            active_style_source=root / "style_runtime/active.json",
            active_manifest_source=root / "style_runtime/manifest.json",
            expected_active_manifest_digest="a" * 64,
            calibration_source=(
                root / "style_candidates/calibration/current/calibration-candidates.json"
            ),
            expected_calibration_fingerprint="b" * 64,
            private_data_dir=root,
            true_name_terms=("synthetic-owner-name",),
            approved_nicknames=("synthetic-approved-nickname",),
        ),
    )


def make_orchestrator(
    root: Path,
    *,
    style: FakeStyleProvider | object | None = None,
    provider: FakeProvider | RaisingProvider | None = None,
    trusted_user_id: str = "synthetic-user",
) -> DialogueOrchestrator:
    config = synthetic_dialogue_config(root)

    # Use provided style adapter or create a real SingleUserStyleAdapter
    if style is None:
        style_adapter = SingleUserStyleAdapter(
            trusted_user_id=trusted_user_id,
            **config.style.as_call_kwargs(),
        )
    else:
        style_adapter = style  # type: ignore[assignment]

    return DialogueOrchestrator(
        config=config,
        style_adapter=style_adapter,
        provider=provider or FakeProvider(),
        ledger=MonthlyUsageLedger(root / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
        test_results_dir=root / "test-results",
    )


def trusted_sample_request_id() -> str:
    """Return the trusted user_id that matches sample_request()."""
    return "synthetic-user"
