from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from dialogue_support import (
    FIXED_NOW,
    FakeProvider,
    FakeStyleProvider,
    RaisingProvider,
    make_orchestrator,
    matched_guidance,
    sample_request,
    synthetic_dialogue_config,
    trusted_sample_request_id,
)

from companion.contracts import DialogueRequest, StyleReplyGuidance
from companion.dialogue import DialogueMode
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import DialogueSessionState, ProviderResult
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.provider import (
    InvalidProviderResponse,
    OpenAIProviderUnavailable,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailableKind,
    ProviderUnavailableStage,
)
from companion.dialogue.usage import MonthlyUsageLedger, UsageLedgerError

RETRIED_UNAVAILABLE_KINDS = {
    ProviderUnavailableKind.DNS_UNAVAILABLE,
    ProviderUnavailableKind.CONNECT_UNAVAILABLE,
    ProviderUnavailableKind.TLS_UNAVAILABLE,
    ProviderUnavailableKind.HTTP_408_TIMEOUT,
    ProviderUnavailableKind.HTTP_5XX,
    ProviderUnavailableKind.TRANSPORT_UNKNOWN,
}


@pytest.mark.parametrize(
    ("guidance", "expect_style"),
    (
        (matched_guidance(), True),
        (StyleReplyGuidance(status="no_match", style_version="synthetic-v1"), False),
        (StyleReplyGuidance(status="unavailable"), False),
    ),
)
def test_reply_preserves_guidance_state_and_only_styles_matched(
    tmp_path: Path,
    guidance: StyleReplyGuidance,
    expect_style: bool,
) -> None:
    style = FakeStyleProvider(guidance)
    provider = FakeProvider("synthetic safe answer")
    run = make_orchestrator(tmp_path, style=style, provider=provider).reply(
        sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert style.call_count == 1
    assert provider.call_count == 1
    assert run.trace.guidance_status is guidance.status
    assert ("synthetic-style-example" in provider.system_text) is expect_style
    assert run.response.disclosed_ai_identity is True
    assert run.response.billable is True


def test_style_receives_exact_trusted_sources_and_independent_digest(tmp_path: Path) -> None:
    style = FakeStyleProvider()
    config = synthetic_dialogue_config(tmp_path)
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=style,
        provider=FakeProvider(),
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    )

    orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert style.calls == [
        (
            (
                "synthetic-user",  # user_id comes first in new adapter interface
                "synthetic current request",
                ("synthetic prior context",),
                None,
            ),
            {},  # No kwargs with new adapter interface
        )
    ]


@pytest.mark.parametrize(
    ("failure", "blocked_reason"),
    (
        (ProviderTimeout("provider timeout"), "provider_timeout"),
        (ProviderError("provider unavailable"), "provider_unavailable"),
        (InvalidProviderResponse("invalid provider response"), "invalid_provider_response"),
        (RuntimeError("synthetic provider detail"), "provider_unavailable"),
    ),
)
def test_paid_failure_returns_local_fallback_after_allowed_attempts(
    tmp_path: Path,
    failure: Exception,
    blocked_reason: str,
) -> None:
    provider = RaisingProvider(failure)
    run = make_orchestrator(tmp_path, provider=provider).reply(
        sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert provider.call_count == (2 if isinstance(failure, ProviderTimeout) else 1)
    assert run.response.blocked_reason == blocked_reason
    assert run.response.billable is True
    assert "本地模型" not in run.response.text
    assert run.response.disclosed_ai_identity is True


@pytest.mark.parametrize("kind", tuple(ProviderUnavailableKind))
def test_diagnostic_provider_failures_keep_public_fallback_semantics(
    tmp_path: Path, kind: ProviderUnavailableKind
) -> None:
    provider = RaisingProvider(
        OpenAIProviderUnavailable(kind, ProviderUnavailableStage.TRANSPORT_CALL)
    )
    run = make_orchestrator(tmp_path, provider=provider).reply(
        sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert provider.call_count == (2 if kind in RETRIED_UNAVAILABLE_KINDS else 1)
    assert run.response.blocked_reason == "provider_unavailable"
    assert run.response.billable is True
    assert run.response.disclosed_ai_identity is True
    assert run.trace.provider_attempted is True
    state = json.loads(
        (tmp_path / "runtime" / "dialogue-usage-2026-07.json").read_text(encoding="utf-8")
    )
    assert run.trace.usage_reliable is False
    assert "stage" not in run.response.model_dump()
    assert "stage" not in run.trace.model_dump()
    assert state["reservations"] == {}


@pytest.mark.parametrize(
    "kind",
    (
        ProviderUnavailableKind.DNS_UNAVAILABLE,
        ProviderUnavailableKind.CONNECT_UNAVAILABLE,
        ProviderUnavailableKind.TLS_UNAVAILABLE,
        ProviderUnavailableKind.REDIRECT_REFUSED,
        ProviderUnavailableKind.HTTP_400_BAD_REQUEST,
        ProviderUnavailableKind.HTTP_401_UNAUTHORIZED,
        ProviderUnavailableKind.HTTP_403_FORBIDDEN,
        ProviderUnavailableKind.HTTP_404_NOT_FOUND,
        ProviderUnavailableKind.HTTP_429_RATE_LIMITED,
    ),
)
def test_unbilled_provider_failure_releases_reservation_without_increasing_settled(
    tmp_path: Path, kind: ProviderUnavailableKind
) -> None:
    config = synthetic_dialogue_config(tmp_path)
    ledger = MonthlyUsageLedger(tmp_path / "runtime", config.budget)
    (prior,) = ledger.reserve(count=1, now=FIXED_NOW)
    ledger.settle(prior, usage=None, now=FIXED_NOW)
    state_path = tmp_path / "runtime" / "dialogue-usage-2026-07.json"
    before = json.loads(state_path.read_text(encoding="utf-8"))
    provider = RaisingProvider(
        OpenAIProviderUnavailable(kind, ProviderUnavailableStage.TRANSPORT_CALL)
    )
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=ledger,
        clock=lambda: FIXED_NOW,
    )

    run = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    after = json.loads(state_path.read_text(encoding="utf-8"))
    assert after["settled_cny"] == before["settled_cny"]
    assert after["reservations"] == {}
    assert provider.call_count == (2 if kind in RETRIED_UNAVAILABLE_KINDS else 1)
    assert run.response.blocked_reason == "provider_unavailable"
    assert run.response.text.endswith(
        "我这边暂时连不上聊天服务，过会儿再试吧。"  # noqa: RUF001
    )
    assert run.response.billable is True
    assert run.trace.provider_attempted is True
    assert run.trace.usage_reliable is False


@pytest.mark.parametrize(
    ("failure", "blocked_reason"),
    (
        *(
            (
                OpenAIProviderUnavailable(kind, ProviderUnavailableStage.TRANSPORT_CALL),
                "provider_unavailable",
            )
            for kind in (
                ProviderUnavailableKind.HTTP_5XX,
                ProviderUnavailableKind.HTTP_408_TIMEOUT,
                ProviderUnavailableKind.HTTP_4XX_OTHER,
                ProviderUnavailableKind.TRANSPORT_UNKNOWN,
            )
        ),
        (ProviderTimeout("provider timeout"), "provider_timeout"),
        (InvalidProviderResponse("invalid provider response"), "invalid_provider_response"),
        (ProviderError("provider unavailable"), "provider_unavailable"),
        (RuntimeError("synthetic provider detail"), "provider_unavailable"),
    ),
)
def test_uncertain_provider_failure_settles_full_reservation(
    tmp_path: Path, failure: Exception, blocked_reason: str
) -> None:
    provider = RaisingProvider(failure)
    run = make_orchestrator(tmp_path, provider=provider).reply(
        sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    state = json.loads(
        (tmp_path / "runtime" / "dialogue-usage-2026-07.json").read_text(encoding="utf-8")
    )
    attempts = 2 if (
        isinstance(failure, ProviderTimeout)
        or (
            isinstance(failure, OpenAIProviderUnavailable)
            and failure.kind in RETRIED_UNAVAILABLE_KINDS
        )
    ) else 1
    assert state["settled_cny"] == f"{attempts}.000000"
    assert state["reservations"] == {}
    assert provider.call_count == attempts
    assert run.response.blocked_reason == blocked_reason
    assert run.response.billable is True
    assert run.trace.usage_reliable is False


@pytest.mark.parametrize(
    ("failure", "operation"),
    (
        (
            OpenAIProviderUnavailable(
                ProviderUnavailableKind.HTTP_401_UNAUTHORIZED,
                ProviderUnavailableStage.TRANSPORT_CALL,
            ),
            "cancel_unused",
        ),
        (ProviderTimeout("provider timeout"), "settle"),
    ),
)
def test_provider_failure_ledger_error_returns_configuration_fallback_after_allowed_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception, operation: str
) -> None:
    config = synthetic_dialogue_config(tmp_path)
    ledger = MonthlyUsageLedger(tmp_path / "runtime", config.budget)
    settle = Mock(side_effect=UsageLedgerError("SYNTHETIC_LEDGER_FAILURE"))
    cancel_unused = Mock(side_effect=UsageLedgerError("SYNTHETIC_LEDGER_FAILURE"))
    monkeypatch.setattr(ledger, "settle", settle)
    monkeypatch.setattr(ledger, "cancel_unused", cancel_unused)
    provider = RaisingProvider(failure)
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=ledger,
        clock=lambda: FIXED_NOW,
    )

    run = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    called, unused = (
        (cancel_unused, settle) if operation == "cancel_unused" else (settle, cancel_unused)
    )
    attempts = 2 if isinstance(failure, ProviderTimeout) else 1
    assert called.call_count == attempts
    unused.assert_not_called()
    assert called.call_args.kwargs["now"] == FIXED_NOW
    state = json.loads(
        (tmp_path / "runtime" / "dialogue-usage-2026-07.json").read_text(encoding="utf-8")
    )
    assert state["settled_cny"] == "0.000000"
    assert state["reservations"] == {
        entry.args[0].reservation_id: "1.000000" for entry in called.call_args_list
    }
    assert provider.call_count == attempts
    assert run.response.blocked_reason == "configuration_unavailable"
    assert run.response.text.endswith(
        "我这边的设置出了点问题，现在没法正常回复。让主人检查一下吧。"  # noqa: RUF001
    )
    assert run.response.billable is True
    assert run.trace.provider_attempted is True
    assert run.trace.usage_reliable is False
    assert "SYNTHETIC_LEDGER_FAILURE" not in run.model_dump_json()


def test_unsafe_provider_text_is_discarded_without_retry(tmp_path: Path) -> None:
    provider = FakeProvider("我代表主人承诺这笔交易。")
    run = make_orchestrator(tmp_path, provider=provider).reply(
        sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert provider.call_count == 1
    assert run.response.blocked_reason == "unsafe_output"
    assert "这笔交易" not in run.response.text
    assert run.response.billable is True


def test_malformed_provider_result_is_invalid_response_without_retry(tmp_path: Path) -> None:
    class MalformedProvider:
        def __init__(self) -> None:
            self.call_count = 0

        def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
            self.call_count += 1
            return ProviderResult.model_construct(text="", usage=None)

    provider = MalformedProvider()
    config = synthetic_dialogue_config(tmp_path)
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    )

    run = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert provider.call_count == 1
    assert run.response.blocked_reason == "invalid_provider_response"
    assert run.response.billable is True
    assert run.trace.provider_attempted is True


def test_invalid_request_makes_no_style_or_provider_call(tmp_path: Path) -> None:
    style = FakeStyleProvider()
    provider = FakeProvider()
    invalid = DialogueRequest.model_construct(
        user_id="synthetic-user",
        text="",
        recent_context=["synthetic context"],
    )

    run = make_orchestrator(tmp_path, style=style, provider=provider).reply(invalid)

    assert style.call_count == 0
    assert provider.call_count == 0
    assert run.response.blocked_reason == "invalid_request"
    assert run.response.billable is False


def test_budget_rejection_makes_no_provider_call(tmp_path: Path) -> None:
    config = synthetic_dialogue_config(tmp_path)
    ledger = MonthlyUsageLedger(tmp_path / "runtime", config.budget)
    ledger.reserve(count=100, now=FIXED_NOW)
    style = FakeStyleProvider()
    provider = FakeProvider()
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=style,
        provider=provider,
        ledger=ledger,
        clock=lambda: FIXED_NOW,
    )

    run = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert style.call_count == 1
    assert provider.call_count == 0
    assert run.response.blocked_reason == "budget_exceeded"
    assert run.response.billable is False


def test_unexpected_style_failure_makes_no_provider_call(tmp_path: Path) -> None:
    class RaisingStyle:
        def accepts_user(self, user_id: str) -> bool:
            return True

        def guidance_for(self, *args: object, **kwargs: object) -> StyleReplyGuidance:
            raise RuntimeError("SYNTHETIC_STYLE_FAILURE_SENTINEL")

    provider = FakeProvider()
    config = synthetic_dialogue_config(tmp_path)
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=RaisingStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    )

    run = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert provider.call_count == 0
    assert run.response.blocked_reason == "configuration_unavailable"
    assert run.response.billable is False
    assert "SYNTHETIC_STYLE_FAILURE_SENTINEL" not in run.model_dump_json()


def test_missing_usage_settles_full_reservation(tmp_path: Path) -> None:
    class MissingUsageProvider(FakeProvider):
        def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
            self.call_count += 1
            self.requests.append(tuple(messages))
            return ProviderResult(text=self.text, usage=None)

    provider = MissingUsageProvider()
    run = make_orchestrator(tmp_path, provider=provider).reply(
        sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    state = json.loads(
        (tmp_path / "runtime" / "dialogue-usage-2026-07.json").read_text(encoding="utf-8")
    )
    assert run.trace.usage_reliable is False
    assert state["settled_cny"] == "1.000000"
    assert state["reservations"] == {}


def test_same_soft_lock_is_suppressed_then_repeated_output_is_blocked(tmp_path: Path) -> None:
    class SequentialProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self._texts = iter(("synthetic safe answer", "synthetic-soft-lock repeated"))

        def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
            self.text = next(self._texts)
            return super().complete(messages)

    provider = SequentialProvider()
    orchestrator = make_orchestrator(tmp_path, provider=provider, style=FakeStyleProvider())

    first = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())
    second = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert first.trace.used_soft_lock is True
    assert "synthetic-soft-lock" in provider.requests[0][0].content
    assert "synthetic-soft-lock" not in provider.requests[1][0].content
    assert second.response.blocked_reason == "mechanical_style_repeat"
    assert provider.call_count == 2


def test_sessions_are_isolated_by_user_id(tmp_path: Path) -> None:
    provider = FakeProvider()
    orchestrator = make_orchestrator(tmp_path, provider=provider, style=FakeStyleProvider())

    first = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())
    second = orchestrator.reply(
        DialogueRequest(user_id="synthetic-user-2", text="synthetic request 2", recent_context=[]),
        trusted_user_id="synthetic-user-2",
    )

    assert first.response.disclosed_ai_identity is True
    assert second.response.disclosed_ai_identity is True
    assert "synthetic-soft-lock" in provider.requests[0][0].content
    assert "synthetic-soft-lock" in provider.requests[1][0].content


def test_first_fallback_discloses_identity_and_next_success_does_not_repeat(
    tmp_path: Path,
) -> None:
    class FailOnceProvider(FakeProvider):
        def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
            if self.call_count == 0:
                self.call_count += 1
                self.requests.append(tuple(messages))
                raise ProviderError("provider unavailable")
            return super().complete(messages)

    provider = FailOnceProvider()
    orchestrator = make_orchestrator(tmp_path, provider=provider)

    first = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())
    second = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert first.response.disclosed_ai_identity is True
    assert first.response.text.startswith("我是 kiki，一个 AI，")  # noqa: RUF001
    assert second.response.disclosed_ai_identity is False
    assert not second.response.text.startswith("我是 kiki，一个 AI，")  # noqa: RUF001


def test_session_state_contains_only_three_non_content_fields(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(tmp_path)

    orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    state = orchestrator._sessions["synthetic-user"]
    assert set(DialogueSessionState.model_fields) == {
        "last_completed_at",
        "identity_disclosed",
        "previous_soft_lock",
    }
    serialized = state.model_dump_json()
    assert all(
        sentinel not in serialized
        for sentinel in (
            "synthetic current request",
            "synthetic prior context",
            "synthetic safe answer",
            "synthetic-style-example",
        )
    )


def test_idle_seconds_uses_last_completed_time_and_aware_clock(tmp_path: Path) -> None:
    style = FakeStyleProvider()
    now = FIXED_NOW
    config = synthetic_dialogue_config(tmp_path)
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=style,
        provider=FakeProvider(),
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: now,
    )

    orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())
    now += timedelta(seconds=37)
    orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    # New adapter interface: (user_id, text, recent_context, idle_seconds)
    assert style.calls[0][0][3] is None  # idle_seconds is 4th argument
    assert style.calls[1][0][3] == 37


def test_baseline_mode_never_sends_matched_style_material(tmp_path: Path) -> None:
    provider = FakeProvider()

    run = make_orchestrator(tmp_path, provider=provider, style=FakeStyleProvider()).reply(
        sample_request(), mode=DialogueMode.BASELINE, trusted_user_id=trusted_sample_request_id()
    )

    assert "synthetic-style-example" not in provider.system_text
    assert run.trace.mode is DialogueMode.BASELINE
    assert run.trace.used_soft_lock is False


def test_ensure_paid_capacity_is_non_mutating(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(tmp_path)

    orchestrator.ensure_paid_capacity(13)

    assert not (tmp_path / "runtime" / "dialogue-usage-2026-07.json").exists()


def test_orchestrator_enforces_identity_verification_even_without_memory(
    tmp_path: Path,
) -> None:
    """Identity verification is now required even without memory gateway."""
    provider = FakeProvider()
    orchestrator = make_orchestrator(tmp_path, provider=provider)

    # Mismatched identity should fail
    run = orchestrator.reply(sample_request(), trusted_user_id="different-synthetic-user")

    assert run.response.blocked_reason == "untrusted_user"
    assert provider.call_count == 0

    # Matching identity should succeed
    run2 = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert run2.response.blocked_reason is None
    assert provider.call_count == 1
