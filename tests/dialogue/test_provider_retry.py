"""One retry for a single dialogue turn, using synthetic input and fake providers."""

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from dialogue_support import (
    FIXED_NOW,
    FakeProvider,
    FakeStyleProvider,
    make_orchestrator,
    sample_request,
)

from companion.dialogue.ab import ABStatus
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import DialogueTrace, ProviderResult
from companion.dialogue.provider import (
    InvalidProviderResponse,
    OpenAIProviderUnavailable,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailableKind,
    ProviderUnavailableStage,
)
from companion.dialogue.usage import BudgetExceeded


def test_two_provider_calls_trace_round_trips_through_validation() -> None:
    trace = DialogueTrace(
        guidance_status="matched", provider_attempted=True, provider_call_count=2,
    )
    restored = DialogueTrace.model_validate(trace.model_dump())
    assert restored == trace
    assert restored.provider_call_count == 2


def unavailable(kind: ProviderUnavailableKind) -> OpenAIProviderUnavailable:
    return OpenAIProviderUnavailable(kind, ProviderUnavailableStage.TRANSPORT_CALL)


class SequenceProvider(FakeProvider):
    def __init__(self, *outcomes: Exception | str) -> None:
        super().__init__()
        self.outcomes = outcomes

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        result = super().complete(messages)
        outcome = self.outcomes[self.call_count - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return result.model_copy(update={"text": outcome})


@pytest.mark.parametrize("failure", [
    ProviderTimeout("synthetic timeout"),
    *(unavailable(kind) for kind in (
        ProviderUnavailableKind.HTTP_5XX,
        ProviderUnavailableKind.CONNECT_UNAVAILABLE,
        ProviderUnavailableKind.DNS_UNAVAILABLE,
        ProviderUnavailableKind.TLS_UNAVAILABLE,
        ProviderUnavailableKind.TRANSPORT_UNKNOWN,
        ProviderUnavailableKind.HTTP_408_TIMEOUT,
    )),
])
def test_recoverable_failure_retries_same_prompt_and_settles_each_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    provider = SequenceProvider(failure, "synthetic second answer")
    style = FakeStyleProvider()
    runtime = make_orchestrator(tmp_path, style=style, provider=provider)
    reserve = Mock(wraps=runtime._ledger.reserve)
    settle = Mock(wraps=runtime._ledger.settle)
    cancel = Mock(wraps=runtime._ledger.cancel_unused)
    complete = Mock(wraps=runtime._complete_once)
    monkeypatch.setattr(runtime._ledger, "reserve", reserve)
    monkeypatch.setattr(runtime._ledger, "settle", settle)
    monkeypatch.setattr(runtime._ledger, "cancel_unused", cancel)
    monkeypatch.setattr(runtime, "_complete_once", complete)

    run = runtime.reply(sample_request(), trusted_user_id="synthetic-user")

    assert run.response.blocked_reason is None
    assert run.response.text.endswith("synthetic second answer")
    assert run.trace.provider_call_count == provider.call_count == 2
    assert provider.requests[0] == provider.requests[1]
    assert style.call_count == 1
    assert reserve.call_args_list == [call(count=1, now=FIXED_NOW)] * 2
    first, second = [entry.kwargs for entry in complete.call_args_list]
    for key in first.keys() - {"reservation", "attempt_outcome"}:
        assert first[key] is second[key]
    assert first["reservation"] != second["reservation"]
    assert first["attempt_outcome"].provider_failure == (
        failure.kind.value if isinstance(failure, OpenAIProviderUnavailable) else "timeout"
    )
    assert second["attempt_outcome"].provider_failure is None
    unbilled = isinstance(failure, OpenAIProviderUnavailable) and failure.kind in {
        ProviderUnavailableKind.CONNECT_UNAVAILABLE,
        ProviderUnavailableKind.DNS_UNAVAILABLE,
        ProviderUnavailableKind.TLS_UNAVAILABLE,
    }
    assert settle.call_count == (1 if unbilled else 2)
    assert cancel.call_count == int(unbilled)
    first_operation = cancel if unbilled else settle
    assert first_operation.call_args_list[0].args[0] == first["reservation"]
    assert settle.call_args.args[0] == second["reservation"]
    for entry in [*settle.call_args_list, *cancel.call_args_list]:
        assert entry.kwargs["now"] is FIXED_NOW
    snapshot = runtime._ledger.snapshot(FIXED_NOW)
    assert snapshot.reserved_cny == Decimal("0")
    assert snapshot.settled_cny == Decimal("0.000020") + (0 if unbilled else 1)


@pytest.mark.parametrize("failure,blocked_reason", [
    *((unavailable(kind), "provider_unavailable") for kind in (
        ProviderUnavailableKind.HTTP_429_RATE_LIMITED,
        ProviderUnavailableKind.HTTP_401_UNAUTHORIZED,
        ProviderUnavailableKind.HTTP_400_BAD_REQUEST,
        ProviderUnavailableKind.HTTP_403_FORBIDDEN,
        ProviderUnavailableKind.HTTP_404_NOT_FOUND,
        ProviderUnavailableKind.HTTP_4XX_OTHER,
        ProviderUnavailableKind.REDIRECT_REFUSED,
    )),
    (InvalidProviderResponse("synthetic invalid"), "invalid_provider_response"),
    (ProviderError("synthetic provider error"), "provider_unavailable"),
    (RuntimeError("synthetic unexpected error"), "provider_unavailable"),
])
def test_nonrecoverable_failure_returns_original_run_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception, blocked_reason: str,
) -> None:
    provider = SequenceProvider(failure)
    runtime = make_orchestrator(tmp_path, style=FakeStyleProvider(), provider=provider)
    original_complete = runtime._complete_once
    runs = []

    def complete(**kwargs):
        run = original_complete(**kwargs)
        runs.append(run)
        return run

    monkeypatch.setattr(runtime, "_complete_once", complete)
    run = runtime.reply(sample_request(), trusted_user_id="synthetic-user")
    assert provider.call_count == run.trace.provider_call_count == 1
    assert run is runs[0]
    assert run.model_dump_json() == runs[0].model_dump_json()
    assert run.response.blocked_reason == blocked_reason
    assert run.response.billable is True


@pytest.mark.parametrize("second,failure,blocked_reason", [
    ("synthetic recovered answer", None, None),
    (ProviderTimeout("synthetic second timeout"), "timeout", "provider_timeout"),
    (unavailable(ProviderUnavailableKind.HTTP_401_UNAUTHORIZED),
     "http_401_unauthorized", "provider_unavailable"),
    (InvalidProviderResponse("synthetic invalid"), "invalid_response", "invalid_provider_response"),
])
def test_retry_returns_second_run_and_observes_only_last_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    second: Exception | str, failure: str | None, blocked_reason: str | None,
) -> None:
    provider = SequenceProvider(ProviderTimeout("synthetic first timeout"), second)
    runtime = make_orchestrator(tmp_path, style=FakeStyleProvider(), provider=provider)
    observations, runs = [], []
    runtime._turn_observer = observations.append
    original_complete = runtime._complete_once

    def complete(**kwargs):
        run = original_complete(**kwargs)
        runs.append(run)
        return run

    monkeypatch.setattr(runtime, "_complete_once", complete)
    ticks = iter((1.0, 1.25, 2.0, 2.6))
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    run = runtime.reply(sample_request(), trusted_user_id="synthetic-user")

    assert run.trace.provider_call_count == provider.call_count == 2
    assert run.response.blocked_reason == blocked_reason
    assert run == runs[1].model_copy(update={
        "trace": runs[1].trace.model_copy(update={"provider_call_count": 2}),
    })
    assert len(observations) == 1
    assert observations[0].provider_failure == failure
    assert observations[0].provider_elapsed_ms == 600
    assert observations[0].reply_chars == len(run.response.text)
    assert observations[0].blocked_reason == blocked_reason


@pytest.mark.parametrize("error", [
    BudgetExceeded("synthetic exhausted budget"), RuntimeError("synthetic ledger failure"),
])
def test_retry_reservation_failure_returns_first_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    provider = SequenceProvider(ProviderTimeout("synthetic timeout"))
    runtime = make_orchestrator(tmp_path, style=FakeStyleProvider(), provider=provider)
    original_reserve, original_complete = runtime._ledger.reserve, runtime._complete_once
    runs = []

    def reserve(*, count, now):
        if runs:
            raise error
        return original_reserve(count=count, now=now)

    def complete(**kwargs):
        run = original_complete(**kwargs)
        runs.append(run)
        return run

    reserve_spy = Mock(side_effect=reserve)
    monkeypatch.setattr(runtime._ledger, "reserve", reserve_spy)
    monkeypatch.setattr(runtime, "_complete_once", complete)
    run = runtime.reply(sample_request(), trusted_user_id="synthetic-user")
    assert run is runs[0]
    assert provider.call_count == run.trace.provider_call_count == 1
    assert run.response.blocked_reason == "provider_timeout"
    assert reserve_spy.call_args_list == [call(count=1, now=FIXED_NOW)] * 2


def test_ab_timeout_does_not_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SequenceProvider(ProviderTimeout("synthetic A"), ProviderTimeout("synthetic B"))
    runtime = make_orchestrator(tmp_path, style=FakeStyleProvider(), provider=provider)
    reserve = Mock(wraps=runtime._ledger.reserve)
    monkeypatch.setattr(runtime._ledger, "reserve", reserve)
    observations = []
    runtime._turn_observer = observations.append
    pair = runtime.reply_ab("synthetic-retry-case", sample_request(),
                            trusted_user_id="synthetic-user")
    assert pair.status is ABStatus.FAILED
    assert pair.request_attempt_count == provider.call_count == 2
    reserve.assert_called_once_with(count=2, now=FIXED_NOW)
    assert observations == []
