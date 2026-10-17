# ruff: noqa: RUF001

"""Turn observation and partner-only retrieval with synthetic inputs and fake providers."""

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from dialogue_support import (
    FIXED_NOW,
    FakeProvider,
    FakeStyleProvider,
    RaisingProvider,
    synthetic_dialogue_config,
)
from pydantic import SecretStr, ValidationError
from test_partner_profile_dialogue import FakeProfileSource
from test_snippet_style import FakeSnippetStyle

import companion.dialogue as dialogue
import companion.dialogue.composition as composition
from companion.contracts import DialogueRequest
from companion.dialogue import DialogueMode
from companion.dialogue.context import CONTEXT_END, CONTEXT_START
from companion.dialogue.models import DialogueSecrets
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.provider import (
    InvalidProviderResponse,
    OpenAIProviderUnavailable,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailableKind,
    ProviderUnavailableStage,
)
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime.settings import Settings


def request(recent_context: Sequence[str] = ("对方：今天好累", "你：嘿嘿。")):
    return DialogueRequest(
        user_id="synthetic-user", text="想睡觉", recent_context=list(recent_context)
    )


def build(root: Path, provider, style, source=None, **kwargs):
    config = synthetic_dialogue_config(root)
    return DialogueOrchestrator(
        config=config,
        snippet_style=style,
        style_adapter=FakeStyleProvider() if style is None else None,
        profile_source=source,
        provider=provider,
        ledger=MonthlyUsageLedger(root / "synthetic-ledger", config.budget),
        clock=lambda: FIXED_NOW,
        **kwargs,
    )


def observation_values():
    return dict(
        blocked_reason=None, input_chars=0, context_count=0, context_chars=0,
        reply_chars=0, similar_count=0, general_count=0, profile_count=0,
        provider_attempted=False, observed_at=FIXED_NOW,
    )


def test_observation_schema_contains_only_diagnostics() -> None:
    counts = {
        "input_chars", "context_count", "context_chars", "reply_chars",
        "similar_count", "general_count", "profile_count",
    }
    assert set(dialogue.TurnObservation.model_fields) == counts | {
        "blocked_reason", "provider_attempted", "provider_failure",
        "provider_elapsed_ms", "system_prompt_sha256", "observed_at",
    }
    observed = dialogue.TurnObservation(**observation_values())
    assert observed.provider_failure is None
    assert observed.provider_elapsed_ms is None
    assert observed.system_prompt_sha256 is None
    for field in counts:
        assert getattr(observed, field) == 0
        positive = dialogue.TurnObservation(**{**observation_values(), field: 1})
        assert getattr(positive, field) == 1
        with pytest.raises(ValidationError) as raised:
            dialogue.TurnObservation(**{**observation_values(), field: -1})
        assert [(error["loc"], error["type"]) for error in raised.value.errors()] == [
            ((field,), "greater_than_equal"),
        ]


@pytest.mark.parametrize("field,value", [
    ("user_id", "synthetic-user"), ("text", "合成输入"),
    ("recent_context", ()), ("reply_text", "合成回复"),
    ("similar", ()), ("general", ()), ("profile_topics", ()),
])
def test_observation_rejects_removed_content_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as raised:
        dialogue.TurnObservation(**{**observation_values(), field: value})
    assert [(error["loc"], error["type"]) for error in raised.value.errors()] == [
        ((field,), "extra_forbidden"),
    ]


def test_observation_records_provider_system_prompt_hash(tmp_path: Path) -> None:
    provider, observations = FakeProvider(), []
    runtime = build(
        tmp_path, provider, FakeSnippetStyle(), turn_observer=observations.append,
    )
    run = runtime.reply(request(), trusted_user_id="synthetic-user")
    assert run.response.blocked_reason is None
    assert provider.call_count == 1
    assert observations[0].system_prompt_sha256 == hashlib.sha256(
        provider.requests[0][0].content.encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("case,reason", [
    ("invalid", "invalid_request"),
    ("untrusted", "untrusted_user"),
    ("budget", "budget_exceeded"),
])
def test_local_returns_have_no_prompt_hash(
    tmp_path: Path, case: str, reason: str,
) -> None:
    provider, observations = FakeProvider(), []
    if case == "budget":
        config = synthetic_dialogue_config(tmp_path)
        ledger = MonthlyUsageLedger(tmp_path / "synthetic-ledger", config.budget)
        ledger.reserve(count=100, now=FIXED_NOW)
    runtime = build(
        tmp_path, provider, FakeSnippetStyle(), turn_observer=observations.append,
    )
    current = request()
    if case == "invalid":
        current = current.model_copy(update={"text": ""})
    trusted = "synthetic-other" if case == "untrusted" else "synthetic-user"
    run = runtime.reply(current, trusted_user_id=trusted)
    assert run.response.blocked_reason == reason
    assert provider.call_count == 0
    assert len(observations) == 1
    assert observations[0].system_prompt_sha256 is None


def test_provider_failure_keeps_prompt_hash(tmp_path: Path) -> None:
    class RecordingTimeoutProvider(FakeProvider):
        def complete(self, messages):
            super().complete(messages)
            raise ProviderTimeout("synthetic timeout")

    provider, observations = RecordingTimeoutProvider(), []
    runtime = build(
        tmp_path, provider, FakeSnippetStyle(), turn_observer=observations.append,
    )
    run = runtime.reply(request(), trusted_user_id="synthetic-user")
    assert run.response.blocked_reason == "provider_timeout"
    assert provider.call_count > 0
    assert len(observations) == 1
    assert observations[0].system_prompt_sha256 == hashlib.sha256(
        provider.requests[-1][0].content.encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("recent,filtered", [
    (("对方：今天好累", "你：嘿嘿。"), ("今天好累",)),
    (("她：今天好累 / 他：早点睡",), ("她：今天好累 / 他：早点睡",)),
    (("你：旧回复", "对方：旧消息", "对方：最近一", "你：新回复", "最近二"),
     ("旧消息", "最近一", "最近二")),
    (("你：嘿嘿。",), ()),
    (("你:半角冒号", " 对方：前导空格", "合成普通前文"),
     ("你:半角冒号", " 对方：前导空格", "合成普通前文")),
])
def test_retrieval_filters_partner_context_but_prompt_preserves_it(
    tmp_path: Path, recent: tuple[str, ...], filtered: tuple[str, ...]
) -> None:
    style, source, provider = FakeSnippetStyle(), FakeProfileSource(), FakeProvider()
    run = build(tmp_path, provider, style, source).reply(
        request(recent), trusted_user_id="synthetic-user"
    )
    assert run.response.blocked_reason is None
    assert style.calls == [("synthetic-user", "想睡觉", filtered)]
    assert source.calls == [("synthetic-user", ["想睡觉", *filtered[-2:]], 3)]
    block = provider.system_text.split(CONTEXT_START)[1].split(CONTEXT_END)[0]
    for item in recent:
        assert item in block


@pytest.mark.parametrize("case,reason", [
    ("normal", None),
    ("blocked", "profile_assertion"),
    ("failure", "provider_timeout"),
])
def test_observer_gets_final_reply_and_retrieval_once(
    tmp_path: Path, case: str, reason: str | None
) -> None:
    provider = (
        RaisingProvider(ProviderTimeout("synthetic timeout"))
        if case == "failure" else FakeProvider("你又没喝水吧" if case == "blocked" else "晚安")
    )
    style, source, observations = FakeSnippetStyle(), FakeProfileSource(), []
    runtime = build(tmp_path, provider, style, source, turn_observer=observations.append)
    before = datetime.now(UTC)
    run = runtime.reply(request(), trusted_user_id="synthetic-user")
    assert len(observations) == 1
    observed = observations[0]
    assert isinstance(observed, dialogue.TurnObservation)
    assert observed.input_chars == 3
    assert observed.context_count == 2
    assert observed.context_chars == 12
    assert observed.reply_chars == len(run.response.text)
    assert observed.blocked_reason == run.response.blocked_reason == reason
    assert observed.similar_count == 1
    assert observed.general_count == 1
    assert observed.profile_count == 1
    assert observed.provider_attempted is run.trace.provider_attempted is True
    assert len(style.calls) == len(source.calls) == 1
    assert before <= observed.observed_at <= datetime.now(UTC)
    assert observed.observed_at.utcoffset() is not None


@pytest.mark.parametrize("case,reason", [
    ("untrusted", "untrusted_user"),
    ("memory_command", "memory_command_invalid"),
    ("clock", "configuration_unavailable"),
    ("snippet_failure", "configuration_unavailable"),
    ("invalid", "invalid_request"),
])
def test_local_returns_are_observed_without_prompt_evidence(
    tmp_path: Path, case: str, reason: str
) -> None:
    observations, source = [], FakeProfileSource()
    style = FakeSnippetStyle(error=RuntimeError("synthetic failure") if
                             case == "snippet_failure" else None)
    runtime = build(tmp_path, FakeProvider(), style, source, turn_observer=observations.append)
    current = request()
    trusted = "synthetic-other" if case == "untrusted" else "synthetic-user"
    if case == "memory_command":
        runtime._memory = object()
        current = current.model_copy(update={"text": "记住"})
    if case == "clock":
        runtime._clock = lambda: datetime(2026, 10, 12)
    if case == "invalid":
        current = current.model_copy(update={"text": ""})
    run = runtime.reply(current, trusted_user_id=trusted)
    assert len(observations) == 1
    observed = observations[0]
    assert observed.blocked_reason == run.response.blocked_reason == reason
    assert observed.reply_chars == len(run.response.text)
    assert observed.similar_count == observed.general_count == observed.profile_count == 0
    assert observed.provider_attempted is run.trace.provider_attempted is False
    if case == "invalid":
        assert observed.input_chars == observed.context_count == observed.context_chars == 0
    else:
        assert observed.input_chars == len(current.text)
        assert observed.context_count == 2
        assert observed.context_chars == 12
    assert observed.provider_failure is None
    assert observed.provider_elapsed_ms is None


@pytest.mark.parametrize("legacy,mode", [
    (True, DialogueMode.STYLED), (False, DialogueMode.BASELINE),
])
def test_no_snippet_evidence_observes_zero_counts(
    tmp_path: Path, legacy: bool, mode: DialogueMode
) -> None:
    observations = []
    runtime = build(
        tmp_path, FakeProvider(), None if legacy else FakeSnippetStyle(), FakeProfileSource(),
        turn_observer=observations.append,
    )
    run = runtime.reply(request(), mode, trusted_user_id="synthetic-user")
    assert run.response.blocked_reason is None
    assert len(observations) == 1
    assert observations[0].similar_count == observations[0].general_count == 0
    assert observations[0].profile_count == (0 if legacy else 1)


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_observer_failure_keeps_prompt_response_and_trace_unchanged(
    tmp_path: Path, error_type: type[Exception]
) -> None:
    calls = []

    def raises(observed):
        calls.append(observed)
        raise error_type("synthetic observer failure")

    providers = [FakeProvider(), FakeProvider()]
    plain = build(tmp_path / "plain", providers[0], FakeSnippetStyle(), turn_observer=None)
    observed = build(tmp_path / "observed", providers[1], FakeSnippetStyle(), turn_observer=raises)
    current = request(("她：合成 / 他：合成", "合成普通前文"))
    expected = plain.reply(current, trusted_user_id="synthetic-user")
    actual = observed.reply(current, trusted_user_id="synthetic-user")
    assert actual == expected
    assert providers[0].requests == providers[1].requests
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_observer_interrupt_propagates(tmp_path: Path, error_type: type[BaseException]) -> None:
    calls = []
    error = error_type("synthetic observer interrupt")

    def raises(observed):
        calls.append(observed)
        raise error

    provider = FakeProvider()
    runtime = build(tmp_path, provider, FakeSnippetStyle(), turn_observer=raises)
    with pytest.raises(error_type) as raised:
        runtime.reply(request(), trusted_user_id="synthetic-user")
    assert raised.value is error
    assert len(calls) == 1
    assert provider.call_count == 1


def test_observation_is_frozen_and_requires_aware_timestamp() -> None:
    values = observation_values()
    observed = dialogue.TurnObservation(**values)
    with pytest.raises(ValidationError, match="frozen"):
        observed.input_chars = 1
    with pytest.raises(ValidationError, match="timezone"):
        dialogue.TurnObservation(**{**values, "observed_at": datetime(2026, 10, 12)})


def test_builder_forwards_observer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, observations = FakeProvider(), []
    monkeypatch.setattr(composition, "OpenAICompatibleProvider", lambda **_: provider)
    settings = Settings(
        environment="test", log_level="INFO", private_data_dir=tmp_path,
        runtime_data_dir=tmp_path / "synthetic-runtime",
        test_results_dir=tmp_path / "synthetic-results",
    )
    runtime = composition.build_real_orchestrator(
        settings, synthetic_dialogue_config(tmp_path),
        DialogueSecrets(api_key=SecretStr("synthetic-key")),
        trusted_user_id="synthetic-user", snippet_style=FakeSnippetStyle(),
        turn_observer=observations.append,
    )
    runtime._clock = lambda: FIXED_NOW
    run = runtime.reply(request(), trusted_user_id="synthetic-user")
    assert len(observations) == 1
    assert observations[0].reply_chars == len(run.response.text)
    assert observations[0].input_chars == 3
    assert observations[0].context_count == 2
    assert observations[0].context_chars == 12
    assert observations[0].similar_count == observations[0].general_count == 1
    assert observations[0].profile_count == 0


def test_observation_evidence_is_not_carried_to_next_turn(tmp_path: Path) -> None:
    observations = []
    runtime = build(
        tmp_path, FakeProvider(), FakeSnippetStyle(), FakeProfileSource(),
        turn_observer=observations.append,
    )
    runtime.reply(request(), trusted_user_id="synthetic-user")
    runtime.reply(request(()), trusted_user_id="synthetic-other")
    runtime.reply(request().model_copy(update={"text": ""}), trusted_user_id="synthetic-user")
    assert len(observations) == 3
    assert observations[0].similar_count == observations[0].general_count == 1
    assert observations[0].profile_count == 1
    assert observations[0].context_count == 2
    assert observations[0].context_chars == 12
    assert observations[1].input_chars == 3
    assert observations[1].context_count == observations[1].context_chars == 0
    assert observations[1].similar_count == observations[1].general_count == 0
    assert observations[1].profile_count == 0
    assert observations[2].input_chars == 0
    assert observations[2].context_count == observations[2].context_chars == 0
    assert observations[2].similar_count == observations[2].general_count == 0
    assert observations[2].profile_count == 0
    assert observations[0].provider_elapsed_ms is not None
    assert observations[1].provider_failure is None
    assert observations[1].provider_elapsed_ms is None


@pytest.mark.parametrize("elapsed,expected_ms", [(0.0, 0), (0.25, 250), (0.2506, 251)])
def test_observer_records_successful_provider_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, elapsed: float, expected_ms: int
) -> None:
    observations = []
    runtime = build(tmp_path, FakeProvider(), None, turn_observer=observations.append)
    ticks = iter((1.0, 1.0 + elapsed))
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    run = runtime.reply(request(), trusted_user_id="synthetic-user")
    observed = observations[0]
    assert run.response.blocked_reason is None
    assert observed.provider_failure is None
    assert type(observed.provider_elapsed_ms) is int
    assert observed.provider_elapsed_ms == expected_ms
    assert "provider_failure" not in run.trace.model_dump()
    assert "provider_elapsed_ms" not in run.trace.model_dump()


@pytest.mark.parametrize("error,failure,blocked_reason", [
    (ProviderTimeout("synthetic"), "timeout", "provider_timeout"),
    (OpenAIProviderUnavailable(
        ProviderUnavailableKind.CONNECT_UNAVAILABLE, ProviderUnavailableStage.OPENER_OPEN,
    ), "connect_unavailable", "provider_unavailable"),
    (OpenAIProviderUnavailable(
        ProviderUnavailableKind.HTTP_5XX, ProviderUnavailableStage.OPENER_OPEN,
    ), "http_5xx", "provider_unavailable"),
    (InvalidProviderResponse("synthetic"), "invalid_response", "invalid_provider_response"),
    (ProviderError("synthetic"), "provider_error", "provider_unavailable"),
    (RuntimeError("synthetic"), "unexpected_error", "provider_unavailable"),
])
def test_observer_records_provider_failure_without_changing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    error: Exception, failure: str, blocked_reason: str,
) -> None:
    observations = []
    plain = build(tmp_path / "plain", RaisingProvider(error), None, turn_observer=None)
    runtime = build(
        tmp_path / "observed", RaisingProvider(error), None, turn_observer=observations.append,
    )
    expected = plain.reply(request(), trusted_user_id="synthetic-user")
    ticks = iter((1.0, 1.25, 2.0, 2.25))
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    actual = runtime.reply(request(), trusted_user_id="synthetic-user")
    assert actual == expected
    assert actual.response.blocked_reason == blocked_reason
    assert observations[0].provider_failure == failure
    assert observations[0].provider_elapsed_ms == 250


def test_observer_records_invalid_provider_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider()
    candidate = provider.complete(())
    candidate = candidate.model_copy(update={"text": ""})
    monkeypatch.setattr(provider, "complete", lambda _: candidate)
    observations = []
    runtime = build(tmp_path, provider, None, turn_observer=observations.append)
    ticks = iter((1.0, 1.25))
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    run = runtime.reply(request(), trusted_user_id="synthetic-user")
    assert run.response.blocked_reason == "invalid_provider_response"
    assert observations[0].provider_failure == "invalid_response"
    assert observations[0].provider_elapsed_ms == 250
