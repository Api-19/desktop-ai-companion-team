from __future__ import annotations

import logging
import traceback
from collections.abc import Sequence
from pathlib import Path

import pytest
from dialogue_support import (
    FIXED_NOW,
    FakeProvider,
    FakeStyleProvider,
    make_orchestrator,
    matched_guidance,
    synthetic_dialogue_config,
    trusted_sample_request_id,
)

from companion.contracts import DialogueRequest, StyleReplyGuidance
from companion.dialogue.context import (
    PreparedDialogueInput,
    PromptBundle,
    PromptMessage,
)
from companion.dialogue.models import ProviderResult
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.provider import ProviderError
from companion.dialogue.usage import MonthlyUsageLedger


@pytest.mark.parametrize("boundary", ("context", "style", "provider"))
def test_fifty_privacy_sentinels_do_not_escape_public_surfaces(
    tmp_path: Path,
    boundary: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinels = tuple(f"SYNTHETIC_PRIVATE_SENTINEL_{index:02d}" for index in range(50))
    request = DialogueRequest(
        user_id="synthetic-user",
        text="synthetic current request",
        recent_context=[" ".join(sentinels)] if boundary == "context" else [],
    )
    style: object | None = None
    provider: object | None = None

    if boundary == "style":

        class RaisingStyle:
            def accepts_user(self, user_id: str) -> bool:
                return True

            def guidance_for(self, *args: object, **kwargs: object) -> StyleReplyGuidance:
                raise RuntimeError(" ".join(sentinels))

        style = RaisingStyle()
    elif boundary == "provider":

        class PrivateErrorProvider:
            call_count = 0

            def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
                self.call_count += 1
                raise ProviderError(" ".join(sentinels))

        provider = PrivateErrorProvider()
    else:
        provider = FakeProvider("synthetic safe answer")

    caplog.set_level(logging.DEBUG)
    orchestrator = make_orchestrator(
        tmp_path,
        style=style,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )
    run = orchestrator.reply(request, trusted_user_id=trusted_sample_request_id())
    captured = capsys.readouterr()
    public_material = "\n".join(
        (
            run.response.text,
            run.trace.model_dump_json(),
            caplog.text,
            captured.out,
            captured.err,
        )
    )

    assert all(sentinel not in public_material for sentinel in sentinels)
    assert all(sentinel not in str(run.response.blocked_reason) for sentinel in sentinels)


def test_private_provider_text_is_discarded_on_output_guard_failure(tmp_path: Path) -> None:
    sentinel = "SYNTHETIC_PRIVATE_PROVIDER_ANSWER"
    guidance = matched_guidance().model_copy(update={"examples": [sentinel]})
    provider = FakeProvider(f"prefix {sentinel}")

    run = make_orchestrator(
        tmp_path,
        style=type(
            "FixedStyle",
            (),
            {
                "accepts_user": lambda self, user_id: True,
                "guidance_for": (
                    lambda self, user_id, text, recent_context, idle_seconds: guidance
                ),
            },
        )(),
        provider=provider,
    ).reply(
        DialogueRequest(
            user_id="synthetic-user",
            text="synthetic current request",
            recent_context=[],
        ),
        trusted_user_id=trusted_sample_request_id(),
    )

    assert run.response.blocked_reason == "unsafe_output"
    assert sentinel not in run.response.text
    assert sentinel not in run.trace.model_dump_json()


@pytest.mark.parametrize("boundary", ("style", "provider"))
def test_held_boundary_exception_is_detached_from_sensitive_traceback(
    tmp_path: Path,
    boundary: str,
) -> None:
    sentinels = tuple(f"SYNTHETIC_HELD_PRIVATE_{index:02d}" for index in range(50))
    inner = RuntimeError(" ".join(sentinels))
    error = ProviderError("synthetic fixed outer error")
    error.__cause__ = inner
    error.__context__ = inner

    class HeldRaisingStyle:
        def __init__(self) -> None:
            self.error = error

        def accepts_user(self, user_id: str) -> bool:
            return True

        def guidance_for(
            self,
            user_id: str,
            text: str,
            recent_context: Sequence[str],
            idle_seconds: int | None,
        ) -> StyleReplyGuidance:
            raise self.error

    class HeldRaisingProvider:
        def __init__(self) -> None:
            self.error = error
            self.call_count = 0

        def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
            self.call_count += 1
            raise self.error

    config = synthetic_dialogue_config(tmp_path)
    style = HeldRaisingStyle() if boundary == "style" else FakeStyleProvider()
    provider = HeldRaisingProvider() if boundary == "provider" else FakeProvider()
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    )
    request = DialogueRequest(
        user_id="synthetic-user",
        text="synthetic current request",
        recent_context=[" ".join(sentinels)],
    )

    orchestrator.reply(request, trusted_user_id=trusted_sample_request_id())

    _assert_detached_exception(error, sentinels)
    _assert_detached_exception(inner, sentinels)


def _assert_detached_exception(error: BaseException, sentinels: tuple[str, ...]) -> None:
    sensitive_types = (
        PreparedDialogueInput,
        PromptBundle,
        PromptMessage,
        StyleReplyGuidance,
    )
    for frame, _line_number in traceback.walk_tb(error.__traceback__):
        assert not any(isinstance(value, sensitive_types) for value in frame.f_locals.values())
        assert all(
            sentinel not in repr(value)
            for value in frame.f_locals.values()
            for sentinel in sentinels
        )
    assert error.__traceback__ is None
    assert error.__cause__ is None
    assert error.__context__ is None
