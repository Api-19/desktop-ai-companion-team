"""Test single-user identity verification for Style integration.

Phase 0: Only one trusted user is allowed. The verified user_id must be
explicitly passed to the Style runtime, and a minimal single-user Style
read adapter must be provided.
"""

from __future__ import annotations

from pathlib import Path

from dialogue_support import (
    FIXED_NOW,
    FakeProvider,
    FakeStyleProvider,
    make_orchestrator,
    matched_guidance,
    sample_request,
    synthetic_dialogue_config,
)

from companion.contracts import DialogueRequest, StyleReplyGuidance
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger


class IdentityCapturingStyleProvider:
    """Fake style provider that captures the user_id parameter."""

    def __init__(self, guidance: StyleReplyGuidance | None = None) -> None:
        self.guidance = guidance or matched_guidance()
        self.call_count = 0
        self.captured_user_ids: list[str | None] = []
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def guidance_for(
        self,
        user_id: str,
        text: str,
        recent_context: list[str],
        idle_seconds: int | None,
    ) -> StyleReplyGuidance:
        self.call_count += 1
        self.calls.append(((user_id, text, recent_context, idle_seconds), {}))
        # Capture user_id (now first positional parameter)
        self.captured_user_ids.append(user_id)
        return self.guidance


def test_reply_without_trusted_user_id_makes_no_style_or_provider_call(
    tmp_path: Path,
) -> None:
    """When trusted_user_id is not provided, fail closed before Style and provider."""
    style = FakeStyleProvider()
    provider = FakeProvider()
    orchestrator = DialogueOrchestrator(
        config=synthetic_dialogue_config(tmp_path),
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", synthetic_dialogue_config(tmp_path).budget),
        clock=lambda: FIXED_NOW,
    )

    run = orchestrator.reply(sample_request(), trusted_user_id=None)

    assert style.call_count == 0, "Style should not be called without trusted_user_id"
    assert provider.call_count == 0, "Provider should not be called without trusted_user_id"
    assert run.response.blocked_reason == "untrusted_user"
    assert run.response.billable is False
    assert "我还没确认你是谁" in run.response.text


def test_reply_with_mismatched_trusted_user_id_makes_no_style_or_provider_call(
    tmp_path: Path,
) -> None:
    """When trusted_user_id does not match request.user_id, fail closed."""
    style = FakeStyleProvider()
    provider = FakeProvider()
    orchestrator = DialogueOrchestrator(
        config=synthetic_dialogue_config(tmp_path),
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", synthetic_dialogue_config(tmp_path).budget),
        clock=lambda: FIXED_NOW,
    )

    request = DialogueRequest(
        user_id="synthetic-user",
        text="synthetic current request",
        recent_context=["synthetic prior context"],
    )
    run = orchestrator.reply(request, trusted_user_id="different-user")

    assert style.call_count == 0, "Style should not be called with mismatched user_id"
    assert provider.call_count == 0, "Provider should not be called with mismatched user_id"
    assert run.response.blocked_reason == "untrusted_user"
    assert run.response.billable is False


def test_reply_with_valid_trusted_user_id_passes_user_id_to_style(
    tmp_path: Path,
) -> None:
    """When trusted_user_id is valid, the verified user_id is passed to Style."""
    style = IdentityCapturingStyleProvider()
    provider = FakeProvider()
    orchestrator = DialogueOrchestrator(
        config=synthetic_dialogue_config(tmp_path),
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", synthetic_dialogue_config(tmp_path).budget),
        clock=lambda: FIXED_NOW,
    )

    request = DialogueRequest(
        user_id="synthetic-user",
        text="synthetic current request",
        recent_context=["synthetic prior context"],
    )
    run = orchestrator.reply(request, trusted_user_id="synthetic-user")

    assert style.call_count == 1, "Style should be called once"
    assert provider.call_count == 1, "Provider should be called once"
    assert len(style.captured_user_ids) == 1
    assert style.captured_user_ids[0] == "synthetic-user", (
        "Verified user_id should be passed to Style"
    )
    assert run.response.blocked_reason is None
    assert run.response.billable is True


def test_reply_ab_without_trusted_user_id_fails(tmp_path: Path) -> None:
    """reply_ab should fail when trusted_user_id is not provided."""
    style = FakeStyleProvider()
    provider = FakeProvider()
    orchestrator = DialogueOrchestrator(
        config=synthetic_dialogue_config(tmp_path),
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", synthetic_dialogue_config(tmp_path).budget),
        clock=lambda: FIXED_NOW,
    )

    pair = orchestrator.reply_ab("test-scenario", sample_request(), trusted_user_id=None)

    assert pair.status == "failed"
    assert style.call_count == 0
    assert provider.call_count == 0


def test_reply_ab_with_valid_trusted_user_id_passes_user_id_to_style(
    tmp_path: Path,
) -> None:
    """reply_ab should pass verified user_id to Style when valid."""
    style = IdentityCapturingStyleProvider()
    provider = FakeProvider()
    orchestrator = DialogueOrchestrator(
        config=synthetic_dialogue_config(tmp_path),
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", synthetic_dialogue_config(tmp_path).budget),
        clock=lambda: FIXED_NOW,
    )

    request = DialogueRequest(
        user_id="synthetic-user",
        text="synthetic current request",
        recent_context=["synthetic prior context"],
    )
    pair = orchestrator.reply_ab("test-scenario", request, trusted_user_id="synthetic-user")

    assert pair.status == "complete"
    assert style.call_count == 1
    assert len(style.captured_user_ids) == 1
    assert style.captured_user_ids[0] == "synthetic-user"


def test_two_different_users_are_isolated(tmp_path: Path) -> None:
    """Two different users should not observe each other's Style data."""
    trusted_user = "trusted-user"
    untrusted_user = "untrusted-user"

    # User 1 makes a successful request
    style1 = IdentityCapturingStyleProvider()
    provider1 = FakeProvider("response for trusted user")
    orchestrator = DialogueOrchestrator(
        config=synthetic_dialogue_config(tmp_path),
        style_adapter=style1,
        provider=provider1,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", synthetic_dialogue_config(tmp_path).budget),
        clock=lambda: FIXED_NOW,
    )

    request1 = DialogueRequest(
        user_id=trusted_user,
        text="request from trusted user",
        recent_context=[],
    )
    run1 = orchestrator.reply(request1, trusted_user_id=trusted_user)
    assert run1.response.blocked_reason is None
    assert style1.call_count == 1
    assert style1.captured_user_ids[0] == trusted_user

    # User 2 tries to access - should fail closed
    style2 = FakeStyleProvider()
    provider2 = FakeProvider("response for untrusted user")
    orchestrator2 = DialogueOrchestrator(
        config=synthetic_dialogue_config(tmp_path),
        style_adapter=style2,
        provider=provider2,
        ledger=MonthlyUsageLedger(
            tmp_path / "runtime2", synthetic_dialogue_config(tmp_path).budget
        ),
        clock=lambda: FIXED_NOW,
    )

    request2 = DialogueRequest(
        user_id=untrusted_user,
        text="request from untrusted user",
        recent_context=[],
    )
    run2 = orchestrator2.reply(request2, trusted_user_id=None)

    # User 2 should be rejected before seeing any Style data
    assert run2.response.blocked_reason == "untrusted_user"
    assert style2.call_count == 0, "Untrusted user should not trigger Style call"
    assert provider2.call_count == 0, "Untrusted user should not trigger provider call"
    assert "trusted user" not in run2.response.text.lower(), (
        "Untrusted user should not see trusted user's data"
    )


def test_verified_non_owner_is_rejected_before_style_memory_or_provider(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    orchestrator = make_orchestrator(
        tmp_path,
        provider=provider,
        trusted_user_id="owner-a",
    )

    run = orchestrator.reply(
        DialogueRequest(user_id="owner-b", text="synthetic request", recent_context=[]),
        trusted_user_id="owner-b",
    )

    assert run.response.blocked_reason == "untrusted_user"
    assert run.trace.provider_attempted is False
    assert provider.call_count == 0
