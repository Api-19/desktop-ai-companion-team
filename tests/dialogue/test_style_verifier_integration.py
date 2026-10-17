from __future__ import annotations

# ruff: noqa: RUF001
import json
from pathlib import Path
from typing import Never

import pytest
from dialogue_support import (
    FIXED_NOW,
    FakeProvider,
    FakeStyleProvider,
    sample_request,
    synthetic_dialogue_config,
    trusted_sample_request_id,
)
from pydantic import SecretStr

from companion.dialogue.ab import build_real_orchestrator
from companion.dialogue.models import (
    DialogueMode,
    DialogueSecrets,
    DialogueSessionState,
)
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime.settings import Settings
from companion.style import DeterministicStyleVerifier


def _orchestrator(
    root: Path,
    *,
    provider: FakeProvider,
    enabled: bool,
) -> DialogueOrchestrator:
    config = synthetic_dialogue_config(root)
    return DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=MonthlyUsageLedger(root / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
        style_verifier=(
            DeterministicStyleVerifier(trusted_user_id="synthetic-user") if enabled else None
        ),
    )


def test_verifier_is_disabled_by_default_and_keeps_old_call_shape(tmp_path: Path) -> None:
    config = synthetic_dialogue_config(tmp_path)
    provider = FakeProvider("以下是一些建议，希望这些信息对您有所帮助。")
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    )

    run = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert run.response.blocked_reason is None
    assert run.trace.verifier_enabled is False
    assert run.trace.verification_action == "not_run"
    assert run.trace.provider_call_count == 1


def test_enabled_verifier_rejects_locally_without_retry_and_settles_once(tmp_path: Path) -> None:
    provider = FakeProvider("以下是一些建议，希望这些信息对您有所帮助。")
    run = _orchestrator(tmp_path, provider=provider, enabled=True).reply(
        sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert provider.call_count == 1
    assert run.response.blocked_reason == "style_verification_rejected"
    assert run.trace.verifier_enabled is True
    assert run.trace.verification_action == "rejected"
    assert run.trace.provider_call_count == 1
    assert run.trace.additional_provider_calls == 0
    assert run.trace.correction_attempts == 0
    state = json.loads(
        (tmp_path / "runtime" / "dialogue-usage-2026-07.json").read_text(encoding="utf-8")
    )
    assert state["reservations"] == {}


def test_enabled_verifier_accepts_safe_output_and_does_not_persist_details(tmp_path: Path) -> None:
    provider = FakeProvider("嗨，今天还好吗？")
    orchestrator = _orchestrator(tmp_path, provider=provider, enabled=True)

    run = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert run.response.blocked_reason is None
    assert run.trace.verification_action == "accepted"
    assert run.trace.provider_call_count == 1
    assert set(DialogueSessionState.model_fields) == {
        "last_completed_at",
        "identity_disclosed",
        "previous_soft_lock",
    }
    serialized = run.trace.model_dump_json()
    assert "synthetic current request" not in serialized
    assert "嗨，今天还好吗" not in serialized


def test_baseline_with_enabled_verifier_preserves_baseline_behavior(tmp_path: Path) -> None:
    provider = FakeProvider("以下是一些建议，希望这些信息对您有所帮助。")
    run = _orchestrator(tmp_path, provider=provider, enabled=True).reply(
        sample_request(),
        mode=DialogueMode.BASELINE,
        trusted_user_id=trusted_sample_request_id(),
    )

    assert run.response.blocked_reason is None
    assert run.trace.verification_action == "accepted"
    assert provider.call_count == 1


def test_existing_output_guard_still_blocks_independently_before_verifier(tmp_path: Path) -> None:
    provider = FakeProvider("我代表主人承诺这笔交易。")
    run = _orchestrator(tmp_path, provider=provider, enabled=True).reply(
        sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert provider.call_count == 1
    assert run.response.blocked_reason == "unsafe_output"
    assert run.trace.verification_action == "not_run"
    assert run.trace.provider_call_count == 1


def test_output_guard_internal_failure_is_not_misattributed_to_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_guard(*args: object, **kwargs: object) -> Never:
        raise RuntimeError("synthetic output guard failure")

    monkeypatch.setattr(
        "companion.dialogue.orchestrator.validate_dialogue_output",
        fail_guard,
    )
    run = _orchestrator(
        tmp_path,
        provider=FakeProvider("synthetic safe answer"),
        enabled=True,
    ).reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    assert run.response.blocked_reason == "unsafe_output"
    assert run.trace.verification_action == "not_run"


def test_verifier_exception_rejects_without_leaking_or_retrying(tmp_path: Path) -> None:
    class RaisingVerifier:
        def verify(self, **kwargs: object) -> Never:
            raise RuntimeError(
                f"private verifier sentinel: {kwargs['request_text']} / {kwargs['response_text']}"
            )

    provider = FakeProvider("PRIVATE_RESPONSE_SENTINEL")
    config = synthetic_dialogue_config(tmp_path)
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
        style_verifier=RaisingVerifier(),
    )

    run = orchestrator.reply(sample_request(), trusted_user_id=trusted_sample_request_id())

    serialized = run.model_dump_json()
    assert provider.call_count == 1
    assert run.response.blocked_reason == "style_verification_rejected"
    assert run.trace.verification_action == "rejected"
    assert "synthetic current request" not in serialized
    assert "PRIVATE_RESPONSE_SENTINEL" not in serialized


def test_real_builder_exposes_disabled_by_default_and_explicit_enable_switch(
    tmp_path: Path,
) -> None:
    settings = Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=tmp_path / "private",
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )
    config = synthetic_dialogue_config(tmp_path)
    secrets = DialogueSecrets(api_key=SecretStr("synthetic-api-key"))

    disabled = build_real_orchestrator(
        settings,
        config,
        secrets,
        trusted_user_id="synthetic-user",
    )
    enabled = build_real_orchestrator(
        settings,
        config,
        secrets,
        trusted_user_id="synthetic-user",
        enable_style_verifier=True,
    )

    assert disabled._style_verifier is None
    assert isinstance(enabled._style_verifier, DeterministicStyleVerifier)
