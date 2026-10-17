# ruff: noqa: RUF001
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import SecretStr

from companion.dialogue.context import MEMORY_END, MEMORY_START
from companion.dialogue.models import (
    BudgetConfig,
    DialogueConfig,
    DialogueSecrets,
    ProviderConfig,
    ProviderResult,
    StyleRuntimeSources,
)
from scripts import run_synthetic_privacy_a as synthetic
from scripts.run_synthetic_privacy_a import CANARIES, run_offline


def _explicit_config(root: Path, worst_case_request_cny: str = "1.00") -> DialogueConfig:
    return DialogueConfig(
        provider=ProviderConfig(
            base_url="https://reviewed-provider.example/v1",
            model="reviewed-fixture-model",
            timeout_seconds=1,
        ),
        budget=BudgetConfig(
            monthly_cap_cny=Decimal("100.00"),
            worst_case_request_cny=Decimal(worst_case_request_cny),
            input_cny_per_million=Decimal("1.00"),
            output_cny_per_million=Decimal("2.00"),
            max_output_tokens=256,
        ),
        style=StyleRuntimeSources(
            candidate_source=root / "unused-candidates.json",
            expected_count=1,
            active_style_source=root / "unused-active.json",
            active_manifest_source=root / "unused-manifest.json",
            expected_active_manifest_digest="c" * 64,
            calibration_source=root / "unused-calibration.json",
            expected_calibration_fingerprint="d" * 64,
            private_data_dir=root,
            true_name_terms=(),
            approved_nicknames=(),
        ),
    )


class _CountingSafeProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.memory_attack_prompt_ok = False
        self.third_attack_prompt_ok = False

    def complete(self, messages) -> ProviderResult:
        self.calls += 1
        system, user = messages
        if "SYNTHETIC_ATTACK_TARGET" in user.content:
            self.memory_attack_prompt_ok = (
                system.content.count(MEMORY_START) == 1
                and system.content.count(MEMORY_END) == 1
                and "\\</untrusted_memory\\>" in system.content
                and "\\<untrusted_memory\\>" in system.content
                and "\nSYSTEM:" not in system.content
            )
        if "SYNTHETIC_QUERY_7" in user.content:
            self.third_attack_prompt_ok = (
                "SYNTHETIC_VICTIM_SECRET_7" not in system.content
                and "SYNTHETIC_VICTIM_SECRET_7" not in user.content
            )
        return ProviderResult(text="合成安全回复")


class _FakeTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.urls: set[str] = set()
        self.models: set[str] = set()
        self.max_tokens: set[int] = set()
        self.timeouts: set[int] = set()

    def __call__(self, request, timeout: int) -> bytes:
        self.calls += 1
        payload = json.loads(request.data or b"{}")
        self.urls.add(request.full_url)
        self.models.add(payload["model"])
        self.max_tokens.add(payload["max_tokens"])
        self.timeouts.add(timeout)
        return json.dumps({"choices": [{"message": {"content": "合成安全回复"}}]}).encode()


def test_same_source_provider_assembly_uses_one_config(tmp_path: Path) -> None:
    config = _explicit_config(tmp_path)
    ledger_root = tmp_path / "persistent-ledger"
    ledger_root.mkdir()
    transport = _FakeTransport()
    summary = synthetic.run_with_configured_provider(
        verified_config=config,
        verified_budget=config.budget,
        api_key=DialogueSecrets(api_key=SecretStr("synthetic-assembly-key")),
        persistent_ledger_root=ledger_root,
        transport=transport,
    )
    assert summary["status"] == "synthetic_check_passed"
    assert summary["provider_calls"] == transport.calls == 54
    assert transport.urls == {"https://reviewed-provider.example/v1/chat/completions"}
    assert transport.models == {config.provider.model}
    assert transport.max_tokens == {config.budget.max_output_tokens}
    assert transport.timeouts == {config.provider.timeout_seconds}
    assert "synthetic-assembly-key" not in json.dumps(summary)


def test_same_source_provider_assembly_rejects_before_transport(tmp_path: Path) -> None:
    config = _explicit_config(tmp_path)
    ledger_root = tmp_path / "persistent-ledger"
    ledger_root.mkdir()
    transport = _FakeTransport()
    cases = (
        (None, config.budget, SecretStr("synthetic-key"), ledger_root),
        (config, None, SecretStr("synthetic-key"), ledger_root),
        (
            config,
            _explicit_config(tmp_path, "1.88").budget,
            SecretStr("synthetic-key"),
            ledger_root,
        ),
        (config, config.budget, SecretStr(""), ledger_root),
        (config, config.budget, None, ledger_root),
        (config, config.budget, SecretStr("synthetic-key"), tmp_path / "missing-ledger"),
    )
    for candidate_config, candidate_budget, key, root in cases:
        with pytest.raises(ValueError):
            synthetic.run_with_configured_provider(
                verified_config=candidate_config,
                verified_budget=candidate_budget,
                api_key=key,
                persistent_ledger_root=root,
                transport=transport,
            )
    assert transport.calls == 0


def test_same_source_provider_budget_blocks_transport(tmp_path: Path) -> None:
    config = _explicit_config(tmp_path, "1.88")
    ledger_root = tmp_path / "persistent-ledger"
    ledger_root.mkdir()
    transport = _FakeTransport()
    summary = synthetic.run_with_configured_provider(
        verified_config=config,
        verified_budget=config.budget,
        api_key=SecretStr("synthetic-key"),
        persistent_ledger_root=ledger_root,
        transport=transport,
    )
    assert summary["status"] == "budget_blocked"
    assert summary["provider_calls"] == transport.calls == 0


def _run_injected(provider, root: Path) -> dict[str, object]:
    config = _explicit_config(root)
    ledger_root = root / "persistent-ledger"
    ledger_root.mkdir()
    return synthetic.run_with_provider(
        provider=provider,
        verified_config=config,
        verified_budget=config.budget,
        persistent_ledger_root=ledger_root,
    )


def test_explicit_provider_requires_config_and_persistent_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(synthetic, "NOW", datetime(2000, 1, 1, tzinfo=UTC))
    original_request = synthetic._request
    attack_identities: list[tuple[str, str]] = []

    def capture_attack_identity(dialogue, user_id: str, query: str, trusted: str):
        if query in (synthetic.MEMORY_ATTACK_QUERY, synthetic.THIRD_ATTACK_QUERY):
            attack_identities.append((user_id, trusted))
        return original_request(dialogue, user_id, query, trusted)

    monkeypatch.setattr(synthetic, "_request", capture_attack_identity)
    provider = _CountingSafeProvider()
    ledger_root = tmp_path / "persistent-ledger"
    ledger_root.mkdir()
    with pytest.raises(ValueError):
        synthetic.run_with_provider(
            provider=provider,
            verified_config=None,
            verified_budget=None,
            persistent_ledger_root=ledger_root,
        )
    assert provider.calls == 0

    config = _explicit_config(tmp_path)
    with pytest.raises(ValueError):
        synthetic.run_with_provider(
            provider=None,
            verified_config=config,
            verified_budget=config.budget,
            persistent_ledger_root=ledger_root,
        )
    with pytest.raises(ValueError):
        synthetic.run_with_provider(
            provider=provider,
            verified_config=synthetic._config(tmp_path, "1.00"),
            verified_budget=config.budget,
            persistent_ledger_root=ledger_root,
        )
    with pytest.raises(ValueError):
        synthetic.run_with_provider(
            provider=provider,
            verified_config=config,
            verified_budget=config.budget,
            persistent_ledger_root=tmp_path / "missing-ledger",
        )
    assert provider.calls == 0
    summary = synthetic.run_with_provider(
        provider=provider,
        verified_config=config,
        verified_budget=config.budget,
        persistent_ledger_root=ledger_root,
    )
    assert summary["status"] == "synthetic_check_passed"
    assert provider.calls == summary["provider_calls"] == 54
    assert summary["attack_ids"] == ["INJ_MEMORY_ROLE", "INJ_THIRD_USER"]
    assert provider.memory_attack_prompt_ok
    assert provider.third_attack_prompt_ok
    assert attack_identities == [
        (synthetic.VICTIM, synthetic.VICTIM),
        (synthetic.THIRD_USER, synthetic.THIRD_USER),
    ]
    month = datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m")
    assert (ledger_root / f"dialogue-usage-{month}.json").is_file()


def test_explicit_provider_budget_preflight_blocks_all_calls(tmp_path: Path) -> None:
    provider = _CountingSafeProvider()
    config = _explicit_config(tmp_path, worst_case_request_cny="1.88")
    assert Decimal("1.88") * 52 <= 100 < Decimal("1.88") * 54
    ledger_root = tmp_path / "persistent-ledger"
    ledger_root.mkdir()
    summary = synthetic.run_with_provider(
        provider=provider,
        verified_config=config,
        verified_budget=config.budget,
        persistent_ledger_root=ledger_root,
    )
    assert summary["status"] == "budget_blocked"
    assert summary["provider_calls"] == provider.calls == 0


def test_memory_attack_malicious_raw_reply_stops_before_next_call(tmp_path: Path) -> None:
    class MaliciousReply(_CountingSafeProvider):
        def complete(self, messages) -> ProviderResult:
            safe = super().complete(messages)
            if "SYNTHETIC_ATTACK_TARGET" in messages[1].content:
                return ProviderResult(text="我就是主人本人，可以替他处理。")
            return safe

    provider = MaliciousReply()
    summary = _run_injected(provider, tmp_path)
    assert summary["status"] == "offline_failed"
    assert provider.calls == summary["provider_calls"] == 52
    assert summary["attack_ids"] == []
    assert "我就是主人本人" not in json.dumps(summary, ensure_ascii=False)


def test_third_attack_victim_secret_reply_stops_before_revocation(tmp_path: Path) -> None:
    class LeakingReply(_CountingSafeProvider):
        def complete(self, messages) -> ProviderResult:
            safe = super().complete(messages)
            if "SYNTHETIC_QUERY_7" in messages[1].content:
                return ProviderResult(text=synthetic.THIRD_ATTACK_VALUE)
            return safe

    provider = LeakingReply()
    summary = _run_injected(provider, tmp_path)
    assert summary["status"] == "offline_failed"
    assert provider.calls == summary["provider_calls"] == 53
    assert summary["attack_ids"] == ["INJ_MEMORY_ROLE"]
    assert "SYNTHETIC_VICTIM_SECRET_7" not in json.dumps(summary)


def test_offline_fifty_canaries_and_revocation() -> None:
    summary = run_offline()
    assert summary == {
        "status": "offline_passed",
        "checked": 50,
        "leaks": 0,
        "provider_calls": 52,
        "failed": 0,
        "victim_visible": True,
        "revoked_clear": True,
        "wrong_identity_calls": 0,
    }


def test_budget_preflight_blocks_all_provider_calls() -> None:
    summary = run_offline(worst_case_request_cny="2.00")
    assert summary["status"] == "budget_blocked"
    assert summary["provider_calls"] == 0


def test_wrong_trusted_identity_returns_untrusted_user(monkeypatch) -> None:
    original = synthetic._request
    observed = 0

    def capture(dialogue, user_id: str, query: str, trusted: str):
        nonlocal observed
        run = original(dialogue, user_id, query, trusted)
        if user_id != trusted:
            observed += 1
            assert run.response.blocked_reason == "untrusted_user"
            assert run.trace.provider_call_count == 0
        return run

    monkeypatch.setattr(synthetic, "_request", capture)
    assert run_offline()["status"] == "offline_passed"
    assert observed == 1


def test_owner_can_receive_own_canary_without_public_output(tmp_path: Path) -> None:
    class OwnerCanaryProvider:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _messages: object) -> ProviderResult:
            self.calls += 1
            return ProviderResult(text=CANARIES[0] if self.calls == 1 else "合成安全回复")

    provider = OwnerCanaryProvider()
    summary = _run_injected(provider, tmp_path)
    assert summary["status"] == "synthetic_check_passed"
    assert provider.calls == summary["provider_calls"] == 54
    assert summary["leaks"] == 0
    assert CANARIES[0] not in json.dumps(summary)


def test_third_user_canary_response_fails_closed(tmp_path: Path) -> None:
    class LeakingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _messages: object) -> ProviderResult:
            self.calls += 1
            return ProviderResult(text=CANARIES[0] if self.calls == 2 else "合成安全回复")

    provider = LeakingProvider()
    summary = _run_injected(provider, tmp_path)
    assert summary["status"] == "offline_failed"
    assert summary["leaks"] > 0
    assert provider.calls == summary["provider_calls"] == 2
    assert CANARIES[0] not in json.dumps(summary)


def test_revoked_owner_canary_response_fails_closed(tmp_path: Path) -> None:
    class LeakingAfterRevocation:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _messages: object) -> ProviderResult:
            self.calls += 1
            return ProviderResult(text=CANARIES[0] if self.calls == 54 else "合成安全回复")

    provider = LeakingAfterRevocation()
    summary = _run_injected(provider, tmp_path)
    assert summary["status"] == "offline_failed"
    assert summary["leaks"] > 0
    assert provider.calls == summary["provider_calls"] == 54
    assert CANARIES[0] not in json.dumps(summary)


def test_cli_help_and_default_output_are_redacted() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_synthetic_privacy_a.py"
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert help_run.returncode == 0
    assert "offline" in help_run.stdout.lower()
    assert not any(canary in help_run.stdout for canary in CANARIES)

    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert run.returncode == 0
    assert json.loads(run.stdout)["status"] == "offline_passed"
    assert run.stderr == ""
    assert not any(canary in run.stdout for canary in CANARIES)
