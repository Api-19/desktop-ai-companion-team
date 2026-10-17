from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from dialogue_support import FakeProvider, make_orchestrator

from companion.dialogue import (
    ABSuiteSummary,
    BudgetExceeded,
    DialogueMode,
    SmokeSummary,
    UsageLedgerError,
    UsageReviewRequired,
    run_baseline_smoke,
)
from companion.dialogue.ab import DIALOGUE_EVALUATION_USER_ID


def load_script() -> ModuleType:
    script_path = Path("scripts/run_dialogue_ab.py").resolve()
    spec = importlib.util.spec_from_file_location("run_dialogue_ab", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_smoke_uses_one_fixed_baseline_request(tmp_path: Path) -> None:
    private_answer = "synthetic private smoke answer"
    orchestrator = make_orchestrator(
        tmp_path,
        provider=FakeProvider(private_answer),
        trusted_user_id=DIALOGUE_EVALUATION_USER_ID,
    )
    results_dir = tmp_path / "test-results"

    summary = run_baseline_smoke(orchestrator, results_dir)

    assert summary == SmokeSummary(
        status="complete",
        succeeded=True,
        request_attempt_count=1,
    )
    evidence_files = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and private_answer in path.read_text(encoding="utf-8")
    ]
    assert len(evidence_files) == 1
    assert evidence_files[0].is_relative_to(results_dir)
    assert evidence_files[0].name == "answer.json"


@pytest.mark.parametrize(
    "untrusted_root",
    (
        Path.cwd(),
        Path("arbitrary-results"),
        Path("test-results") / ".." / "arbitrary-alias",
    ),
)
def test_baseline_smoke_rejects_unbound_results_roots_before_cloud_call_or_write(
    tmp_path: Path,
    untrusted_root: Path,
) -> None:
    provider = FakeProvider("synthetic private smoke answer")
    orchestrator = make_orchestrator(tmp_path, provider=provider)
    supplied_root = untrusted_root if untrusted_root.is_absolute() else tmp_path / untrusted_root

    with pytest.raises(ValueError, match=r"^invalid dialogue results directory$"):
        run_baseline_smoke(orchestrator, supplied_root)

    assert provider.call_count == 0
    assert not list(supplied_root.resolve().glob("dialogue-smoke-*"))


def test_script_checks_thirteen_requests_before_any_cloud_call(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = load_script()
    events: list[object] = []
    builder_kwargs: dict[str, object] = {}
    settings = SimpleNamespace(test_results_dir=tmp_path / "test-results")

    class FakeOrchestrator:
        def ensure_paid_capacity(self, count: int) -> None:
            events.append(("capacity", count))

    orchestrator = FakeOrchestrator()
    monkeypatch.setattr(module, "load_settings", lambda root: settings)
    monkeypatch.setattr(module, "load_dialogue_config", lambda settings: (object(), object()))
    monkeypatch.setattr(
        module,
        "build_real_orchestrator",
        lambda settings, config, secrets, **kwargs: builder_kwargs.update(kwargs) or orchestrator,
    )
    monkeypatch.setattr(
        module,
        "run_baseline_smoke",
        lambda actual, results_dir: (
            events.append(("smoke", DialogueMode.BASELINE, results_dir))
            or SmokeSummary(status="complete", succeeded=True, request_attempt_count=1)
        ),
    )
    monkeypatch.setattr(module, "load_ab_scenarios", lambda path: ())
    monkeypatch.setattr(
        module,
        "run_ab_suite",
        lambda actual, scenarios, results_dir: (
            events.append(("suite", len(scenarios)))
            or ABSuiteSummary(
                run_id="synthetic-run",
                scenario_count=0,
                request_attempt_count=0,
                succeeded_count=0,
                failed_count=0,
                blocked_count=0,
                paired_count=0,
            )
        ),
    )

    exit_code = module.main()

    assert exit_code == 0
    assert builder_kwargs == {"trusted_user_id": DIALOGUE_EVALUATION_USER_ID}
    assert events == [
        ("capacity", 13),
        ("smoke", DialogueMode.BASELINE, settings.test_results_dir),
        ("suite", 0),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "synthetic-run",
        "scenario_count": 0,
        "request_attempt_count": 0,
        "succeeded_count": 0,
        "failed_count": 0,
        "blocked_count": 0,
        "paired_count": 0,
    }


def test_framework20_preflight_binds_owner_and_never_calls_provider(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    module = load_script()
    owner = "synthetic-framework-owner"
    events: list[object] = []
    builder_kwargs: dict[str, object] = {}
    settings = SimpleNamespace(test_results_dir=tmp_path / "results")
    config = SimpleNamespace(style_framework_owner_id=owner)
    scenarios = tuple(object() for _ in range(20))

    class FakeOrchestrator:
        def ensure_paid_capacity(self, count: int) -> None:
            events.append(("capacity", count))

        def preflight_framework20(self, actual, *, trusted_user_id, results_dir) -> None:
            events.append(("preflight", len(actual), trusted_user_id, results_dir))

    monkeypatch.setattr(module, "load_settings", lambda root: settings)
    monkeypatch.setattr(module, "load_dialogue_config", lambda actual: (config, object()))
    monkeypatch.setattr(
        module,
        "build_real_orchestrator",
        lambda *args, **kwargs: builder_kwargs.update(kwargs) or FakeOrchestrator(),
    )
    monkeypatch.setattr(module, "load_framework20_scenarios", lambda path: scenarios)
    monkeypatch.setattr(
        module, "run_baseline_smoke", lambda *args, **kwargs: pytest.fail("provider smoke ran")
    )
    monkeypatch.setattr(
        module, "run_ab_suite", lambda *args, **kwargs: pytest.fail("provider suite ran")
    )

    assert (
        module.main("framework20", enable_production_personalization=True, preflight_only=True) == 0
    )
    assert builder_kwargs == {
        "trusted_user_id": owner,
        "enable_production_personalization": True,
    }
    assert events == [
        ("capacity", 41),
        ("preflight", 20, owner, settings.test_results_dir),
    ]
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "status": "preflight_ready",
        "scenario_count": 20,
        "provider_attempt_count": 0,
    }
    assert owner not in output
    assert list(settings.test_results_dir.glob("**/*")) == []


@pytest.mark.parametrize(
    ("suite", "enabled", "preflight"),
    (("framework20", False, False), ("legacy", True, False), ("style_refresh", False, True)),
)
def test_framework20_flags_fail_before_loading_configuration(
    monkeypatch, capsys, suite: str, enabled: bool, preflight: bool
) -> None:
    module = load_script()
    monkeypatch.setattr(module, "load_settings", lambda root: pytest.fail("configuration loaded"))
    assert (
        module.main(suite, enable_production_personalization=enabled, preflight_only=preflight) == 2
    )
    assert json.loads(capsys.readouterr().out) == {"status": "configuration_unavailable"}


def test_framework20_requires_explicit_owner_before_build(monkeypatch, capsys) -> None:
    module = load_script()
    monkeypatch.setattr(module, "load_settings", lambda root: object())
    monkeypatch.setattr(
        module,
        "load_dialogue_config",
        lambda settings: (SimpleNamespace(style_framework_owner_id=None), object()),
    )
    monkeypatch.setattr(
        module, "build_real_orchestrator", lambda *args, **kwargs: pytest.fail("built")
    )
    assert module.main("framework20", enable_production_personalization=True) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "configuration_unavailable"}


@pytest.mark.parametrize("smoke_succeeded", (False, True))
def test_framework20_provider_run_uses_bound_owner_after_preflight(
    monkeypatch, capsys, tmp_path: Path, smoke_succeeded: bool
) -> None:
    module = load_script()
    owner = "synthetic-framework-owner"
    settings = SimpleNamespace(test_results_dir=tmp_path / "results")
    events: list[object] = []
    scenarios = tuple(object() for _ in range(20))

    class FakeOrchestrator:
        def ensure_paid_capacity(self, count: int) -> None:
            events.append(("capacity", count))

        def preflight_framework20(self, actual, *, trusted_user_id, results_dir) -> None:
            events.append(("preflight", trusted_user_id, results_dir))

    def smoke(actual, results_dir, *, trusted_user_id):
        events.append(("smoke", trusted_user_id, results_dir))
        return SmokeSummary(status="complete", succeeded=smoke_succeeded, request_attempt_count=1)

    def suite(actual, items, results_dir, *, trusted_user_id):
        events.append(("suite", trusted_user_id, len(items), results_dir))
        return ABSuiteSummary(
            run_id="synthetic-framework-run",
            scenario_count=20,
            request_attempt_count=40,
            succeeded_count=20,
            failed_count=0,
            blocked_count=0,
            paired_count=20,
        )

    monkeypatch.setattr(module, "load_settings", lambda root: settings)
    monkeypatch.setattr(
        module,
        "load_dialogue_config",
        lambda actual: (SimpleNamespace(style_framework_owner_id=owner), object()),
    )
    monkeypatch.setattr(
        module, "build_real_orchestrator", lambda *args, **kwargs: FakeOrchestrator()
    )
    monkeypatch.setattr(module, "load_framework20_scenarios", lambda path: scenarios)
    monkeypatch.setattr(module, "run_baseline_smoke", smoke)
    monkeypatch.setattr(module, "run_ab_suite", suite)

    assert module.main("framework20", enable_production_personalization=True) == (
        0 if smoke_succeeded else 1
    )
    assert events == [
        ("capacity", 41),
        ("preflight", owner, settings.test_results_dir),
        ("smoke", owner, settings.test_results_dir),
        *([("suite", owner, 20, settings.test_results_dir)] if smoke_succeeded else []),
    ]
    assert owner not in capsys.readouterr().out


def test_script_style_refresh_mode_checks_fifteen_and_runs_exact_seven(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = load_script()
    events: list[object] = []
    settings = SimpleNamespace(test_results_dir=tmp_path / "test-results")
    scenarios = tuple(object() for _ in range(7))

    class FakeOrchestrator:
        def ensure_paid_capacity(self, count: int) -> None:
            events.append(("capacity", count))

    monkeypatch.setattr(module, "load_settings", lambda root: settings)
    monkeypatch.setattr(module, "load_dialogue_config", lambda settings: (object(), object()))
    monkeypatch.setattr(
        module,
        "build_real_orchestrator",
        lambda settings, config, secrets, **kwargs: FakeOrchestrator(),
    )
    monkeypatch.setattr(
        module,
        "run_baseline_smoke",
        lambda actual, results_dir: SmokeSummary(
            status="complete",
            succeeded=True,
            request_attempt_count=1,
        ),
    )
    monkeypatch.setattr(
        module,
        "load_style_refresh_scenarios",
        lambda full, refresh: scenarios,
    )
    monkeypatch.setattr(
        module,
        "run_ab_suite",
        lambda actual, actual_scenarios, results_dir: (
            events.append(("suite", len(actual_scenarios)))
            or ABSuiteSummary(
                run_id="synthetic-refresh-run",
                scenario_count=7,
                request_attempt_count=14,
                succeeded_count=7,
                failed_count=0,
                blocked_count=0,
                paired_count=7,
            )
        ),
    )

    assert module.main("style_refresh") == 0
    assert events == [("capacity", 15), ("suite", 7)]
    assert json.loads(capsys.readouterr().out)["scenario_count"] == 7


def test_script_stops_before_ab_when_smoke_fails(monkeypatch, capsys, tmp_path: Path) -> None:
    module = load_script()
    suite_calls = 0
    settings = SimpleNamespace(test_results_dir=tmp_path / "test-results")

    class FakeOrchestrator:
        def ensure_paid_capacity(self, count: int) -> None:
            assert count == 13

    def fail_if_called(*args: object, **kwargs: object) -> None:
        nonlocal suite_calls
        suite_calls += 1

    monkeypatch.setattr(module, "load_settings", lambda root: settings)
    monkeypatch.setattr(module, "load_dialogue_config", lambda settings: (object(), object()))
    monkeypatch.setattr(
        module,
        "build_real_orchestrator",
        lambda settings, config, secrets, **kwargs: FakeOrchestrator(),
    )
    monkeypatch.setattr(
        module,
        "run_baseline_smoke",
        lambda orchestrator, results_dir: SmokeSummary(
            status="failed",
            succeeded=False,
            request_attempt_count=1,
        ),
    )
    monkeypatch.setattr(module, "run_ab_suite", fail_if_called)

    exit_code = module.main()

    assert exit_code == 1
    assert suite_calls == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "succeeded": False,
        "request_attempt_count": 1,
    }


def test_script_configuration_failure_is_fixed_and_content_free(monkeypatch, capsys) -> None:
    module = load_script()

    def unavailable(root: Path) -> None:
        raise RuntimeError("synthetic private path and secret detail")

    monkeypatch.setattr(module, "load_settings", unavailable)

    exit_code = module.main()

    assert exit_code == 2
    assert capsys.readouterr().out == '{"status":"configuration_unavailable"}\n'


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_exit_code"),
    (
        (BudgetExceeded("private budget detail"), "budget_exceeded", 1),
        (UsageReviewRequired("private review detail"), "usage_review_required", 2),
        (UsageLedgerError("private ledger path"), "usage_unavailable", 2),
    ),
)
def test_script_preflight_usage_failures_are_distinct_content_free_and_zero_call(
    monkeypatch,
    capsys,
    tmp_path: Path,
    error: Exception,
    expected_status: str,
    expected_exit_code: int,
) -> None:
    module = load_script()
    smoke_calls = 0
    settings = SimpleNamespace(test_results_dir=tmp_path / "test-results")

    class FakeOrchestrator:
        def ensure_paid_capacity(self, count: int) -> None:
            assert count == 13
            raise error

    def smoke_if_called(*args: object, **kwargs: object) -> None:
        nonlocal smoke_calls
        smoke_calls += 1

    monkeypatch.setattr(module, "load_settings", lambda root: settings)
    monkeypatch.setattr(module, "load_dialogue_config", lambda settings: (object(), object()))
    monkeypatch.setattr(
        module,
        "build_real_orchestrator",
        lambda settings, config, secrets, **kwargs: FakeOrchestrator(),
    )
    monkeypatch.setattr(module, "run_baseline_smoke", smoke_if_called)

    exit_code = module.main()

    assert exit_code == expected_exit_code
    assert smoke_calls == 0
    captured = capsys.readouterr()
    assert captured.out == f'{{"status":"{expected_status}"}}\n'
    assert captured.err == ""
    assert "private" not in captured.out


@pytest.mark.parametrize("failure_stage", ("scenario_loading", "suite_write"))
def test_script_runtime_failure_is_fixed_and_content_free(
    monkeypatch,
    capsys,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    module = load_script()
    private_detail = "synthetic private response C:/private/scenarios.json"
    settings = SimpleNamespace(test_results_dir=tmp_path / "test-results")

    class FakeOrchestrator:
        def ensure_paid_capacity(self, count: int) -> None:
            assert count == 13

    def raise_private_error(*args: object, **kwargs: object) -> None:
        raise RuntimeError(private_detail)

    monkeypatch.setattr(module, "load_settings", lambda root: settings)
    monkeypatch.setattr(module, "load_dialogue_config", lambda settings: (object(), object()))
    monkeypatch.setattr(
        module,
        "build_real_orchestrator",
        lambda settings, config, secrets, **kwargs: FakeOrchestrator(),
    )
    monkeypatch.setattr(
        module,
        "run_baseline_smoke",
        lambda orchestrator, results_dir: SmokeSummary(
            status="complete",
            succeeded=True,
            request_attempt_count=1,
        ),
    )
    monkeypatch.setattr(
        module,
        "load_ab_scenarios",
        raise_private_error if failure_stage == "scenario_loading" else lambda path: (),
    )
    monkeypatch.setattr(
        module,
        "run_ab_suite",
        raise_private_error if failure_stage == "suite_write" else lambda *args: None,
    )

    exit_code = module.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == '{"status":"run_failed"}\n'
    assert captured.err == ""
    assert private_detail not in captured.out
