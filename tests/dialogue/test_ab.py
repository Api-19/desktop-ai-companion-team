from __future__ import annotations

import json
import traceback
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

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

from companion.contracts import DialogueRequest, StyleReplyGuidance
from companion.dialogue import (
    STYLE_REFRESH_SCENARIO_IDS,
    ABScenario,
    ABStatus,
    DialogueMode,
    DialogueOrchestrator,
    MonthlyUsageLedger,
    load_ab_scenarios,
    load_style_refresh_scenarios,
    run_ab_suite,
    run_baseline_smoke,
)
from companion.dialogue.ab import FRAMEWORK20_SCENARIO_IDS, ABPair, load_framework20_scenarios
from companion.dialogue.context import STYLE_END, STYLE_START, PromptMessage
from companion.dialogue.models import DialogueConfig, ProviderResult


def make_ab_orchestrator(
    root: Path,
    *,
    guidance: StyleReplyGuidance | None = None,
    worst_case: str = "1.00",
) -> tuple[DialogueOrchestrator, FakeStyleProvider, FakeProvider]:
    style = FakeStyleProvider(guidance)
    provider = FakeProvider()
    config = synthetic_dialogue_config(root)
    config = config.model_copy(
        update={
            "budget": config.budget.model_copy(
                update={"worst_case_request_cny": Decimal(worst_case)}
            )
        }
    )
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(root / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
        test_results_dir=root / "test-results",
    )
    return orchestrator, style, provider


class _ZeroCallMemoryGateway:
    def __init__(self) -> None:
        self.call_count = 0

    def create_memory(self, request: object) -> object:
        self.call_count += 1
        raise AssertionError("A/B must not create memory")

    def retrieve_memories(self, user_id: str, query: str, limit: int = 5) -> object:
        self.call_count += 1
        raise AssertionError("A/B must not retrieve memory")


def without_style_block(
    messages: tuple[PromptMessage, ...],
) -> tuple[PromptMessage, ...]:
    system = messages[0].content
    if STYLE_START in system:
        prefix, rest = system.split(STYLE_START, 1)
        _, suffix = rest.split(STYLE_END, 1)
        system = prefix + suffix
    return (messages[0].model_copy(update={"content": system}), *messages[1:])


def test_load_ab_scenarios_accepts_the_six_public_synthetic_cases() -> None:
    scenarios = load_ab_scenarios(Path("config/dialogue-ab-scenarios.json"))

    assert [(item.scenario_id, item.text, item.idle_seconds) for item in scenarios] == [
        ("greeting-2", "晚上好", 21600),
        ("courtesy-1", "我帮你弄好了", 0),
        ("courtesy-2", "别打断我", 0),
        ("weather-1", "天气转冷", 0),
        ("work_study-1", "作业写完了", 0),
        ("meals-2", "我还没吃饭", 0),
    ]
    assert all(item.recent_context == () for item in scenarios)


def test_full_dialogue_scenarios_match_the_style_runtime_fixture_exactly() -> None:
    dialogue = load_ab_scenarios(Path("config/dialogue-ab-full-scenarios.json"))
    style_payload = json.loads(
        Path("fixtures/synthetic/style_runtime_guidance_scenarios.json").read_text(encoding="utf-8")
    )
    dialogue_records = [
        {
            "scenario_id": item.scenario_id,
            "text": item.text,
            "recent_context": list(item.recent_context),
            "idle_seconds": item.idle_seconds,
        }
        for item in dialogue
    ]
    style_records = [
        {
            "scenario_id": item["scenario_id"],
            "text": item["text"],
            "recent_context": item["recent_context"],
            "idle_seconds": item["idle_seconds"],
        }
        for item in style_payload
    ]

    assert len(dialogue_records) == len(style_records) == 20
    assert len({item["scenario_id"] for item in dialogue_records}) == 20
    assert len({item["scenario_id"] for item in style_records}) == 20
    assert dialogue_records == style_records


def test_framework20_uses_exact_public_scenario_order() -> None:
    scenarios = load_framework20_scenarios(Path("config/dialogue-ab-full-scenarios.json"))
    assert tuple(item.scenario_id for item in scenarios) == FRAMEWORK20_SCENARIO_IDS
    assert len(scenarios) == 20


@pytest.mark.parametrize("change", ("missing", "duplicate", "reordered"))
def test_framework20_rejects_changed_scenario_ids(tmp_path: Path, change: str) -> None:
    records = json.loads(Path("config/dialogue-ab-full-scenarios.json").read_text(encoding="utf-8"))
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1]["scenario_id"] = records[0]["scenario_id"]
    else:
        records[0], records[1] = records[1], records[0]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match=r"^invalid framework20 scenarios$"):
        load_framework20_scenarios(path)


def test_style_refresh_scenarios_are_exact_authoritative_subset() -> None:
    full = load_ab_scenarios(Path("config/dialogue-ab-full-scenarios.json"))
    selected = load_style_refresh_scenarios(
        Path("config/dialogue-ab-full-scenarios.json"),
        Path("config/dialogue-ab-style-refresh-scenarios.json"),
    )

    assert tuple(item.scenario_id for item in selected) == STYLE_REFRESH_SCENARIO_IDS
    assert len(selected) == len({item.scenario_id for item in selected}) == 7
    assert selected == tuple(
        item for item in full if item.scenario_id in STYLE_REFRESH_SCENARIO_IDS
    )


@pytest.mark.parametrize("changed_field", ("text", "idle_seconds"))
def test_style_refresh_scenarios_reject_authoritative_record_drift(
    tmp_path: Path,
    changed_field: str,
) -> None:
    refresh = json.loads(
        Path("config/dialogue-ab-style-refresh-scenarios.json").read_text(encoding="utf-8")
    )
    if changed_field == "text":
        refresh[0]["text"] = "synthetic changed text"
    else:
        refresh[0]["idle_seconds"] += 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(refresh), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^invalid style refresh scenarios$"):
        load_style_refresh_scenarios(
            Path("config/dialogue-ab-full-scenarios.json"),
            changed,
        )


def test_load_ab_scenarios_rejects_unknown_fields(tmp_path: Path) -> None:
    source = tmp_path / "scenarios.json"
    source.write_text(
        json.dumps(
            [
                {
                    "scenario_id": "synthetic-case",
                    "text": "synthetic input",
                    "recent_context": [],
                    "idle_seconds": 0,
                    "unexpected": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid A/B scenarios"):
        load_ab_scenarios(source)


def test_reply_ab_builds_guidance_once_and_changes_only_style_block(tmp_path: Path) -> None:
    orchestrator, style, provider = make_ab_orchestrator(tmp_path)
    pair = orchestrator.reply_ab(
        "weather-1",
        DialogueRequest(
            user_id="synthetic-ab-user",
            text="天气转冷",
            recent_context=[],
        ),
        evaluation_idle_seconds=0,
        trusted_user_id="synthetic-ab-user",
    )

    assert style.call_count == 1
    assert provider.call_count == 2
    assert without_style_block(provider.requests[0]) == without_style_block(provider.requests[1])
    assert {answer.disclosed_ai_identity for answer in pair.answers.values()} == {True}
    assert pair.status is ABStatus.COMPLETE


def test_owner_bound_smoke_and_suite_use_same_trusted_identity(tmp_path: Path) -> None:
    orchestrator, style, provider = make_ab_orchestrator(tmp_path)
    owner = "synthetic-framework-owner"
    scenarios = load_framework20_scenarios(Path("config/dialogue-ab-full-scenarios.json"))

    smoke = run_baseline_smoke(orchestrator, tmp_path / "test-results", trusted_user_id=owner)
    summary = run_ab_suite(
        orchestrator, scenarios[:1], tmp_path / "test-results", trusted_user_id=owner
    )

    assert smoke.succeeded
    assert summary.paired_count == 1
    assert provider.call_count == 3
    assert [call[0][0] for call in style.calls] == [owner, owner]


def test_memory_enabled_reply_ab_rejects_missing_trusted_user_before_side_effects(
    tmp_path: Path,
) -> None:
    config = synthetic_dialogue_config(tmp_path)
    style = FakeStyleProvider()
    provider = FakeProvider()
    gateway = _ZeroCallMemoryGateway()
    runtime = tmp_path / "runtime"
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(runtime, config.budget),
        clock=lambda: FIXED_NOW,
        memory_gateway=gateway,  # type: ignore[arg-type]
    )

    pair = orchestrator.reply_ab("memory-guard", sample_request())

    assert pair.status is ABStatus.FAILED
    assert style.call_count == 0
    assert gateway.call_count == 0
    assert provider.call_count == 0
    assert not runtime.exists()


def test_memory_enabled_reply_ab_accepts_matching_builtin_trusted_user(
    tmp_path: Path,
) -> None:
    config = synthetic_dialogue_config(tmp_path)
    style = FakeStyleProvider()
    provider = FakeProvider()
    gateway = _ZeroCallMemoryGateway()
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=style,
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
        memory_gateway=gateway,  # type: ignore[arg-type]
    )
    request = sample_request()

    pair = orchestrator.reply_ab(
        "memory-guard",
        request,
        trusted_user_id=request.user_id,
    )

    assert pair.status is ABStatus.COMPLETE
    assert style.call_count == 1
    assert gateway.call_count == 0
    assert provider.call_count == 2


def test_reply_ab_skips_fake_pair_for_no_match(tmp_path: Path) -> None:
    orchestrator, _, provider = make_ab_orchestrator(
        tmp_path,
        guidance=StyleReplyGuidance(status="no_match", style_version="synthetic-v1"),
    )

    pair = orchestrator.reply_ab(
        "no_match_case", sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert pair.status is ABStatus.NOT_MATCHED
    assert pair.answers == {}
    assert pair.answer_key == {}
    assert provider.call_count == 0


def test_reply_ab_reserves_both_costs_before_first_call(tmp_path: Path) -> None:
    orchestrator, _, provider = make_ab_orchestrator(tmp_path, worst_case="50.01")

    pair = orchestrator.reply_ab(
        "budget_case", sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert pair.status is ABStatus.BUDGET_EXCEEDED
    assert provider.call_count == 0


@pytest.mark.parametrize(
    ("scenario_id", "expected"),
    (
        ("parity-even", (DialogueMode.BASELINE, DialogueMode.STYLED)),
        ("parity-odd", (DialogueMode.STYLED, DialogueMode.BASELINE)),
    ),
)
def test_reply_ab_uses_deterministic_blind_labels(
    tmp_path: Path,
    scenario_id: str,
    expected: tuple[DialogueMode, DialogueMode],
) -> None:
    orchestrator, _, _ = make_ab_orchestrator(tmp_path)

    pair = orchestrator.reply_ab(
        scenario_id, sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert pair.answer_key == {"A": expected[0], "B": expected[1]}
    assert set(pair.answers) == {"A", "B"}


def test_reply_ab_does_not_mutate_normal_session_state(tmp_path: Path) -> None:
    orchestrator, _, _ = make_ab_orchestrator(tmp_path)

    orchestrator.reply_ab(
        "snapshot-case",
        sample_request(),
        evaluation_idle_seconds=17,
        trusted_user_id=trusted_sample_request_id(),
    )

    assert orchestrator._sessions == {}


def test_reply_ab_keeps_one_failed_side_absent_and_does_not_retry(tmp_path: Path) -> None:
    class FailOnceProvider(FakeProvider):
        def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
            if self.call_count == 0:
                self.call_count += 1
                self.requests.append(tuple(messages))
                raise RuntimeError("synthetic one-side failure")
            return super().complete(messages)

    config = synthetic_dialogue_config(tmp_path)
    provider = FailOnceProvider()
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
    )

    pair = orchestrator.reply_ab(
        "partial-case", sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert pair.status is ABStatus.PARTIAL
    assert provider.call_count == 2
    assert len(pair.answers) == 1
    assert set(pair.answer_key) == {"A", "B"}


def test_run_ab_suite_separates_answers_and_keys_under_results_root(tmp_path: Path) -> None:
    orchestrator, _, _ = make_ab_orchestrator(tmp_path)
    results_root = tmp_path / "test-results"
    scenarios = (
        ABScenario(
            scenario_id="synthetic-case",
            text="synthetic input",
            recent_context=(),
            idle_seconds=0,
        ),
    )

    summary = run_ab_suite(orchestrator, scenarios, results_root)

    run_dir = results_root / summary.run_id
    answers = json.loads((run_dir / "answers.json").read_text(encoding="utf-8"))
    keys = json.loads((run_dir / "answer-key.json").read_text(encoding="utf-8"))
    assert summary.scenario_count == 1
    assert summary.paired_count == 1
    assert set(answers[0]) == {"scenario_id", "status", "answers"}
    assert set(keys[0]) == {"scenario_id", "answer_key"}
    assert set(answers[0]["answers"]) == {"A", "B"}
    assert set(keys[0]["answer_key"]) == {"A", "B"}


@pytest.mark.parametrize(
    "untrusted_root",
    (
        Path.cwd(),
        Path("arbitrary-results"),
        Path("test-results") / ".." / "arbitrary-alias",
    ),
)
def test_run_ab_suite_rejects_unbound_results_roots_before_calls_or_writes(
    tmp_path: Path,
    untrusted_root: Path,
) -> None:
    orchestrator, _, provider = make_ab_orchestrator(tmp_path)
    supplied_root = untrusted_root if untrusted_root.is_absolute() else tmp_path / untrusted_root
    scenarios = (
        ABScenario(
            scenario_id="synthetic-case",
            text="synthetic input",
            recent_context=(),
            idle_seconds=0,
        ),
    )

    with pytest.raises(ValueError, match=r"^invalid dialogue results directory$"):
        run_ab_suite(orchestrator, scenarios, supplied_root)

    assert provider.call_count == 0
    assert not list(supplied_root.resolve().glob("dialogue-ab-*"))


class _SensitiveZeroCallProvider(FakeProvider):
    def __init__(self, config: DialogueConfig) -> None:
        super().__init__()
        self.config = config
        self.secret = SecretStr("synthetic-evidence-root-secret")


@pytest.mark.parametrize("operation", ("smoke", "suite"))
def test_rejected_results_root_traceback_retains_no_dialogue_boundary_material(
    tmp_path: Path,
    operation: str,
) -> None:
    config = synthetic_dialogue_config(tmp_path)
    provider = _SensitiveZeroCallProvider(config)
    orchestrator = DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "runtime", config.budget),
        clock=lambda: FIXED_NOW,
        test_results_dir=tmp_path / "trusted-results",
    )
    untrusted_root = tmp_path / "synthetic-rejected-results-root"
    scenario = ABScenario(
        scenario_id="synthetic-evidence-case",
        text="synthetic-evidence-scenario-sentinel",
        recent_context=(),
        idle_seconds=0,
    )

    with pytest.raises(
        ValueError,
        match=r"^invalid dialogue results directory$",
    ) as caught:
        if operation == "smoke":
            run_baseline_smoke(orchestrator, untrusted_root)
        else:
            run_ab_suite(orchestrator, (scenario,), untrusted_root)

    assert provider.call_count == 0
    assert not untrusted_root.exists()
    _assert_sanitized_results_root_traceback(caught.value)


def _assert_sanitized_results_root_traceback(error: Exception) -> None:
    forbidden_types = (
        ABScenario,
        DialogueConfig,
        DialogueOrchestrator,
        Path,
        SecretStr,
        _SensitiveZeroCallProvider,
    )
    forbidden_values = (
        "synthetic-evidence-root-secret",
        "synthetic-rejected-results-root",
        "synthetic-evidence-scenario-sentinel",
    )
    assert error.__cause__ is None
    assert error.__context__ is None
    for frame, _ in traceback.walk_tb(error.__traceback__):
        filename = frame.f_code.co_filename.replace("/", "\\")
        if "\\src\\companion\\dialogue\\" not in filename:
            continue
        assert not any(isinstance(value, forbidden_types) for value in frame.f_locals.values())
        assert all(
            forbidden not in repr(value)
            for value in frame.f_locals.values()
            for forbidden in forbidden_values
        )


def test_reply_ab_reports_actual_attempt_count(tmp_path: Path) -> None:
    orchestrator, _, provider = make_ab_orchestrator(tmp_path)

    pair = orchestrator.reply_ab(
        "synthetic-case", sample_request(), trusted_user_id=trusted_sample_request_id()
    )
    no_match, _, _ = make_ab_orchestrator(
        tmp_path / "no-match",
        guidance=StyleReplyGuidance(status="no_match", style_version="synthetic-v1"),
    )
    skipped = no_match.reply_ab(
        "synthetic-case", sample_request(), trusted_user_id=trusted_sample_request_id()
    )

    assert pair.request_attempt_count == provider.call_count == 2
    assert skipped.request_attempt_count == 0


@pytest.mark.parametrize("value", (-1, 3, True, 1.0, "1"))
def test_ab_pair_attempt_count_is_strict(value: object) -> None:
    with pytest.raises(ValueError):
        ABPair(scenario_id="synthetic-case", status=ABStatus.PARTIAL, request_attempt_count=value)


def test_run_ab_suite_prefers_reported_count_and_keeps_legacy_inference(
    tmp_path: Path,
) -> None:
    class ReportedOrchestrator:
        def __init__(self, count: int | None) -> None:
            self.count = count

        def resolve_test_results_dir(self, supplied: Path) -> Path:
            return supplied.resolve()

        def reply_ab(self, scenario_id, request, evaluation_idle_seconds=None, **kwargs):
            return ABPair(
                scenario_id=scenario_id,
                status=ABStatus.PARTIAL,
                answer_key={"A": DialogueMode.BASELINE, "B": DialogueMode.STYLED},
                request_attempt_count=self.count,
            )

    scenarios = (
        ABScenario(
            scenario_id="synthetic-case",
            text="synthetic input",
            recent_context=(),
            idle_seconds=0,
        ),
    )

    reported = run_ab_suite(ReportedOrchestrator(1), scenarios, tmp_path / "reported")
    legacy = run_ab_suite(ReportedOrchestrator(None), scenarios, tmp_path / "legacy")

    assert reported.request_attempt_count == 1
    assert legacy.request_attempt_count == 2
    keys = (tmp_path / "reported" / reported.run_id / "answers.json").read_text(encoding="utf-8")
    assert "request_attempt_count" not in keys and "baseline" not in keys
