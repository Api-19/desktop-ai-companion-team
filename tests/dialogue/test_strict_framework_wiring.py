from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from dialogue_support import FakeProvider
from strict_framework_env import (
    APPROVED_COUNT,
    OWNER,
    PHASE1_FIELDS,
    RUN_AT,
    T0,
    StrictEnv,
    build_strict_env,
)

from companion.contracts import DialogueRequest
from companion.dialogue.ab import (
    ABStatus,
    build_real_orchestrator,
    load_framework20_scenarios,
    run_ab_suite,
)
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import DialogueMode
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.style import (
    SingleTrustedUserStyleEvidenceStore,
    SingleTrustedUserStyleProfileStore,
    UserStyleSnapshotReadResult,
)

BASELINE_FIRST = "strict-pair-1"
STYLED_FIRST = "strict-pair-0"
KNOWN_TEXT = "谢谢你帮我"
OPEN_TEXT = "今天随便聊聊吧"
MONTH = "2026-09"
BASELINE_COST = Decimal("0.00002")
SECRET_MARKER = "synthetic-tamper-marker"


class HookProvider(FakeProvider):
    """Run one synthetic mutation while the first lane is on the wire."""

    def __init__(self, hook: Callable[[], None] | None = None) -> None:
        super().__init__("好的谢谢你。")
        self.hook = hook

    def complete(self, messages: Sequence[PromptMessage]):
        result = super().complete(messages)
        if self.hook is not None:
            hook, self.hook = self.hook, None
            hook()
        return result


@pytest.fixture
def env(tmp_path: Path) -> StrictEnv:
    return build_strict_env(tmp_path)


def _build(env: StrictEnv, provider: FakeProvider | None = None) -> DialogueOrchestrator:
    orchestrator = build_real_orchestrator(
        env.settings,
        env.config,
        env.secrets,
        trusted_user_id=OWNER,
        enable_production_personalization=True,
    )
    orchestrator._provider = provider or HookProvider()
    orchestrator._clock = lambda: RUN_AT
    return orchestrator


def _request(text: str = KNOWN_TEXT) -> DialogueRequest:
    return DialogueRequest(user_id=OWNER, text=text, recent_context=[])


def _ledger_state(env: StrictEnv) -> dict[str, object]:
    ledger = MonthlyUsageLedger(env.settings.runtime_data_dir / "dialogue", env.config.budget)
    state = ledger._load_month(MONTH)
    return {"reservations": dict(state.reservations), "settled": state.settled_cny}


def _tree(root: Path) -> set[tuple[str, int]]:
    return {(str(path), path.stat().st_mtime_ns) for path in root.rglob("*")}


def _rewrite_admin(env: StrictEnv, text: str) -> None:
    (env.root / "admin" / "config.toml").write_text(text, encoding="utf-8")


def _revoke(env: StrictEnv) -> Callable[[], None]:
    return lambda: env.phase1.revoke_consent(user_id=OWNER, updated_at=T0 + timedelta(minutes=10))


def _approval_revision(env: StrictEnv) -> Callable[[], None]:
    return lambda: env.approvals.set_approved(
        OWNER, env.package, index=APPROVED_COUNT, approved=True
    )


def _profile_regenerated(env: StrictEnv) -> Callable[[], None]:
    return lambda: env.regenerate_profile(T0 + timedelta(minutes=10))


def _phase5_republished(env: StrictEnv) -> Callable[[], None]:
    return lambda: env.publish_phase5(
        ((2, "gratitude"), (3, "apology")), T0 + timedelta(minutes=10)
    )


def _import_drift(env: StrictEnv) -> Callable[[], None]:
    def mutate() -> None:
        path = env.root / "admin" / "import" / "current.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[0]["text"] = "synthetic drifted reply"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    return mutate


MUTATIONS = {
    "revoked": _revoke,
    "approval_revision": _approval_revision,
    "profile_regenerated": _profile_regenerated,
    "phase5_republished": _phase5_republished,
    "import_drift": _import_drift,
}


def test_builder_assembles_one_strict_gateway_from_admin_config(env: StrictEnv) -> None:
    before = _tree(env.root)

    orchestrator = _build(env)

    assert _tree(env.root) == before
    personalization = orchestrator._style_personalization
    profile = personalization._profile
    evidence = personalization._evidence
    assert type(profile) is SingleTrustedUserStyleProfileStore
    assert type(evidence) is SingleTrustedUserStyleEvidenceStore
    assert profile._strict_source_binding is True
    assert evidence._strict_source_binding is True
    assert profile._source_gateway is evidence._source_gateway
    assert orchestrator._style_verifier is not None


def test_correct_strict_sources_pass_framework20_preflight_without_provider(
    env: StrictEnv,
) -> None:
    provider = HookProvider()
    orchestrator = _build(env, provider)
    scenarios = load_framework20_scenarios(Path("config/dialogue-ab-full-scenarios.json"))

    orchestrator.preflight_framework20(
        scenarios, trusted_user_id=OWNER, results_dir=env.settings.test_results_dir
    )

    assert provider.call_count == 0
    assert not env.settings.test_results_dir.exists()


def _swap_fields(env: StrictEnv, first: str, second: str) -> None:
    sources = env.config.style_framework
    env.config = env.config.model_copy(
        update={
            "style_framework": sources.model_copy(
                update={first: getattr(sources, second), second: getattr(sources, first)}
            )
        }
    )


def _legacy_paths(env: StrictEnv) -> None:
    sources = env.config.style_framework
    env.config = env.config.model_copy(
        update={
            "style_framework": sources.model_copy(
                update={
                    field: env.root / "framework" / f"{field}.json"
                    for field in type(sources).model_fields
                }
            )
        }
    )


def _loose_mode(env: StrictEnv) -> None:
    (env.root / "style-framework" / "phase1" / "profile.json").chmod(0o644)


def _symlinked_phase5(env: StrictEnv) -> None:
    folder = env.root / "style-framework" / "phase5"
    moved = env.root / "style-framework" / "phase5-real"
    folder.rename(moved)
    folder.symlink_to(moved, target_is_directory=True)


BUILD_FAILURES: dict[str, Callable[[StrictEnv], None]] = {
    "missing_owner": lambda env: setattr(
        env, "config", env.config.model_copy(update={"style_framework_owner_id": None})
    ),
    "config_owner_mismatch": lambda env: setattr(
        env, "config", env.config.model_copy(update={"style_framework_owner_id": "other-owner"})
    ),
    "admin_owner_mismatch": lambda env: _rewrite_admin(
        env, 'enabled = true\nowner_id = "other-owner"\nimport_format = "standard_json"\n'
    ),
    "import_format_disabled": lambda env: _rewrite_admin(
        env, f'enabled = true\nowner_id = "{OWNER}"\n'
    ),
    "admin_disabled": lambda env: _rewrite_admin(
        env, f'enabled = false\nowner_id = "{OWNER}"\nimport_format = "standard_json"\n'
    ),
    "legacy_paths": _legacy_paths,
    "reordered_phase1": lambda env: _swap_fields(env, PHASE1_FIELDS[0], PHASE1_FIELDS[1]),
    "loose_mode": _loose_mode,
    "symlinked_phase5": _symlinked_phase5,
}


@pytest.mark.parametrize("fault", sorted(BUILD_FAILURES))
def test_builder_rejects_untrusted_admin_binding_before_network(
    env: StrictEnv, fault: str, capsys: pytest.CaptureFixture[str]
) -> None:
    BUILD_FAILURES[fault](env)

    with pytest.raises(ValueError, match="production personalization configuration unavailable"):
        build_real_orchestrator(
            env.settings,
            env.config,
            env.secrets,
            trusted_user_id=OWNER,
            enable_production_personalization=True,
        )

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_builder_rejects_trusted_user_other_than_admin_owner(env: StrictEnv) -> None:
    with pytest.raises(ValueError, match="production personalization configuration unavailable"):
        build_real_orchestrator(
            env.settings,
            env.config,
            env.secrets,
            trusted_user_id="other-owner",
            enable_production_personalization=True,
        )


@pytest.mark.parametrize("fault", ("approval_revision", "import_drift", "revoked"))
def test_stale_sources_fail_preflight_and_pair_with_zero_requests(
    env: StrictEnv, fault: str
) -> None:
    provider = HookProvider()
    orchestrator = _build(env, provider)
    MUTATIONS[fault](env)()
    scenarios = load_framework20_scenarios(Path("config/dialogue-ab-full-scenarios.json"))

    with pytest.raises(ValueError, match="framework preflight unavailable"):
        orchestrator.preflight_framework20(
            scenarios, trusted_user_id=OWNER, results_dir=env.settings.test_results_dir
        )
    pair = orchestrator.reply_ab(BASELINE_FIRST, _request(), trusted_user_id=OWNER)

    assert pair.status is ABStatus.FAILED
    assert pair.request_attempt_count == 0
    assert provider.call_count == 0
    assert not (env.settings.runtime_data_dir / "dialogue").exists()


def test_corrupt_phase5_refuses_pair_before_reservation(env: StrictEnv) -> None:
    provider = HookProvider()
    orchestrator = _build(env, provider)
    (env.root / "style-framework" / "phase5" / "evidence.json").write_text("{}", encoding="utf-8")

    pair = orchestrator.reply_ab(BASELINE_FIRST, _request(), trusted_user_id=OWNER)
    run = orchestrator.reply(_request(), trusted_user_id=OWNER)

    assert pair.status is ABStatus.FAILED
    assert pair.request_attempt_count == 0
    assert run.response.blocked_reason == "configuration_unavailable"
    assert run.trace.provider_attempted is False
    assert provider.call_count == 0
    assert not (env.settings.runtime_data_dir / "dialogue").exists()


def test_unchanged_sources_complete_pair_with_two_attempts(env: StrictEnv) -> None:
    provider = HookProvider()
    orchestrator = _build(env, provider)

    pair = orchestrator.reply_ab(BASELINE_FIRST, _request(), trusted_user_id=OWNER)

    assert pair.status is ABStatus.COMPLETE
    assert pair.request_attempt_count == 2
    assert provider.call_count == 2
    assert _ledger_state(env) == {"reservations": {}, "settled": BASELINE_COST * 2}


@pytest.mark.parametrize("fault", sorted(MUTATIONS))
def test_first_lane_mutation_blocks_second_styled_send_and_cancels_reservation(
    env: StrictEnv, fault: str
) -> None:
    provider = HookProvider(MUTATIONS[fault](env))
    orchestrator = _build(env, provider)

    pair = orchestrator.reply_ab(BASELINE_FIRST, _request(), trusted_user_id=OWNER)

    assert provider.call_count == 1
    assert pair.status is ABStatus.PARTIAL
    assert pair.request_attempt_count == 1
    assert set(pair.answers) == {"A"}
    assert pair.answer_key == {"A": DialogueMode.BASELINE, "B": DialogueMode.STYLED}
    assert _ledger_state(env) == {"reservations": {}, "settled": BASELINE_COST}
    restarted = MonthlyUsageLedger(env.settings.runtime_data_dir / "dialogue", env.config.budget)
    restarted.ensure_available(count=2, now=RUN_AT)


def test_styled_first_refusal_still_reports_independent_baseline(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = HookProvider()
    orchestrator = _build(env, provider)
    reserve = orchestrator._ledger.reserve

    def reserve_then_revoke(count, now):
        reservations = reserve(count=count, now=now)
        _revoke(env)()
        return reservations

    monkeypatch.setattr(orchestrator._ledger, "reserve", reserve_then_revoke)

    pair = orchestrator.reply_ab(STYLED_FIRST, _request(), trusted_user_id=OWNER)

    assert pair.answer_key == {"A": DialogueMode.STYLED, "B": DialogueMode.BASELINE}
    assert provider.call_count == 1
    assert pair.status is ABStatus.PARTIAL
    assert pair.request_attempt_count == 1
    assert set(pair.answers) == {"B"}
    assert _ledger_state(env) == {"reservations": {}, "settled": BASELINE_COST}


def test_direct_styled_reply_rechecks_before_send(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = HookProvider()
    orchestrator = _build(env, provider)
    reserve = orchestrator._ledger.reserve

    def reserve_then_regenerate(count, now):
        reservations = reserve(count=count, now=now)
        _profile_regenerated(env)()
        return reservations

    monkeypatch.setattr(orchestrator._ledger, "reserve", reserve_then_regenerate)

    run = orchestrator.reply(_request(), trusted_user_id=OWNER)

    assert provider.call_count == 0
    assert run.response.blocked_reason == "configuration_unavailable"
    assert run.response.billable is False
    assert run.trace.provider_attempted is False
    assert _ledger_state(env) == {"reservations": {}, "settled": Decimal("0")}


def test_metadata_cache_update_is_not_content_drift(env: StrictEnv) -> None:
    orchestrator = _build(env)

    def other_route_updates_cache() -> None:
        from companion.dialogue.context import prepare_dialogue_input

        orchestrator._style_personalization.compile(
            user_id=OWNER,
            prepared=prepare_dialogue_input(_request("对不起我来晚了")),
            idle_seconds=None,
            observed_at=RUN_AT + timedelta(minutes=5),
        )

    orchestrator._provider.hook = other_route_updates_cache

    pair = orchestrator.reply_ab(BASELINE_FIRST, _request(), trusted_user_id=OWNER)

    assert pair.status is ABStatus.COMPLETE
    assert pair.request_attempt_count == 2


def test_open_set_pair_reads_no_phase5_resources(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _build(env)

    def fail_load(*args, **kwargs):
        raise AssertionError("phase5 read")

    monkeypatch.setattr(SingleTrustedUserStyleEvidenceStore, "_load_state", fail_load)

    pair = orchestrator.reply_ab(BASELINE_FIRST, _request(OPEN_TEXT), trusted_user_id=OWNER)

    assert pair.status is ABStatus.COMPLETE
    assert pair.request_attempt_count == 2


def test_standalone_baseline_reads_no_framework_sources(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _build(env)

    def fail(*args, **kwargs):
        raise AssertionError("source read")

    monkeypatch.setattr(SingleTrustedUserStyleProfileStore, "read_snapshot_state", fail)
    monkeypatch.setattr(SingleTrustedUserStyleEvidenceStore, "retrieve_evidence", fail)
    monkeypatch.setattr(type(env.gateway), "current_state", fail)

    run = orchestrator.reply(_request(), DialogueMode.BASELINE, trusted_user_id=OWNER)

    assert run.response.blocked_reason is None
    assert run.trace.provider_call_count == 1


def test_default_builder_never_loads_admin_or_framework_sources(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    import companion.admin.app as admin_app

    def fail(*args, **kwargs):
        raise AssertionError("source read")

    monkeypatch.setattr(admin_app, "load_admin_config", fail)
    monkeypatch.setattr(SingleTrustedUserStyleProfileStore, "read_snapshot_state", fail)
    monkeypatch.setattr(SingleTrustedUserStyleEvidenceStore, "retrieve_evidence", fail)

    orchestrator = build_real_orchestrator(
        env.settings, env.config, env.secrets, trusted_user_id=OWNER
    )
    orchestrator._provider = HookProvider()
    orchestrator._clock = lambda: RUN_AT
    orchestrator.reply(_request(), DialogueMode.BASELINE, trusted_user_id=OWNER)
    orchestrator.reply(_request(), trusted_user_id=OWNER)

    assert orchestrator._style_personalization is None


def test_recheck_exception_is_silent_and_cancels(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    orchestrator = _build(env)
    profile = orchestrator._style_personalization._profile
    original = profile.read_snapshot_state

    def raise_on_recheck():
        def raising(user_id):
            raise RuntimeError(SECRET_MARKER)

        monkeypatch.setattr(profile, "read_snapshot_state", raising)

    orchestrator._provider.hook = raise_on_recheck

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pair = orchestrator.reply_ab(BASELINE_FIRST, _request(), trusted_user_id=OWNER)

    assert original is not None
    assert pair.status is ABStatus.PARTIAL
    assert pair.request_attempt_count == 1
    assert SECRET_MARKER not in pair.model_dump_json()
    captured = capsys.readouterr()
    assert SECRET_MARKER not in captured.out + captured.err
    assert _ledger_state(env) == {"reservations": {}, "settled": BASELINE_COST}


def test_tampered_recheck_model_does_not_leak_through_pydantic(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    orchestrator = _build(env)
    profile = orchestrator._style_personalization._profile

    def tamper():
        genuine = profile.read_snapshot_state(OWNER)
        forged = UserStyleSnapshotReadResult.model_construct(
            status=genuine.status,
            snapshot=genuine.snapshot.model_copy(update={"evidence_count": SECRET_MARKER}),
        )
        monkeypatch.setattr(profile, "read_snapshot_state", lambda user_id: forged)

    orchestrator._provider.hook = tamper

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pair = orchestrator.reply_ab(BASELINE_FIRST, _request(), trusted_user_id=OWNER)

    assert pair.status is ABStatus.PARTIAL
    assert pair.request_attempt_count == 1
    captured = capsys.readouterr()
    assert SECRET_MARKER not in captured.out + captured.err + pair.model_dump_json()


def test_cancel_failure_keeps_reservation_held_without_output(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = HookProvider(_revoke(env))
    orchestrator = _build(env, provider)

    def fail_cancel(reservation, now):
        raise RuntimeError(SECRET_MARKER)

    monkeypatch.setattr(orchestrator._ledger, "cancel_unused", fail_cancel)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pair = orchestrator.reply_ab(BASELINE_FIRST, _request(), trusted_user_id=OWNER)

    assert provider.call_count == 1
    assert pair.status is ABStatus.PARTIAL
    assert pair.request_attempt_count == 1
    state = _ledger_state(env)
    assert len(state["reservations"]) == 1
    assert state["settled"] == BASELINE_COST
    captured = capsys.readouterr()
    assert SECRET_MARKER not in captured.out + captured.err + pair.model_dump_json()


def test_suite_reports_actual_attempts_after_mid_run_revocation(env: StrictEnv) -> None:
    scenarios = load_framework20_scenarios(Path("config/dialogue-ab-full-scenarios.json"))
    provider = HookProvider(_revoke(env))
    orchestrator = _build(env, provider)

    summary = run_ab_suite(
        orchestrator, scenarios, env.settings.test_results_dir, trusted_user_id=OWNER
    )

    assert summary.request_attempt_count == provider.call_count
    assert summary.request_attempt_count < 40
    run_dir = env.settings.test_results_dir / summary.run_id
    answers = (run_dir / "answers.json").read_text(encoding="utf-8")
    assert "baseline" not in answers and "styled" not in answers
    assert _ledger_state(env)["reservations"] == {}


def _script_with_real_builder(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch, provider: FakeProvider
) -> object:
    from test_ab_script import load_script

    module = load_script()
    real_builder = module.build_real_orchestrator

    def build(*args, **kwargs):
        orchestrator = real_builder(*args, **kwargs)
        orchestrator._provider = provider
        orchestrator._clock = lambda: RUN_AT
        return orchestrator

    monkeypatch.setattr(module, "load_settings", lambda root: env.settings)
    monkeypatch.setattr(module, "load_dialogue_config", lambda settings: (env.config, env.secrets))
    monkeypatch.setattr(module, "build_real_orchestrator", build)
    return module


def test_script_preflight_uses_real_strict_builder_without_provider(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = HookProvider()
    module = _script_with_real_builder(env, monkeypatch, provider)

    status = module.main("framework20", enable_production_personalization=True, preflight_only=True)

    output = capsys.readouterr().out
    assert status == 0
    assert json.loads(output) == {
        "status": "preflight_ready",
        "scenario_count": 20,
        "provider_attempt_count": 0,
    }
    assert provider.call_count == 0
    assert OWNER not in output


@pytest.mark.parametrize(
    ("fault", "expected"),
    (
        ("admin_owner_mismatch", "configuration_unavailable"),
        ("legacy_paths", "configuration_unavailable"),
        ("approval_revision", "preflight_unavailable"),
    ),
)
def test_script_refuses_untrusted_sources_before_provider(
    env: StrictEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fault: str,
    expected: str,
) -> None:
    if fault in BUILD_FAILURES:
        BUILD_FAILURES[fault](env)
    else:
        MUTATIONS[fault](env)()
    provider = HookProvider()
    module = _script_with_real_builder(env, monkeypatch, provider)

    status = module.main("framework20", enable_production_personalization=True)

    assert status == 2
    assert json.loads(capsys.readouterr().out) == {"status": expected}
    assert provider.call_count == 0
    assert not (env.settings.runtime_data_dir / "dialogue").exists()


def test_script_full_run_reports_actual_attempts_after_revocation(
    env: StrictEnv, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = HookProvider()
    module = _script_with_real_builder(env, monkeypatch, provider)
    # The smoke request re-arms the hook, so revocation lands inside the first suite lane.
    provider.hook = lambda: setattr(provider, "hook", _revoke(env))

    status = module.main("framework20", enable_production_personalization=True)

    summary = json.loads(capsys.readouterr().out)
    assert status == 1
    assert provider.call_count == 2
    assert summary["request_attempt_count"] == 1
    assert summary["paired_count"] == 0
    assert _ledger_state(env)["reservations"] == {}
