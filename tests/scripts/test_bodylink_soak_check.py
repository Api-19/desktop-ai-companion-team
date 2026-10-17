"""Synthetic evidence for independent, frozen soak acceptance."""

import copy
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from companion.bodylink.transport import TransportStats
from scripts import bodylink_soak_faults as faults
from scripts import bodylink_soak_resources as resources
from scripts import bodylink_soak_support as soak


def synthetic_report(scenario="steady"):
    duration = 7200.0 if scenario == "steady" else 60.0
    report = dict(
        schema_version=1,
        scenario=scenario,
        requested_seconds=duration,
        elapsed_seconds=duration,
        started_unix_s=1000.0 if scenario == "baseline" else 2000.0,
        finished_unix_s=1060.0 if scenario == "baseline" else 2000.0 + duration,
        run_started_monotonic_s=100.0,
        run_complete=True,
        interrupted=False,
        turns=dict(planned=1, started=1, completed=1, failed=0, aborted=0, missed=0),
        events=[],
        workers={},
        resources={},
        warmup_seconds=30.0,
        resource_sample_interval_seconds=5.0,
        resource_growth={},
        faults=[],
        limits_sha256="a" * 64,
        unavailable=list(soak.UNAVAILABLE_METRICS),
        cleanup=dict(live_children=0, residual_files=0),
    )
    for role in ("brain", "body"):
        stats = dataclasses.asdict(TransportStats(sessions_started=1))
        report["workers"][role] = {
            "1": {key: dict(stats) for key in ("start", "run_end", "after_close")}
        }
        samples = [
            dict(
                at_s=100.0 + offset,
                rss_bytes=1000 + min(offset, 40),
                thread_count=4 + int(offset >= 35),
                fd_count=8 + int(offset >= 40),
                cpu_seconds=offset / 100.0,
            )
            for offset in range(0, int(duration) + 1, 5)
        ]
        report["resources"][role] = {"1": samples}
        report["resource_growth"][role] = {
            "1": resources.resource_growth(
                [resources.ResourceSample(**s) for s in samples],
                run_started_monotonic_s=100.0,
                warmup_seconds=30.0,
            )
        }
    report["events"] = [
        dict(role="body", incarnation=1, kind=kind, phase="run", at_s=at, values=values)
        for kind, at, values in (
            ("recording_ready", 132.0, dict(count=1)),
            ("playback_started", 133.0, dict(count=1)),
            ("playback_completed", 134.0, dict(count=1, duplicates=0)),
        )
    ]
    if scenario == "faults":
        for case in faults.FAULT_CASES:
            reason = {"body_restart": "target_not_running", "body_pause": "target_suspended"}
            report["faults"].append(
                dict(
                    ordinal=case.ordinal,
                    kind=case.kind,
                    phase=case.phase,
                    applied=True,
                    released=True,
                    complete=True,
                    source_session_changed=True,
                    quiet_without_replay=True,
                    recovery_turn_completed=True,
                    liveness_seconds=4.0,
                    device_stop_seconds="unavailable" if case.kind in reason else 1.0,
                    handshake_recovery_seconds=0.5,
                    turn_recovery_seconds=2.0,
                    unavailable_reason=reason.get(case.kind, "none"),
                    unexpected_exits=[],
                )
            )
    return report


def synthetic_limits(baseline_sha256="b" * 64):
    return dict(
        schema_version=1,
        source_baseline_sha256=baseline_sha256,
        warmup_seconds=30.0,
        resource_sample_interval_seconds=5.0,
        **{
            role: dict(rss_growth_bytes=10, thread_growth=1, fd_growth=1)
            for role in ("brain", "body")
        },
    )


def write_inputs(tmp_path, report=None):
    baseline_path = tmp_path / "baseline.json"
    soak.write_report(baseline_path, synthetic_report("baseline"))
    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    limits_path = tmp_path / "limits.json"
    soak.write_report(limits_path, synthetic_limits(digest))
    report = copy.deepcopy(report if report is not None else synthetic_report())
    report["limits_sha256"] = hashlib.sha256(limits_path.read_bytes()).hexdigest()
    report_path = tmp_path / "report.json"
    soak.write_report(report_path, report)
    return [
        "--report",
        str(report_path),
        "--baseline",
        str(baseline_path),
        "--limits",
        str(limits_path),
        "--output",
        str(tmp_path / "verdict.json"),
    ]


def evaluate(report=None, baseline=None, limits=None):
    from scripts import bodylink_soak_check as check

    return check.evaluate_steady(
        report if report is not None else synthetic_report(),
        baseline if baseline is not None else synthetic_report("baseline"),
        limits if limits is not None else synthetic_limits(),
        baseline_sha256="b" * 64,
        limits_sha256="a" * 64,
    )


def test_complete_steady_report_passes_only_with_prebound_limits(tmp_path):
    from scripts import bodylink_soak_check as check

    args = write_inputs(tmp_path)
    assert check.main(args) == 0
    assert json.loads((tmp_path / "verdict.json").read_text()) == dict(
        schema_version=1,
        scenario="steady",
        verdict="pass",
        two_hours_passed=True,
        faults_passed=False,
        reasons=[],
    )
    report, digest = check.load_report(tmp_path / "report.json")
    assert digest == hashlib.sha256((tmp_path / "report.json").read_bytes()).hexdigest()
    report["limits_sha256"] = "unavailable"
    assert evaluate(report)["reasons"] == ["limits_not_frozen_before_run"]


def test_short_run_cannot_claim_two_hours():
    report = synthetic_report()
    report["requested_seconds"] = report["elapsed_seconds"] = 2.0
    result = evaluate(report)
    assert result["verdict"] == "incomplete"
    assert result["reasons"] == ["duration_insufficient"]
    assert not result["two_hours_passed"]


@pytest.mark.parametrize("field,value", [("run_complete", False), ("interrupted", True)])
def test_interrupted_run_is_incomplete(field, value):
    report = synthetic_report()
    report[field] = value
    assert evaluate(report)["reasons"] == ["run_incomplete"]


@pytest.mark.parametrize(
    "missing", ["snapshot", "counter", "duplicates", "samples", "rss", "gap", "origin"]
)
def test_missing_required_metrics_are_incomplete(missing):
    report = synthetic_report()
    reason = "missing_resource_metrics"
    if missing == "snapshot":
        report["workers"]["brain"]["1"]["after_close"] = "unavailable"
        reason = "missing_transport_metrics"
    elif missing == "counter":
        report["workers"]["brain"]["1"]["run_end"]["tx_audio_frames"] = "unavailable"
        reason = "missing_transport_metrics"
    elif missing == "duplicates":
        report["events"][-1]["values"]["duplicates"] = "unavailable"
        reason = "missing_transport_metrics"
    elif missing == "samples":
        report["resources"]["brain"]["1"] = report["resources"]["brain"]["1"][-2:]
    elif missing == "rss":
        report["resources"]["body"]["1"][6]["rss_bytes"] = "unavailable"
    elif missing == "gap":
        del report["resources"]["body"]["1"][100:103]
    else:
        report["run_started_monotonic_s"] = "unavailable"
    result = evaluate(report)
    assert result["verdict"] == "incomplete"
    assert result["reasons"] == [reason]


def test_known_unobservable_metrics_remain_unavailable():
    report = synthetic_report()
    before = copy.deepcopy(report)
    assert evaluate(report)["verdict"] == "pass"
    assert report == before
    assert report["unavailable"] == list(soak.UNAVAILABLE_METRICS)


def test_one_available_duplicate_counter_does_not_hide_missing_later_evidence():
    report = synthetic_report()
    report["events"].append(copy.deepcopy(report["events"][-1]))
    report["events"][-1]["at_s"] += 1
    report["events"][-1]["values"]["duplicates"] = "unavailable"
    result = evaluate(report)
    assert result["verdict"] == "incomplete"
    assert result["reasons"] == ["missing_transport_metrics"]


def test_limits_changed_after_run_are_rejected(tmp_path):
    from scripts import bodylink_soak_check as check

    args = write_inputs(tmp_path)
    path = tmp_path / "limits.json"
    limits = json.loads(path.read_text())
    limits["body"]["rss_growth_bytes"] += 1
    path.write_text(json.dumps(limits))
    assert check.main(args) == 1
    assert json.loads((tmp_path / "verdict.json").read_text())["reasons"] == [
        "limits_not_frozen_before_run"
    ]


@pytest.mark.parametrize("problem", ["digest", "later", "scenario", "incomplete", "growth"])
def test_limits_must_reference_preceding_baseline(problem):
    baseline, limits = synthetic_report("baseline"), synthetic_limits()
    reason = "invalid_baseline"
    if problem == "digest":
        limits["source_baseline_sha256"] = "c" * 64
        reason = "baseline_digest_mismatch"
    elif problem == "later":
        baseline["finished_unix_s"] = 2000.1
    elif problem == "scenario":
        baseline["scenario"] = "steady"
    elif problem == "incomplete":
        baseline["interrupted"] = True
    else:
        baseline["resource_growth"]["body"]["1"]["rss_growth_bytes"] = "unavailable"
    assert evaluate(baseline=baseline, limits=limits)["reasons"] == [reason]


@pytest.mark.parametrize("field", ["rss_growth_bytes", "thread_growth", "fd_growth"])
def test_resource_growth_exact_boundary_and_excess(field):
    limits = synthetic_limits()
    assert evaluate(limits=limits)["verdict"] == "pass"
    limits["body"][field] -= 1
    result = evaluate(limits=limits)
    assert result["verdict"] == "fail"
    assert result["reasons"] == ["resource_growth_exceeded"]


def test_resource_summary_is_recomputed_from_valid_post_warmup_samples():
    report = synthetic_report()
    assert report["resource_growth"]["body"]["1"]["rss_growth_bytes"] == 10
    assert evaluate(report)["verdict"] == "pass"
    report["resource_growth"]["body"]["1"]["rss_growth_bytes"] = 0
    assert evaluate(report)["reasons"] == ["resource_summary_mismatch"]


def test_cleanup_close_is_not_runtime_disconnect():
    report = synthetic_report()
    for role in ("brain", "body"):
        report["workers"][role]["1"]["after_close"]["sessions_ended_closed"] = 1
        report["events"].append(
            dict(
                role=role,
                incarnation=1,
                kind="session_ended",
                phase="cleanup",
                at_s=7301.0,
                values=dict(
                    count=1,
                    reason=0,
                    last_ended_session_valid_rx_monotonic_s=7300.0,
                    last_session_ended_monotonic_s=7301.0,
                ),
            )
        )
    assert evaluate(report)["verdict"] == "pass"


@pytest.mark.parametrize("problem", ["event", "counter", "session", "incarnation", "fault"])
def test_runtime_disconnect_or_restart_fails_steady(problem):
    report = synthetic_report()
    if problem == "event":
        report["events"].append(
            dict(
                role="brain",
                incarnation=1,
                kind="session_ended",
                phase="run",
                at_s=7300.0,
                values=dict(
                    count=1,
                    reason=0,
                    last_ended_session_valid_rx_monotonic_s=7299.0,
                    last_session_ended_monotonic_s=7300.0,
                ),
            )
        )
    elif problem == "counter":
        report["workers"]["brain"]["1"]["run_end"]["sessions_ended_peer_closed"] = 1
    elif problem == "session":
        report["workers"]["body"]["1"]["run_end"]["sessions_started"] = 2
    elif problem == "incarnation":
        report["events"][0]["incarnation"] = 2
    else:
        report["faults"] = synthetic_report("faults")["faults"][:1]
    result = evaluate(report)
    assert result["verdict"] == "fail"
    assert "unexpected_disconnect" in result["reasons"]


def test_multiple_incarnations_lack_steady_transport_evidence():
    report = synthetic_report()
    report["workers"]["body"]["2"] = copy.deepcopy(report["workers"]["body"]["1"])
    assert evaluate(report)["reasons"] == ["missing_transport_metrics", "missing_resource_metrics"]


@pytest.mark.parametrize("problem", ["disconnect", "protocol", "timeout"])
def test_periodic_runtime_snapshots_are_checked(problem):
    report = synthetic_report()
    stats = dict(report["workers"]["brain"]["1"]["run_end"])
    stats[
        {
            "disconnect": "sessions_ended_peer_closed",
            "protocol": "protocol_errors",
            "timeout": "sessions_ended_send_timeout",
        }[problem]
    ] = 1
    report["events"].append(
        dict(role="brain", incarnation=1, kind="snapshot", phase="run", at_s=500.0, values=stats)
    )
    assert evaluate(report)["verdict"] == "fail"


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "wrong_phase",
        "wrong_ordinal",
        "unapplied",
        "unreleased",
        "incomplete",
        "liveness",
        "handshake",
        "turn",
        "interrupted",
    ],
)
def test_faults_require_exact_twelve_cases(problem):
    from scripts import bodylink_soak_check as check

    report = synthetic_report("faults")
    case = report["faults"][0]
    if problem == "missing":
        report["faults"].pop()
    elif problem == "duplicate":
        report["faults"][-1] = dict(case)
    elif problem == "wrong_phase":
        case["phase"] = "playing"
    elif problem == "wrong_ordinal":
        case["ordinal"] = 99
    elif problem == "interrupted":
        report["interrupted"] = True
    else:
        field, value = {
            "unapplied": ("applied", False),
            "unreleased": ("released", False),
            "incomplete": ("complete", False),
            "liveness": ("liveness_seconds", "unavailable"),
            "handshake": ("handshake_recovery_seconds", "unavailable"),
            "turn": ("turn_recovery_seconds", "unavailable"),
        }[problem]
        case[field] = value
    result = check.evaluate_faults(report)
    assert result["verdict"] == "incomplete"
    assert result["reasons"] == ["fault_coverage_incomplete"]


def test_faults_and_steady_verdicts_are_separate():
    from scripts import bodylink_soak_check as check

    report = synthetic_report("faults")
    report["resource_growth"]["body"]["1"]["rss_growth_bytes"] = 999999999
    assert check.evaluate_faults(report) == dict(
        schema_version=1,
        scenario="faults",
        verdict="pass",
        two_hours_passed=False,
        faults_passed=True,
        reasons=[],
    )
    assert evaluate()["two_hours_passed"]
    assert not evaluate()["faults_passed"]


@pytest.mark.parametrize(
    "kind,reason,passed",
    [
        ("body_restart", "target_not_running", True),
        ("body_pause", "target_suspended", True),
        ("body_pause", "target_not_running", False),
        ("brain_restart", "target_not_running", False),
        ("downlink_blackhole", "target_suspended", False),
        ("body_restart", "missing_event", False),
    ],
)
def test_unavailable_stop_is_allowed_only_for_frozen_cases(kind, reason, passed):
    from scripts import bodylink_soak_check as check

    report = synthetic_report("faults")
    for case in report["faults"]:
        if case["kind"] == kind:
            case["device_stop_seconds"] = "unavailable"
            case["unavailable_reason"] = reason
    result = check.evaluate_faults(report)
    assert result["verdict"] == ("pass" if passed else "incomplete")


def test_fault_failure_reasons_are_ordered_and_deduplicated():
    from scripts import bodylink_soak_check as check

    report = synthetic_report("faults")
    for case in report["faults"][:2]:
        case["liveness_seconds"] = 4.001
        case["device_stop_seconds"] = 1.001
        case["source_session_changed"] = False
        case["quiet_without_replay"] = False
        case["unexpected_exits"] = [
            dict(kind="unexpected_worker_exit", role="brain", incarnation=1, exitcode=-9)
        ]
    report["cleanup"]["live_children"] = 1
    result = check.evaluate_faults(report)
    assert result["verdict"] == "fail"
    assert result["reasons"] == [
        "liveness_limit_exceeded",
        "device_stop_limit_exceeded",
        "recovery_failure",
        "unexpected_worker_exit",
        "cleanup_failure",
    ]


def test_steady_failure_reasons_are_ordered_and_do_not_require_tx_rx_equality():
    report = synthetic_report()
    report["workers"]["brain"]["1"]["run_end"]["tx_audio_frames"] = 99
    assert evaluate(report)["verdict"] == "pass"
    report["turns"]["failed"] = 1
    report["workers"]["body"]["1"]["run_end"]["protocol_errors"] = 1
    report["events"][-1]["values"]["duplicates"] = 1
    report["resource_growth"]["brain"]["1"]["fd_growth"] = 2
    report["cleanup"]["residual_files"] = 1
    assert evaluate(report)["reasons"] == [
        "turn_failure",
        "transport_failure",
        "audio_safety_failure",
        "resource_summary_mismatch",
        "cleanup_failure",
    ]


@pytest.mark.parametrize("field", ["warmup_seconds", "resource_sample_interval_seconds"])
def test_resource_configuration_must_match(field):
    baseline = synthetic_report("baseline")
    baseline[field] += 1
    assert evaluate(baseline=baseline)["reasons"] == ["resource_configuration_mismatch"]


@pytest.mark.parametrize("scenario", ["baseline", "faults"])
def test_cli_separates_baseline_and_faults(tmp_path, scenario):
    from scripts import bodylink_soak_check as check

    path = tmp_path / "report.json"
    soak.write_report(path, synthetic_report(scenario))
    output = tmp_path / "verdict.json"
    assert check.main(["--report", str(path), "--output", str(output)]) == (
        1 if scenario == "baseline" else 0
    )
    result = json.loads(output.read_text())
    assert result["reasons"] == (["baseline_not_acceptance"] if scenario == "baseline" else [])
    assert not result["two_hours_passed"]


@pytest.mark.parametrize(
    "problem",
    [
        "missing_baseline",
        "missing_limits",
        "fault_options",
        "override",
        "abbreviation",
        "existing_output",
    ],
)
def test_cli_errors_are_fixed_and_do_not_overwrite(tmp_path, capsys, problem):
    from scripts import bodylink_soak_check as check

    args = write_inputs(tmp_path)
    if problem in ("missing_baseline", "missing_limits"):
        index = args.index("--baseline" if problem == "missing_baseline" else "--limits")
        del args[index : index + 2]
    elif problem == "fault_options":
        (tmp_path / "report.json").write_text(json.dumps(synthetic_report("faults")))
    elif problem == "override":
        args += ["--duration-seconds", "1"]
    elif problem == "abbreviation":
        args[0] = "--rep"
    else:
        (tmp_path / "verdict.json").write_text("preserve")
    assert check.main(args) == 2
    assert capsys.readouterr().err == check.CHECK_ERROR + "\n"
    if problem == "existing_output":
        assert (tmp_path / "verdict.json").read_text() == "preserve"
    else:
        assert not (tmp_path / "verdict.json").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "utf8",
        "json",
        "top_extra",
        "nested_extra",
        "bool",
        "nan",
        "missing",
        "origin_int",
        "duplicate_key",
        "event_extra",
        "sample_extra",
    ],
)
def test_load_report_strict_schema(tmp_path, problem):
    from scripts import bodylink_soak_check as check

    report = synthetic_report()
    if problem == "top_extra":
        report["payload"] = "secret"
    elif problem == "nested_extra":
        report["workers"]["body"]["1"]["start"]["payload"] = "secret"
    elif problem == "bool":
        report["turns"]["planned"] = True
    elif problem == "nan":
        report["elapsed_seconds"] = float("nan")
    elif problem == "missing":
        del report["run_started_monotonic_s"]
    elif problem == "origin_int":
        report["run_started_monotonic_s"] = 100
    elif problem == "event_extra":
        report["events"][0]["values"]["payload"] = "secret"
    elif problem == "sample_extra":
        report["resources"]["brain"]["1"][0]["payload"] = "secret"
    raw = json.dumps(report).encode()
    if problem == "utf8":
        raw = b"\xff"
    elif problem == "json":
        raw = b"{"
    elif problem == "duplicate_key":
        raw = raw.replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1')
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=r"^invalid soak report$"):
        check.load_report(path)


def test_direct_script_entrypoint(tmp_path):
    args = write_inputs(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/bodylink_soak_check.py", *args],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert Path(args[-1]).exists()
