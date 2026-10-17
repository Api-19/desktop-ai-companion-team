"""Offline acceptance of payload-free soak evidence and prebound limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import fields
from itertools import pairwise
from pathlib import Path

from companion.bodylink.transport import TransportStats

if __package__:
    from .bodylink_soak_faults import DEVICE_STOP_LIMIT_S, FAULT_CASES, LIVENESS_LIMIT_S
    from .bodylink_soak_resources import ResourceSample, resource_growth
    from .bodylink_soak_support import (
        END_REASONS,
        UNAVAILABLE,
        UNAVAILABLE_METRICS,
        _artifact_path,
        load_frozen_limits,
        write_report,
    )
else:
    from bodylink_soak_faults import DEVICE_STOP_LIMIT_S, FAULT_CASES, LIVENESS_LIMIT_S
    from bodylink_soak_resources import ResourceSample, resource_growth
    from bodylink_soak_support import (
        END_REASONS,
        UNAVAILABLE,
        UNAVAILABLE_METRICS,
        _artifact_path,
        load_frozen_limits,
        write_report,
    )

CHECK_ERROR = "bodylink soak check failed"
ROLES = ("brain", "body")
GROWTH_KEYS = ("rss_growth_bytes", "thread_growth", "fd_growth")
RESOURCE_KEYS = ("rss_bytes", "thread_count", "fd_count", "cpu_seconds")
STATS_KEYS = {field.name for field in fields(TransportStats)}
TIME_STATS = {key for key in STATS_KEYS if key.endswith("_monotonic_s")}
REQUIRED_COUNTS = STATS_KEYS - TIME_STATS - {"connection_attempts", "connection_failures"}
REPORT_KEYS = {
    "schema_version",
    "scenario",
    "requested_seconds",
    "elapsed_seconds",
    "started_unix_s",
    "finished_unix_s",
    "run_started_monotonic_s",
    "run_complete",
    "interrupted",
    "turns",
    "events",
    "workers",
    "resources",
    "warmup_seconds",
    "resource_sample_interval_seconds",
    "resource_growth",
    "faults",
    "limits_sha256",
    "unavailable",
    "cleanup",
}
FAULT_KEYS = {
    "ordinal",
    "kind",
    "phase",
    "applied",
    "released",
    "complete",
    "source_session_changed",
    "quiet_without_replay",
    "recovery_turn_completed",
    "liveness_seconds",
    "device_stop_seconds",
    "handshake_recovery_seconds",
    "turn_recovery_seconds",
    "unavailable_reason",
    "unexpected_exits",
}
FAULT_FLAGS = (
    "applied",
    "released",
    "complete",
    "source_session_changed",
    "quiet_without_replay",
    "recovery_turn_completed",
)
FAULT_TIMES = (
    "liveness_seconds",
    "device_stop_seconds",
    "handshake_recovery_seconds",
    "turn_recovery_seconds",
)


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("invalid soak report")


def _keys(value: object, expected: set[str]) -> None:
    _require(type(value) is dict and set(value) == expected)


def _number(value: object) -> bool:
    return _finite(value) and value >= 0


def _finite(value: object) -> bool:
    return type(value) in (int, float) and (type(value) is int or math.isfinite(value))


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def _stats(value: object) -> None:
    _keys(value, STATS_KEYS)
    for key, number in value.items():
        _require(
            number == UNAVAILABLE or (_number(number) if key in TIME_STATS else _count(number))
        )


def _resource_values(value: object) -> None:
    _keys(value, set(RESOURCE_KEYS))
    for key, number in value.items():
        _require(
            number == UNAVAILABLE or (_number(number) if key == "cpu_seconds" else _count(number))
        )


def _event(event: object) -> None:
    _keys(event, {"role", "incarnation", "kind", "phase", "at_s", "values"})
    _require(event["role"] in ROLES and _count(event["incarnation"]) and event["incarnation"] > 0)
    _require(event["phase"] in ("startup", "run", "cleanup") and _number(event["at_s"]))
    kind, values = event["kind"], event["values"]
    _require(type(kind) is str and type(values) is dict)
    if kind == "snapshot":
        _stats(values)
    elif kind == "resource_sample":
        _resource_values(values)
    else:
        if kind == "ready":
            _require(set(values) in ({"pid"}, {"mute_ready"}, {"pid", "port", "mute_ready"}))
        elif kind == "session_started":
            _keys(values, {"count", "changed"})
        elif kind == "session_ended":
            required = {
                "count",
                "reason",
                "last_ended_session_valid_rx_monotonic_s",
                "last_session_ended_monotonic_s",
            }
            _require(
                required
                <= set(values)
                <= required
                | {
                    "callback_at_s",
                    "input_closed_at_s",
                    "speaker_returned_at_s",
                }
            )
        elif kind in ("playback_completed", "playback_stopped"):
            _keys(values, {"count", "duplicates"})
        elif kind in ("recording_ready", "playback_started", "turn_completed", "turn_failed"):
            _keys(values, {"count"})
        elif kind in ("worker_error", "stopped"):
            _keys(values, {"code"})
        else:
            _require(False)
        for key, number in values.items():
            if key in ("changed", "mute_ready"):
                _require(type(number) is bool)
            else:
                _require(
                    number == UNAVAILABLE
                    or (_number(number) if key.endswith("_s") else _count(number))
                )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def load_report(path: Path) -> tuple[dict[str, object], str]:
    """Read exactly one allowed artifact, validating its entire payload-free schema."""
    raw = _artifact_path(path).read_bytes()
    try:
        report = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError):
        raise ValueError("invalid soak report") from None
    _keys(report, REPORT_KEYS)
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1)
    _require(report["scenario"] in ("steady", "faults", "baseline"))
    for key in (
        "requested_seconds",
        "elapsed_seconds",
        "started_unix_s",
        "finished_unix_s",
        "warmup_seconds",
        "resource_sample_interval_seconds",
    ):
        _require(_number(report[key]))
    _require(report["resource_sample_interval_seconds"] > 0)
    origin = report["run_started_monotonic_s"]
    _require(origin == UNAVAILABLE or (type(origin) is float and _number(origin)))
    for key in ("run_complete", "interrupted"):
        _require(type(report[key]) is bool)
    digest = report["limits_sha256"]
    _require(
        type(digest) is str
        and (digest == UNAVAILABLE or re.fullmatch(r"[0-9a-f]{64}", digest) is not None)
    )
    _keys(report["turns"], {"planned", "started", "completed", "failed", "aborted", "missed"})
    _require(all(_count(value) for value in report["turns"].values()))
    _keys(report["cleanup"], {"live_children", "residual_files"})
    _require(all(value == UNAVAILABLE or _count(value) for value in report["cleanup"].values()))
    _require(
        type(report["unavailable"]) is list and report["unavailable"] == list(UNAVAILABLE_METRICS)
    )
    for section in ("workers", "resources", "resource_growth"):
        _keys(report[section], set(ROLES))
        for incarnations in report[section].values():
            _require(type(incarnations) is dict)
            for incarnation, value in incarnations.items():
                _require(re.fullmatch(r"[1-9][0-9]*", incarnation) is not None)
                if section == "workers":
                    _keys(value, {"start", "run_end", "after_close"})
                    for snapshot in value.values():
                        if snapshot != UNAVAILABLE:
                            _stats(snapshot)
                elif section == "resources":
                    _require(type(value) is list)
                    for sample in value:
                        _keys(sample, {"at_s", *RESOURCE_KEYS})
                        _require(_number(sample["at_s"]))
                        _resource_values({key: sample[key] for key in RESOURCE_KEYS})
                else:
                    _keys(value, {*GROWTH_KEYS, "cpu_seconds"})
                    for key, number in value.items():
                        # CPU is a measured difference, not an acceptance threshold.
                        _require(
                            number == UNAVAILABLE
                            or (_finite(number) if key == "cpu_seconds" else _count(number))
                        )
    _require(type(report["events"]) is list)
    for event in report["events"]:
        _event(event)
    _require(type(report["faults"]) is list)
    for case in report["faults"]:
        _keys(case, FAULT_KEYS)
        _require(_count(case["ordinal"]) and type(case["kind"]) is str)
        _require(case["phase"] in ("recording", "playing"))
        for key in FAULT_FLAGS:
            _require(type(case[key]) is bool)
        for key in FAULT_TIMES:
            value = case[key]
            _require(value == UNAVAILABLE or _finite(value))
        _require(
            case["unavailable_reason"]
            in (
                "none",
                "target_not_running",
                "target_suspended",
                "missing_event",
            )
        )
        _require(type(case["unexpected_exits"]) is list)
        for worker_exit in case["unexpected_exits"]:
            _keys(worker_exit, {"kind", "role", "incarnation", "exitcode"})
            _require(
                worker_exit["kind"] == "unexpected_worker_exit" and worker_exit["role"] in ROLES
            )
            _require(_count(worker_exit["incarnation"]) and worker_exit["incarnation"] > 0)
            _require(type(worker_exit["exitcode"]) is int)
    return report, hashlib.sha256(raw).hexdigest()


def _result(scenario: str, verdict: str, reasons: list[str]) -> dict[str, object]:
    return dict(
        schema_version=1,
        scenario=scenario,
        verdict=verdict,
        two_hours_passed=scenario == "steady" and verdict == "pass",
        faults_passed=scenario == "faults" and verdict == "pass",
        reasons=reasons,
    )


def _transport_complete(report: dict[str, object]) -> bool:
    for incarnations in report["workers"].values():
        if len(incarnations) != 1:
            return False
        snapshots = next(iter(incarnations.values()))
        if any(
            snapshot == UNAVAILABLE or any(snapshot[key] == UNAVAILABLE for key in REQUIRED_COUNTS)
            for snapshot in snapshots.values()
        ):
            return False
    playback = [
        event
        for event in report["events"]
        if event["role"] == "body"
        and event["phase"] == "run"
        and event["kind"] in ("playback_completed", "playback_stopped")
    ]
    return (
        not any(
            event["values"][key] == UNAVAILABLE
            for event in report["events"]
            if event["kind"] == "snapshot" and event["phase"] == "run"
            for key in REQUIRED_COUNTS
        )
        and bool(playback)
        and all(event["values"]["duplicates"] != UNAVAILABLE for event in playback)
    )


def _resources_complete(report: dict[str, object]) -> bool:
    origin = report["run_started_monotonic_s"]
    if origin == UNAVAILABLE:
        return False
    interval = report["resource_sample_interval_seconds"]
    for role in ROLES:
        incarnations = report["resources"][role]
        if len(incarnations) != 1 or incarnations.keys() != report["workers"][role].keys():
            return False
        samples = next(iter(incarnations.values()))
        if sum(s["at_s"] >= origin + report["warmup_seconds"] for s in samples) < 3:
            return False
        if any(s[key] == UNAVAILABLE for s in samples for key in RESOURCE_KEYS[:3]):
            return False
        if any(not 0 < b["at_s"] - a["at_s"] <= 2 * interval for a, b in pairwise(samples)):
            return False
        if samples[0]["at_s"] > origin + 2 * interval or (
            samples[-1]["at_s"] < origin + report["elapsed_seconds"] - 2 * interval
        ):
            return False
    return True


def _valid_baseline(baseline: dict[str, object], report: dict[str, object]) -> bool:
    return bool(
        baseline["scenario"] == "baseline"
        and baseline["run_complete"]
        and not baseline["interrupted"]
        and baseline["finished_unix_s"] <= report["started_unix_s"]
        and all(
            len(baseline["resource_growth"][role]) == 1
            and all(
                value[key] != UNAVAILABLE
                for value in baseline["resource_growth"][role].values()
                for key in GROWTH_KEYS
            )
            for role in ROLES
        )
    )


def _audio_unsafe(events: list[dict[str, object]]) -> bool:
    offline = False
    for event in sorted(events, key=lambda e: e["at_s"]):
        if event["role"] != "body" or event["phase"] != "run":
            continue
        if event["values"].get("duplicates", 0) != 0:
            return True
        if event["kind"] == "session_ended":
            offline = True
        elif event["kind"] == "session_started":
            offline = False
        elif offline and event["kind"] in ("recording_ready", "playback_started"):
            return True
    return False


def evaluate_steady(
    report: dict[str, object],
    baseline: dict[str, object],
    limits: dict[str, object],
    *,
    baseline_sha256: str,
    limits_sha256: str,
) -> dict[str, object]:
    """Judge already validated in-memory evidence; never read or amend limits."""
    reasons = []
    if not report["run_complete"] or report["interrupted"]:
        reasons.append("run_incomplete")
    if (
        report["requested_seconds"] < 7200
        or report["elapsed_seconds"] < report["requested_seconds"]
    ):
        reasons.append("duration_insufficient")
    if not _transport_complete(report):
        reasons.append("missing_transport_metrics")
    if not _resources_complete(report):
        reasons.append("missing_resource_metrics")
    if not _valid_baseline(baseline, report):
        reasons.append("invalid_baseline")
    if limits["source_baseline_sha256"] != baseline_sha256:
        reasons.append("baseline_digest_mismatch")
    if report["limits_sha256"] == UNAVAILABLE or report["limits_sha256"] != limits_sha256:
        reasons.append("limits_not_frozen_before_run")
    if any(
        report[key] != baseline[key] or report[key] != limits[key]
        for key in ("warmup_seconds", "resource_sample_interval_seconds")
    ):
        reasons.append("resource_configuration_mismatch")
    if reasons:
        return _result("steady", "incomplete", reasons)

    runtime = [event for event in report["events"] if event["phase"] == "run"]
    end_stats = [next(iter(report["workers"][role].values()))["run_end"] for role in ROLES]
    runtime_stats = end_stats + [
        next(iter(report["workers"][role].values()))["start"] for role in ROLES
    ]
    runtime_stats += [event["values"] for event in runtime if event["kind"] == "snapshot"]
    if (
        report["faults"]
        or any(
            event["kind"] == "session_ended"
            or str(event["incarnation"]) not in report["workers"][event["role"]]
            for event in runtime
        )
        or any(stats["sessions_started"] != 1 for stats in end_stats)
        or any(
            stats["sessions_started"] > 1
            or any(stats[f"sessions_ended_{reason}"] != 0 for reason in END_REASONS)
            for stats in runtime_stats
        )
    ):
        reasons.append("unexpected_disconnect")
    turns = report["turns"]
    if not (
        turns["planned"] > 0
        and turns["started"] == turns["completed"] == turns["planned"]
        and all(turns[key] == 0 for key in ("failed", "aborted", "missed"))
    ):
        reasons.append("turn_failure")
    if any(
        stats[key] != 0
        for stats in runtime_stats
        for key in ("protocol_errors", "sessions_ended_send_timeout")
    ):
        reasons.append("transport_failure")
    if _audio_unsafe(report["events"]):
        reasons.append("audio_safety_failure")
    recalculated = {
        role: {
            incarnation: resource_growth(
                [ResourceSample(**sample) for sample in samples],
                run_started_monotonic_s=report["run_started_monotonic_s"],
                warmup_seconds=report["warmup_seconds"],
            )
            for incarnation, samples in report["resources"][role].items()
        }
        for role in ROLES
    }
    if recalculated != report["resource_growth"]:
        reasons.append("resource_summary_mismatch")
    if any(
        growth[key] > limits[role][key]
        for role in ROLES
        for growth in recalculated[role].values()
        for key in GROWTH_KEYS
    ):
        reasons.append("resource_growth_exceeded")
    if any(value != 0 for value in report["cleanup"].values()):
        reasons.append("cleanup_failure")
    return _result("steady", "fail" if reasons else "pass", reasons)


def evaluate_faults(report: dict[str, object]) -> dict[str, object]:
    """Judge the frozen fault matrix independently of duration and RSS limits."""
    cases = report["faults"]
    expected = {(case.ordinal, case.kind, case.phase) for case in FAULT_CASES}
    complete = (
        report["run_complete"]
        and not report["interrupted"]
        and len(cases) == len(FAULT_CASES)
        and {(case["ordinal"], case["kind"], case["phase"]) for case in cases} == expected
    )
    exempt = {"body_restart": "target_not_running", "body_pause": "target_suspended"}
    for case in cases:
        complete = complete and all(case[key] for key in ("applied", "released", "complete"))
        complete = complete and all(
            case[key] != UNAVAILABLE and case[key] >= 0
            for key in (
                "liveness_seconds",
                "handshake_recovery_seconds",
                "turn_recovery_seconds",
            )
        )
        if case["kind"] in exempt:
            complete = (
                complete
                and case["device_stop_seconds"] == UNAVAILABLE
                and (case["unavailable_reason"] == exempt[case["kind"]])
            )
        else:
            complete = (
                complete
                and case["device_stop_seconds"] != UNAVAILABLE
                and (case["device_stop_seconds"] >= 0)
                and (case["unavailable_reason"] == "none")
            )
    if not complete:
        return _result("faults", "incomplete", ["fault_coverage_incomplete"])
    reasons = []
    if any(case["liveness_seconds"] > LIVENESS_LIMIT_S for case in cases):
        reasons.append("liveness_limit_exceeded")
    if any(
        case["device_stop_seconds"] > DEVICE_STOP_LIMIT_S
        for case in cases
        if case["kind"] not in exempt
    ):
        reasons.append("device_stop_limit_exceeded")
    if any(
        not case[key]
        for case in cases
        for key in (
            "source_session_changed",
            "quiet_without_replay",
            "recovery_turn_completed",
        )
    ):
        reasons.append("recovery_failure")
    if any(case["unexpected_exits"] for case in cases):
        reasons.append("unexpected_worker_exit")
    if any(value != 0 for value in report["cleanup"].values()):
        reasons.append("cleanup_failure")
    return _result("faults", "fail" if reasons else "pass", reasons)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(CHECK_ERROR)


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _Parser(add_help=False, allow_abbrev=False)
        parser.add_argument("--report", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--baseline", type=Path)
        parser.add_argument("--limits", type=Path)
        args = parser.parse_args(argv)
        report, _ = load_report(args.report)
        if report["scenario"] == "steady":
            if args.baseline is None or args.limits is None:
                raise ValueError(CHECK_ERROR)
            baseline, baseline_sha256 = load_report(args.baseline)
            limits, limits_sha256 = load_frozen_limits(args.limits)
            result = evaluate_steady(
                report,
                baseline,
                limits,
                baseline_sha256=baseline_sha256,
                limits_sha256=limits_sha256,
            )
        elif report["scenario"] == "faults":
            if args.baseline is not None or args.limits is not None:
                raise ValueError(CHECK_ERROR)
            result = evaluate_faults(report)
        else:
            result = _result("baseline", "incomplete", ["baseline_not_acceptance"])
        write_report(args.output, result)
    except (ValueError, OSError, argparse.ArgumentError):
        print(CHECK_ERROR, file=sys.stderr)
        return 2
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
