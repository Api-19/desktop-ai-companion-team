"""Real spawned workers, loopback mutual TLS, and bounded synthetic runs."""

import copy
import hashlib
import json
import os
import signal
import sys
import threading

import pytest

from scripts import bodylink_soak as soak_cli
from scripts import bodylink_soak_check as checker
from scripts import bodylink_soak_faults as faults
from scripts import bodylink_soak_support as soak

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="macOS-only soak process supervisor"
)


def _thread_checked_worker(spec, channel):
    soak.worker_main(spec, channel)
    assert not any(thread.name == "soak-resources" for thread in threading.enumerate())


def test_two_workers_complete_fixed_reply_turn(monkeypatch):
    monkeypatch.setattr(soak, "worker_main", _thread_checked_worker)
    report = soak.run_soak(soak.RunConfig(0.6, 30, "synthetic fixed answer"))
    assert report["run_complete"]
    assert report["turns"] == dict(planned=1, started=1, completed=1, failed=0, aborted=0, missed=0)
    pids = {
        event["values"]["pid"]
        for event in report["events"]
        if event["kind"] == "ready" and "pid" in event["values"]
    }
    assert len(pids) == 2
    assert os.getpid() not in pids
    for role in ("brain", "body"):
        samples = [
            event
            for event in report["events"]
            if event["role"] == role and event["kind"] == "resource_sample"
        ]
        assert len(samples) == 2
        assert samples[0]["phase"] == samples[-1]["phase"] == "run"
        assert report["resources"][role]["1"] == [
            dict(at_s=sample["at_s"], **sample["values"]) for sample in samples
        ]
        snapshots = report["workers"][role]["1"]
        assert all(isinstance(snapshots[key], dict) for key in ("start", "run_end", "after_close"))
        assert snapshots["after_close"]["sessions_started"] == 1
        assert snapshots["after_close"]["tx_audio_frames"] > 0
        assert snapshots["after_close"]["rx_audio_frames"] > 0
        assert (
            sum(
                snapshots["after_close"][f"sessions_ended_{reason}"]
                for reason in (
                    "closed",
                    "peer_closed",
                    "liveness_timeout",
                    "protocol_error",
                    "send_timeout",
                )
            )
            == 1
        )
    assert report["cleanup"] == dict(live_children=0, residual_files=0)


def test_short_real_process_run_cannot_pass_two_hour_checker(tmp_path):
    report_path = tmp_path / "short.json"
    assert (
        soak_cli.main(
            [
                "--duration-seconds",
                "0.4",
                "--fixed-reply",
                "synthetic short answer",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text())
    assert report["run_complete"]
    # Only the smoke is real; baseline and limits here are synthetic checker inputs.
    baseline = copy.deepcopy(report)
    baseline["scenario"] = "baseline"
    baseline["finished_unix_s"] = report["started_unix_s"]
    baseline["resource_growth"] = {
        role: {"1": dict(rss_growth_bytes=0, thread_growth=0, fd_growth=0, cpu_seconds=0.0)}
        for role in ("brain", "body")
    }
    baseline_path = tmp_path / "synthetic-baseline.json"
    soak.write_report(baseline_path, baseline)
    limits_path = tmp_path / "synthetic-limits.json"
    soak.write_report(
        limits_path,
        dict(
            schema_version=1,
            source_baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            warmup_seconds=report["warmup_seconds"],
            resource_sample_interval_seconds=5.0,
            **{
                role: dict(rss_growth_bytes=0, thread_growth=0, fd_growth=0)
                for role in ("brain", "body")
            },
        ),
    )
    output = tmp_path / "verdict.json"
    assert (
        checker.main(
            [
                "--report",
                str(report_path),
                "--baseline",
                str(baseline_path),
                "--limits",
                str(limits_path),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    verdict = json.loads(output.read_text())
    assert verdict["verdict"] == "incomplete"
    assert "duration_insufficient" in verdict["reasons"]
    assert verdict["two_hours_passed"] is False


def test_steady_run_does_not_overlap_or_replay_slots():
    report = soak.run_soak(soak.RunConfig(0.12, 0.02, "answer"))
    turns = report["turns"]
    assert turns["planned"] == 6
    assert turns["started"] == turns["completed"] == 1
    assert turns["missed"] == 5
    assert not report["run_complete"]


def test_unexpected_worker_exit_fails_run(monkeypatch):
    original = soak.start_worker

    def start(spec):
        worker = original(spec)
        if spec.role == "body":
            worker.process.terminate()
        return worker

    monkeypatch.setattr(soak, "start_worker", start)
    report = soak.run_soak(soak.RunConfig(0.2, 30, "answer"))
    assert not report["run_complete"]
    assert report["cleanup"] == dict(live_children=0, residual_files=0)


def test_interrupted_run_is_incomplete(monkeypatch):
    stop = threading.Event()
    original = soak.start_worker

    def start(spec):
        worker = original(spec)
        if spec.role == "body":
            stop.set()
        return worker

    monkeypatch.setattr(soak, "start_worker", start)
    report = soak.run_soak(soak.RunConfig(0.2, 30, "answer"), stop_requested=stop)
    assert report["interrupted"]
    assert not report["run_complete"]
    assert report["turns"]["planned"] == sum(
        report["turns"][key] for key in ("completed", "failed", "aborted", "missed")
    )


def test_shutdown_reaps_workers_and_removes_workspace(monkeypatch):
    workers = []
    original = soak.start_worker

    def start(spec):
        worker = original(spec)
        workers.append((spec, worker))
        return worker

    monkeypatch.setattr(soak, "start_worker", start)
    report = soak.run_soak(soak.RunConfig(0.2, 30, "answer"))
    for spec, worker in workers:
        assert not worker.process.is_alive()
        assert not spec.run_root.parent.exists()
        worker.stop()
    assert report["cleanup"] == dict(live_children=0, residual_files=0)


def _failed_recognizer_worker(spec, channel):
    def fail(self, audio):
        raise ValueError("synthetic recognizer failure")

    soak._Recognizer.recognize = fail
    soak.worker_main(spec, channel)


def test_turn_failure_fails_run_without_restarting(monkeypatch):
    monkeypatch.setattr(soak, "worker_main", _failed_recognizer_worker)
    report = soak.run_soak(soak.RunConfig(0.4, 30, "answer"))
    assert not report["run_complete"]
    assert report["turns"]["failed"] == report["turns"]["started"] == 1
    assert report["turns"]["completed"] == 0
    assert all(set(incarnations) == {"1"} for incarnations in report["workers"].values())
    assert report["cleanup"] == dict(live_children=0, residual_files=0)


def test_interrupt_during_turn_aborts_and_closes_both_workers(monkeypatch):
    stop = threading.Event()
    original = soak.OwnedWorker.drain_events

    def drain(self):
        events = original(self)
        if any(event.kind == "recording_ready" for event in events):
            stop.set()
        return events

    monkeypatch.setattr(soak.OwnedWorker, "drain_events", drain)
    report = soak.run_soak(soak.RunConfig(0.4, 30, "answer"), stop_requested=stop)
    assert report["interrupted"]
    assert not report["run_complete"]
    assert report["turns"]["aborted"] == 1
    assert all(
        snapshots["1"]["after_close"] != "unavailable" for snapshots in report["workers"].values()
    )
    assert report["cleanup"] == dict(live_children=0, residual_files=0)


def _no_button_worker(spec, channel):
    from companion.contracts.body import ButtonPressed

    original = soak._Body.send_message

    def send(self, message):
        if not isinstance(message, ButtonPressed):
            original(self, message)

    soak._Body.send_message = send
    soak.worker_main(spec, channel)


def test_unfinished_turn_has_bounded_grace_and_is_aborted(monkeypatch):
    monkeypatch.setattr(soak, "worker_main", _no_button_worker)
    report = soak.run_soak(soak.RunConfig(0.05, 30, "answer"))
    assert not report["run_complete"]
    assert report["turns"] == dict(planned=1, started=1, completed=0, failed=0, aborted=1, missed=0)
    assert 5.05 <= report["elapsed_seconds"] < 6
    assert report["cleanup"] == dict(live_children=0, residual_files=0)


def test_steady_run_completes_multiple_separate_slots():
    # This checks scheduling, not subsecond latency on a shared CI runner.
    report = soak.run_soak(soak.RunConfig(4, 3, "answer"))
    assert report["run_complete"], {"turns": report["turns"], "cleanup": report["cleanup"]}
    assert report["turns"] == dict(planned=2, started=2, completed=2, failed=0, aborted=0, missed=0)
    assert all(event["values"].get("duplicates", 0) == 0 for event in report["events"])


# Every matrix item uses real spawned processes and production heartbeat defaults.
@pytest.fixture(scope="module", params=faults.FAULT_CASES, ids=lambda c: f"{c.kind}-{c.phase}")
def fault_result(request):
    supervisor = soak.SoakSupervisor(soak.RunConfig(1, 30, "synthetic fault answer", "faults"))
    try:
        supervisor.start()
        result = supervisor.run_case(request.param)
        yield supervisor, result
    finally:
        supervisor.close()
    assert supervisor.report["cleanup"] == dict(live_children=0, residual_files=0)


def test_each_fault_case_recovers_with_new_button_turn(fault_result):
    supervisor, result = fault_result
    assert result["complete"], result
    assert result["applied"] and result["released"]
    assert result["source_session_changed"]
    assert result["recovery_turn_completed"]
    assert result["unexpected_exits"] == []
    assert 0 <= result["liveness_seconds"] <= faults.LIVENESS_LIMIT_S
    assert result["handshake_recovery_seconds"] <= result["turn_recovery_seconds"]
    assert supervisor.report["turns"]["completed"] == 2
    assert supervisor.report["turns"]["aborted"] == 1
    assert set(result) == {
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
    before = len(supervisor.events)
    with pytest.raises(ValueError, match=r"^fault already applied$"):
        supervisor.run_case(faults.FAULT_CASES[result["ordinal"] - 1])
    assert len(supervisor.events) == before
    if result["phase"] == "recording":
        # Exactly the baseline's two presses, fault press and recovery's two presses.
        assert sum(e.kind == "recording_ready" for e in supervisor.events) == 3
    if "restart" in result["kind"]:
        role = result["kind"].split("_")[0]
        assert set(supervisor.report["workers"][role]) == {"1", "2"}
        assert supervisor.expected_exits == {(role, 1)}
        assert set(supervisor.report["resources"][role]) == {"1", "2"}
        assert all(supervisor.report["resources"][role].values())
    else:
        assert all(set(value) == {"1"} for value in supervisor.report["workers"].values())


def test_one_way_blackhole_expires_receiving_endpoint_with_default_deadline(fault_result):
    supervisor, result = fault_result
    if "blackhole" not in result["kind"]:
        return
    role = "brain" if result["kind"] == "uplink_blackhole" else "body"
    ends = [
        e for e in supervisor.report["events"] if e["role"] == role and e["kind"] == "session_ended"
    ]
    assert ends[0]["values"]["reason"] == 2
    values = ends[0]["values"]
    assert result["liveness_seconds"] == (
        values["last_session_ended_monotonic_s"] - values["last_ended_session_valid_rx_monotonic_s"]
    )
    assert 2.9 <= result["liveness_seconds"] <= 4


def test_responsive_body_stops_active_fake_device_within_one_second(fault_result):
    supervisor, result = fault_result
    if result["kind"] in ("body_restart", "body_pause"):
        return
    assert 0 <= result["device_stop_seconds"] <= 1
    assert result["unavailable_reason"] == "none"
    ends = [
        e
        for e in supervisor.report["events"]
        if e["role"] == "body" and e["kind"] == "session_ended"
    ]
    values = ends[0]["values"]
    key = "input_closed_at_s" if result["phase"] == "recording" else "speaker_returned_at_s"
    assert result["device_stop_seconds"] == max(0, values[key] - values["callback_at_s"])


def test_suspended_or_killed_body_never_claims_unobserved_stop(fault_result):
    _, result = fault_result
    if result["kind"] == "body_restart":
        assert result["device_stop_seconds"] == "unavailable"
        assert result["unavailable_reason"] == "target_not_running"
    elif result["kind"] == "body_pause":
        assert result["unavailable_reason"] == "target_suspended"
        # This value covers only the callback after SIGCONT; the hold interval is unavailable.
        assert result["device_stop_seconds"] == "unavailable"
        supervisor = fault_result[0]
        ended = next(e for e in supervisor.events if e.role == "body" and e.kind == "session_ended")
        key = "input_closed_at_s" if result["phase"] == "recording" else "speaker_returned_at_s"
        assert max(0, ended.values[key] - ended.values["callback_at_s"]) <= 1


def test_recovery_does_not_replay_old_turn_before_button(fault_result):
    _, result = fault_result
    assert result["quiet_without_replay"]


def test_fault_duration_budget_does_not_skip_matrix():
    report = soak.run_soak(soak.RunConfig(0.1, 30, "answer", "faults"))
    assert not report["run_complete"]
    assert not report["faults"] or not report["faults"][-1]["complete"]
    assert len(report["faults"]) < 12
    assert "two_hours_passed" not in report


def test_unexpected_exit_during_fault_is_failure(monkeypatch):
    original = faults.signal_owned

    def signal_and_kill_other(worker, sig, *, owned):
        original(worker, sig, owned=owned)
        if sig == signal.SIGSTOP:
            other = next(w for w in owned if w is not worker and w.process.is_alive())
            other.process.kill()
            other.process.join()

    monkeypatch.setattr(faults, "signal_owned", signal_and_kill_other)
    supervisor = soak.SoakSupervisor(soak.RunConfig(1, 30, "answer", "faults"))
    try:
        supervisor.start()
        result = supervisor.run_case(faults.FAULT_CASES[4])
        assert not result["complete"]
        assert result["unexpected_exits"][0]["kind"] == "unexpected_worker_exit"
        assert supervisor.failed
    finally:
        supervisor.close()


def test_interrupt_resumes_owned_children_and_closes_forwarder(monkeypatch):
    original = faults.signal_owned
    stop = threading.Event()
    signals = []

    def interrupt(worker, sig, *, owned):
        original(worker, sig, owned=owned)
        signals.append(sig)
        if sig == signal.SIGSTOP:
            stop.set()

    monkeypatch.setattr(faults, "signal_owned", interrupt)
    supervisor = soak.SoakSupervisor(soak.RunConfig(1, 30, "answer", "faults"), stop_requested=stop)
    supervisor.start()
    try:
        result = supervisor.run_case(faults.FAULT_CASES[6])
        assert not result["complete"]
    finally:
        supervisor.close()
    assert signals == [signal.SIGSTOP, signal.SIGCONT]
    assert not supervisor.forwarder.thread.is_alive()
    assert all(not w.process.is_alive() for w in supervisor.owned)
    assert supervisor.report["cleanup"] == dict(live_children=0, residual_files=0)


def test_restart_expects_only_killed_incarnation(monkeypatch):
    original = soak.start_worker

    def kill_replacement(spec):
        worker = original(spec)
        if spec.incarnation == 2:
            worker.process.kill()
            worker.process.join()
        return worker

    monkeypatch.setattr(soak, "start_worker", kill_replacement)
    supervisor = soak.SoakSupervisor(soak.RunConfig(1, 30, "answer", "faults"))
    try:
        supervisor.start()
        result = supervisor.run_case(faults.FAULT_CASES[0])
        assert not result["complete"]
        assert result["unexpected_exits"] == [
            dict(
                kind="unexpected_worker_exit",
                role="brain",
                incarnation=2,
                exitcode=-signal.SIGKILL,
            )
        ]
    finally:
        supervisor.close()


def test_unreached_fault_phase_is_incomplete_and_never_injected(monkeypatch):
    original = soak.SoakSupervisor.command
    presses = 0

    def omit_fault_press(self, role, kind):
        nonlocal presses
        if kind == "press":
            presses += 1
            if presses == 3:
                self.deadline = soak.time.monotonic() + 0.05
                return
        original(self, role, kind)

    monkeypatch.setattr(soak.SoakSupervisor, "command", omit_fault_press)
    supervisor = soak.SoakSupervisor(soak.RunConfig(1, 30, "answer", "faults"))
    try:
        supervisor.start()
        result = supervisor.run_case(faults.FAULT_CASES[0])
        assert not result["complete"] and not result["applied"] and not result["released"]
        assert not supervisor.applied_ordinals
        assert not supervisor.expected_exits
        assert result["unavailable_reason"] == "missing_event"
    finally:
        supervisor.close()


def _unavailable_resource_worker(spec, channel):
    from scripts.bodylink_soak_resources import ResourceSample

    soak.sample_self_resources = lambda: ResourceSample(
        soak.time.monotonic(), "unavailable", "unavailable", "unavailable", "unavailable"
    )
    soak.worker_main(spec, channel)


def test_unavailable_resources_survive_ipc_without_failing_tool(monkeypatch):
    monkeypatch.setattr(soak, "worker_main", _unavailable_resource_worker)
    report = soak.run_soak(soak.RunConfig(0.4, 30, "answer", warmup_seconds=0))
    assert report["run_complete"]
    for role in ("brain", "body"):
        assert len(report["resources"][role]["1"]) == 2
        for sample in report["resources"][role]["1"]:
            assert all(
                sample[field] == "unavailable"
                for field in (
                    "rss_bytes",
                    "thread_count",
                    "fd_count",
                    "cpu_seconds",
                )
            )
        assert set(report["resource_growth"][role]["1"].values()) == {"unavailable"}
