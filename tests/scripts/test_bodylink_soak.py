"""CLI contract, exclusive artifact I/O, and exact report shape."""

import dataclasses
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from companion.bodylink.transport import TransportStats
from scripts import bodylink_soak as cli
from scripts import bodylink_soak_support as soak


@pytest.fixture(scope="module")
def report():
    if sys.platform != "darwin":
        pytest.skip("macOS-only soak process supervisor")
    return soak.run_soak(soak.RunConfig(1.1, 30, "synthetic answer"))


def test_report_has_exact_schema_and_unavailable_values(report):
    assert set(report) == {
        "schema_version",
        "scenario",
        "requested_seconds",
        "elapsed_seconds",
        "started_unix_s",
        "run_started_monotonic_s",
        "finished_unix_s",
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
    assert report["schema_version"] == 1
    assert report["scenario"] == "steady"
    assert report["limits_sha256"] == "unavailable"
    assert report["warmup_seconds"] == 30.0
    assert report["resource_sample_interval_seconds"] == 5.0
    assert report["faults"] == []
    assert report["unavailable"] == list(soak.UNAVAILABLE_METRICS)
    assert report["run_complete"]
    assert 1.1 <= report["elapsed_seconds"] < 6.1
    assert report["finished_unix_s"] >= report["started_unix_s"]
    assert set(report["workers"]) == {"brain", "body"}
    stats_keys = {field.name for field in dataclasses.fields(TransportStats)}
    kinds = {
        "ready",
        "session_started",
        "session_ended",
        "recording_ready",
        "playback_started",
        "playback_completed",
        "playback_stopped",
        "turn_completed",
        "turn_failed",
        "snapshot",
        "resource_sample",
        "worker_error",
        "stopped",
    }
    for event in report["events"]:
        assert set(event) == {"role", "incarnation", "kind", "phase", "at_s", "values"}
        assert event["kind"] in kinds
        assert event["phase"] in ("startup", "run", "cleanup")
        assert all(
            type(value) in (int, float, bool) or value == "unavailable"
            for value in event["values"].values()
        )
        if event["kind"] == "snapshot":
            assert set(event["values"]) == stats_keys
        elif event["kind"] == "resource_sample":
            assert set(event["values"]) == {"rss_bytes", "thread_count", "fd_count", "cpu_seconds"}
            assert all(
                type(value) in (int, float) or value == "unavailable"
                for value in event["values"].values()
            )
    for role in ("brain", "body"):
        samples = report["resources"][role]["1"]
        assert len(samples) == 2
        assert all(
            set(sample)
            == {
                "at_s",
                "rss_bytes",
                "thread_count",
                "fd_count",
                "cpu_seconds",
            }
            for sample in samples
        )
        assert report["resource_growth"][role]["1"] == dict(
            rss_growth_bytes="unavailable",
            thread_growth="unavailable",
            fd_growth="unavailable",
            cpu_seconds="unavailable",
        )
        snapshots = report["workers"][role]["1"]
        assert set(snapshots) == {"start", "run_end", "after_close"}
        assert all(set(snapshot) == stats_keys for snapshot in snapshots.values())
        assert (
            sum(event["kind"] == "snapshot" and event["role"] == role for event in report["events"])
            >= 4
        )


def test_report_monotonic_origin_matches_growth_input(report):
    origin = report["run_started_monotonic_s"]
    assert type(origin) is float
    for role, incarnations in report["resources"].items():
        for incarnation, samples in incarnations.items():
            assert soak.resource_growth(
                [soak.ResourceSample(**sample) for sample in samples],
                run_started_monotonic_s=origin,
                warmup_seconds=report["warmup_seconds"],
            ) == report["resource_growth"][role][incarnation]


def test_existing_output_is_not_overwritten(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        soak.write_report(path, {})
    assert path.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "relative",
    [
        "private_data/out.json",
        "runtime_data/out.json",
        "test-results/out.json",
    ],
)
def test_artifact_path_boundary(tmp_path, relative):
    path = tmp_path / relative
    with pytest.raises(ValueError, match=r"^invalid soak artifact path$"):
        soak.write_report(path, {})
    assert not path.exists()


@pytest.mark.parametrize(
    "relative",
    ["Library/Application Support/x.json", ".ssh/x.json", ".config/x.json", ".local/share/x.json"],
)
def test_user_application_artifacts_are_rejected(relative):
    with pytest.raises(ValueError, match=r"^invalid soak artifact path$"):
        soak.write_report(Path.home() / relative, {})


def _args(tmp_path):
    return [
        "--duration-seconds",
        "0.4",
        "--fixed-reply",
        "answer",
        "--output",
        str(tmp_path / "out.json"),
    ]


def test_cli_returns_0_1_2_without_two_hour_claim(tmp_path, monkeypatch, capsys, report):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(cli, "run_soak", lambda *args, **kwargs: report)
    assert cli.main(_args(tmp_path)) == 0
    assert json.loads((tmp_path / "out.json").read_text())["run_complete"]
    assert cli.main(_args(tmp_path)) == 1
    monkeypatch.setattr(cli, "run_soak", lambda *args, **kwargs: {"run_complete": False})
    args = _args(tmp_path)
    args[-1] = str(tmp_path / "failed.json")
    assert cli.main(args) == 1
    assert cli.main([]) == 2
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [cli.SOAK_ERROR, cli.SOAK_ERROR, cli.SOAK_ARGUMENT_ERROR]
    assert not any(key in json.dumps(report) for key in ("readiness", "two_hours_passed"))


@pytest.mark.parametrize(
    "option,value",
    [
        ("--duration-seconds", "nan"),
        ("--duration-seconds", "inf"),
        ("--duration-seconds", "0"),
        ("--duration-seconds", "-1"),
        ("--turn-interval-seconds", "-inf"),
        ("--turn-interval-seconds", "0"),
        ("--fixed-reply", "  "),
        ("--scenario", "other"),
        ("--host", "localhost"),
        ("--cert-dir", "secret"),
        ("--settings", "secret"),
        ("--provider", "secret"),
        ("--debug-log", "secret"),
        ("--duration", "1"),
    ],
)
def test_invalid_arguments_do_not_start_run(tmp_path, monkeypatch, capsys, option, value):
    def forbidden(*args, **kwargs):
        pytest.fail("argument error created runtime")

    monkeypatch.setattr(cli, "run_soak", forbidden)
    assert cli.main([*_args(tmp_path), option, value]) == 2
    assert capsys.readouterr().err == cli.SOAK_ARGUMENT_ERROR + "\n"


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_other_platforms_fail_before_tempdir_or_worker(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported platform touched runtime")

    monkeypatch.setattr(soak.tempfile, "TemporaryDirectory", forbidden)
    monkeypatch.setattr(soak, "start_worker", forbidden)
    assert cli.main(_args(tmp_path)) == 2
    with pytest.raises(ValueError):
        soak.run_soak(soak.RunConfig(1, 30, "answer"))


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only soak process supervisor")
def test_cli_real_script_entry(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/bodylink_soak.py", *_args(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert json.loads((tmp_path / "out.json").read_text())["turns"]["completed"] == 1


def test_cli_interrupt_handler_only_sets_event(tmp_path, monkeypatch):
    # Validate the supported-platform branch with a mocked runtime.
    monkeypatch.setattr(sys, "platform", "darwin")
    def interrupted(config, *, stop_requested):
        signal.raise_signal(signal.SIGTERM)
        assert stop_requested.is_set()
        return {"run_complete": False, "interrupted": True}

    monkeypatch.setattr(cli, "run_soak", interrupted)
    assert cli.main(_args(tmp_path)) == 1
    assert json.loads((tmp_path / "out.json").read_text())["interrupted"]


def test_owned_worker_stop_is_bounded_and_only_reaps_its_process():
    class Process:
        def __init__(self):
            self.alive = True
            self.calls = []

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.calls.append(("join", timeout))

        def terminate(self):
            self.calls.append(("terminate",))

        def kill(self):
            self.calls.append(("kill",))
            self.alive = False

    class Channel:
        def __init__(self):
            self.commands = []

        def send(self, command):
            self.commands.append(command)

    process, channel = Process(), Channel()
    worker = soak.OwnedWorker(process, channel)
    worker.stop()
    worker.stop()
    assert worker.forced
    assert channel.commands == [soak.WorkerCommand("stop")]
    assert process.calls == [
        ("join", 5.0),
        ("terminate",),
        ("join", 5.0),
        ("kill",),
        ("join", None),
    ]


@pytest.mark.parametrize(
    "instance",
    [
        soak.RunConfig(1, 30, "answer"),
        soak.WorkerSpec("brain", 1, 0, Path("cert"), Path("run"), "answer"),
        soak.WorkerCommand("snapshot"),
        soak.WorkerEvent("brain", 1, "snapshot", "run", 0.0, {}),
    ],
)
def test_public_value_types_are_frozen_dataclasses(instance):
    field = dataclasses.fields(instance)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field.name, None)


def test_output_parent_must_exist(tmp_path, capsys):
    args = _args(tmp_path)
    args[-1] = str(tmp_path / "missing" / "out.json")
    assert cli.main(args) == 2
    assert capsys.readouterr().err == cli.SOAK_ARGUMENT_ERROR + "\n"


def test_parent_io_failure_uses_fixed_argument_error(tmp_path, monkeypatch, capsys):
    def fail(path):
        raise PermissionError("private path error canary")

    monkeypatch.setattr(Path, "is_dir", fail)
    assert cli.main(_args(tmp_path)) == 2
    assert capsys.readouterr().err == cli.SOAK_ARGUMENT_ERROR + "\n"


def test_faults_choice_reaches_run_config(tmp_path, monkeypatch):
    # Validate the supported-platform branch with a mocked runtime.
    monkeypatch.setattr(sys, "platform", "darwin")
    seen = []

    def run(config, **kwargs):
        seen.append(config)
        return {"run_complete": False}

    monkeypatch.setattr(cli, "run_soak", run)
    assert cli.main([*_args(tmp_path), "--scenario", "faults"]) == 1
    assert seen == [soak.RunConfig(0.4, 30, "answer", "faults")]


def test_default_duration_is_two_hours_without_running_it(tmp_path, monkeypatch):
    # Validate the supported-platform branch with a mocked runtime.
    monkeypatch.setattr(sys, "platform", "darwin")
    seen = []

    def run(config, **kwargs):
        seen.append(config)
        return {"run_complete": True}

    monkeypatch.setattr(cli, "run_soak", run)
    args = _args(tmp_path)[2:]
    assert cli.main(args) == 0
    assert seen[0].duration_seconds == cli.STEADY_DURATION_S == 7200.0
    assert seen[0].warmup_seconds == 30.0
    assert seen[0].limits_path is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--warmup-seconds", "-1"],
        ["--warmup-seconds", "nan"],
        ["--warmup-seconds", "inf"],
        ["--scenario", "baseline"],
        ["--scenario", "baseline", "--duration-seconds", "44.99"],
        ["--scenario", "baseline", "--limits", "limits.json"],
        ["--scenario", "faults", "--limits", "limits.json"],
    ],
)
def test_resource_argument_errors_before_runtime(tmp_path, monkeypatch, extra):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid arguments started runtime")

    monkeypatch.setattr(cli, "run_soak", forbidden)
    assert cli.main([*_args(tmp_path), *extra]) == 2


def test_baseline_duration_must_be_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_soak", lambda *a, **kw: pytest.fail("started baseline"))
    assert cli.main([*_args(tmp_path)[2:], "--scenario", "baseline"]) == 2


def test_baseline_collects_without_generating_limits(monkeypatch):
    # Validate the supported-platform branch with a mocked runtime.
    monkeypatch.setattr(sys, "platform", "darwin")
    from scripts.bodylink_soak_resources import ResourceSample

    class Clock:
        now = 100.0

        def monotonic(self):
            return self.now

        def wait(self, timeout):
            self.now += timeout
            return False

        def is_set(self):
            return False

    clock = Clock()
    monkeypatch.setattr(soak.time, "monotonic", clock.monotonic)

    def start(self):
        self.origin = clock.now
        for role in ("brain", "body"):
            self.report["resources"][role]["1"] = [
                dataclasses.asdict(
                    ResourceSample(
                        100 + i * 5,
                        100 + i,
                        3,
                        5,
                        i * 0.1,
                    )
                )
                for i in range(5)
            ]

    def begin(self, **kwargs):
        self.report["turns"]["started"] += 1
        self.report["turns"]["completed"] += 1

    def finish(self):
        self.report["elapsed_seconds"] = clock.now - self.origin

    monkeypatch.setattr(soak.SoakSupervisor, "start", start)
    monkeypatch.setattr(soak.SoakSupervisor, "begin_turn", begin)
    monkeypatch.setattr(soak.SoakSupervisor, "finish_observation", finish)
    monkeypatch.setattr(soak.SoakSupervisor, "run_case", lambda *a: pytest.fail("injected fault"))
    report = soak.run_soak(
        soak.RunConfig(20, 10, "answer", "baseline", warmup_seconds=5), stop_requested=clock
    )
    assert report["scenario"] == "baseline"
    assert report["turns"] == dict(planned=2, started=2, completed=2, failed=0, aborted=0, missed=0)
    assert report["elapsed_seconds"] == pytest.approx(20, abs=0.01)
    assert report["limits_sha256"] == "unavailable"
    assert report["faults"] == []
    assert report["resource_growth"]["brain"]["1"] == dict(
        rss_growth_bytes=3,
        thread_growth=0,
        fd_growth=0,
        cpu_seconds=pytest.approx(0.3),
    )
    assert not any(
        key in json.dumps(report) for key in ("two_hours_passed", "readiness", "limits_path")
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only soak process supervisor")
def test_limits_digest_is_bound_before_workers_start(tmp_path, monkeypatch):
    import hashlib

    from test_bodylink_soak_resources import limits_document

    path = tmp_path / "limits.json"
    original = json.dumps(limits_document()).encode()
    path.write_bytes(original)
    reader = soak.load_frozen_limits
    reads = []

    def read(path):
        reads.append(path)
        return reader(path)

    monkeypatch.setattr(soak, "load_frozen_limits", read)
    launch = soak.start_worker
    generate = soak.generate_pairing
    tempdir = soak.tempfile.TemporaryDirectory

    def check():
        assert reads == [path]

    def directory(*args, **kwargs):
        check()
        return tempdir(*args, **kwargs)

    def pairing(*args, **kwargs):
        check()
        return generate(*args, **kwargs)

    def start(spec):
        check()
        path.write_text("modified during run")
        return launch(spec)

    monkeypatch.setattr(soak.tempfile, "TemporaryDirectory", directory)
    monkeypatch.setattr(soak, "generate_pairing", pairing)
    monkeypatch.setattr(soak, "start_worker", start)
    report = soak.run_soak(soak.RunConfig(0.4, 30, "answer", limits_path=path))
    assert report["run_complete"]
    assert reads == [path]
    assert report["limits_sha256"] == hashlib.sha256(original).hexdigest()
    artifact = tmp_path / "report.json"
    soak.write_report(artifact, report)
    assert json.loads(artifact.read_bytes())["limits_sha256"] == report["limits_sha256"]
    assert path.read_text() == "modified during run"


def test_limits_warmup_mismatch_is_argument_error_before_workspace(tmp_path, monkeypatch):
    from test_bodylink_soak_resources import limits_document

    path = tmp_path / "limits.json"
    path.write_text(json.dumps(limits_document()))
    monkeypatch.setattr(
        soak.tempfile, "TemporaryDirectory", lambda *a, **kw: pytest.fail("created workspace")
    )
    assert cli.main([*_args(tmp_path), "--warmup-seconds", "0", "--limits", str(path)]) == 2


def test_all_real_run_tests_use_explicit_short_duration():
    import ast

    paths = [
        *Path("tests/scripts").glob("test_bodylink_soak*.py"),
        Path("tests/integration/test_bodylink_soak_processes.py"),
    ]
    for path in paths:
        tree = ast.parse(path.read_text())
        for function in tree.body:
            if not isinstance(function, ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "RunConfig"
                ):
                    duration = (
                        node.args[0]
                        if node.args
                        else next(kw.value for kw in node.keywords if kw.arg == "duration_seconds")
                    )
                    assert isinstance(duration, ast.Constant), (path, node.lineno)
                    ceiling = (
                        50
                        if function.name
                        in {
                            "test_baseline_collects_without_generating_limits",
                            "test_growth_uses_peak_after_warmup_and_same_incarnation",
                        }
                        else 10
                    )
                    assert 0 < duration.value <= ceiling, (path, node.lineno)
        source = path.read_text()
        if "subprocess.run(" in source:
            assert '"--duration-seconds"' in source


@pytest.mark.parametrize(
    "scenario,duration,warmup", [("baseline", 15, 0), ("baseline", 45, 30), ("steady", 0.4, 0)]
)
def test_valid_resource_cli_config_without_real_baseline(
    tmp_path, monkeypatch, scenario, duration, warmup
):
    # Validate the supported-platform branch with a mocked runtime.
    monkeypatch.setattr(sys, "platform", "darwin")
    seen = []

    def run(config, **kwargs):
        seen.append(config)
        return {"run_complete": True}

    monkeypatch.setattr(cli, "run_soak", run)
    assert (
        cli.main(
            [
                *_args(tmp_path),
                "--scenario",
                scenario,
                "--duration-seconds",
                str(duration),
                "--warmup-seconds",
                str(warmup),
            ]
        )
        == 0
    )
    assert seen[0].scenario == scenario
    assert seen[0].duration_seconds == duration
    assert seen[0].warmup_seconds == warmup
