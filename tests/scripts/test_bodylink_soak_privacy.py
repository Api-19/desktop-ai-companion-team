"""Privacy checks at actual WAV, worker startup, wire and artifact I/O boundaries."""

import base64
import dataclasses
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

from companion.bodylink.pairing import generate_pairing
from companion.contracts.body import PROTOCOL_VERSION, Fault
from companion.speech.controller_models import SpeechState
from scripts import bodylink_soak as cli
from scripts import bodylink_soak_check as checker
from scripts import bodylink_soak_faults as faults
from scripts import bodylink_soak_support as soak

REPLY = "synthetic-reply-secret-canary"
ERROR = "synthetic-exception-secret-canary"
FAULT = "synthetic-fault-detail-secret-canary"


def _until(predicate, brain):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        brain.step()
        time.sleep(0.01)
    assert predicate()


def _rig(root, resources):
    pairing = generate_pairing(root / "pairing", soak.BODY_ID)
    for role in ("brain", "body"):
        (root / role).mkdir()
    brain_spec = soak.WorkerSpec("brain", 1, 0, pairing.brain_dir, root / "brain", REPLY)
    brain_events = soak._Events(brain_spec)
    brain = soak._Brain(brain_spec, brain_events, resources)
    body_spec = soak.WorkerSpec("body", 1, brain.link.port, pairing.body_dir, root / "body", REPLY)
    body_events = soak._Events(body_spec)
    body = soak._Body(body_spec, body_events, resources)
    _until(brain.link.mute_gate.microphone_enabled, brain)
    return brain, body, brain_events, body_events


@pytest.mark.parametrize("recognizer_fails", [False, True])
def test_synthetic_wav_removed_after_success_and_failure(tmp_path, monkeypatch, recognizer_fails):
    read = soak.read_owned_pcm_wav
    reads = []

    def inspect(audio, *, expected_sample_rate_hz):
        wav = read(audio, expected_sample_rate_hz=expected_sample_rate_hz)
        assert wav.samples_int16 == tuple([1000, -1000] * 2400)
        files = list((tmp_path / "brain").glob("*.wav"))
        assert len(files) == 1
        reads.append(files[0])
        if recognizer_fails:
            raise ValueError(ERROR)
        return wav

    monkeypatch.setattr(soak, "read_owned_pcm_wav", inspect)
    with ExitStack() as resources:
        brain, body, _, _ = _rig(tmp_path, resources)
        body.core.button_pressed()
        _until(lambda: brain.controller.state is SpeechState.RECORDING, brain)
        observed = []

        def recording_ready():
            while not body.events.pending.empty():
                observed.append(body.events.pending.get_nowait())
            return any(event.kind == "recording_ready" for event in observed)

        _until(recording_ready, brain)
        body.core.button_pressed()
        _until(lambda: brain.completed + brain.failed == 1, brain)
        assert brain.failed == int(recognizer_fails)
        assert len(reads) == 1
        assert all(not path.exists() for path in reads)
        assert list((tmp_path / "brain").iterdir()) == []
    assert not list(tmp_path.rglob("*.wav"))


def _guarded_worker(spec, channel):
    """Runs inside each real spawned child, before actual runtime construction."""
    from companion.runtime import settings, speech_chat_mac
    from companion.speech import local_audio_devices, runtime_assembly

    def denied(*args, **kwargs):
        raise AssertionError("forbidden startup I/O")

    settings.load_settings = denied
    speech_chat_mac._build_orchestrator = denied
    runtime_assembly.assemble_body_speech_runtime = denied
    local_audio_devices.load_sounddevice_runtime = denied
    cert_dir = os.path.abspath(spec.cert_dir)
    run_root = os.path.abspath(spec.run_root)

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if set(path.parts) & {"private_data", "runtime_data", "test-results", ".ssh"}:
                denied()
            absolute = os.path.abspath(path)
            if absolute.startswith(str(spec.cert_dir.parent)):
                assert absolute.startswith(cert_dir + os.sep)
            if absolute.startswith(str(spec.run_root.parent)) and "/pairing/" not in absolute:
                assert absolute.startswith(run_root + os.sep)
            if path.suffix in {".onnx", ".env", ".toml", ".yaml", ".yml"}:
                denied()
        elif event == "socket.connect":
            assert args[1][0] == "127.0.0.1"
        elif event == "import" and args[0].split(".")[0] in {"sounddevice", "sherpa_onnx"}:
            denied()

    sys.addaudithook(audit)
    soak.worker_main(spec, channel)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only soak process supervisor")
def test_workers_never_load_models_devices_settings_or_provider(monkeypatch):
    monkeypatch.setattr(soak, "worker_main", _guarded_worker)
    report = soak.run_soak(soak.RunConfig(0.4, 30, REPLY))
    assert report["run_complete"]
    assert report["turns"]["completed"] == 1
    supervisor = soak.SoakSupervisor(soak.RunConfig(1, 30, REPLY, "faults"))
    try:
        supervisor.start()
        assert supervisor.run_case(faults.FAULT_CASES[0])["complete"]
    finally:
        supervisor.close()


def _error_worker(spec, channel):
    def fail(*args):
        raise ValueError(ERROR)

    soak._Brain = fail
    soak.worker_main(spec, channel)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only soak process supervisor")
def test_events_report_and_console_contain_no_payload(tmp_path, monkeypatch, capfd):
    report = soak.run_soak(soak.RunConfig(0.4, 30, REPLY))
    assert report["run_complete"]
    resource_events = [e for e in report["events"] if e["kind"] == "resource_sample"]
    assert {e["role"] for e in resource_events} == {"brain", "body"}
    assert all(
        set(e["values"])
        == {
            "rss_bytes",
            "thread_count",
            "fd_count",
            "cpu_seconds",
        }
        for e in resource_events
    )
    assert all(
        type(value) in (int, float) or value == "unavailable"
        for e in resource_events
        for value in e["values"].values()
    )
    with ExitStack() as resources:
        rig_root = tmp_path / "rig"
        rig_root.mkdir()
        brain, body, brain_events, body_events = _rig(rig_root, resources)
        body.send_message(
            Fault(
                protocol_version=PROTOCOL_VERSION,
                seq=0,
                session_id=brain.link.current_session_id,
                kind="speaker_unavailable",
                detail=FAULT,
            )
        )
    extra_events = []
    for events in (brain_events, body_events):
        while not events.pending.empty():
            extra_events.append(dataclasses.asdict(events.pending.get_nowait()))
    with monkeypatch.context() as patch:
        patch.setattr(soak, "worker_main", _error_worker)
        failed = soak.run_soak(soak.RunConfig(0.1, 30, REPLY))
        assert not failed["run_complete"]
        errors = [event for event in failed["events"] if event["kind"] == "worker_error"]
        assert errors and all(event["values"] == {"code": 1} for event in errors)
    supervisor = soak.SoakSupervisor(soak.RunConfig(1, 30, REPLY, "faults"))
    try:
        supervisor.start()
        assert supervisor.run_case(faults.FAULT_CASES[10])["complete"]
    finally:
        supervisor.close()
    with monkeypatch.context() as patch:
        patch.setattr(soak, "worker_main", _error_worker)
        fault_failed = soak.run_soak(soak.RunConfig(1, 30, REPLY, "faults"))
        assert any(e["kind"] == "worker_error" for e in fault_failed["events"])
    with monkeypatch.context() as patch:

        def fail_after_event(self, case):
            raise ValueError(ERROR)

        patch.setattr(soak.SoakSupervisor, "run_case", fail_after_event)
        fault_exception = soak.run_soak(soak.RunConfig(1, 30, REPLY, "faults"))
        assert not fault_exception["run_complete"]
    path = tmp_path / "evidence.json"
    soak.write_report(path, report)
    faults_path = tmp_path / "faults.json"
    soak.write_report(
        faults_path,
        dict(
            recovered=supervisor.report,
            worker_error=fault_failed,
            exception=fault_exception,
        ),
    )
    assert (
        cli.main(
            [
                "--duration-seconds",
                "nan",
                "--fixed-reply",
                REPLY,
                "--output",
                str(tmp_path / "never.json"),
            ]
        )
        == 2
    )
    captured = capfd.readouterr()
    output = (
        path.read_text(encoding="utf-8")
        + faults_path.read_text(encoding="utf-8")
        + json.dumps(failed)
        + json.dumps(extra_events)
        + captured.out
        + captured.err
    )
    for payload in (
        REPLY,
        ERROR,
        FAULT,
        soak.SYNTHETIC_TRANSCRIPT,
        repr(soak.INPUT_PCM),
        repr(soak.OUTPUT_PCM),
        soak.INPUT_PCM.hex(),
        base64.b64encode(soak.OUTPUT_PCM).decode(),
        str(tmp_path),
        "--fixed-reply",
        "Traceback",
        "ValueError",
    ):
        assert payload not in output


@pytest.mark.parametrize(
    "relative",
    [
        "private_data/limits.json",
        "runtime_data/limits.json",
        "test-results/limits.json",
        "Library/limits.json",
        ".ssh/limits.json",
        ".config/limits.json",
        ".local/share/limits.json",
    ],
)
@pytest.mark.parametrize("entrypoint", ["reader", "checker"])
def test_limits_reader_rejects_private_paths_before_open(
    tmp_path, monkeypatch, capsys, relative, entrypoint
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("opened forbidden limits path")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    if entrypoint == "reader":
        with pytest.raises(ValueError, match=r"^invalid soak artifact path$"):
            soak.load_frozen_limits(tmp_path / relative)
    else:
        from test_bodylink_soak_check import synthetic_report

        monkeypatch.setattr(checker, "load_report", lambda path: (synthetic_report(), "b" * 64))
        assert (
            checker.main(
                [
                    "--report",
                    "synthetic-report.json",
                    "--baseline",
                    "synthetic-baseline.json",
                    "--limits",
                    str(tmp_path / relative),
                    "--output",
                    str(tmp_path / "never.json"),
                ]
            )
            == 2
        )
        assert capsys.readouterr().err == checker.CHECK_ERROR + "\n"


def test_threshold_artifacts_and_errors_contain_no_payload(tmp_path, capsys):
    from test_bodylink_soak_resources import limits_document

    canary = "synthetic-threshold-body-secret"
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(limits_document() | {"body_text": canary}))
    with pytest.raises(ValueError, match=r"^invalid frozen soak limits$") as error:
        soak.load_frozen_limits(path)
    assert (
        cli.main(
            [
                "--duration-seconds",
                "0.4",
                "--fixed-reply",
                REPLY,
                "--limits",
                str(path),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "report.json").exists()
    output = str(error.value) + str(capsys.readouterr())
    assert canary not in output and str(path) not in output


@pytest.mark.parametrize(
    "key,payload",
    [
        ("body_text", "synthetic-checker-body-canary"),
        ("pcm", base64.b64encode(soak.INPUT_PCM).decode()),
        ("source_path", "/synthetic-private-path-canary/chat.json"),
    ],
)
def test_checker_rejects_payload_keys_without_echo(tmp_path, capsys, key, payload):
    from test_bodylink_soak_check import synthetic_report

    path = tmp_path / "input.json"
    soak.write_report(path, synthetic_report("faults") | {key: payload})
    output = tmp_path / "verdict.json"
    assert checker.main(["--report", str(path), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == checker.CHECK_ERROR + "\n"
    assert payload not in captured.err and str(path) not in captured.err
    assert not output.exists()


def test_checker_output_contains_only_verdict_codes(tmp_path, capsys):
    from test_bodylink_soak_check import synthetic_report, write_inputs

    report = synthetic_report()
    report["workers"]["body"]["1"]["run_end"]["tx_application_bytes"] = 987654321
    args = write_inputs(tmp_path, report)
    assert checker.main(args) == 0
    result = (tmp_path / "verdict.json").read_text()
    assert json.loads(result) == dict(
        schema_version=1,
        scenario="steady",
        verdict="pass",
        two_hours_passed=True,
        faults_passed=False,
        reasons=[],
    )
    assert "987654321" not in result and str(tmp_path) not in result
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_checker_report_reader_rejects_private_paths_before_open(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("opened forbidden report path")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError, match=r"^invalid soak artifact path$"):
        checker.load_report(tmp_path / "private_data" / "report.json")
