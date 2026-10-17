"""Owned synthetic bodylink workers and a payload-free, macOS-only supervisor."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import queue
import re
import struct
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if __package__:
    from .bodylink_soak_resources import (
        RESOURCE_SAMPLE_INTERVAL_S,
        ResourceSample,
        resource_growth,
        sample_self_resources,
    )
else:
    from bodylink_soak_resources import (
        RESOURCE_SAMPLE_INTERVAL_S,
        ResourceSample,
        resource_growth,
        sample_self_resources,
    )

if TYPE_CHECKING:
    from .bodylink_soak_faults import FaultCase

from companion.bodylink.body import BodyClient
from companion.bodylink.body_main import BodyCore, DevelopmentMuteSource
from companion.bodylink.brain_ports import BrainBodyLink
from companion.bodylink.pairing import body_ssl_context, brain_ssl_context, generate_pairing
from companion.bodylink.transport import TransportStats
from companion.contracts import TranscriptConfidenceStatus
from companion.contracts.body import ButtonPressed, Fault, PlayAudio, PlaybackFinished
from companion.runtime.speech_chat_body import FixedReplyDialogue
from companion.speech.controller import SpeechController
from companion.speech.controller_models import RecognitionInput, SpeechState
from companion.speech.local_wav import OwnedWavFile, read_owned_pcm_wav
from companion.speech.pcm_audio import PcmAudio
from companion.speech.recognition_gate import AudioQualityEvidence

TURN_FINISH_GRACE_S = 5.0
WORKER_STOP_TIMEOUT_S = 5.0
STARTUP_TIMEOUT_S = 10.0
BODY_ID = "synthetic-soak-body"
SYNTHETIC_TRANSCRIPT = "synthetic soak transcript"
INPUT_PCM = struct.pack("<4800h", *([1000, -1000] * 2400))
OUTPUT_PCM = struct.pack("<4410h", *([1000, -1000] * 2205))
UNAVAILABLE = "unavailable"
UNAVAILABLE_METRICS = (
    "tls_failures",
    "handshake_failures",
    "rtt",
    "queue_bytes",
    "queue_high_water",
    "stale_drop_count",
    "missing_frame_timeout_count",
    "missing_receipt_timeout_count",
)
END_REASONS = ("closed", "peer_closed", "liveness_timeout", "protocol_error", "send_timeout")
Role = Literal["brain", "body"]
Phase = Literal["startup", "run", "cleanup"]
Value = int | float | bool | Literal["unavailable"]


@dataclass(frozen=True)
class RunConfig:
    duration_seconds: float
    turn_interval_seconds: float
    fixed_reply: str
    scenario: Literal["steady", "faults", "baseline"] = "steady"
    warmup_seconds: float = 30.0
    limits_path: Path | None = None


@dataclass(frozen=True)
class WorkerSpec:
    role: Role
    incarnation: int
    port: int
    cert_dir: Path
    run_root: Path
    fixed_reply: str


@dataclass(frozen=True)
class WorkerCommand:
    kind: Literal[
        "press", "start_run", "end_run", "snapshot", "stop", "hold_playback", "release_playback"
    ]


@dataclass(frozen=True)
class WorkerEvent:
    role: Role
    incarnation: int
    kind: str
    phase: Phase
    at_s: float
    values: dict[str, Value]


def _stats(stats: TransportStats) -> dict[str, Value]:
    return {
        "sessions_started": stats.sessions_started,
        "sessions_ended_liveness_timeout": stats.sessions_ended_liveness_timeout,
        "sessions_ended_protocol_error": stats.sessions_ended_protocol_error,
        "sessions_ended_peer_closed": stats.sessions_ended_peer_closed,
        "sessions_ended_closed": stats.sessions_ended_closed,
        "sessions_ended_send_timeout": stats.sessions_ended_send_timeout,
        "protocol_errors": stats.protocol_errors,
        "tx_control_messages": stats.tx_control_messages,
        "rx_control_messages": stats.rx_control_messages,
        "tx_audio_frames": stats.tx_audio_frames,
        "rx_audio_frames": stats.rx_audio_frames,
        "tx_application_bytes": stats.tx_application_bytes,
        "rx_application_bytes": stats.rx_application_bytes,
        "connection_attempts": stats.connection_attempts,
        "connection_failures": stats.connection_failures,
        "last_valid_rx_monotonic_s": stats.last_valid_rx_monotonic_s,
        "last_session_started_monotonic_s": stats.last_session_started_monotonic_s,
        "last_session_ended_monotonic_s": stats.last_session_ended_monotonic_s,
        "last_ended_session_valid_rx_monotonic_s": stats.last_ended_session_valid_rx_monotonic_s,
    }


class _Events:
    def __init__(self, spec: WorkerSpec) -> None:
        self.spec = spec
        self.phase: Phase = "startup"
        self.pending: queue.SimpleQueue[WorkerEvent] = queue.SimpleQueue()

    def emit(self, kind: str, **values: Value) -> None:
        self.pending.put(
            WorkerEvent(
                self.spec.role,
                self.spec.incarnation,
                kind,
                self.phase,
                time.monotonic(),
                values,
            )
        )

    def flush(self, channel: Connection) -> None:
        while not self.pending.empty():
            channel.send(self.pending.get_nowait())


class _ResourceSampler:
    def __init__(self, events: _Events) -> None:
        self.events = events
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(target=self.run, name="soak-resources")

    def sample(self) -> None:
        sample = sample_self_resources()
        self.events.pending.put(
            WorkerEvent(
                self.events.spec.role,
                self.events.spec.incarnation,
                "resource_sample",
                self.events.phase,
                sample.at_s,
                dict(
                    rss_bytes=sample.rss_bytes,
                    thread_count=sample.thread_count,
                    fd_count=sample.fd_count,
                    cpu_seconds=sample.cpu_seconds,
                ),
            )
        )

    def start(self) -> None:
        self.sample()
        self.thread.start()

    def run(self) -> None:
        while not self.stop_requested.wait(RESOURCE_SAMPLE_INTERVAL_S):
            self.sample()

    def close(self) -> None:
        if self.thread.ident is not None and not self.stop_requested.is_set():
            self.stop_requested.set()
            self.thread.join()
            self.sample()


class _Recognizer:
    def recognize(self, audio: OwnedWavFile) -> RecognitionInput:
        wav = read_owned_pcm_wav(audio, expected_sample_rate_hz=16000)
        if struct.pack(f"<{wav.frame_count}h", *wav.samples_int16) != INPUT_PCM:
            raise ValueError("synthetic audio mismatch")
        return RecognitionInput(
            audio=AudioQualityEvidence(
                sample_rate_hz=16000,
                channels=1,
                sample_width_bytes=2,
                frame_count=wav.frame_count,
                rms=1000 / 32768,
                peak_abs=1000 / 32768,
                clipping_fraction=0.0,
                estimated_snr_db=20.0,
            ),
            vad_passed=True,
            text=SYNTHETIC_TRANSCRIPT,
            confidence=None,
            confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
            confidence_threshold=None,
        )


class _Synthesizer:
    def synthesize(self, text: str) -> PcmAudio:
        return PcmAudio(pcm=OUTPUT_PCM, sample_rate_hz=44100)

    def close(self) -> None:
        pass


class _Clock:
    def monotonic(self) -> float:
        return time.monotonic()


class _Brain:
    def __init__(self, spec: WorkerSpec, events: _Events, resources: ExitStack) -> None:
        self.events = events
        self.link = BrainBodyLink(
            host="127.0.0.1",
            port=0,
            ssl_context=brain_ssl_context(spec.cert_dir),
            body_id=BODY_ID,
            run_root=spec.run_root,
        )
        resources.callback(self.link.close)
        self.controller = SpeechController(
            trusted_user_id="synthetic-soak-owner",
            recorder=self.link.recorder,
            recognizer=_Recognizer(),
            dialogue=FixedReplyDialogue(spec.fixed_reply),
            synthesizer=_Synthesizer(),
            player=self.link.player,
            mute_gate=self.link.mute_gate,
            clock=_Clock(),
            session_id_supplier=lambda: self.link.current_session_id,
            session_playback_starter=self.link.player.start_for_session,
        )
        resources.callback(self.controller.shutdown)
        self.active = False
        self.completed = 0
        self.failed = 0
        self.started_sessions = 0
        self.previous_session: str | None = None
        self.ended_sessions = dict.fromkeys(END_REASONS, 0)
        self.ready = False
        self.link.start()
        events.emit("ready", pid=os.getpid(), port=self.link.port, mute_ready=False)

    def observe_sessions(self) -> None:
        stats = _stats(self.link.stats_snapshot())
        if stats["sessions_started"] > self.started_sessions:
            self.started_sessions = stats["sessions_started"]
            session = self.link.current_session_id
            self.events.emit(
                "session_started",
                count=self.started_sessions,
                changed=self.previous_session is not None and session != self.previous_session,
            )
            self.previous_session = session
        for code, reason in enumerate(END_REASONS):
            count = stats[f"sessions_ended_{reason}"]
            if count > self.ended_sessions[reason]:
                self.ended_sessions[reason] = count
                self.events.emit(
                    "session_ended",
                    count=sum(self.ended_sessions.values()),
                    reason=code,
                    last_ended_session_valid_rx_monotonic_s=stats[
                        "last_ended_session_valid_rx_monotonic_s"
                    ],
                    last_session_ended_monotonic_s=stats["last_session_ended_monotonic_s"],
                )

    def step(self) -> None:
        self.observe_sessions()
        if not self.link.mute_gate.microphone_enabled():
            self.ready = False
        if not self.ready and self.link.mute_gate.microphone_enabled():
            self.ready = True
            self.events.emit("ready", mute_ready=True)
        for item in self.link.drain_body_inputs():
            if item.session_id != self.link.current_session_id:
                continue
            if isinstance(item, ButtonPressed):
                self.active = True
                self.controller.button_pressed()
            else:
                self.controller.mute_changed()
        self.controller.tick()
        for event in self.link.player.drain_events():
            if event.succeeded:
                self.controller.playback_finished(event.playback_token)
            else:
                self.controller.playback_failed(event.playback_token)
        if self.active and self.controller.last_issue is not None:
            self.failed += 1
            self.active = False
            self.events.emit("turn_failed", count=self.failed)
        elif self.active and self.controller.state is SpeechState.IDLE:
            self.completed += 1
            self.active = False
            self.events.emit("turn_completed", count=self.completed)


class _InputHandle:
    def __init__(self) -> None:
        self.closed = False
        self.closed_at_s: Value = UNAVAILABLE

    def close(self) -> None:
        if not self.closed:
            self.closed_at_s = time.monotonic()
            self.closed = True


class _Microphone:
    sample_rate = 16000

    def __init__(self, events: _Events) -> None:
        self.events = events
        self.open_count = 0
        self.handle: _InputHandle | None = None

    def open(self, on_audio: Callable[[bytes], None]) -> _InputHandle:
        on_audio(INPUT_PCM)
        self.open_count += 1
        self.events.emit("recording_ready", count=self.open_count)
        self.handle = _InputHandle()
        return self.handle


@dataclass
class _PlaybackObservation:
    returned_at_s: Value = UNAVAILABLE


class _Speaker:
    def __init__(self, events: _Events) -> None:
        self.events = events
        self.lock = threading.Lock()
        self.active: threading.Event | None = None
        self.count = 0
        self.release = threading.Event()
        self.release.set()
        self.observation: _PlaybackObservation | None = None

    def play(self, pcm: bytes, sample_rate: int, stop: threading.Event) -> bool:
        if pcm != OUTPUT_PCM or sample_rate != 44100:
            raise ValueError("synthetic audio mismatch")
        with self.lock:
            self.active = stop
        self.count += 1
        observation = _PlaybackObservation()
        self.observation = observation
        self.events.emit("playback_started", count=self.count)
        try:
            while not self.release.is_set():
                if stop.wait(0.01):
                    return False
            return not stop.wait(0.1)
        finally:
            observation.returned_at_s = time.monotonic()
            with self.lock:
                self.active = None

    def abort(self) -> None:
        with self.lock:
            if self.active is not None:
                self.active.set()


class _Display:
    def show(self, face: object) -> None:
        pass


class _Body:
    def __init__(self, spec: WorkerSpec, events: _Events, resources: ExitStack) -> None:
        self.events = events
        self.microphone = _Microphone(events)
        self.speaker = _Speaker(events)
        self.pending_ends: queue.SimpleQueue[
            tuple[float, dict[str, Value], _InputHandle | None, _PlaybackObservation | None]
        ] = queue.SimpleQueue()
        self.observing_end = None
        self.previous_session: str | None = None
        self.core = BodyCore(
            microphone=self.microphone,
            speaker=self.speaker,
            mute_source=DevelopmentMuteSource(),
            display=_Display(),
        )
        self.session: str | None = None
        self.tokens: set[int] = set()
        self.duplicates = 0
        self.sessions = 0
        self.ends = 0
        self.completed = 0
        self.stopped = 0
        self.failed = 0
        self.link = BodyClient(
            host="127.0.0.1",
            port=spec.port,
            ssl_context=body_ssl_context(spec.cert_dir),
            body_id=BODY_ID,
            output_sample_rates=(44100,),
            on_session_started=self.on_started,
            on_session_ended=self.on_ended,
            on_message=self.on_message,
            on_frame=self.core.on_frame,
        )
        resources.callback(self.link.close)
        resources.callback(self.core.close)
        self.core.start(self)
        self.link.start()
        events.emit("ready", pid=os.getpid())

    def on_started(self, session_id: str) -> None:
        self.session = session_id
        self.tokens.clear()
        self.sessions += 1
        self.core.on_session_started(session_id)
        self.events.emit(
            "session_started",
            count=self.sessions,
            changed=self.previous_session is not None and session_id != self.previous_session,
        )
        self.previous_session = session_id

    def on_ended(self, session_id: str, reason: str) -> None:
        callback_at = time.monotonic()
        self.core.on_session_ended(session_id, reason)
        self.session = None
        self.tokens.clear()
        self.ends += 1
        stats = self.link.stats_snapshot()
        self.pending_ends.put(
            (
                callback_at,
                dict(
                    count=self.ends,
                    reason=END_REASONS.index(reason),
                    callback_at_s=callback_at,
                    last_ended_session_valid_rx_monotonic_s=stats.last_ended_session_valid_rx_monotonic_s,
                    last_session_ended_monotonic_s=stats.last_session_ended_monotonic_s,
                ),
                self.microphone.handle,
                self.speaker.observation,
            )
        )

    def on_message(self, message: object) -> None:
        if isinstance(message, PlayAudio) and message.session_id == self.session:
            if message.playback_token in self.tokens:
                self.duplicates += 1
                return
            self.tokens.add(message.playback_token)
        self.core.on_message(message)

    def send_message(self, message: object) -> None:
        self.link.send_message(message)
        if isinstance(message, PlaybackFinished):
            if message.result == "completed":
                self.completed += 1
                self.events.emit(
                    "playback_completed", count=self.completed, duplicates=self.duplicates
                )
            elif message.result == "stopped":
                self.stopped += 1
                self.events.emit("playback_stopped", count=self.stopped, duplicates=self.duplicates)
            else:
                self.failed += 1
                self.events.emit("turn_failed", count=self.failed)
        elif isinstance(message, Fault):
            self.failed += 1
            self.events.emit("turn_failed", count=self.failed)

    def send_frame(self, session_id: str, frame: object) -> None:
        self.link.send_frame(session_id, frame)

    def step(self) -> None:
        if self.observing_end is None and not self.pending_ends.empty():
            self.observing_end = self.pending_ends.get_nowait()
        if self.observing_end is None:
            return
        callback_at, values, handle, playback = self.observing_end
        closed = handle.closed_at_s if handle is not None else UNAVAILABLE
        returned = playback.returned_at_s if playback is not None else UNAVAILABLE
        waiting = (handle is not None and closed == UNAVAILABLE) or (
            playback is not None and returned == UNAVAILABLE
        )
        if waiting and time.monotonic() - callback_at < 1.1:
            return
        self.events.emit(
            "session_ended", **values, input_closed_at_s=closed, speaker_returned_at_s=returned
        )
        self.observing_end = None


def worker_main(spec: WorkerSpec, channel: Connection) -> None:
    events = _Events(spec)
    try:
        with ExitStack() as resources:
            runtime = (_Brain if spec.role == "brain" else _Body)(spec, events, resources)
            sampler = _ResourceSampler(events)
            resources.callback(sampler.close)
            next_snapshot = math.inf
            stopping = False
            while not stopping:
                while channel.poll():
                    command = channel.recv()
                    if command.kind == "press":
                        runtime.core.button_pressed()
                    elif command.kind == "hold_playback":
                        runtime.speaker.release.clear()
                    elif command.kind == "release_playback":
                        runtime.speaker.release.set()
                    elif command.kind == "stop":
                        stopping = True
                    else:
                        if command.kind == "start_run":
                            events.phase = "run"
                            sampler.start()
                            next_snapshot = time.monotonic() + 1.0
                        elif command.kind == "end_run":
                            sampler.close()
                            events.phase = "cleanup"
                            next_snapshot = math.inf
                        events.emit("snapshot", **_stats(runtime.link.stats_snapshot()))
                if not stopping:
                    runtime.step()
                if time.monotonic() >= next_snapshot:
                    events.emit("snapshot", **_stats(runtime.link.stats_snapshot()))
                    next_snapshot += 1.0
                events.flush(channel)
                if not stopping:
                    time.sleep(0.05)
            events.phase = "cleanup"
        if spec.role == "brain":
            runtime.observe_sessions()
        else:
            runtime.step()
        events.emit("snapshot", **_stats(runtime.link.stats_snapshot()))
        events.emit("stopped", code=0)
        events.flush(channel)
    except Exception:
        events.emit("worker_error", code=1)
        with suppress(BrokenPipeError, OSError):
            events.flush(channel)
    finally:
        channel.close()


class OwnedWorker:
    def __init__(self, process: multiprocessing.Process, channel: Connection) -> None:
        self.process = process
        self.channel = channel
        self.forced = False
        self.stopped = False
        self.stop_requested = False
        self.eof = False
        self.exit_observed = False

    def send(self, command: WorkerCommand) -> None:
        self.channel.send(command)

    def drain_events(self) -> tuple[WorkerEvent, ...]:
        events = []
        while not self.eof and self.channel.poll():
            try:
                events.append(self.channel.recv())
            except EOFError:
                self.eof = True
        return tuple(events)

    def request_stop(self) -> None:
        if not self.stop_requested:
            self.stop_requested = True
            if self.process.is_alive():
                with suppress(BrokenPipeError, EOFError, OSError):
                    self.send(WorkerCommand("stop"))

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.request_stop()
        self.process.join(WORKER_STOP_TIMEOUT_S)
        if self.process.is_alive():
            self.forced = True
            self.process.terminate()
            self.process.join(WORKER_STOP_TIMEOUT_S)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()


def start_worker(spec: WorkerSpec) -> OwnedWorker:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=worker_main, args=(spec, child), name=f"soak-{spec.role}")
    try:
        process.start()
    except Exception:
        parent.close()
        raise
    finally:
        child.close()
    return OwnedWorker(process, parent)


def _event_dict(event: WorkerEvent) -> dict[str, object]:
    return dict(
        role=event.role,
        incarnation=event.incarnation,
        kind=event.kind,
        phase=event.phase,
        at_s=event.at_s,
        values=event.values,
    )


class SoakSupervisor:
    """One run's workspace, children and event collector, shared by both scenarios."""

    def __init__(self, config: RunConfig, *, stop_requested: threading.Event | None = None) -> None:
        self.config = config
        limits_sha256 = UNAVAILABLE
        if config.limits_path is not None:
            limits, limits_sha256 = load_frozen_limits(config.limits_path)
            if config.warmup_seconds != limits["warmup_seconds"]:
                raise ValueError("invalid frozen soak limits")
        self.stop_requested = stop_requested if stop_requested is not None else threading.Event()
        planned = (
            math.ceil(config.duration_seconds / config.turn_interval_seconds)
            if config.scenario != "faults"
            else 36
        )
        self.report = dict(
            schema_version=1,
            scenario=config.scenario,
            requested_seconds=config.duration_seconds,
            elapsed_seconds=0.0,
            started_unix_s=time.time(),
            run_started_monotonic_s=UNAVAILABLE,
            finished_unix_s=time.time(),
            run_complete=False,
            interrupted=False,
            turns=dict(planned=planned, started=0, completed=0, failed=0, aborted=0, missed=0),
            events=[],
            workers={role: {} for role in ("brain", "body")},
            resources={role: {} for role in ("brain", "body")},
            warmup_seconds=config.warmup_seconds,
            resource_sample_interval_seconds=RESOURCE_SAMPLE_INTERVAL_S,
            resource_growth={role: {} for role in ("brain", "body")},
            faults=[],
            limits_sha256=limits_sha256,
            unavailable=list(UNAVAILABLE_METRICS),
            cleanup=dict(live_children=0, residual_files=UNAVAILABLE),
        )
        self.workers: dict[Role, OwnedWorker] = {}
        self.specs: dict[OwnedWorker, WorkerSpec] = {}
        self.owned: tuple[OwnedWorker, ...] = ()
        self.events: list[WorkerEvent] = []
        self.ready: set[Role] = set()
        self.sessions: set[Role] = set()
        self.gate_ready = False
        self.port = 0
        self.failed = False
        self.closing = False
        self.closed = False
        self.active = False
        self.auto_press = False
        self.second_press = False
        self.brain_done = False
        self.body_done = False
        self.origin: float | None = None
        self.deadline = math.inf
        self.end_snapshots = False
        self.workspace: Path | None = None
        self.directory: tempfile.TemporaryDirectory | None = None
        self.forwarder = None
        self.applied_ordinals: set[int] = set()
        self.paused: list[OwnedWorker] = []
        self.expected_exits: set[tuple[Role, int]] = set()
        self.observed_exits: set[OwnedWorker] = set()
        self.unexpected_exits: list[dict[str, object]] = []

    def consume(self) -> None:
        if self.forwarder is not None and self.forwarder.failed:
            self.failed = True
        for worker in self.owned:
            spec = self.specs[worker]
            current = self.workers[spec.role] is worker
            for event in worker.drain_events():
                self.events.append(event)
                self.report["events"].append(_event_dict(event))
                if current and event.kind == "ready":
                    self.ready.add(spec.role)
                    if spec.role == "brain":
                        self.port = event.values.get("port", self.port)
                        self.gate_ready |= event.values.get("mute_ready", False)
                elif current and event.kind == "session_started":
                    self.sessions.add(spec.role)
                elif event.kind == "worker_error":
                    self.failed = True
                elif current and event.kind == "session_ended":
                    self.sessions.discard(spec.role)
                    self.gate_ready = False
                    if self.config.scenario != "faults" and not self.closing:
                        self.failed = True
                elif event.kind == "stopped" and not self.closing:
                    self.failed = True
                elif current and event.kind == "turn_failed":
                    if self.config.scenario != "faults":
                        self.failed = True
                    if self.active:
                        self.report["turns"]["failed"] += 1
                        self.active = False
                elif current and event.kind == "recording_ready" and self.active:
                    if self.auto_press and not self.second_press:
                        self.command("body", "press")
                        self.second_press = True
                elif current and event.kind == "turn_completed" and self.active:
                    self.brain_done = True
                elif current and event.kind == "playback_completed" and self.active:
                    self.body_done = True
                if event.kind == "resource_sample":
                    self.report["resources"][spec.role][str(event.incarnation)].append(
                        dict(at_s=event.at_s, **event.values)
                    )
                if event.kind == "snapshot":
                    snapshots = self.report["workers"][spec.role][str(event.incarnation)]
                    if event.phase == "run" and snapshots["start"] == UNAVAILABLE:
                        snapshots["start"] = event.values
                    elif event.phase == "cleanup":
                        key = "run_end" if snapshots["run_end"] == UNAVAILABLE else "after_close"
                        snapshots[key] = event.values
            if worker.process.exitcode is not None and worker not in self.observed_exits:
                self.observed_exits.add(worker)
                worker.exit_observed = True
                if not self.closing and (spec.role, spec.incarnation) not in self.expected_exits:
                    self.unexpected_exits.append(
                        dict(
                            kind="unexpected_worker_exit",
                            role=spec.role,
                            incarnation=spec.incarnation,
                            exitcode=worker.process.exitcode,
                        )
                    )
                    self.failed = True
        if self.active and self.brain_done and self.body_done:
            self.report["turns"]["completed"] += 1
            self.active = False

    def command(self, role: Role, kind: str) -> None:
        self.workers[role].send(WorkerCommand(kind))

    def launch(self, role: Role, incarnation: int) -> None:
        root = self.workspace / f"{role}-{incarnation}"
        root.mkdir()
        cert_dir = self.pairing.brain_dir if role == "brain" else self.pairing.body_dir
        port = self.body_port if role == "body" else 0
        spec = WorkerSpec(role, incarnation, port, cert_dir, root, self.config.fixed_reply)
        worker = start_worker(spec)
        self.workers[role] = worker
        self.specs[worker] = spec
        self.owned += (worker,)
        self.report["workers"][role][str(incarnation)] = dict.fromkeys(
            ("start", "run_end", "after_close"), UNAVAILABLE
        )
        self.report["resources"][role][str(incarnation)] = []

    def wait_for(
        self, predicate: Callable[[], bool], timeout: float, *, hold: bool = False
    ) -> bool:
        deadline = min(time.monotonic() + timeout, self.deadline)
        while True:
            self.consume()
            if self.failed or self.stop_requested.is_set():
                return False
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return hold and time.monotonic() < self.deadline
            self.stop_requested.wait(0.01)

    def has_event(self, role: Role, kind: str, offset: int) -> bool:
        return any(e.role == role and e.kind == kind for e in self.events[offset:])

    def latest_event(self, role: Role, kind: str, offset: int) -> WorkerEvent | None:
        return next(
            (e for e in reversed(self.events[offset:]) if e.role == role and e.kind == kind), None
        )

    def start(self) -> None:
        if sys.platform != "darwin":
            raise ValueError("invalid soak platform")
        self.directory = tempfile.TemporaryDirectory(prefix="bodylink-soak-", dir="/private/tmp")
        self.workspace = Path(self.directory.name)
        self.pairing = generate_pairing(self.workspace / "pairing", BODY_ID)
        self.body_port = 0
        self.launch("brain", 1)
        if not self.wait_for(lambda: self.port != 0, STARTUP_TIMEOUT_S):
            return
        self.body_port = self.port
        if self.config.scenario == "faults":
            if __package__:
                from .bodylink_soak_faults import TransparentForwarder
            else:
                from bodylink_soak_faults import TransparentForwarder
            self.forwarder = TransparentForwarder(upstream_port=self.port)
            self.body_port = self.forwarder.start()
        self.launch("body", 1)
        if not self.wait_for(
            lambda: len(self.ready) == len(self.sessions) == 2 and self.gate_ready,
            STARTUP_TIMEOUT_S,
        ):
            return
        for role in self.workers:
            self.command(role, "start_run")
        if self.wait_for(
            lambda: all(
                self.report["workers"][role]["1"]["start"] != UNAVAILABLE for role in self.workers
            ),
            STARTUP_TIMEOUT_S,
        ):
            self.origin = time.monotonic()
            self.report["run_started_monotonic_s"] = self.origin
            self.report["started_unix_s"] = time.time()

    def restart(self, role: Role) -> None:
        worker = self.workers[role]
        spec = self.specs[worker]
        # Only this exact killed incarnation is expected to exit.
        self.expected_exits.add((role, spec.incarnation))
        worker.process.kill()
        worker.process.join()
        self.consume()
        if self.failed or self.stop_requested.is_set():
            return
        self.sessions.discard(role)
        self.ready.discard(role)
        self.gate_ready = False
        if role == "brain":
            self.port = 0
        self.launch(role, spec.incarnation + 1)
        if role == "brain":
            if not self.wait_for(lambda: self.port != 0, STARTUP_TIMEOUT_S):
                return
            self.forwarder.set_upstream_port(self.port)
        self.command(role, "start_run")

    def begin_turn(self, *, auto_press: bool = False) -> None:
        self.active = True
        self.auto_press = auto_press
        self.second_press = self.brain_done = self.body_done = False
        self.report["turns"]["started"] += 1
        self.command("body", "press")

    def abort_turn(self) -> None:
        if self.active:
            self.report["turns"]["aborted"] += 1
            self.active = False

    def complete_turn(self, timeout: float = STARTUP_TIMEOUT_S) -> bool:
        offset = len(self.events)
        self.begin_turn(auto_press=True)
        done = self.wait_for(lambda: not self.active, timeout)
        return bool(
            done
            and self.brain_done
            and self.body_done
            and not any(e.kind == "turn_failed" for e in self.events[offset:])
        )

    def run_case(self, case: FaultCase) -> dict[str, object]:
        if __package__:
            from .bodylink_soak_faults import run_fault_case
        else:
            from bodylink_soak_faults import run_fault_case
        return run_fault_case(self, case)

    def run_steady(self) -> None:
        planned = self.report["turns"]["planned"]
        turns = self.report["turns"]
        next_slot = 0
        deadline = self.origin + self.config.duration_seconds
        while time.monotonic() < deadline and not self.failed and not self.stop_requested.is_set():
            self.consume()
            now = time.monotonic()
            if self.failed or self.stop_requested.is_set() or now >= deadline:
                break
            while (
                next_slot < planned
                and self.origin + (next_slot * self.config.turn_interval_seconds) <= now
            ):
                next_slot += 1
                if self.active:
                    turns["missed"] += 1
                else:
                    self.begin_turn(auto_press=True)
            self.stop_requested.wait(0.005)
        if not self.failed and not self.stop_requested.is_set():
            turns["missed"] += planned - next_slot
        self.wait_for(lambda: not self.active, TURN_FINISH_GRACE_S)
        self.consume()
        self.abort_turn()

    def finish_observation(self) -> None:
        self.report["elapsed_seconds"] = time.monotonic() - self.origin
        self.report["finished_unix_s"] = time.time()
        self.deadline = math.inf
        for worker in self.workers.values():
            if worker.process.is_alive():
                worker.send(WorkerCommand("end_run"))
        self.end_snapshots = self.wait_for(
            lambda: all(
                self.report["workers"][role][str(self.specs[worker].incarnation)]["run_end"]
                != UNAVAILABLE
                for role, worker in self.workers.items()
            ),
            WORKER_STOP_TIMEOUT_S,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.abort_turn()
        self.closing = True
        if self.forwarder is not None and self.forwarder.running:
            self.forwarder.set_blackhole(None)
        if self.paused:
            if __package__:
                from .bodylink_soak_faults import signal_owned
            else:
                from bodylink_soak_faults import signal_owned
            import signal

            for worker in self.paused:
                signal_owned(worker, signal.SIGCONT, owned=self.owned)
            self.paused.clear()
        for role in ("body", "brain"):
            for worker in self.owned:
                if self.specs[worker].role == role:
                    if self.config.scenario != "faults":
                        worker.stop()
                    else:
                        worker.request_stop()
        if self.forwarder is not None:
            self.forwarder.close()
        for worker in self.owned:
            worker.stop()
        self.consume()
        for worker in self.owned:
            worker.channel.close()
        self.report["cleanup"]["live_children"] = sum(w.process.is_alive() for w in self.owned)
        if self.directory is not None:
            self.directory.cleanup()
        self.report["cleanup"]["residual_files"] = int(
            self.workspace is not None and self.workspace.exists()
        )
        self.report["interrupted"] = self.stop_requested.is_set()
        turns = self.report["turns"]
        turns["missed"] += turns["planned"] - sum(
            turns[key] for key in ("completed", "failed", "aborted", "missed")
        )
        if self.origin is not None:
            for role, incarnations in self.report["resources"].items():
                for incarnation, samples in incarnations.items():
                    self.report["resource_growth"][role][incarnation] = resource_growth(
                        [ResourceSample(**sample) for sample in samples],
                        run_started_monotonic_s=self.origin,
                        warmup_seconds=self.config.warmup_seconds,
                    )


def run_soak(
    config: RunConfig, *, stop_requested: threading.Event | None = None
) -> dict[str, object]:
    if sys.platform != "darwin":
        raise ValueError("invalid soak platform")
    supervisor = SoakSupervisor(config, stop_requested=stop_requested)
    try:
        supervisor.start()
        if supervisor.origin is not None:
            if config.scenario != "faults":
                supervisor.run_steady()
            else:
                if __package__:
                    from .bodylink_soak_faults import FAULT_CASES
                else:
                    from bodylink_soak_faults import FAULT_CASES
                supervisor.deadline = supervisor.origin + config.duration_seconds
                for case in FAULT_CASES:
                    if time.monotonic() >= supervisor.deadline:
                        break
                    if not supervisor.run_case(case)["complete"]:
                        break
            supervisor.finish_observation()
    except Exception:
        supervisor.failed = True
    finally:
        supervisor.close()
    report = supervisor.report
    if config.scenario != "faults":
        scenario_complete = not any(report["turns"][key] for key in ("failed", "aborted", "missed"))
    else:
        scenario_complete = len(report["faults"]) == 12 and all(
            case["complete"] for case in report["faults"]
        )
    after_close = len(supervisor.workers) == 2 and all(
        report["workers"][role][str(supervisor.specs[worker].incarnation)]["after_close"]
        != UNAVAILABLE
        for role, worker in supervisor.workers.items()
    )
    report["run_complete"] = bool(
        supervisor.origin is not None
        and supervisor.end_snapshots
        and after_close
        and not supervisor.failed
        and not supervisor.stop_requested.is_set()
        and scenario_complete
        and all(
            not worker.forced
            and (
                worker.process.exitcode == 0
                or (supervisor.specs[worker].role, supervisor.specs[worker].incarnation)
                in supervisor.expected_exits
            )
            for worker in supervisor.owned
        )
        and report["cleanup"] == dict(live_children=0, residual_files=0)
    )
    return report


def _artifact_path(path: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    home = Path.home()
    forbidden_roots = (
        home / "Library",
        home / ".ssh",
        home / ".config",
        home / ".local" / "share",
        Path("/Library/Application Support"),
    )
    if set(path.parts) & {"private_data", "runtime_data", "test-results"} or any(
        path.is_relative_to(root) for root in forbidden_roots
    ):
        raise ValueError("invalid soak artifact path")
    return path


def load_frozen_limits(path: Path) -> tuple[dict[str, object], str]:
    raw = _artifact_path(path).read_bytes()
    invalid = "invalid frozen soak limits"
    try:
        limits = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(invalid) from None
    if type(limits) is not dict or set(limits) != {
        "schema_version",
        "source_baseline_sha256",
        "warmup_seconds",
        "resource_sample_interval_seconds",
        "brain",
        "body",
    }:
        raise ValueError(invalid)
    if (
        type(limits["schema_version"]) is not int
        or limits["schema_version"] != 1
        or type(limits["source_baseline_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", limits["source_baseline_sha256"]) is None
        or type(limits["warmup_seconds"]) not in (int, float)
        or (type(limits["warmup_seconds"]) is float and not math.isfinite(limits["warmup_seconds"]))
        or limits["warmup_seconds"] < 0
        or type(limits["resource_sample_interval_seconds"]) not in (int, float)
        or limits["resource_sample_interval_seconds"] != RESOURCE_SAMPLE_INTERVAL_S
    ):
        raise ValueError(invalid)
    for role in ("brain", "body"):
        values = limits[role]
        if (
            type(values) is not dict
            or set(values) != {"rss_growth_bytes", "thread_growth", "fd_growth"}
            or any(type(value) is not int or value < 0 for value in values.values())
        ):
            raise ValueError(invalid)
    return limits, hashlib.sha256(raw).hexdigest()


def write_report(path: Path, report: dict[str, object]) -> None:
    path = _artifact_path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
