"""Owned POSIX faults and an opaque, bounded loopback TCP forwarder."""

from __future__ import annotations

import errno
import os
import queue
import selectors
import signal
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .bodylink_soak_support import OwnedWorker, SoakSupervisor

LIVENESS_LIMIT_S = 4.0
DEVICE_STOP_LIMIT_S = 1.0
FAULT_HOLD_S = 5.0
RECOVERY_OBSERVATION_TIMEOUT_S = 30.0
RECOVERY_QUIET_S = 1.0
FORWARD_BUFFER_BYTES = 65536

FaultKind = Literal[
    "brain_restart",
    "body_restart",
    "brain_pause",
    "body_pause",
    "uplink_blackhole",
    "downlink_blackhole",
]


@dataclass(frozen=True, slots=True)
class FaultCase:
    ordinal: int
    kind: FaultKind
    phase: Literal["recording", "playing"]


FAULT_CASES = tuple(
    FaultCase(2 * index + phase_index + 1, kind, phase)
    for index, kind in enumerate(
        (
            "brain_restart",
            "body_restart",
            "brain_pause",
            "body_pause",
            "uplink_blackhole",
            "downlink_blackhole",
        )
    )
    for phase_index, phase in enumerate(("recording", "playing"))
)


def signal_owned(worker: OwnedWorker, sig: int, *, owned: tuple[OwnedWorker, ...]) -> None:
    if not any(worker is candidate for candidate in owned):
        raise ValueError("worker is not owned")
    if worker.process.is_alive():
        try:
            os.kill(worker.process.pid, sig)
        except ProcessLookupError:
            worker.exit_observed = True
    else:
        worker.exit_observed = True


@dataclass(eq=False)
class _Connection:
    body: socket.socket
    brain: socket.socket
    connecting: bool
    uplink: bytearray = field(default_factory=bytearray)
    downlink: bytearray = field(default_factory=bytearray)


class TransparentForwarder:
    def __init__(self, *, upstream_port: int) -> None:
        self.upstream_port = upstream_port
        self.blackhole: Literal["uplink", "downlink"] | None = None
        self.connections: list[_Connection] = []
        self.commands: queue.SimpleQueue[tuple[Callable[[], None], threading.Event]] = (
            queue.SimpleQueue()
        )
        self.selector = selectors.DefaultSelector()
        self.wake_read, self.wake_write = socket.socketpair()
        self.wake_read.setblocking(False)
        self.listener = socket.socket()
        self.running = False
        self.failed = False
        self.thread = threading.Thread(target=self._run, name="soak-forwarder")

    def start(self) -> int:
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.setblocking(False)
        self.selector.register(self.listener, selectors.EVENT_READ, "listen")
        self.selector.register(self.wake_read, selectors.EVENT_READ, "wake")
        self.running = True
        self.thread.start()
        return self.listener.getsockname()[1]

    def _command(self, action: Callable[[], None]) -> None:
        done = threading.Event()
        self.commands.put((action, done))
        self.wake_write.send(b"!")
        while not done.wait(0.05):
            if not self.thread.is_alive():
                return

    def set_upstream_port(self, port: int) -> None:
        self._command(lambda: setattr(self, "upstream_port", port))

    def set_blackhole(self, direction: Literal["uplink", "downlink"] | None) -> None:
        def apply() -> None:
            # Bytes already received while held must never become post-release traffic.
            if self.blackhole is not None:
                for connection in tuple(self.connections):
                    source = connection.body if self.blackhole == "uplink" else connection.brain
                    try:
                        while source.recv(FORWARD_BUFFER_BYTES):
                            pass
                        self._drop(connection)
                    except BlockingIOError:
                        pass
                    except OSError:
                        self._drop(connection)
            self.blackhole = direction
            if direction is not None:
                for connection in self.connections:
                    (connection.uplink if direction == "uplink" else connection.downlink).clear()

        self._command(apply)

    def _drop(self, connection: _Connection) -> None:
        if connection not in self.connections:
            return
        self.connections.remove(connection)
        for sock in (connection.body, connection.brain):
            if sock in self.selector.get_map():
                self.selector.unregister(sock)
            sock.close()
        connection.uplink.clear()
        connection.downlink.clear()

    def _interest(self, connection: _Connection) -> None:
        for sock, incoming, outgoing, direction in (
            (connection.body, connection.uplink, connection.downlink, "uplink"),
            (connection.brain, connection.downlink, connection.uplink, "downlink"),
        ):
            mask = 0
            if not connection.connecting:
                if self.blackhole == direction or len(incoming) < FORWARD_BUFFER_BYTES:
                    mask |= selectors.EVENT_READ
                if outgoing:
                    mask |= selectors.EVENT_WRITE
            elif sock is connection.brain:
                mask = selectors.EVENT_WRITE
            registered = sock in self.selector.get_map()
            if mask:
                data = (connection, direction)
                if registered:
                    self.selector.modify(sock, mask, data)
                else:
                    self.selector.register(sock, mask, data)
            elif registered:
                self.selector.unregister(sock)

    def _transfer(self, connection: _Connection, direction: str, mask: int) -> None:
        sock = connection.body if direction == "uplink" else connection.brain
        incoming = connection.uplink if direction == "uplink" else connection.downlink
        outgoing = connection.downlink if direction == "uplink" else connection.uplink
        try:
            if connection.connecting:
                if sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                    self._drop(connection)
                else:
                    connection.connecting = False
                return
            if mask & selectors.EVENT_READ:
                size = (
                    FORWARD_BUFFER_BYTES
                    if self.blackhole == direction
                    else (FORWARD_BUFFER_BYTES - len(incoming))
                )
                data = sock.recv(size)
                if not data:
                    self._drop(connection)
                    return
                if self.blackhole != direction:
                    incoming.extend(data)
            if mask & selectors.EVENT_WRITE and outgoing:
                del outgoing[: sock.send(outgoing)]
        except BlockingIOError:
            pass
        except OSError:
            self._drop(connection)

    def _run(self) -> None:
        try:
            while self.running:
                for key, mask in self.selector.select():
                    if key.data == "wake":
                        self.wake_read.recv(4096)
                        while not self.commands.empty():
                            action, done = self.commands.get_nowait()
                            try:
                                action()
                            finally:
                                done.set()
                    elif key.data == "listen":
                        body, _ = self.listener.accept()
                        brain = socket.socket()
                        body.setblocking(False)
                        brain.setblocking(False)
                        result = brain.connect_ex(("127.0.0.1", self.upstream_port))
                        connection = _Connection(body, brain, result != 0)
                        self.connections.append(connection)
                        if result not in (0, errno.EINPROGRESS):
                            self._drop(connection)
                    else:
                        connection, direction = key.data
                        if connection in self.connections:
                            self._transfer(connection, direction, mask)
                for connection in self.connections:
                    self._interest(connection)
        except Exception:
            # This thread is the network I/O boundary; never print wire data or exceptions.
            self.failed = True
        finally:
            self.running = False
            for connection in tuple(self.connections):
                self._drop(connection)
            self.listener.close()
            self.wake_read.close()
            self.selector.close()

    def close(self) -> None:
        if self.thread.is_alive():
            self._command(lambda: setattr(self, "running", False))
        if self.thread.ident is not None:
            self.thread.join()
        else:
            self.listener.close()
            self.wake_read.close()
            self.selector.close()
        self.wake_write.close()


def run_fault_case(supervisor: SoakSupervisor, case: FaultCase) -> dict[str, object]:
    if case.ordinal in supervisor.applied_ordinals:
        raise ValueError("fault already applied")
    result = dict(
        ordinal=case.ordinal,
        kind=case.kind,
        phase=case.phase,
        applied=False,
        released=False,
        complete=False,
        source_session_changed=False,
        quiet_without_replay=False,
        recovery_turn_completed=False,
        liveness_seconds="unavailable",
        device_stop_seconds="unavailable",
        handshake_recovery_seconds="unavailable",
        turn_recovery_seconds="unavailable",
        unavailable_reason="missing_event",
        unexpected_exits=[],
    )
    supervisor.report["faults"].append(result)
    exit_offset = len(supervisor.unexpected_exits)
    fault_offset: int | None = None
    source = (
        "body" if case.kind in ("brain_restart", "brain_pause", "downlink_blackhole") else "brain"
    )
    try:
        if not supervisor.complete_turn():
            return result
        stage_offset = len(supervisor.events)
        if case.phase == "playing":
            supervisor.command("body", "hold_playback")
        supervisor.begin_turn()
        if not supervisor.wait_for(
            lambda: supervisor.has_event("body", "recording_ready", stage_offset),
            RECOVERY_OBSERVATION_TIMEOUT_S,
        ):
            return result
        if case.phase == "playing":
            supervisor.command("body", "press")
            if not supervisor.wait_for(
                lambda: supervisor.has_event("body", "playback_started", stage_offset),
                RECOVERY_OBSERVATION_TIMEOUT_S,
            ):
                return result
        supervisor.applied_ordinals.add(case.ordinal)
        result["applied"] = True
        supervisor.abort_turn()
        fault_offset = len(supervisor.events)
        if case.kind.endswith("restart"):
            role = "brain" if case.kind == "brain_restart" else "body"
            supervisor.restart(role)
        elif case.kind.endswith("pause"):
            role = "brain" if case.kind == "brain_pause" else "body"
            worker = supervisor.workers[role]
            supervisor.paused.append(worker)
            signal_owned(worker, signal.SIGSTOP, owned=supervisor.owned)
            if not supervisor.wait_for(lambda: False, FAULT_HOLD_S, hold=True):
                return result
            signal_owned(worker, signal.SIGCONT, owned=supervisor.owned)
            supervisor.paused.remove(worker)
        else:
            direction = "uplink" if case.kind == "uplink_blackhole" else "downlink"
            supervisor.forwarder.set_blackhole(direction)
            if not supervisor.wait_for(lambda: False, FAULT_HOLD_S, hold=True):
                return result
            supervisor.forwarder.set_blackhole(None)
        released_at = time.monotonic()
        result["released"] = True
        recovery_deadline = released_at + RECOVERY_OBSERVATION_TIMEOUT_S
        if not supervisor.wait_for(
            lambda: (
                all(
                    supervisor.has_event(role, "session_started", fault_offset)
                    for role in ("brain", "body")
                )
                and supervisor.gate_ready
            ),
            RECOVERY_OBSERVATION_TIMEOUT_S,
        ):
            return result
        result["handshake_recovery_seconds"] = time.monotonic() - released_at
        source_start = supervisor.latest_event(source, "session_started", fault_offset)
        result["source_session_changed"] = source_start.values["changed"]
        supervisor.command("body", "release_playback")
        quiet_offset = len(supervisor.events)
        if not supervisor.wait_for(lambda: False, RECOVERY_QUIET_S, hold=True):
            return result
        result["quiet_without_replay"] = not any(
            event.kind in ("recording_ready", "playback_started")
            for event in supervisor.events[quiet_offset:]
        )
        if not result["quiet_without_replay"]:
            return result
        if not supervisor.complete_turn(timeout=max(0, recovery_deadline - time.monotonic())):
            return result
        result["recovery_turn_completed"] = True
        result["turn_recovery_seconds"] = time.monotonic() - released_at
        result["complete"] = time.monotonic() <= recovery_deadline
        return result
    finally:
        if fault_offset is not None:
            ended = next(
                (
                    e
                    for e in supervisor.events[fault_offset:]
                    if e.role == source and e.kind == "session_ended"
                ),
                None,
            )
            if ended is not None:
                valid_rx = ended.values["last_ended_session_valid_rx_monotonic_s"]
                ended_at = ended.values["last_session_ended_monotonic_s"]
                if valid_rx != "unavailable" and ended_at != "unavailable":
                    result["liveness_seconds"] = ended_at - valid_rx
            body_ended = next(
                (
                    e
                    for e in supervisor.events[fault_offset:]
                    if e.role == "body" and e.kind == "session_ended"
                ),
                None,
            )
            stop_seconds = "unavailable"
            if body_ended is not None:
                key = "input_closed_at_s" if case.phase == "recording" else "speaker_returned_at_s"
                stopped_at = body_ended.values[key]
                if stopped_at != "unavailable":
                    stop_seconds = max(0.0, stopped_at - body_ended.values["callback_at_s"])
            if case.kind == "body_restart":
                result["unavailable_reason"] = "target_not_running"
            elif case.kind == "body_pause":
                result["unavailable_reason"] = "target_suspended"
            else:
                result["device_stop_seconds"] = stop_seconds
                result["unavailable_reason"] = (
                    "none" if stop_seconds != "unavailable" else "missing_event"
                )
            liveness = result["liveness_seconds"]
            stopped = case.kind == "body_restart" or (
                stop_seconds != "unavailable" and 0 <= stop_seconds <= DEVICE_STOP_LIMIT_S
            )
            result["complete"] = bool(
                result["complete"]
                and result["source_session_changed"]
                and stopped
                and liveness != "unavailable"
                and 0 <= liveness <= LIVENESS_LIMIT_S
            )
        result["unexpected_exits"] = supervisor.unexpected_exits[exit_offset:]
        if result["unexpected_exits"] or supervisor.failed or supervisor.stop_requested.is_set():
            result["complete"] = False
