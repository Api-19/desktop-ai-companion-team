"""Fault matrix, owned signal boundary and opaque TCP forwarding."""

import multiprocessing
import os
import signal
import socket
import threading
from contextlib import suppress

import pytest

from scripts import bodylink_soak_faults as faults
from scripts import bodylink_soak_support as soak


def test_fault_matrix_has_exact_twelve_cases():
    assert [(case.ordinal, case.kind, case.phase) for case in faults.FAULT_CASES] == [
        (2 * index + phase_index + 1, kind, phase)
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
    ]


def test_fault_is_applied_once():
    supervisor = soak.SoakSupervisor(soak.RunConfig(1, 30, "answer", "faults"))
    supervisor.applied_ordinals.add(1)
    with pytest.raises(ValueError, match=r"^fault already applied$"):
        supervisor.run_case(faults.FAULT_CASES[0])


def _wait_child(event):
    event.wait(10)


def test_fault_targets_only_owned_children(monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    event = ctx.Event()
    process = ctx.Process(target=_wait_child, args=(event,))
    parent, child = ctx.Pipe()
    process.start()
    worker = soak.OwnedWorker(process, parent)
    foreign = soak.OwnedWorker(process, child)
    calls = []
    original = os.kill

    def kill(pid, sig):
        calls.append((pid, sig))
        original(pid, sig)

    monkeypatch.setattr(faults.os, "kill", kill)
    try:
        with pytest.raises(ValueError, match=r"^worker is not owned$"):
            faults.signal_owned(foreign, signal.SIGCONT, owned=(worker,))
        assert calls == []
        faults.signal_owned(worker, signal.SIGCONT, owned=(worker,))
        assert calls == [(process.pid, signal.SIGCONT)]
        event.set()
        process.join(5)
        faults.signal_owned(worker, signal.SIGCONT, owned=(worker,))
        faults.signal_owned(worker, signal.SIGCONT, owned=(worker,))
        assert len(calls) == 1
        assert worker.exit_observed
    finally:
        event.set()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        child.close()


@pytest.mark.parametrize("direction", ["uplink", "downlink"])
def test_forwarder_discards_blackhole_bytes_without_replay(direction):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(3)
        forwarder = faults.TransparentForwarder(upstream_port=server.getsockname()[1])
        try:
            port = forwarder.start()
            with socket.create_connection(("127.0.0.1", port), timeout=3) as body:
                brain, _ = server.accept()
                with brain:
                    brain.settimeout(3)
                    source, target = (body, brain) if direction == "uplink" else (brain, body)
                    source.sendall(b"before")
                    assert target.recv(6) == b"before"
                    forwarder.set_blackhole(direction)
                    source.sendall(b"discard" * 20000)
                    # A round trip on the other direction barriers the selector read loop.
                    target.sendall(b"barrier")
                    assert source.recv(7) == b"barrier"
                    forwarder.set_blackhole(None)
                    source.sendall(b"after")
                    assert target.recv(100) == b"after"
        finally:
            forwarder.close()
        assert not forwarder.thread.is_alive()
        assert not forwarder.connections
        assert forwarder.listener.fileno() == -1


def test_close_joins_idle_forwarder():
    forwarder = faults.TransparentForwarder(upstream_port=1)
    forwarder.start()
    forwarder.close()
    forwarder.close()
    assert not forwarder.thread.is_alive()
    assert threading.current_thread() is not forwarder.thread


def test_forwarder_backpressure_bounds_buffers_and_close_discards_them():
    import time

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(3)
        forwarder = faults.TransparentForwarder(upstream_port=server.getsockname()[1])
        try:
            port = forwarder.start()
            with socket.create_connection(("127.0.0.1", port), timeout=3) as body:
                brain, _ = server.accept()
                with brain:
                    brain.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
                    body.setblocking(False)
                    deadline = time.monotonic() + 3
                    filled = False
                    while time.monotonic() < deadline:
                        with suppress(BlockingIOError):
                            body.send(b"x" * faults.FORWARD_BUFFER_BYTES)
                        for connection in forwarder.connections:
                            assert len(connection.uplink) <= faults.FORWARD_BUFFER_BYTES
                            assert len(connection.downlink) <= faults.FORWARD_BUFFER_BYTES
                            filled |= len(connection.uplink) == faults.FORWARD_BUFFER_BYTES
                        if filled:
                            break
                        time.sleep(0.001)
                    assert filled
                    connection = forwarder.connections[0]
                    forwarder.close()
                    assert not connection.uplink and not connection.downlink
        finally:
            forwarder.close()


def test_forwarder_new_connections_use_updated_upstream():
    with socket.socket() as first, socket.socket() as second:
        for server in (first, second):
            server.bind(("127.0.0.1", 0))
            server.listen()
            server.settimeout(3)
        forwarder = faults.TransparentForwarder(upstream_port=first.getsockname()[1])
        try:
            port = forwarder.start()
            for server in (first, second):
                forwarder.set_upstream_port(server.getsockname()[1])
                with socket.create_connection(("127.0.0.1", port), timeout=3) as body:
                    brain, _ = server.accept()
                    with brain:
                        brain.settimeout(3)
                        body.sendall(b"opaque")
                        assert brain.recv(6) == b"opaque"
        finally:
            forwarder.close()


def test_forwarder_network_failure_closes_without_hanging_or_logging(monkeypatch, capfd):
    import time

    def fail(*args):
        raise OSError("synthetic-forwarder-exception-payload")

    monkeypatch.setattr(faults.TransparentForwarder, "_transfer", fail)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(3)
        forwarder = faults.TransparentForwarder(upstream_port=server.getsockname()[1])
        try:
            port = forwarder.start()
            with socket.create_connection(("127.0.0.1", port), timeout=3) as body:
                body.sendall(b"opaque")
                brain, _ = server.accept()
                brain.close()
                deadline = time.monotonic() + 3
                while forwarder.thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert forwarder.failed
        finally:
            forwarder.close()
        assert not forwarder.thread.is_alive()
        captured = capfd.readouterr()
        assert captured.out == captured.err == ""
