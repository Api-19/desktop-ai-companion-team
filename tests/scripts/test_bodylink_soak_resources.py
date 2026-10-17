"""Native self-only counters, frozen limits and per-incarnation growth."""

import ctypes
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bodylink_soak_resources as resources
from scripts import bodylink_soak_support as soak


def test_native_resource_structures_match_macos_abi():
    assert ctypes.sizeof(resources.ProcTaskInfo) == 96
    names = (
        "pti_virtual_size",
        "pti_resident_size",
        "pti_total_user",
        "pti_total_system",
        "pti_threads_user",
        "pti_threads_system",
        "pti_policy",
        "pti_faults",
        "pti_pageins",
        "pti_cow_faults",
        "pti_messages_sent",
        "pti_messages_received",
        "pti_syscalls_mach",
        "pti_syscalls_unix",
        "pti_csw",
        "pti_threadnum",
        "pti_numrunning",
        "pti_priority",
    )
    for index, name in enumerate(names):
        assert getattr(resources.ProcTaskInfo, name).offset == (
            index * 8 if index < 6 else 48 + (index - 6) * 4
        )
    assert ctypes.sizeof(resources.ProcFdInfo) == 8
    assert resources.ProcFdInfo.proc_fd.offset == 0
    assert resources.ProcFdInfo.proc_fdtype.offset == 4
    sample = resources.ResourceSample(1.0, 1, 2, 3, 4.0)
    assert not hasattr(sample, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.at_s = 2.0


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libproc resource counters")
def test_native_self_sample_has_current_rss_threads_and_fds():
    sample = resources.sample_self_resources()
    assert sample.at_s > 0
    for name in ("rss_bytes", "thread_count", "fd_count"):
        value = getattr(sample, name)
        assert type(value) is int and value > 0
    assert type(sample.cpu_seconds) is float and sample.cpu_seconds >= 0


class NativeCall:
    def __init__(self, task_size=96, fd_sizes=(16, 24)):
        self.task_size = task_size
        self.fd_sizes = iter(fd_sizes)
        self.calls = []

    def __call__(self, pid, flavor, arg, buffer, size):
        self.calls.append((pid, flavor, arg, size))
        assert pid == os.getpid()
        if flavor == 4:
            task = ctypes.cast(buffer, ctypes.POINTER(resources.ProcTaskInfo)).contents
            task.pti_resident_size = 123456
            task.pti_threadnum = 7
            return self.task_size
        assert flavor == 1
        assert (buffer is None) == (size == 0)
        return next(self.fd_sizes)


def _native(monkeypatch, call):
    def load(path):
        assert path == "/usr/lib/libproc.dylib"
        return SimpleNamespace(proc_pidinfo=call)

    monkeypatch.setattr(resources.ctypes, "CDLL", load)


def test_native_call_uses_exact_abi_and_current_values(monkeypatch):
    call = NativeCall()
    _native(monkeypatch, call)
    monkeypatch.setattr(
        resources.resource,
        "getrusage",
        lambda who: SimpleNamespace(
            ru_utime=1.25,
            ru_stime=0.5,
            ru_maxrss=999999,
        ),
    )
    sample = resources.sample_self_resources()
    assert (sample.rss_bytes, sample.thread_count, sample.fd_count, sample.cpu_seconds) == (
        123456,
        7,
        3,
        1.75,
    )
    assert call.argtypes == [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    assert call.restype is ctypes.c_int
    assert call.calls == [(os.getpid(), 4, 0, 96), (os.getpid(), 1, 0, 0), (os.getpid(), 1, 0, 272)]


@pytest.mark.parametrize("size", [-1, 0, 95, 97])
def test_native_short_reads_are_unavailable(monkeypatch, size):
    _native(monkeypatch, NativeCall(task_size=size, fd_sizes=(16, 7)))
    sample = resources.sample_self_resources()
    assert sample.rss_bytes == sample.thread_count == sample.fd_count == "unavailable"
    assert sample.cpu_seconds >= 0


@pytest.mark.parametrize("returned", [272, 280, 0, -1, 23])
def test_fd_list_full_buffer_is_unavailable(monkeypatch, returned):
    call = NativeCall(fd_sizes=(16, returned))
    _native(monkeypatch, call)
    assert resources.sample_self_resources().fd_count == "unavailable"
    assert len(call.calls) == 3


@pytest.mark.parametrize("size", [0, -1])
def test_fd_size_query_failure_is_unavailable(monkeypatch, size):
    call = NativeCall(fd_sizes=(size,))
    _native(monkeypatch, call)
    assert resources.sample_self_resources().fd_count == "unavailable"
    assert len(call.calls) == 2


def test_library_load_failure_preserves_all_unavailable(monkeypatch, capsys):
    def fail(path):
        raise OSError("synthetic error payload")

    monkeypatch.setattr(resources.ctypes, "CDLL", fail)
    sample = resources.sample_self_resources()
    assert dataclasses.astuple(sample)[1:] == ("unavailable",) * 4
    assert capsys.readouterr() == ("", "")


def test_structure_errors_are_not_hidden(monkeypatch):
    def fail(path):
        raise TypeError("bad structure")

    monkeypatch.setattr(resources.ctypes, "CDLL", fail)
    with pytest.raises(TypeError):
        resources.sample_self_resources()


def test_growth_uses_peak_after_warmup_and_same_incarnation():
    supervisor = soak.SoakSupervisor(soak.RunConfig(50, 30, "answer", warmup_seconds=10))
    supervisor.origin = 100.0

    def add(role, incarnation, at, rss, threads, fds, cpu):
        supervisor.report["resources"][role].setdefault(str(incarnation), []).append(
            dataclasses.asdict(resources.ResourceSample(at, rss, threads, fds, cpu))
        )

    add("brain", 1, 109, 9999, 100, 100, 1)
    add("brain", 1, 110, 100, 5, 10, 2)
    add("brain", 1, 115, 170, 9, 12, 4)
    add("brain", 1, 120, 80, 4, 8, 5)
    add("brain", 2, 121, 900, 50, 50, 1)
    add("body", 1, 110, 200, 8, 15, 1)
    add("body", 1, 115, 190, 7, 14, 2)
    add("body", 1, 120, 180, 6, 13, 3)
    supervisor.close()
    assert supervisor.report["resource_growth"] == {
        "brain": {
            "1": dict(rss_growth_bytes=70, thread_growth=4, fd_growth=2, cpu_seconds=3),
            "2": dict(
                rss_growth_bytes="unavailable",
                thread_growth="unavailable",
                fd_growth="unavailable",
                cpu_seconds=0,
            ),
        },
        "body": {"1": dict(rss_growth_bytes=0, thread_growth=0, fd_growth=0, cpu_seconds=2)},
    }


def test_missing_sample_never_becomes_zero():
    samples = [resources.ResourceSample(10 + i * 5, 100, 5, 10, i) for i in range(3)]
    samples[1] = resources.ResourceSample(15, "unavailable", 5, "unavailable", "unavailable")
    assert resources.resource_growth(samples, run_started_monotonic_s=10, warmup_seconds=0) == dict(
        rss_growth_bytes="unavailable",
        thread_growth=0,
        fd_growth="unavailable",
        cpu_seconds="unavailable",
    )
    for count in (0, 1, 2):
        growth = resources.resource_growth(
            samples[:count], run_started_monotonic_s=10, warmup_seconds=0
        )
        assert all(
            growth[name] == "unavailable"
            for name in (
                "rss_growth_bytes",
                "thread_growth",
                "fd_growth",
            )
        )


def limits_document():
    return dict(
        schema_version=1,
        source_baseline_sha256="a" * 64,
        warmup_seconds=30.0,
        resource_sample_interval_seconds=5.0,
        brain=dict(rss_growth_bytes=100, thread_growth=1, fd_growth=2),
        body=dict(rss_growth_bytes=200, thread_growth=2, fd_growth=3),
    )


def test_frozen_limits_digest_is_original_bytes(tmp_path):
    path = tmp_path / "limits.json"
    raw = json.dumps(limits_document(), indent=2).encode() + b"\n"
    path.write_bytes(raw)
    assert soak.load_frozen_limits(path) == (limits_document(), hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize(
    "change",
    [
        {"extra": "payload"},
        {"schema_version": True},
        {"schema_version": 2},
        {"source_baseline_sha256": "A" * 64},
        {"source_baseline_sha256": "a" * 63},
        {"source_baseline_sha256": 1},
        {"warmup_seconds": True},
        {"warmup_seconds": -1},
        {"warmup_seconds": float("nan")},
        {"warmup_seconds": float("inf")},
        {"warmup_seconds": "30"},
        {"resource_sample_interval_seconds": True},
        {"resource_sample_interval_seconds": 4.0},
        {"brain": {}},
        {"body": {"rss_growth_bytes": True, "thread_growth": 1, "fd_growth": 2}},
        {"body": {"rss_growth_bytes": 1.0, "thread_growth": 1, "fd_growth": 2}},
        {"body": {"rss_growth_bytes": -1, "thread_growth": 1, "fd_growth": 2}},
        {"body": {"rss_growth_bytes": 1, "thread_growth": 1, "fd_growth": 2, "text": "payload"}},
    ],
)
def test_invalid_frozen_limits_are_fixed_errors(tmp_path, change):
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(limits_document() | change))
    with pytest.raises(ValueError, match=r"^invalid frozen soak limits$"):
        soak.load_frozen_limits(path)


@pytest.mark.parametrize("raw", [b"{", b"[]", b"null", b"\xff", b"{}"])
def test_malformed_limits_are_fixed_errors(tmp_path, raw):
    path = tmp_path / "limits.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=r"^invalid frozen soak limits$"):
        soak.load_frozen_limits(path)


def test_limits_oserror_is_preserved(tmp_path):
    with pytest.raises(FileNotFoundError):
        soak.load_frozen_limits(tmp_path / "missing.json")


def test_resource_sampler_waits_five_seconds_and_joins():
    spec = soak.WorkerSpec("brain", 1, 0, Path("cert"), Path("run"), "answer")
    events = soak._Events(spec)
    sampler = soak._ResourceSampler(events)

    class Stop:
        def __init__(self):
            self.stopped = False
            self.waits = []
            self.results = iter((False, False, True))

        def wait(self, seconds):
            self.waits.append(seconds)
            return next(self.results)

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

    stop = Stop()
    sampler.stop_requested = stop
    sampler.start()
    sampler.thread.join()
    sampler.close()
    sampler.close()
    assert not sampler.thread.is_alive()
    assert stop.waits == [5.0, 5.0, 5.0]
    samples = []
    while not events.pending.empty():
        samples.append(events.pending.get_nowait())
    assert len(samples) == 4
    assert all(sample.kind == "resource_sample" for sample in samples)
