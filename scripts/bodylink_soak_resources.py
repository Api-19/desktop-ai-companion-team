"""Self-process macOS counters and per-incarnation resource growth."""

from __future__ import annotations

import ctypes
import os
import resource
import time
from dataclasses import dataclass
from typing import Literal

RESOURCE_SAMPLE_INTERVAL_S = 5.0
PROC_PIDTASKINFO = 4
PROC_PIDLISTFDS = 1
Unavailable = Literal["unavailable"]


@dataclass(frozen=True, slots=True)
class ResourceSample:
    at_s: float
    rss_bytes: int | Unavailable
    thread_count: int | Unavailable
    fd_count: int | Unavailable
    cpu_seconds: float | Unavailable


class ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


class ProcFdInfo(ctypes.Structure):
    _fields_ = [("proc_fd", ctypes.c_int32), ("proc_fdtype", ctypes.c_uint32)]


def sample_self_resources() -> ResourceSample:
    at_s = time.monotonic()
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:
        return ResourceSample(at_s, "unavailable", "unavailable", "unavailable", "unavailable")
    pidinfo = library.proc_pidinfo
    pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    pidinfo.restype = ctypes.c_int
    pid = os.getpid()
    task = ProcTaskInfo()
    rss = threads = fds = "unavailable"
    if pidinfo(pid, PROC_PIDTASKINFO, 0, ctypes.byref(task), ctypes.sizeof(task)) == 96:
        rss = task.pti_resident_size
        threads = task.pti_threadnum
    needed = pidinfo(pid, PROC_PIDLISTFDS, 0, None, 0)
    if needed > 0:
        capacity = needed + 256
        buffer = ctypes.create_string_buffer(capacity)
        returned = pidinfo(pid, PROC_PIDLISTFDS, 0, buffer, capacity)
        if 0 < returned < capacity and returned % ctypes.sizeof(ProcFdInfo) == 0:
            fds = returned // ctypes.sizeof(ProcFdInfo)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ResourceSample(at_s, rss, threads, fds, usage.ru_utime + usage.ru_stime)


def resource_growth(
    samples: list[ResourceSample], *, run_started_monotonic_s: float, warmup_seconds: float
) -> dict[str, int | float | Unavailable]:
    remaining = [s for s in samples if s.at_s >= run_started_monotonic_s + warmup_seconds]
    growth: dict[str, int | float | Unavailable] = {}
    for field, key in (
        ("rss_bytes", "rss_growth_bytes"),
        ("thread_count", "thread_growth"),
        ("fd_count", "fd_growth"),
        ("cpu_seconds", "cpu_seconds"),
    ):
        values = [getattr(sample, field) for sample in remaining]
        if not values or "unavailable" in values or (field != "cpu_seconds" and len(values) < 3):
            growth[key] = "unavailable"
        elif field == "cpu_seconds":
            growth[key] = values[-1] - values[0]
        else:
            growth[key] = max(0, max(values) - values[0])
    return growth
