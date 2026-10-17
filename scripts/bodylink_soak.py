"""Short synthetic bodylink run; exit zero is tool completion, never soak acceptance."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
from pathlib import Path

if __package__:
    from .bodylink_soak_support import RunConfig, run_soak, write_report
else:
    from bodylink_soak_support import RunConfig, run_soak, write_report

SOAK_ERROR = "bodylink soak failed"
SOAK_ARGUMENT_ERROR = "invalid bodylink soak arguments"
STEADY_DURATION_S = 7200.0


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(SOAK_ARGUMENT_ERROR)


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(SOAK_ARGUMENT_ERROR)
    return number


def _nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(SOAK_ARGUMENT_ERROR)
    return number


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _Parser(add_help=False, allow_abbrev=False)
        parser.add_argument("--duration-seconds", type=_positive)
        parser.add_argument("--turn-interval-seconds", default=30.0, type=_positive)
        parser.add_argument("--fixed-reply", required=True)
        parser.add_argument(
            "--scenario", default="steady", choices=("steady", "faults", "baseline")
        )
        parser.add_argument("--warmup-seconds", default=30.0, type=_nonnegative)
        parser.add_argument("--limits", type=Path)
        parser.add_argument("--output", required=True, type=Path)
        args = parser.parse_args(argv)
        if (
            sys.platform != "darwin"
            or not args.fixed_reply.strip()
            or not args.output.parent.is_dir()
            or (args.limits is not None and args.scenario != "steady")
            or (
                args.scenario == "baseline"
                and (
                    args.duration_seconds is None
                    or args.duration_seconds < args.warmup_seconds + 15
                )
            )
        ):
            raise ValueError(SOAK_ARGUMENT_ERROR)
        if args.duration_seconds is None:
            args.duration_seconds = STEADY_DURATION_S
    except (ValueError, OSError, argparse.ArgumentError):
        print(SOAK_ARGUMENT_ERROR, file=sys.stderr)
        return 2

    stop = threading.Event()
    previous = {}

    def interrupted(signum: int, frame: object) -> None:
        stop.set()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        try:
            report = run_soak(
                RunConfig(
                    args.duration_seconds,
                    args.turn_interval_seconds,
                    args.fixed_reply,
                    args.scenario,
                    args.warmup_seconds,
                    args.limits,
                ),
                stop_requested=stop,
            )
        except ValueError:
            print(SOAK_ARGUMENT_ERROR, file=sys.stderr)
            return 2
        write_report(args.output, report)
        if report["run_complete"]:
            return 0
    except (OSError, ValueError):
        pass
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(SOAK_ERROR, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
