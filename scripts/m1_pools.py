"""Build private M1 event pools offline; emit only status and counts.

Events join same-conversation snippets by time, normalized inputs/partner text,
or consecutive role-labelled turns. Exact inputs require 12 characters; shared
turn pairs require 20. Exposure takes precedence over sealing the last stage's
remaining tail. Unselected calibration/test events become M2 reserves.
Feasibility-only manifests select no events; supported tiers keep all training
events and sample calibration/test events by stage. A count audits removed
short-input/overlap edges across the final pools without emitting their IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import combinations, pairwise
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from companion.privacy.paired_snippets import PairedSnippet, read_snippets
from companion.runtime.settings import load_settings

_BEIJING = timezone(timedelta(hours=8))
_POOLS = ("exposed", "m2_sealed", "train", "calib", "test", "m2_reserve")
_TIERS = (("300/60/90", 60, 90), ("200/40/60", 40, 60))


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("invalid arguments")


@dataclass(frozen=True)
class _Stage:
    start: date
    end: date
    label: str


@dataclass
class _Event:
    event_id: str
    stage: str
    start: datetime
    end: datetime
    snippet_ids: list[str]
    pool: str = "train"
    selected: bool = False

    def manifest(self) -> dict:
        return {
            "event_id": self.event_id, "stage": self.stage, "pool": self.pool,
            "start": self.start.astimezone(_BEIJING).isoformat(),
            "end": self.end.astimezone(_BEIJING).isoformat(),
            "snippet_ids": self.snippet_ids, "predict_snippet_id": self.event_id,
            "selected": self.selected,
        }


def _read_date(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("invalid stage date")
    return date.fromisoformat(value)


def _read_stages(path: Path) -> list[_Stage]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = payload["stages"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("timeline requires stages")
    stages = []
    for entry in entries:
        stage = _Stage(_read_date(entry["from"]), _read_date(entry["to"]), entry["label"])
        if stage.start > stage.end or not isinstance(stage.label, str):
            raise ValueError("invalid stage")
        stages.append(stage)
    return stages


def _stage_for(timestamp: datetime, stages: list[_Stage]) -> str:
    day = timestamp.astimezone(_BEIJING).date()
    for stage in stages:
        if stage.start <= day <= stage.end:
            return stage.label
    raise ValueError("snippet date outside timeline")


def _normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def _bigrams(text: str) -> set[str]:
    return {text[index:index + 2] for index in range(len(text) - 1)}


@dataclass(frozen=True)
class _Text:
    x: str
    p: str
    x_bigrams: set[str]
    p_bigrams: set[str]
    overlaps: set[tuple[str, str]]


def _text(snippet: PairedSnippet) -> _Text:
    input_lines = [*snippet.context_turns, *("她:" + turn for turn in snippet.partner_turns)]
    x = _normalize("\n".join(input_lines))
    p = _normalize("".join(snippet.partner_turns))
    turns = [_normalize(line) for line in (
        *input_lines, *("他:" + turn for turn in snippet.owner_reply),
    )]
    # Keep the old >=12 overlaps for the audit; event edges enforce >=20 below.
    return _Text(x, p, _bigrams(x), _bigrams(p), {
        (left, right) for left, right in pairwise(turns) if len(left) + len(right) >= 12
    })


def _similar(left: str, right: str, a: set[str], b: set[str]) -> bool:
    return len(left) >= 12 and len(right) >= 12 and 5 * len(a & b) >= 4 * len(a | b)


def _connected(left: _Text, right: _Text) -> bool:
    return (
        _similar(left.x, right.x, left.x_bigrams, right.x_bigrams)
        or _similar(left.p, right.p, left.p_bigrams, right.p_bigrams)
        or any(len(a) + len(b) >= 20 for a, b in left.overlaps & right.overlaps)
    )


def _events(snippets: list[PairedSnippet], stages: list[_Stage]) -> list[_Event]:
    ordered = sorted(snippets, key=lambda item: (item.started_at, item.snippet_id))
    labels = [_stage_for(item.started_at, stages) for item in ordered]
    parents = list(range(len(ordered)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        parents[root(right)] = root(left)

    conversations: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(ordered):
        conversations[item.conversation_id].append(index)
    texts = [_text(item) for item in ordered]
    for indices in conversations.values():
        for left, right in pairwise(indices):
            if ordered[right].started_at - ordered[left].started_at <= timedelta(minutes=30):
                join(left, right)
        for left, right in combinations(indices, 2):
            if _connected(texts[left], texts[right]):
                join(left, right)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ordered)):
        components[root(index)].append(index)
    events = []
    for indices in components.values():
        first, last = ordered[indices[0]], ordered[indices[-1]]
        events.append(_Event(
            first.snippet_id, labels[indices[0]], first.started_at, last.started_at,
            [ordered[index].snippet_id for index in indices],
        ))
    return sorted(events, key=lambda event: (event.start, event.event_id))


def _assign_pools(
    events: list[_Event], exposed: set[str], labels: list[str], last_label: str,
) -> None:
    for event in events:
        if exposed.intersection(event.snippet_ids):
            event.pool = "exposed"
    tail = [event for event in events if event.pool == "train" and event.stage == last_label]
    if tail:
        for event in tail[-math.ceil(len(tail) / 5):]:
            event.pool = "m2_sealed"
    for label in labels:
        remaining = [event for event in events if event.stage == label and event.pool == "train"]
        count = len(remaining)
        if count < 3:
            continue
        calib, test = count * 2 // 15, count * 3 // 15
        train = count - calib - test
        for index, event in enumerate(remaining):
            event.pool = "train" if index < train else "calib" if index < train + calib else "test"


def _stage_quotas(sizes: dict[str, int], quota: int) -> dict[str, int] | None:
    """Hamilton allocation with a lower bound of one per nonempty stage.

    Remainder ties follow timeline order. If enforcing the lower bound exceeds
    the quota, remove slots from the smallest remainders while preserving one.
    """
    sizes = {label: size for label, size in sizes.items() if size}
    total = sum(sizes.values())
    if total < quota or len(sizes) > quota:
        return None
    assigned = {label: max(1, quota * size // total) for label, size in sizes.items()}
    remainders = {label: quota * size % total for label, size in sizes.items()}
    difference = quota - sum(assigned.values())
    if difference > 0:
        for label in sorted(sizes, key=lambda label: -remainders[label])[:difference]:
            assigned[label] += 1
    while difference < 0:
        for label in sorted(sizes, key=lambda label: remainders[label]):
            if assigned[label] > 1:
                assigned[label] -= 1
                difference += 1
                if difference == 0:
                    break
    return assigned


def _select(events: list[_Event], labels: list[str], seed: int) -> str:
    count = sum(event.pool in {"train", "calib", "test"} for event in events)
    if count < 300:
        return "feasibility_only"
    pools = {
        pool: {label: [event for event in events if event.pool == pool and event.stage == label]
               for label in labels}
        for pool in ("calib", "test")
    }
    for tier, calib, test in _TIERS[0 if count >= 450 else 1:]:
        quotas = {
            pool: _stage_quotas({label: len(items) for label, items in pools[pool].items()}, quota)
            for pool, quota in (("calib", calib), ("test", test))
        }
        if any(allocation is None for allocation in quotas.values()):
            continue
        generator = random.Random(seed)
        for event in events:
            event.selected = event.pool == "train"
        for pool, allocation in quotas.items():
            for label, quota in allocation.items():
                for event in generator.sample(pools[pool][label], quota):
                    event.selected = True
        for event in events:
            if event.pool in {"calib", "test"} and not event.selected:
                event.pool = "m2_reserve"
        return tier
    return "feasibility_only"


def _relaxed_cross_pool_links(snippets: list[PairedSnippet], events: list[_Event]) -> int:
    pools = {snippet_id: event.pool for event in events for snippet_id in event.snippet_ids}
    conversations: dict[str, list[tuple[str, _Text]]] = defaultdict(list)
    for snippet in snippets:
        conversations[snippet.conversation_id].append((snippet.snippet_id, _text(snippet)))
    count = 0
    for items in conversations.values():
        for (left_id, left), (right_id, right) in combinations(items, 2):
            if pools[left_id] == pools[right_id]:
                continue
            old_edge = left.x == right.x or bool(left.overlaps & right.overlaps)
            if old_edge and not _connected(left, right):
                count += 1
    return count


def _write_manifest(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as file:
        temporary = Path(file.name)
        try:
            os.fchmod(file.fileno(), 0o600)
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build(args: argparse.Namespace) -> dict:
    snippets: dict[str, PairedSnippet] = {}
    for path in (args.library, args.holdout):
        for snippet in read_snippets(path):
            snippets.setdefault(snippet.snippet_id, snippet)
    exposed = {snippet.snippet_id for path in args.exposed for snippet in read_snippets(path)}
    stages = _read_stages(args.timeline)
    labels = list(dict.fromkeys(stage.label for stage in stages))
    events = _events(list(snippets.values()), stages)
    _assign_pools(events, exposed, labels, stages[-1].label)
    tier = _select(events, labels, args.seed)
    relaxed_cross_pool_links = _relaxed_cross_pool_links(list(snippets.values()), events)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.out_dir / "m1_pools.json", {
        "version": 1, "seed": args.seed, "tier": tier,
        "events": [event.manifest() for event in events],
    })
    return {
        "status": "ok", "snippets": len(snippets), "events": len(events), "tier": tier,
        "exposed_events": sum(event.pool == "exposed" for event in events),
        "exposed_snippets_matched": len(exposed.intersection(snippets)),
        "relaxed_cross_pool_links": relaxed_cross_pool_links,
        "pool_counts": {
            pool: {label: sum(event.pool == pool and event.stage == label for event in events)
                   for label in labels}
            for pool in _POOLS
        },
        "selected": {pool: sum(event.pool == pool and event.selected for event in events)
                     for pool in ("train", "calib", "test")},
    }


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _Parser()
        for option in ("library", "holdout", "timeline", "out-dir"):
            parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument("--exposed", type=Path, action="append", default=[])
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args(argv)
        private_root = load_settings(Path.cwd()).private_data_dir.resolve()
        args.out_dir = args.out_dir.resolve()
        if args.out_dir == private_root or not args.out_dir.is_relative_to(private_root):
            print('{"status":"out_dir_not_private"}')
            return 2
        result = _build(args)
    except Exception:
        # Never disclose input values or exception details at this CLI boundary.
        print('{"status":"failed"}')
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
