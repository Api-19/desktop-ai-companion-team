"""Offline style-material gate research; standard library only.

Reports contain identifiers, scores and labels, never context/reply text. Missing
app labels count as unknown. Candidate metrics exclude unknowns except the
unknown/all ratio; empty-event rates use events with known labels and their known
candidates. Calibration cutoffs use all dev candidates (otherwise all cal
candidates); fitting uses only cal yes/no labels. Synthetic data never fit the
calibration. Random decisions reuse G's probability estimates for Brier/ECE;
those two metrics diagnose the score mapping rather than the random selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median

# Frozen conflict table: context pattern, candidate reply pattern.
RULE_CONFLICTS = ((r"不想睡|不睡|睡不着|还要赶|熬夜赶", r"早点睡|快睡|去睡|安安|晚安"),)
GROUP_KINDS = {"same_topic_opposite", "diff_topic_same_action", "overlap_wrong_action"}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("invalid arguments")


def norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def norm_lines(lines: list[str]) -> str:
    return norm("\n".join(lines))


def dice(left: str, right: str) -> float:
    """Multiset bigram Dice on already normalized strings."""
    a = Counter(left[i:i + 2] for i in range(len(left) - 1))
    b = Counter(right[i:i + 2] for i in range(len(right) - 1))
    if not a or not b:
        return 0.0
    return 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values()))


def lcs(left: str, right: str) -> int:
    """Length of longest common contiguous substring, O(min(n,m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    longest = 0
    for a in left:
        current = [0] * (len(right) + 1)
        for i, b in enumerate(right, 1):
            if a == b:
                current[i] = previous[i - 1] + 1
                longest = max(longest, current[i])
        previous = current
    return longest


def rule_conflict(context: str, reply: str) -> bool:
    return any(re.search(c, context) and re.search(r, reply) for c, r in RULE_CONFLICTS)


def weak_label(true_reply: str, reply: str) -> str:
    a, b = norm(true_reply), norm(reply)
    d2, longest = dice(a, b), lcs(a, b)
    if len(a) >= 6 and len(b) >= 6 and d2 >= 0.70 and longest >= 4:
        return "P"
    if d2 <= 0.10 and longest < 2:
        return "U"
    return "uncertain"


def _string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(line, str) for line in value)


def load_events(path: Path) -> list[dict]:
    events = []
    seen = set()
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("invalid event")
            eid = event["event_id"]
            if not isinstance(eid, str) or eid in seen:
                raise ValueError("invalid event id")
            seen.add(eid)
            if event["split"] not in ("dev", "cal", "test", "synthetic"):
                raise ValueError("invalid split")
            if not _string_list(event["context"]) or not isinstance(event["candidates"], list):
                raise ValueError("invalid event text")
            for field in ("group", "group_kind"):
                if field in event and not isinstance(event[field], str):
                    raise ValueError("invalid group")
            if "true_reply" in event and event["true_reply"] is not None and not _string_list(
                event["true_reply"],
            ):
                raise ValueError("invalid true reply")
            candidates = set()
            for candidate in event["candidates"]:
                if not isinstance(candidate, dict):
                    raise ValueError("invalid candidate")
                cid = candidate["cand_id"]
                if not isinstance(cid, str) or cid in candidates:
                    raise ValueError("invalid candidate id")
                candidates.add(cid)
                if candidate["source"] not in ("similar", "general"):
                    raise ValueError("invalid source")
                if not _string_list(candidate["c"]) or not _string_list(candidate["r"]):
                    raise ValueError("invalid candidate text")
            labels = event.get("labels", {})
            if not isinstance(labels, dict):
                raise ValueError("invalid labels")
            for cid, label in labels.items():
                if cid not in candidates or not isinstance(label, dict) or label["app"] not in (
                    "yes", "no", "unknown",
                ):
                    raise ValueError("invalid label")
            events.append(event)
    return events


def load_scores(path: Path) -> dict[tuple[str, str], float]:
    scores = {}
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("invalid score record")
            eid, cid, score = record["event_id"], record["cand_id"], record["score"]
            if not isinstance(eid, str) or not isinstance(cid, str):
                raise ValueError("invalid score identifiers")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("invalid score")
            key = (eid, cid)
            if key in scores:
                raise ValueError("duplicate score")
            scores[key] = score
    return scores


def score_candidates(
    events: list[dict], scorer: str, scores: dict[tuple[str, str], float] | None = None,
) -> list[dict]:
    lengths = [len(norm_lines(c["r"])) for e in events for c in e["candidates"]]
    center = median(lengths) if lengths else 0
    rows = []
    for event in events:
        context = norm_lines(event["context"])
        for candidate in event["candidates"]:
            eid, cid = event["event_id"], candidate["cand_id"]
            reply = norm_lines(candidate["r"])
            if scorer == "char":
                score = dice(context, norm_lines(candidate["c"]))
            elif scorer == "length":
                score = -abs(len(reply) - center)
            elif scorer == "vector":
                if scores is None or (eid, cid) not in scores:
                    raise ValueError("missing vector score")
                score = scores[eid, cid]
            else:
                raise ValueError("invalid scorer")
            row = {
                "event_id": eid, "cand_id": cid, "split": event["split"],
                "source": candidate["source"], "score": score,
                "app": event.get("labels", {}).get(cid, {}).get("app", "unknown"),
                "rule_conflict": bool(rule_conflict(context, reply)),
            }
            if event["split"] == "synthetic":
                row.update(group=event["group"], group_kind=event["group_kind"])
            rows.append(row)
    return rows


def calibrate(rows: list[dict]) -> dict:
    dev_scores = [r["score"] for r in rows if r["split"] == "dev"]
    cal_scores = [r["score"] for r in rows if r["split"] == "cal"]
    cutoff_scores = dev_scores or cal_scores
    cutoff = median(cutoff_scores) if cutoff_scores else None
    bins = {name: {"yes": 0, "no": 0, "n": 0, "p_hat": None} for name in ("high", "low")}
    for row in rows:
        if row["split"] == "cal" and row["app"] in ("yes", "no"):
            name = "high" if row["score"] >= cutoff else "low"
            bins[name][row["app"]] += 1
            bins[name]["n"] += 1
    for bucket in bins.values():
        if bucket["n"] >= 5:
            bucket["p_hat"] = (bucket["yes"] + 1) / (bucket["n"] + 2)
    high, low = bins["high"]["p_hat"], bins["low"]["p_hat"]
    status = "ok" if all(b["n"] >= 5 for b in bins.values()) else "insufficient_data"
    if high is not None and low is not None and high < low:
        status = "non_monotonic"
    return {
        "cutoff": cutoff, "cutoff_source": "dev" if dev_scores else "cal",
        "bins": bins, "calibration_status": status,
    }


def apply_gate(rows: list[dict], calibration: dict, threshold: float = 0.80) -> list[dict]:
    decided = []
    for row in rows:
        cutoff = calibration["cutoff"]
        name = "high" if cutoff is not None and row["score"] >= cutoff else "low"
        p_hat = calibration["bins"][name]["p_hat"]
        keep = (
            calibration["calibration_status"] != "non_monotonic"
            and not row["rule_conflict"] and p_hat is not None and p_hat >= threshold
        )
        decided.append({**row, "bin": name, "p_hat": p_hat, "keep": keep})
    return decided


def rate(numerator: int, denominator: int) -> dict:
    if denominator == 0:
        value, interval = None, None
    else:
        value = numerator / denominator
        z = 1.959963984540054
        factor = 1 + z * z / denominator
        center = (value + z * z / (2 * denominator)) / factor
        radius = z * math.sqrt(
            value * (1 - value) / denominator + z * z / (4 * denominator * denominator),
        ) / factor
        interval = [max(0, center - radius), min(1, center + radius)]
    return {"value": value, "numerator": numerator, "denominator": denominator,
            "wilson95": interval}


def evaluate(rows: list[dict]) -> dict:
    test = [r for r in rows if r["split"] == "test"]
    known = [r for r in test if r["app"] in ("yes", "no")]
    confusion = {name: 0 for name in ("TP", "FP", "TN", "FN")}
    events = defaultdict(list)
    for row in known:
        key = ("TP" if row["app"] == "yes" else "FP") if row["keep"] else (
            "FN" if row["app"] == "yes" else "TN"
        )
        confusion[key] += 1
        events[row["event_id"]].append(row)
    tp, fp, tn, fn = (confusion[name] for name in ("TP", "FP", "TN", "FN"))
    rates = {
        "false_discovery_rate": rate(fp, tp + fp),
        "inapplicable_pass_rate": rate(fp, fp + tn),
        "miss_rate": rate(fn, tp + fn),
        "retention_rate": rate(tp + fp, len(known)),
        "empty_event_rate": rate(sum(not any(r["keep"] for r in e) for e in events.values()),
                                 len(events)),
        "unknown_rate": rate(len(test) - len(known), len(test)),
        "rule_rejection_rate": rate(sum(r["rule_conflict"] for r in known), len(known)),
    }
    estimated = [r for r in known if r["p_hat"] is not None]
    n = len(estimated)
    brier = fmean((r["p_hat"] - (r["app"] == "yes")) ** 2 for r in estimated) if n else None
    ece_bins = {}
    weighted_error = 0.0
    for name in ("high", "low"):
        bucket = [r for r in estimated if r["bin"] == name]
        count = len(bucket)
        probability = fmean(r["p_hat"] for r in bucket) if count else None
        observed = sum(r["app"] == "yes" for r in bucket) / count if count else None
        if count:
            weighted_error += count * abs(probability - observed)
        ece_bins[name] = {"n": count, "p_hat": probability, "observed_rate": observed}
    return {
        "confusion": confusion, "rates": rates,
        "brier": {"value": brier, "n": n, "excluded": len(known) - n},
        "ece": {"value": weighted_error / n if n else None, "n": n,
                "excluded": len(known) - n, "bins": ece_bins},
    }


def evaluate_by_source(rows: list[dict]) -> dict:
    return {source: evaluate([r for r in rows if r["source"] == source])
            for source in ("similar", "general")}


def random_decisions(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    events = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] == "test":
            events[row["event_id"]].append(i)
    selected = set()
    for indices in events.values():
        count = sum(rows[i]["keep"] for i in indices)
        eligible = [i for i in indices if not rows[i]["rule_conflict"]]
        selected.update(rng.sample(eligible, count))
    return [{**row, "keep": i in selected} for i, row in enumerate(rows)]


def _summarize(values: list) -> dict | list:
    first = values[0]
    if isinstance(first, dict):
        return {key: _summarize([value[key] for value in values]) for key in first}
    # Wilson intervals can be null when no denominator exists for a seed.
    available = [value for value in values if value is not None]
    if available and isinstance(available[0], list):
        return [_summarize([value[i] for value in available]) for i in range(len(available[0]))]
    return {"mean": fmean(available) if available else None,
            "min": min(available) if available else None,
            "max": max(available) if available else None, "n": len(available)}


def random_baseline(rows: list[dict]) -> dict:
    runs = [random_decisions(rows, seed) for seed in range(20)]
    return {
        "seeds": list(range(20)), "probability_basis": "gate_calibration",
        "metrics": _summarize([evaluate(run) for run in runs]),
        "by_source": _summarize([evaluate_by_source(run) for run in runs]),
    }


def synthetic_diagnostics(rows: list[dict]) -> dict:
    synthetic = [r for r in rows if r["split"] == "synthetic"]
    threshold = median(r["score"] for r in synthetic) if synthetic else None
    groups = defaultdict(list)
    for row in synthetic:
        groups[row["group"]].append(row)
    results = []
    for group, members in groups.items():
        kinds = {r["group_kind"] for r in members}
        if len(members) != 2 or len(kinds) != 1 or not kinds <= GROUP_KINDS:
            raise ValueError("invalid synthetic group")
        scenes = {r["event_id"].rsplit("-", 1)[-1]: r for r in members}
        if set(scenes) != {"s1", "s2"}:
            raise ValueError("invalid synthetic scene ids")
        keep = {s: r["score"] >= threshold and not r["rule_conflict"] for s, r in scenes.items()}
        kind = members[0]["group_kind"]
        if kind == "same_topic_opposite":
            correct = keep["s1"] and not keep["s2"]
        elif kind == "diff_topic_same_action":
            correct = keep["s1"] and keep["s2"]
        else:
            correct = not keep["s1"] and not keep["s2"]
        results.append({"group": group, "group_kind": kind,
                        "s1_keep": keep["s1"], "s2_keep": keep["s2"], "correct": correct})
    correct_groups = sum(r["correct"] for r in results)
    return {"threshold": threshold, "correct_groups": correct_groups,
            "total_groups": len(results), "passed": correct_groups >= 10, "groups": results}


def analyze(
    events: list[dict], scorer: str, threshold: float = 0.80,
    scores: dict[tuple[str, str], float] | None = None,
) -> dict:
    rows = score_candidates(events, scorer, scores)
    calibration = calibrate(rows)
    decisions = apply_gate(rows, calibration, threshold)
    weak = []
    counts = {"P": 0, "U": 0, "uncertain": 0}
    for event in events:
        if event.get("true_reply") is not None:
            for candidate in event["candidates"]:
                label = weak_label(norm_lines(event["true_reply"]), norm_lines(candidate["r"]))
                counts[label] += 1
                weak.append({"event_id": event["event_id"], "cand_id": candidate["cand_id"],
                             "Y_use": label})
    scorers = ["char", "length"] + (["vector"] if scores is not None else [])
    return {
        "status": "ok", "scorer": scorer, "threshold": threshold,
        "events": len(events), "candidates": len(rows), "calibration": calibration,
        "decisions": decisions, "metrics": evaluate(decisions),
        "by_source": evaluate_by_source(decisions), "random_baseline": random_baseline(decisions),
        "synthetic": {name: synthetic_diagnostics(score_candidates(events, name, scores))
                      for name in scorers},
        "weak_labels": {"counts": counts, "candidates": weak},
    }


def write_output(path: Path, text: str) -> None:
    """Atomic UTF-8 output in an existing directory, including secure overwrites."""
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as file:
        temporary = Path(file.name)
        try:
            os.fchmod(file.fileno(), 0o600)
            file.write(text)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _Parser()
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--scorer", choices=("char", "length", "vector"), required=True)
        parser.add_argument("--scores-file", type=Path)
        parser.add_argument("--threshold", type=float, default=0.80)
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args(argv)
        if not args.out.parent.is_dir() or not math.isfinite(args.threshold) or not (
            0 <= args.threshold <= 1
        ):
            raise ValueError("invalid output or threshold")
        if args.scorer == "vector" and args.scores_file is None:
            raise ValueError("vector scorer requires a scores file")
        events = load_events(args.input)
        scores = load_scores(args.scores_file) if args.scores_file else None
        report = analyze(events, args.scorer, args.threshold, scores)
        write_output(args.out, json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception:
        print('{"status":"failed"}')
        return 1
    print(json.dumps({"status": "ok", "events": report["events"],
                      "candidates": report["candidates"],
                      "calibration_status": report["calibration"]["calibration_status"]},
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
