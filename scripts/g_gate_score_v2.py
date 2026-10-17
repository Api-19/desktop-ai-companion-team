"""Score any gate arm's predictions against a v2 answer key (D18 §4.3).

Scene decision = the arm's ``c1`` decision; a missing row is "missing" (an
error and a technical failure). Any technical failure makes the scene wrong.
Groups are event ids without the trailing ``-s1``/``-s2``.
The output JSON holds only event ids and numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DECIDABLE = ("T1", "T2", "T3", "T4", "T5", "T6")
APPS = ("yes", "no", "unknown")
DECISIONS = ("keep", "drop", "unknown")
EXPECTED = {"yes": "keep", "no": "drop", "unknown": "unknown"}


def group_id(event_id: str) -> str:
    return re.sub(r"-s[12]$", "", event_id)


def score(scenes: dict, key: dict, pred: list[dict]) -> dict:
    decisions, failed = {}, set()
    for row in pred:
        if row["cand_id"] != "c1":
            continue
        if row["event_id"] in decisions:
            raise ValueError(f"duplicate c1 prediction: {row['event_id']}")
        decisions[row["event_id"]] = row["decision"]
        if row["technical_failure"]:
            failed.add(row["event_id"])

    scene_ok, groups, group_type = {}, {}, {}
    confusion = {app: dict.fromkeys(DECISIONS, 0) for app in APPS}
    counts = dict.fromkeys(("positives_kept", "negatives_kept", "insufficient_kept",
                            "permission_negatives_kept", "unknown_count", "missing"), 0)
    for eid, label in key.items():
        app, decision = label["app"], decisions.get(eid, "missing")
        kept = decision == "keep"
        scene_ok[eid] = int(decision == EXPECTED[app] and eid not in failed)
        gid = group_id(eid)
        groups[gid] = groups.get(gid, 1) & scene_ok[eid]
        group_type[gid] = label["type"]
        if decision == "missing":
            counts["missing"] += 1
            failed.add(eid)
        else:
            confusion[app][decision] += 1
        counts["unknown_count"] += decision == "unknown"
        if label["type"] in DECIDABLE:
            counts["positives_kept"] += app == "yes" and kept
            counts["negatives_kept"] += app == "no" and kept
        counts["insufficient_kept"] += app == "unknown" and kept
        permission = scenes[eid].get("permission")
        counts["permission_negatives_kept"] += (
            app == "no" and permission is not None and kept
        )

    by_type = {t: sum(ok for g, ok in groups.items() if group_type[g] == t) for t in DECIDABLE}
    result = {
        "decidable_groups_correct": sum(by_type.values()),
        "type_groups_correct": by_type,
        "insufficient_groups_correct": sum(
            ok for g, ok in groups.items() if group_type[g] == "T7"
        ),
        **counts,
        "technical_failures": len(failed),
        "confusion": confusion,
    }
    checks = {
        "decidable_groups_correct": result["decidable_groups_correct"] >= 32,
        "type_groups_correct": all(n >= 5 for n in by_type.values()),
        "positives_kept": result["positives_kept"] >= 31,
        "negatives_kept": result["negatives_kept"] <= 3,
        "insufficient_groups_correct": result["insufficient_groups_correct"] >= 5,
        "insufficient_kept": result["insufficient_kept"] == 0,
        "permission_negatives_kept": result["permission_negatives_kept"] == 0,
    }
    result.update(passed=all(checks.values()), checks=checks, groups=groups, scenes=scene_ok)
    return result


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    scenes = {scene["event_id"]: scene for scene in _jsonl(args.scenes)}
    key = json.loads(args.key.read_text(encoding="utf-8-sig"))
    result = score(scenes, key, _jsonl(args.pred))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("groups", "scenes", "confusion")}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
