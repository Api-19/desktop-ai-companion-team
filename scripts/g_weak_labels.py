# ruff: noqa: RUF001
"""Gate weak labels from public LiveChat comment→reply pairs (offline research).

Subcommands (all outputs live under ``--root``, default ``~/.rcpem/data/weak``):

- ``split``: merge the three RawDialogueData files, normalise, drop exact
  (comment, reply) duplicates, split streamers 60/20/20 → ``livechat_split.json``.
- ``sample``: filter train-streamer pairs and draw ``--n`` sources → ``<out>/sources.jsonl``.
- ``cped-sources``: extract turns from official CPED CSV splits.
- ``import``: validate external generations against sources and record prompt provenance.
- ``generate``: one local-model call per source builds a "yes" rewrite and a
  "no" minimal edit, validated in code → ``<out>/pairs.jsonl``.
- ``audit``: blind review items (no labels) plus a separate key.

The backend is injected; only ``main`` constructs ``MlxBackend``. Run real
inference with ``~/.rcpem/venv/bin/python``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

if __package__:
    from . import g_gate_llm as llm
else:
    import g_gate_llm as llm

SEED = 20261016
RAW_FILES = tuple(f"RawDialogueData__{part}_data.jsonl" for part in ("train", "dev", "test"))
DEFAULT_ROOT = Path.home() / ".rcpem" / "data" / "weak"
DEFAULT_RAW = Path.home() / ".rcpem" / "data" / "livechat" / "jsonl"
DEFAULT_MODEL = str(
    Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit"
    / "snapshots/c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"
)
PER_SPEAKER = 5
KINDS = ("paraphrase", "cross_topic")
EDITS = ("state", "need", "person")
MAX_X = 60
MAX_TOKENS = 200
PROMPT_VERSION = "weak-v1"

SYSTEM = (
    "你在为“历史回复能不能拿来回应新消息”的判断模型制作训练数据。"
    "给你一条直播间观众评论 c，以及主播当时对它的回复 r。请写两条新的观众评论。\n"
    "pos：一条新评论 x，主播原来的回复 r 放在 x 下面仍然是自然、说得通的回应，"
    "回应方式和原来同类。"
    "可以是同一件事换个说法（kind 写 paraphrase），也可以换一个话题（kind 写 cross_topic）。"
    "r 里默认的情况（事情做没做完、观众想要什么、说的是谁）在 x 下必须仍然成立，不能让 r 说错事。\n"
    "neg：只改动 c 里的一处得到 x，让 r 放在 x 下面变得不合适。"
    "只能改下面三种之一，edit 写对应的词："
    "state 事情的完成状态（例如“要去”改成“已经去过了”）；"
    "need 观众的需求或要求（例如“想听”改成“别唱了”）；"
    "person 涉及的人（例如“我”改成“我妈”）。"
    "其余的字尽量保持不变；不能只改语气，也不能只换同义词。\n"
    "两条 x 都不超过 40 字，不能和 c 或 r 一模一样。\n"
    "示例一：\n"
    "c：主播今天晚上唱歌吗\nr：唱呀，等下就开唱\n"
    '输出：{"pos":{"x":"今晚还有歌听吗","kind":"paraphrase"},'
    '"neg":{"x":"主播今天晚上别唱歌了","edit":"need"}}\n'
    "示例二：\n"
    "c：我明天要去考驾照了\nr：加油，肯定能过\n"
    '输出：{"pos":{"x":"下周我要去面试了","kind":"cross_topic"},'
    '"neg":{"x":"我昨天考完驾照了","edit":"state"}}\n'
    "只输出一行紧凑的 JSON 对象，不要其他文字，格式："
    '{"pos":{"x":"……","kind":"paraphrase|cross_topic"},"neg":{"x":"……","edit":"state|need|person"}}'
)
USER_TEMPLATE = "c：{c}\nr：{r}"
PROMPT_SHA256 = hashlib.sha256((SYSTEM + "\n" + USER_TEMPLATE).encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ---------- split ----------

def load_pairs(raw_dir: Path) -> tuple[list[tuple[str, str, str]], int]:
    """Merged, normalised, (c, r)-deduplicated pairs in file order, plus the raw count."""
    seen, pairs, raw = set(), [], 0
    for name in RAW_FILES:
        for streamer, c, r in read_jsonl(raw_dir / name):
            raw += 1
            item = (norm(streamer), norm(c), norm(r))
            if item[1:] in seen:
                continue
            seen.add(item[1:])
            pairs.append(item)
    return pairs, raw


def split_streamers(streamers) -> dict[str, list[str]]:
    order = sorted(set(streamers))
    random.Random(SEED).shuffle(order)
    n = len(order)
    n_train, n_dev = n * 6 // 10, n * 2 // 10
    return {"train": order[:n_train], "dev": order[n_train:n_train + n_dev],
            "test": order[n_train + n_dev:]}


def run_split(raw_dir: Path, root: Path) -> dict:
    pairs, raw = load_pairs(raw_dir)
    counts = Counter(s for s, _, _ in pairs)
    parts = split_streamers(counts)
    result = {"seed": SEED, **parts,
              "raw_pairs": raw, "dedup_pairs": len(pairs),
              "pairs_per_split": {k: sum(counts[s] for s in v) for k, v in parts.items()},
              "pairs_per_speaker": dict(sorted(counts.items()))}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "livechat_split.json", result)
    return result


# ---------- sample ----------

def char_dice(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a) + len(b))


def strip_echo(c: str, r: str) -> str:
    """Drop a fuzzy leading echo and join the remaining nonempty clauses."""
    clauses = [part for part in re.split(r"[。！？!?，,…~～\s]+", r) if part]
    if clauses and char_dice(c, clauses[0]) >= 0.5:
        clauses.pop(0)
    return "，".join(clauses)


def reject_reason(c: str, r: str) -> str | None:
    if not 6 <= len(c) <= 40:
        return "c_length"
    if not 4 <= len(r) <= 60:
        return "r_length"
    if any(token in (c + r).lower() for token in ("http", "www")):
        return "url"
    if char_dice(c, r) >= 0.8:
        return "dice"
    if len(strip_echo(c, r)) < 4:
        return "echo_remainder"
    return None


def sample_sources(pairs, streamers: set[str], n: int, exclude: list[Path] = (),
                   id_prefix: str = "", split_name: str = "train") -> tuple[list[dict], dict]:
    excluded = set()
    for path in exclude:
        excluded.update((norm(row["c"]), norm(row["r"])) for row in read_jsonl(path))
    reasons, eligible, split_pairs = Counter(), [], 0
    for streamer, c, r in pairs:
        if streamer not in streamers:
            continue
        split_pairs += 1
        reason = reject_reason(c, r)
        if reason:
            reasons[reason] += 1
        elif (norm(c), norm(strip_echo(c, r))) in excluded:
            reasons["excluded"] += 1
        else:
            eligible.append((streamer, c, r))
    random.Random(SEED).shuffle(eligible)
    taken, chosen = Counter(), []
    for streamer, c, r in eligible:
        if len(chosen) == n:
            break
        if taken[streamer] < PER_SPEAKER:
            taken[streamer] += 1
            chosen.append({"src_id": f"{id_prefix}src{len(chosen):05d}", "speaker": streamer,
                           "source": "livechat", "show": None, "split": split_name,
                           "c": c, "r": strip_echo(c, r), "r_raw": r})
    stats = {"split_pairs": split_pairs, "rejected": dict(reasons),
             "eligible_pairs": len(eligible), "sampled": len(chosen),
             "speakers_sampled": len(taken)}
    return chosen, stats


def run_sample(raw_dir: Path, root: Path, out: Path, n: int,
               exclude: list[Path] = (), id_prefix: str = "",
               split_name: str = "train") -> dict:
    split = json.loads((root / "livechat_split.json").read_text(encoding="utf-8"))
    pairs, _ = load_pairs(raw_dir)
    sources, stats = sample_sources(pairs, set(split[split_name]), n, exclude, id_prefix,
                                    split_name)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sources.jsonl", sources)
    write_json(out / "sample_summary.json", stats)
    return stats


# ---------- CPED sources ----------

def read_cped(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def id_order(value: str) -> tuple:
    """Sort numeric ID components numerically, including IDs with prefixes."""
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", value))


def run_cped_sources(csv_path: Path, all_csv: list[Path], split: str,
                     n: int, out: Path, exclude: list[Path] = (),
                     id_prefix: str = "") -> dict:
    excluded = set()
    for path in exclude:
        excluded.update((norm(row["c"]), norm(row["r"]))
                        for row in read_jsonl(path))
    appearances = Counter()
    for path in all_csv:
        appearances.update({row["Speaker"] for row in read_cped(path)})
    shared = {speaker for speaker, count in appearances.items() if count >= 2}
    rows = sorted(read_cped(csv_path),
                  key=lambda row: (id_order(row["Dialogue_ID"]),
                                   id_order(row["Utterance_ID"])))
    dialogues = {}
    for row in rows:
        dialogues.setdefault((row["TV_ID"], row["Dialogue_ID"]), []).append(row)
    eligible, reasons = [], Counter()
    for (show, _), dialogue in dialogues.items():
        turns = []
        for row in dialogue:
            if not turns or turns[-1][0]["Speaker"] != row["Speaker"]:
                turns.append([])
            turns[-1].append(row)
        for a, b in pairwise(turns):
            c = "／".join(row["Utterance"] for row in a[-2:])
            r = "／".join(row["Utterance"] for row in b[:3])
            if a[0]["Speaker"] in shared or b[0]["Speaker"] in shared:
                reason = "shared_speaker"
            elif not 4 <= len(c) <= 60:
                reason = "c_length"
            elif not 4 <= len(r) <= 80:
                reason = "r_length"
            elif all(row["DA"] in {"greeting", "acknowledge"} for row in b):
                reason = "da_trivial"
            elif any(token in (c + r).lower() for token in ("http", "www")):
                reason = "url"
            elif (norm(c), norm(r)) in excluded:
                reason = "excluded"
            else:
                reason = None
            if reason:
                reasons[reason] += 1
            else:
                eligible.append({"source": "cped", "split": split, "show": show,
                                 "speaker": b[0]["Speaker"], "c": c, "r": r})
    random.Random(SEED).shuffle(eligible)
    chosen, taken = [], Counter()
    for source in eligible:
        if len(chosen) == n:
            break
        if taken[source["speaker"]] < PER_SPEAKER:
            taken[source["speaker"]] += 1
            chosen.append({"src_id": f"{id_prefix}cped-{split}-{len(chosen):05d}", **source})
    stats = {"rejected": dict(reasons), "eligible_pairs": len(eligible),
             "sampled": len(chosen), "speakers_sampled": len(taken),
             "shows_sampled": len({source["show"] for source in chosen})}
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sources.jsonl", chosen)
    write_json(out / "sample_summary.json", stats)
    return stats


# ---------- generate ----------

def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def check_common_side(side, field: str, values: tuple[str, ...], c: str,
                      r: str) -> tuple[str | None, str | None]:
    """Return (x, None) when valid, else (None, reason)."""
    if not isinstance(side, dict) or "x" not in side or field not in side:
        return None, "missing_key"
    if side[field] not in values:
        return None, f"bad_{field}"
    x = side["x"]
    if not isinstance(x, str) or not x.strip():
        return None, "empty_x"
    x = x.strip()
    if len(x) > MAX_X:
        return None, "x_too_long"
    if x == c:
        return None, "x_equals_c"
    if x == r:
        return None, "x_equals_r"
    return x, None


def check_side(side, field: str, values: tuple[str, ...], c: str, r: str,
               neg: bool) -> tuple[str | None, str | None]:
    x, reason = check_common_side(side, field, values, c, r)
    if reason:
        return None, reason
    if neg and x.startswith(c):
        return None, "append_only"
    if not neg and char_dice(x, c) >= 0.9:
        return None, "near_copy"
    if neg and edit_distance(x, c) > max(4, 0.5 * len(c)):
        return None, "edit_too_large"
    return x, None


def validated_pairs(src: dict, data: dict, gen_model: str, prompt_version: str,
                    prompt_sha256: str, reasons: dict,
                    by_variant: dict | None = None) -> list[dict]:
    rows = []
    v3 = "pos_near" in data
    sides = (("yes", "pos_near", "kind", ("near",)),
             ("yes", "pos_far", "kind", KINDS),
             ("no", "neg_near", "edit", EDITS),
             ("no", "neg_far", "edit", EDITS)) if v3 else (
                 ("yes", "pos", "kind", KINDS), ("no", "neg", "edit", EDITS))
    for label, key, field, values in sides:
        if v3 and key.endswith("far"):
            x, reason = check_common_side(data.get(key), field, values, src["c"], src["r"])
            if reason is None and char_dice(x, src["c"]) >= 0.6:
                x, reason = None, "not_far"
        else:
            x, reason = check_side(data.get(key), field, values, src["c"], src["r"],
                                   neg=v3 or label == "no")
        if v3 and by_variant is not None:
            counts = by_variant.setdefault(key, {"kept": 0, "dropped": Counter()})
            if reason:
                counts["dropped"][reason] += 1
            else:
                counts["kept"] += 1
        if reason:
            reasons[label][reason] += 1
            continue
        rows.append({"pair_id": f"{src['src_id']}-{key if v3 else label}",
                     **({"variant": key} if v3 else {}), "src_id": src["src_id"],
                     "speaker": src["speaker"], "source": src["source"],
                     "show": src["show"], "split": src["split"], "c": src["c"],
                     "r": src["r"], "x": x, "label": label, field: data[key][field],
                     "gen_model": gen_model, "prompt_version": prompt_version,
                     "prompt_sha256": prompt_sha256})
    return rows


def generate_pairs(sources: list[dict], backend, gen_model: str) -> tuple[list[dict], dict]:
    rows, reasons = [], {"yes": Counter(), "no": Counter()}
    seconds = 0.0
    for src in sources:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TEMPLATE.format(c=src["c"], r=src["r"])}]
        started = time.perf_counter()
        try:
            text = backend.generate(messages, MAX_TOKENS)
        except Exception:
            text = None
        seconds += time.perf_counter() - started
        try:
            data = json.loads(text) if text is not None else None
        except ValueError:
            data = None
        if not isinstance(data, dict):
            reason = "generate_error" if text is None else "json_error"
            reasons["yes"][reason] += 1
            reasons["no"][reason] += 1
            continue
        rows.extend(validated_pairs(src, data, gen_model, PROMPT_VERSION,
                                    PROMPT_SHA256, reasons))
    kept = Counter(row["label"] for row in rows)
    stats = {"sources": len(sources), "kept": dict(kept),
             "dropped": {k: dict(v) for k, v in reasons.items()},
             "seconds_per_source": round(seconds / len(sources), 3) if sources else None,
             "gen_model": gen_model, "prompt_version": PROMPT_VERSION,
             "prompt_sha256": PROMPT_SHA256}
    return rows, stats


# ---------- import ----------

def run_import(sources_path: Path, gen_path: Path, prompt_file: Path,
               gen_model: str, out: Path) -> dict:
    sources = {src["src_id"]: src for src in read_jsonl(sources_path)}
    prompt_hash = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    rows, seen = [], set()
    reasons = {"yes": Counter(), "no": Counter()}
    by_variant = {}
    unknown, duplicate = 0, 0
    for data in read_jsonl(gen_path):
        src_id = data["src_id"]
        if src_id not in sources:
            unknown += 1
            continue
        if src_id in seen:
            duplicate += 1
            continue
        seen.add(src_id)
        rows.extend(validated_pairs(sources[src_id], data, gen_model, "import-v1",
                                    prompt_hash, reasons, by_variant))
    stats = {"kept": dict(Counter(row["label"] for row in rows)),
             "dropped": {label: dict(counts) for label, counts in reasons.items()},
             "unknown_src": unknown, "duplicate_src": duplicate,
             "missing_gen": len(sources.keys() - seen)}
    if by_variant:
        stats["by_variant"] = by_variant
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "pairs.jsonl", rows)
    write_json(out / "import_summary.json", stats)
    return stats


# ---------- audit ----------

def build_audit(pairs: list[dict], n: int) -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    half = n // 2
    chosen = []
    for label in ("yes", "no"):
        group = [p for p in pairs if p["label"] == label]
        chosen += group if len(group) <= half else rng.sample(group, half)
    rng.shuffle(chosen)
    items, key = [], {}
    for i, pair in enumerate(chosen, 1):
        item_id = f"a{i:04d}"
        items.append({"item_id": item_id, "c": pair["c"], "r": pair["r"], "x": pair["x"]})
        field = "kind" if pair["label"] == "yes" else "edit"
        key[item_id] = {"pair_id": pair["pair_id"], "label": pair["label"], field: pair[field]}
    return items, key


# ---------- cli ----------

def out_dir(root: Path, sub: str) -> Path:
    path = Path(sub)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit("--out must be a subdirectory name under the weak-label root")
    return root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_split = sub.add_parser("split")
    p_split.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p_sample.add_argument("--n", type=int, required=True)
    p_sample.add_argument("--out", required=True)
    p_sample.add_argument("--exclude", type=Path, action="append", default=[])
    p_sample.add_argument("--id-prefix", default="")
    p_sample.add_argument("--split", choices=("train", "dev", "test"), default="train")
    p_cped = sub.add_parser("cped-sources")
    p_cped.add_argument("--csv", type=Path, required=True)
    p_cped.add_argument("--all-csv", type=Path, nargs=3, required=True,
                        metavar=("TRAIN", "VALID", "TEST"))
    p_cped.add_argument("--split", choices=("train", "valid", "test"), required=True)
    p_cped.add_argument("--n", type=int, required=True)
    p_cped.add_argument("--out", required=True)
    p_cped.add_argument("--exclude", type=Path, action="append", default=[])
    p_cped.add_argument("--id-prefix", default="")
    p_import = sub.add_parser("import")
    p_import.add_argument("--sources", type=Path, required=True)
    p_import.add_argument("--gen", type=Path, required=True)
    p_import.add_argument("--prompt-file", type=Path, required=True)
    p_import.add_argument("--gen-model", required=True)
    p_import.add_argument("--out", required=True)
    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--model", default=DEFAULT_MODEL)
    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--out", required=True)
    p_audit.add_argument("--n", type=int, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "split":
        result = run_split(args.raw_dir, args.root)
        summary = {k: result[k] for k in ("raw_pairs", "dedup_pairs", "pairs_per_split")}
        summary["streamers"] = {k: len(result[k]) for k in ("train", "dev", "test")}
    elif args.cmd == "sample":
        summary = run_sample(args.raw_dir, args.root, out_dir(args.root, args.out), args.n,
                             args.exclude, args.id_prefix, args.split)
    elif args.cmd == "cped-sources":
        summary = run_cped_sources(args.csv, args.all_csv, args.split, args.n,
                                   out_dir(args.root, args.out), args.exclude,
                                   args.id_prefix)
    elif args.cmd == "import":
        summary = run_import(args.sources, args.gen, args.prompt_file, args.gen_model,
                             out_dir(args.root, args.out))
    elif args.cmd == "generate":
        out = out_dir(args.root, args.out)
        os.environ["HF_HUB_OFFLINE"] = "1"
        backend = llm.MlxBackend(args.model)
        rows, summary = generate_pairs(read_jsonl(out / "sources.jsonl"), backend,
                                       llm.model_name(args.model))
        write_jsonl(out / "pairs.jsonl", rows)
        write_json(out / "generate_summary.json", summary)
    else:
        out = out_dir(args.root, args.out)
        items, key = build_audit(read_jsonl(out / "pairs.jsonl"), args.n)
        write_jsonl(out / "audit_items.jsonl", items)
        write_json(out / "audit_key.json", key)
        summary = {"items": len(items), "labels": dict(Counter(v["label"] for v in key.values()))}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
