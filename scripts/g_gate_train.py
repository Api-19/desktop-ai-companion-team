# ruff: noqa: RUF001
"""Fine-tuned classifier gate arm (offline research; public weak-label pairs, synthetic scenes).

Fine-tunes GLiClass or Laya as a binary "reference applies / does not apply"
classifier on weak-label pairs, early-stops on dev AUC (max 3 epochs), fits one
temperature on dev logits (min NLL) and keeps a candidate iff calibrated p_yes >= 0.5.

Subcommands: ``train`` (MPS only), ``predict`` (synthetic scene jsonl or pairs
jsonl → rows in the ``g_gate_llm._row`` format, mode ``ft-<arm>``) and ``report``
(AUC, ECE, Brier, accuracy by variant/source; plus the surface-feature baseline
when ``--pairs`` is given). torch/laya/gliclass are imported only inside the
arm loaders. Run with ``~/.rcpem/venv-cls/bin/python`` and ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from . import g_gate_cls as cls
    from . import g_gate_llm as llm
else:
    import g_gate_cls as cls
    import g_gate_llm as llm

ARMS = ("gliclass", "laya")
SEED = 20261016
DEV_FRAC = 0.2
MAX_EPOCHS = 3
DEV_SPLITS = ("dev", "valid")
HYPER = {
    "gliclass": {"lr": 1e-5, "head_lr": 1e-5, "batch": 8, "weight_decay": 0.01,
                 "warmup_frac": 0.1, "max_length": 512},
    "laya": {"lr": 1e-5, "head_lr": 1e-5, "batch": 8, "weight_decay": 0.01, "warmup_frac": 0.1},
}
META_FILE = "gate_meta.json"


# ---------- text ----------

def clauses(text: str) -> list[str]:
    return [part.strip() for part in text.split("／") if part.strip()]


def pair_to_event(pair: dict) -> dict:
    return {"event_id": pair["pair_id"], "context": clauses(pair["x"]), "permission": None,
            "candidates": [{"cand_id": "c1", "c": clauses(pair["c"]), "r": clauses(pair["r"])}]}


def render_pair(pair: dict) -> str:
    event = pair_to_event(pair)
    return cls.render_text(event, event["candidates"][0])


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                            for r in rows), encoding="utf-8")


# ---------- splits ----------

def group_split(pairs: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out ``round(frac * groups)`` src_ids (sorted, then seeded shuffle)."""
    groups = sorted({p["src_id"] for p in pairs})
    random.Random(seed).shuffle(groups)
    held = set(groups[:round(len(groups) * frac)])
    return ([p for p in pairs if p["src_id"] not in held],
            [p for p in pairs if p["src_id"] in held])


def select_splits(pairs: list[dict], final: bool):
    """(train, dev, test, note). Without dev/valid rows, dev is a grouped 80/20 split of train."""
    train = [p for p in pairs if p["split"] == "train"]
    dev = [p for p in pairs if p["split"] in DEV_SPLITS]
    test = [p for p in pairs if p["split"] == "test"] if final else []
    if dev:
        return train, dev, test, "split-field"
    train, dev = group_split(train, DEV_FRAC, SEED)
    note = f"temp-dev: src_id grouped {1 - DEV_FRAC:.0%}/{DEV_FRAC:.0%} of train, seed {SEED}"
    return train, dev, test, note


# ---------- metrics and calibration ----------

def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _log1pexp(z: float) -> float:
    return z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))


def nll(logits: list[float], labels: list[int], temperature: float) -> float:
    """Mean binary cross-entropy of sigmoid(logit / T)."""
    total = 0.0
    for z, y in zip(logits, labels, strict=True):
        s = z / temperature
        total += _log1pexp(-s) if y else _log1pexp(s)
    return total / len(logits)


def fit_temperature(logits: list[float], labels: list[int]) -> float:
    """Golden-section search on a = 1/T in [0.01, 50] (NLL is convex in a); never worse than T=1."""
    lo, hi = 0.01, 50.0
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(200):
        a1, a2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if nll(logits, labels, 1 / a1) <= nll(logits, labels, 1 / a2):
            hi = a2
        else:
            lo = a1
    t = 2 / (lo + hi)
    return t if nll(logits, labels, t) <= nll(logits, labels, 1.0) else 1.0


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney AUC with average ranks for ties; None if one class is missing."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos_ranks = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    return (pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier(probs: list[float], labels: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def ece(probs: list[float], labels: list[int], bins: int = 10) -> float:
    """Reliability of p_yes: 10 equal-width bins, |mean p_yes - positive rate| weighted by size."""
    buckets = defaultdict(list)
    for p, y in zip(probs, labels, strict=True):
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    return sum(abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b)) * len(b)
               for b in buckets.values()) / len(probs)


def decide(p_yes: float) -> str:
    return "keep" if p_yes >= 0.5 else "drop"


def metrics(probs: list[float], labels: list[int]) -> dict:
    return {"auc": auc(probs, labels), "ece": ece(probs, labels), "brier": brier(probs, labels),
            "accuracy": sum((p >= 0.5) == bool(y) for p, y in zip(probs, labels, strict=True))
            / len(probs)}


def _rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def report(rows: list[dict]) -> dict:
    """Metrics on calibrated p_yes (and ``raw`` on fields.p_raw); accuracy by variant/source."""
    labels = [int(r["label"] == "yes") for r in rows]
    result = {"n": len(rows), **metrics([r["p_yes"] for r in rows], labels)}
    result["raw"] = metrics([r["fields"]["p_raw"] for r in rows], labels)
    for key in ("variant", "source"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append((r["decision"] == "keep") == (r["label"] == "yes"))
        result[f"by_{key}"] = {g: {"n": len(v), "accuracy": sum(v) / len(v)}
                               for g, v in sorted(groups.items())}
    return result


# ---------- surface-feature baseline (same features as the D21 shortcut check) ----------

NEG = ("别", "不要", "不用", "没", "不")
META = ("请", "直接", "核对", "不是问", "我问的是", "我说的是", "已经", "明确", "从没", "从未",
        "并不", "无关")


def dice(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 2 * sum((Counter(a) & Counter(b)).values()) / (len(a) + len(b))


def surface_features(pair: dict) -> list[float]:
    x, c, r = pair["x"], pair["c"], pair["r"]
    return [len(x), len(x) / max(1, len(c)), dice(x, c), dice(x, r), sum(w in x for w in NEG),
            sum(w in x for w in META), x.count("？") + x.count("?"),
            x.count("，") + x.count("／")]


def surface_baseline(train: list[dict], dev: list[dict]) -> dict:
    """Logistic regression on surface features, fit on train, AUC on dev (needs sklearn)."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000).fit(
        [surface_features(p) for p in train], [p["label"] == "yes" for p in train])
    probs = model.predict_proba([surface_features(p) for p in dev])[:, 1].tolist()
    return {"n_train": len(train), "n_dev": len(dev),
            "auc": auc(probs, [int(p["label"] == "yes") for p in dev])}


# ---------- prediction rows ----------

def _row(event_id, cand_id, arm_name, model, logit, temperature) -> dict:
    p_yes = sigmoid(logit / temperature)
    return {"event_id": event_id, "cand_id": cand_id, "mode": f"ft-{arm_name}", "model": model,
            "decision": decide(p_yes), "p_yes": p_yes,
            "fields": {"logit": logit, "p_raw": sigmoid(logit), "temperature": temperature},
            "technical_failure": False}


def _batched_logits(arm, texts: list[str], batch: int) -> list[float]:
    out = []
    for start in range(0, len(texts), batch):
        out.extend(float(v) for v in arm.predict_logits(texts[start:start + batch]))
    return out


def _arm_name(model: str) -> str:
    return model.split("@", 1)[0].removeprefix("ft-")


def predict_pairs(pairs: list[dict], arm, temperature: float, model: str, batch: int) -> list[dict]:
    logits = _batched_logits(arm, [render_pair(p) for p in pairs], batch)
    rows = []
    for p, z in zip(pairs, logits, strict=True):
        row = _row(p["pair_id"], "c1", _arm_name(model), model, z, temperature)
        row.update(pair_id=p["pair_id"], label=p["label"], variant=p["variant"],
                   source=p["source"])
        rows.append(row)
    return rows


def predict_events(events: list[dict], arm, temperature: float, model: str,
                   batch: int) -> list[dict]:
    items = [(e, c) for e in events for c in e["candidates"]]
    logits = _batched_logits(arm, [cls.render_text(e, c) for e, c in items], batch)
    return [_row(e["event_id"], c["cand_id"], _arm_name(model), model, z, temperature)
            for (e, c), z in zip(items, logits, strict=True)]


# ---------- arms (torch / laya / gliclass imported lazily) ----------

def require_mps(torch):
    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available; refusing to train or predict on CPU")


class GliclassArm:
    """Binary logit = logit(合适) - logit(不合适) over the two fixed GLiClass labels."""

    name = "gliclass"

    def __init__(self, path: str, device):
        import torch
        from gliclass import GLiClassModel
        from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        self.torch = torch
        self.model = GLiClassModel.from_pretrained(path, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.pipe = UniEncoderZeroShotClassificationPipeline(
            self.model, self.tokenizer, max_length=HYPER["gliclass"]["max_length"],
            classification_type="single-label", device=device, progress_bar=False)
        self.device = device

    def forward(self, texts: list[str]):
        inputs = self.pipe.prepare_inputs(texts, cls.GLICLASS_LABELS, same_labels=True)
        logits = self.model(**inputs, max_num_classes=len(cls.GLICLASS_LABELS)).logits.float()
        return logits[:, 0] - logits[:, 1]

    def param_groups(self, hyper):
        enc = [p for n, p in self.model.named_parameters() if n.startswith("encoder_model.")]
        head = [p for n, p in self.model.named_parameters() if not n.startswith("encoder_model.")]
        return [{"params": enc, "lr": hyper["lr"]}, {"params": head, "lr": hyper["head_lr"]}]

    def save(self, path: Path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def predict_logits(self, texts):
        with self.torch.no_grad():
            return self.forward(texts).cpu().tolist()


class LayaArm:
    """Binary logit = logit(true) - logit(false) of Laya's noul head on the zero-shot question."""

    name = "laya"

    def __init__(self, path: str, device):
        import torch
        from laya import common
        from laya import train as lt

        self.torch, self.common, self.lt = torch, common, lt
        self.model, self.tok, self.cfg = lt.load_checkpoint(path)
        self.max_len, self.head_max_len = self.cfg["max_len"], self.cfg["head_max_len"]
        self.q = lt.to_internal("q", {"type": "noul", "instructions": cls.LAYA_QUESTION})
        self.model.to(device).eval()
        self.device = device

    def forward(self, texts: list[str]):
        chunk = []
        for text in texts:
            state_ids = self.lt.encode_state(self.tok, text, self.max_len)
            item, reason = self.lt.make_item(self.tok, self.q, [0.5, 0.5], state_ids,
                                             self.head_max_len, self.max_len)
            if reason is not None:
                raise ValueError(f"laya item rejected: {reason}")
            chunk.append(self.lt.encode_item(self.tok, item, self.max_len, self.head_max_len))
        batch = self.common.collate_items([chunk], self.tok.pad_token_id)
        logits = self.lt._forward(self.model, batch, self.device, amp=False, detach_encoder=False)
        return logits[:, 1] - logits[:, 0]

    def param_groups(self, hyper):
        enc = [p for n, p in self.model.named_parameters() if n.startswith("encoder.")]
        head = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
        return [{"params": enc, "lr": hyper["lr"]}, {"params": head, "lr": hyper["head_lr"]}]

    def save(self, path: Path):
        self.lt.save_checkpoint(self.model, self.tok, self.cfg, str(path))

    def predict_logits(self, texts):
        with self.torch.no_grad():
            return self.forward(texts).cpu().tolist()


def load_arm(arm: str, path: str):
    import torch

    require_mps(torch)
    device = torch.device("mps")
    return (GliclassArm if arm == "gliclass" else LayaArm)(path, device)


# ---------- training ----------

def _peak_mb(torch) -> dict:
    return {"mps_driver_mb": round(torch.mps.driver_allocated_memory() / 2**20),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)}


def train(arm, train_pairs: list[dict], dev_pairs: list[dict], out: Path, hyper: dict,
          log=print) -> dict:
    """Train with BCE; after each epoch score dev; save the best-AUC epoch; stop on no gain."""
    import torch

    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    texts = [render_pair(p) for p in train_pairs]
    ys = [float(p["label"] == "yes") for p in train_pairs]
    dev_texts = [render_pair(p) for p in dev_pairs]
    dev_y = [int(p["label"] == "yes") for p in dev_pairs]
    model = arm.model
    optimizer = torch.optim.AdamW(arm.param_groups(hyper), weight_decay=hyper["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    batch = hyper["batch"]
    total_steps = MAX_EPOCHS * math.ceil(len(texts) / batch)
    warmup = max(1, round(total_steps * hyper["warmup_frac"]))
    decay = max(1, total_steps - warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / decay)))
    best, history, peak = None, [], {"mps_driver_mb": 0, "max_rss_mb": 0}
    model.eval()
    zero_logits = _batched_logits(arm, dev_texts, batch)
    zero_shot = metrics([sigmoid(z) for z in zero_logits], dev_y)
    log(f"epoch 0 (zero-shot) dev {_rounded(zero_shot)}")
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        order = list(range(len(texts)))
        random.Random(SEED + epoch).shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            logits = arm.forward([texts[i] for i in idx])
            target = torch.tensor([ys[i] for i in idx], device=logits.device)
            loss = loss_fn(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)
            now = _peak_mb(torch)
            peak = {k: max(peak[k], now[k]) for k in peak}
        train_seconds = time.perf_counter() - started
        model.eval()
        dev_logits = _batched_logits(arm, dev_texts, batch)
        dev_auc = auc(dev_logits, dev_y)
        record = {"epoch": epoch, "train_loss": total / len(texts),
                  "train_seconds": train_seconds,
                  "dev_auc": dev_auc, "dev_nll": nll(dev_logits, dev_y, 1.0),
                  "epoch_seconds": time.perf_counter() - started}
        history.append(record)
        log(f"epoch {epoch} {_rounded(record)}")
        if best is not None and (dev_auc or 0) <= (best["dev_auc"] or 0):
            break
        best = {**record, "logits": dev_logits}
        arm.save(out)
    temperature = fit_temperature(best["logits"], dev_y)
    return {"history": history, "best_epoch": best["epoch"], "temperature": temperature,
            "dev_logits": best["logits"], "zero_shot_dev": zero_shot, "peak_memory": peak,
            "nll_before": nll(best["logits"], dev_y, 1.0),
            "nll_after": nll(best["logits"], dev_y, temperature)}


def _pairs_from(paths: list[Path]) -> list[dict]:
    pairs = [p for path in paths for p in read_jsonl(path)]
    ids = [p["pair_id"] for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair_id across --pairs files")
    return pairs


def cmd_train(args) -> int:
    pairs = _pairs_from(args.pairs)
    train_pairs, dev_pairs, test_pairs, note = select_splits(pairs, args.final)
    arm = load_arm(args.arm, args.model)
    hyper = {**HYPER[args.arm], **{k: v for k, v in (("lr", args.lr), ("head_lr", args.head_lr),
                                                       ("batch", args.batch)) if v is not None}}
    result = train(arm, train_pairs, dev_pairs, args.out, hyper)
    meta = {"arm": args.arm, "base_model": cls.model_name(args.arm, args.model),
            "pairs": [str(p) for p in args.pairs], "split_note": note, "seed": SEED,
            "n_train": len(train_pairs), "n_dev": len(dev_pairs), "hyper": hyper,
            "max_epochs": MAX_EPOCHS, "loss": "binary cross-entropy",
            **{k: v for k, v in result.items() if k != "dev_logits"}}
    model = f"ft-{args.arm}@{args.out.name}"
    dev_rows = [_row(p["pair_id"], "c1", args.arm, model, z, result["temperature"])
                | {"pair_id": p["pair_id"], "label": p["label"], "variant": p["variant"],
                   "source": p["source"]}
                for p, z in zip(dev_pairs, result["dev_logits"], strict=True)]
    write_jsonl(args.out / "dev_pred.jsonl", dev_rows)
    meta["dev_report"] = report(dev_rows)
    meta["surface_baseline_dev"] = surface_baseline(train_pairs, dev_pairs)
    if test_pairs:
        best = load_arm(args.arm, str(args.out))
        test_rows = predict_pairs(test_pairs, best, result["temperature"], model, hyper["batch"])
        write_jsonl(args.out / "test_pred.jsonl", test_rows)
        meta["test_report"] = report(test_rows)
    (args.out / META_FILE).write_text(json.dumps(_rounded(meta), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(_rounded({k: meta[k] for k in (
        "arm", "split_note", "n_train", "n_dev", "best_epoch", "temperature", "nll_before",
        "nll_after", "peak_memory", "dev_report", "surface_baseline_dev")}), ensure_ascii=False))
    return 0


def cmd_predict(args) -> int:
    meta = json.loads((args.ckpt / META_FILE).read_text(encoding="utf-8"))
    rows_in = read_jsonl(args.input)
    arm = load_arm(meta["arm"], str(args.ckpt))
    model = f"ft-{meta['arm']}@{args.ckpt.name}"
    batch = HYPER[meta["arm"]]["batch"]
    if rows_in and "pair_id" in rows_in[0]:
        rows = predict_pairs(rows_in, arm, meta["temperature"], model, batch)
        summary = report(rows)
    else:
        events = llm.load_events(args.input)
        rows = predict_events(events, arm, meta["temperature"], model, batch)
        summary = llm.summarize(events, rows)
    write_jsonl(args.out, rows)
    print(json.dumps(_rounded(summary), ensure_ascii=False))
    return 0


def cmd_report(args) -> int:
    result = report(read_jsonl(args.pred))
    if args.pairs:
        pairs = _pairs_from(args.pairs)
        train_pairs, dev_pairs, _, note = select_splits(pairs, final=False)
        result["surface_baseline_dev"] = {**surface_baseline(train_pairs, dev_pairs),
                                          "split_note": note}
    print(json.dumps(_rounded(result), ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--pairs", type=Path, action="append", required=True)
    p_train.add_argument("--arm", choices=ARMS, required=True)
    p_train.add_argument("--model", required=True, help="local base model snapshot path")
    p_train.add_argument("--out", type=Path, required=True, help="checkpoint directory")
    p_train.add_argument("--final", action="store_true", help="also evaluate the test split once")
    p_train.add_argument("--lr", type=float, help="encoder learning rate (default per arm)")
    p_train.add_argument("--head-lr", type=float, help="head learning rate (default per arm)")
    p_train.add_argument("--batch", type=int, help="batch size (default per arm)")
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--input", type=Path, required=True)
    p_pred.add_argument("--ckpt", type=Path, required=True)
    p_pred.add_argument("--out", type=Path, required=True)
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--pred", type=Path, required=True)
    p_rep.add_argument("--pairs", type=Path, action="append",
                       help="pairs files for the surface-feature baseline on the same dev split")
    args = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return {"train": cmd_train, "predict": cmd_predict, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
