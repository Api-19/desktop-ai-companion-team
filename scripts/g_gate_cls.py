# ruff: noqa: RUF001
"""Zero-shot classifier gate arm (offline research; synthetic data only).

Scores each candidate with an off-the-shelf zero-shot classifier (Laya binary
question or GLiClass single-label) and keeps it iff p_yes >= 0.5. Models are
injected; laya/gliclass are imported only by the default loader used in
``main``. Run real inference with ``~/.rcpem/venv-cls/bin/python`` and
``HF_HUB_OFFLINE=1``. Output rows follow ``g_gate_llm``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

if __package__:
    from . import g_gate_llm as llm
else:
    import g_gate_llm as llm

BACKENDS = ("laya", "gliclass")
LAYA_QUESTION = "把参考回复的回应方式用在当前对话里合适吗？"
GLICLASS_LABELS = ["合适", "不合适"]
MODE = "cls"


def render_text(event: dict, candidate: dict) -> str:
    text = (
        "当前对话：\n" + "\n".join(event["context"])
        + "\n参考·当时对方说：\n" + "\n".join(candidate["c"])
        + "\n参考·他的回复：\n" + "\n".join(candidate["r"])
    )
    if event.get("permission") is not None:
        text += "\n许可：" + event["permission"]["note"]
    return text


def laya_scorer(agent):
    def scorer(text: str) -> tuple[float, dict]:
        result = agent.system_one(text, {"q": {"type": "noul", "instructions": LAYA_QUESTION}})
        return result["answers"]["q"]["noul"], {"truncated": result["usage"]["truncated"]}

    return scorer


def gliclass_scorer(pipeline):
    def scorer(text: str) -> tuple[float, dict]:
        scores = pipeline(text, GLICLASS_LABELS, classification_type="single-label",
                          return_hierarchical=True)[0]
        s_yes, s_no = scores["合适"], scores["不合适"]
        return s_yes / (s_yes + s_no), {"s_yes": s_yes, "s_no": s_no}

    return scorer


def _row(event, candidate, model, decision, p_yes=None, fields=None, failed=False):
    return {
        "event_id": event["event_id"], "cand_id": candidate["cand_id"], "mode": MODE,
        "model": model, "decision": decision, "p_yes": p_yes, "fields": fields or {},
        "technical_failure": failed,
    }


def classify(events: list[dict], scorer, model: str) -> list[dict]:
    """One row per candidate; any scorer error or invalid probability is a technical failure."""
    rows = []
    for event in events:
        for candidate in event["candidates"]:
            try:
                p_yes, fields = scorer(render_text(event, candidate))
                p_yes = float(p_yes)
                if not 0.0 <= p_yes <= 1.0:
                    raise ValueError("probability out of range")
            except Exception:
                rows.append(_row(event, candidate, model, "drop", failed=True))
                continue
            rows.append(_row(event, candidate, model, "keep" if p_yes >= 0.5 else "drop",
                             p_yes, fields))
    return rows


def model_name(backend: str, path: str) -> str:
    match = re.search(r"/snapshots/([0-9a-f]+)", path)
    return f"{backend}@{match[1] if match else Path(path).name}"


def load_scorer(backend: str, path: str):
    if backend == "laya":
        import laya

        return laya_scorer(laya.load(path))
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    pipeline = ZeroShotClassificationPipeline(
        GLiClassModel.from_pretrained(path), AutoTokenizer.from_pretrained(path),
        classification_type="single-label", device="cpu", progress_bar=False,
    )
    return gliclass_scorer(pipeline)


def main(argv: list[str] | None = None, loader=load_scorer) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--model", required=True, help="local model snapshot path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    events = llm.load_events(args.input)
    model = model_name(args.backend, args.model)
    started = time.perf_counter()
    try:
        scorer = loader(args.backend, args.model)
    except Exception:
        scorer = None
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if scorer is None:
        rows = [_row(e, c, model, "drop", failed=True) for e in events for c in e["candidates"]]
    else:
        rows = classify(events, scorer, model)
    seconds = time.perf_counter() - started
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    summary = llm.summarize(events, rows)
    summary.update(mode=MODE, model=model, load_seconds=round(load_seconds, 2),
                   seconds_per_candidate=round(seconds / len(rows), 3) if rows else None)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
