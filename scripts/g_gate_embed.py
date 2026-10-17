"""Generate offline cosine scores with a pinned, locally available encoder.

No dependency installation or model download occurs here. HF IDs must already
be cached; a local model path also works. score_events accepts a fake encoder
without importing sentence_transformers. Only main loads that optional package.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
from pathlib import Path

if __package__:
    from . import g_gate
else:
    import g_gate


def score_events(events: list[dict], encoder) -> list[dict]:
    texts = []
    keys = []
    for event in events:
        for candidate in event["candidates"]:
            keys.append((event["event_id"], candidate["cand_id"]))
            texts.extend([g_gate.norm_lines(event["context"]), g_gate.norm_lines(candidate["c"])])
    if not texts:
        return []
    vectors = encoder.encode(texts, normalize_embeddings=True)
    if len(vectors) != len(texts):
        raise ValueError("encoder returned incorrect vector count")
    scores = []
    for i, (eid, cid) in enumerate(keys):
        left, right = vectors[2 * i], vectors[2 * i + 1]
        if not len(left) or len(left) != len(right):
            raise ValueError("invalid encoder dimensions")
        # Recompute cosine so injected encoders follow the same mathematical contract.
        numerator = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
        magnitude = math.sqrt(sum(float(x) ** 2 for x in left)) * math.sqrt(
            sum(float(x) ** 2 for x in right),
        )
        score = numerator / magnitude if magnitude else 0.0
        if not math.isfinite(score):
            raise ValueError("non-finite encoder score")
        scores.append({"event_id": eid, "cand_id": cid, "score": score})
    return scores


def write_scores(path: Path, scores: list[dict]) -> None:
    g_gate.write_output(path, "".join(json.dumps(row, allow_nan=False) + "\n" for row in scores))


def main(argv: list[str] | None = None) -> int:
    try:
        parser = g_gate._Parser()
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--model", required=True)
        parser.add_argument("--revision", required=True)
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args(argv)
        if not args.out.parent.is_dir() or not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
            raise ValueError("invalid output or revision")
        events = g_gate.load_events(args.input)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Library progress and error text must not leak input/model details to the CLI.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(args.model, revision=args.revision, local_files_only=True)
            scores = score_events(events, encoder)
        write_scores(args.out, scores)
    except Exception:
        print('{"status":"failed"}')
        return 1
    print(json.dumps({"status": "ok", "events": len(events), "candidates": len(scores)},
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
