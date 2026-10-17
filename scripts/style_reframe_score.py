"""Score local blind-review choices without printing ballot contents."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from style_reframe_eval import score_choices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--candidate-arm", default="full")
    args = parser.parse_args(argv)
    try:
        choices = json.loads(args.choices.read_text(encoding="utf-8"))
        answer_key = json.loads(args.answer_key.read_text(encoding="utf-8"))
        result = score_choices(choices, answer_key, candidate_arm=args.candidate_arm)
    except (OSError, ValueError):
        print('{"status":"invalid_input"}')
        return 2
    print(json.dumps({"status": "scored", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
