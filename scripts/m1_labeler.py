# ruff: noqa: RUF001
"""Batch first-action labeling; CLI requires --real, tests inject a fake provider.

Gold uses LabelStore's item -> {label, at} format. Reports contain only labels
and aggregate metrics. Undefined kappa is zero for an empty intersection and
one for unanimous agreement; absent gold classes are omitted from macro-F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from companion.admin.labeling import _validate_tasks
from companion.dialogue.config import load_dialogue_config
from companion.dialogue.context import PromptMessage
from companion.dialogue.provider import DialogueProvider, OpenAICompatibleProvider, ProviderError
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime.settings import load_settings

CODES = tuple("ABCDE")
MODEL_CODES = (*CODES, "?", "invalid", "not_run")
_LABEL_LINE = re.compile(r"^\s*\[?(r\d{3})\]?\s*[:：]\s*([A-E?？])\s*$")
_SYSTEM = (
    '你是对话标注员。下面每条是一段聊天：可能有前文，然后是"她"说的话，最后是"他"的回复。'
    '只给"他"的回复标一个标签，标的是他回复中第一个有内容的动作。'
    '开头的纯附和词（可以可以、嗯嗯、哦哦、好呀、ok、嘿嘿、行吧、哎呦等）先跳过；'
    '整条回复只有附和词时标 A。只有表情时按表情作用标：安慰类（如[拥抱]）标 A，'
    '搞怪类（如[猪头]）标 D。只看这段对话本身，不要推测两人关系处在什么阶段。\n标签：\n'
    '{labels}\n? 无法判断\n边界：目的是要她哄、要她关注、表达喜欢标 E；'
    '目的是逗她、损她、搞笑标 D；一边接话一边损她标 D；'
    '针对她刚说的话反问标 A，问她新的情况标 B。\n'
    '输出格式：每条一行，"编号:标签"，不要输出其他任何文字。'
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("invalid arguments")


class RunLimits:
    """Same per-run accounting and 0.20 CNY reserve as style_reframe_eval."""

    def __init__(self, ledger: MonthlyUsageLedger, max_cny: Decimal, max_requests: int):
        self.ledger = ledger
        self.max_cny = max_cny
        self.max_requests = max_requests
        self.completed_requests = 0
        self.stopped = None
        self.start = Decimal("0")
        self.current = Decimal("0")
        if self.refresh():
            self.start = self.current

    def refresh(self) -> bool:
        try:
            self.current = self.ledger.snapshot(datetime.now(UTC)).accounted_cny
        except Exception:
            self.stopped = "spend_cap"
            return False
        return True

    @property
    def spent_cny(self) -> str:
        return format(self.current - self.start, ".2f")

    def before_request(self) -> bool:
        if self.stopped is not None or not self.refresh():
            return False
        if self.completed_requests >= self.max_requests:
            self.stopped = "request_cap"
        elif self.current - self.start + Decimal("0.20") > self.max_cny:
            self.stopped = "spend_cap"
        return self.stopped is None


def build_messages(tasks: dict, item_ids: Sequence[str]) -> list[PromptMessage]:
    labels = "\n".join(f'{label["code"]} {label["name"]}：{label["hint"]}'
                       for label in tasks["labels"])
    entries = []
    for item_id in item_ids:
        item = tasks["items"][item_id]
        lines = [f"[{item_id}]", *(f"前文 {line}" for line in item["context"]),
                 *(f"她：{line}" for line in item["partner"]),
                 *(f"他：{line}" for line in item["reply"])]
        entries.append("\n".join(lines))
    return [PromptMessage(role="system", content=_SYSTEM.format(labels=labels)),
            PromptMessage(role="user", content="\n\n".join(entries))]


def parse_labels(text: str, item_ids: Sequence[str]) -> dict[str, str]:
    seen = {item_id: set() for item_id in item_ids}
    for line in text.splitlines():
        match = _LABEL_LINE.fullmatch(line)
        if match is not None and match[1] in seen:
            seen[match[1]].add(match[2].replace("？", "?"))
    return {item_id: next(iter(codes)) if len(codes) == 1 else "invalid"
            for item_id, codes in seen.items()}


def compute_metrics(gold: dict, predictions: dict[str, str]) -> dict:
    confusion = {code: dict.fromkeys(MODEL_CODES, 0) for code in CODES}
    for item_id, predicted in predictions.items():
        if item_id in gold and gold[item_id]["label"] in CODES:
            confusion[gold[item_id]["label"]][predicted] += 1
    per_class = {}
    for code in CODES:
        support = sum(confusion[code].values())
        predicted_count = sum(confusion[other][code] for other in CODES)
        tp = confusion[code][code]
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted_count) if support + predicted_count else 0.0
        per_class[code] = dict(support=support, precision=precision, recall=recall, f1=f1)
    supported = [entry for entry in per_class.values() if entry["support"]]
    macro_f1 = sum(entry["f1"] for entry in supported) / len(supported) if supported else 0.0
    row_totals = {code: sum(confusion[code][pred] for pred in CODES) for code in CODES}
    col_totals = {code: sum(confusion[other][code] for other in CODES) for code in CODES}
    count = sum(row_totals.values())
    kappa = 0.0
    if count:
        observed = sum(confusion[code][code] for code in CODES) / count
        expected = sum(row_totals[code] * col_totals[code] for code in CODES) / count**2
        kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    rejected = sum(code not in CODES for code in predictions.values())
    reject_rate = rejected / len(predictions) if predictions else 0.0
    return dict(kappa=kappa, macro_f1=macro_f1, reject_rate=reject_rate,
                per_class=per_class, confusion=confusion)


def is_passed(metrics: dict) -> bool:
    return (metrics["kappa"] >= 0.60 and metrics["macro_f1"] >= 0.75
            and metrics["reject_rate"] <= 0.15
            and all(entry["recall"] > 0 for entry in metrics["per_class"].values()
                    if entry["support"] > 0))


def _validate_inputs(tasks: dict, round_id: str, gold: dict) -> list[str]:
    _validate_tasks(tasks)
    if {label["code"] for label in tasks["labels"]} != set(CODES):
        raise ValueError("invalid label codes")
    round_ = next(entry for entry in tasks["rounds"] if entry["id"] == round_id)
    order = round_["order"]
    if any(re.fullmatch(r"r\d{3}", item_id) is None for item_id in order):
        raise ValueError("invalid item ids")
    if not isinstance(gold, dict):
        raise ValueError("invalid gold")
    for item_id, answer in gold.items():
        if (item_id not in order or not isinstance(answer, dict)
                or set(answer) != {"label", "at"}
                or answer["label"] not in (*CODES, "?")
                or not isinstance(answer["at"], str)
                or datetime.fromisoformat(answer["at"]).tzinfo is None):
            raise ValueError("invalid gold answer")
    return order


def run_labeler(
    tasks: dict, round_id: str, gold: dict, out_dir: Path, *, provider: DialogueProvider,
    model: str = "fake", ledger: MonthlyUsageLedger | None = None,
    max_cny: Decimal | None = None, max_requests: int | None = None,
    batch_size: int = 20, min_interval: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    order = _validate_inputs(tasks, round_id, gold)
    predictions = dict.fromkeys(order, "not_run")
    limits = RunLimits(ledger, max_cny, max_requests) if ledger is not None else None
    requests = 0
    last_start = None
    for offset in range(0, len(order), batch_size):
        batch = order[offset:offset + batch_size]
        messages = build_messages(tasks, batch)
        if last_start is not None:
            delay = min_interval - (monotonic() - last_start)
            if delay > 0:
                sleep(delay)
        if limits is not None and not limits.before_request():
            break
        reservation = ledger.reserve(1, datetime.now(UTC))[0] if ledger is not None else None
        last_start = monotonic()
        requests += 1
        if limits is not None:
            limits.completed_requests += 1
        try:
            result = provider.complete(messages)
        except ProviderError:
            if ledger is not None:
                ledger.settle(reservation, None, datetime.now(UTC))
            predictions.update(dict.fromkeys(batch, "invalid"))
        else:
            if ledger is not None:
                ledger.settle(reservation, result.usage, datetime.now(UTC))
            predictions.update(parse_labels(result.text, batch))
    if limits is not None:
        limits.refresh()
    metrics = compute_metrics(gold, predictions)
    system = _SYSTEM.format(labels="\n".join(
        f'{label["code"]} {label["name"]}：{label["hint"]}' for label in tasks["labels"]))
    report = dict(version=1, model=model, prompt_sha256=hashlib.sha256(system.encode()).hexdigest(),
                  labels=predictions, metrics=metrics, passed=is_passed(metrics), requests=requests,
                  spent_cny=limits.spent_cny if limits is not None else "0.00")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = out_dir / f"labeler_{round_id}_{timestamp}.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(report, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = _Parser()
        for name in ("tasks", "gold", "out-dir"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--round", required=True)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--min-interval", type=float, default=3.0)
        parser.add_argument("--real", action="store_true")
        parser.add_argument("--max-cny", type=Decimal)
        parser.add_argument("--max-requests", type=int)
        args = parser.parse_args(argv)
        if not args.real:
            print('{"status":"real_required"}')
            return 2
        if (args.max_cny is None or not args.max_cny.is_finite() or args.max_cny <= 0
                or args.max_requests is None or args.max_requests < 0 or args.batch_size <= 0
                or not math.isfinite(args.min_interval) or args.min_interval < 0):
            raise ValueError("invalid limits")
        settings = load_settings(Path.cwd())
        private_root = settings.private_data_dir.resolve()
        out_dir = args.out_dir.resolve()
        if out_dir == private_root or not out_dir.is_relative_to(private_root):
            print('{"status":"out_dir_not_private"}')
            return 2
        tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
        gold = json.loads(args.gold.read_text(encoding="utf-8"))
        _validate_inputs(tasks, args.round, gold)
        config, secrets = load_dialogue_config(settings, require_style_manifest=False)
        provider = OpenAICompatibleProvider(config=config.provider, api_key=secrets.api_key,
                                            max_output_tokens=config.budget.max_output_tokens)
        ledger = MonthlyUsageLedger(settings.runtime_data_dir / "dialogue", config.budget)
        report = run_labeler(tasks, args.round, gold, out_dir, provider=provider,
                             model=config.provider.model, ledger=ledger,
                             max_cny=args.max_cny, max_requests=args.max_requests,
                             batch_size=args.batch_size, min_interval=args.min_interval)
        metrics = report["metrics"]
        summary = dict(status="ok", requests=report["requests"], spent_cny=report["spent_cny"],
                       kappa=metrics["kappa"], macro_f1=metrics["macro_f1"],
                       reject_rate=metrics["reject_rate"], passed=report["passed"],
                       per_class={code: {"recall": entry["recall"]}
                                  for code, entry in metrics["per_class"].items()})
    except Exception:
        print('{"status":"failed"}')
        return 1
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
