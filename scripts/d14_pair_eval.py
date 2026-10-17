# ruff: noqa: RUF001
"""Evaluate two snippet cards with isolated retrieval and blind paired scoring."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from companion.contracts import DialogueRequest  # noqa: E402
from companion.dialogue import build_real_orchestrator, load_dialogue_config  # noqa: E402
from companion.dialogue.context import prepare_dialogue_input  # noqa: E402
from companion.dialogue.models import (  # noqa: E402
    BudgetConfig,
    DialogueConfig,
    ProviderConfig,
    StyleRuntimeSources,
)
from companion.dialogue.observer import TurnObservation, TurnObserver  # noqa: E402
from companion.dialogue.orchestrator import DialogueOrchestrator, SnippetStyleSource  # noqa: E402
from companion.dialogue.shared_terms import (  # noqa: E402
    SharedTerm,
    load_shared_terms,
)
from companion.dialogue.usage import MonthlyUsageLedger  # noqa: E402
from companion.privacy.paired_snippets import PairedSnippet, read_snippets  # noqa: E402
from companion.runtime.settings import load_settings  # noqa: E402
from companion.style.snippet_evidence import SnippetStyleAdapter  # noqa: E402
from scripts import style_reframe_eval as ev  # noqa: E402

GROUPS = ("base", "variant")
STYLE_VERSION = "d14-pair-eval-v1"
RECHECK_SIZE = 6
STAGE_TOTALS = {"screen": 8, "combo": 30}
FAILURES = ("接法", "措辞", "理解", "超时拦截")
ERROR_TYPES = ("接法", "措辞", "理解")

FORMAT_JS = '''function resultLine(rows) {
  const order = ["接法", "措辞", "理解", "超时拦截"];
  const pick = (types) => order.filter((t) => types.includes(t)).join("");
  return rows.map((row, i) => {
    const a = pick(row.fa), b = pick(row.fb);
    const tail = a || b ? "/" + (a ? "A" + a : "") + (b ? "B" + b : "") : "";
    return String(i + 1) + (row.pref || "?") + "/" + (row.ok || "?") + tail;
  }).join(",");
}'''

GLUE_JS = '''function collect() {
  const rows = [];
  document.querySelectorAll("section.item").forEach((s) => {
    const n = s.dataset.number;
    const many = (name) => Array.from(
      s.querySelectorAll(`input[name="${name}"]:checked`)).map((e) => e.value);
    const one = (name) => many(name)[0] || "";
    rows.push({pref: one(`pref-${n}`), ok: one(`ok-${n}`),
               fa: many(`fail-${n}-A`), fb: many(`fail-${n}-B`)});
  });
  document.getElementById("result").value = resultLine(rows);
}
document.addEventListener("change", collect);
collect();'''


def status(name: str, code: int, **fields) -> int:
    print(json.dumps({"status": name, **fields}, ensure_ascii=False))
    return code


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def line_key(text: str) -> str:
    text = normalize(text)
    return text[2:] if text.startswith(("她:", "他:")) else text


def bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def filter_library(library: Sequence[PairedSnippet], occupied: set[str],
                   eval_snippets: Sequence[PairedSnippet]):
    eval_replies = {normalize("".join(s.owner_reply)) for s in eval_snippets} - {""}
    eval_lines = {
        text for s in eval_snippets
        for text in [*(line_key(t) for t in s.context_turns),
                     *(normalize(t) for t in s.partner_turns)]
        if len(text) >= 5
    }
    eval_partners = [normalize("".join(s.partner_turns)) for s in eval_snippets]
    removed = dict.fromkeys(("occupied", "same_reply", "context_overlap", "near_duplicate"), 0)
    kept = []
    for snippet in library:
        lines = {
            text for text in [*(line_key(t) for t in snippet.context_turns),
                              *(normalize(t) for t in snippet.partner_turns),
                              *(normalize(t) for t in snippet.owner_reply)]
            if len(text) >= 5
        }
        partner = normalize("".join(snippet.partner_turns))
        if snippet.snippet_id in occupied:
            reason = "occupied"
        elif normalize("".join(snippet.owner_reply)) in eval_replies:
            reason = "same_reply"
        elif lines & eval_lines:
            reason = "context_overlap"
        elif any(
            (partner and partner == other)
            or (len(partner) >= 12 and len(other) >= 12
                and 5 * len(bigrams(partner) & bigrams(other))
                >= 4 * len(bigrams(partner) | bigrams(other)))
            for other in eval_partners
        ):
            reason = "near_duplicate"
        else:
            kept.append(snippet)
            continue
        removed[reason] += 1
    return kept, removed


def build_fake_orchestrator(
    root: Path, *, snippet_style: SnippetStyleSource,
    snippet_habit_card: str, turn_observer: TurnObserver,
    snippet_shared_terms: Sequence[SharedTerm] | None = None,
    snippet_session_check: bool = False,
) -> DialogueOrchestrator:
    config = DialogueConfig(
        provider=ProviderConfig(
            base_url="https://example.invalid/v1", model="fake", timeout_seconds=1
        ),
        budget=BudgetConfig(
            monthly_cap_cny=Decimal("100.00"),
            worst_case_request_cny=Decimal("0.10"),
            input_cny_per_million=Decimal("1"),
            output_cny_per_million=Decimal("2"),
            max_output_tokens=256,
        ),
        style=StyleRuntimeSources(
            candidate_source=root / "candidates",
            expected_count=1,
            active_style_source=root / "active",
            active_manifest_source=root / "manifest",
            expected_active_manifest_digest="a" * 64,
            calibration_source=root / "calibration",
            expected_calibration_fingerprint="b" * 64,
            private_data_dir=root,
            true_name_terms=(),
            approved_nicknames=(),
        ),
    )
    return DialogueOrchestrator(
        config=config,
        provider=ev.FakeProvider(),
        style_adapter=None,
        snippet_style=snippet_style,
        snippet_habit_card=snippet_habit_card,
        snippet_shared_terms=snippet_shared_terms,
        snippet_session_check=snippet_session_check,
        ledger=MonthlyUsageLedger(root / "ledger", config.budget),
        clock=lambda: datetime.now(UTC),
        turn_observer=turn_observer,
    )


def first_prompt_observer(group_info: dict) -> TurnObserver:
    def observe(obs: TurnObservation) -> None:
        if group_info["system_prompt_sha256"] is None and obs.system_prompt_sha256 is not None:
            group_info["system_prompt_sha256"] = obs.system_prompt_sha256
    return observe


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def read_jsonl(path: Path) -> dict:
    return {row["item_id"]: row for row in (
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    )}


def h(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def write_blind(out_dir: Path, item_ids: Sequence[str], items: dict, replies: dict,
                *, kind: str, seed: int, rng: random.Random) -> None:
    key = {"kind": kind, "seed": seed, "items": []}
    page = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8"><title>盲评</title>',
        "</head><body>",
    ]
    for number, item_id in enumerate(item_ids, 1):
        order = list(GROUPS)
        rng.shuffle(order)
        key["items"].append({"number": number, "item_id": item_id,
                             "A": order[0], "B": order[1]})
        item = items[item_id]
        section = [f'<section class="item" data-number="{number}">',
                   f"<h2>第 {number} 题</h2>", '<div class="context">']
        section.extend(f"<p>{h(line)}</p>" for line in item["recent_context"])
        section.extend(["</div>", f'<p class="current">当前消息：{h(item["text"])}</p>'])
        for label, group in zip("AB", order, strict=True):
            row = replies[group][item_id]
            section.append(f'<div class="reply" data-label="{label}"><h3>{label}</h3>'
                           f'<p>{h(row["reply"])}</p>')
            if row["blocked_reason"] is not None:
                section.append('<p class="blocked">拦截：'
                               f'{html.escape(row["blocked_reason"])}</p>')
            section.append("</div>")
        for name, choices in (
            ("pref", (("A", "更像 A"), ("B", "更像 B"), ("并列", "并列"),
                      ("都不像", "都不像"), ("超出语境", "超出语境"))),
            ("ok", (("A", "A"), ("B", "B"), ("AB", "都可"), ("无", "都不可"))),
        ):
            section.extend(
                f'<label><input type="radio" name="{name}-{number}" '
                f'value="{value}">{text}</label>' for value, text in choices
            )
        section.extend(
            f'<label><input type="checkbox" name="fail-{number}-{label}" '
            f'value="{failure}">{label} {failure}</label>'
            for label in "AB" for failure in FAILURES
        )
        section.append("</section>")
        page.append("\n".join(section))
    page.extend([
        '<textarea id="result" readonly></textarea>',
        '<script id="d14-format">' + FORMAT_JS + "</script>",
        "<script>" + GLUE_JS + "</script>", "</body></html>",
    ])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blind.html").write_text("\n".join(page), encoding="utf-8")
    write_json(out_dir / "answer_key.json", key)


def run(args: argparse.Namespace) -> int:
    active_groups = ("base",) if args.only_base else GROUPS
    if not args.only_base and (args.variant_card is None or args.variant_strata is None):
        return status("invalid_input", 2)
    max_cny = None
    if args.real or args.max_cny is not None:
        try:
            max_cny = Decimal(args.max_cny) if args.max_cny is not None else None
        except InvalidOperation:
            raise ValueError("invalid max cny") from None
        if max_cny is None or not max_cny.is_finite() or max_cny <= 0:
            raise ValueError("invalid max cny")
    cards, groups, terms = {}, {}, {}
    for group in active_groups:
        data = getattr(args, f"{group}_card").read_bytes()
        text = data.decode("utf-8")
        if not text.strip():
            raise ValueError("empty habit card")
        cards[group] = text
        terms_path = getattr(args, f"{group}_terms")
        terms[group] = load_shared_terms(terms_path) if terms_path is not None else None
        groups[group] = {
            "card_sha256": sha256(data).hexdigest(),
            "terms_sha256": sha256(terms_path.read_bytes()).hexdigest()
            if terms_path is not None else None,
            "session_check": getattr(args, f"{group}_session_check"),
            "general_strata": getattr(args, f"{group}_strata"),
            "system_prompt_sha256": None, "completed": 0, "blocked": {},
        }
    pools_data = args.pools.read_bytes()
    pools = json.loads(pools_data)["pools"]
    if (args.pool == "unassigned" or args.pool not in pools
            or (args.items_file is None and not pools[args.pool])):
        return status("invalid_pool", 2)
    split_data = args.split.read_bytes()
    split = {event["event_id"]: event for event in json.loads(split_data)["events"]}
    occupied_events = [event for pool, events in pools.items() if pool != "unassigned"
                       for event in events]
    if (len(set(occupied_events)) != len(occupied_events)
            or any(event not in split for event in occupied_events)):
        return status("invalid_pools", 2)
    holdout = read_snippets(args.holdout)
    library = read_snippets(args.library)
    by_id = {}
    for snippet in [*holdout, *library]:
        by_id.setdefault(snippet.snippet_id, snippet)
    items, eval_snippets = [], []
    source_rows = None
    items_data = None
    missing = over_budget = 0
    if args.items_file is not None:
        try:
            items_data = args.items_file.read_bytes()
            source_rows = json.loads(items_data)
        except (OSError, ValueError):
            return status("invalid_items", 2)
        if not isinstance(source_rows, list):
            return status("invalid_items", 2)
        seen = set()
        for row in source_rows:
            if (not isinstance(row, dict)
                    or not isinstance(row.get("id"), str) or not row["id"].strip()
                    or row["id"] in seen
                    or not isinstance(row.get("text"), str) or not row["text"].strip()
                    or not isinstance(row.get("recent_context"), list)
                    or len(row["recent_context"]) > 10
                    or any(not isinstance(line, str) for line in row["recent_context"])):
                return status("invalid_items", 2)
            seen.add(row["id"])
            items.append(ev.EvalItem(row["id"], row["text"], row["recent_context"]))
    else:
        for event_id in pools[args.pool]:
            snippet = by_id.get(event_id)
            if snippet is None:
                missing += 1
                continue
            selected = ev.items_from_snippets([snippet])
            if not selected:
                over_budget += 1
                continue
            items.extend(selected)
            eval_snippets.append(snippet)
    if missing or over_budget:
        return status("invalid_items", 2, missing=missing, over_budget=over_budget)
    planned = len(active_groups) * len(items)
    if not args.real and planned > args.max_requests:
        return status("request_limit_exceeded", 2, planned_requests=planned)
    if args.real:
        settings = load_settings(Path.cwd())
        if not args.out_dir.resolve().is_relative_to(settings.test_results_dir.resolve()):
            return status("invalid_out_dir", 2)
        config, secrets = load_dialogue_config(settings, require_style_manifest=False)
    occupied = {sid for event in occupied_events for sid in [event, *split[event]["snippet_ids"]]}
    kept, removed = filter_library(library, occupied, eval_snippets)
    library_sha256 = sha256("".join(
        json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n" for s in kept
    ).encode("utf-8")).hexdigest()
    style_snippets = [e for s in kept if (e := s.as_style_snippet()) is not None]
    replies = {group: [] for group in active_groups}
    with tempfile.TemporaryDirectory(prefix="d14-pair-") as temporary:
        orchestrators = {}
        for group in active_groups:
            adapter = SnippetStyleAdapter(
                snippets=style_snippets, partner_user_id=args.partner_user_id,
                style_version=STYLE_VERSION, clock=lambda: datetime.now(UTC),
                general_strata=groups[group]["general_strata"],
            )
            observer = first_prompt_observer(groups[group])
            if args.real:
                orchestrators[group] = build_real_orchestrator(
                    settings, config, secrets, trusted_user_id=args.partner_user_id,
                    snippet_style=adapter, snippet_habit_card=cards[group], turn_observer=observer,
                    snippet_shared_terms=terms[group],
                    snippet_session_check=groups[group]["session_check"],
                )
            else:
                orchestrators[group] = build_fake_orchestrator(
                    Path(temporary) / group, snippet_style=adapter,
                    snippet_habit_card=cards[group], turn_observer=observer,
                    snippet_shared_terms=terms[group],
                    snippet_session_check=groups[group]["session_check"],
                )
        limits = None
        if args.real:
            orchestrators["base"].ensure_paid_capacity(1)
            limits = ev.RunLimits(
                MonthlyUsageLedger(settings.runtime_data_dir / "dialogue", config.budget),
                max_cny, args.max_requests,
            )
        runners = {group: ev.snippet_runner(orchestrators[group], args.partner_user_id)
                   for group in active_groups}
        for item in items:
            for group in active_groups:
                if limits is not None and not limits.before_request():
                    break
                result = runners[group](item)
                if limits is not None:
                    limits.completed_requests += 1
                replies[group].append({"item_id": item.item_id, "reply": result.response.text,
                                       "blocked_reason": result.response.blocked_reason})
            if limits is not None and limits.stopped is not None:
                break
        if limits is not None:
            if limits.stopped != "spend_cap":
                limits.refresh()
            completed, spent, stopped = (
                limits.completed_requests, limits.spent_cny, limits.stopped,
            )
        else:
            completed = sum(map(len, replies.values()))
            spent, stopped = "0.00", None
    item_rows = []
    for item in items:
        prepared = prepare_dialogue_input(DialogueRequest(
            user_id="d14-blind", text=item.text, recent_context=item.recent_context,
        ))
        item_rows.append({"item_id": item.item_id, "text": prepared.text,
                          "recent_context": list(prepared.recent_context)})
    if source_rows is not None:
        item_rows = [{**source, **prepared} for prepared, source in
                     zip(item_rows, source_rows, strict=True)]
    complete = min(map(len, replies.values()))
    presented_ids = [item.item_id for item in items[:complete]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "items.jsonl", item_rows)
    for group in active_groups:
        write_jsonl(args.out_dir / f"replies_{group}.jsonl", replies[group])
        groups[group]["completed"] = len(replies[group])
        groups[group]["blocked"] = dict(Counter(
            row["blocked_reason"] for row in replies[group] if row["blocked_reason"] is not None
        ))
    if not args.only_base:
        write_blind(
            args.out_dir, presented_ids,
            {item.item_id: row for item, row in zip(items, item_rows, strict=True)},
            {group: {row["item_id"]: row for row in rows} for group, rows in replies.items()},
            kind="pair", seed=args.seed, rng=random.Random(args.seed),
        )
    mode = "real" if args.real else "fake"
    manifest = {
        "version": 1, "mode": mode, "seed": args.seed, "pool": args.pool,
        "pools_sha256": sha256(pools_data).hexdigest(),
        "split_sha256": sha256(split_data).hexdigest(), "library_sha256": library_sha256,
        "library_before": len(library), "library_after": len(kept), "removed": removed,
        "style_evidence_excluded": len(kept) - len(style_snippets), "groups": groups,
        "item_ids": [item.item_id for item in items], "presented_item_ids": presented_ids,
        "planned_requests": planned, "completed_requests": completed,
        "spent_cny": spent, "stopped": stopped,
    }
    if items_data is not None:
        manifest["items_file_sha256"] = sha256(items_data).hexdigest()
    if args.only_base:
        manifest["only_base"] = True
    write_json(args.out_dir / "manifest.json", manifest)
    if stopped is not None:
        return status("stopped", 3, reason=stopped, completed_requests=completed, spent_cny=spent)
    return status("complete", 0, mode=mode, planned_requests=planned,
                  completed_requests=completed, spent_cny=spent)


def recheck(args: argparse.Namespace) -> int:
    settings = load_settings(Path.cwd())
    if (args.out_dir.resolve() == args.source_dir.resolve()
            or not args.out_dir.resolve().is_relative_to(settings.test_results_dir.resolve())):
        return status("invalid_out_dir", 2)
    key_data = (args.source_dir / "answer_key.json").read_bytes()
    key = json.loads(key_data)
    if key["kind"] != "pair":
        raise ValueError("invalid source kind")
    items = read_jsonl(args.source_dir / "items.jsonl")
    replies = {group: read_jsonl(args.source_dir / f"replies_{group}.jsonl") for group in GROUPS}
    if len(key["items"]) < RECHECK_SIZE:
        return status("insufficient_items", 2)
    rng = random.Random(args.seed)
    picked = rng.sample([row["item_id"] for row in key["items"]], RECHECK_SIZE)
    write_blind(args.out_dir, picked, items, replies, kind="recheck", seed=args.seed, rng=rng)
    write_json(args.out_dir / "manifest.json", {
        "version": 1, "kind": "recheck", "seed": args.seed,
        "source_answer_key_sha256": sha256(key_data).hexdigest(), "item_ids": picked,
    })
    return status("complete", 0, mode="recheck", items=RECHECK_SIZE)


def parse_results(path: Path, key: dict) -> dict:
    failure_pattern = r"(?:接法)?(?:措辞)?(?:理解)?(?:超时拦截)?"
    pattern = re.compile(
        r"([1-9]\d*)(A|B|并列|都不像|超出语境)/(AB|A|B|无)"
        rf"(?:/(?:A({failure_pattern}))?(?:B({failure_pattern}))?)?"
    )
    text = re.sub(r"\s", "", path.read_text(encoding="utf-8").replace("，", ","))
    by_number = {row["number"]: row for row in key["items"]}
    results, numbers = {}, set()
    for part in text.split(","):
        match = pattern.fullmatch(part)
        if match is None or part.endswith("/"):
            raise ValueError("invalid result")
        number, preference, acceptable, fa, fb = match.groups()
        number = int(number)
        if number in numbers or fa == "" or fb == "":
            raise ValueError("invalid result")
        numbers.add(number)
        row = by_number[number]
        results[row["item_id"]] = {
            "preference": row[preference] if preference in ("A", "B") else preference,
            "acceptable": {row[label] for label in acceptable if label in "AB"},
            "failures": {row[label]: {f for f in FAILURES if f in types}
                         for label, types in (("A", fa or ""), ("B", fb or ""))},
        }
    if numbers != set(by_number):
        raise ValueError("invalid result numbers")
    return results


def score(args: argparse.Namespace) -> int:
    if ((args.require_wins and args.stage == "combo")
            or (args.stage == "screen" and (args.recheck_results or args.recheck_key))
            or (args.recheck_results is None) != (args.recheck_key is None)):
        return status("invalid_input", 2)
    key = json.loads(args.answer_key.read_bytes())
    if key["kind"] != "pair":
        raise ValueError("invalid key kind")
    results = parse_results(args.results, key)
    recheck_summary = None
    if args.recheck_results is not None:
        recheck_key = json.loads(args.recheck_key.read_bytes())
        if recheck_key["kind"] != "recheck" or len(recheck_key["items"]) != RECHECK_SIZE:
            raise ValueError("invalid recheck key")
        rechecked = parse_results(args.recheck_results, recheck_key)
        recheck_summary = {"total": RECHECK_SIZE, "agree": sum(
            row["preference"] == results[item_id]["preference"]
            for item_id, row in rechecked.items()
        )}
    preferences = Counter(row["preference"] for row in results.values())
    acceptable = {g: sum(g in row["acceptable"] for row in results.values()) for g in GROUPS}
    failures = {g: {f: sum(f in row["failures"][g] for row in results.values())
                    for f in FAILURES} for g in GROUPS}
    errors = {g: sum(bool(set(ERROR_TYPES) & row["failures"][g]) for row in results.values())
              for g in GROUPS}
    total, wins, losses = len(key["items"]), preferences["variant"], preferences["base"]
    checks = {}
    if total != STAGE_TOTALS[args.stage]:
        verdict = "insufficient_items"
    else:
        if args.stage == "screen":
            checks["errors_reduced_by_2"] = errors["base"] - errors["variant"] >= 2
            checks["acceptable_not_lower"] = acceptable["variant"] >= acceptable["base"]
            if args.require_wins:
                checks["wins_at_least_5"] = wins >= 5
        else:
            checks["wins_at_least_18"] = wins >= 18
            checks["wins_over_losses"] = wins > losses
            checks["acceptable_not_lower"] = acceptable["variant"] >= acceptable["base"]
            checks["comprehension_not_higher"] = (
                failures["variant"]["理解"] <= failures["base"]["理解"]
            )
            if recheck_summary is not None:
                checks["recheck_agree_at_least_5"] = recheck_summary["agree"] >= 5
        verdict = ("fail" if not all(checks.values()) else
                   "incomplete" if args.stage == "combo" and recheck_summary is None else "pass")
    return status(
        "scored", 0, stage=args.stage, total=total, wins=wins, losses=losses,
        ties=preferences["并列"], neither=preferences["都不像"],
        out_of_context=preferences["超出语境"], acceptable=acceptable, failures=failures,
        errors=errors, recheck=recheck_summary, checks=checks, verdict=verdict,
        manual_checks=["越权或串会话为 0"] if args.stage == "combo" else [],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    for name in ("pools", "split", "holdout", "library", "base-card", "out-dir"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--variant-card", type=Path)
    run_parser.add_argument("--items-file", type=Path)
    run_parser.add_argument("--only-base", action="store_true")
    for name in ("pool", "partner-user-id"):
        run_parser.add_argument(f"--{name}", required=True)
    for group in GROUPS:
        run_parser.add_argument(f"--{group}-strata", choices=("length", "lines"),
                                required=group == "base")
        run_parser.add_argument(f"--{group}-terms", type=Path)
        run_parser.add_argument(f"--{group}-session-check", action="store_true")
    run_parser.add_argument("--max-requests", type=int, required=True)
    run_parser.add_argument("--max-cny")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--real", action="store_true")
    recheck_parser = commands.add_parser("recheck")
    for name in ("source-dir", "out-dir"):
        recheck_parser.add_argument(f"--{name}", type=Path, required=True)
    recheck_parser.add_argument("--seed", type=int, required=True)
    score_parser = commands.add_parser("score")
    for name in ("results", "answer-key"):
        score_parser.add_argument(f"--{name}", type=Path, required=True)
    score_parser.add_argument("--stage", choices=tuple(STAGE_TOTALS), required=True)
    score_parser.add_argument("--require-wins", action="store_true")
    for name in ("recheck-results", "recheck-key"):
        score_parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args(argv)
    if args.command == "score":
        try:
            return score(args)
        except (OSError, ValueError, KeyError, TypeError):
            return status("invalid_input", 2)
    try:
        return run(args) if args.command == "run" else recheck(args)
    except (OSError, ValueError):
        return status("configuration_unavailable", 2)
    except Exception:
        return status("run_failed", 1)


if __name__ == "__main__":
    raise SystemExit(main())
