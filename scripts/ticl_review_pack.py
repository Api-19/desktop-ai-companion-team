# ruff: noqa: RUF001
"""Build a blind difference review page and validate its exported claims."""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from companion.privacy.paired_snippets import read_snippets  # noqa: E402
from companion.runtime.settings import SettingsError, load_settings  # noqa: E402

GROUP_FILES = {"a": "base", "b": "variant"}
LIMITS = {"situation": 80, "difference": 160, "applies": 80, "exceptions": 80}
MAX_CLAIMS = 4
MIN_SUPPORT = 2

EXPORT_JS = """function buildExport(meta, notes, claims) {
  const fields = ["situation", "difference", "applies", "exceptions"];
  const noteMap = {};
  notes.forEach((n) => {
    const text = n.text.trim();
    if (text) noteMap[String(n.number)] = text;
  });
  const rows = claims.map((c) => {
    const row = {};
    fields.forEach((f) => { row[f] = c[f].trim(); });
    row.support = Array.from(new Set(c.support.map(Number))).sort((x, y) => x - y);
    return row;
  }).filter((row) => row.support.length || fields.some((f) => row[f]));
  return JSON.stringify({version: 1, group: meta.group, manifest_sha256: meta.manifest,
                         notes: noteMap, claims: rows});
}
"""

GLUE_JS = """function collect() {
  const meta = {group: document.body.dataset.group, manifest: document.body.dataset.manifest};
  const notes = Array.from(document.querySelectorAll("textarea.note")).map(
    (e) => ({number: Number(e.dataset.number), text: e.value}));
  const claims = Array.from(document.querySelectorAll("fieldset.claim")).map((f) => {
    const k = f.dataset.claim;
    const text = (name) => f.querySelector(`textarea[name="${name}-${k}"]`).value;
    return {situation: text("situation"), difference: text("difference"),
            applies: text("applies"), exceptions: text("exceptions"),
            support: Array.from(f.querySelectorAll(`input[name="support-${k}"]:checked`))
              .map((e) => e.value)};
  });
  document.getElementById("export").value = buildExport(meta, notes, claims);
}
document.addEventListener("input", collect);
document.addEventListener("change", collect);
collect();
"""


def status(name: str, code: int, **fields) -> int:
    print(json.dumps({"status": name, **fields}, ensure_ascii=False))
    return code


def h(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path) -> dict:
    return {row["item_id"]: row for row in (
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    )}


def build(args: argparse.Namespace) -> int:
    settings = load_settings(Path.cwd())
    if not args.out_dir.resolve().is_relative_to(settings.test_results_dir.resolve()):
        return status("invalid_out_dir", 2)
    manifest = (args.run_dir / "manifest.json").read_bytes()
    manifest_sha256 = sha256(manifest).hexdigest()
    presented = json.loads(manifest)["presented_item_ids"]
    if not presented:
        return status("insufficient_items", 2)
    items = read_jsonl(args.run_dir / "items.jsonl")
    replies = read_jsonl(args.run_dir / f"replies_{GROUP_FILES[args.group]}.jsonl")
    snippets = {}
    for path in (args.holdout, args.library):
        for snippet in read_snippets(path):
            snippets.setdefault(snippet.snippet_id, snippet)
    missing = sum(item_id not in items or item_id not in replies or item_id not in snippets
                  for item_id in presented)
    if missing:
        return status("invalid_items", 2, missing=missing)

    page = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8"><title>TICL 差异审阅</title></head>',
        f'<body data-manifest="{manifest_sha256}" data-group="{args.group}">',
    ]
    for n, item_id in enumerate(presented, 1):
        item = items[item_id]
        reply = replies[item_id]
        lines = [f'<section class="item" data-number="{n}">', f"<h2>第 {n} 题</h2>",
                 '<div class="context">']
        lines.extend(f"<p>{h(line)}</p>" for line in item["recent_context"])
        lines.extend(["</div>", f'<p class="current">当前消息：{h(item["text"])}</p>',
                      f'<div class="model"><h3>试答</h3><p>{h(reply["reply"])}</p>'])
        if reply["blocked_reason"] is not None:
            lines.append(f'<p class="blocked">拦截：{html.escape(reply["blocked_reason"])}</p>')
        lines.extend(["</div>", '<div class="owner"><h3>真答</h3>'])
        lines.extend(f"<p>{h(line)}</p>" for line in snippets[item_id].owner_reply)
        lines.extend(["</div>",
                      f'<label>差异备注<textarea class="note" data-number="{n}">'
                      '</textarea></label>',
                      "</section>"])
        page.append("\n".join(lines))
    page.append('<section class="claims">')
    for k in range(1, MAX_CLAIMS + 1):
        page.append(f'<fieldset class="claim" data-claim="{k}"><legend>差异说明 {k}</legend>')
        for label, field in zip(("情境", "差异", "适用", "例外"), LIMITS, strict=True):
            page.append(f'<label>{label}<textarea name="{field}-{k}"></textarea></label>')
        for n in range(1, len(presented) + 1):
            page.append(f'<label><input type="checkbox" name="support-{k}" value="{n}">{n}</label>')
        page.append("</fieldset>")
    page.extend(["</section>", '<textarea id="export" readonly></textarea>',
                 '<script id="ticl-export">' + EXPORT_JS + "</script>",
                 "<script>" + GLUE_JS + "</script>", "</body></html>"])
    pack = {"version": 1, "group": args.group, "manifest_sha256": manifest_sha256,
            "items": [{"number": n, "event_id": item_id, "snippet_id": item_id}
                      for n, item_id in enumerate(presented, 1)]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "review.html").write_text("\n".join(page), encoding="utf-8")
    write_json(args.out_dir / "pack.json", pack)
    return status("complete", 0, group=args.group, items=len(presented))


def export_ok(export) -> bool:
    return (
        isinstance(export, dict)
        and isinstance(export.get("group"), str)
        and isinstance(export.get("manifest_sha256"), str)
        and isinstance(export.get("claims"), list)
        and all(
            isinstance(c, dict)
            and all(isinstance(c.get(f), str) for f in LIMITS)
            and isinstance(c.get("support"), list)
            and all(type(x) is int for x in c["support"])
            for c in export["claims"]
        )
    )


def validate(args: argparse.Namespace) -> int:
    settings = load_settings(Path.cwd())
    if not args.out_dir.resolve().is_relative_to(settings.test_results_dir.resolve()):
        return status("invalid_out_dir", 2)
    export = json.loads(args.export.read_text(encoding="utf-8"))
    pack = json.loads(args.pack.read_bytes())
    if not export_ok(export):
        return status("invalid_input", 2)
    if (export["group"] != pack["group"]
            or export["manifest_sha256"] != pack["manifest_sha256"]):
        return status("pack_mismatch", 2)
    claims = export["claims"]
    if not claims:
        return status("no_claims", 2)
    if len(claims) > MAX_CLAIMS:
        return status("too_many_claims", 2, claims=len(claims))
    event_ids = {item["number"]: item["event_id"] for item in pack["items"]}
    errors = []
    normalized = []
    for k, claim in enumerate(claims, 1):
        values = {}
        for field, limit in LIMITS.items():
            value = " ".join(claim[field].split())
            values[field] = value
            if not value:
                errors.append({"claim": k, "field": field, "reason": "empty"})
            elif len(value) > limit:
                errors.append({"claim": k, "field": field, "reason": "too_long"})
        distinct = set(claim["support"])
        if len(distinct) < MIN_SUPPORT:
            errors.append({"claim": k, "field": "support", "reason": "too_few"})
        if any(number not in event_ids for number in distinct):
            errors.append({"claim": k, "field": "support", "reason": "unknown"})
        normalized.append((values, distinct))
    if errors:
        return status("invalid_claims", 2, errors=errors)
    lines = ["# TICL 差异说明草稿：待主人逐条确认，未写入习惯卡",
             f'# manifest_sha256={pack["manifest_sha256"]} group={pack["group"]}']
    for k, (values, distinct) in enumerate(normalized, 1):
        ids = ", ".join(event_ids[number] for number in sorted(distinct))
        lines.extend(["", f"【差异 {k}】", f'情境：{values["situation"]}',
                      f'差异（试答→真答）：{values["difference"]}',
                      f'适用：{values["applies"]}', f'例外：{values["exceptions"]}',
                      f"训练 ID：{ids}"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "card_addendum.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status("complete", 0, claims=len(claims))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    for option in ("--run-dir", "--library", "--holdout", "--out-dir"):
        build_parser.add_argument(option, type=Path, required=True)
    build_parser.add_argument("--group", choices=("a", "b"), required=True)
    validate_parser = commands.add_parser("validate")
    for option in ("--export", "--pack", "--out-dir"):
        validate_parser.add_argument(option, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            return build(args)
        return validate(args)
    except SettingsError:
        return status("configuration_unavailable", 2)
    except (OSError, ValueError, KeyError, TypeError):
        return status("invalid_input", 2)


if __name__ == "__main__":
    raise SystemExit(main())
