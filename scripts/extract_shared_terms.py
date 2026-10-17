# ruff: noqa: RUF001
"""Extract shared nickname/in-joke candidates for local owner review (D14 item 3).

build:    train-pool snippets (minus D14 screening pools) -> candidates.json + review.html
validate: owner's exported JSON + candidates.json -> dictionary.json (no example lines)

Local only: no network, no provider. stdout carries status and counts, never terms.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from companion.privacy.paired_snippets import PairedSnippet  # noqa: E402
from companion.privacy.veiled import load_veiled_terms  # noqa: E402
from companion.runtime.settings import load_settings  # noqa: E402

EXCLUDED_POOLS = frozenset({"combo", "shape_screen", "ticl_screen", "dict_screen", "rcpem_m15f"})
DEFAULT_VEILED = ROOT / "config/privacy/veiled_terms.json"
MAX_CANDIDATES = 30
MAX_EXAMPLES = 2
MIN_USES = 3
MIN_DAYS = 2
MIN_LEN, MAX_CJK_LEN, MAX_LATIN_LEN = 2, 6, 12
FRAGMENT_RATIO = 0.8
# A part is "loose" when the term accounts for at most this share of the part's uses.
LOOSE_SHARE = 0.5
GENERAL_SMOOTHING = 1
MAX_FIELD_CHARS = 120
SPEAKERS = ("owner", "partner", "both")
PERMISSIONS = ("allowed", "understand_only", "disabled", "delete")
ENTRY_KEYS = frozenset({"id", "term", "meaning", "occasion", "speaker", "addressee",
                        "permission"})

# Particles, pronouns and the most frequent function characters split CJK runs.
COMMON_CHARS = frozenset(
    "的了吗呢吧啊呀哦噢嘛啦哈嗯呜哇诶欸哎呵嘿嘻哼嗷"
    "我你他她它们是在不也就都还和很又才这那哪么个"
)
# Everyday words: masked before n-grams are formed, so they never become candidates
# and never glue neighbouring characters into cross-word fragments.
_COMMON_WORDS_TEXT = """
什么 怎么 为什么 怎么样 不知道 知道 今天 明天 昨天 后天 现在 刚才 刚刚 时候 已经
可以 可能 应该 觉得 感觉 喜欢 需要 想要 一起 一下 一点 一个 一样 一直 一会 有点
没有 没事 不是 不要 不会 不用 不行 还是 但是 因为 所以 如果 然后 而且 或者 虽然
就是 真的 其实 好像 只是 还有 这样 那样 这么 那么 这里 那里 这个 那个 哪个 哪里
晚安 早安 早上 晚上 中午 下午 上午 睡觉 起床 吃饭 回家 上班 下班 上课 下课 工作
学校 公司 东西 事情 问题 时间 地方 朋友 同学 老师 妈妈 爸爸 家里 外面 手机 消息
电话 视频 照片 图片 回来 出去 过来 过去 看看 看到 听到 说话 告诉 记得 忘了 开心
难过 生气 害怕 担心 辛苦 好的 好吧 对的 是的 行吧 收到 谢谢 不客气 对不起 没关系
哈哈 嘿嘿 嘻嘻 呜呜 嗯嗯 好好 多少 几点 知道了 我们 你们 他们 她们 自己 别人 大家
怎么办 干嘛 在吗 在干嘛 吃了 睡了 到了 走了 来了 去了 好了 完了 对了 算了 等等
休息 洗澡 天气 下雨 外卖 好吃 好看 好玩 厉害 可爱 漂亮 舒服 难受 太好 非常 特别
"""
COMMON_WORDS = tuple(sorted(set(_COMMON_WORDS_TEXT.split()), key=lambda w: (-len(w), w)))

_PLACEHOLDER = re.compile(r"\[[^\[\]\s]{1,10}\]")
_MASK = re.compile("|".join(map(re.escape, COMMON_WORDS)))
_CJK_RUN = re.compile(r"[一-鿿]+")
_LATIN = re.compile(r"(?<![a-z])[a-z]+(?![a-z])")

EXPORT_JS = """function buildExport(candidatesSha, rows) {
  const entries = rows.map((r) => ({
    id: r.id, term: r.term, meaning: r.meaning.trim(), occasion: r.occasion.trim(),
    speaker: r.speaker, addressee: r.addressee.trim(), permission: r.permission,
  }));
  return JSON.stringify({version: 1, candidates_sha256: candidatesSha, entries: entries});
}
"""

GLUE_JS = """function collect() {
  const rows = Array.from(document.querySelectorAll("section.term")).map((s) => {
    const id = s.dataset.id;
    const text = (name) => s.querySelector(`input[name="${name}"]`).value;
    const pick = (name) => {
      const e = s.querySelector(`input[name="${name}-${id}"]:checked`);
      return e ? e.value : "";
    };
    return {id: id, term: s.dataset.term, meaning: text("meaning"),
            occasion: text("occasion"), speaker: pick("speaker"),
            addressee: text("addressee"), permission: pick("permission")};
  });
  document.getElementById("export").value = buildExport(
    document.body.dataset.candidates, rows);
}
function copyExport() {
  const box = document.getElementById("export");
  box.select();
  navigator.clipboard.writeText(box.value);
}
document.addEventListener("input", collect);
document.addEventListener("change", collect);
collect();
"""

SPEAKER_LABELS = {"owner": "主人", "partner": "她", "both": "双方"}
PERMISSION_LABELS = {"allowed": "可用", "understand_only": "仅理解", "disabled": "停用",
                     "delete": "删除"}


def status(name: str, code: int, **fields) -> int:
    print(json.dumps({"status": name, **fields}, ensure_ascii=False))
    return code


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def message_terms(line: str, min_len: int = MIN_LEN) -> set[str]:
    """Distinct n-grams in one message (CJK min_len-6 chars, Latin whole words)."""
    text = normalize(_PLACEHOLDER.sub(" ", line))
    terms = {word for word in _LATIN.findall(text) if MIN_LEN <= len(word) <= MAX_LATIN_LEN}
    for run in _CJK_RUN.findall(_MASK.sub(" ", text)):
        for part in "".join(" " if c in COMMON_CHARS else c for c in run).split():
            for size in range(min_len, min(MAX_CJK_LEN, len(part)) + 1):
                terms.update(part[i:i + size] for i in range(len(part) - size + 1))
    return terms


def turns(snippet: PairedSnippet):
    yield from (("partner", line) for line in snippet.partner_turns)
    yield from (("owner", line) for line in snippet.owner_reply)


def allowed_snippet_ids(split: dict, pools: dict) -> set[str] | None:
    """Snippet ids of train events outside the excluded D14 pools; None if pools are invalid."""
    events = {event["event_id"]: event for event in split["events"]}
    if any(event not in events for members in pools.values() for event in members):
        return None
    excluded_events = {event for pool in EXCLUDED_POOLS for event in pools.get(pool, [])}

    def members(event_id: str) -> set[str]:
        return {event_id, *events[event_id]["snippet_ids"]}

    allowed = set().union(*(members(e) for e, event in events.items()
                            if event["pool"] == "train" and e not in excluded_events))
    return allowed - set().union(*(members(e) for e in excluded_events))


def read_allowed(paths: Sequence[Path], allowed: set[str]) -> list[PairedSnippet]:
    """Validate only allowed lines; other snippets are skipped by id before validation."""
    found: dict[str, PairedSnippet] = {}
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["snippet_id"] in allowed and row["snippet_id"] not in found:
                    found[row["snippet_id"]] = PairedSnippet.model_validate(row)
    return sorted(found.values(), key=lambda s: (s.started_at, s.snippet_id))


def load_general_freq(path: Path) -> dict[str, int]:
    """jieba-format "word freq [tag]" lines, keyed by normalized word (highest freq wins)."""
    freqs: dict[str, int] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            fields = line.split()
            if len(fields) >= 2:
                word = normalize(fields[0])
                freqs[word] = max(freqs.get(word, 0), int(fields[1]))
    return freqs


def default_general_dict() -> Path | None:
    spec = find_spec("jieba")
    return None if spec is None else Path(spec.origin).parent / "dict.txt"


def free_combination(term: str, total: Counter) -> bool:
    """True when some split of the term has both halves mostly used apart from it."""
    limit = total[term] / LOOSE_SHARE
    return any(total[term[:i]] >= limit and total[term[i:]] >= limit
               for i in range(1, len(term)))


class Blocklist:
    def __init__(self, veiled_path: Path, exclude_terms: Sequence[str]) -> None:
        self.lexicon = load_veiled_terms(veiled_path)
        raw = json.loads(veiled_path.read_text(encoding="utf-8-sig"))
        self.veiled_terms = [normalize(term) for tier in ("high", "suspect")
                             for group in raw[tier].values() for term in group]
        self.exclude_terms = [normalize(term) for term in exclude_terms if term.strip()]
        self.exclude_bigrams = {term[i:i + 2] for term in self.exclude_terms
                                for i in range(len(term) - 1)}

    def veiled(self, text: str) -> bool:
        hits = self.lexicon.scan(text)
        return bool(hits.high or hits.suspect)

    def veiled_term(self, term: str) -> bool:
        return self.veiled(term) or any(term in word for word in self.veiled_terms)

    def excluded_term(self, term: str) -> bool:
        return any(term[i:i + 2] in self.exclude_bigrams for i in range(len(term) - 1))

    def excluded_line(self, line: str) -> bool:
        text = normalize(line)
        return any(term in text for term in self.exclude_terms)


def extract(snippets: Sequence[PairedSnippet], blocklist: Blocklist,
            general: dict[str, int]) -> tuple[list[dict], dict]:
    counts: dict[str, Counter] = {"owner": Counter(), "partner": Counter()}
    days: dict[str, set] = {}
    for snippet in snippets:
        day = snippet.started_at.date()
        for speaker, line in turns(snippet):
            # Single characters are counted only to judge free combinations.
            for term in message_terms(line, min_len=1):
                counts[speaker][term] += 1
                days.setdefault(term, set()).add(day)
    total = counts["owner"] + counts["partner"]
    frequent = {term for term, uses in total.items()
                if len(term) >= MIN_LEN and uses >= MIN_USES and len(days[term]) >= MIN_DAYS}
    # A fragment yields to a longer frequent term that keeps most of its uses.
    covered: dict[str, int] = {}
    for term in frequent:
        for size in range(MIN_LEN, len(term)):
            for i in range(len(term) - size + 1):
                part = term[i:i + size]
                covered[part] = max(covered.get(part, 0), total[term])
    stats = Counter()
    ranked = []
    for term in frequent:
        if covered.get(term, 0) >= FRAGMENT_RATIO * total[term]:
            continue
        if blocklist.veiled_term(term):
            stats["dropped_veiled"] += 1
            continue
        if blocklist.excluded_term(term):
            stats["dropped_excluded"] += 1
            continue
        if term not in general and free_combination(term, total):
            stats["dropped_free_combination"] += 1
            continue
        shared = 2 if counts["owner"][term] and counts["partner"][term] else 1
        usage = total[term] * shared * min(len(days[term]), 5)
        score = usage / (general.get(term, 0) + GENERAL_SMOOTHING)
        ranked.append((-score, -len(term), term))
    ranked.sort()
    stats["dropped_over_limit"] = max(0, len(ranked) - MAX_CANDIDATES)
    candidates = []
    for index, (_, _, term) in enumerate(ranked[:MAX_CANDIDATES], start=1):
        term_days = sorted(days[term])
        candidates.append({
            "id": f"t{index:02d}", "term": term,
            "owner_count": counts["owner"][term], "partner_count": counts["partner"][term],
            "general_freq": general.get(term, 0),
            "first_seen": term_days[0].isoformat(), "last_seen": term_days[-1].isoformat(),
            "examples": examples(term, snippets, blocklist),
        })
    return candidates, stats


def examples(term: str, snippets: Sequence[PairedSnippet], blocklist: Blocklist) -> list[dict]:
    """Earliest and latest clean lines with the term, from different snippets."""
    hits = [{"snippet_id": s.snippet_id, "speaker": speaker, "text": line}
            for s in snippets for speaker, line in turns(s)
            if term in message_terms(line)
            and not blocklist.veiled(line) and not blocklist.excluded_line(line)]
    if not hits:
        return []
    last = next((hit for hit in reversed(hits) if hit["snippet_id"] != hits[0]["snippet_id"]),
                None)
    return [hits[0]] if last is None else [hits[0], last]


def h(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def radios(name: str, labels: dict[str, str], checked: str | None = None) -> str:
    return "".join(
        f'<label><input type="radio" name="{name}" value="{value}"'
        f'{" checked" if value == checked else ""}>{label}</label>'
        for value, label in labels.items()
    )


def review_page(candidates: Sequence[dict], candidates_sha: str) -> str:
    page = ['<!doctype html><html lang="zh"><head><meta charset="utf-8">',
            "<title>梗与昵称候选审定</title></head>",
            f'<body data-candidates="{candidates_sha}">',
            "<h1>梗与昵称候选审定</h1>",
            "<p>只在本机打开。每项填写含义、场合、说话者、称呼对象与许可；"
            "默认“仅理解”。填完后复制页底的导出文本。</p>"]
    for item in candidates:
        item_id = item["id"]
        page.append(f'<section class="term" data-id="{item_id}" '
                    f'data-term="{html.escape(item["term"])}">')
        page.append(f"<h2>{item_id}　{h(item['term'])}</h2>")
        page.append(f"<p>主人 {item['owner_count']} 次，她 {item['partner_count']} 次；"
                    f"{item['first_seen']} 至 {item['last_seen']}</p><ul>")
        page.extend(f"<li>{SPEAKER_LABELS[ex['speaker']]}（{h(ex['snippet_id'])}）："
                    f"{h(ex['text'])}</li>" for ex in item["examples"])
        page.append("</ul>")
        for field, label in (("meaning", "含义"), ("occasion", "场合"), ("addressee", "对象")):
            page.append(f'<label>{label}<input type="text" name="{field}" '
                        f'maxlength="{MAX_FIELD_CHARS}"></label>')
        page.append(f"<div>说话者{radios(f'speaker-{item_id}', SPEAKER_LABELS)}</div>")
        permission = radios(f"permission-{item_id}", PERMISSION_LABELS, "understand_only")
        page.append(f"<div>许可{permission}</div>")
        page.append("</section>")
    page.extend(['<textarea id="export" readonly></textarea>',
                 '<button type="button" onclick="copyExport()">复制导出文本</button>',
                 f'<script id="terms-export">{EXPORT_JS}</script>',
                 f"<script>{GLUE_JS}</script></body></html>"])
    return "\n".join(page)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def out_dir_ok(out_dir: Path) -> bool:
    settings = load_settings(ROOT)
    return out_dir.resolve().is_relative_to(settings.test_results_dir.resolve())


def build(args: argparse.Namespace) -> int:
    if not out_dir_ok(args.out_dir):
        return status("invalid_out_dir", 2)
    general_dict = args.general_dict or default_general_dict()
    if general_dict is None:
        return status("general_dict_unavailable", 2)
    exclude_terms: list = []
    if args.exclude_terms is not None:
        exclude_terms = json.loads(args.exclude_terms.read_text(encoding="utf-8-sig"))
        if not isinstance(exclude_terms, list) or not all(
                isinstance(term, str) for term in exclude_terms):
            return status("invalid_exclude_terms", 2)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    pools = json.loads(args.pools.read_text(encoding="utf-8"))["pools"]
    allowed = allowed_snippet_ids(split, pools)
    if allowed is None:
        return status("invalid_pools", 2)
    snippets = read_allowed(args.snippets, allowed)
    candidates, stats = extract(snippets, Blocklist(args.veiled, exclude_terms),
                                load_general_freq(general_dict))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.out_dir / "candidates.json"
    write_json(candidates_path, {"version": 1, "snippets_used": len(snippets),
                                 "candidates": candidates})
    candidates_sha = sha256(candidates_path.read_bytes()).hexdigest()
    (args.out_dir / "review.html").write_text(review_page(candidates, candidates_sha),
                                              encoding="utf-8")
    return status("ok", 0, snippets_used=len(snippets), candidates=len(candidates),
                  dropped_veiled=stats["dropped_veiled"],
                  dropped_excluded=stats["dropped_excluded"],
                  dropped_free_combination=stats["dropped_free_combination"],
                  dropped_over_limit=stats["dropped_over_limit"])


def entry_error(entry: object, terms: dict[str, str]) -> bool:
    if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
        return True
    if any(not isinstance(value, str) or len(value) > MAX_FIELD_CHARS
           for value in entry.values()):
        return True
    if terms.get(entry["id"]) != entry["term"] or entry["permission"] not in PERMISSIONS:
        return True
    if entry["permission"] == "delete":
        return False
    return not entry["meaning"].strip() or entry["speaker"] not in SPEAKERS


def validate(args: argparse.Namespace) -> int:
    if not out_dir_ok(args.out_dir):
        return status("invalid_out_dir", 2)
    candidates_data = args.candidates.read_bytes()
    terms = {item["id"]: item["term"] for item in json.loads(candidates_data)["candidates"]}
    try:
        export = json.loads(args.export.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return status("invalid_export", 2)
    entries = export.get("entries") if isinstance(export, dict) else None
    if (not isinstance(export, dict) or export.get("version") != 1
            or export.get("candidates_sha256") != sha256(candidates_data).hexdigest()
            or not isinstance(entries, list)
            or any(entry_error(entry, terms) for entry in entries)
            or sorted(entry["id"] for entry in entries) != sorted(terms)):
        return status("invalid_export", 2)
    kept = [{key: entry[key].strip() for key in
             ("term", "meaning", "occasion", "speaker", "addressee", "permission")}
            for entry in entries if entry["permission"] != "delete"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "dictionary.json", {"version": 1, "entries": kept})
    return status("ok", 0, entries=len(kept), deleted=len(entries) - len(kept))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--snippets", type=Path, action="extend", nargs="+",
                              required=True)
    for name in ("split", "pools", "out-dir"):
        build_parser.add_argument(f"--{name}", type=Path, required=True)
    build_parser.add_argument("--veiled", type=Path, default=DEFAULT_VEILED)
    build_parser.add_argument("--exclude-terms", type=Path)
    build_parser.add_argument("--general-dict", type=Path)
    validate_parser = commands.add_parser("validate")
    for name in ("export", "candidates", "out-dir"):
        validate_parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    return build(args) if args.command == "build" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
