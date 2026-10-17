"""Extract local self-stated profile candidates; expose only counts or fixed errors.

Counts are per message for scanned/processed/owner/media, otherwise per sentence.
Dry runs count classifications and duplicates, but candidates_added stays zero.
Service calls still perform normal database initialization and lifecycle cleanup.
Use a separate ledger for each profile user; keep it after deleting candidates.
"""

# Chinese punctuation is intentional in the extraction rules.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from companion.importer.models import MessageDirection
from companion.importer.standard_json import StandardJsonAdapter
from companion.memory import MemoryAuthorizationError, MemoryService
from companion.memory.profile import PROFILE_PERMISSION

_HABIT = re.compile("每天|经常|总是|老是|一般|从来不|很少|习惯")
_PREFERENCE = re.compile("不爱|不喜欢|讨厌|喜欢|爱")
_SENTENCES = re.compile(r"[^。.？！!?\r\n]+[。.？！!?]?[\"'”’」』]*")
_QUOTES = re.compile(r"[\"'“”‘’「」『』]|说(?:过)?\s*(?:[：:]|我|自己|他|她|朋友|室友|同学)")
_PARTICLES = re.compile(r"[啊呀呢吧哦噢啦呐嘛]+(?=[，,。.！!？?\s]|$)")
_TOPICS = (
    ("喝水", re.compile("喝水|水")),
    ("作息", re.compile("熬夜|晚睡|早睡|起床|睡觉")),
    ("奶茶", re.compile("奶茶")),
    ("咖啡", re.compile("咖啡")),
    ("运动", re.compile("运动|跑步|健身")),
    ("吃辣", re.compile("辣")),
    ("早餐", re.compile("早饭|早餐")),
    ("零食", re.compile("零食")),
)
_EXCLUSIONS = (
    ("question", re.compile(r"吗|[？?]")),
    ("hypothetical", re.compile("如果|要是|假如|万一")),
    ("past", re.compile("以前|之前|小时候|那时候|曾经")),
    # Identify explicit quotations before checking their quoted subject.
    ("quote", _QUOTES),
    ("third_party", re.compile(r"^(?!我(?!妈|爸))")),
    ("health", re.compile("病|药|医院|疼|痛|吃药|确诊|抑郁|焦虑")),
)
_COUNT_KEYS = (
    "scanned",
    "self_statements",
    "candidates_added",
    "skipped_duplicate",
    "skipped_processed",
    "excluded_owner",
    "excluded_question",
    "excluded_hypothetical",
    "excluded_past",
    "excluded_third_party",
    "excluded_health",
    "excluded_quote",
    "excluded_other",
)


class _ArgumentError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError("invalid_arguments")


def _read_ledger(path: Path) -> set[str]:
    if not path.exists():
        return set()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, list) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in saved
    ):
        raise ValueError("invalid ledger")
    return set(saved)


def _write_ledger(path: Path, processed: set[str]) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(json.dumps(sorted(processed), separators=(",", ":")) + "\n")
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _extract(records, processed, existing):
    counts = dict.fromkeys(_COUNT_KEYS, 0)
    counts["scanned"] = len(records)
    pending = []
    for record in records:
        digest = hashlib.sha256(record.message_id.encode("utf-8")).hexdigest()
        if digest in processed:
            counts["skipped_processed"] += 1
            continue
        candidates = []
        pending.append((digest, candidates))
        if record.direction == MessageDirection.OUTBOUND:
            counts["excluded_owner"] += 1
            continue
        if record.content_type != "text" or not record.text:
            counts["excluded_other"] += 1
            continue
        for match in _SENTENCES.finditer(record.text):
            sentence = match.group().strip()
            habit = _HABIT.search(sentence)
            if "我" not in sentence or not (habit or _PREFERENCE.search(sentence)):
                counts["excluded_other"] += 1
                continue
            counts["self_statements"] += 1
            excluded = next((name for name, rule in _EXCLUSIONS if rule.search(sentence)), None)
            if excluded is not None:
                counts[f"excluded_{excluded}"] += 1
                continue
            topic = next((name for name, rule in _TOPICS if rule.search(sentence)), None)
            if topic is None:
                counts["excluded_other"] += 1
                continue
            text = _PARTICLES.sub("", sentence)[:60]
            key = (topic, text)
            if key in existing:
                counts["skipped_duplicate"] += 1
                continue
            existing.add(key)
            candidates.append(
                dict(
                    category="habit" if habit else "preference",
                    topic=topic,
                    text=text,
                    source="self_stated",
                    evidence_date=record.timestamp.date(),
                    evidence_quote=sentence[:30],
                    confidence="high" if re.search("每天|从来不|总是", sentence) else "med",
                )
            )
    return counts, pending


def _run(args: argparse.Namespace) -> dict[str, int]:
    args.chat = args.chat.resolve()
    args.memory_db = args.memory_db.resolve()
    args.ledger = args.ledger.resolve()
    database_files = {
        args.memory_db,
        *(Path(str(args.memory_db) + suffix) for suffix in ("-wal", "-shm", "-journal")),
    }
    if args.chat in database_files or args.ledger in database_files or args.ledger == args.chat:
        raise ValueError("paths must be distinct")
    if not args.user_id.strip() or len(args.user_id) > 120:
        raise ValueError("invalid user")
    records = StandardJsonAdapter().parse(args.chat).records
    processed = _read_ledger(args.ledger)
    now = datetime.now(UTC)
    service = MemoryService(args.memory_db, clock=lambda: now)
    if not args.dry_run:
        consent = service.get_consent(args.user_id)
        if (
            consent is None
            or not consent.granted
            or PROFILE_PERMISSION not in consent.allowed_fields
        ):
            raise MemoryAuthorizationError("profile_consent_required")
    existing = {(item.topic, item.text) for item in service.list_profile(args.user_id)}
    counts, pending = _extract(records, processed, existing)
    if args.dry_run or not pending:
        return counts
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    for digest, candidates in pending:
        for candidate in candidates:
            service.add_profile_candidate(user_id=args.user_id, **candidate)
            counts["candidates_added"] += 1
        processed.add(digest)
        _write_ledger(args.ledger, processed)
    return counts


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _Parser(add_help=False)
        for name in ("chat", "memory-db", "ledger"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--user-id", required=True)
        parser.add_argument("--dry-run", action="store_true")
        result = _run(parser.parse_args(argv))
    except MemoryAuthorizationError:
        print("profile_consent_required")
        return 2
    except _ArgumentError:
        print("invalid_arguments")
        return 2
    except Exception:
        # File, import and service failures must never disclose source content.
        print("failed")
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
