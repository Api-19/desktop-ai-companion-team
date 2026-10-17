# ruff: noqa: RUF001
"""Private paired-snippet storage and page fragments for the local admin."""

from __future__ import annotations

import html
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from companion.contracts import ChatRecord
from companion.privacy.paired_snippets import (
    PairedSnippet,
    build_paired_snippets,
    delete_snippets,
    read_snippets,
    sample_for_audit,
    split_holdout,
    write_snippets,
)

REDACTION_GROUPS = ("name", "address", "account", "private_event")
TERM_FIELDS = (*REDACTION_GROUPS, "intimate")
_FIELD_LABELS = {
    "name": "真实姓名",
    "address": "地址",
    "account": "账号",
    "private_event": "私密事件词",
    "intimate": "亲密称呼",
}
_MAX_TERM_CHARS = 40
_MAX_TERMS = 500
_AUDIT_SIZE = 50
_SHANGHAI = ZoneInfo("Asia/Shanghai")


class SnippetStoreError(RuntimeError):
    """The private snippet directory or one of its files is unsafe."""


def parse_terms(form: Mapping[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for field in TERM_FIELDS:
        terms = list(
            dict.fromkeys(line.strip() for line in form[field].splitlines() if line.strip())
        )
        if any(len(term) > _MAX_TERM_CHARS for term in terms):
            raise ValueError("snippet terms invalid")
        groups[field] = terms
    if sum(len(terms) for terms in groups.values()) > _MAX_TERMS:
        raise ValueError("snippet terms invalid")
    return groups


def parse_range(start: str, end: str) -> tuple[datetime, datetime] | None:
    if not start and not end:
        return None
    begin, finish = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if begin.tzinfo is not None or finish.tzinfo is not None or begin > finish:
        raise ValueError("snippet range invalid")
    return begin.replace(tzinfo=_SHANGHAI), finish.replace(tzinfo=_SHANGHAI)


class SnippetStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.terms_path = root / "terms.json"
        self.library_path = root / "library.jsonl"
        self.holdout_path = root / "holdout.jsonl"
        self.audit_path = root / "audit_sample.jsonl"
        self.deleted_path = root / "deleted_ids.json"

    @property
    def snippet_paths(self) -> tuple[Path, Path, Path]:
        return self.library_path, self.holdout_path, self.audit_path

    def _directory(self, *, create: bool) -> bool:
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            info = self.root.lstat()
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise SnippetStoreError("snippet directory unsafe")
        return True

    def _exists(self, path: Path) -> bool:
        if not self._directory(create=False):
            return False
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o777 != 0o600
        ):
            raise SnippetStoreError("snippet file unsafe")
        return True

    def _read(self, path: Path) -> list[PairedSnippet]:
        return read_snippets(path) if self._exists(path) else []

    def terms(self) -> dict[str, list[str]]:
        if not self._exists(self.terms_path):
            return {field: [] for field in TERM_FIELDS}
        saved = json.loads(self.terms_path.read_text(encoding="utf-8"))
        groups = {field: saved["redaction_terms"].get(field, []) for field in REDACTION_GROUPS}
        groups["intimate"] = saved["intimate_terms"]
        return groups

    def exclusions(self) -> tuple[set[str], list[tuple[datetime, datetime]]]:
        if not self._exists(self.deleted_path):
            return set(), []
        saved = json.loads(self.deleted_path.read_text(encoding="utf-8"))
        ranges = [
            (datetime.fromisoformat(start), datetime.fromisoformat(end))
            for start, end in saved["ranges"]
        ]
        return set(saved["snippet_ids"]), ranges

    def counts(self) -> tuple[int, int, int]:
        library, holdout, audit = (len(self._read(path)) for path in self.snippet_paths)
        return library, holdout, audit

    def audit_sample(self) -> list[PairedSnippet]:
        return self._read(self.audit_path)

    def _write_json(self, path: Path, payload: object) -> None:
        self._exists(path)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def generate(
        self,
        records: Sequence[ChatRecord],
        *,
        groups: Mapping[str, list[str]],
        created_at: datetime,
        seed: int,
    ) -> tuple[int, int, int, int]:
        self._directory(create=True)
        self._write_json(
            self.terms_path,
            {
                "redaction_terms": {
                    field: groups[field] for field in REDACTION_GROUPS if groups[field]
                },
                "intimate_terms": groups["intimate"],
            },
        )
        deleted_ids, deleted_ranges = self.exclusions()
        built = build_paired_snippets(
            list(records),
            created_at=created_at,
            context_turns=10,
            exclude_near_nontext=True,
            pseudonymize=True,
            redaction_terms={field: groups[field] for field in REDACTION_GROUPS if groups[field]},
        )
        snippets = [
            item
            for item in built
            if item.snippet_id not in deleted_ids
            and not any(start <= item.started_at <= end for start, end in deleted_ranges)
        ]
        library, holdout = split_holdout(snippets)
        audit = sample_for_audit(library, size=_AUDIT_SIZE, seed=seed)
        for path, items in zip(self.snippet_paths, (library, holdout, audit), strict=True):
            self._exists(path)
            write_snippets(path, items)
        return len(snippets), len(library), len(holdout), len(audit)

    def delete(
        self,
        snippet_ids: frozenset[str],
        started_between: tuple[datetime, datetime] | None,
    ) -> int:
        if not all(self._exists(path) for path in self.snippet_paths):
            raise SnippetStoreError("snippet library unavailable")
        deleted_ids, deleted_ranges = self.exclusions()
        matched_ids = {
            item.snippet_id
            for path in self.snippet_paths
            for item in read_snippets(path)
            if item.snippet_id in snippet_ids
            or (
                started_between is not None
                and started_between[0] <= item.started_at <= started_between[1]
            )
        }
        deleted_ids.update(matched_ids)
        if started_between is not None:
            deleted_ranges.append(started_between)
        self._write_json(
            self.deleted_path,
            {
                "snippet_ids": sorted(deleted_ids),
                "ranges": [[start.isoformat(), end.isoformat()] for start, end in deleted_ranges],
            },
        )
        delete_snippets(
            self.snippet_paths, snippet_ids=snippet_ids, started_between=started_between
        )
        return len(matched_ids)


def render_form(
    *, csrf: str, token: str, terms: Mapping[str, list[str]], counts: tuple[int, int, int]
) -> str:
    fields = "".join(
        f"<p><label>{_FIELD_LABELS[field]}（每行一个）"
        f'<textarea name="{field}">{html.escape(chr(10).join(terms[field]))}</textarea>'
        "</label></p>"
        for field in TERM_FIELDS
    )
    library, holdout, audit = counts
    return (
        "<h1>成对片段</h1>"
        f"<p>片段库：{library}</p><p>留出集：{holdout}</p><p>审核样本：{audit}</p>"
        '<form method="post" action="/admin/style/snippets">'
        f'<input type="hidden" name="csrf" value="{csrf}">'
        f'<input type="hidden" name="snippets_token" value="{token}">'
        f"{fields}"
        "<p>名字与地点会替换成固定化名（小甲、甲地…），按每行顺序编号；"
        "含图片等非文字消息附近的对话不会进入片段库。</p>"
        '<label><input type="checkbox" name="confirm" value="yes">'
        "确认生成片段库（会覆盖现有片段库与留出集）</label>"
        "<button>生成片段库</button></form>"
        '<p><a href="/admin/style/snippets/audit">抽样审核</a></p>'
    )


def render_result(counts: tuple[int, int, int, int]) -> str:
    total, library, holdout, audit = counts
    return (
        "<h1>片段库已生成</h1>"
        f"<p>片段：{total}</p><p>片段库：{library}</p>"
        f"<p>留出集：{holdout}</p><p>审核样本：{audit}</p>"
        '<p><a href="/admin/style/snippets/audit">下一步：抽样审核</a></p>'
    )


def render_audit(
    *, csrf: str, token: str, sample: Sequence[PairedSnippet], excluded: tuple[int, int]
) -> str:
    items = "".join(
        "<li><label>"
        f'<input type="checkbox" name="delete_{item.snippet_id}" value="yes">'
        f"{html.escape(item.started_at.astimezone(_SHANGHAI).isoformat())}</label>"
        + (
            "<p>前文：" + "<br>".join(html.escape(line) for line in item.context_turns) + "</p>"
            if item.context_turns else ""
        )
        + f"<p>对方：{'<br>'.join(html.escape(line) for line in item.partner_turns)}</p>"
        f"<p>主人：{'<br>'.join(html.escape(line) for line in item.owner_reply)}</p></li>"
        for item in sample
    )
    return (
        "<h1>片段抽样审核</h1>"
        '<form method="post" action="/admin/style/snippets/audit">'
        f'<input type="hidden" name="csrf" value="{csrf}">'
        f'<input type="hidden" name="audit_token" value="{token}">'
        f"<ol>{items}</ol>"
        "<p>按时间段删除（北京时间）："
        '<input type="datetime-local" name="from"> 至 '
        '<input type="datetime-local" name="to"></p>'
        "<button>删除所选片段</button></form>"
        f"<p>已永久排除：{excluded[0]} 段 + {excluded[1]} 个时间段</p>"
    )
