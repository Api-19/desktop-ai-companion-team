# ruff: noqa: RUF001
"""Structure checks for the synthetic D14 session-check scenario file."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SESSION_FILE = ROOT / "fixtures/synthetic/d14_session_check_v1.json"
CATEGORIES = {
    "future_vs_past", "person_swap", "serial_correction", "negation_limit",
    "resolved_reopened", "interleaved_topics", "ambiguous_reference", "example_or_ai_as_fact",
}
CRISIS = re.compile("自杀|自伤|想死|杀人|伤害")
# Reviewed shared-term words must not appear in the session-check scenarios.
SHARED_TERMS = ("呃呃", "牛逼", "笑死", "乖奥", "dnys", "安安", "小宝", "宝宝", "乖乖", "坤")


def load():
    return json.loads(SESSION_FILE.read_text(encoding="utf-8"))


def test_session_file_has_eight_unique_items_one_per_category():
    rows = load()
    assert isinstance(rows, list)
    assert len(rows) == 8
    assert len({row["id"] for row in rows}) == 8
    assert sorted(row["category"] for row in rows) == sorted(CATEGORIES)


def test_session_items_have_required_fields():
    for row in load():
        assert isinstance(row["id"], str) and row["id"].strip()
        assert isinstance(row["text"], str) and row["text"].strip()
        context = row["recent_context"]
        assert isinstance(context, list) and 0 <= len(context) <= 10
        assert all(isinstance(line, str) and line.startswith(("对方：", "你："))
                   for line in context)
        expect = row["expect"]
        for key in ("facts", "forbidden"):
            assert isinstance(expect[key], list) and 1 <= len(expect[key]) <= 3
            assert all(isinstance(entry, str) and entry.strip() for entry in expect[key])
        assert isinstance(expect["ok_reply_hint"], str) and expect["ok_reply_hint"].strip()


def test_session_file_has_no_crisis_words_or_shared_terms():
    raw = SESSION_FILE.read_text(encoding="utf-8")
    assert CRISIS.search(raw) is None
    assert [term for term in SHARED_TERMS if term in raw] == []
