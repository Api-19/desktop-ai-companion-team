# ruff: noqa: RUF001
"""Private response-strategy labels and pages, isolated by round."""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_BASE = "/admin/style/labels"


def _validate_tasks(tasks: Any) -> None:
    if (
        not isinstance(tasks, dict)
        or type(tasks.get("version")) is not int
        or tasks["version"] != 1
        or not isinstance(tasks.get("labels"), list)
        or not tasks["labels"]
        or not isinstance(tasks.get("items"), dict)
        or not isinstance(tasks.get("rounds"), list)
    ):
        raise ValueError("invalid labeling tasks")
    codes = set()
    for label in tasks["labels"]:
        if (
            not isinstance(label, dict)
            or not isinstance(label.get("code"), str)
            or not 1 <= len(label["code"]) <= 2
            or label["code"] == "?"
            or label["code"] in codes
            or not isinstance(label.get("name"), str)
            or not isinstance(label.get("hint"), str)
        ):
            raise ValueError("invalid labeling labels")
        codes.add(label["code"])
    for item in tasks["items"].values():
        if not isinstance(item, dict) or any(
            not isinstance(item.get(field), list)
            or any(not isinstance(line, str) for line in item[field])
            for field in ("context", "partner", "reply")
        ):
            raise ValueError("invalid labeling item")
    round_ids = set()
    for round_ in tasks["rounds"]:
        if (
            not isinstance(round_, dict)
            or not isinstance(round_.get("id"), str)
            or re.fullmatch(r"[a-z0-9_-]{1,32}", round_["id"]) is None
            or round_["id"] in round_ids
            or not isinstance(round_.get("title"), str)
            or "opens_on" not in round_
            or not isinstance(round_.get("order"), list)
            or any(
                not isinstance(item_id, str) or item_id not in tasks["items"]
                for item_id in round_["order"]
            )
            or len(set(round_["order"])) != len(round_["order"])
        ):
            raise ValueError("invalid labeling round")
        opens_on = round_["opens_on"]
        if opens_on is not None:
            if (
                not isinstance(opens_on, str)
                or re.fullmatch(r"\d{4}-\d{2}-\d{2}", opens_on) is None
            ):
                raise ValueError("invalid labeling opening date")
            date.fromisoformat(opens_on)
        if round_.get("kind", "strategy") not in ("strategy", "focus"):
            raise ValueError("invalid labeling round kind")
        if round_.get("kind") == "focus":
            for item_id in round_["order"]:
                _validate_units(tasks["items"][item_id].get("units"))
        round_ids.add(round_["id"])


def _validate_units(units: Any) -> None:
    if not isinstance(units, list) or not 1 <= len(units) <= 12:
        raise ValueError("invalid focus units")
    ids = set()
    for unit in units:
        if (
            not isinstance(unit, dict)
            or set(unit) != {"id", "text"}
            or not isinstance(unit["id"], str)
            or re.fullmatch(r"[a-z0-9_-]{1,16}", unit["id"]) is None
            or unit["id"] in ids
            or not isinstance(unit["text"], str)
            or not 1 <= len(unit["text"]) <= 60
        ):
            raise ValueError("invalid focus unit")
        ids.add(unit["id"])


class LabelStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def tasks(self) -> dict[str, Any]:
        tasks = json.loads((self.root / "label_tasks.json").read_text(encoding="utf-8"))
        _validate_tasks(tasks)
        return tasks

    @staticmethod
    def _round(tasks: dict[str, Any], round_id: str) -> dict[str, Any]:
        for round_ in tasks["rounds"]:
            if round_["id"] == round_id:
                return round_
        raise ValueError("unknown labeling round")

    def answers(self, round_id: str) -> dict[str, dict[str, str]]:
        tasks = self.tasks()
        round_ = self._round(tasks, round_id)
        path = self.root / f"labels_{round_id}.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        codes = {label["code"] for label in tasks["labels"]} | {"?"}
        if not isinstance(saved, dict):
            raise ValueError("invalid labeling answers")
        for item_id, answer in saved.items():
            if (
                item_id not in round_["order"]
                or not isinstance(answer, dict)
                or set(answer) != {"label", "at"}
                or not isinstance(answer["label"], str)
                or answer["label"] not in codes
                or not isinstance(answer["at"], str)
                or datetime.fromisoformat(answer["at"]).tzinfo is None
            ):
                raise ValueError("invalid labeling answer")
        return saved

    def next_item(self, round_id: str) -> str | None:
        round_ = self._round(self.tasks(), round_id)
        answers = self.answers(round_id)
        return next((item_id for item_id in round_["order"] if item_id not in answers), None)

    def _write_json(self, path: Path, payload: object) -> None:
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

    def record(self, round_id: str, item_id: str, label: str) -> None:
        tasks = self.tasks()
        round_ = self._round(tasks, round_id)
        if item_id not in round_["order"] or label not in {
            "?",
            *(entry["code"] for entry in tasks["labels"]),
        }:
            raise ValueError("invalid labeling selection")
        answers = self.answers(round_id)
        answers[item_id] = {"label": label, "at": datetime.now(UTC).isoformat()}
        self._write_json(self.root / f"labels_{round_id}.json", answers)

    def undo(self, round_id: str) -> None:
        answers = self.answers(round_id)
        if not answers:
            return
        latest = max(answers, key=lambda item_id: datetime.fromisoformat(answers[item_id]["at"]))
        del answers[latest]
        self._write_json(self.root / f"labels_{round_id}.json", answers)


def round_is_open(round_: dict[str, Any], now: datetime) -> bool:
    return round_["opens_on"] is None or (
        now.astimezone(_SHANGHAI).date() >= date.fromisoformat(round_["opens_on"])
    )


def render_round_list(tasks: dict[str, Any], progress: dict[str, int], now: datetime) -> str:
    lines = ["<h1>回应策略标注</h1><ul>"]
    for round_ in tasks["rounds"]:
        state = (
            "已开放"
            if round_is_open(round_, now)
            else (f"将于 {html.escape(round_['opens_on'])} 开放")
        )
        lines.append(
            f'<li><a href="{_BASE}/{html.escape(round_["id"])}">'
            f"{html.escape(round_['title'])}</a> "
            f"{progress[round_['id']]}/{len(round_['order'])} · {state}</li>"
        )
    return "".join(lines) + "</ul>"


def dialogue_lines(item: dict[str, Any], *, reply: bool) -> list[str]:
    lines = []
    if item["context"]:
        lines.append(
            '<p><small><font color="gray">前文：'
            + "<br>".join(html.escape(line) for line in item["context"])
            + "</font></small></p>"
        )
    lines.extend(f"<p>她：{html.escape(line)}</p>" for line in item["partner"])
    if reply:
        lines.extend(f"<p><strong>他：{html.escape(line)}</strong></p>" for line in item["reply"])
    return lines


def render_round(
    *,
    tasks: dict[str, Any],
    round_: dict[str, Any],
    marked: int,
    item_id: str | None,
    opened: bool,
    csrf: str = "",
    token: str = "",
) -> str:
    lines = [
        f"<h1>{html.escape(round_['title'])}</h1>",
        f"<p>进度：{marked}/{len(round_['order'])}</p>",
        f'<p><a href="{_BASE}">返回标注轮次</a></p>',
    ]
    if not opened:
        return "".join(lines) + f"<p>将于 {html.escape(round_['opens_on'])} 开放</p>"
    lines.append("<h2>标签说明</h2><ul>")
    for label in tasks["labels"]:
        lines.append(
            f"<li>{html.escape(label['code'])} · {html.escape(label['name'])}："
            f"{html.escape(label['hint'])}</li>"
        )
    lines.append("<li>? · 无法判断</li></ul>")
    current = html.escape(item_id if item_id is not None else "")
    hidden = (
        f'<input type="hidden" name="csrf" value="{html.escape(csrf)}">'
        f'<input type="hidden" name="labels_token" value="{html.escape(token)}">'
        f'<input type="hidden" name="item_id" value="{current}">'
    )
    form_start = f'<form method="post" action="{_BASE}/{html.escape(round_["id"])}">'
    if item_id is None:
        lines.append("<p>本轮已完成</p>")
    else:
        lines.extend(dialogue_lines(tasks["items"][item_id], reply=True))
        lines.append(form_start + hidden + '<input type="hidden" name="action" value="label">')
        for code in [*(label["code"] for label in tasks["labels"]), "?"]:
            lines.append(
                f'<button type="submit" name="label" value="{html.escape(code)}">'
                f"{html.escape(code)}</button>"
            )
        lines.append("</form>")
    lines.append(
        form_start + hidden + '<input type="hidden" name="action" value="undo">'
        '<input type="hidden" name="label" value="">'
        '<button type="submit">撤销上一条</button></form>'
    )
    return "".join(lines)
