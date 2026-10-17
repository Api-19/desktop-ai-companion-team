# ruff: noqa: RUF001
"""Two-step response-focus labels: acceptable focus first, the real reply second."""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from typing import Any

from companion.admin.labeling import LabelStore, dialogue_lines

_BASE = "/admin/style/labels"
_FORM_FIELDS = frozenset({"csrf", "labels_token", "action", "item_id", "step"})
_SPECIAL = {"@none": "no_unique", "@insufficient": "insufficient"}
_SPECIAL_NAMES = {"no_unique": "无唯一首选", "insufficient": "上下文不足"}


def _unit_ids(item: dict[str, Any]) -> list[str]:
    return [unit["id"] for unit in item["units"]]


def step1_answer(item: dict[str, Any], acceptable: list[str], preferred: str) -> dict[str, Any]:
    """Build a first-step answer, enforcing the selection rules."""
    ids = _unit_ids(item)
    if acceptable != [unit for unit in ids if unit in acceptable]:
        raise ValueError("invalid focus selection")
    choice = _SPECIAL.get(preferred, "unit")
    if choice != "insufficient" and not acceptable:
        raise ValueError("focus selection needs an acceptable unit")
    if choice == "unit" and preferred not in acceptable:
        raise ValueError("preferred focus must be acceptable")
    return {
        "acceptable": acceptable,
        "choice": choice,
        "preferred": preferred if choice == "unit" else None,
    }


def _step2_valid(item: dict[str, Any], responded: list[str]) -> bool:
    return len(set(responded)) == len(responded) and set(responded) <= set(_unit_ids(item))


def _timestamp_valid(value: Any) -> bool:
    return isinstance(value, str) and datetime.fromisoformat(value).tzinfo is not None


class FocusLabelStore(LabelStore):
    def _focus_round(self, round_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        tasks = self.tasks()
        round_ = self._round(tasks, round_id)
        if round_.get("kind") != "focus":
            raise ValueError("not a focus round")
        return tasks, round_

    def _read(self, round_id: str, step: int) -> dict[str, Any]:
        try:
            saved = json.loads(
                (self.root / f"labels_{round_id}_step{step}.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return {}
        if not isinstance(saved, dict):
            raise ValueError("invalid focus answers")
        return saved

    def steps(self, round_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        tasks, round_ = self._focus_round(round_id)
        first = self._read(round_id, 1)
        second = self._read(round_id, 2)
        for item_id, answer in first.items():
            if (
                item_id not in round_["order"]
                or not isinstance(answer, dict)
                or set(answer) != {"acceptable", "choice", "preferred", "at"}
                or not isinstance(answer["acceptable"], list)
                or not all(isinstance(unit, str) for unit in answer["acceptable"])
                or not isinstance(answer["preferred"], str | None)
                or not _timestamp_valid(answer["at"])
            ):
                raise ValueError("invalid focus step1 answer")
            preferred = answer["preferred"] if answer["choice"] == "unit" else next(
                (key for key, value in _SPECIAL.items() if value == answer["choice"]), ""
            )
            rebuilt = step1_answer(tasks["items"][item_id], answer["acceptable"], preferred)
            if rebuilt != {key: answer[key] for key in rebuilt}:
                raise ValueError("invalid focus step1 answer")
        for item_id, answer in second.items():
            if (
                item_id not in first
                or not isinstance(answer, dict)
                or set(answer) != {"responded", "at"}
                or not isinstance(answer["responded"], list)
                or not _step2_valid(tasks["items"][item_id], answer["responded"])
                or not _timestamp_valid(answer["at"])
            ):
                raise ValueError("invalid focus step2 answer")
        return first, second

    def record_step1(
        self, round_id: str, item_id: str, acceptable: list[str], preferred: str
    ) -> None:
        tasks, round_ = self._focus_round(round_id)
        first, _ = self.steps(round_id)
        if item_id not in round_["order"] or item_id in first:
            raise ValueError("focus step1 is already saved")
        answer = step1_answer(tasks["items"][item_id], acceptable, preferred)
        first[item_id] = {**answer, "at": datetime.now(UTC).isoformat()}
        self._write_json(self.root / f"labels_{round_id}_step1.json", first)

    def record_step2(self, round_id: str, item_id: str, responded: list[str]) -> None:
        tasks, _ = self._focus_round(round_id)
        first, second = self.steps(round_id)
        if (
            item_id not in first
            or item_id in second
            or not _step2_valid(tasks["items"][item_id], responded)
        ):
            raise ValueError("invalid focus step2")
        second[item_id] = {"responded": responded, "at": datetime.now(UTC).isoformat()}
        self._write_json(self.root / f"labels_{round_id}_step2.json", second)

    def undo(self, round_id: str) -> None:
        """Remove the latest save; a first step stays while its second step exists."""
        first, second = self.steps(round_id)
        candidates = [(answer["at"], 2, item_id) for item_id, answer in second.items()] + [
            (answer["at"], 1, item_id) for item_id, answer in first.items()
            if item_id not in second
        ]
        if not candidates:
            return
        _, step, item_id = max(
            candidates, key=lambda entry: (datetime.fromisoformat(entry[0]), entry[1])
        )
        answers = second if step == 2 else first
        del answers[item_id]
        self._write_json(self.root / f"labels_{round_id}_step{step}.json", answers)


def focus_position(
    round_: dict[str, Any], first: dict[str, Any], second: dict[str, Any]
) -> tuple[str | None, int | None]:
    item_id = next((item for item in round_["order"] if item not in second), None)
    if item_id is None:
        return None, None
    return item_id, 2 if item_id in first else 1


def parse_focus_form(
    form: dict[str, str], item: dict[str, Any] | None, item_id: str | None, step: int | None
) -> tuple[str, Any]:
    """Return ("undo", None), ("step1", (acceptable, preferred)) or ("step2", responded)."""
    if (
        not set(form) >= _FORM_FIELDS
        or form["item_id"] != (item_id or "")
        or form["step"] != (str(step) if step else "")
    ):
        raise ValueError("stale focus form")
    extra = set(form) - _FORM_FIELDS
    if form["action"] == "undo" and not extra:
        return "undo", None
    if form["action"] != "label" or item is None:
        raise ValueError("invalid focus action")
    ids = _unit_ids(item)
    if step == 1:
        if "preferred" not in extra or not extra <= {f"accept_{unit}" for unit in ids} | {
            "preferred"
        } or any(form[key] != "yes" for key in extra - {"preferred"}):
            raise ValueError("invalid focus step1 form")
        acceptable = [unit for unit in ids if f"accept_{unit}" in form]
        step1_answer(item, acceptable, form["preferred"])
        return "step1", (acceptable, form["preferred"])
    if not extra <= {f"{prefix}_{unit}" for prefix in ("hit", "rank") for unit in ids}:
        raise ValueError("invalid focus step2 form")
    hits = [unit for unit in ids if f"hit_{unit}" in form]
    ranks = {unit: form.get(f"rank_{unit}", "") for unit in ids}
    if (
        any(form[f"hit_{unit}"] != "yes" for unit in hits)
        or any(ranks[unit] for unit in ids if unit not in hits)
        or any(re.fullmatch(r"[1-9][0-9]?", ranks[unit]) is None for unit in hits)
        or sorted(int(ranks[unit]) for unit in hits) != list(range(1, len(hits) + 1))
    ):
        raise ValueError("invalid focus order")
    return "step2", sorted(hits, key=lambda unit: int(ranks[unit]))


def render_focus_round(
    *,
    tasks: dict[str, Any],
    round_: dict[str, Any],
    marked: int,
    item_id: str | None,
    step: int | None,
    first: dict[str, Any] | None,
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
    hidden = (
        f'<input type="hidden" name="csrf" value="{html.escape(csrf)}">'
        f'<input type="hidden" name="labels_token" value="{html.escape(token)}">'
        f'<input type="hidden" name="item_id" value="{html.escape(item_id or "")}">'
        f'<input type="hidden" name="step" value="{step or ""}">'
    )
    form_start = f'<form method="post" action="{_BASE}/{html.escape(round_["id"])}">'
    if item_id is None:
        lines.append("<p>本轮已完成</p>")
    else:
        item = tasks["items"][item_id]
        texts = {unit["id"]: html.escape(unit["text"]) for unit in item["units"]}
        if step == 1:
            lines.append("<h2>第一步：只看前文和她的话，选可接受的回应焦点</h2>")
            lines.extend(dialogue_lines(item, reply=False))
            lines.append(form_start + hidden + '<input type="hidden" name="action" value="label">')
            lines.append("<fieldset><legend>可接受焦点（可多选）</legend>")
            lines.extend(
                f'<label><input type="checkbox" name="accept_{html.escape(unit)}" value="yes">'
                f"{text}</label><br>"
                for unit, text in texts.items()
            )
            lines.append("</fieldset><fieldset><legend>首选（单选）</legend>")
            options = [
                *texts.items(),
                ("@none", _SPECIAL_NAMES["no_unique"]),
                ("@insufficient", _SPECIAL_NAMES["insufficient"]),
            ]
            lines.extend(
                f'<label><input type="radio" name="preferred" value="{html.escape(value)}">'
                f"{text}</label><br>"
                for value, text in options
            )
            lines.append("</fieldset><button type=\"submit\">保存第一步</button></form>")
        else:
            assert first is not None
            lines.append("<h2>第二步：看他的真实回复，选实际回应了哪些</h2>")
            lines.extend(dialogue_lines(item, reply=True))
            accepted = "、".join(texts[unit] for unit in first["acceptable"]) or "无"
            preferred = (
                texts[first["preferred"]]
                if first["choice"] == "unit"
                else _SPECIAL_NAMES[first["choice"]]
            )
            lines.append(f"<p>第一步（只读）：可接受：{accepted}；首选：{preferred}</p>")
            lines.append(form_start + hidden + '<input type="hidden" name="action" value="label">')
            lines.append("<fieldset><legend>实际回应了（可不选；勾选的填先后 1、2、3…）</legend>")
            lines.extend(
                f'<label><input type="checkbox" name="hit_{html.escape(unit)}" value="yes">'
                f"{text}</label> 顺序"
                f'<input name="rank_{html.escape(unit)}" inputmode="numeric" size="2"><br>'
                for unit, text in texts.items()
            )
            lines.append("</fieldset><button type=\"submit\">保存第二步</button></form>")
    lines.append(
        form_start + hidden + '<input type="hidden" name="action" value="undo">'
        '<button type="submit">撤销上一次保存</button></form>'
    )
    return "".join(lines)
