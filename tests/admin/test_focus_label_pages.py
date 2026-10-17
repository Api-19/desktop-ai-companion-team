# ruff: noqa: RUF001
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_label_pages import NOW, label_form, make_app, task_data
from test_local_entry import call
from test_mapping_confirmation import ORIGIN

BASE = "/admin/style/labels"
SNAPSHOT = Path(__file__).parent / "snapshots" / "label_strategy_pages.html"


def focus_data():
    data = task_data()
    for index in range(1, 4):
        data["items"][f"f{index}"] = {
            "context": [f"焦点前文{index}"],
            "partner": [f"焦点她{index}"],
            "reply": [f"焦点他回复{index}", f"焦点他回复后续{index}"],
            "units": [
                {"id": "u1", "text": f"今天太忙了{index}"},
                {"id": "u2", "text": f"延续前文事项{index}"},
                {"id": "whole", "text": "整体关系"},
            ],
        }
    data["rounds"] += [
        {"id": "fr", "title": "焦点轮", "kind": "focus", "opens_on": None,
         "order": ["f1", "f2", "f3"]},
        {"id": "fr2", "title": "焦点重标", "kind": "focus", "opens_on": "2026-10-09",
         "order": ["f1"]},
        {"id": "fr3", "title": "焦点未开", "kind": "focus", "opens_on": "2026-10-16",
         "order": ["f1"]},
    ]
    return data


def focus_app(tmp_path, **kwargs):
    return make_app(tmp_path, data=focus_data(), **kwargs)


def page_form(app, cookie, round_id="fr"):
    status, _, page = call(app, f"{BASE}/{round_id}", cookie=cookie)
    assert status == 200
    base = {
        name: re.search(rf'name="{name}" value="([^"]*)"', page).group(1)
        for name in ("csrf", "labels_token", "item_id", "step")
    }
    return base, page


def step1(app, cookie, accept=("u1",), preferred="u1", round_id="fr", **extra):
    form, _ = page_form(app, cookie, round_id)
    assert form["step"] == "1"
    form["action"] = "label"
    form.update({f"accept_{unit}": "yes" for unit in accept})
    if preferred is not None:
        form["preferred"] = preferred
    form.update(extra)
    return post(app, cookie, form, round_id)


def step2(app, cookie, ranks=None, round_id="fr", **extra):
    form, _ = page_form(app, cookie, round_id)
    assert form["step"] == "2"
    form["action"] = "label"
    ranks = {"u2": "1", "u1": "2"} if ranks is None else ranks
    for unit in ("u1", "u2", "whole"):
        form[f"rank_{unit}"] = ranks.get(unit, "")
        if unit in ranks:
            form[f"hit_{unit}"] = "yes"
    form.update(extra)
    return post(app, cookie, form, round_id)


def undo(app, cookie, round_id="fr"):
    form, _ = page_form(app, cookie, round_id)
    form["action"] = "undo"
    return post(app, cookie, form, round_id)


def post(app, cookie, form, round_id="fr"):
    return call(
        app, f"{BASE}/{round_id}", method="POST", cookie=cookie, origin=ORIGIN, form=form
    )[0]


def saved(root, round_id, step):
    path = root / f"labels_{round_id}_step{step}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def snapshot_pages(app, cookie):
    pages = []
    for path in [BASE, f"{BASE}/r1", f"{BASE}/r3"]:
        status, _, page = call(app, path, cookie=cookie)
        assert status == 200
        page = re.sub(r'name="csrf" value="[^"]*"', 'name="csrf" value="CSRF"', page)
        page = re.sub(
            r'name="labels_token" value="[^"]*"', 'name="labels_token" value="TOKEN"', page
        )
        pages.append(page)
    return "\n===\n".join(pages)


def test_strategy_pages_are_byte_identical_to_snapshot(tmp_path):
    app, cookie, _ = make_app(tmp_path)
    form = label_form(app, cookie, label="E")
    assert post(app, cookie, form, "r1") == 303
    assert snapshot_pages(app, cookie) == SNAPSHOT.read_text(encoding="utf-8")


def test_explicit_strategy_kind_and_unrelated_units_keep_old_behavior(tmp_path):
    data = task_data()
    data["rounds"][0]["kind"] = "strategy"
    data["items"]["i1"]["units"] = "旧文件里不属于焦点轮的字段"
    app, cookie, _ = make_app(tmp_path, data=data)
    status, _, page = call(app, f"{BASE}/r1", cookie=cookie)
    assert status == 200 and 'name="label" value="A"' in page and "step" not in page


INVALID_FOCUS = [
    ("kind", "other"),
    ("kind", 1),
    ("kind", None),
    ("units", None),
    ("units", "u1"),
    ("units", []),
    ("units", [{"id": f"u{n}", "text": "t"} for n in range(13)]),
    ("units", [{"id": "u1", "text": "a"}, {"id": "u1", "text": "b"}]),
    ("units", [{"id": "U1", "text": "a"}]),
    ("units", [{"id": "x" * 17, "text": "a"}]),
    ("units", [{"id": "", "text": "a"}]),
    ("units", [{"id": 1, "text": "a"}]),
    ("units", [{"id": "u1", "text": ""}]),
    ("units", [{"id": "u1", "text": "字" * 61}]),
    ("units", [{"id": "u1", "text": None}]),
    ("units", [{"id": "u1"}]),
    ("units", [{"id": "u1", "text": "a", "extra": 1}]),
    ("units", ["u1"]),
]


@pytest.mark.parametrize("field,value", INVALID_FOCUS)
def test_invalid_kind_or_units_make_tasks_unavailable(tmp_path, field, value):
    data = focus_data()
    if field == "kind":
        data["rounds"][3]["kind"] = value
    elif value is None:
        del data["items"]["f2"]["units"]
    else:
        data["items"]["f2"]["units"] = value
    app, cookie, _ = make_app(tmp_path, data=data)
    for path in [BASE, f"{BASE}/fr", f"{BASE}/r1"]:
        status, _, page = call(app, path, cookie=cookie)
        assert status == 503 and "标注任务未就绪" in page


def test_valid_unit_limits_are_accepted(tmp_path):
    data = focus_data()
    data["items"]["f1"]["units"] = [
        {"id": f"u-{n}_{'x' * 11}", "text": "字" * 60} for n in range(10, 22)
    ]
    app, cookie, _ = make_app(tmp_path, data=data)
    status, _, page = call(app, f"{BASE}/fr", cookie=cookie)
    assert status == 200 and page.count('type="checkbox"') == 12


def test_step1_page_hides_reply_and_offers_units(tmp_path):
    app, cookie, _ = focus_app(tmp_path)
    form, page = page_form(app, cookie)
    assert form["item_id"] == "f1" and form["step"] == "1"
    assert "焦点他回复" not in page and "他：" not in page
    assert "焦点前文1" in page and "焦点她1" in page
    for unit in ("u1", "u2", "whole"):
        assert f'name="accept_{unit}" value="yes"' in page
        assert f'name="preferred" value="{unit}"' in page
    assert "今天太忙了1" in page and "整体关系" in page
    assert 'name="preferred" value="@none"' in page and "无唯一首选" in page
    assert 'name="preferred" value="@insufficient"' in page and "上下文不足" in page
    assert "rank_" not in page and "0/3" in page


def test_step1_saves_and_reveals_step2_with_readonly_choice(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie, accept=("whole", "u1"), preferred="whole") == 303
    answer = saved(root, "fr", 1)["f1"]
    assert answer["acceptable"] == ["u1", "whole"]
    assert answer["choice"] == "unit" and answer["preferred"] == "whole"
    assert datetime.fromisoformat(answer["at"]).tzinfo
    assert not (root / "labels_fr_step2.json").exists()
    assert not (root / "labels_fr.json").exists()
    form, page = page_form(app, cookie)
    assert form["item_id"] == "f1" and form["step"] == "2"
    assert "焦点他回复1" in page and "焦点他回复后续1" in page
    for unit in ("u1", "u2", "whole"):
        assert f'name="hit_{unit}" value="yes"' in page
        assert f'name="rank_{unit}"' in page
    assert "accept_" not in page and 'name="preferred"' not in page
    assert "第一步" in page and "首选：整体关系" in page
    assert "可接受：今天太忙了1、整体关系" in page
    assert answer["at"] not in page


@pytest.mark.parametrize(
    "accept,preferred,choice,stored",
    [
        ((), "@insufficient", "insufficient", None),
        (("u2",), "@insufficient", "insufficient", None),
        (("u1", "u2"), "@none", "no_unique", None),
    ],
)
def test_step1_special_choices(tmp_path, accept, preferred, choice, stored):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie, accept=accept, preferred=preferred) == 303
    answer = saved(root, "fr", 1)["f1"]
    assert answer["acceptable"] == list(accept)
    assert answer["choice"] == choice and answer["preferred"] is stored


@pytest.mark.parametrize(
    "accept,preferred,extra",
    [
        ((), "@none", {}),
        ((), "u1", {}),
        (("u1",), "u2", {}),
        (("u1",), None, {}),
        (("u1",), "missing", {}),
        (("u1",), "u1", {"accept_missing": "yes"}),
        (("u1",), "u1", {"accept_u2": "no"}),
        (("u1",), "u1", {"rank_u1": "1"}),
        (("u1",), "u1", {"label": "A"}),
        (("u1",), "u1", {"step": "2"}),
    ],
)
def test_step1_rules_reject_without_writing(tmp_path, accept, preferred, extra):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie, accept=accept, preferred=preferred, **extra) == 400
    assert not (root / "labels_fr_step1.json").exists()


def test_step2_saves_ordered_hits_and_advances(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie) == 303
    assert step2(app, cookie, {"u2": "1", "whole": "3", "u1": "2"}) == 303
    answer = saved(root, "fr", 2)["f1"]
    assert answer["responded"] == ["u2", "u1", "whole"]
    form, page = page_form(app, cookie)
    assert form["item_id"] == "f2" and form["step"] == "1"
    assert "1/3" in page and "焦点他回复" not in page


def test_step2_allows_zero_hits(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie) == 303
    assert step2(app, cookie, {}) == 303
    assert saved(root, "fr", 2)["f1"]["responded"] == []


@pytest.mark.parametrize(
    "ranks,extra",
    [
        ({"u1": "1", "u2": "1"}, {}),
        ({"u1": "1", "u2": "3"}, {}),
        ({"u1": "2"}, {}),
        ({"u1": "0"}, {}),
        ({"u1": "01"}, {}),
        ({"u1": "x"}, {}),
        ({"u1": ""}, {}),
        ({}, {"rank_u1": "1"}),
        ({"u1": "1"}, {"hit_u1": "no"}),
        ({"u1": "1"}, {"hit_missing": "yes"}),
        ({"u1": "1"}, {"accept_u1": "yes"}),
        ({"u1": "1"}, {"step": "1"}),
    ],
)
def test_step2_order_rules_reject_without_writing(tmp_path, ranks, extra):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie) == 303
    before = (root / "labels_fr_step1.json").read_bytes()
    assert step2(app, cookie, ranks, **extra) == 400
    assert not (root / "labels_fr_step2.json").exists()
    assert (root / "labels_fr_step1.json").read_bytes() == before


def test_step2_cannot_be_submitted_before_step1(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    form, _ = page_form(app, cookie)
    form.update(action="label", step="2", hit_u1="yes", rank_u1="1", rank_u2="", rank_whole="")
    assert post(app, cookie, form) == 400
    assert list(root.glob("labels_*")) == []


def test_step1_is_frozen_once_saved(tmp_path):
    from companion.admin.focus_labeling import FocusLabelStore

    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie) == 303
    store = FocusLabelStore(root)
    with pytest.raises(ValueError):
        store.record_step1("fr", "f1", ["u2"], "u2")
    assert step2(app, cookie, {"u1": "1"}) == 303
    before = (root / "labels_fr_step1.json").read_bytes()
    with pytest.raises(ValueError):
        store.record_step1("fr", "f1", ["u2"], "u2")
    with pytest.raises(ValueError):
        store.record_step2("fr", "f1", ["u2"])
    # 表单把旧题目当作第一步提交也不行。
    form, _ = page_form(app, cookie)
    form.update(action="label", item_id="f1", accept_u2="yes", preferred="u2")
    assert post(app, cookie, form) == 400
    assert (root / "labels_fr_step1.json").read_bytes() == before


def test_undo_removes_only_latest_save_step_by_step(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie) == 303
    assert step2(app, cookie) == 303
    assert step1(app, cookie, accept=("u2",), preferred="u2") == 303
    assert undo(app, cookie) == 303
    assert list(saved(root, "fr", 1)) == ["f1"] and list(saved(root, "fr", 2)) == ["f1"]
    form, _ = page_form(app, cookie)
    assert (form["item_id"], form["step"]) == ("f2", "1")
    assert undo(app, cookie) == 303
    assert list(saved(root, "fr", 1)) == ["f1"] and saved(root, "fr", 2) == {}
    form, page = page_form(app, cookie)
    assert (form["item_id"], form["step"]) == ("f1", "2") and "0/3" in page
    assert undo(app, cookie) == 303
    assert saved(root, "fr", 1) == {}
    form, _ = page_form(app, cookie)
    assert (form["item_id"], form["step"]) == ("f1", "1")
    assert undo(app, cookie) == 303
    assert saved(root, "fr", 1) == {} and saved(root, "fr", 2) == {}


def test_undo_never_removes_step1_while_its_step2_exists(tmp_path):
    from companion.admin.focus_labeling import FocusLabelStore

    _, _, root = focus_app(tmp_path)
    store = FocusLabelStore(root)
    store.record_step1("fr", "f1", ["u1"], "u1")
    store.record_step2("fr", "f1", ["u1"])
    first = saved(root, "fr", 1)
    second = saved(root, "fr", 2)
    first["f1"]["at"] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    store._write_json(root / "labels_fr_step1.json", first)
    store.undo("fr")
    assert saved(root, "fr", 1) == first
    assert saved(root, "fr", 2) == {} and second


def test_completed_round_offers_only_undo(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    for _ in range(3):
        assert step1(app, cookie) == 303
        assert step2(app, cookie, {}) == 303
    status, _, page = call(app, f"{BASE}/fr", cookie=cookie)
    assert status == 200 and "本轮已完成" in page and "3/3" in page
    assert "焦点" not in page.replace("焦点轮", "") and 'value="label"' not in page
    assert undo(app, cookie) == 303
    assert len(saved(root, "fr", 2)) == 2 and len(saved(root, "fr", 1)) == 3


def test_step_files_are_private_and_separate(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    original = (root / "label_tasks.json").read_bytes()
    assert step1(app, cookie) == 303
    assert step2(app, cookie) == 303
    for step in (1, 2):
        path = root / f"labels_fr_step{step}.json"
        assert path.stat().st_mode & 0o777 == 0o600
    assert set(saved(root, "fr", 1)["f1"]) == {"acceptable", "choice", "preferred", "at"}
    assert set(saved(root, "fr", 2)["f1"]) == {"responded", "at"}
    assert sorted(path.name for path in root.iterdir()) == [
        "label_tasks.json", "labels_fr_step1.json", "labels_fr_step2.json"
    ]
    assert (root / "label_tasks.json").read_bytes() == original


def test_future_focus_round_shows_date_only_and_rejects_post(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    status, _, page = call(app, f"{BASE}/fr3", cookie=cookie)
    assert status == 200 and "将于 2026-10-16 开放" in page
    assert "<form" not in page and "焦点她" not in page and "整体关系" not in page
    form, _ = page_form(app, cookie)
    form.update(action="label", accept_u1="yes", preferred="u1")
    assert post(app, cookie, form, "fr3") == 403
    assert list(root.glob("labels_*")) == []


@pytest.mark.parametrize(
    "now,opened",
    [(datetime(2026, 10, 8, 15, 59, 59, tzinfo=UTC), False), (NOW, True)],
)
def test_focus_opening_date_uses_beijing_midnight(tmp_path, now, opened):
    app, cookie, _ = focus_app(tmp_path, now=now)
    status, _, page = call(app, f"{BASE}/fr2", cookie=cookie)
    assert status == 200 and ("<form" in page) == opened


@pytest.mark.parametrize("field", ["csrf", "labels_token"])
def test_focus_tokens_follow_strategy_rules(tmp_path, field):
    app, cookie, root = focus_app(tmp_path)
    form, _ = page_form(app, cookie)
    form.update(action="label", accept_u1="yes", preferred="u1")
    form[field] = "wrong"
    assert post(app, cookie, form) == (403 if field == "csrf" else 303)
    assert list(root.glob("labels_*")) == []


def test_focus_once_token_cannot_be_reused_or_cross_rounds(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    form, _ = page_form(app, cookie)
    form.update(action="label", accept_u1="yes", preferred="u1")
    assert post(app, cookie, form) == 303
    before = (root / "labels_fr_step1.json").read_bytes()
    assert post(app, cookie, form) == 303
    assert (root / "labels_fr_step1.json").read_bytes() == before
    other, _ = page_form(app, cookie, "fr2")
    other.update(action="label", accept_u1="yes", preferred="u1")
    assert post(app, cookie, other) == 403
    assert not (root / "labels_fr2_step1.json").exists()


def test_focus_rounds_do_not_echo_other_rounds(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie, accept=("u2",), preferred="u2") == 303
    assert post(app, cookie, label_form(app, cookie, label="E"), "r1") == 303
    form, page = page_form(app, cookie, "fr2")
    assert (form["item_id"], form["step"]) == ("f1", "1")
    assert "checked" not in page and "焦点他回复" not in page
    assert saved(root, "fr", 1)["f1"]["at"] not in page
    (root / "labels_fr_step1.json").write_text("FR_ANSWER_MUST_NOT_BE_READ")
    (root / "labels_r1.json").write_text("R1_ANSWER_MUST_NOT_BE_READ")
    assert step1(app, cookie, round_id="fr2") == 303
    assert list(saved(root, "fr2", 1)) == ["f1"]


def test_list_counts_completed_focus_items(tmp_path):
    app, cookie, _ = focus_app(tmp_path)
    assert step1(app, cookie) == 303
    status, _, page = call(app, BASE, cookie=cookie)
    assert status == 200 and "焦点轮</a> 0/3" in page
    assert step2(app, cookie) == 303
    status, _, page = call(app, BASE, cookie=cookie)
    assert "焦点轮</a> 1/3" in page and "将于 2026-10-16 开放" in page


def test_corrupt_step_file_is_unavailable(tmp_path):
    app, cookie, root = focus_app(tmp_path)
    assert step1(app, cookie) == 303
    data = saved(root, "fr", 1)
    data["f1"]["preferred"] = "u2"
    (root / "labels_fr_step1.json").write_text(json.dumps(data))
    status, _, page = call(app, f"{BASE}/fr", cookie=cookie)
    assert status == 503 and "标注任务未就绪" in page


def test_focus_text_is_html_escaped(tmp_path):
    data = focus_data()
    text = '<script>"x"</script>'
    data["items"]["f1"]["units"][0]["text"] = text
    data["items"]["f1"]["partner"] = [text]
    app, cookie, _ = make_app(tmp_path, data=data)
    status, _, page = call(app, f"{BASE}/fr", cookie=cookie)
    assert status == 200 and "<script>" not in page
    assert "&lt;script&gt;&quot;x&quot;&lt;/script&gt;" in page
