# ruff: noqa: RUF001
import json
import re
from datetime import UTC, datetime

import pytest
from test_local_entry import call, config, settings
from test_mapping_confirmation import ORIGIN, unlock

from companion.admin.app import AdminApp, build_admin_app, load_admin_config

BASE = "/admin/style/labels"
NOW = datetime(2026, 10, 8, 16, tzinfo=UTC)  # 北京时间 10 月 9 日零点


def task_data():
    return {
        "version": 1,
        "labels": [
            {"code": code, "name": f"说明{code}", "hint": f"提示{code}"} for code in "ABCDE"
        ],
        "items": {
            f"i{index}": {
                "context": [f"合成前文{index}"],
                "partner": [f"合成她{index}", f"合成她后续{index}"],
                "reply": [f"合成他{index}", f"合成他后续{index}"],
            }
            for index in range(1, 5)
        },
        "rounds": [
            {"id": "r1", "title": "第一轮", "opens_on": None, "order": ["i1", "i2", "i3", "i4"]},
            {"id": "r2", "title": "第二轮", "opens_on": "2026-10-09", "order": ["i1"]},
            {"id": "r3", "title": "第三轮", "opens_on": "2026-10-16", "order": ["i1"]},
        ],
    }


def make_app(tmp_path, *, data=None, root=True, now=NOW):
    label_root = tmp_path / "synthetic-labels" if root else None
    if label_root is not None:
        label_root.mkdir(mode=0o700)
        (label_root / "label_tasks.json").write_text(
            json.dumps(task_data() if data is None else data), encoding="utf-8"
        )
    app = AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        label_root=label_root,
        wall_clock=lambda: now,
    )
    return app, unlock(app), label_root


def label_form(app, cookie, round_id="r1", *, action="label", label="A"):
    status, _, page = call(app, f"{BASE}/{round_id}", cookie=cookie)
    assert status == 200
    return {
        "csrf": re.search(r'name="csrf" value="([^"]*)"', page).group(1),
        "labels_token": re.search(r'name="labels_token" value="([^"]*)"', page).group(1),
        "item_id": re.search(r'name="item_id" value="([^"]*)"', page).group(1),
        "action": action,
        "label": label,
    }


def submit(app, cookie, form, round_id="r1", *, origin=ORIGIN):
    return call(app, f"{BASE}/{round_id}", method="POST", cookie=cookie, origin=origin, form=form)


@pytest.mark.parametrize("path", [BASE, f"{BASE}/r1"])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_requires_logged_in_session(tmp_path, path, method):
    app, _, _ = make_app(tmp_path)
    status, _, page = call(app, path, method=method, origin=ORIGIN)
    assert status == 403
    assert "合成" not in page


def test_future_round_has_date_and_no_form_and_rejects_post(tmp_path):
    app, cookie, root = make_app(tmp_path)
    status, _, page = call(app, f"{BASE}/r3", cookie=cookie)
    assert status == 200
    assert "将于 2026-10-16 开放" in page
    assert "<form" not in page and "labels_token" not in page
    assert "合成她" not in page
    form = label_form(app, cookie)
    assert submit(app, cookie, form, "r3")[0] == 403
    assert not (root / "labels_r3.json").exists()


@pytest.mark.parametrize(
    "now,opened",
    [
        (datetime(2026, 10, 8, 15, 59, 59, tzinfo=UTC), False),
        (NOW, True),
    ],
)
def test_opening_date_uses_beijing_midnight(tmp_path, now, opened):
    app, cookie, _ = make_app(tmp_path, now=now)
    status, _, page = call(app, f"{BASE}/r2", cookie=cookie)
    assert status == 200
    assert ("<form" in page) == opened


def test_sequential_labels_are_private_and_advance_progress(tmp_path):
    app, cookie, root = make_app(tmp_path)
    original = (root / "label_tasks.json").read_bytes()
    for index, label in enumerate(["A", "?", "E"], 1):
        form = label_form(app, cookie, label=label)
        assert form["item_id"] == f"i{index}"
        status, headers, page = submit(app, cookie, form)
        assert status == 303 and headers["Location"] == f"{BASE}/r1"
        assert "合成" not in page
    path = root / "labels_r1.json"
    answers = json.loads(path.read_text())
    assert list(answers) == ["i1", "i2", "i3"]
    assert [answer["label"] for answer in answers.values()] == ["A", "?", "E"]
    assert all(datetime.fromisoformat(answer["at"]).tzinfo for answer in answers.values())
    assert path.stat().st_mode & 0o777 == 0o600
    assert (root / "label_tasks.json").read_bytes() == original
    status, _, page = call(app, f"{BASE}/r1", cookie=cookie)
    assert status == 200 and "3/4" in page and "合成她4" in page
    assert 'name="item_id" value="i4"' in page
    assert "合成她1" not in page
    for answer in answers.values():
        assert answer["at"] not in page
    for label in "ABCDE?":
        assert f'name="label" value="{label}"' in page
    assert "撤销上一条" in page


@pytest.mark.parametrize(
    "field,value",
    [
        ("label", "Z"),
        ("item_id", "i4"),
        ("item_id", "unknown"),
        ("csrf", "wrong"),
        ("labels_token", "wrong"),
        ("action", "other"),
        ("csrf", "错误"),
        ("labels_token", "错误"),
    ],
)
def test_invalid_submission_does_not_change_answers(tmp_path, field, value):
    app, cookie, root = make_app(tmp_path)
    assert submit(app, cookie, label_form(app, cookie))[0] == 303
    path = root / "labels_r1.json"
    before = path.read_bytes()
    form = label_form(app, cookie)
    form[field] = value
    status, headers, _ = submit(app, cookie, form)
    expected = 303 if field == "labels_token" else 403 if field == "csrf" else 400
    assert status == expected
    if status == 303:
        assert headers["Location"] == f"{BASE}/r1"
    assert path.read_bytes() == before


def test_token_reuse_and_stale_item_do_not_skip_items(tmp_path):
    app, cookie, root = make_app(tmp_path)
    form = label_form(app, cookie)
    assert submit(app, cookie, form)[0] == 303
    path = root / "labels_r1.json"
    before = path.read_bytes()
    status, headers, _ = submit(app, cookie, form)
    assert status == 303 and headers["Location"] == f"{BASE}/r1"
    assert path.read_bytes() == before
    assert len(json.loads(before)) == 1
    status, _, page = call(app, headers["Location"], cookie=cookie)
    assert status == 200 and 'name="item_id" value="i2"' in page
    fresh = label_form(app, cookie)
    fresh["item_id"] = form["item_id"]
    assert 400 <= submit(app, cookie, fresh)[0] < 500
    assert path.read_bytes() == before


def test_tokens_are_bound_to_round_and_session_and_latest_get(tmp_path):
    app, cookie, root = make_app(tmp_path)
    form = label_form(app, cookie)
    assert submit(app, cookie, form, "r2")[0] == 403
    form = label_form(app, cookie)
    other_cookie = unlock(app)
    assert submit(app, other_cookie, form)[0] == 403
    stale = label_form(app, cookie)
    label_form(app, cookie)
    status, headers, _ = submit(app, cookie, stale)
    assert status == 303 and headers["Location"] == f"{BASE}/r1"
    assert not (root / "labels_r1.json").exists()


@pytest.mark.parametrize("fault", ["missing", "round"])
def test_invalid_once_token_with_valid_csrf_redirects_without_writing(tmp_path, fault):
    app, cookie, root = make_app(tmp_path)
    form = label_form(app, cookie, "r2")
    if fault == "missing":
        del form["labels_token"]
    else:
        label_form(app, cookie, "r1")
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    status, headers, _ = submit(app, cookie, form, "r2")
    assert status == 303 and headers["Location"] == f"{BASE}/r2"
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    status, _, page = call(app, headers["Location"], cookie=cookie)
    assert status == 200 and 'name="item_id" value="i1"' in page


def test_undo_removes_latest_and_returns_to_that_item(tmp_path):
    app, cookie, root = make_app(tmp_path)
    for _ in range(3):
        assert submit(app, cookie, label_form(app, cookie))[0] == 303
    assert submit(app, cookie, label_form(app, cookie, action="undo", label=""))[0] == 303
    assert list(json.loads((root / "labels_r1.json").read_text())) == ["i1", "i2"]
    status, _, page = call(app, f"{BASE}/r1", cookie=cookie)
    assert status == 200 and "2/4" in page and 'name="item_id" value="i3"' in page


def test_completed_round_can_undo_without_echoing_labels(tmp_path):
    app, cookie, root = make_app(tmp_path)
    for _ in range(4):
        assert submit(app, cookie, label_form(app, cookie))[0] == 303
    status, _, page = call(app, f"{BASE}/r1", cookie=cookie)
    assert status == 200 and "本轮已完成" in page and "4/4" in page
    assert "合成她" not in page and 'name="action" value="label"' not in page
    assert submit(app, cookie, label_form(app, cookie, action="undo", label=""))[0] == 303
    assert len(json.loads((root / "labels_r1.json").read_text())) == 3


def test_round_answers_never_leak_into_other_round(tmp_path):
    app, cookie, root = make_app(tmp_path)
    assert submit(app, cookie, label_form(app, cookie, label="E"))[0] == 303
    raw = (root / "labels_r1.json").read_text()
    answer = json.loads(raw)["i1"]
    status, _, page = call(app, f"{BASE}/r2", cookie=cookie)
    assert status == 200 and "0/1" in page and "合成她1" in page
    assert raw not in page and answer["at"] not in page
    assert '"label": "E"' not in page and "labels_r1.json" not in page
    assert not (root / "labels_r2.json").exists()
    # r2 不读取 r1 答案, 连损坏的 r1 文件也不会影响它。
    (root / "labels_r1.json").write_text("R1_ANSWER_MUST_NOT_BE_READ")
    assert call(app, f"{BASE}/r2", cookie=cookie)[0] == 200
    status, _, response = submit(app, cookie, label_form(app, cookie, "r2", label="?"), "r2")
    assert status == 303 and "R1_ANSWER" not in response


def test_all_input_text_is_html_escaped(tmp_path):
    data = task_data()
    text = '<script>alert("synthetic")</script>'
    data["labels"][0] = {"code": '<"', "name": text, "hint": text}
    data["rounds"][0]["title"] = text
    data["items"]["i1"] = {field: [text] for field in ("context", "partner", "reply")}
    app, cookie, _ = make_app(tmp_path, data=data)
    for path in [BASE, f"{BASE}/r1"]:
        status, _, page = call(app, path, cookie=cookie)
        assert status == 200 and "<script>" not in page
        assert "&lt;script&gt;alert(&quot;synthetic&quot;)&lt;/script&gt;" in page
    assert 'value="&lt;&quot;"' in page
    assert "<strong>他：" in page
    assert '<small><font color="gray">' in page


INVALID_TASKS = [
    ("version", 2),
    ("version", True),
    ("labels", {}),
    ("labels", []),
    ("labels.0.code", "ABC"),
    ("labels.0.code", ""),
    ("labels.0.code", "?"),
    ("labels.0.code", "B"),
    ("labels.0.name", None),
    ("labels.0.hint", 2),
    ("items", []),
    ("items.i1.context", "text"),
    ("items.i1.partner", [None]),
    ("items.i1.reply", None),
    ("rounds", {}),
    ("rounds.0.id", "../escape"),
    ("rounds.0.id", "R1"),
    ("rounds.0.id", "x" * 33),
    ("rounds.0.id", "r2"),
    ("rounds.0.title", None),
    ("rounds.0.opens_on", "2026-02-30"),
    ("rounds.0.opens_on", "20261009"),
    ("rounds.0.opens_on", 20261009),
    ("rounds.0.order", ["missing"]),
    ("rounds.0.order", ["i1", "i1"]),
    ("rounds.0.order", [1]),
    ("rounds.0.order", "i1"),
]


@pytest.mark.parametrize("field,value", INVALID_TASKS)
def test_invalid_tasks_are_unavailable(tmp_path, field, value):
    data = task_data()
    parts = field.split(".")
    target = data
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    target[int(parts[-1]) if isinstance(target, list) else parts[-1]] = value
    app, cookie, _ = make_app(tmp_path, data=data)
    for path in [BASE, f"{BASE}/r1"]:
        status, _, page = call(app, path, cookie=cookie)
        assert status == 503 and "标注任务未就绪" in page
        assert "<form" not in page


@pytest.mark.parametrize("problem", ["none", "missing", "json", "schema", "encoding"])
def test_missing_or_unreadable_tasks_are_unavailable(tmp_path, problem):
    app, cookie, root = make_app(tmp_path, root=problem != "none")
    if root:
        path = root / "label_tasks.json"
        if problem == "missing":
            path.unlink()
        elif problem == "encoding":
            path.write_bytes(b"\xff")
        else:
            path.write_text("{" if problem == "json" else "{}")
    for route in [BASE, f"{BASE}/r1"]:
        status, _, page = call(app, route, cookie=cookie)
        assert status == 503 and "标注任务未就绪" in page


def test_list_shows_only_counts_and_opening_states_and_home_link(tmp_path):
    app, cookie, root = make_app(tmp_path)
    assert submit(app, cookie, label_form(app, cookie, label="E"))[0] == 303
    status, _, page = call(app, BASE, cookie=cookie)
    assert status == 200 and "1/4" in page and "0/1" in page
    assert "已开放" in page and "将于 2026-10-16 开放" in page
    for round_id in ["r1", "r2", "r3"]:
        assert f'href="{BASE}/{round_id}"' in page
    answer = json.loads((root / "labels_r1.json").read_text())["i1"]
    assert answer["at"] not in page and "合成她" not in page
    status, _, page = call(app, "/admin/home", cookie=cookie)
    assert status == 200
    assert f'<li><a href="{BASE}">回应策略标注</a></li>' in page


def test_product_assembly_uses_m1_directory(tmp_path):
    root = tmp_path / "synthetic-private"
    config(root)
    trusted = settings(root)
    app = build_admin_app(trusted, load_admin_config(trusted), port=12345)
    assert app._label_root == root / "style" / "m1"


def test_store_overwrite_and_undo_use_latest_timestamp(tmp_path):
    from companion.admin.labeling import LabelStore

    _, _, root = make_app(tmp_path)
    store = LabelStore(root)
    assert store.tasks() == task_data()
    assert store.answers("r1") == {} and store.next_item("r1") == "i1"
    for item_id in ["i1", "i2", "i1"]:
        store.record("r1", item_id, "E")
    assert len(store.answers("r1")) == 2
    assert store.next_item("r1") == "i3"
    store.undo("r1")
    assert set(store.answers("r1")) == {"i2"}
    assert store.next_item("r1") == "i1"
    store.undo("r1")
    store.undo("r1")
    assert store.answers("r1") == {}
    before = (root / "labels_r1.json").read_bytes()
    for round_id, item_id, label in [("r1", "i1", "Z"), ("r2", "i2", "A")]:
        with pytest.raises(ValueError):
            store.record(round_id, item_id, label)
    assert (root / "labels_r1.json").read_bytes() == before


@pytest.mark.parametrize(
    "path,method",
    [
        (f"{BASE}/missing", "GET"),
        (f"{BASE}/r1/extra", "GET"),
        (f"{BASE}/r1", "PUT"),
        (BASE, "POST"),
    ],
)
def test_unknown_round_or_method_is_not_found(tmp_path, path, method):
    app, cookie, _ = make_app(tmp_path)
    assert call(app, path, method=method, cookie=cookie, origin=ORIGIN)[0] == 404
