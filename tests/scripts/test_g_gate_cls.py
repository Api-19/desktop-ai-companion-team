# ruff: noqa: RUF001
"""Zero-shot classifier gate arm with injected fake models; laya/gliclass/torch never imported."""

import json
import sys

import pytest

from scripts import g_gate_cls as cls

LAYA_PATH = (
    "/m/models--convaiinnovations--laya-multilingual/snapshots/"
    "1720e3e3357cfe1e281542e223f8273b0890ca34"
)
GLICLASS_PATH = (
    "/m/models--knowledgator--gliclass-multilang-mini/snapshots/"
    "0bd888b6c3ef9fca5f0a9d407bddfbbc7623486b"
)
ROW_KEYS = {"event_id", "cand_id", "mode", "model", "decision", "p_yes", "fields",
            "technical_failure"}


def event(eid="e1", permission=None, n=1):
    return {
        "event_id": eid, "context": ["她：合成话一", "他：合成话二"], "permission": permission,
        "candidates": [{"cand_id": f"c{i}", "source": "similar", "c": ["她：合成旧话"],
                        "r": ["合成回复一", "合成回复二"]} for i in range(1, n + 1)],
    }


class FakeLaya:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return {"answers": {"q": {"type": "noul", "noul": value}},
                "usage": {"truncated": False}}


class FakeGliclass:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def __call__(self, texts, labels, **kwargs):
        self.calls.append((texts, labels, kwargs))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return [dict(zip(labels, value, strict=True))]


def test_text_is_frozen_without_permission():
    assert cls.render_text(event(), event()["candidates"][0]) == (
        "当前对话：\n她：合成话一\n他：合成话二\n参考·当时对方说：\n她：合成旧话"
        "\n参考·他的回复：\n合成回复一\n合成回复二"
    )


def test_text_appends_permission_note():
    item = event(permission={"nickname_ok": False, "note": "不许用昵称。"})
    assert cls.render_text(item, item["candidates"][0]).endswith(
        "\n参考·他的回复：\n合成回复一\n合成回复二\n许可：不许用昵称。",
    )


def test_laya_question_and_p_yes():
    fake = FakeLaya([0.73])
    scorer = cls.laya_scorer(fake)
    p_yes, fields = scorer("文本")
    assert p_yes == 0.73
    assert fields == {"truncated": False}
    state, questions = fake.calls[0]
    assert state == "文本"
    assert questions == {"q": {"type": "noul",
                               "instructions": "把参考回复的回应方式用在当前对话里合适吗？"}}


def test_gliclass_labels_and_normalised_p_yes():
    fake = FakeGliclass([(0.3, 0.1)])
    p_yes, fields = cls.gliclass_scorer(fake)("文本")
    assert p_yes == pytest.approx(0.75)
    assert fields == {"s_yes": 0.3, "s_no": 0.1}
    texts, labels, kwargs = fake.calls[0]
    assert texts == "文本"
    assert labels == ["合适", "不合适"]
    assert kwargs["classification_type"] == "single-label"
    assert kwargs["return_hierarchical"] is True


def test_gliclass_already_normalised():
    p_yes, _ = cls.gliclass_scorer(FakeGliclass([(0.2, 0.8)]))("文本")
    assert p_yes == pytest.approx(0.2)


@pytest.mark.parametrize(("p", "decision"), [(0.5, "keep"), (0.9, "keep"), (0.4999, "drop"),
                                             (0.0, "drop")])
def test_threshold_decision(p, decision):
    rows = cls.classify([event()], cls.laya_scorer(FakeLaya([p])), "laya@x")
    assert rows == [{"event_id": "e1", "cand_id": "c1", "mode": "cls", "model": "laya@x",
                     "decision": decision, "p_yes": p, "fields": {"truncated": False},
                     "technical_failure": False}]


def test_classify_passes_frozen_text_per_candidate():
    fake = FakeGliclass([(0.9, 0.1), (0.1, 0.9)])
    item = event(n=2, permission={"nickname_ok": True, "note": "可以。"})
    rows = cls.classify([item], cls.gliclass_scorer(fake), "gliclass@x")
    assert [r["decision"] for r in rows] == ["keep", "drop"]
    assert [c[0] for c in fake.calls] == [cls.render_text(item, c) for c in item["candidates"]]


def test_inference_error_is_technical_failure_without_retry():
    fake = FakeLaya([RuntimeError("boom"), 0.8])
    rows = cls.classify([event(n=2)], cls.laya_scorer(fake), "laya@x")
    assert rows[0] == {"event_id": "e1", "cand_id": "c1", "mode": "cls", "model": "laya@x",
                       "decision": "drop", "p_yes": None, "fields": {},
                       "technical_failure": True}
    assert rows[1]["decision"] == "keep" and not rows[1]["technical_failure"]
    assert len(fake.calls) == 2


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.1])
def test_out_of_range_probability_is_technical_failure(bad):
    rows = cls.classify([event()], cls.laya_scorer(FakeLaya([bad])), "laya@x")
    assert rows[0]["technical_failure"] and rows[0]["p_yes"] is None


def test_model_name_uses_backend_and_revision():
    assert cls.model_name("laya", LAYA_PATH) == "laya@1720e3e3357cfe1e281542e223f8273b0890ca34"
    assert cls.model_name("gliclass", "/x/local-dir") == "gliclass@local-dir"


def write_events(path, events):
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
                    encoding="utf-8")


@pytest.mark.parametrize(("backend", "path", "fake"), [
    ("laya", LAYA_PATH, lambda: cls.laya_scorer(FakeLaya([0.9, 0.1]))),
    ("gliclass", GLICLASS_PATH, lambda: cls.gliclass_scorer(FakeGliclass([(0.9, 0.1),
                                                                         (0.1, 0.9)]))),
])
def test_main_with_injected_loader(tmp_path, capsys, backend, path, fake):
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_events(src, [event("e1"), event("e2")])
    seen = []

    def loader(name, model_path):
        seen.append((name, model_path))
        return fake()

    code = cls.main(["--input", str(src), "--backend", backend, "--model", path,
                     "--out", str(out)], loader=loader)
    assert code == 0
    assert seen == [(backend, path)]
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [set(r) for r in rows] == [ROW_KEYS, ROW_KEYS]
    assert [r["decision"] for r in rows] == ["keep", "drop"]
    assert {r["model"] for r in rows} == {cls.model_name(backend, path)}
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary["technical_failures"] == 0 and summary["candidates"] == 2
    for name in ("laya", "gliclass", "torch"):
        assert name not in sys.modules


def test_main_load_failure_marks_every_candidate(tmp_path, capsys):
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_events(src, [event("e1", n=2)])

    def loader(name, model_path):
        raise OSError("no model")

    assert cls.main(["--input", str(src), "--backend", "laya", "--model", LAYA_PATH,
                     "--out", str(out)], loader=loader) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["decision"], r["p_yes"], r["technical_failure"]) for r in rows] == [
        ("drop", None, True), ("drop", None, True)]
    assert json.loads(capsys.readouterr().out)["technical_failures"] == 2


def test_rejects_unknown_backend(tmp_path):
    with pytest.raises(SystemExit):
        cls.main(["--input", str(tmp_path / "x"), "--backend", "bert", "--model", "m",
                  "--out", str(tmp_path / "o")], loader=lambda *_: None)
