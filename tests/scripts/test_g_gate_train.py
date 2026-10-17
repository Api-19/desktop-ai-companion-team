# ruff: noqa: RUF001
"""Fine-tuned gate arm helpers with fake models and synthetic pairs; never reads ~/.rcpem."""

import json
import math

import pytest

from scripts import g_gate_score_v2 as score_v2
from scripts import g_gate_train as gt

ROW_KEYS = {"event_id", "cand_id", "mode", "model", "decision", "p_yes", "fields",
            "technical_failure"}


def pair(src="s1", variant="pos_near", label="yes", split="train", source="cped",
         x="合成当前一／合成当前二", c="合成旧话一／合成旧话二", r="合成回复一／合成回复二"):
    return {"pair_id": f"{src}-{variant}", "variant": variant, "src_id": src, "speaker": "甲",
            "source": source, "show": None, "split": split, "c": c, "r": r, "x": x,
            "label": label}


def four(src, split="train", source="cped"):
    return [pair(src, v, "yes" if v.startswith("pos") else "no", split, source)
            for v in ("pos_near", "pos_far", "neg_near", "neg_far")]


class FakeArm:
    """Returns a fixed logit per rendered text; records calls."""

    def __init__(self, by_text=None, default=0.0):
        self.by_text = by_text or {}
        self.default = default
        self.calls = []

    def predict_logits(self, texts):
        self.calls.append(list(texts))
        return [self.by_text.get(t, self.default) for t in texts]


# ---------- rendering ----------

def test_pair_to_event_splits_clauses_and_has_no_permission():
    event = gt.pair_to_event(pair(x="现在一／现在二", c="旧一／ 旧二 ／", r="回一"))
    assert event["event_id"] == "s1-pos_near"
    assert event["context"] == ["现在一", "现在二"]
    assert event["permission"] is None
    assert event["candidates"] == [{"cand_id": "c1", "c": ["旧一", "旧二"], "r": ["回一"]}]


def test_render_pair_matches_zero_shot_format():
    text = gt.render_pair(pair(x="现在一／现在二", c="旧一／旧二", r="回一／回二"))
    assert text == ("当前对话：\n现在一\n现在二\n参考·当时对方说：\n旧一\n旧二"
                    "\n参考·他的回复：\n回一\n回二")
    assert "许可" not in text


# ---------- splits ----------

def test_group_split_keeps_src_ids_together_and_is_deterministic():
    pairs = [p for i in range(20) for p in four(f"s{i:02d}")]
    train, dev = gt.group_split(pairs, 0.2, gt.SEED)
    assert len(train) + len(dev) == len(pairs)
    assert {p["src_id"] for p in train}.isdisjoint({p["src_id"] for p in dev})
    assert len({p["src_id"] for p in dev}) == 4
    assert all(sum(q["src_id"] == p["src_id"] for q in dev) == 4 for p in dev)
    again = gt.group_split(list(reversed(pairs)), 0.2, gt.SEED)
    assert {p["pair_id"] for p in again[1]} == {p["pair_id"] for p in dev}


def test_select_splits_uses_split_field_when_dev_present():
    pairs = four("a") + four("b", "valid") + four("c", "dev") + four("d", "test")
    train, dev, test, note = gt.select_splits(pairs, final=False)
    assert {p["src_id"] for p in train} == {"a"}
    assert {p["src_id"] for p in dev} == {"b", "c"}
    assert test == []
    assert note == "split-field"
    _, _, test, _ = gt.select_splits(pairs, final=True)
    assert {p["src_id"] for p in test} == {"d"}


def test_select_splits_falls_back_to_grouped_temp_dev_when_train_only():
    pairs = [p for i in range(10) for p in four(f"s{i}")]
    train, dev, test, note = gt.select_splits(pairs, final=False)
    assert note.startswith("temp-dev")
    assert dev and test == []
    assert {p["src_id"] for p in train}.isdisjoint({p["src_id"] for p in dev})


# ---------- calibration and metrics ----------

def test_temperature_fit_does_not_raise_nll():
    logits = [4.0, 3.0, -2.5, 5.0, -4.0, 2.0, -1.0, 3.5]
    labels = [1, 0, 0, 1, 1, 1, 0, 0]
    t = gt.fit_temperature(logits, labels)
    assert t > 1.0  # overconfident logits get softened
    assert gt.nll(logits, labels, t) <= gt.nll(logits, labels, 1.0)


def test_temperature_fit_on_perfectly_separated_data_still_no_worse():
    logits, labels = [0.5, 0.2, -0.3, -0.6], [1, 1, 0, 0]
    t = gt.fit_temperature(logits, labels)
    assert t < 1.0
    assert gt.nll(logits, labels, t) <= gt.nll(logits, labels, 1.0)


def test_auc_with_ties():
    assert gt.auc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]) == 1.0
    assert gt.auc([0.1, 0.3, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert gt.auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    assert gt.auc([0.9, 0.4, 0.4, 0.1], [1, 1, 0, 0]) == pytest.approx(0.875)


def test_brier_and_ece():
    assert gt.brier([1.0, 0.0, 0.5], [1, 0, 1]) == pytest.approx(0.25 / 3)
    # bin [0.9,1.0]: mean p 0.95, positives 1/2 -> gap 0.45, weight 2/4
    # bin [0.1,0.2): mean p 0.15, positives 0/2 -> gap 0.15, weight 2/4
    assert gt.ece([0.95, 0.95, 0.15, 0.15], [1, 0, 0, 0]) == pytest.approx(0.30)
    assert gt.ece([1.0, 0.0], [1, 0]) == 0.0  # p=1.0 falls in the last bin


def test_sigmoid_is_stable():
    assert gt.sigmoid(1000.0) == 1.0
    assert gt.sigmoid(-1000.0) == pytest.approx(0.0)
    assert gt.sigmoid(0.0) == 0.5


# ---------- predict rows ----------

def test_predict_pairs_rows_carry_pair_id_label_and_calibrated_p():
    pairs = [pair("a", "pos_near", "yes"), pair("a", "neg_far", "no", x="不同的当前话")]
    arm = FakeArm({gt.render_pair(pairs[0]): 2.0, gt.render_pair(pairs[1]): -2.0})
    rows = gt.predict_pairs(pairs, arm, temperature=2.0, model="ft-laya@ck", batch=8)
    assert [set(r) - ROW_KEYS for r in rows] == [{"pair_id", "label", "variant", "source"}] * 2
    first = rows[0]
    assert first["event_id"] == "a-pos_near" and first["cand_id"] == "c1"
    assert first["mode"] == "ft-laya" and first["model"] == "ft-laya@ck"
    assert first["p_yes"] == pytest.approx(gt.sigmoid(1.0))
    assert first["fields"]["p_raw"] == pytest.approx(gt.sigmoid(2.0))
    assert first["decision"] == "keep" and rows[1]["decision"] == "drop"
    assert first["technical_failure"] is False
    assert (first["pair_id"], first["label"], first["variant"], first["source"]) == (
        "a-pos_near", "yes", "pos_near", "cped")


def test_predict_events_rows_score_with_g_gate_score_v2():
    events = []
    for g in range(1, 3):
        for s, ctx in (("s1", "合成可用"), ("s2", "合成不可用")):
            events.append({"event_id": f"T1-G{g}-{s}", "context": [f"她：{ctx}{g}"],
                           "permission": None,
                           "candidates": [{"cand_id": "c1", "c": ["她：旧"], "r": ["回"]}]})
    arm = FakeArm({gt.cls.render_text(e, e["candidates"][0]): (3.0 if "s1" in e["event_id"]
                                                                else -3.0) for e in events})
    rows = gt.predict_events(events, arm, temperature=1.0, model="ft-gliclass@ck", batch=3)
    assert all(set(r) == ROW_KEYS for r in rows)
    assert {r["mode"] for r in rows} == {"ft-gliclass"}
    assert {r["decision"] for r in rows} <= {"keep", "drop"}
    key = {e["event_id"]: {"app": "yes" if "s1" in e["event_id"] else "no", "type": "T1"}
           for e in events}
    scenes = {e["event_id"]: e for e in events}
    result = score_v2.score(scenes, key, json.loads(json.dumps(rows)))
    assert result["type_groups_correct"]["T1"] == 2
    assert result["technical_failures"] == 0
    assert [len(c) for c in arm.calls] == [3, 1]


def test_decision_threshold_is_half():
    assert gt.decide(0.5) == "keep"
    assert gt.decide(0.4999) == "drop"


# ---------- report ----------

def test_report_metrics_by_variant_and_source():
    rows = []
    for i, (variant, label, p, source) in enumerate([
        ("pos_near", "yes", 0.9, "cped"), ("pos_far", "yes", 0.4, "livechat"),
        ("neg_near", "no", 0.6, "cped"), ("neg_far", "no", 0.1, "livechat"),
    ]):
        rows.append({"event_id": f"p{i}", "cand_id": "c1", "mode": "ft-laya", "model": "m",
                     "decision": gt.decide(p), "p_yes": p, "fields": {"p_raw": p},
                     "technical_failure": False, "pair_id": f"p{i}", "label": label,
                     "variant": variant, "source": source})
    rep = gt.report(rows)
    assert rep["n"] == 4
    assert rep["auc"] == pytest.approx(0.75)
    assert rep["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)
    assert rep["accuracy"] == 0.5
    assert rep["by_variant"] == {"neg_far": {"n": 1, "accuracy": 1.0},
                                 "neg_near": {"n": 1, "accuracy": 0.0},
                                 "pos_far": {"n": 1, "accuracy": 0.0},
                                 "pos_near": {"n": 1, "accuracy": 1.0}}
    assert rep["by_source"]["cped"] == {"n": 2, "accuracy": 0.5}
    assert rep["raw"]["auc"] == pytest.approx(0.75)


def test_surface_features_match_shortcut_check():
    feats = gt.surface_features({"x": "别去了，请直接说？", "c": "去吧", "r": "好"})
    assert len(feats) == 8
    assert feats[0] == len("别去了，请直接说？")
    assert feats[4] == 1  # 别 (NEG words counted by presence)
    assert feats[5] == 2  # 请, 直接
    assert feats[6] == 1 and feats[7] == 1


def test_require_mps_refuses_cpu():
    class NoMps:
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

    with pytest.raises(SystemExit):
        gt.require_mps(NoMps)


def test_nll_matches_definition():
    value = gt.nll([0.0, 2.0], [1, 0], 1.0)
    assert value == pytest.approx((math.log(2) + math.log(1 + math.exp(2.0))) / 2)
