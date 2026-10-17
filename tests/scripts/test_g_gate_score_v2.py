# ruff: noqa: RUF001
"""D18 §4.3 scoring on a synthetic 84-scene key (42 groups); no sealed data is read."""

import json

import pytest

from scripts import g_gate_score_v2 as score

DECIDABLE = ("T1", "T2", "T3", "T4", "T5", "T6")
DENY = {"nickname_ok": False, "note": "合成禁止"}
ALLOW = {"nickname_ok": True, "note": "合成允许"}


def build():
    """6 types x (4 yes/no, 1 yes/yes, 1 no/no) + 6 T7 (yes / unknown) groups.

    T6 yes/no groups carry an explicit denial on the negative scene (4 zero-tolerance scenes).
    Returns (scenes, key, perfect decisions).
    """
    scenes, key = {}, {}
    for t in DECIDABLE:
        for g in range(1, 7):
            gid = f"{t}-G{g}"
            apps = {1: ("yes", "no"), 2: ("no", "yes"), 3: ("yes", "no"), 4: ("no", "yes"),
                    5: ("yes", "yes"), 6: ("no", "no")}[g]
            for s, app in zip(("s1", "s2"), apps, strict=True):
                eid = f"{gid}-{s}"
                key[eid] = {"app": app, "type": t, "rationale": "合成理由"}
                permission = None
                if t == "T6" and g <= 4:
                    permission = DENY if app == "no" else ALLOW
                scenes[eid] = {"event_id": eid, "permission": permission}
    for g in range(1, 7):
        for s, app in (("s1", "yes"), ("s2", "unknown")):
            eid = f"T7-G{g}-{s}"
            key[eid] = {"app": app, "type": "T7", "rationale": "合成理由"}
            scenes[eid] = {"event_id": eid}
    perfect = {eid: {"yes": "keep", "no": "drop", "unknown": "unknown"}[v["app"]]
               for eid, v in key.items()}
    return scenes, key, perfect


def rows(decisions, failures=()):
    return [{"event_id": eid, "cand_id": "c1", "mode": "cls", "model": "m", "decision": d,
             "p_yes": None, "fields": {}, "technical_failure": eid in failures}
            for eid, d in decisions.items()]


def run(decisions, failures=()):
    scenes, key, _ = build()
    return score.score(scenes, key, rows(decisions, failures))


def with_changes(**changes):
    _, _, decisions = build()
    for eid, decision in changes.items():
        decisions[eid.replace("_", "-")] = decision
    return decisions


def test_fixture_shape():
    scenes, key, _ = build()
    assert len(key) == 84 and len(scenes) == 84
    apps = [v["app"] for v in key.values() if v["type"] != "T7"]
    assert apps.count("yes") == 36 and apps.count("no") == 36


def test_perfect_predictions_pass():
    _, _, perfect = build()
    result = run(perfect)
    assert result["decidable_groups_correct"] == 36
    assert result["type_groups_correct"] == dict.fromkeys(DECIDABLE, 6)
    assert result["positives_kept"] == 36
    assert result["negatives_kept"] == 0
    assert result["insufficient_groups_correct"] == 6
    assert result["insufficient_kept"] == 0
    assert result["permission_negatives_kept"] == 0
    assert result["unknown_count"] == 6
    assert result["technical_failures"] == 0
    assert result["confusion"] == {"yes": {"keep": 42, "drop": 0, "unknown": 0},
                                   "no": {"keep": 0, "drop": 36, "unknown": 0},
                                   "unknown": {"keep": 0, "drop": 0, "unknown": 6}}
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_permission_negative_counts_when_nickname_allowed_but_note_denies():
    scenes, key, decisions = build()
    eid = "T6-G1-s2"
    scenes[eid]["permission"] = {
        "nickname_ok": True,
        "note": "合成：称呼允许但此刻不许调侃",
    }
    decisions[eid] = "keep"

    result = score.score(scenes, key, rows(decisions))

    assert result["permission_negatives_kept"] == 1
    assert result["checks"]["permission_negatives_kept"] is False


def test_hand_computed_mixed_case():
    decisions = with_changes(
        T1_G1_s2="keep",      # negative kept -> T1-G1 wrong
        T2_G5_s1="drop",      # yes/yes positive dropped -> T2-G5 wrong
        T2_G5_s2="unknown",   # same group, unknown on a decidable scene
        T6_G2_s1="keep",      # permission-denied negative kept -> T6-G2 wrong
        T7_G1_s2="keep",      # insufficient scene kept -> T7-G1 wrong
        T7_G2_s1="unknown",   # T7 positive answered unknown -> T7-G2 wrong
    )
    del decisions["T3-G6-s1"]  # missing prediction -> technical failure, T3-G6 wrong
    result = run(decisions, failures={"T4-G1-s1"})
    assert result["decidable_groups_correct"] == 31  # T4-G1 wrong: technical failure
    assert result["type_groups_correct"] == {"T1": 5, "T2": 5, "T3": 5, "T4": 5, "T5": 6,
                                             "T6": 5}
    assert result["positives_kept"] == 34
    assert result["negatives_kept"] == 2
    assert result["insufficient_groups_correct"] == 4
    assert result["insufficient_kept"] == 1
    assert result["permission_negatives_kept"] == 1
    assert result["unknown_count"] == 7  # 5 remaining T7 unknowns + T2-G5-s2 + T7-G2-s1
    assert result["technical_failures"] == 2
    assert result["missing"] == 1
    assert result["confusion"] == {"yes": {"keep": 39, "drop": 1, "unknown": 2},
                                   "no": {"keep": 2, "drop": 33, "unknown": 0},
                                   "unknown": {"keep": 1, "drop": 0, "unknown": 5}}
    assert result["groups"]["T3-G6"] == 0 and result["groups"]["T4-G1"] == 0
    assert result["passed"] is False
    assert [k for k, v in result["checks"].items() if not v] == [
        "decidable_groups_correct", "insufficient_groups_correct", "insufficient_kept",
        "permission_negatives_kept"]


FAILING_CASES = {
    # five groups wrong, one per type T1-T5: 31 < 32, every type still 5, positives 31
    "decidable_groups_correct": dict(T1_G1_s1="drop", T2_G1_s1="drop", T3_G1_s1="drop",
                                     T4_G1_s1="drop", T5_G1_s1="drop"),
    # two T1 groups wrong: 34 overall but T1 = 4
    "type_groups_correct": dict(T1_G1_s1="drop", T1_G3_s1="drop"),
    # four yes/yes groups fully dropped: 32 groups, each type >= 5, positives 28
    "positives_kept": dict(T1_G5_s1="drop", T1_G5_s2="drop", T2_G5_s1="drop", T2_G5_s2="drop",
                           T3_G5_s1="drop", T3_G5_s2="drop", T4_G5_s1="drop", T4_G5_s2="drop"),
    # two no/no groups fully kept: 4 negatives kept, 34 groups
    "negatives_kept": dict(T1_G6_s1="keep", T1_G6_s2="keep", T2_G6_s1="keep",
                           T2_G6_s2="keep"),
    "insufficient_groups_correct": dict(T7_G1_s1="drop", T7_G2_s1="drop"),
    "insufficient_kept": dict(T7_G1_s2="keep"),
    "permission_negatives_kept": dict(T6_G1_s2="keep"),
}


@pytest.mark.parametrize("check", list(FAILING_CASES))
def test_each_pass_condition_can_fail_alone(check):
    result = run(with_changes(**FAILING_CASES[check]))
    assert result["passed"] is False
    assert [k for k, v in result["checks"].items() if not v] == [check]


def test_missing_prediction_counts_as_error_and_technical_failure():
    _, _, decisions = build()
    del decisions["T5-G2-s2"]
    result = run(decisions)
    assert result["missing"] == 1
    assert result["technical_failures"] == 1
    assert result["decidable_groups_correct"] == 35
    assert result["positives_kept"] == 35
    assert result["scenes"]["T5-G2-s2"] == 0
    assert sum(sum(v.values()) for v in result["confusion"].values()) == 83


def test_missing_insufficient_scene_is_not_a_correct_unknown():
    _, _, decisions = build()
    del decisions["T7-G3-s2"]
    result = run(decisions)
    assert result["insufficient_groups_correct"] == 5
    assert result["insufficient_kept"] == 0


def test_technical_failure_scene_is_wrong_even_if_decision_matches():
    _, _, decisions = build()
    assert decisions["T1-G6-s1"] == "drop"  # key no
    result = run(decisions, failures={"T1-G6-s1"})
    assert result["scenes"]["T1-G6-s1"] == 0
    assert result["groups"]["T1-G6"] == 0
    assert result["technical_failures"] == 1
    assert result["decidable_groups_correct"] == 35
    assert result["negatives_kept"] == 0
    assert result["confusion"]["no"]["drop"] == 36


def test_only_candidate_c1_counts():
    scenes, key, decisions = build()
    pred = rows(decisions)
    pred.append({"event_id": "T1-G1-s2", "cand_id": "c2", "mode": "cls", "model": "m",
                 "decision": "keep", "p_yes": None, "fields": {}, "technical_failure": True})
    result = score.score(scenes, key, pred)
    assert result["passed"] is True and result["technical_failures"] == 0


def test_cli_writes_numbers_only(tmp_path, capsys):
    scenes, key, decisions = build()
    decisions["T1-G1-s1"] = "drop"
    paths = {name: tmp_path / name for name in ("scenes.jsonl", "key.json", "pred.jsonl",
                                                 "out.json")}
    scene_rows = [{**s, "context": ["她：私密文本不应出现"]} for s in scenes.values()]
    paths["scenes.jsonl"].write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scene_rows), encoding="utf-8")
    paths["key.json"].write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
    paths["pred.jsonl"].write_text(
        "".join(json.dumps(r) + "\n" for r in rows(decisions)), encoding="utf-8")
    assert score.main(["--scenes", str(paths["scenes.jsonl"]), "--key", str(paths["key.json"]),
                       "--pred", str(paths["pred.jsonl"]), "--out", str(paths["out.json"])]) == 0
    text = paths["out.json"].read_text(encoding="utf-8")
    assert "私密" not in text and "合成" not in text
    result = json.loads(text)
    assert result["decidable_groups_correct"] == 35 and result["passed"] is True

    def leaves(value):
        if isinstance(value, dict):
            for v in value.values():
                yield from leaves(v)
        else:
            yield value

    assert all(isinstance(v, (int, float, bool)) for v in leaves(result))
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert json.loads(out[0])["passed"] is True


def test_duplicate_c1_prediction_rejected():
    scenes, key, decisions = build()
    pred = rows(decisions)
    pred.append(dict(pred[0]))
    with pytest.raises(ValueError, match="duplicate"):
        score.score(scenes, key, pred)
