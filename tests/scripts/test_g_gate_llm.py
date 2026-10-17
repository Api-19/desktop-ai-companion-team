# ruff: noqa: RUF001
"""LLM gate contract tests with an injected fake backend; mlx is never imported."""

import hashlib
import json
import sys

import pytest

from scripts import g_gate_llm as llm


class FakeBackend:
    name = "fake/model@00000000"

    def __init__(self, generate=(), probs=None):
        self.outputs = list(generate)
        self.probs = list(probs or [])
        self.calls = []

    def generate(self, messages, max_tokens):
        self.calls.append(("generate", messages, max_tokens))
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def choice_probs(self, messages, options):
        self.calls.append(("choice", messages, options))
        item = self.probs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def event(n=1, permission=None, **extra):
    item = {
        "event_id": "e1", "context": ["她：合成话"], "permission": permission,
        "candidates": [{"cand_id": f"c{i}", "source": "similar", "c": ["她：合成旧话"],
                        "r": ["合成回复一", "合成回复二"]} for i in range(n)],
    }
    item.update(extra)
    return item


def stage1(*foci):
    return json.dumps({"foci": [{"id": f"f{i}", "focus": "焦点", "actions": acts}
                                for i, acts in enumerate(foci, 1)]}, ensure_ascii=False)


def stage2(a, content, permission=None, t="说明"):
    value = {"a": a, "t": t, "content": content}
    if permission is not None:
        value["permission"] = permission
    return json.dumps(value, ensure_ascii=False)


ROW_KEYS = {"event_id", "cand_id", "mode", "model", "decision", "p_yes", "fields",
            "technical_failure"}


def test_mlx_is_not_imported():
    assert "mlx" not in sys.modules and "mlx_lm" not in sys.modules


def test_direct_rows_threshold_and_unparseable_unknown():
    backend = FakeBackend(probs=[("是", {"是": 0.5, "否": 0.5}), ("否", {"是": 0.2, "否": 0.8}),
                                 (None, {"是": 0.9, "否": 0.1})])
    rows = llm.judge_event(event(3), "direct", backend)
    assert [set(r) for r in rows] == [ROW_KEYS] * 3
    assert [r["decision"] for r in rows] == ["keep", "drop", "unknown"]
    assert [r["p_yes"] for r in rows] == [0.5, 0.2, 0.9]
    assert all(r["mode"] == "direct" and r["model"] == backend.name for r in rows)
    assert not any(r["technical_failure"] for r in rows)
    assert all(call[2] == ("是", "否") for call in backend.calls)


def test_content_only_mapping_and_token_cap():
    backend = FakeBackend(generate=["兼容", " 冲突\n", "未知"])
    rows = llm.judge_event(event(3), "content_only", backend)
    assert [r["decision"] for r in rows] == ["keep", "drop", "unknown"]
    assert [r["fields"]["content"] for r in rows] == ["compatible", "conflict", "unknown"]
    assert all(r["p_yes"] is None and set(r) == ROW_KEYS for r in rows)
    assert all(call[2] <= 4 for call in backend.calls)


def test_content_only_unparseable_is_technical_failure():
    rows = llm.judge_event(event(), "content_only", FakeBackend(generate=["大概兼容吧"]))
    assert rows[0]["decision"] == "drop" and rows[0]["technical_failure"] is True


def test_ga_row_format_stage1_once_and_token_caps():
    backend = FakeBackend(generate=[stage1(["A", "B"]), stage2("A", {"f1": "compatible"}),
                                    stage2("B", {"f1": "compatible"})])
    rows = llm.judge_event(event(2), "ga", backend)
    assert [r["decision"] for r in rows] == ["keep", "keep"]
    assert all(set(r) == ROW_KEYS and r["p_yes"] is None for r in rows)
    assert rows[0]["fields"]["a_m"] == "A" and rows[0]["fields"]["t_m"] == "说明"
    assert rows[0]["fields"]["foci"][0]["actions"] == ["A", "B"]
    assert [c[2] for c in backend.calls] == [256, 192, 192]
    stage2_user = backend.calls[1][1][1]["content"]
    assert "【焦点】" in stage2_user and "合成回复二" in stage2_user
    assert "合成回复" not in backend.calls[0][1][1]["content"]  # stage 1 never sees candidates


@pytest.mark.parametrize(("foci", "judged", "expected"), [
    ([["B"]], ("A", {"f1": "compatible"}), "drop"),                      # a_m not in A(f)
    ([["A"]], ("A", {"f1": "conflict"}), "drop"),                        # content conflict
    ([["A"]], ("A", {"f1": "unknown"}), "unknown"),                      # content unknown
    ([["A"]], ("unknown", {"f1": "compatible"}), "unknown"),             # action unknown
    ([["unknown"]], ("A", {"f1": "compatible"}), "unknown"),             # stage1 unknown
    ([["A"], ["D"]], ("D", {"f1": "conflict", "f2": "compatible"}), "keep"),  # any focus
    ([["A"], ["D"]], ("A", {"f1": "conflict", "f2": "compatible"}), "drop"),  # pairing kept
])
def test_ga_combination_rules(foci, judged, expected):
    backend = FakeBackend(generate=[stage1(*foci), stage2(judged[0], judged[1])])
    assert llm.judge_event(event(), "ga", backend)[0]["decision"] == expected


@pytest.mark.parametrize(("permission", "expected"), [
    ("ok", "keep"), ("violation", "drop"), ("unknown", "unknown"),
])
def test_ga_permission_veto(permission, expected):
    perm = {"nickname_ok": False, "note": "合成许可"}
    backend = FakeBackend(generate=[stage1(["A"]), stage2("A", {"f1": "compatible"}, permission)])
    row = llm.judge_event(event(permission=perm), "ga", backend)[0]
    assert row["decision"] == expected and row["fields"]["permission"] == permission
    assert "合成许可" in backend.calls[0][1][1]["content"]


def test_ga_permission_key_required_only_when_permission_present():
    missing = FakeBackend(generate=[stage1(["A"]), stage2("A", {"f1": "compatible"})])
    row = llm.judge_event(event(permission={"note": "x"}), "ga", missing)[0]
    assert row["technical_failure"] is True
    extra = FakeBackend(generate=[stage1(["A"]), stage2("A", {"f1": "compatible"}, "ok")])
    assert llm.judge_event(event(), "ga", extra)[0]["technical_failure"] is True


@pytest.mark.parametrize("bad_stage1", [
    "```json\n" + stage1(["A"]) + "\n```",          # strict JSON only
    json.dumps({"foci": []}),
    json.dumps({"foci": [{"id": "f1", "focus": "x", "actions": ["Z"]}]}),
    json.dumps({"foci": [{"id": "f1", "focus": "x", "actions": ["A", "unknown"]}]}),
    RuntimeError("generation failed"),
])
def test_ga_stage1_failure_drops_every_candidate(bad_stage1):
    backend = FakeBackend(generate=[bad_stage1])
    rows = llm.judge_event(event(2), "ga", backend)
    assert [(r["decision"], r["technical_failure"]) for r in rows] == [("drop", True)] * 2
    assert len(backend.calls) == 1  # no retry, no stage 2


@pytest.mark.parametrize("bad_stage2", [
    stage2("A", {"f2": "compatible"}),               # wrong focus ids
    stage2("A", {"f1": "maybe"}),
    stage2("Q", {"f1": "compatible"}),
    "not json",
    RuntimeError("generation failed"),
])
def test_ga_stage2_failure_is_technical_and_other_candidates_continue(bad_stage2):
    backend = FakeBackend(generate=[stage1(["A"]), bad_stage2, stage2("A", {"f1": "compatible"})])
    rows = llm.judge_event(event(2), "ga", backend)
    assert (rows[0]["decision"], rows[0]["technical_failure"]) == ("drop", True)
    assert (rows[1]["decision"], rows[1]["technical_failure"]) == ("keep", False)


def test_direct_backend_error_is_technical_failure():
    rows = llm.judge_event(event(), "direct", FakeBackend(probs=[RuntimeError("x")]))
    assert rows[0] | {"fields": {}} == {
        "event_id": "e1", "cand_id": "c0", "mode": "direct", "model": FakeBackend.name,
        "decision": "drop", "p_yes": None, "fields": {}, "technical_failure": True,
    }


def test_labels_four_merges_a_and_b():
    prompt = llm.stage1_system("four")
    assert "接话/追问" in prompt and "B：" not in prompt
    assert "B：追问关心" in llm.stage1_system("five")
    four = FakeBackend(generate=[stage1(["A"]), stage2("A", {"f1": "compatible"})])
    assert llm.judge_event(event(), "ga", four, labels="four")[0]["decision"] == "keep"
    rejects_b = FakeBackend(generate=[stage1(["B"])])
    assert llm.judge_event(event(), "ga", rejects_b, labels="four")[0]["technical_failure"]
    b_action = FakeBackend(generate=[stage1(["A"]), stage2("B", {"f1": "compatible"})])
    assert llm.judge_event(event(), "ga", b_action, labels="four")[0]["technical_failure"]


def test_load_events_accepts_practice_fixture_and_optional_permission(tmp_path):
    events = llm.load_events(llm.Path("fixtures/synthetic/g_gate_practice_v1.jsonl"))
    assert len(events) == 24 and any(e["permission"] for e in events)
    path = tmp_path / "x.jsonl"
    item = event()
    del item["permission"]
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    assert llm.load_events(path)[0]["event_id"] == "e1"
    path.write_text(json.dumps(event(permission="yes")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        llm.load_events(path)


def test_summarize_group_and_counts():
    events = [event(event_id="g-s1", group="G", labels={"c0": {"app": "yes"}}),
              event(event_id="g-s2", group="G", labels={"c0": {"app": "no"}})]
    rows = [{"event_id": "g-s1", "cand_id": "c0", "decision": "keep", "technical_failure": False},
            {"event_id": "g-s2", "cand_id": "c0", "decision": "unknown",
             "technical_failure": False}]
    summary = llm.summarize(events, rows)
    assert (summary["groups_correct"], summary["pos_kept"], summary["neg_kept"],
            summary["unknown"]) == (1, 1, 0, 1)
    rows[1]["decision"] = "keep"
    assert llm.summarize(events, rows)["groups_correct"] == 0


def test_main_load_failure_marks_all_technical(tmp_path, monkeypatch, capsys):
    def boom(path):
        raise RuntimeError("no model")

    monkeypatch.setattr(llm, "MlxBackend", boom)
    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps(event(2), ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    assert llm.main(["--input", str(source), "--model", "/nope", "--mode", "ga",
                     "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["decision"], r["technical_failure"]) for r in rows] == [("drop", True)] * 2
    assert json.loads(capsys.readouterr().out)["technical_failures"] == 2


FROZEN_PROMPT_SHA256 = {  # M15-G2a v5, frozen on the practice set; change requires a new contract
    "direct": "da2056c3f9e4be4a767ef064e1e37a43915ffee2a7873b46b43c445e064b0926",
    "content_only": "a12f69f983ad40e75f39e5bfb1eabf99ab760c3933a6a4a1d336099ec3b5ad10",
    "stage1_five": "4524438abf37a602716014f50c6cdb0c77e103d4cba7bf51aecbc7b5f405a2b3",
    "stage1_four": "9520ec586f29b73492ca243f50c1bd6f3656d87f1da986585ce03b865063ad71",
    "stage2_five_f1": "21c592aad611b4529497f12b59b948993db068a7735e1aa7afebd14ffc96e8c2",
    "stage2_five_f1_perm": "df147104f7758bdd0a59819339a289103c56b392909a06e56d82f261872040e8",
    "stage2_four_f1": "2a75fd14f58fceb7f7f1c178102863140ab71f4fc42196cb0f18505dbd6f7c72",
}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_frozen_system_prompts_unchanged():
    rendered = {
        "direct": llm.DIRECT_SYSTEM, "content_only": llm.CONTENT_SYSTEM,
        "stage1_five": llm.stage1_system("five"), "stage1_four": llm.stage1_system("four"),
        "stage2_five_f1": llm.stage2_system("five", ["f1"], False),
        "stage2_five_f1_perm": llm.stage2_system("five", ["f1"], True),
        "stage2_four_f1": llm.stage2_system("four", ["f1"], False),
    }
    assert {k: _sha(v) for k, v in rendered.items()} == FROZEN_PROMPT_SHA256


def test_frozen_user_message_templates_unchanged():
    item = event(permission={"note": "合成许可"})
    candidate = item["candidates"][0]
    foci = [{"id": "f1", "focus": "焦点", "actions": ["A"]}]
    messages = {
        "direct": llm.direct_messages(item, candidate),
        "content_only": llm.content_messages(item, candidate),
        "stage1": llm.stage1_messages(item, "five"),
        "stage2": llm.stage2_messages(item, candidate, foci, "five"),
    }
    assert {k: _sha(json.dumps(v, ensure_ascii=False)) for k, v in messages.items()} == {
        "direct": "5976c5604388568e43299f29b036da23a0c6790cf20802c810ff85dba8b6d11b",
        "content_only": "1825a3362dcc5a0a60f0009c69cf330eddcdfa3b360f833932dedfd0c7d0efbe",
        "stage1": "55aa2c1d48a0cc9d43a1ed44528172f086c05ac8f0bae8e5587a3307efc7328b",
        "stage2": "4beaa76e4d03f81e2069bad43ea3ee403e369a95a82129316232d55c3063a99d",
    }


# ---- ga2: offline candidate card + premise check + sufficiency (M15-G3a) ----

def card(actions=("A", "A"), premises=("state",), nickname=False, tease=False):
    return json.dumps({
        "msgs": [{"i": i, "action": a, "evidence": "证据"} for i, a in enumerate(actions, 1)],
        "persons": ["她"],
        "premises": [{"id": f"p{i}", "text": "抽象前提", "kind": k}
                     for i, k in enumerate(premises, 1)],
        "needs_permission": {"nickname": nickname, "tease": tease},
    }, ensure_ascii=False)


def judged2(premises, permission=None, key_missing=False):
    value = {"premises": premises, "key_missing": key_missing}
    if permission is not None:
        value["permission"] = permission
    return json.dumps(value, ensure_ascii=False)


def run_ga2(outputs, tmp_path, item=None):
    backend = FakeBackend(generate=outputs)
    cache = llm.CardCache(tmp_path / "cards")
    rows = llm.judge_event(item or event(), "ga2", backend, card_cache=cache)
    return rows, backend, cache


def test_ga2_in_modes_and_row_fields(tmp_path):
    assert "ga2" in llm.MODES
    rows, backend, _ = run_ga2([stage1(["A"]), card(), judged2({"p1": "supported"})], tmp_path)
    row = rows[0]
    assert set(row) == ROW_KEYS and row["mode"] == "ga2" and row["decision"] == "keep"
    assert {"card_key", "card", "foci", "premises", "permission", "key_missing"} <= set(
        row["fields"])
    assert row["fields"]["permission"] is None and row["technical_failure"] is False
    assert [c[2] for c in backend.calls] == [256, llm.MAX_TOKENS["card"], llm.MAX_TOKENS["judge2"]]
    card_user = backend.calls[1][1][1]["content"]
    assert "合成旧话" in card_user and "合成回复二" in card_user
    assert "合成话" not in card_user.replace("合成旧话", "")  # card never sees the current scene
    judge_user = backend.calls[2][1][1]["content"]
    assert "合成话" in judge_user and "p1" in judge_user and "【焦点】" in judge_user


@pytest.mark.parametrize(("foci", "card_kw", "judged", "expected"), [
    # a. permission violation drops even when everything else is fine
    ([["A"]], {}, ({"p1": "supported"}, "violation", True), "drop"),
    # b. any contradicted premise drops, even with key_missing
    ([["A"]], {"premises": ("state", "need")},
     ({"p1": "supported", "p2": "contradicted"}, "ok", True), "drop"),
    # c. any message action not accepted by some focus drops (first ok, second not)
    ([["A"]], {"actions": ("A", "B")}, ({"p1": "supported"}, "ok", False), "drop"),
    ([["unknown"]], {"actions": ("A",)}, ({"p1": "supported"}, "ok", False), "drop"),
    ([["A"], ["B"]], {"actions": ("A", "B")}, ({"p1": "supported"}, "ok", False), "keep"),
    # d. all supported and permission ok -> keep
    ([["A"]], {}, ({"p1": "supported"}, "ok", False), "keep"),
    # e. not_given / permission unknown -> unknown iff key_missing, else drop
    ([["A"]], {}, ({"p1": "not_given"}, "ok", True), "unknown"),
    ([["A"]], {}, ({"p1": "not_given"}, "ok", False), "drop"),
    ([["A"]], {}, ({"p1": "supported"}, "unknown", True), "unknown"),
    ([["A"]], {}, ({"p1": "supported"}, "unknown", False), "drop"),
])
def test_combine2_branches(foci, card_kw, judged, expected):
    focus_list = [{"id": f"f{i}", "focus": "x", "actions": a} for i, a in enumerate(foci, 1)]
    card_value = llm.parse_card(card(**card_kw), {"r": ["x"] * len(card_kw.get("actions", "AA"))},
                                "five")
    result = {"premises": judged[0], "permission": judged[1], "key_missing": judged[2]}
    assert llm.combine2(focus_list, card_value, result) == expected


def test_combine2_without_scene_permission():
    foci = [{"id": "f1", "focus": "x", "actions": ["A"]}]
    card_value = llm.parse_card(card(), {"r": ["x", "y"]}, "five")
    assert llm.combine2(foci, card_value, {"premises": {"p1": "supported"}, "permission": None,
                                           "key_missing": False}) == "keep"


def test_ga2_card_cache_hit_skips_generation(tmp_path):
    item = event(2)  # two candidates with identical c/r share one card
    rows, backend, cache = run_ga2(
        [stage1(["A"]), card(), judged2({"p1": "supported"}), judged2({"p1": "supported"})],
        tmp_path, item)
    assert [r["decision"] for r in rows] == ["keep", "keep"]
    assert len(backend.calls) == 4  # stage1 + one card + two judge2
    assert (cache.hits, cache.misses) == (1, 1)
    assert rows[0]["fields"]["card_key"] == rows[1]["fields"]["card_key"]
    stored = tmp_path / "cards" / (rows[0]["fields"]["card_key"] + ".json")
    assert json.loads(stored.read_text(encoding="utf-8")) == rows[0]["fields"]["card"]
    again = FakeBackend(generate=[stage1(["A"]), judged2({"p1": "supported"})])
    fresh = llm.CardCache(tmp_path / "cards")
    rows2 = llm.judge_event(event(1), "ga2", again, card_cache=fresh)
    assert rows2[0]["decision"] == "keep" and (fresh.hits, fresh.misses) == (1, 0)
    assert len(again.calls) == 2


def test_card_key_depends_on_text_version_and_model():
    candidate = {"c": ["她：甲"], "r": ["乙", "丙"]}
    base = llm.card_key(candidate, "five", "m1")
    assert base == llm.card_key(dict(candidate), "five", "m1")
    assert base != llm.card_key({"c": ["她：甲"], "r": ["乙丙"]}, "five", "m1")
    assert base != llm.card_key(candidate, "five", "m2")
    assert base != llm.card_key(candidate, "four", "m1")


@pytest.mark.parametrize(("outputs", "stage"), [
    ([RuntimeError("x")], "stage1"),
    ([stage1(["A"]), RuntimeError("x")], "card"),
    ([stage1(["A"]), card(actions=("A",))], "card"),                 # msgs != len(r)
    ([stage1(["A"]), card(premises=())], "card"),                    # 1-5 premises
    ([stage1(["A"]), card(actions=("A", "Q"))], "card"),
    ([stage1(["A"]), card(), "not json"], "judge2"),
    ([stage1(["A"]), card(), judged2({"p2": "supported"})], "judge2"),  # must cover card
    ([stage1(["A"]), card(), judged2({"p1": "maybe"})], "judge2"),
    ([stage1(["A"]), card(), judged2({"p1": "supported"}, "ok")], "judge2"),  # no scene perm
])
def test_ga2_any_stage_failure_is_technical(outputs, stage, tmp_path):
    rows, _, _ = run_ga2(outputs, tmp_path)
    assert (rows[0]["decision"], rows[0]["technical_failure"]) == ("drop", True)
    assert rows[0]["fields"]["stage"] == stage
    assert not list((tmp_path / "cards").glob("*.json")) or stage == "judge2"


def test_ga2_permission_key_required_only_with_scene_permission(tmp_path):
    perm = {"nickname_ok": False, "note": "合成许可"}
    missing, _, _ = run_ga2([stage1(["A"]), card(), judged2({"p1": "supported"})],
                            tmp_path / "a", event(permission=perm))
    assert missing[0]["technical_failure"] is True
    present, judged_backend, _ = run_ga2(
        [stage1(["A"]), card(), judged2({"p1": "supported"}, "violation")],
        tmp_path / "b", event(permission=perm))
    assert present[0]["decision"] == "drop" and present[0]["fields"]["permission"] == "violation"
    assert "合成许可" in judged_backend.calls[2][1][1]["content"]
    assert "合成许可" not in judged_backend.calls[1][1][1]["content"]


def test_main_ga2_requires_card_cache(tmp_path):
    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps(event(), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        llm.main(["--input", str(source), "--model", "/nope", "--mode", "ga2",
                  "--out", str(tmp_path / "o.jsonl")])


def test_summarize_unknown_label_requires_unknown_decision():
    events = [event(event_id="u-s1", group="U", labels={"c0": {"app": "yes"}}),
              event(event_id="u-s2", group="U", labels={"c0": {"app": "unknown"}})]
    rows = [{"event_id": "u-s1", "cand_id": "c0", "decision": "keep", "technical_failure": False},
            {"event_id": "u-s2", "cand_id": "c0", "decision": "drop", "technical_failure": False}]
    assert llm.summarize(events, rows)["groups_correct"] == 0
    rows[1]["decision"] = "unknown"
    summary = llm.summarize(events, rows)
    assert summary["groups_correct"] == 1
    assert (summary["unk_total"], summary["unk_correct"], summary["unk_kept"]) == (1, 1, 0)


def test_load_events_accepts_practice_v2():
    events = llm.load_events(llm.Path("fixtures/synthetic/g_gate_practice_v2.jsonl"))
    groups = {e["group"] for e in events}
    assert len(events) >= 12 and len(groups) >= 6
    kinds = defaultdict_count(e["group_kind"] for e in events if e["event_id"].endswith("-s1"))
    assert kinds.get("T7", 0) >= 2
    assert any(lab["app"] == "unknown" for e in events for lab in e["labels"].values())


def defaultdict_count(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


FROZEN_GA2_SHA256 = {  # M15-G3a repair r1-2 (card v13), frozen on practice v1+v2
    "stage1_ga2_five": "746d2782a9a82a08d5b33b4c152ab8bfb69db1a3653a3b1848503cdcc0282136",
    "stage1_ga2_four": "551582ca4f34ad48a7839855fe4f135b000bf35a41067b45aab6eaf0f3d6b12e",
    "card_five": "78d316354fd3406aef4c4070649a0a91c8a297bf186995e69e429d29e1e0c641",
    "card_four": "a32195c8a00a10ee8adc5123b20803b7350fd8c56f1292250a2839938c812eaa",
    "judge2_p1": "4bfa96f5118d8951495ef98415cc761ba0bcabe848caa075cf46fdd790e9dca3",
    "judge2_p1_perm": "eaa6a54ce93c7dc883162eaafc9f9d48e3930b71bcbaa61947a7e72a90477717",
}


def test_frozen_ga2_system_prompts_unchanged():
    assert llm.CARD_PROMPT_VERSION == "ga2-card-v13"
    rendered = {
        "stage1_ga2_five": llm.stage1_ga2_system("five"),
        "stage1_ga2_four": llm.stage1_ga2_system("four"),
        "card_five": llm.card_system("five"), "card_four": llm.card_system("four"),
        "judge2_p1": llm.judge2_system(["p1"], False),
        "judge2_p1_perm": llm.judge2_system(["p1"], True),
    }
    assert {k: _sha(v) for k, v in rendered.items()} == FROZEN_GA2_SHA256


def test_frozen_ga2_user_message_templates_and_card_key_unchanged():
    item = event(permission={"note": "合成许可"})
    candidate = item["candidates"][0]
    card_value = {"msgs": [], "persons": ["她"],
                  "premises": [{"id": "p1", "text": "抽象前提", "kind": "state"}],
                  "needs_permission": {"nickname": False, "tease": False}}
    foci = [{"id": "f1", "focus": "焦点", "actions": ["A"]}]
    messages = {
        "stage1_ga2": llm.stage1_ga2_messages(item, "five"),
        "card": llm.card_messages(candidate, "five"),
        "judge2": llm.judge2_messages(item, candidate, card_value, foci),
    }
    assert {k: _sha(json.dumps(v, ensure_ascii=False)) for k, v in messages.items()} == {
        "stage1_ga2": "6a941d104108edaa4f5de28e154b77c258317adf38da97f006ec3b9856032d34",
        "card": "936881996094802816b2120475000d4e81d73ef463bc0f46c863718d3afefe30",
        "judge2": "c23301a66193fffd119e981a1815790b94efb90b0ff1b2f951f9d67593c4dbdc",
    }
    assert llm.card_key(candidate, "five", FakeBackend.name) == (
        "26d3475b95b81db5ff2d7c9e74fa85ce25b8702523b4cb354b905435b2b51900")


def test_ga2_uses_own_stage1_prompt_and_ga_keeps_frozen_one(tmp_path):
    _, backend, _ = run_ga2([stage1(["A"]), card(), judged2({"p1": "supported"})], tmp_path)
    assert backend.calls[0][1] == llm.stage1_ga2_messages(event(), "five")
    assert backend.calls[0][1][0]["content"] == llm.stage1_ga2_system("five")
    assert llm.stage1_ga2_system("five") != llm.stage1_system("five")
    assert all(f"{code}：" in llm.stage1_ga2_system("five") for code in "ABCDE")
    ga = FakeBackend(generate=[stage1(["A"]), stage2("A", {"f1": "compatible"})])
    llm.judge_event(event(), "ga", ga)
    assert ga.calls[0][1][0]["content"] == llm.stage1_system("five")
