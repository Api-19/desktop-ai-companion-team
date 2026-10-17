# ruff: noqa: RUF001
"""Contract tests for scripts/d14_pair_eval.py (synthetic data and fake provider only)."""

from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.contracts import DialogueRequest
from companion.dialogue.context import MAX_CURRENT_CHARS, prepare_dialogue_input
from companion.dialogue.models import ProviderResult, ProviderUsage
from companion.dialogue.observer import TurnObservation
from companion.dialogue.provider import ProviderTimeout
from companion.privacy.paired_snippets import PairedSnippet, write_snippets
from scripts import d14_pair_eval as d14
from scripts import style_reframe_eval as ev

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/d14_pair_eval.py"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
GROUPS = ("base", "variant")
BASE_CARD = "基线卡：短句为主\n"
VARIANT_CARD = "变体卡：先接话再展开\n"


def prompt_observation(prompt_hash):
    return TurnObservation(
        blocked_reason=None, input_chars=4, context_count=0, context_chars=0,
        reply_chars=4, similar_count=0, general_count=0, profile_count=0,
        provider_attempted=prompt_hash is not None,
        system_prompt_sha256=prompt_hash, observed_at=T0,
    )


def test_prompt_hash_none_when_no_provider_call():
    group_info = {"system_prompt_sha256": None}
    observer = d14.first_prompt_observer(group_info)

    observer(prompt_observation(None))

    assert group_info["system_prompt_sha256"] is None


def test_prompt_hash_records_first_provider_call_only():
    group_info = {"system_prompt_sha256": None}
    observer = d14.first_prompt_observer(group_info)
    first_hash = sha(b"first actual provider system prompt")

    observer(prompt_observation(None))
    observer(prompt_observation(first_hash))
    observer(prompt_observation(sha(b"later provider system prompt")))
    observer(prompt_observation(None))

    assert group_info["system_prompt_sha256"] == first_hash


def snip(sid, partner, reply, context=()):
    return PairedSnippet(
        snippet_id=sid, conversation_id="synthetic", started_at=T0,
        partner_turns=list(partner), owner_reply=list(reply),
        source_fingerprint="synthetic", context_turns=list(context),
    )


E1 = snip("e1", ["我今天下班路上看到一只很胖的橘猫"], ["好 的", "知道了"],
          ["他：晚上吃什么呢<还没想好>&", "她：哈哈哈"])
E2 = snip("e2", ["周末要不要一起去爬山"], ["可以啊早点出发"],
          ["她：今天下雨了好冷啊", "她：哈哈哈"])
E4 = snip("e4", ["好困"], ["那快睡"])
LIBRARY = [
    E2,
    snip("e1b", ["随便聊聊天气"], ["嗯嗯"]),
    snip("e3", ["晚饭吃了吗宝"], ["吃了吃了"]),
    # Occupied takes precedence over the same-reply rule.
    snip("e3b", ["你在干什么呀"], ["好的知道了"]),
    snip("u1", ["明天几点的高铁"], ["八点半的那班"]),
    snip("dup_reply", ["那我先去洗澡了"], ["好的", "知道了"]),
    snip("ctx_overlap", ["我们周六见吧"], ["行没问题"], ["她：今天下雨了 好冷啊"]),
    snip("near_dup", ["我今天下班路上看到一只很胖的橘猫呀"], ["真的假的"]),
    snip("dup_short", ["好困"], ["我也是"]),
    # Lines shorter than five normalized characters never count as overlap.
    snip("keep1", ["今天加班到很晚好累"], ["辛苦啦早点休息"], ["她：哈哈哈"]),
    # Kept in the filtered library but rejected as style evidence (4 partner turns).
    snip("long4", ["一", "二", "三", "四个字呢"], ["好"]),
]
KEPT_IDS = ["u1", "keep1", "long4"]
POOLS = {"version": 1, "seed": 3,
         "pools": {"screen_shape": ["e2", "e1", "e4"], "combo": ["e3"], "unassigned": ["u1"]}}
SPLIT = {"events": [
    {"event_id": "e1", "pool": "test", "snippet_ids": ["e1", "e1b"],
     "predict_snippet_id": "e1", "stage": "s1"},
    {"event_id": "e2", "pool": "train", "snippet_ids": ["e2"],
     "predict_snippet_id": "e2", "stage": "s1"},
    {"event_id": "e4", "pool": "train", "snippet_ids": ["e4"],
     "predict_snippet_id": "e4", "stage": "s2"},
    {"event_id": "e3", "pool": "calib", "snippet_ids": ["e3", "e3b"],
     "predict_snippet_id": "e3", "stage": "s2"},
    {"event_id": "u1", "pool": "train", "snippet_ids": ["u1"],
     "predict_snippet_id": "u1", "stage": "s3"},
]}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_data(tmp_path, *, pools=None, split=None, holdout=None, library=None):
    write_snippets(tmp_path / "holdout.jsonl", [E1, E4] if holdout is None else holdout)
    write_snippets(tmp_path / "library.jsonl", LIBRARY if library is None else library)
    (tmp_path / "pools.json").write_text(json.dumps(POOLS if pools is None else pools),
                                         encoding="utf-8")
    (tmp_path / "split.json").write_text(json.dumps(SPLIT if split is None else split),
                                         encoding="utf-8")
    (tmp_path / "base.txt").write_text(BASE_CARD, encoding="utf-8")
    (tmp_path / "variant.txt").write_text(VARIANT_CARD, encoding="utf-8")
    return [
        "run",
        "--pools", str(tmp_path / "pools.json"), "--pool", "screen_shape",
        "--split", str(tmp_path / "split.json"),
        "--holdout", str(tmp_path / "holdout.jsonl"),
        "--library", str(tmp_path / "library.jsonl"),
        "--base-card", str(tmp_path / "base.txt"), "--base-strata", "length",
        "--variant-card", str(tmp_path / "variant.txt"), "--variant-strata", "lines",
        "--partner-user-id", "synthetic-partner",
        "--out-dir", str(tmp_path / "out"),
        "--max-requests", "6", "--seed", "7",
    ]


def set_arg(args, option, value):
    args[args.index(option) + 1] = value
    return args


def record_fake_provider(monkeypatch, *, fail=None):
    """Patch the shared fake provider; instances are named by first-call order."""
    calls, instances = [], []

    def complete(self, messages):
        if self not in instances:
            instances.append(self)
        group = GROUPS[instances.index(self)]
        calls.append((group, messages[0].content, messages[1].content))
        if fail is not None and fail(group, messages[1].content):
            raise ProviderTimeout("synthetic timeout")
        label = "基线" if group == "base" else "变体"
        return ProviderResult(
            text=f"{label}回复{messages[1].content}",
            usage=ProviderUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(ev.FakeProvider, "complete", complete)
    return calls


def capture_adapters(monkeypatch):
    captured = []
    original = d14.SnippetStyleAdapter

    def adapter(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(d14, "SnippetStyleAdapter", adapter)
    return captured


def forbid_assembly(monkeypatch):
    for name in ("build_fake_orchestrator", "build_real_orchestrator", "SnippetStyleAdapter",
                 "load_dialogue_config"):
        monkeypatch.setattr(d14, name, lambda *a, **kw: pytest.fail("assembly"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def expected_prepared(snippet):
    item = ev.items_from_snippets([snippet])[0]
    prepared = prepare_dialogue_input(DialogueRequest(
        user_id="d14-blind", text=item.text, recent_context=item.recent_context))
    return {"item_id": snippet.snippet_id, "text": prepared.text,
            "recent_context": list(prepared.recent_context)}


def expected_orders(seed, count):
    rng = random.Random(seed)
    orders = []
    for _ in range(count):
        order = list(GROUPS)
        rng.shuffle(order)
        orders.append(order)
    return orders


def as_html(text):
    return html.escape(text).replace("\n", "<br>")


def section(page, number):
    match = re.search(rf'<section class="item" data-number="{number}">(.*?)</section>',
                      page, re.DOTALL)
    assert match is not None
    return match.group(1)


# --- run: library isolation -------------------------------------------------------------


def test_library_isolation_counts_digest_and_retrieval_input(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)
    record_fake_provider(monkeypatch)
    adapters = capture_adapters(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "out/manifest.json").read_text(encoding="utf-8"))
    assert manifest["removed"] == {
        "occupied": 4, "same_reply": 1, "context_overlap": 1, "near_duplicate": 2,
    }
    assert (manifest["library_before"], manifest["library_after"]) == (11, 3)
    assert manifest["style_evidence_excluded"] == 1
    kept = [s for s in LIBRARY if s.snippet_id in KEPT_IDS]
    assert [s.snippet_id for s in kept] == KEPT_IDS
    digest = sha("".join(
        json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n" for s in kept
    ).encode("utf-8"))
    assert manifest["library_sha256"] == digest
    expected_style = [s.as_style_snippet() for s in kept[:2]]
    assert [a["snippets"] for a in adapters] == [expected_style, expected_style]
    assert [a["general_strata"] for a in adapters] == ["length", "lines"]
    assert {a["partner_user_id"] for a in adapters} == {"synthetic-partner"}
    assert {a["style_version"] for a in adapters} == {"d14-pair-eval-v1"}


def test_occupied_uses_every_non_unassigned_pool_and_split_snippets(tmp_path, monkeypatch,
                                                                    capsys):
    pools = json.loads(json.dumps(POOLS))
    pools["pools"]["combo"] = []
    pools["pools"]["unassigned"] = ["u1", "e3"]
    args = make_data(tmp_path, pools=pools)
    record_fake_provider(monkeypatch)
    adapters = capture_adapters(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "out/manifest.json").read_text(encoding="utf-8"))
    # e3 sits in "unassigned" and stays; e3b's reply equals e1's normalized reply.
    assert manifest["removed"] == {
        "occupied": 2, "same_reply": 2, "context_overlap": 1, "near_duplicate": 2,
    }
    assert manifest["library_after"] == 4
    style_ids = {"e3", "u1", "keep1"}
    assert adapters[0]["snippets"] == [
        s.as_style_snippet() for s in LIBRARY if s.snippet_id in style_ids
    ]


# --- run: outputs ------------------------------------------------------------------------


def test_fake_run_outputs_items_replies_key_and_status(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)
    calls = record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "complete", "mode": "fake", "planned_requests": 6,
        "completed_requests": 6, "spent_cny": "0.00",
    }
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "answer_key.json", "blind.html", "items.jsonl", "manifest.json",
        "replies_base.jsonl", "replies_variant.jsonl",
    ]
    assert read_jsonl(out / "items.jsonl") == [expected_prepared(s) for s in (E2, E1, E4)]
    texts = [s.partner_turns[-1] for s in (E2, E1, E4)]
    assert [(group, user) for group, _, user in calls] == [
        (group, text) for text in texts for group in GROUPS
    ]
    for group, label in (("base", "基线"), ("variant", "变体")):
        rows = read_jsonl(out / f"replies_{group}.jsonl")
        # The orchestrator may prepend its first-turn identity line; the reply is response.text.
        assert [(r["item_id"], r["blocked_reason"]) for r in rows] == [
            ("e2", None), ("e1", None), ("e4", None),
        ]
        assert [set(r) for r in rows] == [{"item_id", "reply", "blocked_reason"}] * 3
        for row, text in zip(rows, texts, strict=True):
            assert row["reply"].endswith(f"{label}回复{text}")
    key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    assert key == {"kind": "pair", "seed": 7, "items": [
        {"number": n, "item_id": sid, "A": order[0], "B": order[1]}
        for n, (sid, order) in enumerate(zip(("e2", "e1", "e4"), expected_orders(7, 3),
                                             strict=True), 1)
    ]}


def test_blind_html_shows_prepared_context_and_shuffled_replies(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)
    record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    out = tmp_path / "out"
    page = (out / "blind.html").read_text(encoding="utf-8")
    key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    replies = {g: read_jsonl(out / f"replies_{g}.jsonl") for g in GROUPS}
    for index, (row, item) in enumerate(zip(key["items"], read_jsonl(out / "items.jsonl"),
                                            strict=True)):
        part = section(page, row["number"])
        assert f"<h2>第 {row['number']} 题</h2>" in part
        shown = [f"<p>{as_html(line)}</p>" for line in item["recent_context"]]
        positions = [part.index(line) for line in shown]
        assert positions == sorted(positions)
        assert f'<p class="current">当前消息：{as_html(item["text"])}</p>' in part
        a = ('<div class="reply" data-label="A"><h3>A</h3>'
             f'<p>{as_html(replies[row["A"]][index]["reply"])}</p>')
        b = ('<div class="reply" data-label="B"><h3>B</h3>'
             f'<p>{as_html(replies[row["B"]][index]["reply"])}</p>')
        assert a in part and b in part and part.index(a) < part.index(b)
        n = row["number"]
        for value, text in zip(("A", "B", "并列", "都不像", "超出语境"),
                               ("更像 A", "更像 B", "并列", "都不像", "超出语境"), strict=True):
            assert (f'<label><input type="radio" name="pref-{n}" value="{value}">{text}</label>'
                    in part)
        for value, text in zip(("A", "B", "AB", "无"), ("A", "B", "都可", "都不可"), strict=True):
            assert (f'<label><input type="radio" name="ok-{n}" value="{value}">{text}</label>'
                    in part)
        for label in "AB":
            for failure in ("接法", "措辞", "理解", "超时拦截"):
                assert (f'<label><input type="checkbox" name="fail-{n}-{label}" '
                        f'value="{failure}">{label} {failure}</label>') in part
    assert "他：晚上吃什么呢&lt;还没想好&gt;&amp;" in page
    assert '<textarea id="result" readonly></textarea>' in page
    assert '<script id="d14-format">' in page


def test_blind_html_hides_groups_ids_and_cards(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)
    record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    page = (tmp_path / "out/blind.html").read_text(encoding="utf-8")
    for hidden in ("base", "variant", "基线卡", "变体卡"):
        assert hidden not in page
    assert re.search(r"\be[124]\b", page) is None


def test_reply_newlines_render_as_breaks(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)

    def complete(self, messages):
        return ProviderResult(text="第一行\n第二行",
                              usage=ProviderUsage(prompt_tokens=1, completion_tokens=1,
                                                  total_tokens=2))

    monkeypatch.setattr(ev.FakeProvider, "complete", complete)
    assert d14.main(args) == 0
    capsys.readouterr()
    page = (tmp_path / "out/blind.html").read_text(encoding="utf-8")
    assert "<p>第一行<br>第二行</p>" in page


def test_blocked_replies_are_shown_and_counted(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)
    record_fake_provider(monkeypatch, fail=lambda group, text: group == "variant"
                         and text == E1.partner_turns[-1])
    assert d14.main(args) == 0
    capsys.readouterr()
    out = tmp_path / "out"
    variant = read_jsonl(out / "replies_variant.jsonl")
    assert [row["blocked_reason"] for row in variant] == [None, "provider_timeout", None]
    assert variant[1]["reply"]
    key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    label = "A" if key["items"][1]["A"] == "variant" else "B"
    part = section((out / "blind.html").read_text(encoding="utf-8"), 2)
    block = part[part.index(f'data-label="{label}"'):]
    assert '<p class="blocked">拦截：provider_timeout</p>' in block.split("</div>")[0]
    assert part.count('<p class="blocked">') == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["groups"]["variant"]["blocked"] == {"provider_timeout": 1}
    assert manifest["groups"]["base"]["blocked"] == {}
    assert manifest["groups"]["variant"]["completed"] == 3


def test_manifest_hashes_and_first_system_prompt(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)
    calls = record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "out/manifest.json").read_text(encoding="utf-8"))
    first = {}
    for group, system, _ in calls:
        first.setdefault(group, system)
    assert BASE_CARD.strip() in first["base"] and VARIANT_CARD.strip() in first["variant"]
    assert manifest == {
        "version": 1, "mode": "fake", "seed": 7, "pool": "screen_shape",
        "pools_sha256": sha((tmp_path / "pools.json").read_bytes()),
        "split_sha256": sha((tmp_path / "split.json").read_bytes()),
        "library_sha256": manifest["library_sha256"],
        "library_before": 11, "library_after": 3,
        "removed": manifest["removed"], "style_evidence_excluded": 1,
        "groups": {
            group: {
                "card_sha256": sha(card.encode("utf-8")),
                "general_strata": strata,
                "terms_sha256": None,
                "session_check": False,
                "system_prompt_sha256": sha(first[group].encode("utf-8")),
                "completed": 3,
                "blocked": {},
            }
            for group, card, strata in (("base", BASE_CARD, "length"),
                                        ("variant", VARIANT_CARD, "lines"))
        },
        "item_ids": ["e2", "e1", "e4"],
        "presented_item_ids": ["e2", "e1", "e4"],
        "planned_requests": 6, "completed_requests": 6, "spent_cny": "0.00",
        "stopped": None,
    }


def test_cards_are_passed_verbatim_to_orchestrators(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)
    built = []
    original = d14.build_fake_orchestrator

    def build(root, **kwargs):
        built.append(kwargs)
        return original(root, **kwargs)

    monkeypatch.setattr(d14, "build_fake_orchestrator", build)
    record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    assert [kw["snippet_habit_card"] for kw in built] == [BASE_CARD, VARIANT_CARD]
    assert all(callable(kw["turn_observer"]) for kw in built)
    assert built[0]["snippet_style"] is not built[1]["snippet_style"]


# --- run: validation before assembly ---------------------------------------------------------


@pytest.mark.parametrize("pool, pools_update", [
    ("unassigned", None), ("missing_pool", None), ("screen_shape", {"screen_shape": []}),
])
def test_invalid_pool(tmp_path, monkeypatch, capsys, pool, pools_update):
    pools = json.loads(json.dumps(POOLS))
    pools["pools"].update(pools_update or {})
    args = set_arg(make_data(tmp_path, pools=pools), "--pool", pool)
    forbid_assembly(monkeypatch)
    assert d14.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "invalid_pool"}


@pytest.mark.parametrize("pools_update", [
    {"combo": ["ghost"]},
    {"combo": ["e3", "e1"]},
    {"combo": ["e3", "e3"]},
])
def test_invalid_pools_missing_or_duplicate_events(tmp_path, monkeypatch, capsys, pools_update):
    pools = json.loads(json.dumps(POOLS))
    pools["pools"].update(pools_update)
    args = make_data(tmp_path, pools=pools)
    forbid_assembly(monkeypatch)
    assert d14.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "invalid_pools"}


def test_invalid_items_count_missing_and_over_budget(tmp_path, monkeypatch, capsys):
    long_e1 = E1.model_copy(update={"partner_turns": ["长" * (MAX_CURRENT_CHARS + 1)]})
    pools = json.loads(json.dumps(POOLS))
    pools["pools"]["screen_shape"] = ["e2", "e1", "e4", "gone"]
    split = json.loads(json.dumps(SPLIT))
    split["events"].append({"event_id": "gone", "pool": "train", "snippet_ids": ["gone"],
                            "predict_snippet_id": "gone", "stage": "s1"})
    args = make_data(tmp_path, pools=pools, split=split, holdout=[long_e1, E4])
    forbid_assembly(monkeypatch)
    assert d14.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "invalid_items", "missing": 1, "over_budget": 1,
    }


def test_holdout_wins_over_library_for_same_snippet_id(tmp_path, monkeypatch, capsys):
    library_e1 = E1.model_copy(update={"partner_turns": ["库里的另一版本消息内容"]})
    args = make_data(tmp_path, library=[*LIBRARY, library_e1])
    record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    items = read_jsonl(tmp_path / "out/items.jsonl")
    assert items[1]["text"] == E1.partner_turns[-1]


def test_fake_request_limit_before_assembly(tmp_path, monkeypatch, capsys):
    args = set_arg(make_data(tmp_path), "--max-requests", "5")
    forbid_assembly(monkeypatch)
    assert d14.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "request_limit_exceeded", "planned_requests": 6,
    }
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["empty_card", "blank_card", "max_cny_zero", "max_cny_text",
                                  "real_without_max_cny"])
def test_configuration_errors_before_assembly(tmp_path, monkeypatch, capsys, kind):
    args = make_data(tmp_path)
    if kind == "empty_card":
        (tmp_path / "variant.txt").write_text("", encoding="utf-8")
    elif kind == "blank_card":
        (tmp_path / "base.txt").write_text(" \n\t", encoding="utf-8")
    elif kind == "max_cny_zero":
        args += ["--max-cny", "0"]
    elif kind == "max_cny_text":
        args += ["--real", "--max-cny", "abc"]
    else:
        args += ["--real"]
    forbid_assembly(monkeypatch)
    monkeypatch.setattr(d14, "load_settings", lambda *a: pytest.fail("settings"))
    assert d14.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "configuration_unavailable"}


# --- run: real mode (all collaborators stubbed) ------------------------------------------------


def stub_real(tmp_path, monkeypatch, *, results_dir=None):
    state = {"calls": [], "capacities": [], "built": [], "configs": [], "ledgers": []}
    monkeypatch.setattr(d14, "load_settings", lambda root: SimpleNamespace(
        test_results_dir=tmp_path if results_dir is None else results_dir,
        runtime_data_dir=tmp_path / "runtime",
    ))

    def load_config(settings, **kwargs):
        state["configs"].append(kwargs)
        return SimpleNamespace(character_name="kiki", budget=object()), object()

    monkeypatch.setattr(d14, "load_dialogue_config", load_config)

    def build(settings, config, secrets, **kwargs):
        state["built"].append(kwargs)
        group = GROUPS[len(state["built"]) - 1]

        def reply(request, **kw):
            state["calls"].append((group, request.text, kw["trusted_user_id"]))
            return SimpleNamespace(
                response=SimpleNamespace(text=f"真实桩{group[0]}", blocked_reason=None),
                trace=SimpleNamespace(output_block_reason=None),
            )

        return SimpleNamespace(
            reply=reply, ensure_paid_capacity=lambda n: state["capacities"].append((group, n)),
        )

    monkeypatch.setattr(d14, "build_real_orchestrator", build)

    class Ledger:
        def __init__(self, root, budget):
            state["ledgers"].append(root)

        def snapshot(self, now):
            return SimpleNamespace(
                accounted_cny=Decimal("5") + Decimal("0.10") * len(state["calls"]))

    monkeypatch.setattr(d14, "MonthlyUsageLedger", Ledger)
    return state


def test_real_complete_assembly_and_spend(tmp_path, monkeypatch, capsys):
    state = stub_real(tmp_path, monkeypatch)
    adapters = capture_adapters(monkeypatch)
    args = [*set_arg(make_data(tmp_path), "--max-requests", "10"), "--real", "--max-cny", "10"]
    assert d14.main(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "complete", "mode": "real", "planned_requests": 6,
        "completed_requests": 6, "spent_cny": "0.60",
    }
    assert state["configs"] == [{"require_style_manifest": False}]
    assert [kw["trusted_user_id"] for kw in state["built"]] == ["synthetic-partner"] * 2
    assert [kw["snippet_habit_card"] for kw in state["built"]] == [BASE_CARD, VARIANT_CARD]
    assert all(callable(kw["turn_observer"]) for kw in state["built"])
    assert [a["general_strata"] for a in adapters] == ["length", "lines"]
    assert state["capacities"] == [("base", 1)]
    assert state["ledgers"] == [tmp_path / "runtime/dialogue"]
    texts = [s.partner_turns[-1] for s in (E2, E1, E4)]
    assert state["calls"] == [
        (group, text, "synthetic-partner") for text in texts for group in GROUPS
    ]
    manifest = json.loads((tmp_path / "out/manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "real"
    assert manifest["completed_requests"] == 6 and manifest["spent_cny"] == "0.60"
    assert manifest["stopped"] is None
    assert {g: manifest["groups"][g]["system_prompt_sha256"] for g in GROUPS} == {
        "base": None, "variant": None,
    }


@pytest.mark.parametrize("option, limit, reason", [
    ("--max-requests", "3", "request_cap"), ("--max-cny", "0.40", "spend_cap"),
])
def test_real_caps_save_partial_results(tmp_path, monkeypatch, capsys, option, limit, reason):
    state = stub_real(tmp_path, monkeypatch)
    args = [*set_arg(make_data(tmp_path), "--max-requests", "10"), "--real", "--max-cny", "10"]
    set_arg(args, option, limit)
    assert d14.main(args) == 3
    assert json.loads(capsys.readouterr().out) == {
        "status": "stopped", "reason": reason, "completed_requests": 3, "spent_cny": "0.30",
    }
    assert [(g, t) for g, t, _ in state["calls"]] == [
        ("base", E2.partner_turns[-1]), ("variant", E2.partner_turns[-1]),
        ("base", E1.partner_turns[-1]),
    ]
    out = tmp_path / "out"
    assert [row["item_id"] for row in read_jsonl(out / "replies_base.jsonl")] == ["e2", "e1"]
    assert [row["item_id"] for row in read_jsonl(out / "replies_variant.jsonl")] == ["e2"]
    key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    assert [row["item_id"] for row in key["items"]] == ["e2"]
    page = (out / "blind.html").read_text(encoding="utf-8")
    assert 'data-number="1"' in page and 'data-number="2"' not in page
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stopped"] == reason
    assert manifest["item_ids"] == ["e2", "e1", "e4"]
    assert manifest["presented_item_ids"] == ["e2"]
    assert (manifest["completed_requests"], manifest["spent_cny"]) == (3, "0.30")
    assert [manifest["groups"][g]["completed"] for g in GROUPS] == [2, 1]


def test_real_out_dir_must_be_under_test_results(tmp_path, monkeypatch, capsys):
    stub_real(tmp_path, monkeypatch, results_dir=tmp_path / "allowed")
    monkeypatch.setattr(d14, "build_real_orchestrator", lambda *a, **kw: pytest.fail("assembly"))
    monkeypatch.setattr(d14, "load_dialogue_config", lambda *a, **kw: pytest.fail("config"))
    args = [*make_data(tmp_path), "--real", "--max-cny", "10"]
    assert d14.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "invalid_out_dir"}


def test_unexpected_failure_is_run_failed_without_details(tmp_path, monkeypatch, capsys):
    stub_real(tmp_path, monkeypatch)

    def build(*a, **kw):
        return SimpleNamespace(ensure_paid_capacity=lambda n: (_ for _ in ()).throw(
            RuntimeError("synthetic private detail")))

    monkeypatch.setattr(d14, "build_real_orchestrator", build)
    args = [*make_data(tmp_path), "--real", "--max-cny", "10"]
    assert d14.main(args) == 1
    assert json.loads(capsys.readouterr().out) == {"status": "run_failed"}


def test_script_runs_as_file_without_pythonpath(tmp_path):
    args = make_data(tmp_path)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "complete"


# --- recheck -----------------------------------------------------------------------------------


def make_source(tmp_path, count=8):
    source = tmp_path / "results/source"
    source.mkdir(parents=True)
    ids = [f"r{i}" for i in range(count)]
    items = [{"item_id": i, "text": f"消息{i}", "recent_context": [f"她：前文{i}"]} for i in ids]
    (source / "items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
    for group, label in (("base", "甲"), ("variant", "乙")):
        rows = [{"item_id": i, "reply": f"{label}答{i}",
                 "blocked_reason": "provider_timeout" if (group, i) == ("variant", "r3") else None}
                for i in ids]
        (source / f"replies_{group}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    key = {"kind": "pair", "seed": 1, "items": [
        {"number": n, "item_id": i, "A": "base", "B": "variant"} for n, i in enumerate(ids, 1)
    ]}
    (source / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False),
                                            encoding="utf-8")
    monkeypatch_settings = SimpleNamespace(test_results_dir=tmp_path / "results",
                                           runtime_data_dir=tmp_path / "runtime")
    return source, ids, monkeypatch_settings


def test_recheck_samples_six_reshuffles_and_sends_nothing(tmp_path, monkeypatch, capsys):
    source, ids, settings = make_source(tmp_path)
    monkeypatch.setattr(d14, "load_settings", lambda root: settings)
    forbid_assembly(monkeypatch)
    monkeypatch.setattr(ev.FakeProvider, "complete", lambda *a: pytest.fail("request"))
    out = tmp_path / "results/recheck"
    assert d14.main(["recheck", "--source-dir", str(source), "--out-dir", str(out),
                     "--seed", "11"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "complete", "mode": "recheck", "items": 6,
    }
    rng = random.Random(11)
    picked = rng.sample(ids, 6)
    orders = []
    for _ in picked:
        order = list(GROUPS)
        rng.shuffle(order)
        orders.append(order)
    key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    assert key == {"kind": "recheck", "seed": 11, "items": [
        {"number": n, "item_id": i, "A": o[0], "B": o[1]}
        for n, (i, o) in enumerate(zip(picked, orders, strict=True), 1)
    ]}
    page = (out / "blind.html").read_text(encoding="utf-8")
    for n, (i, o) in enumerate(zip(picked, orders, strict=True), 1):
        part = section(page, n)
        assert f"<p>她：前文{i}</p>" in part
        assert f'<p class="current">当前消息：消息{i}</p>' in part
        label = {"base": "甲", "variant": "乙"}
        assert f'<h3>A</h3><p>{label[o[0]]}答{i}</p>' in part
        assert f'<h3>B</h3><p>{label[o[1]]}答{i}</p>' in part
        assert ('<p class="blocked">拦截：provider_timeout</p>' in part) == (i == "r3")
    assert 'data-number="7"' not in page
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "version": 1, "kind": "recheck", "seed": 11,
        "source_answer_key_sha256": sha((source / "answer_key.json").read_bytes()),
        "item_ids": picked,
    }
    assert sorted(p.name for p in out.iterdir()) == ["answer_key.json", "blind.html",
                                                     "manifest.json"]


def test_recheck_needs_six_presented_items(tmp_path, monkeypatch, capsys):
    source, _, settings = make_source(tmp_path, count=5)
    monkeypatch.setattr(d14, "load_settings", lambda root: settings)
    out = tmp_path / "results/recheck"
    assert d14.main(["recheck", "--source-dir", str(source), "--out-dir", str(out),
                     "--seed", "1"]) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "insufficient_items"}
    assert not out.exists()


@pytest.mark.parametrize("target", ["same", "outside"])
def test_recheck_out_dir_boundary(tmp_path, monkeypatch, capsys, target):
    source, _, settings = make_source(tmp_path)
    monkeypatch.setattr(d14, "load_settings", lambda root: settings)
    out = source if target == "same" else tmp_path / "elsewhere"
    assert d14.main(["recheck", "--source-dir", str(source), "--out-dir", str(out),
                     "--seed", "1"]) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "invalid_out_dir"}
    assert not (tmp_path / "elsewhere").exists()


# --- score -------------------------------------------------------------------------------------


def pair_key(count, flip=()):
    return {"kind": "pair", "seed": 0, "items": [
        {"number": n, "item_id": f"q{n}",
         "A": "variant" if n in flip else "base", "B": "base" if n in flip else "variant"}
        for n in range(1, count + 1)
    ]}


def score(tmp_path, capsys, results, key, *extra, recheck=None):
    (tmp_path / "results.txt").write_text(results, encoding="utf-8")
    (tmp_path / "key.json").write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
    args = ["score", "--results", str(tmp_path / "results.txt"),
            "--answer-key", str(tmp_path / "key.json"), *extra]
    if recheck is not None:
        (tmp_path / "recheck.txt").write_text(recheck[0], encoding="utf-8")
        (tmp_path / "recheck_key.json").write_text(json.dumps(recheck[1], ensure_ascii=False),
                                                   encoding="utf-8")
        args += ["--recheck-results", str(tmp_path / "recheck.txt"),
                 "--recheck-key", str(tmp_path / "recheck_key.json")]
    code = d14.main(args)
    return code, json.loads(capsys.readouterr().out)


def test_score_counts_map_labels_per_item(tmp_path, capsys):
    key = pair_key(8, flip={2, 4})
    results = ("1B/AB,2A/A/A理解,3并列/无/A接法措辞B超时拦截,4B/B/B理解,"
               "5都不像/无,6超出语境/A/B接法理解,7A/AB,8B/B")
    code, result = score(tmp_path, capsys, results, key, "--stage", "screen")
    assert code == 0
    assert result == {
        "status": "scored", "stage": "screen", "total": 8,
        "wins": 3, "losses": 2, "ties": 1, "neither": 1, "out_of_context": 1,
        "acceptable": {"base": 4, "variant": 4},
        "failures": {
            "base": {"接法": 1, "措辞": 1, "理解": 1, "超时拦截": 0},
            "variant": {"接法": 1, "措辞": 0, "理解": 2, "超时拦截": 1},
        },
        "errors": {"base": 2, "variant": 2},
        "recheck": None,
        "checks": {"errors_reduced_by_2": False, "acceptable_not_lower": True},
        "verdict": "fail",
        "manual_checks": [],
    }


def screen_results(base_errors, variant_errors, variant_ok, base_ok, wins):
    entries = []
    for n in range(1, 9):
        pref = "B" if n <= wins else "A"
        ok = ("A" if n <= base_ok else "") + ("B" if n <= variant_ok else "")
        fails = ("A理解" if n <= base_errors else "") + ("B措辞" if n <= variant_errors else "")
        entries.append(f"{n}{pref}/{ok or '无'}" + (f"/{fails}" if fails else ""))
    return ",".join(entries)


@pytest.mark.parametrize("errors, oks, wins, require, checks, verdict", [
    ((2, 0), (6, 6), 0, False,
     {"errors_reduced_by_2": True, "acceptable_not_lower": True}, "pass"),
    ((2, 1), (6, 6), 0, False,
     {"errors_reduced_by_2": False, "acceptable_not_lower": True}, "fail"),
    ((3, 1), (6, 5), 0, False,
     {"errors_reduced_by_2": True, "acceptable_not_lower": False}, "fail"),
    ((3, 1), (6, 6), 5, True,
     {"errors_reduced_by_2": True, "acceptable_not_lower": True, "wins_at_least_5": True},
     "pass"),
    ((3, 1), (6, 6), 4, True,
     {"errors_reduced_by_2": True, "acceptable_not_lower": True, "wins_at_least_5": False},
     "fail"),
])
def test_screen_pass_lines(tmp_path, capsys, errors, oks, wins, require, checks, verdict):
    results = screen_results(errors[0], errors[1], oks[1], oks[0], wins)
    extra = ["--stage", "screen", *(["--require-wins"] if require else [])]
    code, result = score(tmp_path, capsys, results, pair_key(8), *extra)
    assert code == 0
    assert result["errors"] == {"base": errors[0], "variant": errors[1]}
    assert result["acceptable"] == {"base": oks[0], "variant": oks[1]}
    assert (result["checks"], result["verdict"]) == (checks, verdict)


def test_errors_count_items_not_checkboxes_and_ignore_timeouts(tmp_path, capsys):
    results = "1A/无/A接法措辞理解超时拦截B超时拦截," + ",".join(
        f"{n}A/AB" for n in range(2, 9))
    _, result = score(tmp_path, capsys, results, pair_key(8), "--stage", "screen")
    assert result["errors"] == {"base": 1, "variant": 0}
    assert result["failures"]["variant"]["超时拦截"] == 1


def combo_results(wins, losses, base_ok=20, variant_ok=20, base_understanding=0,
                  variant_understanding=0):
    entries = []
    for n in range(1, 31):
        pref = "B" if n <= wins else "A" if n <= wins + losses else "并列"
        ok = ("A" if n <= base_ok else "") + ("B" if n <= variant_ok else "")
        fails = (("A理解" if n <= base_understanding else "")
                 + ("B理解" if n <= variant_understanding else ""))
        entries.append(f"{n}{pref}/{ok or '无'}" + (f"/{fails}" if fails else ""))
    return ",".join(entries)


def recheck_pair(original_prefs, recheck_prefs):
    key = {"kind": "recheck", "seed": 5, "items": [
        {"number": n, "item_id": f"q{q}", "A": "variant", "B": "base"}
        for n, q in enumerate(range(1, 7), 1)
    ]}
    del original_prefs
    return ",".join(f"{n}{p}/AB" for n, p in enumerate(recheck_prefs, 1)), key


@pytest.mark.parametrize("kwargs, checks_false, verdict", [
    ({"wins": 18, "losses": 6}, [], "incomplete"),
    ({"wins": 17, "losses": 6}, ["wins_at_least_18"], "fail"),
    ({"wins": 18, "losses": 12}, [], "incomplete"),
    ({"wins": 18, "losses": 6, "variant_ok": 19}, ["acceptable_not_lower"], "fail"),
    ({"wins": 18, "losses": 6, "base_understanding": 2, "variant_understanding": 3},
     ["comprehension_not_higher"], "fail"),
    ({"wins": 18, "losses": 6, "base_understanding": 3, "variant_understanding": 3}, [],
     "incomplete"),
])
def test_combo_pass_lines_without_recheck(tmp_path, capsys, kwargs, checks_false, verdict):
    code, result = score(tmp_path, capsys, combo_results(**kwargs), pair_key(30),
                         "--stage", "combo")
    assert code == 0
    assert set(result["checks"]) == {"wins_at_least_18", "wins_over_losses",
                                     "acceptable_not_lower", "comprehension_not_higher"}
    assert [k for k, v in result["checks"].items() if not v] == checks_false
    assert result["verdict"] == verdict
    assert result["recheck"] is None
    assert result["manual_checks"] == ["越权或串会话为 0"]


def test_combo_wins_must_exceed_losses(tmp_path, capsys):
    key = pair_key(30)
    results = ",".join(f"{n}{'B' if n <= 18 else 'A'}/AB" for n in range(1, 31))
    _, result = score(tmp_path, capsys, results, key, "--stage", "combo")
    assert result["checks"]["wins_over_losses"] is True
    results = ",".join(f"{n}{'B' if n <= 15 else 'A'}/AB" for n in range(1, 31))
    _, result = score(tmp_path, capsys, results, key, "--stage", "combo")
    assert result["checks"]["wins_over_losses"] is False
    assert result["checks"]["wins_at_least_18"] is False


@pytest.mark.parametrize("recheck_prefs, agree, verdict", [
    (["A", "A", "A", "A", "A", "A"], 6, "pass"),
    (["A", "A", "A", "A", "A", "并列"], 5, "pass"),
    (["A", "A", "A", "A", "B", "并列"], 4, "fail"),
])
def test_combo_recheck_agreement_maps_through_both_keys(tmp_path, capsys, recheck_prefs,
                                                        agree, verdict):
    # Original: items 1-18 prefer variant (label B). The recheck key flips labels, so
    # "A" in the recheck also means variant.
    results = combo_results(wins=18, losses=6)
    recheck = recheck_pair(None, recheck_prefs)
    code, result = score(tmp_path, capsys, results, pair_key(30), "--stage", "combo",
                         recheck=recheck)
    assert code == 0
    assert result["recheck"] == {"total": 6, "agree": agree}
    assert result["checks"]["recheck_agree_at_least_5"] is (agree >= 5)
    assert result["verdict"] == verdict


def test_combo_failed_check_beats_missing_recheck(tmp_path, capsys):
    _, result = score(tmp_path, capsys, combo_results(wins=10, losses=6), pair_key(30),
                      "--stage", "combo")
    assert result["verdict"] == "fail"


@pytest.mark.parametrize("stage, count", [("screen", 7), ("screen", 9), ("combo", 29)])
def test_wrong_item_count_is_insufficient(tmp_path, capsys, stage, count):
    results = ",".join(f"{n}B/AB" for n in range(1, count + 1))
    code, result = score(tmp_path, capsys, results, pair_key(count), "--stage", stage)
    assert code == 0
    assert result["total"] == count
    assert (result["checks"], result["verdict"]) == ({}, "insufficient_items")


def test_results_accept_whitespace_chinese_commas_and_any_order(tmp_path, capsys):
    results = " 2A/无，\n1B/AB , 3并列/A/B理解\n"
    code, result = score(tmp_path, capsys, results, pair_key(3), "--stage", "screen")
    assert code == 0
    assert (result["wins"], result["losses"], result["ties"]) == (1, 1, 1)
    assert result["failures"]["variant"]["理解"] == 1


@pytest.mark.parametrize("results", [
    "1B/AB,2A/AB",
    "1B/AB,2A/AB,3A/AB,4A/AB",
    "1B/AB,1A/AB,3A/AB",
    "1B/AB,2A/AB,3?/?",
    "1C/AB,2A/AB,3A/AB",
    "1B/BA,2A/AB,3A/AB",
    "1B/AB/,2A/AB,3A/AB",
    "1B/AB/A,2A/AB,3A/AB",
    "1B/AB/B接法A理解,2A/AB,3A/AB",
    "1B/AB/A理解接法,2A/AB,3A/AB",
    "1B/AB/A跑题,2A/AB,3A/AB",
    "0B/AB,1A/AB,2A/AB",
    "",
])
def test_invalid_results_text(tmp_path, capsys, results):
    code, result = score(tmp_path, capsys, results, pair_key(3), "--stage", "screen")
    assert (code, result) == (2, {"status": "invalid_input"})


@pytest.mark.parametrize("case", ["require_wins_combo", "recheck_screen", "recheck_half",
                                  "key_kind", "recheck_kind", "recheck_size",
                                  "recheck_unknown_item"])
def test_invalid_score_options(tmp_path, capsys, case):
    results = combo_results(wins=18, losses=6)
    key = pair_key(30)
    recheck = recheck_pair(None, ["A"] * 6)
    extra = ["--stage", "combo"]
    if case == "require_wins_combo":
        extra.append("--require-wins")
        recheck = None
    elif case == "recheck_screen":
        extra = ["--stage", "screen"]
        results = ",".join(f"{n}B/AB" for n in range(1, 9))
        key = pair_key(8)
    elif case == "recheck_half":
        (tmp_path / "recheck.txt").write_text(recheck[0], encoding="utf-8")
        extra += ["--recheck-results", str(tmp_path / "recheck.txt")]
        recheck = None
    elif case == "key_kind":
        key["kind"] = "recheck"
        recheck = None
    elif case == "recheck_kind":
        recheck[1]["kind"] = "pair"
    elif case == "recheck_size":
        recheck = (",".join(f"{n}A/AB" for n in range(1, 6)),
                   {"kind": "recheck", "seed": 5, "items": recheck[1]["items"][:5]})
    else:
        recheck[1]["items"][0]["item_id"] = "unknown"
    code, result = score(tmp_path, capsys, results, key, *extra, recheck=recheck)
    assert (code, result) == (2, {"status": "invalid_input"})


def test_missing_results_file_is_invalid_input(tmp_path, capsys):
    (tmp_path / "key.json").write_text(json.dumps(pair_key(3)), encoding="utf-8")
    code = d14.main(["score", "--results", str(tmp_path / "absent.txt"),
                     "--answer-key", str(tmp_path / "key.json"), "--stage", "screen"])
    assert (code, json.loads(capsys.readouterr().out)) == (2, {"status": "invalid_input"})


# --- page result line (JavaScript) ---------------------------------------------------------------


def test_page_result_line_formatter(tmp_path, monkeypatch, capsys):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    args = make_data(tmp_path)
    record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    page = (tmp_path / "out/blind.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="d14-format">(.*?)</script>', page, re.DOTALL)
    assert match is not None
    rows = [
        {"pref": "A", "ok": "AB", "fa": [], "fb": []},
        {"pref": "都不像", "ok": "无", "fa": ["理解", "接法"], "fb": []},
        {"pref": "B", "ok": "B", "fa": [], "fb": ["超时拦截", "措辞", "措辞"]},
        {"pref": "超出语境", "ok": "A", "fa": ["措辞"], "fb": ["接法"]},
        {"pref": "", "ok": "", "fa": [], "fb": []},
    ]
    script = match.group(1) + f"\nprocess.stdout.write(resultLine({json.dumps(rows)}));\n"
    (tmp_path / "format.js").write_text(script, encoding="utf-8")
    output = subprocess.run([node, str(tmp_path / "format.js")], capture_output=True,
                            text=True, check=True).stdout
    assert output == "1A/AB,2都不像/无/A接法理解,3B/B/B措辞超时拦截,4超出语境/A/A措辞B接法,5?/?"


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("term", ["周末", "哈哈哈"])
def test_group_specific_terms_prompt_and_hashes(tmp_path, monkeypatch, capsys, group, term):
    args = make_data(tmp_path)
    path = tmp_path / "terms.json"
    path.write_text(json.dumps({"version": 1, "entries": [{
        "term": term, "meaning": "合成专属含义", "occasion": "合成场合",
        "speaker": "both", "addressee": "合成对象", "permission": "allowed",
    }]}), encoding="utf-8")
    calls = record_fake_provider(monkeypatch)
    assert d14.main([*args, f"--{group}-terms", str(path)]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "out/manifest.json").read_text(encoding="utf-8"))
    first = {}
    for name, system, _ in calls:
        first.setdefault(name, system)
    other = "variant" if group == "base" else "base"
    assert "你们之间的专属说法" in first[group]
    assert "合成专属含义" in first[group]
    assert all("你们之间的专属说法" not in system
               for name, system, _ in calls if name == other)
    assert manifest["groups"][group]["terms_sha256"] == sha(path.read_bytes())
    assert manifest["groups"][other]["terms_sha256"] is None
    for name in GROUPS:
        assert manifest["groups"][name]["system_prompt_sha256"] == sha(
            first[name].encode("utf-8"))


@pytest.mark.parametrize("group", GROUPS)
def test_damaged_terms_are_configuration_unavailable(tmp_path, monkeypatch, capsys, group):
    args = make_data(tmp_path)
    path = tmp_path / "terms.json"
    path.write_text("{broken", encoding="utf-8")
    forbid_assembly(monkeypatch)
    assert d14.main([*args, f"--{group}-terms", str(path)]) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "configuration_unavailable"}


@pytest.mark.parametrize("group", GROUPS)
def test_real_assembly_receives_group_terms(tmp_path, monkeypatch, capsys, group):
    state = stub_real(tmp_path, monkeypatch)
    args = make_data(tmp_path)
    path = tmp_path / "terms.json"
    path.write_text(json.dumps({"version": 1, "entries": [{
        "term": "周末", "meaning": "合成含义", "occasion": "合成场合",
        "speaker": "both", "addressee": "合成对象", "permission": "allowed",
    }]}), encoding="utf-8")
    assert d14.main([*args, "--real", "--max-cny", "10",
                     f"--{group}-terms", str(path)]) == 0
    capsys.readouterr()
    built = dict(zip(GROUPS, state["built"], strict=True))
    assert built[group]["snippet_shared_terms"][0].meaning == "合成含义"
    other = "variant" if group == "base" else "base"
    assert built[other]["snippet_shared_terms"] is None


@pytest.mark.parametrize("group", GROUPS)
def test_group_specific_session_check_prompt_and_manifest(tmp_path, monkeypatch, capsys, group):
    from companion.dialogue.context import SNIPPET_SESSION_CHECK

    args = make_data(tmp_path)
    calls = record_fake_provider(monkeypatch)
    assert d14.main([*args, f"--{group}-session-check"]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "out/manifest.json").read_text(encoding="utf-8"))
    other = "variant" if group == "base" else "base"
    first = {}
    for name, system, _ in calls:
        first.setdefault(name, system)
    assert all(SNIPPET_SESSION_CHECK in system for name, system, _ in calls if name == group)
    assert all(SNIPPET_SESSION_CHECK not in system for name, system, _ in calls if name == other)
    assert first[group] != first[other]
    assert manifest["groups"][group]["session_check"] is True
    assert manifest["groups"][other]["session_check"] is False
    for name in GROUPS:
        assert manifest["groups"][name]["system_prompt_sha256"] == sha(
            first[name].encode("utf-8"))


def test_real_assembly_receives_group_session_check(tmp_path, monkeypatch, capsys):
    state = stub_real(tmp_path, monkeypatch)
    args = make_data(tmp_path)
    assert d14.main([*args, "--real", "--max-cny", "10", "--variant-session-check"]) == 0
    capsys.readouterr()
    assert [kw["snippet_session_check"] for kw in state["built"]] == [False, True]


# --- run: synthetic items and base-only mode -----------------------------------------------


def synthetic_items_args(tmp_path, rows):
    args = make_data(tmp_path)
    path = tmp_path / "items.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return [*args, "--items-file", str(path)]


def test_items_file_outputs_prepared_context_and_preserves_metadata(tmp_path, monkeypatch,
                                                                  capsys):
    row = {"id": "synthetic-q", "item_id": "source-item-id", "text": "今天有什么安排呀",
           "recent_context": ["她：早呀", "  "],
           "expect": {"secret": "METADATA_EXPECT"}, "notes": ["METADATA_NOTES"]}
    args = synthetic_items_args(tmp_path, [row])
    calls = record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    capsys.readouterr()
    saved = read_jsonl(tmp_path / "out/items.jsonl")[0]
    assert saved["recent_context"] == ["她：早呀"]
    assert saved["item_id"] == row["id"]
    assert saved["id"] == row["id"]
    assert saved["text"] == row["text"]
    assert saved["expect"] == row["expect"]
    assert saved["notes"] == row["notes"]
    for _, system, user in calls:
        assert saved["text"] in user
        assert saved["recent_context"][0] in system + user
        assert "METADATA_" not in system + user


@pytest.mark.parametrize("only_base", [False, True])
def test_items_file_preserves_metadata_without_sending_it(tmp_path, monkeypatch, capsys,
                                                         only_base):
    rows = [{"id": "synthetic-q", "text": "今天有什么安排呀", "recent_context": ["她：早呀"],
             "expect": {"secret": "METADATA_EXPECT"}, "group": "METADATA_GROUP",
             "notes": ["METADATA_NOTES"]}]
    args = synthetic_items_args(tmp_path, rows)
    if only_base:
        args.append("--only-base")
    calls = record_fake_provider(monkeypatch)
    adapters = capture_adapters(monkeypatch)
    assert d14.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["planned_requests"] == (1 if only_base else 2)
    out = tmp_path / "out"
    assert read_jsonl(out / "items.jsonl") == [{**rows[0], "item_id": "synthetic-q"}]
    assert [group for group, _, _ in calls] == (["base"] if only_base else list(GROUPS))
    for _, system, user in calls:
        assert "METADATA_" not in system + user
        assert rows[0]["text"] in user
        assert rows[0]["recent_context"][0] in system + user
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["items_file_sha256"] == sha((tmp_path / "items.json").read_bytes())
    assert manifest["item_ids"] == ["synthetic-q"]
    assert manifest["removed"] == {
        "occupied": 4, "same_reply": 0, "context_overlap": 0, "near_duplicate": 0,
    }
    assert len(adapters) == (1 if only_base else 2)
    assert manifest["library_after"] == 7
    assert ("only_base" in manifest) is only_base
    if only_base:
        assert manifest["only_base"] is True
        assert set(manifest["groups"]) == {"base"}
        assert sorted(p.name for p in out.iterdir()) == [
            "items.jsonl", "manifest.json", "replies_base.jsonl",
        ]


@pytest.mark.parametrize("rows", [
    [{"id": "q", "text": "你好", "recent_context": []}] * 2,
    [{"id": "q", "text": "", "recent_context": []}],
    [{"id": "q", "text": "你好", "recent_context": ["早"] * 11}],
    [{"id": "", "text": "你好", "recent_context": []}],
    [{"id": "q", "text": "   ", "recent_context": []}],
    [{"id": "q", "text": "你好", "recent_context": [1]}],
    [{"id": "q", "text": "你好"}],
    {"id": "q", "text": "你好", "recent_context": []},
])
def test_items_file_invalid_rows_return_invalid_items(tmp_path, monkeypatch, capsys, rows):
    args = synthetic_items_args(tmp_path, rows)
    forbid_assembly(monkeypatch)
    assert d14.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "invalid_items"}


def test_only_base_ignores_all_variant_inputs(tmp_path, monkeypatch, capsys):
    args = make_data(tmp_path)
    set_arg(args, "--variant-card", str(tmp_path / "missing-card.txt"))
    args += ["--variant-terms", str(tmp_path / "missing-terms.json"),
             "--variant-session-check", "--only-base"]
    set_arg(args, "--max-requests", "3")
    calls = record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    assert json.loads(capsys.readouterr().out)["planned_requests"] == 3
    assert [group for group, _, _ in calls] == ["base"] * 3
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "items.jsonl", "manifest.json", "replies_base.jsonl",
    ]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["only_base"] is True
    assert set(manifest["groups"]) == {"base"}
    assert manifest["groups"]["base"]["completed"] == 3
    assert manifest["planned_requests"] == manifest["completed_requests"] == 3
    assert "items_file_sha256" not in manifest


@pytest.mark.parametrize("only_base", [False, True])
@pytest.mark.parametrize("missing", ["--variant-card", "--variant-strata", "both"])
def test_variant_required_unless_only_base(tmp_path, monkeypatch, capsys, only_base, missing):
    args = make_data(tmp_path)
    for option in ("--variant-card", "--variant-strata"):
        if missing in (option, "both"):
            index = args.index(option)
            del args[index:index + 2]
    calls = record_fake_provider(monkeypatch)
    if only_base:
        args.append("--only-base")
        assert d14.main(args) == 0
        assert len(calls) == 3
    else:
        assert d14.main(args) == 2
        assert calls == []
    capsys.readouterr()


def test_items_file_request_cap_before_assembly(tmp_path, monkeypatch, capsys):
    args = synthetic_items_args(tmp_path, [
        {"id": "q1", "text": "你好", "recent_context": []},
        {"id": "q2", "text": "晚安", "recent_context": []},
    ])
    args.append("--only-base")
    set_arg(args, "--max-requests", "1")
    forbid_assembly(monkeypatch)
    assert d14.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "request_limit_exceeded", "planned_requests": 2,
    }


def test_items_file_does_not_require_pool_questions(tmp_path, monkeypatch, capsys):
    pools = json.loads(json.dumps(POOLS))
    pools["pools"]["screen_shape"] = []
    args = make_data(tmp_path, pools=pools, holdout=[], library=[])
    path = tmp_path / "items.json"
    path.write_text(json.dumps([
        {"id": "q", "text": "你好", "recent_context": ["她：早"] * 10},
    ]), encoding="utf-8")
    args += ["--items-file", str(path), "--only-base"]
    calls = record_fake_provider(monkeypatch)
    assert d14.main(args) == 0
    assert len(calls) == 1
    capsys.readouterr()
