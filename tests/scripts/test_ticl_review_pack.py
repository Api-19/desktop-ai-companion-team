# ruff: noqa: RUF001
"""Contract tests for scripts/ticl_review_pack.py (synthetic data only, no requests)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.privacy.paired_snippets import PairedSnippet, write_snippets
from companion.runtime.settings import SettingsError
from scripts import ticl_review_pack as ticl

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ticl_review_pack.py"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
IDS = ["evt-alpha-01", "evt-alpha-02", "evt-alpha-03"]
ITEMS = [
    {"item_id": IDS[0], "text": "今天<好>累",
     "recent_context": ["他：吃饭了吗", "她：<b>还没</b> & 等你"]},
    {"item_id": IDS[1], "text": "周末去哪", "recent_context": []},
    {"item_id": IDS[2], "text": "晚安", "recent_context": ["她：困了"]},
]
REPLIES = {
    "base": [("基线答一\n第二行", None), ("基线答二", None), ('基线答"三"', None)],
    "variant": [("变体答一", None), ("", "provider_timeout"), ("变体答三", None)],
}


def snip(sid, partner, reply):
    return PairedSnippet(
        snippet_id=sid, conversation_id="synthetic", started_at=T0,
        partner_turns=list(partner), owner_reply=list(reply),
        source_fingerprint="synthetic",
    )


HOLDOUT = [snip(IDS[0], ["今天<好>累"], ["抱抱<3", "早点睡"]),
           snip(IDS[1], ["周末去哪"], ["去海边吧"])]
LIBRARY = [snip(IDS[0], ["今天<好>累"], ["库里的旧回复"]),
           snip(IDS[2], ["晚安"], ["晚安呀"]),
           snip("lib-other", ["随便"], ["嗯"])]
OWNER = {IDS[0]: ["抱抱&lt;3", "早点睡"], IDS[1]: ["去海边吧"], IDS[2]: ["晚安呀"]}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def stub_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(ticl, "load_settings",
                        lambda root: SimpleNamespace(test_results_dir=tmp_path / "results"))


def make_run(tmp_path, *, presented=None, drop=None):
    """Write a synthetic d14_pair_eval run dir plus snippet files; return build argv."""
    run = tmp_path / "results/run"
    run.mkdir(parents=True)
    write_jsonl(run / "items.jsonl",
                [r for r in ITEMS if (drop, r["item_id"]) != ("items", IDS[1])])
    for group, rows in REPLIES.items():
        write_jsonl(run / f"replies_{group}.jsonl", [
            {"item_id": i, "reply": reply, "blocked_reason": blocked}
            for i, (reply, blocked) in zip(IDS, rows, strict=True)
            if (drop, group, i) != ("replies", "base", IDS[1])
        ])
    manifest = {"version": 1, "mode": "fake", "item_ids": [*IDS, "evt-alpha-04"],
                "presented_item_ids": IDS if presented is None else presented}
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False),
                                       encoding="utf-8")
    holdout = [s for s in HOLDOUT if (drop, s.snippet_id) != ("snippet", IDS[1])]
    write_snippets(tmp_path / "holdout.jsonl", holdout)
    write_snippets(tmp_path / "library.jsonl", LIBRARY)
    return ["build", "--run-dir", str(run), "--group", "a",
            "--library", str(tmp_path / "library.jsonl"),
            "--holdout", str(tmp_path / "holdout.jsonl"),
            "--out-dir", str(tmp_path / "results/pack")]


def set_arg(args, option, value):
    args[args.index(option) + 1] = value
    return args


def run_main(capsys, args):
    code = ticl.main(args)
    return code, json.loads(capsys.readouterr().out)


def section(page, number):
    match = re.search(rf'<section class="item" data-number="{number}">(.*?)</section>',
                      page, re.DOTALL)
    assert match is not None
    return match.group(1)


def build_page(tmp_path, monkeypatch, capsys, group="a", **kwargs):
    stub_settings(tmp_path, monkeypatch)
    args = set_arg(make_run(tmp_path, **kwargs), "--group", group)
    assert run_main(capsys, args)[0] == 0
    return (tmp_path / "results/pack/review.html").read_text(encoding="utf-8")


# --- build: outputs ----------------------------------------------------------------------


def test_build_writes_pack_and_status(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    args = make_run(tmp_path)
    assert run_main(capsys, args) == (0, {"status": "complete", "group": "a", "items": 3})
    out = tmp_path / "results/pack"
    assert sorted(p.name for p in out.iterdir()) == ["pack.json", "review.html"]
    pack = json.loads((out / "pack.json").read_text(encoding="utf-8"))
    assert pack == {
        "version": 1, "group": "a",
        "manifest_sha256": sha((tmp_path / "results/run/manifest.json").read_bytes()),
        "items": [{"number": n, "event_id": i, "snippet_id": i}
                  for n, i in enumerate(IDS, 1)],
    }


@pytest.mark.parametrize(("group", "replies"), [
    ("a", ["基线答一<br>第二行", "基线答二", "基线答&quot;三&quot;"]),
    ("b", ["变体答一", "", "变体答三"]),
])
def test_build_group_selects_reply_file(tmp_path, monkeypatch, capsys, group, replies):
    page = build_page(tmp_path, monkeypatch, capsys, group=group)
    for number, reply in enumerate(replies, 1):
        assert f'<div class="model"><h3>试答</h3><p>{reply}</p>' in section(page, number)
    other = "变体答" if group == "a" else "基线答"
    assert other not in page
    pack = json.loads((tmp_path / "results/pack/pack.json").read_text(encoding="utf-8"))
    assert pack["group"] == group


def test_review_section_shows_context_current_model_owner_and_note(tmp_path, monkeypatch,
                                                                   capsys):
    page = build_page(tmp_path, monkeypatch, capsys)
    assert page.startswith("<!DOCTYPE html>")
    part = section(page, 1)
    marks = [
        "<h2>第 1 题</h2>",
        '<div class="context">',
        "<p>他：吃饭了吗</p>",
        "<p>她：&lt;b&gt;还没&lt;/b&gt; &amp; 等你</p>",
        '<p class="current">当前消息：今天&lt;好&gt;累</p>',
        '<div class="model"><h3>试答</h3><p>基线答一<br>第二行</p>',
        '<div class="owner"><h3>真答</h3>',
        f"<p>{OWNER[IDS[0]][0]}</p>",
        f"<p>{OWNER[IDS[0]][1]}</p>",
        '<label>差异备注<textarea class="note" data-number="1"></textarea></label>',
    ]
    positions = [part.index(mark) for mark in marks]
    assert positions == sorted(positions)
    for number, item_id in enumerate(IDS, 1):
        part = section(page, number)
        assert f'<textarea class="note" data-number="{number}"></textarea>' in part
        for line in OWNER[item_id]:
            assert f"<p>{line}</p>" in part.split('<div class="owner">')[1]
    assert '<div class="context">\n</div>' in section(page, 2)
    assert "库里的旧回复" not in page
    assert 'data-number="4"' not in page


def test_blocked_reply_is_marked(tmp_path, monkeypatch, capsys):
    page = build_page(tmp_path, monkeypatch, capsys, group="b")
    blocked = '<p class="blocked">拦截：provider_timeout</p>'
    assert [blocked in section(page, n) for n in (1, 2, 3)] == [False, True, False]
    model = section(page, 2).split('<div class="model">')[1].split("</div>")[0]
    assert model == f"<h3>试答</h3><p></p>\n{blocked}\n"


def test_review_page_claim_area_export_box_and_meta(tmp_path, monkeypatch, capsys):
    page = build_page(tmp_path, monkeypatch, capsys)
    manifest_sha = sha((tmp_path / "results/run/manifest.json").read_bytes())
    assert f'<body data-manifest="{manifest_sha}" data-group="a">' in page
    claims = re.search(r'<section class="claims">(.*?)</section>', page, re.DOTALL)
    assert claims is not None
    assert claims.group(1).count('<fieldset class="claim"') == 4
    for k in (1, 2, 3, 4):
        fieldset = re.search(rf'<fieldset class="claim" data-claim="{k}">(.*?)</fieldset>',
                             page, re.DOTALL)
        assert fieldset is not None
        body = fieldset.group(1)
        assert f"<legend>差异说明 {k}</legend>" in body
        for label, name in (("情境", "situation"), ("差异", "difference"),
                            ("适用", "applies"), ("例外", "exceptions")):
            assert f'<label>{label}<textarea name="{name}-{k}"></textarea></label>' in body
        for n in (1, 2, 3):
            assert (f'<label><input type="checkbox" name="support-{k}" value="{n}">{n}</label>'
                    in body)
        assert 'value="4"' not in body
    assert '<textarea id="export" readonly></textarea>' in page
    assert re.search(r'<script id="ticl-export">.*?</script>', page, re.DOTALL)
    assert page.index('<section class="claims">') > page.index('data-number="3"')
    assert all(item_id not in page for item_id in IDS)


def test_build_follows_presented_order(tmp_path, monkeypatch, capsys):
    presented = [IDS[2], IDS[0]]
    page = build_page(tmp_path, monkeypatch, capsys, presented=presented)
    assert '<p class="current">当前消息：晚安</p>' in section(page, 1)
    assert '<p class="current">当前消息：今天&lt;好&gt;累</p>' in section(page, 2)
    assert 'data-number="3"' not in page
    pack = json.loads((tmp_path / "results/pack/pack.json").read_text(encoding="utf-8"))
    assert pack["items"] == [{"number": 1, "event_id": IDS[2], "snippet_id": IDS[2]},
                             {"number": 2, "event_id": IDS[0], "snippet_id": IDS[0]}]


# --- build: errors -----------------------------------------------------------------------


@pytest.mark.parametrize("drop", ["items", "replies", "snippet"])
def test_build_missing_item_data(tmp_path, monkeypatch, capsys, drop):
    stub_settings(tmp_path, monkeypatch)
    args = make_run(tmp_path, drop=drop)
    assert run_main(capsys, args) == (2, {"status": "invalid_items", "missing": 1})
    assert not (tmp_path / "results/pack").exists()


def test_build_empty_presented_is_insufficient(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    args = make_run(tmp_path, presented=[])
    assert run_main(capsys, args) == (2, {"status": "insufficient_items"})
    assert not (tmp_path / "results/pack").exists()


def test_build_out_dir_must_be_under_test_results(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    args = set_arg(make_run(tmp_path), "--out-dir", str(tmp_path / "elsewhere"))
    assert run_main(capsys, args) == (2, {"status": "invalid_out_dir"})
    assert not (tmp_path / "elsewhere").exists()


def test_build_unreadable_run_is_invalid_input(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    args = make_run(tmp_path)
    (tmp_path / "results/run/manifest.json").unlink()
    assert run_main(capsys, args) == (2, {"status": "invalid_input"})
    assert not (tmp_path / "results/pack").exists()


def test_settings_error_is_configuration_unavailable(tmp_path, monkeypatch, capsys):
    args = make_run(tmp_path)

    def fail(root):
        raise SettingsError("synthetic")

    monkeypatch.setattr(ticl, "load_settings", fail)
    assert run_main(capsys, args) == (2, {"status": "configuration_unavailable"})
    paths = write_validate_inputs(tmp_path, [claim()])
    assert run_main(capsys, validate_args(tmp_path, *paths)) == (
        2, {"status": "configuration_unavailable"})


def test_script_has_no_network_or_dialogue_imports():
    source = SCRIPT.read_text(encoding="utf-8")
    for name in ("companion.dialogue", "httpx", "urllib", "requests", "socket"):
        assert name not in source


# --- page export (JavaScript) ------------------------------------------------------------


def test_page_export_builder(tmp_path, monkeypatch, capsys):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    page = build_page(tmp_path, monkeypatch, capsys)
    match = re.search(r'<script id="ticl-export">(.*?)</script>', page, re.DOTALL)
    assert match is not None
    meta = {"group": "a", "manifest": "c" * 64}
    notes = [{"number": 1, "text": "  先讲道理了 "}, {"number": 2, "text": "   "},
             {"number": 3, "text": "语气太正式"}]
    claims = [
        {"situation": " 她累的时候 ", "difference": "先抱抱", "applies": "低落",
         "exceptions": "", "support": ["3", "1", "3"]},
        {"situation": " ", "difference": "", "applies": "", "exceptions": "", "support": []},
        {"situation": "", "difference": "", "applies": "", "exceptions": "", "support": ["2"]},
    ]
    script = (match.group(1) + "\nprocess.stdout.write(buildExport("
              f"{json.dumps(meta)}, {json.dumps(notes, ensure_ascii=False)}, "
              f"{json.dumps(claims, ensure_ascii=False)}));\n")
    (tmp_path / "export.js").write_text(script, encoding="utf-8")
    output = subprocess.run([node, str(tmp_path / "export.js")], capture_output=True,
                            text=True, check=True).stdout
    assert json.loads(output) == {
        "version": 1, "group": "a", "manifest_sha256": "c" * 64,
        "notes": {"1": "先讲道理了", "3": "语气太正式"},
        "claims": [
            {"situation": "她累的时候", "difference": "先抱抱", "applies": "低落",
             "exceptions": "", "support": [1, 3]},
            {"situation": "", "difference": "", "applies": "", "exceptions": "",
             "support": [2]},
        ],
    }
    assert list(json.loads(output)) == ["version", "group", "manifest_sha256", "notes",
                                        "claims"]


# --- validate ----------------------------------------------------------------------------

MANIFEST_SHA = "c" * 64
PACK = {"version": 1, "group": "b", "manifest_sha256": MANIFEST_SHA,
        "items": [{"number": n, "event_id": f"train-{n:02d}", "snippet_id": f"train-{n:02d}"}
                  for n in range(1, 6)]}
HEADER = ("# TICL 差异说明草稿：待主人逐条确认，未写入习惯卡\n"
          f"# manifest_sha256={MANIFEST_SHA} group=b\n")


def claim(**overrides):
    row = {"situation": "她抱怨累的时候", "difference": "试答先讲道理，真答先抱抱再问原因",
           "applies": "她情绪低落时", "exceptions": "她在开玩笑时", "support": [1, 2]}
    row.update(overrides)
    return row


def write_validate_inputs(tmp_path, claims, *, export=None, pack=None):
    data = {"version": 1, "group": "b", "manifest_sha256": MANIFEST_SHA,
            "notes": {"1": "备注"}, "claims": claims}
    text = export if isinstance(export, str) else json.dumps(
        data if export is None else export, ensure_ascii=False)
    (tmp_path / "export.json").write_text(text, encoding="utf-8")
    (tmp_path / "pack.json").write_text(json.dumps(PACK if pack is None else pack),
                                        encoding="utf-8")
    return tmp_path / "export.json", tmp_path / "pack.json"


def validate_args(tmp_path, export_path, pack_path, out=None):
    return ["validate", "--export", str(export_path), "--pack", str(pack_path),
            "--out-dir", str(tmp_path / "results/addendum" if out is None else out)]


def validate(tmp_path, monkeypatch, capsys, claims, **kwargs):
    stub_settings(tmp_path, monkeypatch)
    paths = write_validate_inputs(tmp_path, claims, **kwargs)
    return run_main(capsys, validate_args(tmp_path, *paths))


def test_validate_writes_card_addendum(tmp_path, monkeypatch, capsys):
    (tmp_path / "habit_card.txt").write_text("旧卡\n", encoding="utf-8")
    claims = [claim(situation=" 她抱怨\n累的时候 ", support=[3, 1, 3]),
              claim(exceptions="无", support=[5, 2])]
    assert validate(tmp_path, monkeypatch, capsys, claims) == (
        0, {"status": "complete", "claims": 2})
    out = tmp_path / "results/addendum"
    assert [p.name for p in out.iterdir()] == ["card_addendum.txt"]
    assert (out / "card_addendum.txt").read_text(encoding="utf-8") == HEADER + (
        "\n【差异 1】\n"
        "情境：她抱怨 累的时候\n"
        "差异（试答→真答）：试答先讲道理，真答先抱抱再问原因\n"
        "适用：她情绪低落时\n"
        "例外：她在开玩笑时\n"
        "训练 ID：train-01, train-03\n"
        "\n【差异 2】\n"
        "情境：她抱怨累的时候\n"
        "差异（试答→真答）：试答先讲道理，真答先抱抱再问原因\n"
        "适用：她情绪低落时\n"
        "例外：无\n"
        "训练 ID：train-02, train-05\n"
    )
    assert (tmp_path / "habit_card.txt").read_text(encoding="utf-8") == "旧卡\n"


def test_validate_accepts_four_claims(tmp_path, monkeypatch, capsys):
    assert validate(tmp_path, monkeypatch, capsys, [claim()] * 4) == (
        0, {"status": "complete", "claims": 4})
    text = (tmp_path / "results/addendum/card_addendum.txt").read_text(encoding="utf-8")
    assert text.count("【差异 ") == 4


@pytest.mark.parametrize(("claims", "expected"), [
    ([], {"status": "no_claims"}),
    ([claim()] * 5, {"status": "too_many_claims", "claims": 5}),
    ([claim(situation="")] * 5, {"status": "too_many_claims", "claims": 5}),
])
def test_validate_claim_count(tmp_path, monkeypatch, capsys, claims, expected):
    assert validate(tmp_path, monkeypatch, capsys, claims) == (2, expected)
    assert not (tmp_path / "results/addendum").exists()


def test_validate_reports_each_bad_claim(tmp_path, monkeypatch, capsys):
    claims = [
        claim(),
        claim(situation="  ", support=[1]),
        claim(difference="差" * 161, support=[1, 9]),
        claim(applies="\n", exceptions="", support=[9, 9]),
    ]
    assert validate(tmp_path, monkeypatch, capsys, claims) == (2, {
        "status": "invalid_claims",
        "errors": [
            {"claim": 2, "field": "situation", "reason": "empty"},
            {"claim": 2, "field": "support", "reason": "too_few"},
            {"claim": 3, "field": "difference", "reason": "too_long"},
            {"claim": 3, "field": "support", "reason": "unknown"},
            {"claim": 4, "field": "applies", "reason": "empty"},
            {"claim": 4, "field": "exceptions", "reason": "empty"},
            {"claim": 4, "field": "support", "reason": "too_few"},
            {"claim": 4, "field": "support", "reason": "unknown"},
        ],
    })
    assert not (tmp_path / "results/addendum").exists()


@pytest.mark.parametrize(("field", "limit"), [
    ("situation", 80), ("difference", 160), ("applies", 80), ("exceptions", 80),
])
def test_validate_length_limits_after_whitespace_collapse(tmp_path, monkeypatch, capsys,
                                                          field, limit):
    at_limit = "字" * (limit - 2) + " \n 字"
    assert validate(tmp_path, monkeypatch, capsys, [claim(**{field: at_limit})])[0] == 0
    shutil.rmtree(tmp_path / "results")
    over = "字" * limit + "字"
    assert validate(tmp_path, monkeypatch, capsys, [claim(**{field: over})]) == (2, {
        "status": "invalid_claims",
        "errors": [{"claim": 1, "field": field, "reason": "too_long"}],
    })


@pytest.mark.parametrize("change", [
    {"group": "a"}, {"manifest_sha256": "d" * 64},
])
def test_validate_pack_mismatch(tmp_path, monkeypatch, capsys, change):
    export = {"version": 1, "group": "b", "manifest_sha256": MANIFEST_SHA,
              "notes": {}, "claims": [claim()] * 5, **change}
    assert validate(tmp_path, monkeypatch, capsys, None, export=export) == (
        2, {"status": "pack_mismatch"})
    assert not (tmp_path / "results/addendum").exists()


def _without(key):
    row = claim()
    del row[key]
    return row


@pytest.mark.parametrize("export", [
    "不是 JSON",
    {"group": "b", "manifest_sha256": MANIFEST_SHA, "claims": {}},
    {"group": "b", "manifest_sha256": MANIFEST_SHA, "claims": ["文本"]},
    {"group": "b", "manifest_sha256": MANIFEST_SHA, "claims": [_without("applies")]},
    {"group": "b", "manifest_sha256": MANIFEST_SHA, "claims": [claim(situation=5)]},
    {"group": "b", "manifest_sha256": MANIFEST_SHA, "claims": [claim(support=["1", "2"])]},
    {"group": "b", "manifest_sha256": MANIFEST_SHA, "claims": [claim(support=1)]},
    {"group": "b", "claims": [claim()]},
    [claim()],
])
def test_validate_malformed_export_is_invalid_input(tmp_path, monkeypatch, capsys, export):
    assert validate(tmp_path, monkeypatch, capsys, None, export=export) == (
        2, {"status": "invalid_input"})
    assert not (tmp_path / "results/addendum").exists()


def test_validate_missing_pack_is_invalid_input(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    export_path, _ = write_validate_inputs(tmp_path, [claim()])
    args = validate_args(tmp_path, export_path, tmp_path / "absent.json")
    assert run_main(capsys, args) == (2, {"status": "invalid_input"})


def test_validate_out_dir_must_be_under_test_results(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    paths = write_validate_inputs(tmp_path, [claim()])
    args = validate_args(tmp_path, *paths, out=tmp_path / "elsewhere")
    assert run_main(capsys, args) == (2, {"status": "invalid_out_dir"})
    assert not (tmp_path / "elsewhere").exists()


# --- end to end ----------------------------------------------------------------------------


def test_build_then_validate_maps_numbers_to_event_ids(tmp_path, monkeypatch, capsys):
    build_page(tmp_path, monkeypatch, capsys, group="b", presented=[IDS[2], IDS[0], IDS[1]])
    pack_path = tmp_path / "results/pack/pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    export = {"version": 1, "group": "b", "manifest_sha256": pack["manifest_sha256"],
              "notes": {}, "claims": [claim(support=[3, 1])]}
    (tmp_path / "export.json").write_text(json.dumps(export, ensure_ascii=False),
                                          encoding="utf-8")
    args = validate_args(tmp_path, tmp_path / "export.json", pack_path)
    assert run_main(capsys, args) == (0, {"status": "complete", "claims": 1})
    text = (tmp_path / "results/addendum/card_addendum.txt").read_text(encoding="utf-8")
    assert f"# manifest_sha256={pack['manifest_sha256']} group=b\n" in text
    assert text.endswith(f"训练 ID：{IDS[2]}, {IDS[1]}\n")


def test_script_runs_as_file_without_pythonpath(tmp_path):
    args = make_run(tmp_path)
    env = {key: value for key, value in os.environ.items()
           if key != "PYTHONPATH" and not key.startswith("COMPANION_")}
    env["COMPANION_TEST_RESULTS_DIR"] = str(tmp_path / "results")
    result = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == {"status": "complete", "group": "a", "items": 3}
