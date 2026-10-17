# ruff: noqa: RUF001
"""Contract tests for scripts/extract_shared_terms.py (synthetic data only, no requests)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.privacy.paired_snippets import PairedSnippet, write_snippets
from scripts import extract_shared_terms as st

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/extract_shared_terms.py"
T0 = datetime(2026, 3, 1, 20, 0, tzinfo=UTC)
FILL = ["好的", "嗯嗯", "知道了"]
# Synthetic general-Chinese frequency table in jieba's "word freq tag" format.
GENERAL_DICT = {"寝室": 5000, "作业": 8000, "周末": 6000, "教室": 7000, "乖乖": 600}


def snip(sid, partner, reply, day=0):
    return PairedSnippet(
        snippet_id=sid, conversation_id="synthetic", started_at=T0 + timedelta(days=day),
        partner_turns=list(partner), owner_reply=list(reply),
        source_fingerprint="synthetic", context_turns=["她：上一句里的蛋挞仔"],
    )


def base_snippets():
    return [
        # train event e1 (two snippets) + e2: the shared nickname 蛋挞仔
        snip("e1", ["蛋挞仔你在干嘛"], ["在想蛋挞仔"], day=0),
        snip("e1b", ["吃饭没"], ["吃了，蛋挞仔"], day=2),
        snip("e2", ["蛋挞仔晚安"], ["晚安"], day=5),
        snip("e2b", ["今天好累"], ["抱抱"], day=5),
        # calib event: 卷心菜精 never reaches the candidates
        snip("c1", ["卷心菜精来了"], ["卷心菜精"], day=1),
        snip("c1b", ["卷心菜精"], ["卷心菜精"], day=3),
    ]


def split_doc():
    return {"events": [
        {"event_id": "e1", "pool": "train", "snippet_ids": ["e1b"]},
        {"event_id": "e2", "pool": "train", "snippet_ids": ["e2b"]},
        {"event_id": "c1", "pool": "calib", "snippet_ids": ["c1b"]},
    ]}


def stub_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "load_settings",
                        lambda root: SimpleNamespace(test_results_dir=tmp_path / "results"))


def write_general_dict(path, freqs):
    path.write_text("".join(f"{word} {freq} n\n" for word, freq in freqs.items()),
                    encoding="utf-8")
    return path


def write_inputs(tmp_path, snippets=None, split=None, pools=None, exclude=None):
    paths = SimpleNamespace(
        snippets=tmp_path / "library.jsonl", split=tmp_path / "split.json",
        pools=tmp_path / "pools.json", exclude=None,
        out=tmp_path / "results" / "terms",
        general=write_general_dict(tmp_path / "general_dict.txt", GENERAL_DICT),
    )
    write_snippets(paths.snippets, base_snippets() if snippets is None else snippets)
    paths.split.write_text(json.dumps(split or split_doc()), encoding="utf-8")
    paths.pools.write_text(json.dumps({"pools": pools or {"unassigned": []}}), encoding="utf-8")
    if exclude is not None:
        paths.exclude = tmp_path / "exclude.json"
        paths.exclude.write_text(json.dumps(exclude, ensure_ascii=False), encoding="utf-8")
    return paths


def build_args(paths, *extra):
    args = ["build", "--snippets", str(paths.snippets), "--split", str(paths.split),
            "--pools", str(paths.pools), "--out-dir", str(paths.out)]
    if paths.general is not None:
        args += ["--general-dict", str(paths.general)]
    if paths.exclude is not None:
        args += ["--exclude-terms", str(paths.exclude)]
    return args + list(extra)


def run_build(tmp_path, monkeypatch, capsys, **kwargs):
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path, **kwargs)
    capsys.readouterr()
    code = st.main(build_args(paths))
    out = json.loads(capsys.readouterr().out)
    assert code == 0, out
    data = json.loads((paths.out / "candidates.json").read_text(encoding="utf-8"))
    return paths, data, out


def terms_of(data):
    return [c["term"] for c in data["candidates"]]


def add_train_event(split, event_id, snippet_ids):
    split["events"].append({"event_id": event_id, "pool": "train", "snippet_ids": snippet_ids})


# --- extraction ------------------------------------------------------------------------


def test_build_finds_shared_term_with_counts_and_dates(tmp_path, monkeypatch, capsys):
    _, data, out = run_build(tmp_path, monkeypatch, capsys)
    assert terms_of(data) == ["蛋挞仔"]
    item = data["candidates"][0]
    assert item["owner_count"] == 2
    assert item["partner_count"] == 2
    assert item["first_seen"] == "2026-03-01"
    assert item["last_seen"] == "2026-03-06"
    assert out["status"] == "ok"
    assert out["candidates"] == 1
    assert out["snippets_used"] == 4


def test_examples_have_snippet_id_speaker_and_original_line(tmp_path, monkeypatch, capsys):
    _, data, _ = run_build(tmp_path, monkeypatch, capsys)
    examples = data["candidates"][0]["examples"]
    assert examples == [
        {"snippet_id": "e1", "speaker": "partner", "text": "蛋挞仔你在干嘛"},
        {"snippet_id": "e2", "speaker": "partner", "text": "蛋挞仔晚安"},
    ]


def test_context_turns_are_not_counted(tmp_path, monkeypatch, capsys):
    # Every snippet carries a context line with 蛋挞仔; counts must only see own turns.
    _, data, _ = run_build(tmp_path, monkeypatch, capsys)
    assert data["candidates"][0]["partner_count"] == 2


def test_non_train_events_are_not_used(tmp_path, monkeypatch, capsys):
    _, data, _ = run_build(tmp_path, monkeypatch, capsys)
    assert "卷心菜精" not in json.dumps(data, ensure_ascii=False)


@pytest.mark.parametrize("pool", sorted(st.EXCLUDED_POOLS))
def test_excluded_d14_pool_events_are_dropped(tmp_path, monkeypatch, capsys, pool):
    snippets = [*base_snippets(),
                snip("x1", ["奶黄包包"], ["奶黄包包"], day=1),
                snip("x1b", ["奶黄包包呢"], ["奶黄包包"], day=4)]
    split = split_doc()
    add_train_event(split, "x1", ["x1b"])
    _, data, out = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split,
                             pools={pool: ["x1"], "unassigned": []})
    assert "奶黄包包" not in json.dumps(data, ensure_ascii=False)
    assert out["snippets_used"] == 4


def test_other_d14_pools_are_not_excluded(tmp_path, monkeypatch, capsys):
    snippets = [*base_snippets(),
                snip("x1", ["奶黄包包"], ["奶黄包包"], day=1),
                snip("x1b", ["奶黄包包呢"], ["奶黄包包"], day=4)]
    split = split_doc()
    add_train_event(split, "x1", ["x1b"])
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split,
                           pools={"shape_trial": ["x1"]})
    assert "奶黄包包" in terms_of(data)


def test_excluded_snippet_lines_are_never_parsed_as_snippets(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    split = split_doc()
    add_train_event(split, "x1", [])
    paths = write_inputs(tmp_path, split=split, pools={"dict_screen": ["x1"]})
    with paths.snippets.open("a", encoding="utf-8") as file:
        # Not a valid PairedSnippet: validating it would fail the whole build.
        file.write(json.dumps({"snippet_id": "x1", "partner_turns": 7}) + "\n")
        file.write(json.dumps({"snippet_id": "c9", "partner_turns": 7}) + "\n")
    assert st.main(build_args(paths)) == 0


def test_duplicate_snippet_ids_across_files_count_once(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path)
    holdout = tmp_path / "holdout.jsonl"
    write_snippets(holdout, base_snippets())
    code = st.main([*build_args(paths), "--snippets", str(holdout)])
    assert code == 0
    data = json.loads((paths.out / "candidates.json").read_text(encoding="utf-8"))
    assert data["candidates"][0]["owner_count"] == 2


def test_unknown_event_in_pools_is_rejected(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path, pools={"combo": ["missing-event"]})
    assert st.main(build_args(paths)) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_pools"
    assert not paths.out.exists()


def test_rare_or_single_day_terms_are_not_candidates(tmp_path, monkeypatch, capsys):
    snippets = [
        snip("e1", ["芋泥球球"], ["芋泥球球", "芋泥球球"], day=0),  # 3 uses, one day
        snip("e1b", ["棉花糖人"], ["好"], day=0),
        snip("e2", ["棉花糖人"], ["好"], day=3),  # 2 uses, two days
        snip("e2b", ["你好"], ["嗯"], day=4),
    ]
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets)
    assert data["candidates"] == []


def test_common_words_and_interjections_are_not_candidates(tmp_path, monkeypatch, capsys):
    snippets = [
        snip("e1", ["我们今天吃什么", "哈哈哈哈"], ["不知道呀哈哈哈"], day=0),
        snip("e1b", ["我们今天吃什么", "哈哈哈哈"], ["不知道呀哈哈哈"], day=2),
        snip("e2", ["我们今天吃什么", "哈哈哈哈"], ["不知道呀哈哈哈"], day=4),
        snip("e2b", ["我们今天吃什么"], ["不知道呀"], day=6),
    ]
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets)
    assert data["candidates"] == []


def test_longest_stable_form_wins_over_its_fragments(tmp_path, monkeypatch, capsys):
    _, data, _ = run_build(tmp_path, monkeypatch, capsys)
    assert "蛋挞" not in terms_of(data)
    assert "挞仔" not in terms_of(data)


def test_latin_tokens_are_whole_words(tmp_path, monkeypatch, capsys):
    snippets = [
        snip("e1", ["xswl 笑死"], ["xswl"], day=0),
        snip("e1b", ["XSWL"], ["好"], day=2),
        snip("e2", ["睡了"], ["晚安"], day=3),
        snip("e2b", ["嗯"], ["嗯"], day=4),
    ]
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets)
    assert terms_of(data) == ["xswl"]
    assert data["candidates"][0]["owner_count"] == 1
    assert data["candidates"][0]["partner_count"] == 2


def test_redaction_placeholders_are_ignored(tmp_path, monkeypatch, capsys):
    snippets = [
        snip("e1", ["[手机号]给你"], ["[手机号]收到"], day=0),
        snip("e1b", ["[手机号]"], ["好"], day=2),
        snip("e2", ["[手机号]"], ["嗯"], day=3),
        snip("e2b", ["嗯"], ["嗯"], day=4),
    ]
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets)
    assert data["candidates"] == []


def test_shared_terms_rank_above_one_sided_terms(tmp_path, monkeypatch, capsys):
    snippets = [
        snip("e1", ["小番茄酱"], ["花生米粒"], day=0),
        snip("e1b", ["小番茄酱"], ["花生米粒"], day=2),
        snip("e2", ["花生米粒"], ["小番茄酱"], day=4),
        snip("e2b", ["嗯"], ["嗯"], day=6),
    ]
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets)
    # Same total (3); 花生米粒 vs 小番茄酱 are both shared: ties break by term.
    assert terms_of(data) == sorted(["花生米粒", "小番茄酱"])
    one_sided = [
        snip("e1", ["好"], ["花生米粒", "花生米粒"], day=0),
        snip("e1b", ["小番茄酱"], ["好"], day=2),
        snip("e2", ["好"], ["花生米粒"], day=4),
        snip("e2b", ["嗯"], ["小番茄酱"], day=6),
        snip("e3", ["小番茄酱"], ["嗯"], day=7),
    ]
    split = split_doc()
    add_train_event(split, "e3", [])
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=one_sided, split=split)
    assert terms_of(data) == ["小番茄酱", "花生米粒"]


def test_at_most_thirty_candidates(tmp_path, monkeypatch, capsys):
    chars = "甜酸辣苦咸鲜云雾雪雨星月山川河海风花草木林森石玉金银铜铁松柏竹梅兰菊桃李" \
        "杏梨橙柚瓜豆米麦稻谷鱼虾蟹贝龟鹤鹿虎豹狼熊猫狐兔鸽鹰燕雀蝶蜂蚁蛙蛇羊牛马"
    words = [chars[i:i + 2] for i in range(0, 72, 2)]  # 36 terms sharing no characters
    assert len(set(chars)) == 72
    snippets, split = [], {"events": []}
    for index, word in enumerate(words):
        snippets += [snip(f"w{index}a", [word], [word], day=0),
                     snip(f"w{index}b", [word], ["好"], day=1 + index % 3)]
        split["events"].append(
            {"event_id": f"w{index}a", "pool": "train", "snippet_ids": [f"w{index}b"]})
    _, data, out = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    assert len(data["candidates"]) == 30
    assert out["candidates"] == 30
    assert out["dropped_over_limit"] == 6


# --- general-frequency contrast ---------------------------------------------------------


def contrast_snippets():
    """Ordinary words used most, 乖乖 as often, coined 糯米团/芋圆球 rarely; all by both sides."""
    ordinary = ["寝室呢", "作业呢", "周末呢", "教室呢"]
    rows, split = [], {"events": []}
    for day in range(6):
        rows += [snip(f"o{day}", ordinary, [*ordinary, "乖乖"], day=day),
                 snip(f"o{day}b", ["乖乖"], ["好"], day=day)]
        add_train_event(split, f"o{day}", [f"o{day}b"])
    rows += [snip("n1", ["糯米团"], ["芋圆球"], day=0),
             snip("n1b", ["芋圆球"], ["糯米团"], day=3),
             snip("n2", ["糯米团呀"], ["芋圆球"], day=5)]
    add_train_event(split, "n1", ["n1b"])
    add_train_event(split, "n2", [])
    return rows, split


def test_coined_and_relational_terms_rank_above_ordinary_words(tmp_path, monkeypatch, capsys):
    snippets, split = contrast_snippets()
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    terms = terms_of(data)
    assert terms == ["糯米团", "芋圆球", "乖乖", "寝室", "周末", "教室", "作业"]


def test_ordinary_words_lose_their_slots_to_coined_terms(tmp_path, monkeypatch, capsys):
    snippets, split = contrast_snippets()
    monkeypatch.setattr(st, "MAX_CANDIDATES", 3)
    _, data, out = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    assert set(terms_of(data)) == {"糯米团", "芋圆球", "乖乖"}
    assert out["dropped_over_limit"] == 4


def test_candidates_carry_general_freq(tmp_path, monkeypatch, capsys):
    snippets, split = contrast_snippets()
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    freqs = {c["term"]: c["general_freq"] for c in data["candidates"]}
    assert freqs["糯米团"] == 0
    assert freqs["乖乖"] == 600
    assert freqs["作业"] == 8000


def test_free_combinations_of_loose_parts_are_dropped(tmp_path, monkeypatch, capsys):
    # 去寝室 is not in the general table, but 去 and 寝室 mostly occur apart from it.
    snippets, split = contrast_snippets()
    snippets += [snip("g1", ["去寝室"], ["去寝室", "去哪"], day=1),
                 snip("g1b", ["去寝室"], ["去吧", "去上课", "去呀"], day=4)]
    add_train_event(split, "g1", ["g1b"])
    _, data, out = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    assert "去寝室" not in terms_of(data)
    assert out["dropped_free_combination"] == 1


def test_coined_term_built_from_a_loose_word_is_kept(tmp_path, monkeypatch, capsys):
    # 乖乖 is used on its own a lot, but 猪 only ever appears inside 乖乖猪.
    snippets, split = contrast_snippets()
    snippets += [snip("g1", ["乖乖猪"], ["乖乖猪"], day=1),
                 snip("g1b", ["乖乖猪"], ["好"], day=4)]
    add_train_event(split, "g1", ["g1b"])
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    assert "乖乖猪" in terms_of(data)[:4]


def test_general_dict_is_normalized_and_keeps_highest_freq(tmp_path, monkeypatch, capsys):
    snippets = [
        snip("e1", ["xswl 笑死"], ["xswl"], day=0),
        snip("e1b", ["XSWL"], ["好"], day=2),
        snip("e2", ["睡了"], ["晚安"], day=3),
        snip("e2b", ["嗯"], ["嗯"], day=4),
    ]
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path, snippets=snippets)
    paths.general.write_text("XSWL 7 nz\nxswl 40 nz\n", encoding="utf-8")
    assert st.main(build_args(paths)) == 0
    data = json.loads((paths.out / "candidates.json").read_text(encoding="utf-8"))
    assert data["candidates"][0]["general_freq"] == 40


def test_default_general_dict_is_read_from_installed_jieba(tmp_path, monkeypatch, capsys):
    package = tmp_path / "site" / "jieba"
    package.mkdir(parents=True)
    write_general_dict(package / "dict.txt", {"蛋挞仔": 9})
    monkeypatch.setattr(st, "find_spec", lambda name: SimpleNamespace(
        origin=str(package / "__init__.py")) if name == "jieba" else None)
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path)
    paths.general = None
    assert st.main(build_args(paths)) == 0
    data = json.loads((paths.out / "candidates.json").read_text(encoding="utf-8"))
    assert data["candidates"][0]["general_freq"] == 9


def test_missing_jieba_without_general_dict_is_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(st, "find_spec", lambda name: None)
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path)
    paths.general = None
    capsys.readouterr()
    assert st.main(build_args(paths)) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "general_dict_unavailable"}
    assert not paths.out.exists()


def test_script_does_not_import_jieba_segmentation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import jieba" not in source
    assert "from jieba" not in source


# --- privacy filters ----------------------------------------------------------------------


@pytest.mark.parametrize("term", ["蜜桃臀", "安全期"])  # suspect, high
def test_veiled_terms_are_dropped(tmp_path, monkeypatch, capsys, term):
    snippets = [*base_snippets(),
                snip("e3", [f"{term}啦"], [f"{term}吧"], day=1),
                snip("e3b", [f"今天{term}"], [term], day=3)]
    split = split_doc()
    add_train_event(split, "e3", ["e3b"])
    _, data, out = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    assert term not in json.dumps(data, ensure_ascii=False)
    assert terms_of(data) == ["蛋挞仔"]
    assert out["dropped_veiled"] >= 1


def test_fragments_of_veiled_terms_are_dropped(tmp_path, monkeypatch, capsys):
    snippets = [*base_snippets(),
                snip("e3", ["觉觉了"], ["觉觉"], day=1),
                snip("e3b", ["觉觉"], ["觉觉"], day=3)]
    split = split_doc()
    add_train_event(split, "e3", ["e3b"])
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    assert "觉觉" not in terms_of(data)


def test_examples_skip_lines_with_veiled_terms(tmp_path, monkeypatch, capsys):
    snippets = base_snippets()
    snippets[0] = snip("e1", ["蛋挞仔睡觉觉"], ["在想蛋挞仔"], day=0)
    _, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets)
    texts = [e["text"] for e in data["candidates"][0]["examples"]]
    assert all("睡觉觉" not in text for text in texts)
    assert data["candidates"][0]["partner_count"] == 2


def test_exclude_terms_drop_matches_and_fragments(tmp_path, monkeypatch, capsys):
    snippets = [*base_snippets(),
                snip("e3", ["林小满呢"], ["林小满"], day=1),
                snip("e3b", ["小满"], ["小满"], day=3)]
    split = split_doc()
    add_train_event(split, "e3", ["e3b"])
    _, data, out = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split,
                             exclude=["林小满"])
    dumped = json.dumps(data, ensure_ascii=False)
    assert "小满" not in dumped
    assert terms_of(data) == ["蛋挞仔"]
    assert out["dropped_excluded"] >= 1


def test_exclude_terms_must_be_string_list(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path, exclude={"name": "林小满"})
    assert st.main(build_args(paths)) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_exclude_terms"


def test_status_output_never_contains_terms(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path)
    capsys.readouterr()
    st.main(build_args(paths))
    assert "蛋挞仔" not in capsys.readouterr().out


# --- build outputs --------------------------------------------------------------------------


def test_build_rejects_out_dir_outside_test_results(tmp_path, monkeypatch, capsys):
    stub_settings(tmp_path, monkeypatch)
    paths = write_inputs(tmp_path)
    paths.out = tmp_path / "elsewhere"
    assert st.main(build_args(paths)) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_out_dir"
    assert not paths.out.exists()


def test_review_page_fields_and_escaping(tmp_path, monkeypatch, capsys):
    snippets = base_snippets()
    snippets[0] = snip("e1", ["蛋挞仔<b>&"], ["在想蛋挞仔"], day=0)
    paths, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets)
    page = (paths.out / "review.html").read_text(encoding="utf-8")
    assert "蛋挞仔&lt;b&gt;&amp;" in page
    assert "<b>&" not in page
    section = re.search(r'<section class="term".*?</section>', page, re.DOTALL).group()
    for field in ("meaning", "occasion", "addressee"):
        assert f'name="{field}"' in section
    item_id = data["candidates"][0]["id"]
    assert re.findall(rf'name="speaker-{item_id}" value="(\w+)"', section) == [
        "owner", "partner", "both"]
    for label in ("主人", "她", "双方"):
        assert label in section
    assert re.findall(rf'name="permission-{item_id}" value="(\w+)"', section) == [
        "allowed", "understand_only", "disabled", "delete"]
    for label in ("可用", "仅理解", "停用", "删除"):
        assert label in section
    assert re.search(r'value="understand_only" checked', section)
    assert '<textarea id="export" readonly>' in page
    candidates_sha = hashlib.sha256((paths.out / "candidates.json").read_bytes()).hexdigest()
    assert f'data-candidates="{candidates_sha}"' in page
    assert f'data-id="{data["candidates"][0]["id"]}"' in section


def test_page_export_builder(tmp_path, monkeypatch, capsys):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    paths, _, _ = run_build(tmp_path, monkeypatch, capsys)
    page = (paths.out / "review.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="terms-export">(.*?)</script>', page, re.DOTALL)
    assert match is not None
    rows = [{"id": "t01", "term": "蛋挞仔", "meaning": " 对她的昵称 ", "occasion": "",
             "speaker": "both", "addressee": "她", "permission": "allowed"},
            {"id": "t02", "term": "某词", "meaning": "", "occasion": "", "speaker": "",
             "addressee": "", "permission": "delete"}]
    script = (match.group(1) + "\nprocess.stdout.write(buildExport("
              f'"{"a" * 64}", {json.dumps(rows, ensure_ascii=False)}));\n')
    (tmp_path / "export.js").write_text(script, encoding="utf-8")
    output = subprocess.run([node, str(tmp_path / "export.js")], capture_output=True,
                            text=True, check=True).stdout
    assert json.loads(output) == {
        "version": 1, "candidates_sha256": "a" * 64,
        "entries": [
            {"id": "t01", "term": "蛋挞仔", "meaning": "对她的昵称", "occasion": "",
             "speaker": "both", "addressee": "她", "permission": "allowed"},
            {"id": "t02", "term": "某词", "meaning": "", "occasion": "", "speaker": "",
             "addressee": "", "permission": "delete"},
        ],
    }


def test_script_runs_as_file_without_network_imports(tmp_path):
    source = SCRIPT.read_text(encoding="utf-8")
    for name in ("companion.dialogue", "httpx", "urllib", "requests", "socket"):
        assert name not in source
    result = subprocess.run([sys.executable, str(SCRIPT), "validate"], capture_output=True,
                            text=True, cwd=tmp_path)
    assert result.returncode == 2


# --- validate --------------------------------------------------------------------------------


def two_candidates(tmp_path, monkeypatch, capsys):
    snippets = [*base_snippets(),
                snip("e3", ["奶黄包包"], ["奶黄包包"], day=1),
                snip("e3b", ["奶黄包包呢"], ["好"], day=4)]
    split = split_doc()
    add_train_event(split, "e3", ["e3b"])
    paths, data, _ = run_build(tmp_path, monkeypatch, capsys, snippets=snippets, split=split)
    assert len(data["candidates"]) == 2
    return paths, data


def export_for(paths, data, **overrides):
    entries = []
    for index, item in enumerate(data["candidates"]):
        entry = {"id": item["id"], "term": item["term"], "meaning": f"含义{index}",
                 "occasion": "睡前", "speaker": "both", "addressee": "她",
                 "permission": "understand_only"}
        entry.update(overrides.get(item["id"], {}))
        entries.append(entry)
    sha = hashlib.sha256((paths.out / "candidates.json").read_bytes()).hexdigest()
    return {"version": 1, "candidates_sha256": sha, "entries": entries}


def run_validate(tmp_path, paths, export, capsys, out_dir=None):
    export_path = tmp_path / "export.json"
    export_path.write_text(export if isinstance(export, str)
                           else json.dumps(export, ensure_ascii=False), encoding="utf-8")
    capsys.readouterr()
    code = st.main(["validate", "--export", str(export_path),
                    "--candidates", str(paths.out / "candidates.json"),
                    "--out-dir", str(out_dir or paths.out)])
    return code, json.loads(capsys.readouterr().out)


def test_validate_writes_dictionary_without_examples(tmp_path, monkeypatch, capsys):
    paths, data = two_candidates(tmp_path, monkeypatch, capsys)
    first = data["candidates"][0]["id"]
    export = export_for(paths, data, **{first: {"permission": "allowed", "meaning": " 昵称 "}})
    code, out = run_validate(tmp_path, paths, export, capsys)
    assert code == 0, out
    assert out == {"status": "ok", "entries": 2, "deleted": 0}
    dictionary = json.loads((paths.out / "dictionary.json").read_text(encoding="utf-8"))
    assert dictionary["version"] == 1
    assert dictionary["entries"][0] == {
        "term": data["candidates"][0]["term"], "meaning": "昵称", "occasion": "睡前",
        "speaker": "both", "addressee": "她", "permission": "allowed"}
    dumped = json.dumps(dictionary, ensure_ascii=False)
    for word in ("examples", "snippet_id", "owner_count", "first_seen", "蛋挞仔你在干嘛"):
        assert word not in dumped


def test_validate_drops_deleted_entries(tmp_path, monkeypatch, capsys):
    paths, data = two_candidates(tmp_path, monkeypatch, capsys)
    first = data["candidates"][0]
    export = export_for(paths, data, **{first["id"]: {
        "permission": "delete", "meaning": "", "speaker": "", "occasion": "", "addressee": ""}})
    code, out = run_validate(tmp_path, paths, export, capsys)
    assert code == 0, out
    assert out == {"status": "ok", "entries": 1, "deleted": 1}
    dictionary = json.loads((paths.out / "dictionary.json").read_text(encoding="utf-8"))
    assert [e["term"] for e in dictionary["entries"]] == [data["candidates"][1]["term"]]
    assert first["term"] not in json.dumps(dictionary, ensure_ascii=False)


def _mutate(kind, export):
    entries = export["entries"]
    if kind == "version":
        export["version"] = 2
    elif kind == "sha":
        export["candidates_sha256"] = "0" * 64
    elif kind == "unknown_id":
        entries[0]["id"] = "t99"
    elif kind == "duplicate_id":
        entries[1] = dict(entries[0])
    elif kind == "missing_entry":
        entries.pop()
    elif kind == "term_mismatch":
        entries[0]["term"] = "别的词"
    elif kind == "permission":
        entries[0]["permission"] = "maybe"
    elif kind == "speaker":
        entries[0]["speaker"] = "他"
    elif kind == "blank_meaning":
        entries[0]["meaning"] = "   "
    elif kind == "extra_key":
        entries[0]["examples"] = []
    elif kind == "missing_key":
        del entries[0]["addressee"]
    elif kind == "not_string":
        entries[0]["occasion"] = 3
    elif kind == "too_long":
        entries[0]["meaning"] = "长" * (st.MAX_FIELD_CHARS + 1)
    return export


@pytest.mark.parametrize("kind", [
    "version", "sha", "unknown_id", "duplicate_id", "missing_entry", "term_mismatch",
    "permission", "speaker", "blank_meaning", "extra_key", "missing_key", "not_string",
    "too_long",
])
def test_validate_rejects_bad_export(tmp_path, monkeypatch, capsys, kind):
    paths, data = two_candidates(tmp_path, monkeypatch, capsys)
    code, out = run_validate(tmp_path, paths, _mutate(kind, export_for(paths, data)), capsys)
    assert code == 2
    assert out["status"] == "invalid_export"
    assert not (paths.out / "dictionary.json").exists()


def test_validate_rejects_non_json_export(tmp_path, monkeypatch, capsys):
    paths, _ = two_candidates(tmp_path, monkeypatch, capsys)
    code, out = run_validate(tmp_path, paths, "not json {", capsys)
    assert code == 2
    assert out["status"] == "invalid_export"


def test_validate_rejects_out_dir_outside_test_results(tmp_path, monkeypatch, capsys):
    paths, data = two_candidates(tmp_path, monkeypatch, capsys)
    code, out = run_validate(tmp_path, paths, export_for(paths, data), capsys,
                             out_dir=tmp_path / "elsewhere")
    assert code == 2
    assert out["status"] == "invalid_out_dir"
