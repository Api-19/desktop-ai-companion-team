# ruff: noqa: RUF001
"""Weak-label pipeline tests on synthetic data with a fake backend; never reads ~/.rcpem."""

import csv
import hashlib
import json
from collections import Counter

import pytest

from scripts import g_weak_labels as wl


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, max_tokens):
        self.calls.append(messages)
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)


def write_raw(raw_dir, rows_by_part):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for part in ("train", "dev", "test"):
        lines = [json.dumps(row, ensure_ascii=False) for row in rows_by_part.get(part, [])]
        (raw_dir / f"RawDialogueData__{part}_data.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")


def synthetic_rows(n_streamers=10, per=3):
    return [[f"streamer{s}", f"合成评论第{s}号第{i}条话", f"合成回复内容{s}之{i}"]
            for s in range(n_streamers) for i in range(per)]


# ---------- split ----------

def test_split_reproducible_disjoint_and_sized(tmp_path):
    rows = synthetic_rows(n_streamers=11)
    write_raw(tmp_path / "raw", {"train": rows[:20], "dev": rows[20:27], "test": rows[27:]})
    first = wl.run_split(tmp_path / "raw", tmp_path / "w1")
    second = wl.run_split(tmp_path / "raw", tmp_path / "w2")
    assert (tmp_path / "w1" / "livechat_split.json").read_text() == \
        (tmp_path / "w2" / "livechat_split.json").read_text()
    assert first["seed"] == 20261016
    assert (len(first["train"]), len(first["dev"]), len(first["test"])) == (6, 2, 3)
    parts = [set(first[k]) for k in ("train", "dev", "test")]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set().union(*parts) == {f"streamer{s}" for s in range(11)}
    assert second["pairs_per_split"] == first["pairs_per_split"]
    assert sum(first["pairs_per_split"].values()) == 33


def test_split_normalises_and_drops_exact_duplicates_keeping_first(tmp_path):
    write_raw(tmp_path / "raw", {
        "train": [["streamerA", " 你好  主播 ", "你好呀\t朋友"],
                  ["streamerA", "别的话题评论", "回复"]],
        "dev": [["streamerB", "你好 主播", "你好呀 朋友"]],
        "test": [["streamerB", "你好 主播", "不同的回复"]],
    })
    pairs, raw = wl.load_pairs(tmp_path / "raw")
    assert raw == 4
    assert pairs == [("streamerA", "你好 主播", "你好呀 朋友"),
                     ("streamerA", "别的话题评论", "回复"),
                     ("streamerB", "你好 主播", "不同的回复")]
    result = wl.run_split(tmp_path / "raw", tmp_path / "w")
    assert result["dedup_pairs"] == 3
    assert result["pairs_per_speaker"] == {"streamerA": 2, "streamerB": 1}


# ---------- sample filters ----------

@pytest.mark.parametrize("c,r,reason", [
    ("短评论", "一条正常长度的回复", "c_length"),
    ("这" * 41, "一条正常长度的回复", "c_length"),
    ("一条正常长度评论", "好的", "r_length"),
    ("一条正常长度评论", "回" * 61, "r_length"),
    ("看看这个http链接", "一条正常长度的回复", "url"),
    ("一条正常长度评论", "去WWW点看看吧", "url"),
    ("今天天气真不错呀", "今天天气真不错呀哈", "dice"),
    ("主播你吃饭了吗今天", "主播你吃饭了吗嗯嗯哦", "echo_remainder"),
])
def test_filter_rules(c, r, reason):
    assert wl.reject_reason(c, r) == reason


def test_filter_keeps_echo_with_long_remainder_and_plain_pairs():
    assert wl.strip_echo("主播你吃饭了吗今天", "主播你吃饭了吗，吃过了刚吃完") == "吃过了刚吃完"
    assert wl.strip_echo("主播你吃饭了吗今天", "还没吃呢等下吃") == "还没吃呢等下吃"
    assert wl.reject_reason("主播你吃饭了吗今天", "主播你吃饭了吗，吃过了刚吃完") is None
    assert wl.reject_reason("主播唱首歌吧好不好", "等一下马上就唱") is None


def test_sample_train_only_per_streamer_cap_and_reproducible():
    good = [(f"s{s}", f"合成长评论{s}号第{i}条", f"完全不同回复{i}")
            for s in range(3) for i in range(9)]
    pairs = [*good, ("s9", "合成长评论九号第一条", "完全不同回复")]
    first, stats = wl.sample_sources(pairs, {"s0", "s1", "s2"}, 100)
    second, _ = wl.sample_sources(pairs, {"s0", "s1", "s2"}, 100)
    assert first == second
    assert len(first) == 15
    assert all(sum(row["speaker"] == s for row in first) == 5 for s in ("s0", "s1", "s2"))
    assert {row["speaker"] for row in first} == {"s0", "s1", "s2"}
    assert stats["split_pairs"] == 27 and stats["eligible_pairs"] == 27
    limited, _ = wl.sample_sources(pairs, {"s0", "s1", "s2"}, 4)
    assert len(limited) == 4 and limited == first[:4]
    assert set(first[0]) == {"src_id", "source", "split", "speaker", "show", "c", "r", "r_raw"}


def test_sample_counts_rejections():
    pairs = [("s0", "短", "一条正常长度的回复"), ("s0", "一条正常长度评论", "好的"),
             ("s0", "一条正常长度评论", "好呀那就这样吧")]
    chosen, stats = wl.sample_sources(pairs, {"s0"}, 5)
    assert stats["rejected"] == {"c_length": 1, "r_length": 1}
    assert len(chosen) == 1


def test_sample_cli_excludes_existing_pairs_and_prefixes_ids(tmp_path):
    rows = [["s0", "合成评论内容足够长", "完全不同的回复内容"],
            ["s0", "另一条评论内容足够长", "直播间今天网络很好呀"]]
    write_raw(tmp_path / "raw", {"train": rows})
    root = tmp_path / "weak"
    root.mkdir()
    (root / "livechat_split.json").write_text(json.dumps({"train": ["s0"]}))
    excluded = tmp_path / "old.jsonl"
    excluded.write_text(json.dumps({"c": " 合成评论内容足够长 ",
                                    "r": " 完全不同的回复内容 "}) + "\n")
    assert wl.main(["--root", str(root), "sample", "--raw-dir", str(tmp_path / "raw"),
                    "--n", "10", "--out", "sampled", "--exclude", str(excluded),
                    "--id-prefix", "batch3-"]) == 0
    sources = wl.read_jsonl(root / "sampled" / "sources.jsonl")
    stats = json.loads((root / "sampled" / "sample_summary.json").read_text())
    assert len(sources) == 1 and sources[0]["src_id"] == "batch3-src00000"
    assert sources[0]["c"] == "另一条评论内容足够长"
    assert stats["rejected"] == {"excluded": 1}


def test_sample_cli_uses_requested_livechat_split(tmp_path):
    rows = [["train_speaker", "合成评论内容足够长", "完全不同的回复内容"],
            ["dev_speaker", "另一条评论内容足够长", "直播间今天网络很好呀"]]
    write_raw(tmp_path / "raw", {"train": rows, "dev": rows})
    root = tmp_path / "weak"
    root.mkdir()
    (root / "livechat_split.json").write_text(
        json.dumps({"train": ["train_speaker"], "dev": ["dev_speaker"]}))

    assert wl.main(["--root", str(root), "sample", "--raw-dir", str(tmp_path / "raw"),
                    "--split", "dev", "--n", "10", "--out", "sampled"]) == 0

    sources = wl.read_jsonl(root / "sampled" / "sources.jsonl")
    stats = json.loads((root / "sampled" / "sample_summary.json").read_text())
    assert {row["speaker"] for row in sources} == {"dev_speaker"}
    assert all(row["split"] == "dev" for row in sources)
    assert stats["split_pairs"] == 1


# ---------- generate ----------

SRC = {"source": "livechat", "split": "train", "show": None, "src_id": "src00000",
       "speaker": "s0", "c": "我明天要去考驾照了", "r": "加油，肯定能过"}
GOOD_POS = {"x": "下周我要去面试了", "kind": "cross_topic"}
GOOD_NEG = {"x": "我昨天考完驾照了", "edit": "state"}


def test_generate_valid_output_rows():
    backend = FakeBackend([{"pos": GOOD_POS, "neg": GOOD_NEG}])
    rows, stats = wl.generate_pairs([SRC], backend, "fake/model@0")
    assert len(backend.calls) == 1
    assert "我明天要去考驾照了" in backend.calls[0][1]["content"]
    yes, no = rows
    assert yes["label"] == "yes" and yes["kind"] == "cross_topic" and "edit" not in yes
    assert no["label"] == "no" and no["edit"] == "state" and "kind" not in no
    assert yes["pair_id"] == "src00000-yes" and no["pair_id"] == "src00000-no"
    assert set(yes) == {"pair_id", "src_id", "source", "show", "speaker", "split",
                        "c", "r", "x", "label", "kind", "gen_model", "prompt_version",
                        "prompt_sha256"}
    assert yes["split"] == "train" and yes["prompt_sha256"] == wl.PROMPT_SHA256
    assert stats["kept"] == {"yes": 1, "no": 1}


@pytest.mark.parametrize("pos,neg,yes_reason,no_reason", [
    ({"x": "下周我要去面试了"}, {"edit": "state"}, "missing_key", "missing_key"),
    ({"x": "下周我要去面试了", "kind": "other"}, {"x": "我昨天考完驾照了", "edit": "tone"},
     "bad_kind", "bad_edit"),
    ({"x": "  ", "kind": "paraphrase"}, {"x": "", "edit": "need"}, "empty_x", "empty_x"),
    ({"x": "长" * 61, "kind": "paraphrase"}, {"x": "长" * 61, "edit": "need"},
     "x_too_long", "x_too_long"),
    ({"x": "我明天要去考驾照了", "kind": "paraphrase"}, {"x": "我明天要去考驾照了", "edit": "need"},
     "x_equals_c", "x_equals_c"),
    ({"x": "加油，肯定能过", "kind": "paraphrase"}, {"x": "加油，肯定能过", "edit": "need"},
     "x_equals_r", "x_equals_r"),
    (GOOD_POS, {"x": "主播你喜欢什么颜色的衣服", "edit": "person"}, None, "edit_too_large"),
    (GOOD_POS, {"x": SRC["c"] + "啦", "edit": "state"}, None, "append_only"),
    ({"x": "我明天要去考驾照了呀", "kind": "paraphrase"}, GOOD_NEG,
     "near_copy", None),
])
def test_generate_drop_rules(pos, neg, yes_reason, no_reason):
    rows, stats = wl.generate_pairs([SRC], FakeBackend([{"pos": pos, "neg": neg}]), "m")
    labels = {row["label"] for row in rows}
    if yes_reason:
        assert "yes" not in labels and stats["dropped"]["yes"] == {yes_reason: 1}
    else:
        assert "yes" in labels and stats["dropped"]["yes"] == {}
    if no_reason:
        assert "no" not in labels and stats["dropped"]["no"] == {no_reason: 1}
    else:
        assert "no" in labels and stats["dropped"]["no"] == {}


def test_generate_edit_distance_bound_uses_half_length():
    src = {**SRC, "c": "主播今天晚上会唱好几首歌吗我们都在等"}  # len 18 → bound 9
    near = {"x": "主播今天晚上别唱歌了我们都在等", "edit": "need"}
    rows, _ = wl.generate_pairs([src], FakeBackend([{"pos": GOOD_POS, "neg": near}]), "m")
    assert any(row["label"] == "no" for row in rows)
    assert wl.edit_distance("abc", "abd") == 1 and wl.edit_distance("", "ab") == 2


def test_generate_unparseable_missing_side_and_backend_error():
    outputs = ["不是 JSON", RuntimeError("boom"), json.dumps({"pos": GOOD_POS}), "[1, 2]"]
    rows, stats = wl.generate_pairs([SRC] * 4, FakeBackend(outputs), "m")
    assert [row["label"] for row in rows] == ["yes"]
    assert stats["dropped"]["yes"] == {"json_error": 2, "generate_error": 1}
    assert stats["dropped"]["no"] == {"json_error": 2, "generate_error": 1, "missing_key": 1}


# ---------- audit ----------

def make_pairs(n_yes, n_no):
    rows = []
    for i in range(n_yes):
        rows.append({"pair_id": f"y{i}", "label": "yes", "kind": "paraphrase",
                     "c": f"c{i}", "r": f"r{i}", "x": f"xy{i}", "speaker": "s"})
    for i in range(n_no):
        rows.append({"pair_id": f"n{i}", "label": "no", "edit": "need",
                     "c": f"c{i}", "r": f"r{i}", "x": f"xn{i}", "speaker": "s"})
    return rows


def test_audit_balanced_blind_and_reproducible():
    items, key = wl.build_audit(make_pairs(20, 15), 10)
    again, _ = wl.build_audit(make_pairs(20, 15), 10)
    assert items == again
    assert len(items) == 10
    assert all(set(item) == {"item_id", "c", "r", "x"} for item in items)
    labels = [v["label"] for v in key.values()]
    assert labels.count("yes") == 5 and labels.count("no") == 5
    assert set(key) == {item["item_id"] for item in items}
    assert all("pair_id" in v for v in key.values())


def test_audit_takes_all_when_short():
    items, key = wl.build_audit(make_pairs(2, 9), 10)
    labels = [v["label"] for v in key.values()]
    assert labels.count("yes") == 2 and labels.count("no") == 5 and len(items) == 7


def test_cli_end_to_end_without_model(tmp_path, monkeypatch):
    rows = [[f"streamer{s}", f"合成长评论{s}号第{i}条", f"完全不同的回复{i}"]
            for s in range(10) for i in range(3)]
    write_raw(tmp_path / "raw", {"train": rows})
    root = tmp_path / "weak"
    assert wl.main(["--root", str(root), "split", "--raw-dir", str(tmp_path / "raw")]) == 0
    assert wl.main(["--root", str(root), "sample", "--raw-dir", str(tmp_path / "raw"),
                    "--n", "4", "--out", "smoke"]) == 0
    sources = wl.read_jsonl(root / "smoke" / "sources.jsonl")
    split = json.loads((root / "livechat_split.json").read_text())
    assert len(sources) == 4 and all(s["speaker"] in split["train"] for s in sources)

    class Backend(FakeBackend):
        def __init__(self, path):
            super().__init__([{"pos": GOOD_POS,
                               "neg": {"x": "啊" + s["c"][1:], "edit": "state"}}
                              for s in sources])

    monkeypatch.setattr(wl.llm, "MlxBackend", Backend)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert wl.main(["--root", str(root), "generate", "--out", "smoke"]) == 0
    assert len(wl.read_jsonl(root / "smoke" / "pairs.jsonl")) == 8
    assert wl.main(["--root", str(root), "audit", "--out", "smoke", "--n", "4"]) == 0
    assert len(wl.read_jsonl(root / "smoke" / "audit_items.jsonl")) == 4
    with pytest.raises(SystemExit):
        wl.main(["--root", str(root), "audit", "--out", "../escape", "--n", "4"])


@pytest.mark.parametrize("c,r,expected", [
    ("今天你吃饭了吗", "你今天吃饭了吗？已经吃过了！准备开播", "已经吃过了，准备开播"),
    ("今天你吃饭了吗", "等会再吃。先聊会天", "等会再吃，先聊会天"),
    ("今天你吃饭了吗", "你今天吃饭了吗，嗯", "嗯"),
    ("甲乙丙丁", "甲乙戊己，剩下四字", "剩下四字"),
])
def test_fuzzy_echo(c, r, expected):
    assert wl.strip_echo(c, r) == expected


def test_sample_stores_stripped_and_raw_reply():
    raw = "你今天吃饭了吗？已经吃过了！准备开播"
    rows, _ = wl.sample_sources([("s", "今天你吃饭了吗", raw)], {"s"}, 1)
    assert rows == [{"src_id": "src00000", "source": "livechat", "split": "train",
                     "speaker": "s", "show": None, "c": "今天你吃饭了吗",
                     "r": "已经吃过了，准备开播", "r_raw": raw}]
    assert wl.reject_reason("今天你吃饭了吗晚上", "你今天吃饭了吗，嗯") == "echo_remainder"


def write_cped(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "TV_ID", "Dialogue_ID", "Utterance_ID", "Speaker", "DA", "Utterance"])
        writer.writeheader()
        writer.writerows(rows)


def utterance(dialogue, index, speaker, text, da="inform", show="剧一"):
    return dict(TV_ID=show, Dialogue_ID=str(dialogue), Utterance_ID=str(index),
                Speaker=speaker, DA=da, Utterance=text)


def cped_cli(tmp_path, rows, other=(), n=100, out="cped", split="train",
             exclude=(), id_prefix=""):
    paths = [tmp_path / f"{name}.csv" for name in ("train", "valid", "test")]
    write_cped(paths[0], rows)
    write_cped(paths[1], other)
    write_cped(paths[2], [])
    root = tmp_path / "weak"
    argv = ["--root", str(root), "cped-sources", "--csv", str(paths[0]),
                    "--split", split, "--all-csv", *map(str, paths),
                    "--n", str(n), "--out", out, "--id-prefix", id_prefix]
    for path in exclude:
        argv.extend(["--exclude", str(path)])
    assert wl.main(argv) == 0
    return (wl.read_jsonl(root / out / "sources.jsonl"),
            json.loads((root / out / "sample_summary.json").read_text()))


def test_cped_merges_sorts_truncates_and_preserves_metadata(tmp_path):
    rows = [utterance(1, i, "甲" if i < 10 else "乙", f"合成台词第{i}句")
            for i in (12, 2, 11, 1, 13, 3, 10)]
    sources, stats = cped_cli(tmp_path, rows, split="valid")
    assert sources == [{"src_id": "cped-valid-00000", "source": "cped",
                        "split": "valid", "speaker": "乙", "show": "剧一",
                        "c": "合成台词第2句／合成台词第3句",
                        "r": "合成台词第10句／合成台词第11句／合成台词第12句"}]
    assert stats["eligible_pairs"] == stats["sampled"] == 1
    assert stats["speakers_sampled"] == stats["shows_sampled"] == 1


def test_cped_excludes_normalized_pairs_and_prefixes_ids(tmp_path):
    rows = [utterance(1, 1, "甲", "合成评论内容足够长"),
            utterance(1, 2, "乙", "合成回复内容足够长"),
            utterance(2, 1, "丙", "另一条评论内容足够长"),
            utterance(2, 2, "丁", "另一条回复内容足够长")]
    excluded = tmp_path / "old.jsonl"
    excluded.write_text(json.dumps({"c": " 合成评论内容足够长 ",
                                    "r": " 合成回复内容足够长 "}) + "\n")
    extra_excluded = tmp_path / "more-old.jsonl"
    extra_excluded.write_text(json.dumps({"c": "无匹配评论", "r": "无匹配回复"}) + "\n")
    sources, stats = cped_cli(tmp_path, rows, exclude=[excluded, extra_excluded],
                              id_prefix="batch2-")
    assert len(sources) == 1
    assert sources[0]["c"] == "另一条评论内容足够长"
    assert sources[0]["src_id"] == "batch2-cped-train-00000"
    assert stats["rejected"]["excluded"] == 1


def test_cped_rejections_and_dialogue_boundaries(tmp_path):
    cases = [
        ("短", "正常合成回复", "inform", "甲", "乙"),
        ("字" * 61, "正常合成回复", "inform", "甲", "乙"),
        ("正常合成评论", "短", "inform", "甲", "乙"),
        ("正常合成评论", "字" * 81, "inform", "甲", "乙"),
        ("看看HTTP链接", "正常合成回复", "inform", "甲", "乙"),
        ("正常合成评论", "看看WWW链接", "inform", "甲", "乙"),
        ("正常合成评论", "你好合成回复", "greeting", "甲", "乙"),
        ("正常合成评论", "正常合成回复", "inform", "共享", "乙"),
        ("正常合成评论", "正常合成回复", "inform", "甲", "共享"),
    ]
    rows = []
    for i, (c, r, da, a, b) in enumerate(cases):
        rows += [utterance(i, 1, a, c), utterance(i, 2, b, r, da)]
        if da == "greeting":
            rows.append(utterance(i, 3, b, "好的合成回应", "acknowledge"))
    sources, stats = cped_cli(tmp_path, rows, [utterance(99, 1, "共享", "其他切分台词")])
    assert sources == []
    assert stats["rejected"] == {
        "c_length": 2, "r_length": 2, "url": 2, "da_trivial": 1, "shared_speaker": 2}
    assert stats["eligible_pairs"] == stats["sampled"] == 0


def test_cped_cap_reproducibility_and_mixed_da(tmp_path):
    rows = []
    for i in range(18):
        rows += [utterance(i, 1, "甲", f"合成评论第{i}条"),
                 utterance(i, 2, f"乙{i % 2}", "你好合成回复", "greeting"),
                 utterance(i, 3, f"乙{i % 2}", "回答实质内容", "inform")]
    first, stats = cped_cli(tmp_path, rows)
    second, _ = cped_cli(tmp_path, list(reversed(rows)), out="again")
    assert first == second
    assert Counter(row["speaker"] for row in first) == {"乙0": 5, "乙1": 5}
    assert stats["eligible_pairs"] == 18
    limited, _ = cped_cli(tmp_path, rows, n=3, out="limited")
    assert limited == first[:3]


def test_import_cli_validation_counts_and_prompt_bytes(tmp_path):
    src = {**SRC, "src_id": "cped-test-00000", "source": "cped",
           "split": "test", "show": "合成剧", "c": SRC["c"] + "／有些紧张"}
    sources = tmp_path / "sources.jsonl"
    gen = tmp_path / "gen.jsonl"
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes("合成提示词".encode() + bytes([13, 10]))
    wl.write_jsonl(sources, [src, SRC, {**SRC, "src_id": "missing"}])
    wl.write_jsonl(gen, [
        {"src_id": "unknown", "pos": GOOD_POS},
        {"src_id": src["src_id"], "pos": GOOD_POS,
             "neg": {"x": "啊" + src["c"][1:], "edit": "state"}, "extra": "ignored"},
        {"src_id": src["src_id"], "pos": {"x": "", "kind": "bad"}},
        {"src_id": SRC["src_id"], "pos": {"x": SRC["c"], "kind": "paraphrase"},
         "neg": {"x": "完全无关而且编辑距离太大的句子内容", "edit": "need"}},
    ])
    root = tmp_path / "weak"
    assert wl.main(["--root", str(root), "import", "--sources", str(sources),
                    "--gen", str(gen), "--prompt-file", str(prompt),
                    "--gen-model", "external/codex", "--out", "imported"]) == 0
    rows = wl.read_jsonl(root / "imported/pairs.jsonl")
    stats = json.loads((root / "imported/import_summary.json").read_text())
    assert stats["unknown_src"] == stats["duplicate_src"] == stats["missing_gen"] == 1
    assert stats["kept"] == {"yes": 1, "no": 1}
    assert stats["dropped"] == {"yes": {"x_equals_c": 1}, "no": {"edit_too_large": 1}}
    assert len(rows) == 2
    for row in rows:
        assert row["prompt_version"] == "import-v1"
        assert row["prompt_sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
        assert row["gen_model"] == "external/codex"
        assert row["split"] == "test" and row["source"] == "cped"
        assert row["show"] == "合成剧" and row["speaker"] == src["speaker"]
        assert row["c"] == src["c"]


def test_cped_length_boundaries_and_da_uses_entire_turn(tmp_path):
    rows = [
        utterance(1, 1, "甲", "甲" * 29),
        utterance(1, 2, "甲", "乙" * 30),
        utterance(1, 3, "乙", "丙" * 26, "greeting"),
        utterance(1, 4, "乙", "丁" * 26, "acknowledge"),
        utterance(1, 5, "乙", "戊" * 26, "greeting"),
        utterance(1, 6, "乙", "实质回复在第四句", "inform"),
        utterance(2, 1, "甲", "四字评论", show="剧二"),
        utterance(2, 2, "乙", "四字回复", show="剧二"),
    ]
    sources, stats = cped_cli(tmp_path, rows)
    assert sorted((len(row["c"]), len(row["r"])) for row in sources) == [(4, 4), (60, 80)]
    assert stats["shows_sampled"] == 2


def test_import_invalid_first_row_is_still_first_and_not_missing(tmp_path):
    sources, gen, prompt = (tmp_path / name for name in ("sources", "gen", "prompt"))
    wl.write_jsonl(sources, [SRC])
    wl.write_jsonl(gen, [{"src_id": SRC["src_id"]},
                        {"src_id": SRC["src_id"], "pos": GOOD_POS, "neg": GOOD_NEG}])
    prompt.write_bytes(b"synthetic")
    stats = wl.run_import(sources, gen, prompt, "fake", tmp_path / "out")
    assert stats["missing_gen"] == 0 and stats["duplicate_src"] == 1
    assert stats["kept"] == {}
    assert stats["dropped"] == {"yes": {"missing_key": 1}, "no": {"missing_key": 1}}


V3 = {
    "pos_near": {"x": "abcdefghijZl", "kind": "near"},
    "pos_far": {"x": "uvwxyz", "kind": "cross_topic"},
    "neg_near": {"x": "abcdXfghijkl", "edit": "state"},
    "neg_far": {"x": "mnopqr", "edit": "need"},
}
V3_SRC = {**SRC, "c": "abcdefghijkl", "r": "synthetic reply"}


def import_synthetic(tmp_path, data, src=V3_SRC):
    sources, gen, prompt = (tmp_path / name for name in ("sources", "gen", "prompt"))
    wl.write_jsonl(sources, [src])
    wl.write_jsonl(gen, [{"src_id": src["src_id"], **data}])
    prompt.write_bytes(b"synthetic prompt\r\n")
    stats = wl.run_import(sources, gen, prompt, "fake", tmp_path / "out")
    return wl.read_jsonl(tmp_path / "out/pairs.jsonl"), stats


@pytest.mark.parametrize("kind", wl.KINDS)
def test_import_v3_four_sides_and_metadata(tmp_path, kind):
    data = {**V3, "pos_far": {**V3["pos_far"], "kind": kind}}
    rows, stats = import_synthetic(tmp_path, data)
    assert stats["kept"] == {"yes": 2, "no": 2}
    assert stats["dropped"] == {"yes": {}, "no": {}}
    assert stats["by_variant"] == {
        side: {"kept": 1, "dropped": {}} for side in V3}
    for row, side in zip(rows, V3, strict=True):
        field = "kind" if side.startswith("pos") else "edit"
        assert row == {
            **V3_SRC, "pair_id": f"{SRC['src_id']}-{side}", "variant": side,
            "x": data[side]["x"], field: data[side][field],
            "label": "yes" if side.startswith("pos") else "no",
            "gen_model": "fake", "prompt_version": "import-v1",
            "prompt_sha256": hashlib.sha256(b"synthetic prompt\r\n").hexdigest(),
        }


@pytest.mark.parametrize("side", V3)
@pytest.mark.parametrize("fault", ["missing", "bad_tag", "empty", "long", "equals_c",
                                  "equals_r", "overlap", "append"])
def test_import_v3_rejections_are_independent(tmp_path, side, fault):
    data = {key: dict(value) for key, value in V3.items()}
    near = side.endswith("near")
    field = "kind" if side.startswith("pos") else "edit"
    reason = {
        "missing": "missing_key", "bad_tag": f"bad_{field}", "empty": "empty_x",
        "long": "x_too_long", "equals_c": "x_equals_c", "equals_r": "x_equals_r",
        "overlap": "edit_too_large" if near else "not_far",
        "append": "append_only" if near else "not_far",
    }[fault]
    if fault == "missing":
        # pos_near must remain present to select v3, even when its value is missing.
        if side == "pos_near":
            data[side] = None
        else:
            del data[side]
    elif fault == "bad_tag":
        data[side][field] = "invalid"
    else:
        data[side]["x"] = {
            "empty": "  ", "long": "z" * 61, "equals_c": V3_SRC["c"],
            "equals_r": V3_SRC["r"],
            "overlap": "z" * 12 if near else V3["pos_near"]["x"],
            "append": V3_SRC["c"] + "z",
        }[fault]
    rows, stats = import_synthetic(tmp_path, data)
    label = "yes" if side.startswith("pos") else "no"
    assert len(rows) == 3 and side not in {row["variant"] for row in rows}
    assert stats["dropped"] == {label: {reason: 1},
                                "no" if label == "yes" else "yes": {}}
    assert stats["by_variant"] == {
        key: {"kept": 0, "dropped": {reason: 1}} if key == side
        else {"kept": 1, "dropped": {}} for key in V3}


def test_import_v1_exact_compatibility(tmp_path):
    rows, stats = import_synthetic(tmp_path, {"pos": GOOD_POS, "neg": GOOD_NEG}, SRC)
    assert stats == {"kept": {"yes": 1, "no": 1}, "dropped": {"yes": {}, "no": {}},
                     "unknown_src": 0, "duplicate_src": 0, "missing_gen": 0}
    for row, label, side in zip(rows, ("yes", "no"), (GOOD_POS, GOOD_NEG), strict=True):
        assert row == {
            **SRC, **side, "label": label, "pair_id": f"{SRC['src_id']}-{label}",
            "gen_model": "fake", "prompt_version": "import-v1",
            "prompt_sha256": hashlib.sha256(b"synthetic prompt\r\n").hexdigest(),
        }


@pytest.mark.parametrize("side", ["pos_near", "neg_near"])
@pytest.mark.parametrize("c,x,accepted", [
    ("abcde", "zzzze", True),  # distance 4: the minimum bound
    ("abcde", "zzzzz", False),
    ("abcdefghijkl", "zzzzzzghijkl", True),  # distance 6: half the length
    ("abcdefghijkl", "zzzzzzzhijkl", False),
])
def test_import_v3_near_distance_boundaries(tmp_path, side, c, x, accepted):
    data = {key: dict(value) for key, value in V3.items()}
    data[side]["x"] = x
    rows, stats = import_synthetic(tmp_path, data, {**V3_SRC, "c": c})
    assert (side in {row["variant"] for row in rows}) == accepted
    assert stats["by_variant"][side] == {
        "kept": int(accepted), "dropped": {} if accepted else {"edit_too_large": 1}}


@pytest.mark.parametrize("side", ["pos_far", "neg_far"])
@pytest.mark.parametrize("x,accepted", [("abczz", False), ("abzzz", True)])
def test_import_v3_far_dice_boundary(tmp_path, side, x, accepted):
    data = {key: dict(value) for key, value in V3.items()}
    data[side]["x"] = x
    rows, stats = import_synthetic(tmp_path, data, {**V3_SRC, "c": "abcde"})
    assert (side in {row["variant"] for row in rows}) == accepted
    assert stats["by_variant"][side] == {
        "kept": int(accepted), "dropped": {} if accepted else {"not_far": 1}}


def test_import_mixed_versions_aggregate_labels_and_variants(tmp_path):
    sources, gen, prompt = (tmp_path / name for name in ("sources", "gen", "prompt"))
    wl.write_jsonl(sources, [V3_SRC, {**V3_SRC, "src_id": "second"},
                              {**SRC, "src_id": "legacy"}])
    bad = {key: {**value, "x": ""} for key, value in V3.items()}
    wl.write_jsonl(gen, [{"src_id": V3_SRC["src_id"], **bad},
                        {"src_id": "second", **bad},
                        {"src_id": "legacy", "pos": GOOD_POS, "neg": GOOD_NEG},
                        {"src_id": V3_SRC["src_id"], **V3},
                        {"src_id": "unknown", **V3}])
    prompt.write_bytes(b"synthetic")
    stats = wl.run_import(sources, gen, prompt, "fake", tmp_path / "out")
    assert stats == {
        "kept": {"yes": 1, "no": 1},
        "dropped": {"yes": {"empty_x": 4}, "no": {"empty_x": 4}},
        "by_variant": {side: {"kept": 0, "dropped": {"empty_x": 2}} for side in V3},
        "duplicate_src": 1, "unknown_src": 1, "missing_gen": 0,
    }
