"""Offline gate contract tests: synthetic strings and injected encoders only."""

import copy
import importlib
import json
import math
import stat
import subprocess
import sys
from pathlib import Path
from statistics import median

import pytest

from scripts import g_gate as gate
from scripts import g_gate_embed as embed

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures/synthetic/g_gate_minimal_pairs_v1.jsonl"


def event(event_id="e", split="test", apps=("yes",), **overrides):
    item = {
        "event_id": event_id, "split": split, "context": ["合成上下文"],
        "candidates": [
            {"cand_id": f"c{i}", "source": "similar" if i % 2 == 0 else "general",
             "c": ["合成历史"], "r": ["合成答复"]}
            for i in range(len(apps))
        ],
        "labels": {f"c{i}": {"app": app} for i, app in enumerate(apps)},
    }
    item.update(overrides)
    return item


def write_jsonl(path, items):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
    return path


def scored(event_id, score, app="yes", **overrides):
    row = {
        "event_id": event_id, "cand_id": "c0", "split": "cal", "source": "similar",
        "score": score, "app": app, "rule_conflict": False,
    }
    row.update(overrides)
    return row


def fitted_rows():
    return [scored(f"lo{i}", 0, "yes" if i == 0 else "no") for i in range(5)] + [
        scored(f"hi{i}", 1) for i in range(5)
    ]


def test_norm_dice_and_longest_common_substring_hand_calculations():
    assert gate.norm(" \uff21\t\uff22\n\u3000\uff01🙂 ") == "AB!🙂"
    assert gate.norm_lines(["\uff21 ", "\t\uff22🙂"]) == "AB🙂"
    assert gate.dice("aaaa", "aa") == pytest.approx(0.5)  # min(3, 1), not sets
    assert gate.dice("ababa", "aba") == pytest.approx(2 / 3)
    assert gate.dice("a", "a") == 0
    assert gate.dice("", "abc") == 0
    assert gate.lcs("abcXYZdef", "qXYZr") == 3
    assert gate.lcs("abcd", "acbd") == 1  # contiguous, not subsequence
    assert gate.lcs("", "abc") == 0


def test_rule_conflict_overrides_all_scores_and_synthetic_decisions():
    item = event(context=["今晚不想睡", "还要赶报告"])
    item["candidates"][0]["r"] = ["好呀 早点睡", "安安"]
    for scorer in ("char", "length", "vector"):
        rows = gate.score_candidates([item], scorer, {("e", "c0"): 1})
        assert rows[0]["rule_conflict"] is True
        calibration = {"cutoff": -100, "calibration_status": "ok",
                       "bins": {"high": {"p_hat": 1}, "low": {"p_hat": 1}}}
        assert gate.apply_gate(rows, calibration)[0]["keep"] is False
    assert not gate.rule_conflict("今天要睡了", "安安")
    for phrase in ("不想睡", "不睡", "睡不着", "还要赶", "熬夜赶"):
        assert gate.rule_conflict(phrase, "快睡")


def test_length_is_negative_distance_from_file_wide_median():
    items = [event("a"), event("b"), event("c")]
    for item, reply in zip(items, ("a", "abc", "abcdef"), strict=True):
        item["candidates"][0]["r"] = [reply]
    assert [r["score"] for r in gate.score_candidates(items, "length")] == [-2, 0, -3]
    assert gate.score_candidates([], "length") == []


def test_calibration_dev_cutoff_smoothing_and_small_bins():
    rows = [*fitted_rows(), scored("dev", 0.75, split="dev")]
    cal = gate.calibrate(rows)
    assert cal["cutoff"] == 0.75 and cal["cutoff_source"] == "dev"
    assert cal["bins"]["high"] == {"yes": 5, "no": 0, "n": 5, "p_hat": 6 / 7}
    assert cal["bins"]["low"] == {"yes": 1, "no": 4, "n": 5, "p_hat": 2 / 7}
    assert cal["calibration_status"] == "ok"
    small = gate.calibrate(rows[1:])
    assert small["bins"]["low"]["n"] == 4
    assert small["bins"]["low"]["p_hat"] is None
    fallback = gate.calibrate(fitted_rows())
    assert fallback["cutoff"] == 0.5 and fallback["cutoff_source"] == "cal"
    empty = gate.calibrate([])
    assert empty["cutoff"] is None
    assert all(b["p_hat"] is None for b in empty["bins"].values())


def test_non_monotonic_calibration_drops_every_candidate():
    rows = [scored(f"l{i}", 0) for i in range(5)] + [
        scored(f"h{i}", 1, "no") for i in range(5)
    ]
    cal = gate.calibrate(rows)
    assert cal["calibration_status"] == "non_monotonic"
    assert all(not row["keep"] for row in gate.apply_gate(rows, cal, threshold=0))


def test_default_threshold_bin_boundary_and_unknown_not_fit():
    rows = [*fitted_rows(), scored("unknown", 0.5, "unknown")]
    cal = gate.calibrate(rows)
    out = gate.apply_gate(rows, cal)
    assert out[-1]["bin"] == "high" and out[-1]["keep"] is True
    assert sum(b["n"] for b in cal["bins"].values()) == 10
    assert not gate.apply_gate(rows, cal, threshold=0.9)[-1]["keep"]


def metric_rows():
    # TP=1, FP=1, TN=1, FN=1. One additional unknown, one unestimated yes.
    return [
        scored("a", 1, "yes", split="test", keep=True, p_hat=0.8, bin="high"),
        scored("b", 1, "no", split="test", keep=True, p_hat=0.8, bin="high"),
        scored("c", 0, "no", split="test", keep=False, p_hat=0.2, bin="low",
               source="general", rule_conflict=True),
        scored("d", 0, "yes", split="test", keep=False, p_hat=0.2, bin="low",
               source="general"),
        scored("u", 0, "unknown", split="test", keep=True, p_hat=0.9, bin="high"),
    ]


def test_metrics_confusion_rates_wilson_brier_ece_and_unknown():
    rows = metric_rows()
    result = gate.evaluate(rows)
    assert result["confusion"] == {"TP": 1, "FP": 1, "TN": 1, "FN": 1}
    for name in ("false_discovery_rate", "inapplicable_pass_rate", "miss_rate"):
        assert result["rates"][name]["value"] == 0.5
        assert result["rates"][name]["numerator"] == 1
        assert result["rates"][name]["denominator"] == 2
        assert result["rates"][name]["wilson95"] == pytest.approx(
            [0.09453120573423074, 0.9054687942657693]
        )
    rates = result["rates"]
    assert rates["retention_rate"]["numerator"] == 2
    assert rates["retention_rate"]["denominator"] == 4
    assert rates["empty_event_rate"]["value"] == 0.5
    assert rates["unknown_rate"]["value"] == 1 / 5
    assert rates["rule_rejection_rate"]["value"] == 1 / 4
    assert result["brier"] == {"value": pytest.approx(0.34), "n": 4, "excluded": 0}
    assert result["ece"]["value"] == pytest.approx(0.3)
    assert result["ece"]["n"] == 4
    assert result["ece"]["bins"]["high"]["n"] == 2
    groups = gate.evaluate_by_source(rows)
    assert groups["similar"]["confusion"] == {"TP": 1, "FP": 1, "TN": 0, "FN": 0}
    assert groups["general"]["rates"]["retention_rate"]["value"] == 0
    rows.append(scored("x", 0, split="test", keep=False, p_hat=None, bin="low"))
    assert gate.evaluate(rows)["brier"]["excluded"] == 1
    assert gate.evaluate(rows)["ece"]["excluded"] == 1
    assert gate.evaluate([])["rates"]["miss_rate"] == {
        "value": None, "numerator": 0, "denominator": 0, "wilson95": None,
    }
    assert gate.evaluate([])["brier"]["value"] is None


def test_wilson_extreme_counts():
    assert gate.rate(0, 5)["wilson95"] == pytest.approx([0, 0.43448246478317476])
    assert gate.rate(5, 5)["wilson95"] == pytest.approx([0.5655175352168252, 1])


def test_random_same_count_non_conflicting_reproducible_and_all_metrics_summarized():
    rows = []
    for eid, count in (("a", 2), ("b", 1), ("empty", 0)):
        for i in range(6):
            rows.append(scored(eid, i, "yes" if i % 2 else "no", split="test",
                               cand_id=f"c{i}", keep=i < count, rule_conflict=i == 5,
                               p_hat=0.8, bin="high"))
    for seed in range(20):
        selected = gate.random_decisions(rows, seed)
        assert selected == gate.random_decisions(rows, seed)
        assert all(not row["keep"] for row in selected if row["rule_conflict"])
        for eid in ("a", "b", "empty"):
            assert sum(r["keep"] for r in selected if r["event_id"] == eid) == sum(
                r["keep"] for r in rows if r["event_id"] == eid
            )
    assert gate.random_decisions(rows, 0) != gate.random_decisions(rows, 1)
    report = gate.random_baseline(rows)
    assert report["seeds"] == list(range(20))
    summary = report["metrics"]["rates"]["retention_rate"]["value"]
    assert summary == {"mean": 3 / 18, "min": 3 / 18, "max": 3 / 18, "n": 20}
    fp_values = [gate.evaluate(gate.random_decisions(rows, s))["confusion"]["FP"]
                 for s in range(20)]
    assert report["metrics"]["confusion"]["FP"] == {
        "mean": sum(fp_values) / 20, "min": min(fp_values), "max": max(fp_values), "n": 20,
    }
    assert set(report["by_source"]) == {"similar", "general"}
    assert report["metrics"]["brier"]["value"]["n"] == 20


def test_synthetic_fixture_reports_twelve_groups_per_available_scorer():
    events = gate.load_events(FIXTURE)
    scores = {(e["event_id"], c["cand_id"]): i / 24
              for i, e in enumerate(events) for c in e["candidates"]}
    report = gate.analyze(events, "vector", scores=scores)
    assert len(events) == 24
    assert set(report["synthetic"]) == {"char", "length", "vector"}
    for scorer, diagnostic in report["synthetic"].items():
        assert diagnostic["total_groups"] == 12
        assert len(diagnostic["groups"]) == 12
        assert diagnostic["passed"] == (diagnostic["correct_groups"] >= 10)
        assert diagnostic["threshold"] == pytest.approx(
            median(r["score"] for r in gate.score_candidates(events, scorer, scores))
        )
        assert diagnostic["groups"][0]["s1_keep"] in (True, False)
        assert diagnostic["groups"][0]["s2_keep"] is False  # frozen rule wins


def test_synthetic_group_expectations_use_scene_ids_not_input_order():
    rows = []
    for group, kind, scores in (
        ("a", "same_topic_opposite", (1, 0)),
        ("b", "diff_topic_same_action", (1, 1)),
        ("c", "overlap_wrong_action", (0, 0)),
    ):
        for scene, score in enumerate(scores, 1):
            rows.append(scored(f"{group}-s{scene}", score, split="synthetic", group=group,
                               group_kind=kind))
    result = gate.synthetic_diagnostics(list(reversed(rows)))
    assert result["threshold"] == 0.5
    assert result["correct_groups"] == 3
    assert all(group["correct"] for group in result["groups"])


@pytest.mark.parametrize(("a", "b", "expected"), [
    ("abcdef", "abcdef", "P"),
    ("abcde", "abcde", "uncertain"),  # length below six
    ("abcdefghijk", "abcdefghXYZ", "P"),  # Dice exactly .70
    ("abcdefghijk", "abcdefgWXYZ", "uncertain"),
    ("aaaaab", "aabaaa", "uncertain"),  # Dice .80, but L=3
    ("aaaaab", "aaabaa", "P"),  # Dice .80, L=4 exactly
    ("abcdefghijk", "ab123456789", "uncertain"),  # .10, but L=2
    ("abcdef", "aXYZ", "U"),  # one shared character, L=1
    ("abcdef", "XYZ", "U"),
    ("abcdef", "abcXYZ", "uncertain"),
    ("abcdef", "abcdefZZZZZZZZZZ", "uncertain"),
    ("", "", "U"),
    (" \uff21\uff22\uff23", "ABC", "uncertain"),
])
def test_weak_label_boundaries(a, b, expected):
    assert gate.weak_label(a, b) == expected


def test_weak_labels_only_with_true_reply_and_never_override_app():
    items = [event("yes", apps=("no",), true_reply=["合成答复完整六字"]), event("none")]
    items[0]["candidates"][0]["r"] = ["合成答复完整六字"]
    report = gate.analyze(items, "char")
    assert report["weak_labels"]["counts"] == {"P": 1, "U": 0, "uncertain": 0}
    assert report["weak_labels"]["candidates"] == [
        {"event_id": "yes", "cand_id": "c0", "Y_use": "P"},
    ]
    assert report["decisions"][0]["app"] == "no"


@pytest.fixture
def cli(tmp_path, capsys):
    def invoke(items, *extra, scorer="char", output=None):
        input_path = write_jsonl(tmp_path / "input.jsonl", items)
        out = output or tmp_path / "out.json"
        code = gate.main(["--input", str(input_path), "--scorer", scorer,
                          "--out", str(out), *extra])
        captured = capsys.readouterr()
        assert captured.err == ""
        assert len(captured.out.splitlines()) == 1
        return code, json.loads(captured.out), out, captured.out
    return invoke


def test_cli_outputs_no_text_secure_mode_and_script_entrypoint(cli, tmp_path):
    marker = "UNIQUE_SYNTHETIC_CANARY_987654"
    item = event(context=[marker], true_reply=[marker])
    item["candidates"][0].update(c=[marker], r=[marker])
    out = tmp_path / "existing.json"
    out.write_text("old")
    out.chmod(0o644)
    code, summary, out, stdout = cli([item], output=out)
    assert code == 0 and summary["status"] == "ok"
    assert marker not in stdout and marker not in out.read_text()
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    report = json.loads(out.read_text())
    assert report["scorer"] == "char"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/g_gate.py"), "--input",
         str(tmp_path / "input.jsonl"), "--scorer", "length", "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0 and result.stderr == ""
    assert marker not in result.stdout and marker not in out.read_text()


@pytest.mark.parametrize("mutate", [
    lambda e: e.pop("event_id"), lambda e: e.update(event_id=2),
    lambda e: e.update(split="train"), lambda e: e.update(context="bad"),
    lambda e: e.update(context=[1]), lambda e: e.update(candidates={}),
    lambda e: e["candidates"][0].pop("source"),
    lambda e: e["candidates"][0].update(cand_id=False),
    lambda e: e["candidates"][0].update(source="other"),
    lambda e: e["candidates"][0].update(c=[None]),
    lambda e: e["candidates"][0].update(r=None),
    lambda e: e.update(labels=[]), lambda e: e.update(labels={"c0": {"app": "maybe"}}),
    lambda e: e.update(labels={"c0": {}}), lambda e: e.update(labels={"c0": "yes"}),
    lambda e: e.update(group=1), lambda e: e.update(group_kind=[]),
    lambda e: e.update(true_reply="bad"), lambda e: e.update(true_reply=[False]),
    lambda e: e.update(candidates=[e["candidates"][0], e["candidates"][0]]),
])
def test_bad_input_failed_only(cli, mutate):
    item = event()
    mutate(item)
    code, summary, out, stdout = cli([item])
    assert code == 1 and summary == {"status": "failed"}
    assert stdout.strip() == '{"status":"failed"}'
    assert not out.exists()


def test_duplicate_event_ids_and_malformed_json_fail(cli, tmp_path, capsys):
    assert cli([event(), event()])[0] == 1
    path = tmp_path / "bad.jsonl"
    path.write_text('{"context":"SECRET_BAD_JSON"')
    assert gate.main(["--input", str(path), "--scorer", "char", "--out",
                      str(tmp_path / "out.json")]) == 1
    assert capsys.readouterr().out.strip() == '{"status":"failed"}'


@pytest.mark.parametrize("extra", [[], ["--threshold", "nan"], ["--threshold", "1.1"],
                                   ["--threshold", "-0.1"]])
def test_missing_parent_and_bad_threshold_fail(cli, tmp_path, extra):
    code, summary, out, _ = cli([event()], *extra, output=tmp_path / "missing" / "x.json")
    assert code == 1 and summary == {"status": "failed"} and not out.exists()


@pytest.mark.parametrize("threshold", ["nan", "inf", "-0.1", "1.1", "bad"])
def test_invalid_threshold_with_valid_output_parent(cli, threshold):
    assert cli([event()], "--threshold", threshold)[0] == 1


@pytest.mark.parametrize("argv", [[], ["--unknown"], ["--scorer", "rule"]])
def test_argument_errors_only_json(capsys, argv):
    assert gate.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status":"failed"}' and captured.err == ""


def test_vector_scores_require_every_candidate_and_accept_correct_file(cli, tmp_path):
    scores = tmp_path / "scores.jsonl"
    write_jsonl(scores, [{"event_id": "e", "cand_id": "c0", "score": 0.5}])
    assert cli([event(apps=("yes", "no"))], "--scores-file", str(scores), scorer="vector")[0] == 1
    assert cli([event()], scorer="vector")[0] == 1
    code, _, out, _ = cli([event()], "--scores-file", str(scores), scorer="vector")
    assert code == 0
    assert json.loads(out.read_text())["decisions"][0]["score"] == 0.5


def test_vector_cli_requires_scores_even_for_empty_input(cli):
    assert cli([], scorer="vector")[0] == 1


def test_vector_cli_full_calibration_decision_and_evaluation(cli, tmp_path):
    items = [event(f"cal{i}", "cal", apps=("yes" if i < 5 else "no",))
             for i in range(10)]
    items.extend([event("dev", "dev"), event("test", apps=("yes", "no", "unknown")),
                  event("conflict", context=["今晚不想睡 还要赶报告"])])
    items[-1]["candidates"][0]["r"] = ["好呀 早点睡", "安安"]
    records = [{"event_id": e["event_id"], "cand_id": c["cand_id"], "score": (
        0.5 if e["split"] == "dev" else 0 if e["event_id"] in
        {"cal5", "cal6", "cal7", "cal8", "cal9"} else 1
    )} for e in items for c in e["candidates"]]
    path = write_jsonl(tmp_path / "vector.jsonl", records)
    code, summary, out, _ = cli(items, "--scores-file", str(path), scorer="vector")
    assert code == 0 and summary["calibration_status"] == "ok"
    report = json.loads(out.read_text())
    assert report["calibration"]["cutoff_source"] == "dev"
    assert report["calibration"]["cutoff"] == 0.5
    assert report["metrics"]["confusion"] == {"TP": 1, "FP": 1, "TN": 0, "FN": 1}
    assert report["decisions"][-1]["keep"] is False
    assert all(r["keep"] for r in report["decisions"] if r["event_id"] == "test")
    assert report["metrics"]["brier"]["value"] == pytest.approx(38 / 147)


@pytest.mark.parametrize("score", [None, True, "0.1", float("nan"), float("inf")])
def test_vector_scores_reject_invalid_numeric_types(cli, tmp_path, score):
    scores = write_jsonl(tmp_path / "scores.jsonl", [
        {"event_id": "e", "cand_id": "c0", "score": score},
    ])
    assert cli([event()], "--scores-file", str(scores), scorer="vector")[0] == 1


def test_embedding_injection_cosine_and_no_sentence_transformers_import(monkeypatch, tmp_path):
    import builtins

    original_import = builtins.__import__
    imports = []

    def guarded(name, *args, **kwargs):
        imports.append(name)
        if name.startswith("sentence_transformers"):
            raise AssertionError("optional dependency imported in injected path")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.reload(embed)
    item = event(context=[" \uff21", "\uff22 "], apps=("yes", "no"))
    item["candidates"][0]["c"] = ["\uff23 "]
    item["candidates"][1]["c"] = [" \uff24"]
    calls = []

    class FakeEncoder:
        def encode(self, texts, *, normalize_embeddings):
            calls.append((texts, normalize_embeddings))
            return [{"AB": [1.0, 0.0], "C": [0.6, 0.8], "D": [-1.0, 0.0]}[t]
                    for t in texts]

    scores = embed.score_events([item], FakeEncoder())
    assert scores == [{"event_id": "e", "cand_id": "c0", "score": 0.6},
                      {"event_id": "e", "cand_id": "c1", "score": -1.0}]
    assert all(normalized for _, normalized in calls)
    assert not any(name.startswith("sentence_transformers") for name in imports)
    path = tmp_path / "scores.jsonl"
    embed.write_scores(path, scores)
    assert [json.loads(line) for line in path.read_text().splitlines()] == scores
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(gate.score_candidates([item], "vector", gate.load_scores(path))) == 2


def test_embedding_cli_fake_boundary_offline_flags_and_redaction(monkeypatch, tmp_path, capsys):
    import types

    marker = "EMBED_UNIQUE_SYNTHETIC_MARKER"
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    item = event(context=[marker])
    path = write_jsonl(tmp_path / "input.jsonl", [item])
    seen = {}

    class FakeModel:
        def __init__(self, model, **kwargs):
            seen.update(model=model, **kwargs)
            print(marker)
            print(marker, file=sys.stderr)

        def encode(self, texts, *, normalize_embeddings):
            return [[1, 0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(
        SentenceTransformer=FakeModel,
    ))
    out = tmp_path / "scores.jsonl"
    assert embed.main(["--input", str(path), "--model", "cached/model", "--revision", "a" * 40,
                       "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.err == "" and marker not in captured.out and marker not in out.read_text()
    assert json.loads(captured.out) == {"status": "ok", "events": 1, "candidates": 1}
    assert seen["revision"] == "a" * 40 and seen["local_files_only"] is True
    assert seen["model"] == "cached/model"
    assert embed.main([]) == 1
    assert capsys.readouterr().out.strip() == '{"status":"failed"}'


def test_embedding_cli_failure_redacts_library_exception(monkeypatch, tmp_path, capsys):
    import types

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    path = write_jsonl(tmp_path / "input.jsonl", [event()])

    def fail(*args, **kwargs):
        print("SYNTHETIC_LIBRARY_SECRET", file=sys.stderr)
        raise RuntimeError("SYNTHETIC_LIBRARY_SECRET")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(
        SentenceTransformer=fail,
    ))
    assert embed.main(["--input", str(path), "--model", "cached/model", "--revision", "a" * 40,
                       "--out", str(tmp_path / "out.jsonl")]) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status":"failed"}' and captured.err == ""
    assert not (tmp_path / "out.jsonl").exists()


def test_analyze_keeps_input_unchanged():
    items = [event()]
    before = copy.deepcopy(items)
    gate.analyze(items, "char")
    assert items == before
    assert math.isfinite(gate.dice("abc", "abc"))
