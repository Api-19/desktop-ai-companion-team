import json
import os
import stat
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.privacy.paired_snippets import PairedSnippet, write_snippets

ROOT = Path(__file__).resolve().parents[2]
BEIJING = timezone(timedelta(hours=8))
BASE = datetime(2026, 1, 1, tzinfo=BEIJING)
STAGES = [
    {"from": "2026-01-01", "to": "2026-07-31", "label": "early"},
    {"from": "2026-08-01", "to": "2026-12-31", "label": "late"},
]


def snippet(index, *, at=None, conversation="synthetic", partner=None, context=None, reply=None):
    return PairedSnippet(
        snippet_id=f"s-{index:04d}", conversation_id=conversation,
        started_at=at or BASE + timedelta(hours=index),
        partner_turns=partner if partner is not None else [chr(0x4E00 + index)],
        owner_reply=reply if reply is not None else ["合成回答"],
        context_turns=context or [], source_fingerprint="synthetic-fingerprint",
    )


@pytest.fixture
def cli(monkeypatch, tmp_path):
    from scripts import m1_pools

    monkeypatch.setattr(m1_pools, "load_settings", lambda root: SimpleNamespace(
        private_data_dir=tmp_path / "private",
    ))
    return m1_pools


@pytest.fixture
def run(cli, tmp_path, capsys):
    def invoke(items, *, holdout=(), exposed=(), stages=STAGES, seed=None):
        library_path = tmp_path / "library.jsonl"
        holdout_path = tmp_path / "holdout.jsonl"
        timeline_path = tmp_path / "timeline.json"
        write_snippets(library_path, items)
        write_snippets(holdout_path, holdout)
        timeline_path.write_text(json.dumps({"stages": stages, "ignored": "synthetic"}))
        out = tmp_path / "private" / "pools"
        args = ["--library", str(library_path), "--holdout", str(holdout_path),
                "--timeline", str(timeline_path), "--out-dir", str(out)]
        for index, exposed_items in enumerate(exposed):
            path = tmp_path / f"exposed-{index}.jsonl"
            write_snippets(path, exposed_items)
            args.extend(["--exposed", str(path)])
        if seed is not None:
            args.extend(["--seed", str(seed)])
        exit_code = cli.main(args)
        captured = capsys.readouterr()
        assert captured.err == ""
        assert len(captured.out.splitlines()) == 1
        summary = json.loads(captured.out)
        manifest = json.loads((out / "m1_pools.json").read_text()) if exit_code == 0 else None
        return exit_code, summary, manifest

    return invoke


@pytest.mark.parametrize(("seconds", "count"), [(1800, 1), (1801, 2)])
def test_time_boundary(run, seconds, count):
    code, summary, manifest = run([
        snippet(1, at=BASE), snippet(2, at=BASE + timedelta(seconds=seconds)),
    ])
    assert code == 0
    assert summary["events"] == count
    assert manifest["events"][0]["predict_snippet_id"] == "s-0001"


def test_normalized_exact_input_across_days_and_earliest_id(run):
    first = snippet(9, at=BASE, partner=["Ａ Ｂ\tＣ"],  # noqa: RUF001
                    context=["他： 合成前文测试"])  # noqa: RUF001
    later = snippet(1, at=BASE + timedelta(days=5), partner=["A\nBC"],
                    context=["他:合成前文测试"])
    code, summary, manifest = run([later, first])
    assert code == 0 and summary["events"] == 1
    event = manifest["events"][0]
    assert event["event_id"] == event["predict_snippet_id"] == first.snippet_id
    assert event["snippet_ids"] == [first.snippet_id, later.snippet_id]
    assert event["start"] == first.started_at.isoformat()
    assert event["end"] == later.started_at.isoformat()


@pytest.mark.parametrize(("partner_length", "count"), [(1, 2), (9, 2), (10, 1)])
def test_exact_input_requires_twelve_normalized_characters(run, partner_length, count):
    code, summary, _ = run([
        snippet(1, at=BASE, partner=["甲" * partner_length]),
        snippet(2, at=BASE + timedelta(days=5), partner=["甲 " * partner_length]),
    ])
    assert code == 0 and summary["events"] == count


@pytest.mark.parametrize(("rule", "long", "count"), [
    ("x", True, 1), ("x", False, 2), ("p", True, 1), ("p", False, 2),
])
def test_near_duplicate_length_threshold(run, rule, long, count):
    # Repeated bigrams keep similarity high even for the short examples.
    text = "甲乙" * (6 if long else 2)
    partners = [text, text + "甲"]
    contexts = [[], []] if rule == "x" else [["他:独特前文丙丁戊己庚辛壬癸"],
                                                ["他:另一前文天地玄黄宇宙洪荒"]]
    code, summary, _ = run([
        snippet(i + 1, at=BASE + timedelta(days=i * 5),
                partner=[partner], context=contexts[i])
        for i, partner in enumerate(partners)
    ])
    assert code == 0 and summary["events"] == count


def test_exactly_twelve_input_characters_enable_x_similarity(run):
    # NFKC role prefix contributes two characters to x; p stays shorter than 12.
    code, summary, _ = run([
        snippet(1, at=BASE, partner=["甲乙" * 5]),
        snippet(2, at=BASE + timedelta(days=4), partner=["甲乙" * 5 + "甲"]),
    ])
    assert code == 0 and summary["events"] == 1


@pytest.mark.parametrize(("suffix", "count"), [("nop", 1), ("nopq", 2)])
def test_partner_jaccard_exact_point_eight_and_below(run, suffix, count):
    code, summary, _ = run([
        snippet(1, at=BASE, partner=["abcdefghijklm"], context=["甲" * 30]),
        snippet(2, at=BASE + timedelta(days=5), partner=["abcdefghijklm" + suffix],
                context=["乙" * 30]),
    ])
    assert code == 0 and summary["events"] == count


@pytest.mark.parametrize(("shared_lines", "long", "count"), [
    (2, True, 1), (1, True, 2), (2, False, 2),
])
def test_context_overlap(run, shared_lines, long, count):
    partner = "甲乙丙丁戊己庚辛" if long else "甲"
    reply = "壬癸天地玄黄宇宙" if long else "乙"
    context = [f"她:{partner}", f"他:{reply}"][:shared_lines]
    code, summary, _ = run([
        snippet(1, at=BASE, partner=[partner], reply=[reply]),
        snippet(2, at=BASE + timedelta(days=5), partner=["全新问题"], context=context),
    ])
    assert code == 0 and summary["events"] == count


@pytest.mark.parametrize(("reply_length", "count"), [(7, 2), (8, 1)])
def test_context_overlap_nineteen_and_twenty_character_boundary(run, reply_length, count):
    partner = "甲乙丙丁戊己庚辛"
    reply = "壬癸天地玄黄宇宙"[:reply_length]
    code, summary, _ = run([
        snippet(1, at=BASE, partner=[partner], reply=[reply]),
        snippet(2, at=BASE + timedelta(days=5), partner=["新问"],
                context=[f"她:{partner}", f"他:{reply}"]),
    ])
    assert code == 0 and summary["events"] == count


def test_overlap_normalizes_each_line_but_requires_consecutive_lines(run):
    first = snippet(1, at=BASE, partner=["甲乙丙丁戊己庚辛"],
                    reply=["ＡＢＣＤＥＦＧＨ"])  # noqa: RUF001
    second = snippet(2, at=BASE + timedelta(days=3), partner=["戊"],
                     context=["她: 甲乙丙丁戊己庚辛", "他： A B C D E F G H"])  # noqa: RUF001
    third = snippet(3, at=BASE + timedelta(days=6), partner=["己"],
                    context=["她:甲乙丙丁戊己庚辛", "她:插入轮次" + "无关" * 20, "他:ABCDEFGH"])
    code, summary, manifest = run([first, second, third])
    assert code == 0 and summary["events"] == 2
    assert manifest["events"][0]["snippet_ids"] == [first.snippet_id, second.snippet_id]


def test_different_conversations_never_connect(run):
    code, summary, _ = run([
        snippet(1, at=BASE, partner=["相同的十二字符合成消息甲乙丙丁"], conversation="one"),
        snippet(2, at=BASE, partner=["相同的十二字符合成消息甲乙丙丁"], conversation="two"),
    ])
    assert code == 0 and summary["events"] == 2


def test_exposure_propagates_transitively_and_input_ids_deduplicate(run):
    first = snippet(1, at=BASE, partner=["甲乙" * 6])
    second = snippet(2, at=BASE + timedelta(days=5), partner=["甲乙" * 6 + "甲"])
    third = snippet(3, at=second.started_at + timedelta(minutes=30))
    missing = snippet(99)
    code, summary, manifest = run(
        [third, first], holdout=[first, second], exposed=[[second, missing], [second, third]],
    )
    assert code == 0
    assert summary["snippets"] == 3 and summary["events"] == 1
    assert summary["exposed_events"] == 1 and summary["exposed_snippets_matched"] == 2
    assert manifest["events"][0]["pool"] == "exposed"
    assert manifest["events"][0]["snippet_ids"] == [first.snippet_id, second.snippet_id,
                                                  third.snippet_id]


def test_last_stage_tail_uses_remaining_events_and_ceil(run):
    base = datetime(2026, 8, 1, tzinfo=BEIJING)
    items = [snippet(i, at=base + timedelta(hours=i)) for i in range(7)]
    code, _, manifest = run(list(reversed(items)), exposed=[[items[-1]]])
    assert code == 0
    assert [event["pool"] for event in manifest["events"]] == [
        "train", "train", "train", "train", "m2_sealed", "m2_sealed", "exposed",
    ]


def test_each_snippet_must_have_stage_even_when_event_start_is_valid(run):
    first = snippet(1, at=datetime(2026, 7, 31, 23, 50, tzinfo=BEIJING))
    second = snippet(2, at=first.started_at + timedelta(minutes=20))
    code, summary, _ = run([first, second], stages=STAGES[:1])
    assert code == 1 and summary == {"status": "failed"}


def test_event_stage_is_start_beijing_date_and_timeline_end_is_inclusive(run):
    first = snippet(1, at=datetime(2026, 7, 31, 15, 50, tzinfo=UTC))
    second = snippet(2, at=first.started_at + timedelta(minutes=20))
    code, _, manifest = run([first, second])
    assert code == 0 and manifest["events"][0]["stage"] == "early"


@pytest.mark.parametrize(("size", "train", "calib", "test"), [
    (0, 0, 0, 0), (1, 1, 0, 0), (2, 2, 0, 0), (3, 3, 0, 0),
    (15, 10, 2, 3), (17, 12, 2, 3), (29, 21, 3, 5),
])
def test_chronological_split_and_remainder(run, size, train, calib, test):
    code, summary, manifest = run([snippet(i) for i in reversed(range(size))])
    assert code == 0
    assert [event["pool"] for event in manifest["events"]] == (
        ["train"] * train + ["calib"] * calib + ["test"] * test
    )
    assert summary["tier"] == "feasibility_only"
    assert not any(event["selected"] for event in manifest["events"])


@pytest.mark.parametrize(("size", "tier", "selected"), [
    (450, "300/60/90", {"train": 300, "calib": 60, "test": 90}),
    (449, "200/40/60", {"train": 301, "calib": 40, "test": 60}),
    (300, "200/40/60", {"train": 200, "calib": 40, "test": 60}),
    (299, "feasibility_only", {"train": 0, "calib": 0, "test": 0}),
])
def test_tiers_and_train_kept_in_full(run, size, tier, selected):
    code, summary, manifest = run([snippet(i) for i in range(size)])
    assert code == 0 and summary["tier"] == manifest["tier"] == tier
    assert summary["selected"] == selected
    assert manifest["seed"] == 0 and manifest["version"] == 1
    if tier == "feasibility_only":
        assert all(event["pool"] != "m2_reserve" for event in manifest["events"])


@pytest.mark.parametrize(("size", "reserve"), [(449, 48), (600, 50)])
def test_unselected_calibration_and_test_events_become_reserve(run, size, reserve):
    code, summary, manifest = run([snippet(i) for i in range(size)], seed=17)
    assert code == 0
    events = manifest["events"]
    assert sum(event["pool"] == "m2_reserve" for event in events) == reserve
    assert summary["pool_counts"]["m2_reserve"] == {"early": reserve, "late": 0}
    assert all(event["selected"] for event in events if event["pool"] in {"calib", "test"})
    assert all(not event["selected"] for event in events if event["pool"] == "m2_reserve")
    assert all(event["selected"] for event in events if event["pool"] == "train")


def test_relaxed_cross_pool_links_count_snippet_pairs_not_events(run):
    short = [snippet(i, at=BASE + timedelta(days=i * 2), partner=["短语"],
                     conversation="short") for i in range(3)]
    # Two train snippets are one time-connected event: both pairs with the exposed
    # snippet count, while the pair within their own event does not.
    short[2].started_at = short[1].started_at + timedelta(minutes=20)
    overlap = [
        snippet(10, at=BASE, partner=["甲乙丙丁戊己庚辛"], reply=["壬癸天地玄黄宇"],
                conversation="overlap"),
        snippet(11, at=BASE + timedelta(days=3), partner=["新问"],
                context=["她:甲乙丙丁戊己庚辛", "他:壬癸天地玄黄宇"], conversation="overlap"),
    ]
    other_conversation = snippet(20, partner=["短语"], conversation="other")
    code, summary, manifest = run([*short, *overlap, other_conversation],
                                  holdout=[short[0]], exposed=[[short[0], overlap[0]]])
    assert code == 0 and summary["events"] == 5
    assert summary["relaxed_cross_pool_links"] == 3
    assert manifest["events"][0]["pool"] == "exposed"


@pytest.mark.parametrize(("length", "links"), [(11, 0), (12, 1), (19, 1), (20, 0)])
def test_relaxed_overlap_counts_only_old_edges_removed_by_new_threshold(run, length, links):
    partner, reply = "甲" * 4, "乙" * (length - 8)
    first = snippet(1, at=BASE, partner=[partner], reply=[reply])
    later = snippet(2, at=BASE + timedelta(days=5), partner=["新问"],
                    context=[f"她:{partner}", f"他:{reply}"])
    code, summary, _ = run([first, later], exposed=[[first]])
    assert code == 0 and summary["relaxed_cross_pool_links"] == links


def test_relaxed_edge_still_connected_by_current_rule_is_not_counted(run):
    partner = "甲乙丙丁戊己庚辛壬癸子丑"
    first = snippet(1, at=BASE, partner=[partner], context=["她:甲乙丙丁", "他:戊己庚辛"])
    later = snippet(2, at=BASE + timedelta(days=5), partner=[partner],
                    context=["她:甲乙丙丁", "他:戊己庚辛", "他:新"])
    code, summary, _ = run([first, later], exposed=[[first]])
    assert code == 0 and summary["events"] == 1
    assert summary["relaxed_cross_pool_links"] == 0


def test_relaxed_links_use_final_reserve_pool_after_sampling(run):
    items = [snippet(i, conversation=f"conversation-{i}") for i in range(600)]
    code, _, manifest = run(items, seed=17)
    assert code == 0
    calib = next(event for event in manifest["events"] if event["pool"] == "calib")
    reserve = next(event for event in manifest["events"]
                   if event["pool"] == "m2_reserve" and int(event["event_id"][2:]) < 480)
    for event in (calib, reserve):
        item = items[int(event["event_id"][2:])]
        item.conversation_id = "shared-short-input"
        item.partner_turns = ["短语"]
    code, summary, manifest = run(items, seed=17)
    assert code == 0 and summary["events"] == 600
    assert summary["relaxed_cross_pool_links"] == 1
    pools = {event["event_id"]: event["pool"] for event in manifest["events"]}
    assert pools[calib["event_id"]] == "calib"
    assert pools[reserve["event_id"]] == "m2_reserve"


def fragmented_stages(size):
    stages = [{"from": (BASE + timedelta(days=i)).date().isoformat(),
               "to": (BASE + timedelta(days=i)).date().isoformat(), "label": f"stage-{i}"}
              for i in range(size)]
    return [*stages, {"from": "2027-01-01", "to": "2027-01-31", "label": "last"}]


@pytest.mark.parametrize(("groups", "group_size", "tier"), [
    (40, 12, "200/40/60"), (150, 3, "feasibility_only"), (100, 3, "feasibility_only"),
])
def test_insufficient_pool_quota_downgrades(run, groups, group_size, tier):
    items = [snippet(i * group_size + j, at=BASE + timedelta(days=i, minutes=j * 60))
             for i in range(groups) for j in range(group_size)]
    code, summary, manifest = run(items, stages=fragmented_stages(groups))
    assert code == 0 and summary["tier"] == tier
    if tier == "200/40/60":
        assert summary["selected"] == {"train": 360, "calib": 40, "test": 60}
    else:
        assert not any(event["selected"] for event in manifest["events"])


def test_stratified_sampling_largest_remainders_and_seed(run):
    stages = fragmented_stages(3)
    items = [snippet(i * 1000 + j, at=BASE + timedelta(days=i, seconds=j * 31))
             for i, size in enumerate((150, 300, 15)) for j in range(size)]
    # Separate conversations prevent the closely spaced synthetic events from merging.
    for item in items:
        item.conversation_id = item.snippet_id
    code, summary, manifest = run(items, stages=stages, seed=17)
    assert code == 0 and summary["tier"] == "300/60/90"
    selected = {
        pool: {stage: sum(event["selected"] and event["pool"] == pool
                         and event["stage"] == stage for event in manifest["events"])
               for stage in ("stage-0", "stage-1", "stage-2")}
        for pool in ("calib", "test")
    }
    assert selected == {"calib": {"stage-0": 19, "stage-1": 39, "stage-2": 2},
                        "test": {"stage-0": 29, "stage-1": 58, "stage-2": 3}}
    assert run(list(reversed(items)), stages=stages, seed=17)[2] == manifest
    assert run(items, stages=stages, seed=18)[2] != manifest


def test_minimum_one_per_nonempty_stage_and_tie_order(cli):
    assert cli._stage_quotas({"large": 98, "tiny-a": 1, "tiny-b": 1}, 10) == {
        "large": 8, "tiny-a": 1, "tiny-b": 1,
    }
    assert cli._stage_quotas({"a": 3, "b": 3, "c": 3}, 4) == {"a": 2, "b": 1, "c": 1}
    assert cli._stage_quotas({"a": 3, "b": 3, "c": 3}, 2) is None


@pytest.mark.parametrize("destination", ["outside", "root", "symlink"])
def test_out_dir_rejected_before_input_read(cli, tmp_path, monkeypatch, capsys, destination):
    private = tmp_path / "private"
    private.mkdir()
    link = private / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    out = {"outside": tmp_path / "out", "root": private, "symlink": link / "out"}[destination]

    def forbidden(path):
        pytest.fail("snippet input read")

    monkeypatch.setattr(cli, "read_snippets", forbidden)
    assert cli.main(["--library", "missing", "--holdout", "missing", "--timeline", "missing",
                     "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"status": "out_dir_not_private"}


@pytest.mark.parametrize("stages", [
    [], [{"from": "bad synthetic date", "to": "2026-12-31", "label": "early"}],
    [{"from": "2026-12-31", "to": "2026-01-01", "label": "early"}],
    [{"from": "2026-01-01", "to": "2026-12-31", "label": 123}],
    [{"from": "2026-02-01", "to": "2026-12-31", "label": "early"}],
])
def test_timeline_failures_are_sanitized(run, stages):
    code, summary, _ = run([snippet(1)], stages=stages)
    assert code == 1 and summary == {"status": "failed"}


def test_invalid_arguments_and_inputs_are_sanitized(cli, tmp_path, capsys):
    assert cli.main(["--synthetic-invalid-option"]) == 1
    assert capsys.readouterr().out.strip() == '{"status":"failed"}'
    library = tmp_path / "library.jsonl"
    library.write_text('synthetic-content-that-must-not-leak')
    assert cli.main(["--library", str(library), "--holdout", "missing", "--timeline", "missing",
                     "--out-dir", str(tmp_path / "private" / "out")]) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status":"failed"}' and captured.err == ""


def test_manifest_permissions_counts_and_no_text_in_terminal_cli(tmp_path):
    marker = "UNIQUE_SYNTHETIC_TEXT_NEVER_OUTPUT"
    library, holdout, timeline = [tmp_path / name for name in (
        "library.jsonl", "holdout.jsonl", "timeline.json",
    )]
    write_snippets(library, [snippet(1, partner=[marker], context=["她:" + marker],
                                     reply=[marker])])
    write_snippets(holdout, [])
    timeline.write_text(json.dumps({"stages": STAGES}))
    out = tmp_path / "private" / "out"
    out.mkdir(parents=True)
    output = out / "m1_pools.json"
    output.write_text("old synthetic manifest")
    output.chmod(0o644)
    environ = {key: value for key, value in os.environ.items()
               if not key.startswith("COMPANION_") and key != "PYTHONPATH"}
    environ["COMPANION_PRIVATE_DATA_DIR"] = str(tmp_path / "private")
    result = subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/m1_pools.py"),
        "--library", str(library), "--holdout", str(holdout), "--timeline", str(timeline),
        "--out-dir", str(out),
    ], cwd=ROOT, env=environ, capture_output=True, text=True)
    assert result.returncode == 0 and result.stderr == ""
    assert len(result.stdout.splitlines()) == 1
    assert marker not in result.stdout + result.stderr + output.read_text()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(result.stdout) == {
        "status": "ok", "snippets": 1, "events": 1, "tier": "feasibility_only",
        "exposed_events": 0, "exposed_snippets_matched": 0,
        "relaxed_cross_pool_links": 0,
        "pool_counts": {
            pool: {"early": int(pool == "train"), "late": 0}
            for pool in ("exposed", "m2_sealed", "train", "calib", "test", "m2_reserve")
        },
        "selected": {"train": 0, "calib": 0, "test": 0},
    }
    manifest = json.loads(output.read_text())
    assert set(manifest) == {"version", "seed", "tier", "events"}
    assert set(manifest["events"][0]) == {
        "event_id", "stage", "pool", "start", "end", "snippet_ids", "predict_snippet_id",
        "selected",
    }
