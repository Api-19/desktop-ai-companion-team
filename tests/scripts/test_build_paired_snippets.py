import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.importer.models import ImportIssue, ImportSeverity
from companion.importer.standard_json import StandardJsonAdapter

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_paired_snippets.py"
ZERO_MEDIA_COUNTS = {
    "excluded_by_media": 0, "excluded_by_deictic": 0,
    "excluded_by_short_reaction": 0, "context_truncated_at_media": 0,
}


@pytest.fixture
def inputs(tmp_path):
    messages = []
    for i in range(12):
        for owner in (False, True):
            messages.append({
                "schema_version": 1,
                "conversation_id": "synthetic-conversation",
                "message_id": f"synthetic-message-{i}-{owner}",
                "timestamp": (datetime(2026, 10, 6, tzinfo=UTC)
                              + timedelta(hours=i, minutes=int(owner))).isoformat(),
                "source_speaker_id": "synthetic-owner" if owner else "synthetic-contact",
                "source_speaker_name": None,
                "direction": "outbound" if owner else "inbound",
                "content_type": "text",
                "text": ("合成名字收到" if owner else f"合成名字话题{chr(0x4E00 + i * 37)}")
                        if i < 11 else "合成排除标记",
            })
    payloads = {
        "chat": messages,
        "mapping": {"synthetic-owner": "owner", "synthetic-contact": "other"},
        "terms": {"redaction_terms": {"name": ["合成名字"]},
                  "risk_terms": {"private_event": ["合成排除标记"]}},
    }
    paths = {}
    for name, value in payloads.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return paths


def args(inputs, out):
    return [arg for name, path in inputs.items() for arg in (f"--{name}", str(path))] + [
        "--out-dir", str(out), "--seed", "7",
    ]


@pytest.mark.parametrize("with_veiled", [False, True])
def test_terminal_cli_counts_and_private_outputs(tmp_path, inputs, with_veiled):
    out = tmp_path / "private" / "snippets"
    environ = {key: value for key, value in os.environ.items()
               if not key.startswith("COMPANION_") and key != "PYTHONPATH"}
    environ["COMPANION_PRIVATE_DATA_DIR"] = str(tmp_path / "private")
    veiled_args = (["--veiled", str(ROOT / "config/privacy/veiled_terms.json")]
                   if with_veiled else [])
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(SCRIPT), *args(inputs, out), *veiled_args],
        cwd=ROOT, env=environ, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert len(result.stdout.splitlines()) == 1
    counts = json.loads(result.stdout)
    assert counts == {"status": "ok", "records": 24, "snippets": 11,
                      "library": 9, "holdout": 2, "audit": 9,
                      "deleted_ids_matched": 0, "blocked_messages": 0,
        "excluded_by_deleted": 0,
        "style_eligible": 11,
                      "excluded_by_veiled": 0, "flagged_suspect": 0, **ZERO_MEDIA_COUNTS}
    for key, filename in [("library", "library.jsonl"), ("holdout", "holdout.jsonl"),
                          ("audit", "audit_sample.jsonl")]:
        text = (out / filename).read_text(encoding="utf-8")
        assert len(text.splitlines()) == counts[key]
        assert "合成名字" not in text and "合成排除标记" not in text
        assert "[NAME]" in text
        for line in text.splitlines():
            item = json.loads(line)
            assert item["owner_reply"] == ["[NAME]收到"]
    assert json.loads((out / "audit_sample.jsonl").read_text().splitlines()[0]) in [
        json.loads(line) for line in (out / "library.jsonl").read_text().splitlines()
    ]


def load_cli(monkeypatch, tmp_path):
    from scripts import build_paired_snippets as cli
    monkeypatch.setattr(cli, "load_settings",
                        lambda root: SimpleNamespace(private_data_dir=tmp_path / "private"))
    return cli


@pytest.mark.parametrize("destination", ["outside", "root", "symlink"])
def test_outside_private_rejected_before_any_input_read(
    monkeypatch, tmp_path, inputs, capsys, destination,
):
    cli = load_cli(monkeypatch, tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    link = private / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    out = {"outside": tmp_path / "outside", "root": private, "symlink": link / "out"}[
        destination
    ]

    def forbidden(*args, **kwargs):
        pytest.fail("input file opened")

    monkeypatch.setattr(Path, "open", forbidden)
    assert cli.main(args(inputs, out)) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "out_dir_not_private"}
    assert captured.err == ""


@pytest.mark.parametrize("failure", [
    "import_error", "invalid_chat", "mapping", "terms_root", "terms_key", "risk_category",
    "risk_values", "redaction_label", "redaction_values", "terms_null",
])
def test_failures_are_sanitized(monkeypatch, tmp_path, inputs, capsys, failure):
    cli = load_cli(monkeypatch, tmp_path)
    if failure == "import_error":
        batch = StandardJsonAdapter().parse(inputs["chat"])
        batch.issues.append(ImportIssue(
            code="synthetic-error", severity=ImportSeverity.ERROR, detail="合成名字",
        ))
        monkeypatch.setattr(cli.StandardJsonAdapter, "parse", lambda self, path: batch)
    else:
        name, payload = {
            "invalid_chat": (
                "chat", [{"text": "合成名字", "source_speaker_id": "synthetic-owner"}],
            ),
            "mapping": ("mapping", {"synthetic-owner": "other"}),
            "terms_root": ("terms", []),
            "terms_key": ("terms", {"unexpected": {}}),
            "risk_category": ("terms", {"risk_terms": {"invalid": ["合成名字"]}}),
            "risk_values": ("terms", {"risk_terms": {"health": "合成名字"}}),
            "redaction_label": ("terms", {"redaction_terms": {"invalid": ["合成名字"]}}),
            "redaction_values": ("terms", {"redaction_terms": {"name": [123]}}),
            "terms_null": ("terms", {"risk_terms": None}),
        }[failure]
        inputs[name].write_text(json.dumps(payload), encoding="utf-8")
    assert cli.main(args(inputs, tmp_path / "private" / "out")) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status":"failed"}'
    assert captured.err == ""
    assert "合成名字" not in captured.out and "synthetic-owner" not in captured.out


def direction_args(inputs, out):
    return [*args({key: path for key, path in inputs.items() if key != "mapping"}, out),
        "--direction-mapping",
    ]


def test_direction_mapping_infers_speakers_from_batch(monkeypatch, tmp_path, inputs, capsys):
    cli = load_cli(monkeypatch, tmp_path)
    messages = json.loads(inputs["chat"].read_text())
    for message in messages:
        message["source_speaker_id"] = "another-" + message["direction"]
    inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
    inputs["mapping"].unlink()
    out = tmp_path / "private" / "out"
    assert cli.main(direction_args(inputs, out)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "ok", "records": 24, "snippets": 11, "library": 9,
        "holdout": 2, "audit": 9, "deleted_ids_matched": 0, "blocked_messages": 0,
        "excluded_by_deleted": 0,
        "style_eligible": 11,
        "excluded_by_veiled": 0, "flagged_suspect": 0, **ZERO_MEDIA_COUNTS,
    }
    assert "合成" not in captured.out and "another-" not in captured.out


@pytest.mark.parametrize("direction", ["outbound", "inbound"])
@pytest.mark.parametrize("parsed_batch", [False, True])
def test_direction_mapping_rejects_multiple_speakers(
    monkeypatch, tmp_path, inputs, capsys, direction, parsed_batch,
):
    cli = load_cli(monkeypatch, tmp_path)
    if parsed_batch:
        batch = StandardJsonAdapter().parse(inputs["chat"])
        next(record for record in batch.records if record.direction.value == direction
             ).source_speaker_id = "synthetic-extra-speaker"
        monkeypatch.setattr(cli.StandardJsonAdapter, "parse", lambda self, path: batch)
    else:
        messages = json.loads(inputs["chat"].read_text())
        next(message for message in messages if message["direction"] == direction
             )["source_speaker_id"] = "synthetic-extra-speaker"
        inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
    out = tmp_path / "private" / "out"
    assert cli.main(direction_args(inputs, out)) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status":"failed"}'
    assert captured.err == ""
    assert not out.exists()


def test_mapping_options_are_mutually_exclusive(monkeypatch, tmp_path, inputs, capsys):
    cli = load_cli(monkeypatch, tmp_path)
    assert cli.main([*args(inputs, tmp_path / "private" / "out"), "--direction-mapping"]) == 1
    assert capsys.readouterr().out.strip() == '{"status":"failed"}'


def test_cli_deleted_ids_and_inclusive_ranges_apply_before_split(
    monkeypatch, tmp_path, inputs, capsys,
):
    from companion.importer.models import ConfirmedSpeakerMapping
    from companion.importer.speaker_mapping import apply_confirmed_mapping
    from companion.privacy import RiskCategory, build_paired_snippets, read_snippets
    cli = load_cli(monkeypatch, tmp_path)
    records = apply_confirmed_mapping(
        StandardJsonAdapter().parse(inputs["chat"]).records,
        ConfirmedSpeakerMapping(roles=json.loads(inputs["mapping"].read_text())),
    )
    original = build_paired_snippets(records, created_at=datetime.now(UTC),
                                     risk_terms={RiskCategory.PRIVATE_EVENT: ["合成排除标记"]})
    out = tmp_path / "private" / "out"
    out.mkdir(parents=True)
    deleted = {"snippet_ids": [original[0].snippet_id, original[9].snippet_id],
               "ranges": [[original[9].started_at.isoformat(),
                           original[10].started_at.isoformat()]]}
    exclusion_file = out / "deleted_ids.json"
    exclusion_file.write_text(json.dumps(deleted), encoding="utf-8")
    for _ in range(2):
        assert cli.main(direction_args(inputs, out)) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert json.loads(captured.out) == {
            "status": "ok", "records": 24, "snippets": 8, "library": 7,
            "holdout": 1, "audit": 7, "deleted_ids_matched": 2, "blocked_messages": 6,
            "excluded_by_deleted": 3,
            "style_eligible": 8,
            "excluded_by_veiled": 0, "flagged_suspect": 0, **ZERO_MEDIA_COUNTS,
        }
        assert [item.snippet_id for item in read_snippets(out / "library.jsonl")] == [
            item.snippet_id for item in original[1:8]
        ]
        assert [item.snippet_id for item in read_snippets(out / "holdout.jsonl")] == [
            original[8].snippet_id
        ]
        assert json.loads(exclusion_file.read_text()) == deleted


def test_cli_entities_full_text_context_and_style_counts(monkeypatch, tmp_path, inputs, capsys):
    from companion.privacy import read_snippets
    cli = load_cli(monkeypatch, tmp_path)
    messages = json.loads(inputs["chat"].read_text())
    messages[0]["text"] = "合成名字" + "长" * 81
    messages[4]["text"] = messages[5]["text"] = "合成名字 未列姓名 未列地点"
    terms = json.loads(inputs["terms"].read_text())
    terms["redaction_terms"]["name"].append("未列姓名")
    terms["redaction_terms"]["address"] = ["未列地点"]
    inputs["terms"].write_text(json.dumps(terms), encoding="utf-8")
    for i in range(3):
        messages.append({**messages[0], "message_id": f"extra-partner-{i}",
                         "timestamp": (datetime(2026, 10, 6, tzinfo=UTC)
                                       + timedelta(seconds=i + 1)).isoformat(),
                         "text": f"合成名字连发{i}"})
    # Put the next pair in the same segment to exercise complete preceding turns.
    messages[2]["timestamp"] = "2026-10-06T00:02:00+00:00"
    messages[3]["timestamp"] = "2026-10-06T00:03:00+00:00"
    inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
    entities_path = tmp_path / "entities.json"
    entities_path.write_text(json.dumps({"version": 1, "entities": [
        {"id": "third", "kind": "person", "aliases": ["合成名字"], "pseudonym": "小丙"},
    ]}), encoding="utf-8")
    out = tmp_path / "private" / "out"
    assert cli.main([*direction_args(inputs, out),
        "--entities", str(entities_path), "--context-turns", "10", "--full-text",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "ok", "records": 27, "snippets": 11, "library": 9,
        "holdout": 2, "audit": 9, "deleted_ids_matched": 0, "blocked_messages": 0,
        "excluded_by_deleted": 0,
        "style_eligible": 10,
        "excluded_by_veiled": 0, "flagged_suspect": 0, **ZERO_MEDIA_COUNTS,
    }
    library = read_snippets(out / "library.jsonl")
    assert library[0].partner_turns == ["小丙" + "长" * 81] + [f"小丙连发{i}" for i in range(3)]
    assert library[1].context_turns == [
        "她：" + "小丙" + "长" * 81, *[f"她：小丙连发{i}" for i in range(3)], "他：小丙收到",  # noqa: RUF001
    ]
    assert library[0].as_style_snippet() is None
    assert library[1].as_style_snippet() is not None
    assert library[2].partner_turns == library[2].owner_reply == ["小丙 [NAME] [ADDRESS]"]
    assert "小丙" not in captured.out and "合成名字" not in captured.out


def test_cli_exclude_near_nontext(monkeypatch, tmp_path, inputs, capsys):
    cli = load_cli(monkeypatch, tmp_path)
    messages = json.loads(inputs["chat"].read_text())
    messages.append({**messages[0], "message_id": "synthetic-image",
                     "timestamp": "2026-10-06T00:00:30+00:00",
                     "content_type": "image", "text": None})
    inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
    assert cli.main([*args(inputs, tmp_path / "private" / "out"), "--exclude-near-nontext"]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["snippets"] == counts["style_eligible"] == 10
    assert counts["excluded_by_deleted"] == 0
    assert {key: counts[key] for key in ZERO_MEDIA_COUNTS} == {
        **ZERO_MEDIA_COUNTS, "excluded_by_media": 1,
    }


def test_cli_veiled_counts_raw_hits_and_applies_deletions_before_flag_count(
    monkeypatch, tmp_path, inputs, capsys,
):
    from companion.privacy import read_snippets
    cli = load_cli(monkeypatch, tmp_path)
    messages = json.loads(inputs["chat"].read_text())
    messages[0]["text"] = "合成名字法*死*你"
    messages[2]["text"] = "合成名字袜 子 袜子"
    messages[5]["text"] = "合成名字上床"
    messages[6]["text"] = "合成名字胸"
    inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
    terms = json.loads(inputs["terms"].read_text())
    terms["redaction_terms"]["name"] += ["法*死*你", "袜 子", "袜子", "上床", "胸"]
    inputs["terms"].write_text(json.dumps(terms), encoding="utf-8")
    out = tmp_path / "private" / "out"
    out.mkdir(parents=True)
    (out / "deleted_ids.json").write_text(json.dumps({"snippet_ids": [], "ranges": [
        [messages[6]["timestamp"], messages[6]["timestamp"]],
    ]}), encoding="utf-8")
    assert cli.main([*args(inputs, out), "--veiled",
                     str(ROOT / "config/privacy/veiled_terms.json")]) == 0
    captured = capsys.readouterr()
    counts = json.loads(captured.out)
    assert counts == {"status": "ok", "records": 24, "snippets": 9, "library": 8,
                      "holdout": 1, "audit": 8, "deleted_ids_matched": 0, "blocked_messages": 2,
                      "excluded_by_deleted": 1,
                      "style_eligible": 9, "excluded_by_veiled": 1, "flagged_suspect": 2,
                      **ZERO_MEDIA_COUNTS}
    assert captured.err == ""
    assert all(isinstance(value, int) for key, value in counts.items() if key != "status")
    snippets = read_snippets(out / "library.jsonl") + read_snippets(out / "holdout.jsonl")
    assert [item.privacy_flags for item in snippets if item.privacy_flags] == [
        ["body:袜子"], ["ambiguous_sexual:上床"],
    ]
    for bait in ("法*死*你", "袜 子", "袜子", "上床", "胸", "合成名字"):
        assert bait not in captured.out


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("with_veiled", [False, True])
def test_d6_cli_media_counts_and_context_outputs(
    monkeypatch, tmp_path, inputs, capsys, enabled, with_veiled,
):
    from companion.privacy import read_snippets
    cli = load_cli(monkeypatch, tmp_path)
    templates = json.loads(inputs["chat"].read_text())[:2]
    cases = [
        [(False, "text", "问题"), (True, "image", None), (True, "text", "回答")],
        [(False, "text", "这张内容"), (True, "text", "回答")],
        [(True, "image", None), (False, "text", "看看!"), (True, "text", "回答")],
        [(True, "text", "旧前文"), (True, "image", None),
         (True, "text", "普通前文"), (False, "text", "问题"), (True, "text", "回答")],
        [(False, "text", "看看!"), (True, "text", "回答")],
        [(True, "text", "法*死*你"), (True, "image", None),
         (False, "text", "普通问题"), (True, "text", "回答")],
        # First-match attribution: pair media wins over deictic and short reaction.
        [(True, "image", None), (False, "text", "看看"),
         (True, "image", None), (True, "text", "上图内容")],
        # A truncated pair can also be excluded by deictic.
        [(True, "image", None), (False, "text", "下图内容"), (True, "text", "回答")],
    ]
    messages = []
    for case_index, turns in enumerate(cases):
        for turn_index, (owner, content_type, text) in enumerate(turns):
            messages.append({**templates[int(owner)],
                "message_id": f"synthetic-d6-{case_index}-{turn_index}",
                "timestamp": (datetime(2026, 10, 6, tzinfo=UTC)
                              + timedelta(hours=case_index, minutes=turn_index)).isoformat(),
                "content_type": content_type, "text": text,
            })
    inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
    out = tmp_path / "private" / "out"
    options = ["--context-turns", "2"]
    if enabled:
        options.append("--exclude-near-nontext")
    if with_veiled:
        options += ["--veiled", str(ROOT / "config/privacy/veiled_terms.json")]
    assert cli.main([*args(inputs, out), *options]) == 0
    captured = capsys.readouterr()
    counts = json.loads(captured.out)
    assert captured.err == ""
    assert {key: counts[key] for key in ZERO_MEDIA_COUNTS} == ({
        "excluded_by_media": 2, "excluded_by_deictic": 2,
        "excluded_by_short_reaction": 1, "context_truncated_at_media": 4,
    } if enabled else ZERO_MEDIA_COUNTS)
    assert counts["snippets"] == (3 if enabled else 8) - int(with_veiled)
    assert counts["excluded_by_veiled"] == int(with_veiled)
    snippets = read_snippets(out / "library.jsonl") + read_snippets(out / "holdout.jsonl")
    retained_context = next(item for item in snippets if item.started_at.hour == 3).context_turns
    assert retained_context == (
        ["他：普通前文"] if enabled else ["他：旧前文", "他：普通前文"]  # noqa: RUF001
    )
    assert "法*死*你" not in captured.out and "普通前文" not in captured.out


@pytest.mark.parametrize("bad_lexicon", ["missing", "malformed", "invalid"])
def test_cli_veiled_load_failures_are_sanitized(monkeypatch, tmp_path, inputs, capsys, bad_lexicon):
    cli = load_cli(monkeypatch, tmp_path)
    path = tmp_path / "synthetic-sensitive-name.json"
    if bad_lexicon != "missing":
        path.write_text("{合成敏感词" if bad_lexicon == "malformed" else
                        json.dumps({"version": 1, "high": {"sexual": [123]}}), encoding="utf-8")
    out = tmp_path / "private" / "out"
    assert cli.main([*args(inputs, out), "--veiled", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status":"failed"}'
    assert captured.err == ""
    assert not out.exists()


@pytest.mark.parametrize("deletion", ["id", "range"])
@pytest.mark.parametrize("enabled", [False, True])
def test_deleted_pair_messages_block_adjacent_context(
    monkeypatch, tmp_path, inputs, capsys, deletion, enabled,
):
    from companion.privacy import read_snippets
    cli = load_cli(monkeypatch, tmp_path)
    messages = json.loads(inputs["chat"].read_text())
    messages[0]["text"] = "synthetic-deleted-question"
    messages[1]["text"] = "synthetic-deleted-answer"
    # Include a raw media record within the deleted pair's interval.
    hidden = dict(messages[0], message_id="synthetic-hidden",
                  timestamp=(datetime.fromisoformat(messages[0]["timestamp"])
                             + timedelta(seconds=30)).isoformat(),
                  content_type="image", text=None)
    messages.insert(1, hidden)
    inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
    out = tmp_path / "private" / "out"
    baseline_args = [*args(inputs, out), "--context-turns", "4"]
    # The candidate pass uses identical options; keep this pair eligible with media enabled.
    if enabled:
        messages.remove(hidden)
        inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
        baseline_args += ["--exclude-near-nontext"]
    assert cli.main(baseline_args) == 0
    capsys.readouterr()
    original = read_snippets(out / "library.jsonl") + read_snippets(out / "holdout.jsonl")
    first = min(original, key=lambda item: item.started_at)
    saved = {"snippet_ids": [first.snippet_id] if deletion == "id" else [],
             "ranges": [[first.started_at.isoformat(), first.started_at.isoformat()]]
             if deletion == "range" else []}
    (out / "deleted_ids.json").write_text(json.dumps(saved), encoding="utf-8")
    for _ in range(2):
        assert cli.main(baseline_args) == 0
        captured = capsys.readouterr()
        counts = json.loads(captured.out)
        assert captured.err == ""
        assert len(captured.out.splitlines()) == 1
        assert counts["blocked_messages"] == (2 if enabled else 3)
        assert counts["deleted_ids_matched"] == (1 if deletion == "id" else 0)
        assert counts["excluded_by_deleted"] == 1
        assert counts["snippets"] == 10
        assert all(isinstance(value, int) for key, value in counts.items() if key != "status")
        retained = read_snippets(out / "library.jsonl") + read_snippets(out / "holdout.jsonl")
        adjacent = min(retained, key=lambda item: item.started_at)
        assert adjacent.context_turns == []
        for filename in ("library.jsonl", "holdout.jsonl", "audit_sample.jsonl"):
            assert "synthetic-deleted-" not in (out / filename).read_text()
        assert "synthetic-deleted-" not in captured.out


@pytest.mark.parametrize("rule", ["media", "veiled", "risk"])
@pytest.mark.parametrize("deletion", ["id", "range"])
def test_relaxed_deleted_coverage_survives_formal_exclusions(
    monkeypatch, tmp_path, inputs, capsys, rule, deletion,
):
    from companion.importer.models import ConfirmedSpeakerMapping
    from companion.importer.speaker_mapping import apply_confirmed_mapping
    from companion.privacy import build_paired_snippets, read_snippets

    cli = load_cli(monkeypatch, tmp_path)
    templates = json.loads(inputs["chat"].read_text())[:2]
    turns = [(False, "synthetic-deleted-question", 0),
             (True, "synthetic-deleted-answer", 2),
             (False, "ordinary-question", 3), (True, "ordinary-answer", 4),
             (False, "later-question", 60), (True, "later-answer", 61)]
    messages = [{**templates[int(owner)], "message_id": f"synthetic-relaxed-{i}",
                 "timestamp": (datetime(2026, 10, 6, tzinfo=UTC)
                               + timedelta(minutes=minute)).isoformat(), "text": text}
                for i, (owner, text, minute) in enumerate(turns)]
    options = ["--context-turns", "1"]
    if rule == "media":
        messages.insert(1, {**messages[0], "message_id": "synthetic-relaxed-media",
                            "timestamp": (datetime(2026, 10, 6, tzinfo=UTC)
                                          + timedelta(minutes=1)).isoformat(),
                            "content_type": "image", "text": None})
        options += ["--exclude-near-nontext"]
    elif rule == "veiled":
        messages[0]["text"] += "法*死*你"
        options += ["--veiled", str(ROOT / "config/privacy/veiled_terms.json")]
    else:
        messages[0]["text"] += "合成排除标记"
    inputs["chat"].write_text(json.dumps(messages), encoding="utf-8")
    records = apply_confirmed_mapping(
        StandardJsonAdapter().parse(inputs["chat"]).records,
        ConfirmedSpeakerMapping(roles=json.loads(inputs["mapping"].read_text())),
    )
    relaxed = build_paired_snippets(records, created_at=datetime.now(UTC),
                                    excluded_risks=frozenset(), context_turns=1)
    first = relaxed[0]
    out = tmp_path / "private" / "out"
    assert cli.main([*args(inputs, out), *options]) == 0
    capsys.readouterr()
    baseline = read_snippets(out / "library.jsonl") + read_snippets(out / "holdout.jsonl")
    assert first.snippet_id not in {item.snippet_id for item in baseline}
    # With media/veiled rules, the deleted answer would otherwise return as context.
    if rule != "risk":
        assert any("synthetic-deleted-answer" in " ".join(item.context_turns)
                   for item in baseline)
    saved = {"snippet_ids": [first.snippet_id, first.snippet_id, "snip-unmatched"]
             if deletion == "id" else ["snip-unmatched"],
             "ranges": [[first.started_at.isoformat(), first.started_at.isoformat()]]
             if deletion == "range" else []}
    (out / "deleted_ids.json").write_text(json.dumps(saved), encoding="utf-8")
    assert cli.main([*args(inputs, out), *options]) == 0
    captured = capsys.readouterr()
    counts = json.loads(captured.out)
    assert counts["blocked_messages"] == (3 if rule == "media" else 2)
    assert counts["deleted_ids_matched"] == (1 if deletion == "id" else 0)
    assert counts["excluded_by_deleted"] == 0
    retained = read_snippets(out / "library.jsonl") + read_snippets(out / "holdout.jsonl")
    assert retained
    assert all(item.context_turns == [] for item in retained)
    for filename in ("library.jsonl", "holdout.jsonl", "audit_sample.jsonl"):
        assert "synthetic-deleted-" not in (out / filename).read_text()
    assert captured.err == "" and len(captured.out.splitlines()) == 1
    assert all(isinstance(value, int) for key, value in counts.items() if key != "status")
