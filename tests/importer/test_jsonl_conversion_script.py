"""Synthetic-only contract checks for the bounded conversion script."""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from companion.importer.standard_json import StandardJsonAdapter
from companion.runtime.settings import Settings

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/convert_chat_jsonl.py"
spec = importlib.util.spec_from_file_location("conversion_script", SCRIPT)
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


def row(**changes):
    result = dict(
        timestamp=converter.CUTOFF - 100,
        type=1,
        server_id=1,
        is_sender=True,
        sender_id="synthetic-owner",
        sender="虚构",
        conversation_id="synthetic-chat",
        text="  中文\n🙂\t ",
        raw="UNDECODED_SYNTHETIC_RAW",
    )
    result.update(changes)
    return result


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    source = tmp_path / "synthetic.jsonl"
    settings = Settings(
        environment="test",
        log_level="ERROR",
        private_data_dir=root,
        runtime_data_dir=tmp_path,
        test_results_dir=tmp_path,
    )
    return source, settings


def write(source, rows):
    source.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n")


def test_boundary_fidelity_adapter_and_source(inputs):
    source, settings = inputs
    write(
        source,
        [
            row(timestamp=converter.CUTOFF - 1),
            row(timestamp=converter.CUTOFF, text="EXCLUDED_SYNTHETIC"),
            row(
                timestamp=converter.CUTOFF - 2,
                server_id=2,
                is_sender=False,
                sender_id="synthetic-other",
                sender="",
            ),
            row(type=49, server_id=0),
        ],
    )
    before = source.read_bytes(), source.stat()
    result = converter.convert(source, settings, "synthetic-chat")
    payload = json.loads(result.output.read_text())
    assert (result.included, result.nontext, result.later) == (2, 1, 1)
    assert len(payload[0]) == 9
    assert payload[0]["text"] == row()["text"]
    assert payload[0]["source_speaker_id"] == "synthetic-owner"
    assert payload[1]["source_speaker_name"] is None
    assert payload[0]["timestamp"] == "2026-07-17T23:59:59+08:00"
    assert "raw" not in payload[0]
    batch = StandardJsonAdapter().parse(result.output)
    assert len(batch.records) == 2
    assert not hasattr(batch.records[0], "speaker_role")
    after = source.stat()
    assert source.read_bytes() == before[0]
    assert (after.st_ino, after.st_dev, after.st_mode, after.st_mtime_ns) == (
        before[1].st_ino,
        before[1].st_dev,
        before[1].st_mode,
        before[1].st_mtime_ns,
    )
    assert result.output.stat().st_mode & 0o777 == 0o600
    for directory in (
        result.output.parent,
        result.output.parent.parent,
        result.output.parent.parent.parent,
    ):
        assert directory.stat().st_mode & 0o777 == 0o700


def test_custom_through_date_boundary_and_filename(inputs):
    source, settings = inputs
    zone = timezone(timedelta(hours=8))
    included = int(datetime(2026, 8, 1, 23, 59, 59, tzinfo=zone).timestamp())
    later = int(datetime(2026, 8, 2, tzinfo=zone).timestamp())
    write(source, [row(timestamp=included), row(timestamp=later, server_id=2)])

    result = converter.convert(source, settings, "synthetic-chat", through_date=date(2026, 8, 1))

    assert result.output.name == "chat-through-2026-08-01.json"
    assert (result.included, result.later) == (1, 1)


def test_cli_through_date_and_invalid_format(inputs, monkeypatch, capsys):
    source, settings = inputs
    write(source, [row()])
    args = [str(SCRIPT), "--source", str(source), "--private-root", str(settings.private_data_dir),
            "--conversation-id", "synthetic-chat", "--through-date", "2026-08-01"]
    monkeypatch.setattr(sys, "argv", args)
    assert converter.main() == 0
    output_names = {p.name for p in settings.private_data_dir.rglob("*.json")}
    assert "chat-through-2026-08-01.json" in output_names
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", [*args[:-1], "2026/08/01"])
    assert converter.main() == 1
    captured = capsys.readouterr()
    assert captured.err == "conversion_failed\n"


def test_excluded_values_never_decoded(inputs, monkeypatch):
    source, settings = inputs
    write(
        source,
        [
            row(),
            row(type=50, text="FORBIDDEN_SYNTHETIC_BODY"),
            row(
                timestamp=converter.CUTOFF,
                type="FORBIDDEN_SYNTHETIC_TYPE",
                text="FORBIDDEN_SYNTHETIC_BODY",
            ),
        ],
    )
    original = converter._decode
    seen = []

    def observe(token):
        seen.append(token)
        assert b"FORBIDDEN_SYNTHETIC" not in token
        assert b"UNDECODED_SYNTHETIC_RAW" not in token
        return original(token)

    monkeypatch.setattr(converter, "_decode", observe)
    assert converter.convert(source, settings, "synthetic-chat").included == 1
    assert seen


@pytest.mark.parametrize(
    "changes",
    [
        {"timestamp": True},
        {"timestamp": "bad"},
        {"timestamp": 1e100},
        {"type": True},
        {"type": "1"},
        {"server_id": 0},
        {"server_id": -1},
        {"server_id": True},
        {"server_id": "1"},
        {"is_sender": 1},
        {"sender": None},
        {"sender_id": ""},
        {"text": None},
    ],
)
def test_invalid_included_fails_closed(inputs, changes):
    source, settings = inputs
    write(source, [row(**changes)])
    with pytest.raises(converter.ConversionError, match=r"^conversion_failed$"):
        converter.convert(source, settings, "synthetic-chat")
    assert list(settings.private_data_dir.iterdir()) == []


@pytest.mark.parametrize(
    "second",
    [
        row(server_id=1),
        row(server_id=2, conversation_id="other"),
        row(server_id=2, sender_id="other"),
        row(server_id=2, is_sender=False),
    ],
)
def test_identity_conflicts(inputs, second):
    source, settings = inputs
    write(source, [row(), second])
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")


@pytest.mark.parametrize(
    "line",
    [
        '{"timestamp":1,"timestamp":2}',
        '{"timestamp":1,"\\u0074imestamp":2}',
        '{"timestamp":1784304000,"type":49,"raw":"bad\\q"}',
        '{"timestamp":NaN}',
        "{}",
        "[]",
    ],
)
def test_ambiguous_or_malformed_json(inputs, line):
    source, settings = inputs
    source.write_text(line)
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")


@pytest.mark.parametrize(
    "kind",
    ["source_link", "root_link", "ancestor_link", "root_mode", "existing_link", "existing_mode"],
)
def test_path_safety(inputs, tmp_path, kind):
    source, settings = inputs
    write(source, [row()])
    root = settings.private_data_dir
    if kind == "source_link":
        alias = tmp_path / "alias"
        alias.symlink_to(source)
        source = alias
    elif kind == "root_link":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        settings.private_data_dir = alias
    elif kind == "ancestor_link":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        settings.private_data_dir = alias / "private"
    elif kind == "root_mode":
        root.chmod(0o755)
    elif kind == "existing_link":
        (root / "imports").symlink_to(tmp_path, target_is_directory=True)
    else:
        (root / "imports").mkdir(mode=0o755)
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")


def test_validation_failure_cleanup_preserves_existing(inputs, monkeypatch):
    source, settings = inputs
    write(source, [row()])
    imports = settings.private_data_dir / "imports"
    imports.mkdir(mode=0o700)
    sentinel = imports / "sentinel"
    sentinel.write_text("synthetic")

    def reject(*args):
        raise ValueError("SECRET_SYNTHETIC_ERROR")

    monkeypatch.setattr(converter.StandardJsonAdapter, "parse", reject)
    with pytest.raises(converter.ConversionError) as caught:
        converter.convert(source, settings, "synthetic-chat")
    assert str(caught.value) == "conversion_failed"
    assert sentinel.read_text() == "synthetic"
    assert sorted(item.name for item in imports.iterdir()) == ["sentinel"]


def test_repeat_run_no_overwrite(inputs):
    source, settings = inputs
    write(source, [row()])
    first = converter.convert(source, settings, "synthetic-chat").output
    original = first.read_bytes()
    second = converter.convert(source, settings, "synthetic-chat").output
    assert first != second
    assert first.read_bytes() == original


def test_cli_safe_success_and_failure(inputs):
    source, settings = inputs
    write(source, [row()])
    command = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--private-root",
        str(settings.private_data_dir),
        "--conversation-id",
        "synthetic-chat",
    ]
    env = dict(os.environ, PYTHONPATH=str(SCRIPT.parent.parent / "src"))
    success = subprocess.run(command, text=True, capture_output=True, env=env)
    assert success.returncode == 0
    assert json.loads(success.stdout)["included"] == 1
    assert success.stderr == ""
    write(source, [row(text=None, sender="PRIVATE_SYNTHETIC_NAME")])
    failed = subprocess.run(command, text=True, capture_output=True, env=env)
    assert failed.returncode == 1
    assert failed.stdout == ""
    assert failed.stderr == "conversion_failed\n"
    assert str(source) not in success.stdout + failed.stderr


@pytest.mark.parametrize(
    "field", ["timestamp", "type", "server_id", "is_sender", "sender_id", "sender", "text"]
)
def test_missing_required_fields(inputs, field):
    source, settings = inputs
    message = row()
    del message[field]
    write(source, [message])
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")


def test_explicit_conversation_without_source_field(inputs):
    source, settings = inputs
    message = row()
    del message["conversation_id"]
    write(source, [message])
    result = converter.convert(source, settings, "synthetic-chat")
    assert json.loads(result.output.read_text())[0]["conversation_id"] == "synthetic-chat"
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "")


def test_source_hardlink_rejected(inputs):
    source, settings = inputs
    write(source, [row()])
    os.link(source, source.with_suffix(".alias"))
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")


def test_fresh_directory_collision_preserved(inputs, monkeypatch):
    source, settings = inputs
    write(source, [row()])
    imports = settings.private_data_dir / "imports"
    imports.mkdir(mode=0o700)
    standard = imports / "standard-v1"
    standard.mkdir(mode=0o700)
    run = standard / "synthetic-collision"
    run.mkdir(mode=0o700)
    sentinel = run / "sentinel"
    sentinel.write_text("synthetic")

    class Fixed:
        hex = "synthetic-collision"

    monkeypatch.setattr(converter.uuid, "uuid4", lambda: Fixed())
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")
    assert sentinel.read_text() == "synthetic"


def test_publication_collision_preserved(inputs, monkeypatch):
    source, settings = inputs
    write(source, [row()])
    original = converter.os.link

    def collide(src, dst, **kwargs):
        fd = os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=kwargs["dst_dir_fd"])
        os.write(fd, b"existing-synthetic")
        os.close(fd)
        return original(src, dst, **kwargs)

    monkeypatch.setattr(converter.os, "link", collide)
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")
    files = list(settings.private_data_dir.rglob("*.json"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"existing-synthetic"
    assert not list(settings.private_data_dir.rglob(".pending.json"))


@pytest.mark.parametrize("kind", [3, 34, 43, 47, 49, 50, 10000, 10002])
def test_every_nontext_type_and_zero_id_skipped(inputs, kind):
    source, settings = inputs
    write(source, [row(), row(type=kind, server_id=0, text=None, sender=None, is_sender=None)])
    result = converter.convert(source, settings, "synthetic-chat")
    assert (result.included, result.nontext) == (1, 1)


def test_pending_hardlink_rejected(inputs, monkeypatch):
    source, settings = inputs
    write(source, [row()])
    original = converter.StandardJsonAdapter.parse
    alias = source.with_suffix(".alias")

    def link_then_parse(self, path):
        os.link(path, alias)
        return original(self, path)

    monkeypatch.setattr(converter.StandardJsonAdapter, "parse", link_then_parse)
    with pytest.raises(converter.ConversionError) as caught:
        converter.convert(source, settings, "synthetic-chat")
    assert caught.value.__context__ is None
    assert list(settings.private_data_dir.iterdir()) == []
    assert alias.exists()


def test_source_change_during_validation_refuses_publication(inputs, monkeypatch):
    source, settings = inputs
    write(source, [row()])
    original = converter.StandardJsonAdapter.parse

    def mutate(self, path):
        write(source, [row(text="CHANGED_SYNTHETIC")])
        return original(self, path)

    monkeypatch.setattr(converter.StandardJsonAdapter, "parse", mutate)
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")
    assert list(settings.private_data_dir.iterdir()) == []


def test_ancestor_swapped_to_link_during_validation(inputs, monkeypatch):
    source, settings = inputs
    write(source, [row()])
    original = converter.StandardJsonAdapter.parse
    root = settings.private_data_dir
    moved = root.with_name("moved-private")

    def swap(self, path):
        result = original(self, path)
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
        return result

    monkeypatch.setattr(converter.StandardJsonAdapter, "parse", swap)
    with pytest.raises(converter.ConversionError):
        converter.convert(source, settings, "synthetic-chat")
    assert root.is_symlink()
    assert list(moved.iterdir()) == []


def test_markers_disabled_preserves_exact_output(inputs):
    source, settings = inputs
    write(source, [row(), row(type=3, server_id=2, text="SYNTHETIC_SECRET")])
    result = converter.convert(source, settings, "synthetic-chat", keep_nontext_markers=False)
    assert (
        result.output.read_bytes()
        == (
            '[{"schema_version": 1, "conversation_id": "synthetic-chat", "message_id": "1", '
            '"timestamp": "2026-07-17T23:58:20+08:00", '
            '"source_speaker_id": "synthetic-owner", "source_speaker_name": "虚构", '
            '"direction": "outbound", "content_type": "text", "text": "  中文\\n🙂\\t "}]\n'
        ).encode()
    )


def test_nontext_markers_mapping_and_no_content_access(inputs, monkeypatch):
    source, settings = inputs
    kinds = [3, 34, 43, 47, 10000, 10002, 49, 50, 999]
    messages = [
        row(
            type=kind,
            server_id=index,
            text="FORBIDDEN_SYNTHETIC_TEXT",
            content={"payload": "FORBIDDEN_SYNTHETIC_CONTENT"},
            sender="" if index == 1 else "虚构",
        )
        for index, kind in enumerate(kinds, 1)
    ]
    del messages[-1]["text"]
    write(source, messages)
    original = converter._decode

    def observe(token):
        assert b"FORBIDDEN_SYNTHETIC" not in token
        assert b"UNDECODED_SYNTHETIC_RAW" not in token
        return original(token)

    monkeypatch.setattr(converter, "_decode", observe)
    result = converter.convert(source, settings, "synthetic-chat", keep_nontext_markers=True)
    payload = json.loads(result.output.read_text())
    assert [item["content_type"] for item in payload] == [
        "image",
        "voice",
        "video",
        "emoji",
        "system",
        "system",
        "other",
        "other",
        "other",
    ]
    for index, item in enumerate(payload, 1):
        assert item == dict(
            schema_version=1,
            conversation_id="synthetic-chat",
            message_id=str(index),
            timestamp="2026-07-17T23:58:20+08:00",
            source_speaker_id="synthetic-owner",
            source_speaker_name=None if index == 1 else "虚构",
            direction="outbound",
            content_type=item["content_type"],
            text=None,
        )
    assert "FORBIDDEN_SYNTHETIC" not in result.output.read_text()
    assert "UNDECODED_SYNTHETIC_RAW" not in result.output.read_text()
    assert (result.included, result.nontext, result.nontext_kept, result.nontext_skipped) == (
        9,
        9,
        9,
        0,
    )
    batch = StandardJsonAdapter().parse(result.output)
    assert len(batch.records) == 9
    assert all(record.text is None for record in batch.records)


@pytest.mark.parametrize(
    "field,value",
    [
        ("server_id", 0),
        ("server_id", -1),
        ("server_id", True),
        ("server_id", "2"),
        ("server_id", 1),
        ("is_sender", 1),
        ("is_sender", None),
        ("sender_id", ""),
        ("sender_id", None),
        ("sender_id", 2),
        ("timestamp", True),
        ("timestamp", "bad"),
        ("timestamp", None),
        ("timestamp", 10**100),
        ("timestamp", -(10**100)),
        ("sender", None),
        ("sender", 2),
        ("conversation_id", None),
        ("conversation_id", ""),
    ],
)
def test_invalid_nontext_markers_skipped(inputs, field, value):
    source, settings = inputs
    invalid = row(type=3, server_id=2)
    invalid[field] = value
    write(source, [row(), invalid, row(type=34, server_id=3)])
    result = converter.convert(source, settings, "synthetic-chat", keep_nontext_markers=True)
    assert (result.included, result.nontext, result.nontext_kept, result.nontext_skipped) == (
        2,
        2,
        1,
        1,
    )
    assert [item["message_id"] for item in json.loads(result.output.read_text())] == ["1", "3"]


@pytest.mark.parametrize("field", ["server_id", "is_sender", "sender_id", "timestamp", "sender"])
def test_missing_nontext_metadata_skipped(inputs, field):
    source, settings = inputs
    invalid = row(type=3, server_id=2)
    del invalid[field]
    write(source, [row(), invalid])
    result = converter.convert(source, settings, "synthetic-chat", keep_nontext_markers=True)
    assert (result.included, result.nontext_kept, result.nontext_skipped) == (1, 0, 1)


def test_skipped_marker_does_not_reserve_identity(inputs):
    source, settings = inputs
    write(source, [row(type=3, server_id=2, is_sender=None), row(server_id=2)])
    result = converter.convert(source, settings, "synthetic-chat", keep_nontext_markers=True)
    assert (result.included, result.nontext_skipped) == (1, 1)


@pytest.mark.parametrize("first_kind,second_kind", [(1, 3), (3, 1), (3, 34)])
@pytest.mark.parametrize(
    "changes",
    [
        {"sender_id": "synthetic-conflict"},
        {"is_sender": False},
    ],
)
def test_nontext_direction_conflicts_still_raise(first_kind, second_kind, changes):
    messages = [row(type=first_kind), row(type=second_kind, server_id=2, **changes)]
    lines = [(json.dumps(message) + "\n").encode() for message in messages]
    with pytest.raises(ValueError):
        converter._read_records(lines, "synthetic-chat", keep_nontext_markers=True)


@pytest.mark.parametrize("keep", [False, True])
def test_cli_marker_flag_counts_and_cutoff(inputs, monkeypatch, capsys, keep):
    source, settings = inputs
    write(
        source,
        [
            row(),
            row(type=3, server_id=2),
            row(type=34, server_id=0),
            row(type=43, server_id=3, timestamp=converter.CUTOFF),
            row(server_id=4, text="GARBLED_SYNTHETIC\ufffd"),
        ],
    )
    args = [
        str(SCRIPT),
        "--source",
        str(source),
        "--private-root",
        str(settings.private_data_dir),
        "--conversation-id",
        "synthetic-chat",
    ]
    if keep:
        args.append("--keep-nontext-markers")
    monkeypatch.setattr(sys, "argv", args)
    assert converter.main() == 0
    captured = capsys.readouterr()
    stats = json.loads(captured.out)
    expected = dict(
        included=3 if keep else 1,
        nontext=3,
        later=1,
        garbled_text=1,
        source_unchanged=True,
        source_atime_changed=stats["source_atime_changed"],
    )
    if keep:
        expected.update(nontext_kept=2, nontext_skipped=1)
    assert captured.out == json.dumps(expected) + "\n"
    assert captured.err == ""


@pytest.mark.parametrize("bad", ["�", "\x01", "\x00", "\x1f", "\x0b"])
def test_garbled_text_becomes_marker_when_kept(inputs, bad):
    source, settings = inputs
    write(source, [row(), row(server_id=2, text=f"GARBLED_SYNTHETIC{bad}x")])
    result = converter.convert(source, settings, "synthetic-chat", keep_nontext_markers=True)
    payload = json.loads(result.output.read_text())
    assert [item["content_type"] for item in payload] == ["text", "other"]
    assert payload[1]["text"] is None
    assert "GARBLED_SYNTHETIC" not in result.output.read_text()
    assert (
        result.included,
        result.nontext,
        result.nontext_kept,
        result.nontext_skipped,
        result.garbled_text,
    ) == (2, 1, 1, 0, 1)
    assert len(StandardJsonAdapter().parse(result.output).records) == 2


@pytest.mark.parametrize("bad", ["�", "\x01"])
def test_garbled_text_skipped_when_markers_disabled(inputs, bad):
    source, settings = inputs
    write(source, [row(), row(server_id=2, text=f"GARBLED_SYNTHETIC{bad}x")])
    result = converter.convert(source, settings, "synthetic-chat")
    payload = json.loads(result.output.read_text())
    assert [item["message_id"] for item in payload] == ["1"]
    assert (result.included, result.nontext, result.garbled_text) == (1, 1, 1)


def test_tab_newline_carriage_return_text_is_not_garbled(inputs):
    source, settings = inputs
    write(source, [row(text="中文\t\n\r🙂")])
    result = converter.convert(source, settings, "synthetic-chat", keep_nontext_markers=True)
    payload = json.loads(result.output.read_text())
    assert payload[0]["content_type"] == "text"
    assert payload[0]["text"] == "中文\t\n\r🙂"
    assert (result.nontext, result.garbled_text) == (0, 0)
