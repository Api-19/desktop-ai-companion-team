"""Synthetic file-backed snippet source tests using disposable libraries."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from companion.contracts.style import AddresseeRegister, StyleEvidence
from companion.privacy.paired_snippets import PairedSnippet
from companion.style.errors import StyleValidationError
from companion.style.snippet_evidence import SnippetStyleAdapter
from companion.style.snippet_source import FileSnippetStyleSource

DAY = datetime(2026, 10, 6, tzinfo=UTC)


def paired(index: int, *, reply: str | None = None) -> PairedSnippet:
    return PairedSnippet(
        snippet_id=f"synthetic-{index}",
        conversation_id="synthetic-conversation",
        started_at=DAY,
        partner_turns=[f"合成语境{index}"],
        owner_reply=[reply if reply is not None else "回" * (index + 1)],
        source_fingerprint="synthetic-fingerprint",
    )


def write_library(path: Path, snippets: list[PairedSnippet]) -> bytes:
    raw = "".join(item.model_dump_json() + "\n" for item in snippets).encode("utf-8")
    path.write_bytes(raw)
    return raw


def source(path: Path, *, partner_user_id: str = "partner"):
    return FileSnippetStyleSource(
        library_source=path,
        partner_user_id=partner_user_id,
        style_version="v1",
        clock=lambda: DAY,
    )


def adapter(snippets):
    return SnippetStyleAdapter(
        snippets=snippets, partner_user_id="partner", style_version="v1", clock=lambda: DAY
    )


def test_live_source_removes_deleted_snippets_before_next_evidence(tmp_path):
    path = tmp_path / "synthetic-library.jsonl"
    deleted, retained = paired(0), paired(1)
    raw = write_library(path, [deleted, retained])
    subject = source(path)
    first = subject.evidence_for("partner", "合成语境", [])
    assert first.similar == [deleted.to_style_snippet(), retained.to_style_snippet()]
    write_library(path, [retained])

    evidence = subject.evidence_for("partner", "合成语境", [])

    assert evidence.similar == [retained.to_style_snippet()]
    assert deleted.to_style_snippet() not in evidence.similar + evidence.general
    assert subject.content_revision("partner") != hashlib.sha256(raw).hexdigest()


def test_exclusion_only_change_refreshes_evidence_and_revision(tmp_path):
    path = tmp_path / "synthetic-library.jsonl"
    deleted, retained = paired(0), paired(1)
    raw = write_library(path, [deleted, retained])
    subject = source(path)
    revision = subject.content_revision("partner")
    assert subject.evidence_for("partner", "合成语境", []).similar == [
        deleted.to_style_snippet(), retained.to_style_snippet(),
    ]

    (tmp_path / "deleted_ids.json").write_text(
        json.dumps({"snippet_ids": [deleted.snippet_id], "ranges": []}), encoding="utf-8"
    )

    assert path.read_bytes() == raw
    assert subject.content_revision("partner") != revision
    evidence = subject.evidence_for("partner", "合成语境", [])
    assert evidence.similar == [retained.to_style_snippet()]
    assert deleted.to_style_snippet() not in evidence.similar + evidence.general


@pytest.mark.parametrize("size", [12, 60])
def test_unchanged_exclusions_preserve_three_call_reuse_window(tmp_path, size):
    path = tmp_path / "synthetic-library.jsonl"
    snippets = [paired(index) for index in range(size + 1)]
    write_library(path, snippets)
    exclusions = tmp_path / "deleted_ids.json"
    raw = json.dumps({"snippet_ids": [snippets[-1].snippet_id], "ranges": []}).encode()
    exclusions.write_bytes(raw)
    subject = source(path)
    revision = subject.content_revision("partner")
    control = adapter([item.to_style_snippet() for item in snippets[:-1]])
    results = []
    for _ in range(8):
        exclusions.write_bytes(raw)
        for _ in range(4):
            assert subject.content_revision("partner") == revision
        actual = subject.evidence_for("partner", "合成语境", [])
        assert actual == control.evidence_for("partner", "合成语境", [])
        results.append(actual)
    if size == 12:
        assert results[0].similar and results[0].general
        assert all(not (item.similar + item.general) for item in results[1:4])
        assert results[4] == results[0]


@pytest.mark.parametrize("failure", ["json", "utf8", "schema"])
@pytest.mark.parametrize("operation", ["evidence", "revision"])
def test_corrupt_exclusions_never_reuses_old_content(tmp_path, failure, operation):
    path = tmp_path / "synthetic-library.jsonl"
    deleted, retained = paired(0), paired(1)
    raw = write_library(path, [deleted, retained])
    exclusions = tmp_path / "deleted_ids.json"
    valid = json.dumps({"snippet_ids": [deleted.snippet_id], "ranges": []}).encode()
    exclusions.write_bytes(valid)
    subject = source(path)
    revision = subject.content_revision("partner")
    first = subject.evidence_for("partner", "合成语境", [])
    assert first.similar == [retained.to_style_snippet()]
    exclusions.write_bytes({"json": b"{broken", "utf8": b"\xff", "schema": b"{}"}[failure])

    for current in (operation, "revision", "evidence"):
        with pytest.raises(StyleValidationError) as caught:
            if current == "revision":
                subject.content_revision("partner")
            else:
                subject.evidence_for("partner", "合成语境", [])
        error = caught.value
        assert error.code == "snippet_source_unavailable"
        assert error.detail == "snippet source unavailable"
        assert str(error) == "snippet_source_unavailable: snippet source unavailable"
        assert error.__cause__ is None
        assert error.__context__ is None

    assert path.read_bytes() == raw
    exclusions.write_bytes(valid)
    assert subject.content_revision("partner") == revision
    assert subject.evidence_for("partner", "合成语境", []) == first


@pytest.mark.parametrize("failure", ["missing", "json", "utf8", "schema"])
@pytest.mark.parametrize("operation", ["evidence", "revision"])
def test_missing_or_corrupt_library_never_reuses_old_content(tmp_path, failure, operation):
    path = tmp_path / "synthetic-library.jsonl"
    raw = write_library(path, [paired(0)])
    subject = source(path)
    first = subject.evidence_for("partner", "合成语境", [])
    assert first.similar
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes({"json": b"{broken", "utf8": b"\xff", "schema": b"{}"}[failure])

    # Both entry points must fail on every retry, without exposing parser/path details.
    for current in (operation, "revision", "evidence"):
        with pytest.raises(StyleValidationError) as caught:
            if current == "revision":
                subject.content_revision("partner")
            else:
                subject.evidence_for("partner", "合成语境", [])
        error = caught.value
        assert error.code == "snippet_source_unavailable"
        assert error.detail == "snippet source unavailable"
        assert str(error) == "snippet_source_unavailable: snippet source unavailable"
        assert error.__cause__ is None
        assert error.__context__ is None

    # Identical bytes must reload after failure, including resetting old reuse history.
    path.write_bytes(raw)
    assert subject.content_revision("partner") == hashlib.sha256(raw).hexdigest()
    assert subject.evidence_for("partner", "合成语境", []) == first


@pytest.mark.parametrize("size", [12, 60])
def test_unchanged_library_preserves_three_call_reuse_window(tmp_path, size):
    path = tmp_path / "synthetic-library.jsonl"
    snippets = [paired(index) for index in range(size)]
    write_library(path, snippets)
    subject = source(path)
    control = adapter([item.to_style_snippet() for item in snippets])
    results = []
    for _ in range(8):
        actual = subject.evidence_for("partner", "合成语境", [])
        expected = control.evidence_for("partner", "合成语境", [])
        assert actual == expected
        assert actual.model_dump_json() == expected.model_dump_json()
        results.append(actual)
    if size == 12:
        assert results[0].similar and results[0].general
        assert all(not (item.similar + item.general) for item in results[1:4])
        assert results[4] == results[0]


def test_revision_reads_do_not_consume_reuse_window(tmp_path):
    path = tmp_path / "synthetic-library.jsonl"
    snippets = [paired(index) for index in range(60)]
    raw = write_library(path, snippets)
    subject = source(path)
    control = adapter([item.to_style_snippet() for item in snippets])
    for text, context in [("合成语境", []), ("无匹配", ["旧", "合成语境1", "新"])] * 4:
        for _ in range(4):
            assert subject.content_revision("partner") == hashlib.sha256(raw).hexdigest()
        assert subject.evidence_for("partner", text, context) == control.evidence_for(
            "partner", text, context
        )


def test_changed_library_rebuilds_even_when_size_and_mtime_are_preserved(tmp_path):
    path = tmp_path / "synthetic-library.jsonl"
    old = paired(0, reply="旧合成回复")
    raw = write_library(path, [old])
    subject = source(path)
    revision = subject.content_revision("partner")
    first = subject.evidence_for("partner", "合成语境", [])
    metadata = path.stat()
    new = paired(0, reply="新合成回复")
    changed = write_library(path, [new])
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert len(changed) == len(raw) == path.stat().st_size
    assert path.stat().st_mtime_ns == metadata.st_mtime_ns

    assert subject.content_revision("partner") == hashlib.sha256(changed).hexdigest()
    assert subject.content_revision("partner") != revision
    evidence = subject.evidence_for("partner", "合成语境", [])
    assert evidence != first
    assert evidence == adapter([new.to_style_snippet()]).evidence_for("partner", "合成语境", [])


def test_empty_library_and_source_recovery(tmp_path):
    path = tmp_path / "synthetic-library.jsonl"
    path.write_bytes(b"")
    subject = source(path)
    assert subject.content_revision("partner") == hashlib.sha256(b"").hexdigest()
    assert subject.evidence_for("partner", "合成语境", []) == adapter([]).evidence_for(
        "partner", "合成语境", []
    )
    path.unlink()
    with pytest.raises(StyleValidationError, match="snippet_source_unavailable"):
        subject.content_revision("partner")
    snippets = [paired(0)]
    raw = write_library(path, snippets)
    assert subject.content_revision("partner") == hashlib.sha256(raw).hexdigest()
    assert subject.evidence_for("partner", "合成语境", []) == adapter(
        [snippets[0].to_style_snippet()]
    ).evidence_for("partner", "合成语境", [])
    path.write_bytes(b"")
    assert subject.evidence_for("partner", "合成语境", []) == adapter([]).evidence_for(
        "partner", "合成语境", []
    )
    assert subject.content_revision("partner") == hashlib.sha256(b"").hexdigest()


def test_non_partner_revision_and_evidence_preserve_partner_history(tmp_path):
    path = tmp_path / "synthetic-library.jsonl"
    snippets = [paired(index) for index in range(60)]
    raw = write_library(path, snippets)
    subject = source(path)
    control = adapter([item.to_style_snippet() for item in snippets])
    for _ in range(6):
        assert subject.evidence_for("partner", "合成语境", []) == control.evidence_for(
            "partner", "合成语境", []
        )
        path.unlink()
        for user_id in ("owner", "visitor", "another"):
            assert subject.content_revision(user_id) == ""
            assert subject.evidence_for(user_id, "合成语境", []) == StyleEvidence(
                style_version="v1", addressee_register=AddresseeRegister.NEUTRAL
            )
        path.write_bytes(raw)


def test_parsing_filters_and_deduplicates_using_existing_rules(tmp_path):
    path = tmp_path / "synthetic-library.jsonl"
    first, second = paired(0), paired(1)
    duplicate = paired(2).model_copy(
        update={"partner_turns": first.partner_turns, "owner_reply": first.owner_reply}
    )
    long = paired(3, reply="回" * 1000)
    # Universal newlines and blank lines match read_snippets. A JSON string may
    # contain a literal Unicode line separator, which must not split a record.
    unicode_separator = paired(4, reply="合成\u2028回复")
    snippets = [first, duplicate, long, second, unicode_separator]
    raw = (" \r\n" + "\r\n\t\r".join(item.model_dump_json() for item in snippets)).encode()
    path.write_bytes(raw)
    subject = source(path)
    assert long.as_style_snippet() is None
    expected = adapter(
        [converted for item in snippets if (converted := item.as_style_snippet()) is not None]
    ).evidence_for("partner", "合成语境", [])
    assert subject.evidence_for("partner", "合成语境", []) == expected
    assert subject.content_revision("partner") == hashlib.sha256(raw).hexdigest()


def test_refresh_reads_configured_synthetic_file_once_per_call(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-library.jsonl"
    exclusions = tmp_path / "deleted_ids.json"
    raw = write_library(path, [paired(0)])
    read_bytes = Path.read_bytes
    reads = []

    def read_synthetic_only(actual):
        assert actual in (path, exclusions), "source must only read its synthetic snapshot files"
        reads.append(actual)
        return read_bytes(actual)

    # Guard the actual file-read boundary for construction and both refresh callers.
    monkeypatch.setattr(Path, "read_bytes", read_synthetic_only)
    subject = source(path)
    assert reads == [path, exclusions]
    assert subject.content_revision("partner") == hashlib.sha256(raw).hexdigest()
    assert reads == [path, exclusions] * 2
    subject.evidence_for("partner", "合成语境", [])
    assert reads == [path, exclusions] * 3
    raw = write_library(path, [paired(1)])
    assert subject.content_revision("partner") == hashlib.sha256(raw).hexdigest()
    assert reads == [path, exclusions] * 4


class StringSubclass(str):
    pass


@pytest.mark.parametrize("value", [None, 1, StringSubclass("partner"), "", " \t\n", "x" * 121])
def test_constructor_validates_partner_before_loading_with_adapter_errors(tmp_path, value):
    with pytest.raises((TypeError, ValueError)) as expected:
        SnippetStyleAdapter(
            snippets=[], partner_user_id=value, style_version="v1", clock=lambda: DAY
        )
    with pytest.raises(type(expected.value)) as actual:
        source(tmp_path / "missing-synthetic-library.jsonl", partner_user_id=value)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("raw", [None, b"{broken", b"\xff", b"{}"])
def test_constructor_rejects_unavailable_library(tmp_path, raw):
    path = tmp_path / "synthetic-library.jsonl"
    if raw is not None:
        path.write_bytes(raw)
    with pytest.raises(StyleValidationError) as caught:
        source(path)
    assert caught.value.code == "snippet_source_unavailable"
    assert caught.value.detail == "snippet source unavailable"
    assert caught.value.__context__ is None


def test_accepts_user_matches_adapter_even_when_source_is_unavailable(tmp_path):
    path = tmp_path / "synthetic-library.jsonl"
    path.write_bytes(b"")
    subject = source(path)
    control = adapter([])
    path.unlink()
    for value in ("partner", "visitor", "x" * 120, "", " ", "x" * 121, None, StringSubclass("a")):
        assert subject.accepts_user(value) is control.accepts_user(value)
