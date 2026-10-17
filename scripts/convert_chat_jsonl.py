"""Bounded local conversion; callers explicitly supply a trusted private root.

Excluded values are scanned for JSON structure without decoding their contents.
This helper neither imports records into product stores nor assigns final roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from companion.importer.standard_json import StandardJsonAdapter
from companion.runtime.settings import Settings

ZONE = timezone(timedelta(hours=8))
CUTOFF = int(datetime(2026, 7, 18, tzinfo=ZONE).timestamp())
NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
STRING = re.compile(rb'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"')
NONTEXT_TYPES = {
    3: "image",
    34: "voice",
    43: "video",
    47: "emoji",
    10000: "system",
    10002: "system",
}
GARBLED = re.compile("[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


class ConversionError(Exception):
    """Fixed public failure without source values."""


@dataclass(frozen=True)
class Result:
    output: Path
    included: int
    nontext: int
    later: int
    source_atime_changed: bool
    nontext_kept: int = 0
    nontext_skipped: int = 0
    garbled_text: int = 0


def _decode(token: bytes):
    return json.loads(token)


class _Tokens:
    def __init__(self, line: bytes):
        self.line = line
        self.pos = 0

    def whitespace(self):
        while self.pos < len(self.line) and self.line[self.pos : self.pos + 1] in b" \t\r\n":
            self.pos += 1

    def char(self, expected):
        self.whitespace()
        if self.line[self.pos : self.pos + 1] != expected:
            raise ValueError
        self.pos += 1

    def string(self):
        self.whitespace()
        match = STRING.match(self.line, self.pos)
        if match is None:
            raise ValueError
        start = self.pos
        self.pos = match.end()
        return self.line[start : self.pos]

    def value(self, depth=0):
        if depth > 100:
            raise ValueError
        self.whitespace()
        current = self.line[self.pos : self.pos + 1]
        if current == b'"':
            self.string()
        elif current in (b"{", b"["):
            self.pos += 1
            close = b"}" if current == b"{" else b"]"
            self.whitespace()
            if self.line[self.pos : self.pos + 1] != close:
                while True:
                    if current == b"{":
                        self.string()
                        self.char(b":")
                    self.value(depth + 1)
                    self.whitespace()
                    if self.line[self.pos : self.pos + 1] == close:
                        break
                    self.char(b",")
            self.char(close)
        else:
            match = NUMBER.match(self.line, self.pos)
            if match:
                self.pos = match.end()
                return
            for literal in (b"true", b"false", b"null"):
                if self.line.startswith(literal, self.pos):
                    self.pos += len(literal)
                    return
            raise ValueError

    def fields(self):
        self.char(b"{")
        fields = {}
        self.whitespace()
        if self.line[self.pos : self.pos + 1] != b"}":
            while True:
                key = _decode(self.string())
                if key in fields:
                    raise ValueError
                self.char(b":")
                self.whitespace()
                start = self.pos
                self.value()
                fields[key] = (start, self.pos)
                self.whitespace()
                if self.line[self.pos : self.pos + 1] == b"}":
                    break
                self.char(b",")
        self.char(b"}")
        self.whitespace()
        if self.pos != len(self.line):
            raise ValueError
        return fields


def _read_records(
    handle, conversation_id, *, through_date=date(2026, 7, 17), keep_nontext_markers=False
):
    records, ids, speakers = [], set(), {}
    cutoff = int(
        datetime.combine(through_date + timedelta(days=1), datetime.min.time(), ZONE).timestamp()
    )
    nontext = later = 0
    nontext_kept = nontext_skipped = garbled_text = 0
    for raw_line in handle:
        line = raw_line
        fields = _Tokens(line).fields()

        def get(key, fields=fields, line=line):
            start, end = fields[key]
            return _decode(line[start:end])

        try:
            timestamp = get("timestamp")
            if type(timestamp) is not int:
                raise ValueError
            time = datetime.fromtimestamp(timestamp, ZONE)
        except (KeyError, ValueError, OverflowError, OSError):
            if keep_nontext_markers:
                kind = get("type")
                if type(kind) is int and kind != 1:
                    nontext += 1
                    nontext_skipped += 1
                    continue
            raise
        if timestamp >= cutoff:
            later += 1
            continue
        kind = get("type")
        if type(kind) is not int:
            raise ValueError
        text = get("text") if kind == 1 else None
        garbled = isinstance(text, str) and GARBLED.search(text) is not None
        marker = kind != 1 or garbled
        if garbled:
            garbled_text += 1
            text = None
        if marker:
            nontext += 1
            if not keep_nontext_markers:
                continue
        try:
            message_id = get("server_id")
            if type(message_id) is not int or message_id <= 0 or message_id in ids:
                raise ValueError
            sender = get("is_sender")
            if type(sender) is not bool:
                raise ValueError
            speaker = get("sender_id")
            if not isinstance(speaker, str) or not speaker:
                raise ValueError
            conversation = (
                get("conversation_id") if "conversation_id" in fields else conversation_id
            )
            name = get("sender")
            if not isinstance(conversation, str) or not conversation:
                raise ValueError
            if not isinstance(name, str) or (not marker and not isinstance(text, str)):
                raise ValueError
        except (KeyError, ValueError):
            if marker:
                nontext_skipped += 1
                continue
            raise
        ids.add(message_id)
        direction = "outbound" if sender else "inbound"
        if conversation != conversation_id:
            raise ValueError
        if direction in speakers and speakers[direction] != speaker:
            raise ValueError
        speakers[direction] = speaker
        if len(set(speakers.values())) != len(speakers):
            raise ValueError
        records.append(
            dict(
                schema_version=1,
                conversation_id=conversation,
                message_id=str(message_id),
                timestamp=time.isoformat(),
                source_speaker_id=speaker,
                source_speaker_name=name or None,
                direction=direction,
                content_type=NONTEXT_TYPES.get(kind, "other") if marker else "text",
                text=text,
            )
        )
        if marker:
            nontext_kept += 1
    if not records:
        raise ValueError
    return records, nontext, later, nontext_kept, nontext_skipped, garbled_text


def _fingerprint(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns


def _private_directory(fd):
    info = os.fstat(fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError


def _root(path):
    path = path.absolute()
    if ".." in path.parts:
        raise ValueError
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        _private_directory(fd)
        return fd, path
    except BaseException:
        os.close(fd)
        raise


def _convert(
    source, settings, conversation_id, *, through_date=date(2026, 7, 17), keep_nontext_markers=False
):
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError
    opened = []
    created_dirs = []
    created_files = []
    success = False
    try:
        root_fd, root_path = _root(settings.private_data_dir)
        opened.append(root_fd)
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        held_source = os.dup(source_fd)
        opened.append(held_source)
        with os.fdopen(source_fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError
            digest = hashlib.file_digest(handle, "sha256").digest()
            handle.seek(0)
            records, nontext, later, nontext_kept, nontext_skipped, garbled_text = _read_records(
                handle,
                conversation_id,
                through_date=through_date,
                keep_nontext_markers=keep_nontext_markers,
            )
            handle.seek(0)
            if hashlib.file_digest(handle, "sha256").digest() != digest:
                raise ValueError
            after = os.fstat(handle.fileno())
            if _fingerprint(before) != _fingerprint(after) or _fingerprint(
                source.stat(follow_symlinks=False)
            ) != _fingerprint(before):
                raise ValueError

        parent = root_fd
        output_dir = root_path
        directory_chain = [(root_path, root_fd)]
        for index, name in enumerate(("imports", "standard-v1", uuid.uuid4().hex)):
            try:
                os.mkdir(name, 0o700, dir_fd=parent)
                created = True
            except FileExistsError:
                if index == 2:  # fresh run must never reuse an existing directory
                    raise
                created = False
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            opened.append(child)
            if created:
                created_dirs.append((parent, name, os.fstat(child)))
            _private_directory(child)
            parent = child
            output_dir = output_dir / name
            directory_chain.append((output_dir, child))
        pending = ".pending.json"
        file_fd = os.open(
            pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        info = os.fstat(file_fd)
        created_files.append((parent, pending, info))
        if stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1:
            os.close(file_fd)
            raise ValueError
        with os.fdopen(file_fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(records, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        def check_directories():
            for visible, expected_fd in directory_chain:
                visible_fd, _ = _root(visible)
                try:
                    current_dir = os.fstat(visible_fd)
                    expected_dir = os.fstat(expected_fd)
                    if (current_dir.st_dev, current_dir.st_ino) != (
                        expected_dir.st_dev,
                        expected_dir.st_ino,
                    ):
                        raise ValueError
                    _private_directory(expected_fd)
                finally:
                    os.close(visible_fd)

        check_directories()
        StandardJsonAdapter().parse(output_dir / pending)
        with os.fdopen(os.dup(held_source), "rb") as check:
            check.seek(0)
            if (
                _fingerprint(os.fstat(check.fileno())) != _fingerprint(before)
                or os.fstat(check.fileno()).st_nlink != 1
                or hashlib.file_digest(check, "sha256").digest() != digest
                or _fingerprint(source.stat(follow_symlinks=False)) != _fingerprint(before)
            ):
                raise ValueError
            after = os.fstat(check.fileno())
        check_directories()
        current = os.stat(pending, dir_fd=parent, follow_symlinks=False)
        if (
            (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
            or current.st_nlink != 1
            or stat.S_IMODE(current.st_mode) != 0o600
        ):
            raise ValueError
        final = f"chat-through-{through_date.isoformat()}.json"
        # link is an atomic exclusive publication; remove our temporary link immediately.
        os.link(pending, final, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        created_files.append((parent, final, info))
        os.unlink(pending, dir_fd=parent)
        os.fsync(parent)
        if os.stat(final, dir_fd=parent, follow_symlinks=False).st_nlink != 1:
            raise ValueError
        check_directories()
        success = True
        return Result(
            output_dir / final,
            len(records),
            nontext,
            later,
            after.st_atime_ns != before.st_atime_ns,
            nontext_kept,
            nontext_skipped,
            garbled_text,
        )
    finally:
        if not success:
            for parent, name, info in reversed(created_files):
                try:
                    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                        os.unlink(name, dir_fd=parent)
                except OSError:
                    pass
            for parent, name, info in reversed(created_dirs):
                try:
                    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                        os.rmdir(name, dir_fd=parent)
                except OSError:
                    pass
        for fd in reversed(opened):
            os.close(fd)


def convert(
    source: Path,
    settings: Settings,
    conversation_id: str,
    *,
    through_date: date = date(2026, 7, 17),
    keep_nontext_markers: bool = False,
) -> Result:
    try:
        return _convert(
            source,
            settings,
            conversation_id,
            through_date=through_date,
            keep_nontext_markers=keep_nontext_markers,
        )
    except Exception:
        pass
    raise ConversionError("conversion_failed")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ConversionError("conversion_failed")


def _parse_through_date(value):
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError
    return date.fromisoformat(value)


def main():
    try:
        parser = _Parser()
        parser.add_argument("--source", required=True, type=Path)
        parser.add_argument("--private-root", required=True, type=Path)
        parser.add_argument("--conversation-id", required=True)
        parser.add_argument("--through-date", type=_parse_through_date, default=date(2026, 7, 17))
        parser.add_argument("--keep-nontext-markers", action="store_true")
        args = parser.parse_args()
        settings = Settings(
            environment="production",
            log_level="ERROR",
            private_data_dir=args.private_root,
            runtime_data_dir=args.private_root,
            test_results_dir=args.private_root,
        )
        result = convert(
            args.source,
            settings,
            args.conversation_id,
            through_date=args.through_date,
            keep_nontext_markers=args.keep_nontext_markers,
        )
        stats = dict(
            included=result.included,
            nontext=result.nontext,
            later=result.later,
            garbled_text=result.garbled_text,
            source_unchanged=True,
            source_atime_changed=result.source_atime_changed,
        )
        if args.keep_nontext_markers:
            stats.update(nontext_kept=result.nontext_kept, nontext_skipped=result.nontext_skipped)
        print(json.dumps(stats))
        return 0
    except Exception:
        print("conversion_failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
