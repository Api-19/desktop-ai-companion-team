"""Validate frozen, offline v1.1 annotation packages (Unicode code point offsets)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL = "rcpem-public-pilot-v1.1"
FORMAT = "cped-annotation-package-v1"


def require(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{where}: {message}")


def obj(value, where):
    require(isinstance(value, dict), where, "expected object")
    return value


def array(value, where, *, nonempty=False):
    require(isinstance(value, list), where, "expected array")
    require(not nonempty or bool(value), where, "must not be empty")
    return value


def string(value, where):
    require(isinstance(value, str) and bool(value.strip()), where, "expected nonempty string")
    return value


def unique(values, where):
    require(len(values) == len(set(values)), where, "duplicate IDs or values")
    return values


def messages(values, where, known):
    for index, value in enumerate(array(values, where)):
        loc = f"{where}[{index}]"
        obj(value, loc)
        mid = string(value.get("message_id"), loc + ".message_id")
        string(value.get("speaker_id"), loc + ".speaker_id")
        string(value.get("text"), loc + ".text")
        require(mid not in known or known[mid] == value, loc, "conflicting message ID")
        known[mid] = value
    unique([m["message_id"] for m in values], where)


def spans(values, known, where):
    for span in array(values, where, nonempty=True):
        obj(span, where)
        mid = span.get("message_id")
        require(isinstance(mid, str) and mid in known, where, "unknown message_id")
        start, end = span.get("start"), span.get("end")
        text = known[mid]["text"]
        require(
            type(start) is int and type(end) is int and 0 <= start < end <= len(text),
            where,
            "invalid code point range",
        )
        require(span.get("text") == text[start:end], where, "text differs from message slice")


def units(values, known, where):
    ids = []
    for value in array(values, where, nonempty=True):
        obj(value, where)
        ids.append(string(value.get("id"), where + ".id"))
        spans(value.get("spans"), known, where + ".spans")
    unique(ids, where)


def validate_package(package):
    obj(package, "package")
    require(package.get("package_format") == FORMAT, "package", "wrong package_format")
    require(package.get("protocol_version") == PROTOCOL, "package", "wrong protocol_version")
    for key in ("material_version", "batch_id"):
        string(package.get(key), "package." + key)
    stage = package.get("stage")
    require(stage in ("s_only", "fg_stage1", "fg_stage2"), "package", "invalid stage")
    ids = []
    for event in array(package.get("events"), "events", nonempty=True):
        obj(event, "event")
        eid = string(event.get("event_id"), "event.event_id")
        ids.append(eid)
        known = {}
        if stage != "fg_stage2":
            require(event.get("sample_role") in ("base", "supplement"), eid, "invalid sample_role")
            string(event.get("material_ref"), eid + ".material_ref")
            for key in ("context", "partner"):
                messages(event.get(key), eid + "." + key, known)
        if stage == "fg_stage1":
            require(
                "reply" not in event and "segments" not in event,
                eid,
                "stage1 must not contain reply/segments",
            )
            units(event.get("f_units"), known, eid + ".f_units")
            candidates = array(event.get("g_candidates"), eid + ".g_candidates")
            candidate_ids = []
            for candidate in candidates:
                obj(candidate, eid + ".candidate")
                candidate_ids.append(string(candidate.get("candidate_id"), eid + ".candidate_id"))
                for key in ("c", "r"):
                    require(bool(candidate.get(key)), eid, "candidate c/r must not be empty")
                    messages(candidate.get(key), eid + ".candidate." + key, known)
            unique(candidate_ids, eid + ".g_candidates")
        else:
            messages(event.get("reply"), eid + ".reply", known)
            require(bool(event["reply"]), eid, "reply must not be empty")
            reply = {m["message_id"]: m for m in event["reply"]}
            require(
                type(event.get("segmentation_diagnosis")) is bool,
                eid,
                "segmentation_diagnosis must be boolean",
            )
            units(event.get("segments"), reply, eid + ".segments")
            # Flatten coverage: each code point must occur once, in source order, without gaps.
            actual = [
                (s["message_id"], i)
                for unit in event["segments"]
                for s in unit["spans"]
                for i in range(s["start"], s["end"])
            ]
            expected = [(m["message_id"], i) for m in event["reply"] for i in range(len(m["text"]))]
            require(actual == expected, eid, "segments must cover reply in order without overlap")
    unique(ids, "events")
    return package


def load_package(path):
    return validate_package(json.loads(Path(path).read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    args = parser.parse_args()
    try:
        package = load_package(args.package)
    except (OSError, ValueError) as error:
        print(" ".join(str(error).splitlines()))
        return 1
    print(f"package-ok {len(package['events'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
