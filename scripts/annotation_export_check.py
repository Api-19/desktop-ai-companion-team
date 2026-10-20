"""Check an offline annotation export against its frozen package(s)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .annotation_package_check import (
        PROTOCOL,
        array,
        load_package,
        obj,
        require,
        spans,
        string,
        unique,
        validate_package,
    )
else:
    from annotation_package_check import (
        PROTOCOL,
        array,
        load_package,
        obj,
        require,
        spans,
        string,
        unique,
        validate_package,
    )

ACTIONS = {
    "ACK",
    "INFORM",
    "ASK",
    "EVALUATE",
    "VALIDATE",
    "SUPPORT",
    "SUGGEST",
    "REQUEST",
    "COMMIT",
    "REFUSE",
    "SOCIAL",
    "OTHER",
}
FEATURES = {"humor", "affection", "seek_care", "self_disclosure"}
REASONS = {
    "context_missing",
    "ambiguous_intent",
    "joke_or_serious",
    "multi_function",
    "boundary",
    "other",
}
SUBTYPES = {
    "COMMIT": {"promise", "offer"},
    "SOCIAL": {"greet", "thank", "apologize", "close", "topic_shift", "other"},
}


def choices(value, allowed, where, *, nonempty=False):
    array(value, where, nonempty=nonempty)
    require(all(isinstance(v, str) and v in allowed for v in value), where, "invalid selection")
    unique(value, where)


def declaration(value, where):
    obj(value, where)
    for key in ("reply_seen", "review_comments_read"):
        require(type(value.get(key)) is bool, where + "." + key, "must explicitly select yes/no")


def mode(value, where):
    require(value.get("mode") in ("units", "holistic"), where, "select mode")
    require(isinstance(value.get("reason"), str), where, "reason must be a string")


def validate_f1(value, event, exposed):
    loc = event["event_id"] + ".f_step1"
    obj(value, loc)
    require(value.get("locked") is True, loc, "F1 must be locked")
    require(value.get("reply_seen_before_step1") is exposed, loc, "exposure flag differs")
    mode(value, loc)
    units = {u["id"] for u in event["f_units"]}
    choices(value.get("acceptable"), units, loc + ".acceptable")
    status, preferred = value.get("status"), value.get("preferred")
    require(status in ("judged", "insufficient_context"), loc, "invalid status")
    if status == "insufficient_context":
        require(
            value["acceptable"] == [] and preferred == "insufficient_context",
            loc,
            "insufficient_context requires empty acceptable and matching preferred",
        )
    else:
        require(
            bool(value["acceptable"])
            and (preferred == "no_unique" or preferred in value["acceptable"]),
            loc,
            "select acceptable and preferred (selected unit or no_unique)",
        )


def validate_f2(value, event):
    loc = event["event_id"] + ".f_step2"
    obj(value, loc)
    mode(value, loc)
    status = value.get("status")
    require(status in ("judged", "no_match", "insufficient_context"), loc, "select status")
    groups = array(value.get("responded_groups"), loc + ".responded_groups")
    seen = []
    for group in groups:
        choices(group, {u["id"] for u in event["f_units"]}, loc, nonempty=True)
        seen.extend(group)
    unique(seen, loc)
    require(
        bool(groups) if status == "judged" else not groups,
        loc,
        "judged needs groups; no_match/insufficient_context needs empty groups",
    )


def validate_g(record, event, exposed):
    loc = event["event_id"] + ".g"
    require(record.get("g_reply_seen_before_judgment") is exposed, loc, "exposure flag differs")
    candidates = event["g_candidates"]
    pairs = array(record.get("g_pairs"), loc)
    if not candidates:
        require(
            record.get("g_status") == "not_applicable" and not pairs,
            loc,
            "no candidates requires not_applicable and empty pairs",
        )
        string(record.get("g_reason"), loc + ".g_reason")
        return
    require(record.get("g_status") == "annotated", loc, "g_status must be annotated")
    for pair in pairs:
        obj(pair, loc)
    require(
        [p.get("candidate_id") for p in pairs] == [c["candidate_id"] for c in candidates],
        loc,
        "candidate IDs differ",
    )
    for pair, candidate in zip(pairs, candidates, strict=True):
        available = set()
        for key, messages in [
            ("c", candidate["c"]),
            ("r", candidate["r"]),
            ("x", event["context"] + event["partner"]),
        ]:
            expected = [m["message_id"] for m in messages]
            require(pair.get(key + "_message_ids") == expected, loc, key + " source IDs differ")
            available.update(expected)
        require(
            pair.get("label") in ("applicable", "not_applicable", "uncertain"),
            loc,
            "select candidate label",
        )
        string(pair.get("reason"), loc + ".reason")
        choices(pair.get("evidence"), available, loc + ".evidence", nonempty=True)


def validate_s(value, target):
    loc = target["event_id"] + ".s"
    obj(value, loc)
    segments = array(value.get("segments"), loc + ".segments")
    for segment in segments:
        obj(segment, loc)
    require(
        [s.get("id") for s in segments] == [s["id"] for s in target["segments"]],
        loc,
        "segment IDs/order differ",
    )
    known = {m["message_id"]: m for m in target["reply"]}
    for segment, frozen in zip(segments, target["segments"], strict=True):
        where = loc + "." + frozen["id"]
        require(segment.get("spans") == frozen["spans"], where, "frozen spans differ")
        status = segment.get("status")
        require(status in ("judged", "uncertain", "insufficient_context"), where, "select status")
        choices(segment.get("actions"), ACTIONS, where + ".actions", nonempty=status == "judged")
        choices(
            segment.get("candidates"),
            ACTIONS,
            where + ".candidates",
            nonempty=status == "uncertain",
        )
        if status != "judged":
            require(not segment["actions"], where, "unjudged segment actions must be empty")
        if status != "uncertain":
            require(not segment["candidates"], where, "candidates only allowed for uncertain")
        exceptional = (
            status != "judged" or len(segment["actions"]) > 1 or "OTHER" in segment["actions"]
        )
        if exceptional:
            require(
                isinstance(segment.get("reason_code"), str) and segment["reason_code"] in REASONS,
                where,
                "select reason_code",
            )
        if "OTHER" in segment["actions"] or "OTHER" in segment["candidates"]:
            string(segment.get("reason"), where + ".reason")
        require("subtype" in segment, where, "missing subtype (may be null)")
        subtype = segment.get("subtype")
        if subtype is not None:
            if isinstance(subtype, dict):
                require(bool(subtype), where, "empty subtype object")
                for action, sub in subtype.items():
                    require(
                        action in segment["actions"]
                        and action in SUBTYPES
                        and isinstance(sub, str)
                        and sub in SUBTYPES[action],
                        where,
                        "invalid subtype",
                    )
            else:
                require(
                    any(
                        a in segment["actions"] and isinstance(subtype, str) and subtype in opts
                        for a, opts in SUBTYPES.items()
                    ),
                    where,
                    "invalid subtype",
                )
        modifiers = obj(segment.get("modifiers"), where + ".modifiers")
        require(set(modifiers) == FEATURES, where, "all four modifier keys required")
        for key, modifier in modifiers.items():
            obj(modifier, where + "." + key)
            val = modifier.get("value")
            require(val in ("present", "absent", "unclear"), where + "." + key, "select value")
            if val == "present":
                spans(modifier.get("evidence"), known, where + "." + key + ".evidence")
                for evidence in modifier["evidence"]:
                    require(
                        any(
                            evidence["message_id"] == frozen_span["message_id"]
                            and frozen_span["start"]
                            <= evidence["start"]
                            < evidence["end"]
                            <= frozen_span["end"]
                            for frozen_span in frozen["spans"]
                        ),
                        where,
                        "evidence must lie within this segment",
                    )
            elif val == "unclear":
                string(modifier.get("reason"), where + "." + key + ".reason")
            else:
                require(set(modifier) == {"value"}, where, "absent only stores value")
    for objection in array(value.get("segment_objections"), loc + ".segment_objections"):
        obj(objection, loc)
        require(
            isinstance(objection.get("segment_id"), str)
            and objection["segment_id"] in {s["id"] for s in target["segments"]},
            loc,
            "unknown objection segment",
        )
        require(
            objection.get("problem") in ("should_split", "should_merge", "wrong_boundary"),
            loc,
            "invalid boundary problem",
        )
        require(objection.get("reason_code") == "boundary", loc, "boundary reason_code required")
        string(objection.get("proposal"), loc + ".proposal")
    if target["segmentation_diagnosis"]:
        boundaries = array(value.get("free_segmentation"), loc + ".free_segmentation")
        keys = []
        for boundary in boundaries:
            obj(boundary, loc)
            mid, position = boundary.get("message_id"), boundary.get("position")
            require(
                isinstance(mid, str)
                and mid in known
                and type(position) is int
                and 0 < position < len(known[mid]["text"]),
                loc,
                "invalid internal boundary",
            )
            keys.append((mid, position))
        unique(keys, loc)


def validate_export(data, packages):
    for package in packages:
        validate_package(package)
    require(len(packages) in (1, 2), "packages", "use S-only or a stage1/stage2 pair")
    initial = next((p for p in packages if p["stage"] != "fg_stage2"), None)
    require(initial is not None, "packages", "missing initial package")
    is_fg = initial["stage"] == "fg_stage1"
    if is_fg:
        require(
            len(packages) == 2 and {p["stage"] for p in packages} == {"fg_stage1", "fg_stage2"},
            "packages",
            "F/G requires both stages",
        )
        target = next(p for p in packages if p["stage"] == "fg_stage2")
    else:
        require(len(packages) == 1, "packages", "S-only uses one package")
        target = initial
    for key in ("protocol_version", "batch_id", "material_version"):
        require(all(p[key] == initial[key] for p in packages), "packages", key + " differs")
    events = {e["event_id"]: e for e in initial["events"]}
    targets = {e["event_id"]: e for e in target["events"]}
    require(set(events) == set(targets), "packages", "stage event_id sets differ")
    obj(data, "export")
    require(data.get("protocol_version") == PROTOCOL, "export", "wrong protocol_version")
    for key in ("batch_id", "material_version"):
        require(data.get(key) == initial[key], "export", key + " differs")
    string(data.get("annotator_id"), "annotator_id")
    require(data.get("phase") in ("training", "pilot", "adjudication"), "export", "invalid phase")
    declaration(data.get("batch_declaration"), "batch_declaration")
    exceptions = {}
    for exception in array(data.get("event_exposure_exceptions"), "event_exposure_exceptions"):
        obj(exception, "exposure")
        eid = exception.get("event_id")
        require(
            isinstance(eid, str) and eid in events and eid not in exceptions,
            "exposure",
            "unknown or duplicate event ID",
        )
        declaration(exception, "exposure")
        string(exception.get("note"), "exposure.note")
        exceptions[eid] = exception
    records = array(data.get("records"), "records")
    ids = [obj(r, "record").get("event_id") for r in records]
    require(all(isinstance(eid, str) for eid in ids), "records", "invalid event ID")
    unique(ids, "records")
    require(set(ids) == set(events), "records", "event_id set differs")
    for record in records:
        eid = record["event_id"]
        event = events[eid]
        for key in ("material_ref", "sample_role"):
            require(record.get(key) == event[key], eid, key + " differs")
        require("training_material" not in record, eid, "omit training_material")
        exposed = exceptions.get(eid, data["batch_declaration"])["reply_seen"]
        if is_fg:
            validate_f1(record.get("f_step1"), event, exposed)
            validate_f2(record.get("f_step2"), event)
            validate_g(record, event, exposed)
        else:
            require(
                all(
                    key in record and record[key] is None
                    for key in ("f_step1", "f_step2", "g_status")
                ),
                eid,
                "S-only F/G must be null",
            )
            require(
                record.get("g_pairs") == [] and record.get("g_reply_seen_before_judgment") is False,
                eid,
                "S-only has no G judgments",
            )
        validate_s(record.get("s"), targets[eid])
        for objection in array(record.get("unit_objections"), eid + ".unit_objections"):
            obj(objection, eid)
            require(
                is_fg
                and isinstance(objection.get("unit_id"), str)
                and objection["unit_id"] in {u["id"] for u in event["f_units"]},
                eid,
                "unknown F unit objection",
            )
            string(objection.get("problem"), eid + ".problem")
            string(objection.get("proposal"), eid + ".proposal")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export")
    parser.add_argument("--package", action="append", required=True)
    args = parser.parse_args()
    try:
        data = json.loads(Path(args.export).read_text(encoding="utf-8"))
        validate_export(data, [load_package(p) for p in args.package])
    except (OSError, ValueError) as error:
        print(" ".join(str(error).splitlines()))
        return 1
    print(f"export-ok {len(data['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
