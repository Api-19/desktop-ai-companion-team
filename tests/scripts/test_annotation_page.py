"""Offline pilot tooling uses invented fixtures only; no browser or provider required."""

import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "fixtures/synthetic/annotation_packages"
PAGE = ROOT / "tools/annotation_page/annotate.html"
ACTIONS = [
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
]
FEATURES = ["humor", "affection", "seek_care", "self_disclosure"]


def read(name):
    return json.loads((PACKAGES / f"{name}.json").read_text())


def valid_export(stage="s_only"):
    p = read(stage)
    p2 = read("fg_stage2") if stage == "fg_stage1" else p
    records = []
    for event, target in zip(p["events"], p2["events"], strict=True):
        record = {
            "event_id": event["event_id"],
            "material_ref": event["material_ref"],
            "sample_role": event["sample_role"],
            "f_step1": None,
            "f_step2": None,
            "g_status": None,
            "g_reply_seen_before_judgment": False,
            "g_pairs": [],
            "s": {"segments": [], "segment_objections": []},
            "unit_objections": [],
        }
        if stage == "fg_stage1":
            units = [u["id"] for u in event["f_units"]]
            record["f_step1"] = dict(
                locked=True,
                reply_seen_before_step1=False,
                status="judged",
                acceptable=units,
                preferred=units[0],
                mode="units",
                reason="",
            )
            record["f_step2"] = dict(
                status="judged", mode="units", responded_groups=[[units[0]]], reason=""
            )
            record["g_status"] = "annotated" if event["g_candidates"] else "not_applicable"
            if not event["g_candidates"]:
                record["g_reason"] = "没有候选"
            for candidate in event["g_candidates"]:
                record["g_pairs"].append(
                    dict(
                        candidate_id=candidate["candidate_id"],
                        c_message_ids=[m["message_id"] for m in candidate["c"]],
                        r_message_ids=[m["message_id"] for m in candidate["r"]],
                        x_message_ids=[
                            m["message_id"] for m in event["context"] + event["partner"]
                        ],
                        label="applicable",
                        reason="前提成立",
                        evidence=[candidate["c"][0]["message_id"]],
                    )
                )
        for segment in target["segments"]:
            record["s"]["segments"].append(
                dict(
                    **copy.deepcopy(segment),
                    status="judged",
                    actions=["INFORM"],
                    candidates=[],
                    subtype=None,
                    modifiers={k: {"value": "absent"} for k in FEATURES},
                )
            )
        if target["segmentation_diagnosis"]:
            record["s"]["free_segmentation"] = []
        records.append(record)
    return dict(
        protocol_version=p["protocol_version"],
        material_version=p["material_version"],
        annotator_id="tester-a",
        phase="training",
        batch_id=p["batch_id"],
        batch_declaration=dict(reply_seen=False, review_comments_read=False),
        event_exposure_exceptions=[],
        records=records,
    )


def check_export(data, stage="s_only"):
    from scripts.annotation_export_check import validate_export

    packages = [read(stage)]
    if stage == "fg_stage1":
        packages.append(read("fg_stage2"))
    validate_export(data, packages)


@pytest.mark.parametrize(
    "name", ["s_only", "fg_stage1", "fg_stage2", "example_fg_stage1", "example_fg_stage2"]
)
def test_package_cli(name):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/annotation_package_check.py"),
            str(PACKAGES / f"{name}.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"package-ok {len(read(name)['events'])}"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("material_version"),
        lambda p: p.update(stage="unknown"),
        lambda p: p["events"][0].update(sample_role="unknown"),
        lambda p: p["events"][0]["reply"][0].pop("speaker_id"),
        lambda p: p["events"][0].update(segmentation_diagnosis=0),
        lambda p: p["events"].append(copy.deepcopy(p["events"][0])),
        lambda p: p["events"][0]["segments"][0]["spans"][0].update(text="错误"),
        lambda p: p["events"][0]["segments"][0]["spans"][0].update(start=True),
        lambda p: p["events"][0]["segments"].reverse(),
        lambda p: p["events"][0]["segments"].pop(),
    ],
)
def test_bad_package(mutation):
    from scripts.annotation_package_check import validate_package

    p = read("s_only")
    mutation(p)
    with pytest.raises(ValueError):
        validate_package(p)


@pytest.mark.parametrize("key", ["reply", "segments"])
def test_stage1_never_contains_reply(key):
    from scripts.annotation_package_check import validate_package

    p = read("fg_stage1")
    p["events"][0][key] = []
    with pytest.raises(ValueError):
        validate_package(p)


@pytest.mark.parametrize("stage", ["s_only", "fg_stage1"])
def test_valid_exports(stage):
    check_export(valid_export(stage), stage)


def test_authoritative_example():
    from scripts.annotation_export_check import validate_export

    data = json.loads((ROOT / "docs/data/annotation_record.example.json").read_text())
    for record in data["records"]:
        record.pop("training_material", None)
    validate_export(data, [read("example_fg_stage1"), read("example_fg_stage2")])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d, r, s: d.pop("annotator_id"),
        lambda d, r, s: d.update(protocol_version="old"),
        lambda d, r, s: d.update(material_version="wrong"),
        lambda d, r, s: d.update(batch_declaration={"reply_seen": False}),
        lambda d, r, s: d.update(records=[]),
        lambda d, r, s: d["records"].append(copy.deepcopy(r)),
        lambda d, r, s: r.update(material_ref="wrong"),
        lambda d, r, s: s.update(id="wrong"),
        lambda d, r, s: s["spans"][0].update(text="wrong"),
        lambda d, r, s: s.update(status=""),
        lambda d, r, s: s.update(actions=[]),
        lambda d, r, s: s.update(actions=["BAD"]),
        lambda d, r, s: s.update(actions=["ACK", "INFORM"]),
        lambda d, r, s: s.update(status="uncertain", actions=[], candidates=[]),
        lambda d, r, s: s.update(status="uncertain", actions=[], candidates=["ACK"]),
        lambda d, r, s: s.update(status="insufficient_context", actions=[]),
        lambda d, r, s: s.update(actions=["OTHER"], reason_code="other"),
        lambda d, r, s: s.update(subtype="promise"),
        lambda d, r, s: s["modifiers"].pop("humor"),
        lambda d, r, s: s["modifiers"]["humor"].update(value=""),
        lambda d, r, s: s["modifiers"]["humor"].update(value="present"),
        lambda d, r, s: s["modifiers"]["humor"].update(value="unclear"),
        lambda d, r, s: s["modifiers"]["humor"].update(
            value="present", evidence=[dict(message_id="missing", start=0, end=1, text="假")]
        ),
        lambda d, r, s: r["s"].pop("free_segmentation"),
        lambda d, r, s: r["s"].update(free_segmentation=[dict(message_id="missing", position=1)]),
        lambda d, r, s: r["s"].update(
            segment_objections=[
                dict(
                    segment_id=s["id"], problem="should_split", proposal="", reason_code="boundary"
                )
            ]
        ),
        lambda d, r, s: r.update(f_step1={}),
        lambda d, r, s: d.update(
            event_exposure_exceptions=[
                dict(event_id="missing", reply_seen=True, review_comments_read=False, note="例外")
            ]
        ),
    ],
)
def test_required_s_rules(mutation):
    d = valid_export()
    mutation(d, d["records"][0], d["records"][0]["s"]["segments"][0])
    with pytest.raises(ValueError):
        check_export(d)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["f_step1"].update(locked=False),
        lambda r: r["f_step1"].update(acceptable=[]),
        lambda r: r["f_step1"].update(preferred="missing"),
        lambda r: r["f_step1"].update(mode=""),
        lambda r: r["f_step1"].update(reply_seen_before_step1=True),
        lambda r: r["f_step1"].update(status="insufficient_context"),
        lambda r: r["f_step2"].update(status=""),
        lambda r: r["f_step2"].update(mode=""),
        lambda r: r["f_step2"].update(responded_groups=[]),
        lambda r: r["f_step2"].update(responded_groups=[["missing"]]),
        lambda r: r["f_step2"].update(responded_groups=[["u1"], ["u1"]]),
        lambda r: r["f_step2"].update(status="no_match"),
        lambda r: r.update(g_status=None),
        lambda r: r.update(g_pairs=[]),
        lambda r: r.update(g_reply_seen_before_judgment=True),
        lambda r: r["g_pairs"][0].update(label=""),
        lambda r: r["g_pairs"][0].update(reason=""),
        lambda r: r["g_pairs"][0].update(evidence=[]),
        lambda r: r["g_pairs"][0].update(evidence=["missing"]),
        lambda r: r["g_pairs"][0].update(x_message_ids=[]),
        lambda r: r["g_pairs"][0].update(c_message_ids=[]),
    ],
)
def test_required_fg_rules(mutation):
    d = valid_export("fg_stage1")
    mutation(d["records"][0])
    with pytest.raises(ValueError):
        check_export(d, "fg_stage1")


def test_conditional_valid_states_and_unicode():
    d = valid_export()
    s = d["records"][0]["s"]["segments"][0]
    s.update(status="uncertain", actions=[], candidates=["ASK"], reason_code="ambiguous_intent")
    check_export(d)
    s.update(status="insufficient_context", candidates=[], reason_code="context_missing")
    check_export(d)
    s.update(status="judged", actions=["OTHER"], reason_code="other", reason="编造功能描述")
    span = copy.deepcopy(s["spans"][0])
    s["modifiers"]["humor"] = dict(value="present", evidence=[span])
    s["modifiers"]["affection"] = dict(value="unclear", reason="语气不清")
    check_export(d)


def test_cli_errors_are_one_line(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    for script, extra in [
        ("annotation_package_check.py", []),
        ("annotation_export_check.py", ["--package", str(PACKAGES / "s_only.json")]),
    ]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), str(path), *extra],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 1
        assert len(result.stdout.splitlines()) == 1
        assert "Traceback" not in result.stderr


def test_page_is_self_contained():
    html = PAGE.read_text()
    for forbidden in ["http://", "https://", "fetch(", "XMLHttpRequest", "WebSocket"]:
        assert forbidden not in html
    for key in ACTIONS + FEATURES:
        assert key in html
    assert '<script id="annotation-core">' in html
    assert "FileReader" in html
    assert "localStorage" in html


def test_node_build_and_completion(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable")
    core = re.search(r'<script id="annotation-core">(.*?)</script>', PAGE.read_text(), re.S)[1]
    for stage in ["s_only", "fg_stage1"]:
        p = read(stage)
        p2 = read("fg_stage2") if stage == "fg_stage1" else None
        data = valid_export(stage)
        code = (
            core
            + "\n"
            + """
const [pkg, stage2, data] = JSON.parse(process.argv[1]);
let state = AnnotationCore.createState(pkg, data.annotator_id, data.phase);
if (AnnotationCore.completion(state).length === 0) throw Error('blank draft accepted');
state.stage2 = stage2;
state.batch_declaration = data.batch_declaration;
state.records = Object.fromEntries(data.records.map(r => [r.event_id, {...r, _gLocked:true}]));
const errors = AnnotationCore.completion(state);
if (errors.length) throw Error(errors.join('\\n'));
const output = AnnotationCore.buildExport(state);
if (output.legacy_projection || output.records.some(r => r._gLocked))
  throw Error('internal fields');
if (AnnotationCore.codePointOffset('甲😀乙', 3) !== 2) throw Error('UTF16 offset');
state.records[data.records[0].event_id].s.segments[0].actions = [];
if (!AnnotationCore.completion(state).length) throw Error('missing action accepted');
process.stdout.write(JSON.stringify(output));
"""
        )
        result = run_node(node, code, [p, p2, data])
        assert result.returncode == 0, result.stderr
        output = tmp_path / f"{stage}.json"
        output.write_text(result.stdout)
        command = [
            sys.executable,
            str(ROOT / "scripts/annotation_export_check.py"),
            str(output),
            "--package",
            str(PACKAGES / f"{stage}.json"),
        ]
        if p2:
            command += ["--package", str(PACKAGES / "fg_stage2.json")]
        checked = subprocess.run(command, text=True, capture_output=True)
        assert checked.returncode == 0, checked.stdout + checked.stderr


def run_node(node, code, payload):
    # Script and payload go through stdin: Linux caps a single argv string at 128 KiB.
    argv = json.dumps(json.dumps(payload, ensure_ascii=False))
    return subprocess.run(
        [node, "-"],
        input=f"process.argv[1] = {argv};\n{code}",
        text=True,
        capture_output=True,
    )


def run_core(script, payload):
    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable")
    core = re.search(r'<script id="annotation-core">(.*?)</script>', PAGE.read_text(), re.S)[1]
    result = run_node(node, core + "\n" + script, payload)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_node_locks_restore_and_stage2_gate():
    run_core(
        """
const C=AnnotationCore, [pkg,stage2,data,sPkg,sData]=JSON.parse(process.argv[1]);
function blocked(fn) {
  let ok=false; try {fn();} catch(e) {ok=true;}
  if(!ok) throw Error('gate did not block');
}
let state=C.createState(pkg,data.annotator_id,data.phase);
blocked(()=>C.attachStage2(state,stage2));
state.batch_declaration=data.batch_declaration;
for(const e of pkg.events) {
  const r=state.records[e.event_id], source=data.records.find(r=>r.event_id===e.event_id);
  r.f_step1=C.clone(source.f_step1);
  r.g_pairs=C.clone(source.g_pairs);
}
blocked(()=>C.attachStage2(state,stage2));
for(const r of Object.values(state.records)) r._gLocked=true;
const badVersion=C.clone(stage2);badVersion.material_version='wrong';
blocked(()=>C.attachStage2(state,badVersion));
const badIds=C.clone(stage2);badIds.events[0].event_id='wrong';
blocked(()=>C.attachStage2(state,badIds));
if(state.stage2!==null)throw Error('failed attach mutated state');
const ordered=C.clone(stage2);ordered.events.reverse();C.attachStage2(state,ordered);
blocked(()=>C.attachStage2(state,stage2));
state=C.importExport(state,data);
if(C.completion(state).length)throw Error('valid import rejected');
const editF=C.clone(data);editF.records[0].f_step1.preferred='no_unique';
blocked(()=>C.importExport(state,editF));
const editG=C.clone(data);editG.records[0].g_pairs[0].label='uncertain';
blocked(()=>C.importExport(state,editG));
const restored=JSON.parse(JSON.stringify(state));
if(!C.readyForStage2(restored))throw Error('locks lost on restore');
const second=restored.stage2;restored.stage2=null;C.attachStage2(restored,second);
if(C.completion(restored).length)throw Error('draft restore incomplete');
let resumed=C.createState(pkg,data.annotator_id,data.phase);
resumed=C.importExport(resumed,data);C.attachStage2(resumed,stage2);
if(C.completion(resumed).length)throw Error('two-file import resume failed');
let s=C.createState(sPkg,sData.annotator_id,sData.phase);
s.batch_declaration=sData.batch_declaration;
const errs=C.completion(s).filter(e=>e.startsWith(sPkg.events[0].event_id));
if(errs.length!==1||!errs[0].includes('独立切分')) throw Error('precut leaked before diagnosis');
s=C.importExport(s,sData);
const editS=C.clone(sData);editS.records[0].s.free_segmentation=[
  {message_id:'r1',position:1}];
blocked(()=>C.importExport(s,editS));
""",
        [
            read("fg_stage1"),
            read("fg_stage2"),
            valid_export("fg_stage1"),
            read("s_only"),
            valid_export(),
        ],
    )


@pytest.mark.parametrize("key", FEATURES)
@pytest.mark.parametrize("value", ["present", "unclear", None])
def test_each_modifier_requires_explicit_value_and_details(key, value):
    data = valid_export()
    data["records"][0]["s"]["segments"][0]["modifiers"][key] = {"value": value}
    with pytest.raises(ValueError):
        check_export(data)


def test_exposure_overrides_and_f_no_match():
    data = valid_export("fg_stage1")
    data["batch_declaration"]["reply_seen"] = True
    for record in data["records"]:
        record["f_step1"]["reply_seen_before_step1"] = True
        record["g_reply_seen_before_judgment"] = True
    record = data["records"][0]
    data["event_exposure_exceptions"] = [
        dict(
            event_id=record["event_id"],
            reply_seen=False,
            review_comments_read=True,
            note="本事件尚未看过",
        )
    ]
    record["f_step1"]["reply_seen_before_step1"] = False
    record["g_reply_seen_before_judgment"] = False
    record["f_step1"].update(
        status="insufficient_context", acceptable=[], preferred="insufficient_context"
    )
    record["f_step2"].update(status="no_match", responded_groups=[])
    check_export(data, "fg_stage1")
    record["f_step2"]["status"] = "insufficient_context"
    check_export(data, "fg_stage1")


def test_no_g_candidates_requires_reason():
    from scripts.annotation_export_check import validate_export

    p, p2, data = read("fg_stage1"), read("fg_stage2"), valid_export("fg_stage1")
    p["events"][0]["g_candidates"] = []
    r = data["records"][0]
    r.update(g_pairs=[], g_status="not_applicable", g_reason="本事件无候选")
    validate_export(data, [p, p2])
    r.pop("g_reason")
    with pytest.raises(ValueError):
        validate_export(data, [p, p2])


def test_bad_stage_pairs():
    from scripts.annotation_export_check import validate_export

    p, p2, data = read("fg_stage1"), read("fg_stage2"), valid_export("fg_stage1")
    with pytest.raises(ValueError):
        validate_export(data, [p])
    p2["events"][0]["event_id"] = "wrong"
    with pytest.raises(ValueError):
        validate_export(data, [p, p2])
    p2 = read("fg_stage2")
    p2["batch_id"] = "wrong"
    with pytest.raises(ValueError):
        validate_export(data, [p, p2])


def test_evidence_cannot_cross_into_another_segment():
    data = valid_export()
    segments = data["records"][0]["s"]["segments"]
    segments[0]["modifiers"]["humor"] = dict(value="present", evidence=segments[1]["spans"])
    with pytest.raises(ValueError):
        check_export(data)


def test_valid_multi_action_subtypes_and_boundary_objections():
    data = valid_export()
    r = data["records"][0]
    s = r["s"]["segments"][0]
    s.update(
        actions=["COMMIT", "SOCIAL"],
        subtype={"COMMIT": "offer", "SOCIAL": "thank"},
        reason_code="multi_function",
    )
    r["s"]["free_segmentation"] = [{"message_id": "r1", "position": 5}]
    r["s"]["segment_objections"] = [
        dict(
            segment_id=s["id"],
            problem="should_merge",
            reason_code="boundary",
            proposal="与 s2 合并",
        )
    ]
    check_export(data)


@pytest.mark.parametrize(
    "stage,change",
    [
        ("s_only", lambda p: p["events"][0]["segments"][1]["spans"][0].update(start=0)),
        (
            "s_only",
            lambda p: p["events"][0]["segments"][0]["spans"][0].update(message_id="unknown"),
        ),
        ("fg_stage1", lambda p: p["events"][0]["f_units"][0]["spans"][0].update(text="wrong")),
        ("fg_stage1", lambda p: p["events"][0]["g_candidates"][0].pop("c")),
        ("fg_stage1", lambda p: p["events"][0]["g_candidates"][0]["r"][0].pop("text")),
        ("fg_stage2", lambda p: p["events"][0].pop("segmentation_diagnosis")),
    ],
)
def test_package_nested_rules(stage, change):
    from scripts.annotation_package_check import validate_package

    package = read(stage)
    change(package)
    with pytest.raises(ValueError):
        validate_package(package)


@pytest.mark.parametrize("field", ["reason_code", "subtype"])
def test_malformed_json_field_reports_validation_error(field):
    data = valid_export()
    segment = data["records"][0]["s"]["segments"][0]
    segment.update(actions=["COMMIT", "SOCIAL"], reason_code="multi_function")
    segment[field] = []
    with pytest.raises(ValueError):
        check_export(data)


def test_core_and_python_reject_the_same_conditional_omissions():
    # Exercise real completion checks, rather than a second JS fixture generator with defaults.
    cases = []
    for stage in ["s_only", "fg_stage1"]:
        mutations = [
            lambda d: d["records"][1]["s"]["segments"][0].update(status=None),
            lambda d: d["records"][1]["s"]["segments"][0].update(actions=[]),
            lambda d: d["records"][1]["s"]["segments"][0].update(actions=["ACK", "INFORM"]),
            lambda d: d["records"][1]["s"]["segments"][0].update(actions=["OTHER"]),
            lambda d: d["records"][1]["s"]["segments"][0].update(
                status="uncertain", actions=[], candidates=["ASK"]
            ),
            lambda d: d["records"][1]["s"]["segments"][0].update(
                status="insufficient_context", actions=[]
            ),
        ]
        for key in FEATURES:
            for value in [None, "present", "unclear"]:
                mutations.append(
                    lambda d, k=key, v=value: d["records"][1]["s"]["segments"][0][
                        "modifiers"
                    ].update({k: {"value": v}})
                )
        if stage == "fg_stage1":
            mutations.extend(
                [
                    lambda d: d["records"][1]["f_step1"].update(locked=False),
                    lambda d: d["records"][1]["f_step1"].update(preferred="missing"),
                    lambda d: d["records"][1]["f_step1"].update(acceptable=[]),
                    lambda d: d["records"][1]["f_step1"].update(mode=None),
                    lambda d: d["records"][1]["f_step2"].update(status=None),
                    lambda d: d["records"][1]["f_step2"].update(mode=None),
                    lambda d: d["records"][1]["f_step2"].update(responded_groups=[]),
                    lambda d: d["records"][1]["g_pairs"][0].update(label=None),
                    lambda d: d["records"][1]["g_pairs"][0].update(reason=""),
                    lambda d: d["records"][1]["g_pairs"][0].update(evidence=[]),
                ]
            )
        for mutate in mutations:
            data = valid_export(stage)
            mutate(data)
            with pytest.raises(ValueError):
                check_export(data, stage)
            cases.append([read(stage), read("fg_stage2") if stage == "fg_stage1" else None, data])
    run_core(
        """
for(const [pkg,second,data] of JSON.parse(process.argv[1])) {
  const state=AnnotationCore.createState(pkg,data.annotator_id,data.phase);
  state.batch_declaration=data.batch_declaration;
  state.stage2=second;
  state.records=Object.fromEntries(data.records.map(r=>[r.event_id,{...r,_gLocked:true}]));
  if(!AnnotationCore.completion(state).length)throw Error('conditional omission accepted');
}
""",
        cases,
    )


def test_all_page_scripts_parse_without_browser():
    scripts = re.findall(
        r'<script(?: id="annotation-core")?>(.*?)</script>', PAGE.read_text(), re.S
    )
    assert len(scripts) == 2
    run_core("for (const code of JSON.parse(process.argv[1])) new Function(code);", scripts)


@pytest.mark.parametrize("field", ["g_pairs", "segments"])
def test_non_object_array_entry_is_a_validation_error(field):
    data = valid_export("fg_stage1")
    r = data["records"][0]
    if field == "g_pairs":
        r["g_pairs"].insert(0, None)
    else:
        r["s"]["segments"].insert(0, None)
    with pytest.raises(ValueError):
        check_export(data, "fg_stage1")
