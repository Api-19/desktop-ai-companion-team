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
        if stage in ("fg_stage1", "g_only"):
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
        if stage == "g_only":
            record["s"] = None
            records.append(record)
            continue
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
    validate_export(data, [read("v12_example_fg_stage1"), read("v12_example_fg_stage2")])


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
    assert '>清除本机草稿</button>' in html
    assert 'id="clear-draft" type="button" disabled' in html
    assert "localStorage.removeItem(storageKey(state))" in html


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
        ui = re.findall(r"<script>(.*?)</script>", PAGE.read_text(), re.S)[0]
        cleared = run_node(
            node,
            core + "\n" + """
const elements = new Map();
globalThis.document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, {});
  return elements.get(id);
}};
const storage = new Map();
globalThis.localStorage = {
  setItem: (key, value) => storage.set(key, value),
  removeItem: key => storage.delete(key)
};
let confirmed = false, confirmations = 0;
globalThis.confirm = () => { confirmations++; return confirmed; };
""" + ui + "\n" + """
const pkg = JSON.parse(process.argv[1]);
state = C.createState(pkg, 'tester-a', 'training');
save();
const key = storageKey(state);
const otherKeys = [
  'unrelated',
  storageKey({...state, annotator_id:'tester-b'}),
  storageKey({...state, package:{...pkg, batch_id:'another-batch'}}),
  storageKey({...state, package:{...pkg, material_version:'another-version'}})
];
for (const other of otherKeys) localStorage.setItem(other, 'keep');
$('clear-draft').onclick();
if (!storage.has(key)) throw Error('cancel removed draft');
confirmed = true;
$('clear-draft').onclick();
if (storage.has(key)) throw Error('draft retained');
if (otherKeys.some(other => storage.get(other) !== 'keep')) throw Error('other key removed');
if (confirmations !== 2) throw Error('confirmation missing');
if ($('notice').textContent !== '本机草稿已清除') throw Error('clear notice missing');
""",
            p,
        )
        assert cleared.returncode == 0, cleared.stderr
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


def self_export():
    package = read("self_seg_s_only")
    data = valid_export()
    data.update({k: package[k] for k in ("batch_id", "material_version")})
    template = data["records"][0]["s"]["segments"][0]
    for record, event in zip(data["records"], package["events"], strict=True):
        record.update({k: event[k] for k in ("event_id", "material_ref", "sample_role")})
        segments = []
        for message in event["reply"]:
            raw = message["text"]
            start, end = len(raw) - len(raw.lstrip()), len(raw.rstrip())
            segments.append(
                dict(
                    **{
                        k: copy.deepcopy(v)
                        for k, v in template.items()
                        if k not in ("id", "spans")
                    },
                    id=f"s{len(segments) + 1}",
                    spans=[
                        dict(
                            message_id=message["message_id"],
                            start=start,
                            end=end,
                            text=raw[start:end],
                        )
                    ],
                )
            )
        record["s"] = dict(segments=segments, segmentation_source="self", segment_objections=[])
    return data


def test_self_package_cli():
    test_package_cli("self_seg_s_only")


@pytest.mark.parametrize(
    "stage,flag",
    [("s_only", True), ("fg_stage2", True), ("s_only", False), ("fg_stage2", False)],
)
def test_self_package_modes(stage, flag):
    from scripts.annotation_package_check import validate_package

    package = read(stage)
    package["self_segmentation"] = flag
    if flag:
        for event in package["events"]:
            event["segments"] = []
            event.pop("segmentation_diagnosis")
    validate_package(package)
    run_core("AnnotationCore.validatePackage(JSON.parse(process.argv[1]));", package)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["events"][0].update(segments=read("s_only")["events"][0]["segments"]),
        lambda p: p["events"][0].update(segmentation_diagnosis=True),
        lambda p: p["events"][0].update(segmentation_diagnosis=0),
        lambda p: p["events"][0].pop("segments"),
        lambda p: p.update(self_segmentation=1),
        lambda p: p.update(self_segmentation="true"),
    ],
)
def test_bad_self_package(mutation):
    from scripts.annotation_package_check import validate_package

    package = read("self_seg_s_only")
    mutation(package)
    with pytest.raises(ValueError):
        validate_package(package)
    run_core(
        """
let rejected=false;
try {AnnotationCore.validatePackage(JSON.parse(process.argv[1]));} catch(e) {rejected=true;}
if(!rejected)throw Error('invalid self package accepted');
""",
        package,
    )


@pytest.mark.parametrize("flag", [True, False])
def test_stage1_rejects_self_field(flag):
    from scripts.annotation_package_check import validate_package

    package = read("fg_stage1")
    package["self_segmentation"] = flag
    with pytest.raises(ValueError):
        validate_package(package)
    run_core(
        """
let rejected=false;
try {AnnotationCore.validatePackage(JSON.parse(process.argv[1]));} catch(e) {rejected=true;}
if(!rejected)throw Error('stage1 accepted self field');
""",
        package,
    )


def test_valid_self_export():
    check_export(self_export(), "self_seg_s_only")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["segments"][1].update(spans=copy.deepcopy(s["segments"][0]["spans"])),
        lambda s: s["segments"].pop(),
        lambda s: s["segments"][0]["spans"][0].update(text="错误"),
        lambda s: s.pop("segmentation_source"),
        lambda s: s.update(segmentation_source="precut"),
        lambda s: s.update(free_segmentation=[]),
        lambda s: s["segments"].reverse(),
        lambda s: s["segments"][1].update(id="s1"),
        lambda s: s["segments"][0]["spans"].extend(s["segments"][1]["spans"]),
        lambda s: s["segments"][0]["spans"][0].update(start=True),
        lambda s: s["segments"][0]["spans"][0].update(message_id="unknown"),
        lambda s: s["segments"][0].update(actions=[]),
    ],
)
def test_bad_self_exports(mutation):
    data = self_export()
    mutation(data["records"][0]["s"])
    with pytest.raises(ValueError):
        check_export(data, "self_seg_s_only")
    run_core(
        """
const C=AnnotationCore,[pkg,data]=JSON.parse(process.argv[1]);
const state=C.createState(pkg,data.annotator_id,data.phase);
state.batch_declaration=data.batch_declaration;
state.records=Object.fromEntries(data.records.map(r=>[r.event_id,r]));
if(!C.completion(state).length)throw Error('invalid self export accepted');
""",
        [read("self_seg_s_only"), data],
    )


@pytest.mark.parametrize("fg", [False, True])
def test_node_self_segmentation_roundtrip(fg):
    from scripts.annotation_export_check import validate_export

    initial = read("fg_stage1") if fg else read("self_seg_s_only")
    target = read("fg_stage2") if fg else initial
    if fg:
        target["self_segmentation"] = True
        for event in target["events"]:
            event.update(segments=[], segmentation_diagnosis=False)
    output = run_core(
        """
const C=AnnotationCore,[pkg,target,fgData]=JSON.parse(process.argv[1]);
let state=C.createState(pkg,'tester-a','training');
state.batch_declaration={reply_seen:false,review_comments_read:false};
if(fgData) {
  for(const r of fgData.records) state.records[r.event_id]={...C.clone(r),s:null,_gLocked:true};
  C.attachStage2(state,target);
}
if(!C.completion(state).length)throw Error('unconfirmed segmentation accepted');
for(const e of target.events) {
  const r=state.records[e.event_id], m=e.reply[0], length=Array.from(m.text).length;
  const boundaries=fgData?[]:[{message_id:m.message_id,position:1},
    {message_id:m.message_id,position:length-1},{message_id:m.message_id,position:6}]
      .filter(b=>b.position<length);
  C.lockSelfSegmentation(r,e,boundaries);
  if(!fgData && e===target.events[0]) {
    const expected=[{id:'s1',spans:[{message_id:'r1',start:2,end:6,text:'收到😀。'}]},
      {id:'s2',spans:[{message_id:'r1',start:9,end:14,text:'我来画图。'}]},
      {id:'s3',spans:[{message_id:'r2',start:1,end:8,text:'先量一下宽度。'}]}];
    if(!C.same(r.s.segments.map(s=>({id:s.id,spans:s.spans})),expected))
      throw Error('code point/trim/message split');
  }
  for(const s of r.s.segments) {
    s.status='judged';s.actions=['INFORM'];
    for(const k of Object.keys(C.FEATURES))s.modifiers[k]={value:'absent'};
  }
}
const output=C.buildExport(state);
if(output.records.some(r=>r.s.segmentation_source!=='self'||C.has(r.s,'free_segmentation')))
  throw Error('self export fields');
let resumed=C.importExport(C.createState(pkg,'tester-a','training'),output);
if(fgData)C.attachStage2(resumed,target);
if(!C.same(C.buildExport(resumed),output))throw Error('import roundtrip differs');
if(fgData) {
  const restored=C.clone(resumed), second=restored.stage2;
  restored.stage2=null;C.attachStage2(restored,second);
  if(C.completion(restored).length)throw Error('stage2 draft restore failed');
}
const e=target.events[0], r=resumed.records[e.event_id];
r.s.segment_objections.push({segment_id:'s1',problem:'should_split',reason_code:'boundary',proposal:'拆开'});
C.resetSelfSegmentation(r,e);
if(r.s.segments.length||r.s.segment_objections.length||C.has(r.s,'segmentation_source'))
  throw Error('S annotations retained');
if(!C.completion(resumed).length)throw Error('reset still complete');
C.lockSelfSegmentation(r,e,[]);
if(r.s.segments.length!==e.reply.length||r.s.segments.some(s=>s.status!==null||s.actions.length))
  throw Error('zero boundaries/reset labels');
process.stdout.write(JSON.stringify(output));
""",
        [initial, target, valid_export("fg_stage1") if fg else None],
    )
    validate_export(json.loads(output), [initial, target] if fg else [initial])


def test_node_self_segmentation_ui():
    ui = re.findall(r"<script>(.*?)</script>", PAGE.read_text(), re.S)[0]
    run_core(
        """
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.classList={add(){}};}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  setAttribute(){}
}
const elements=new Map();
globalThis.document={createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text}),
  getElementById(id){
    if(!elements.has(id))elements.set(id,new Element('div'));return elements.get(id);
  }};
globalThis.localStorage={setItem(){}};
globalThis.Option=function(text,value){this.textContent=text;this.value=value;};
let confirmed=false, confirmations=0;
globalThis.confirm=()=>{confirmations++;return confirmed;};
""" + ui + "\n" + """
state=C.createState(JSON.parse(process.argv[1]),'tester-a','training');
const flatten=n=>[n,...(n.children||[]).flatMap(flatten)];
const findButton=text=>flatten($('event-view')).find(n=>n.tag==='button'&&n.textContent===text);
renderEvent();
if(findButton('重新切分')||!findButton('确定切分'))throw Error('initial segmentation UI');
const boundary=flatten($('event-view')).find(n=>n.tag==='button'&&n.textContent==='·');
boundary.onclick();
findButton('确定切分').onclick();
const r=state.records[state.package.events[0].event_id];
if(r.s.segmentation_source!=='self'||findButton('确定切分')||!findButton('重新切分'))
  throw Error('confirmation did not lock');
r.s.segments[0].status='judged';r.s.segments[0].actions=['ACK'];
const previous=JSON.stringify(r.s);
findButton('重新切分').onclick();
if(JSON.stringify(r.s)!==previous)throw Error('cancel erased labels');
confirmed=true;findButton('重新切分').onclick();
if(confirmations!==2||r.s.segments.length||!findButton('确定切分'))
  throw Error('reset confirmation or flow');
if($('notice').className==='error')throw Error($('notice').textContent);
""",
        read("self_seg_s_only"),
    )


def test_node_self_import_cannot_bypass_reset():
    run_core(
        """
const C=AnnotationCore,[pkg,data]=JSON.parse(process.argv[1]);
let state=C.importExport(C.createState(pkg,data.annotator_id,data.phase),data);
const next=C.clone(state),e=pkg.events[0],r=next.records[e.event_id];
C.lockSelfSegmentation(r,e,[{message_id:'r1',position:6}]);
for(const s of r.s.segments) {
  s.status='judged';s.actions=['INFORM'];
  for(const k of Object.keys(C.FEATURES))s.modifiers[k]={value:'absent'};
}
const changed=C.buildExport(next);
let rejected=false;
try {C.importExport(state,changed);} catch(error) {rejected=true;}
if(!rejected)throw Error('import bypassed segmentation lock');
const untouched=C.clone(state.records[pkg.events[1].event_id]);
C.resetSelfSegmentation(state.records[e.event_id],e);
if(!C.same(untouched,state.records[pkg.events[1].event_id]))throw Error('other event erased');
state=C.importExport(state,changed);
if(!C.same(C.buildExport(state),changed))throw Error('import after reset failed');
""",
        [read("self_seg_s_only"), self_export()],
    )


@pytest.mark.parametrize("mutation", [
    lambda p: p["events"][0].update(reply=[]),
    lambda p: p["events"][0].update(f_units=[]),
    lambda p: p["events"][0].update(segments=[]),
    lambda p: p["events"][0].update(segmentation_diagnosis=False),
    lambda p: p.update(self_segmentation=False),
    lambda p: p["events"][0]["permission"].pop("note"),
    lambda p: p["events"][0]["permission"].update(tease="invalid"),
    lambda p: p["events"][0].update(g_candidates=[]),
])
def test_g_only_package_rejects(mutation):
    from scripts.annotation_package_check import validate_package

    package = read("g_only")
    mutation(package)
    with pytest.raises(ValueError):
        validate_package(package)


def test_g_only_package_and_export():
    from scripts.annotation_package_check import validate_package

    validate_package(read("g_only"))
    check_export(valid_export("g_only"), "g_only")


@pytest.mark.parametrize("mutation", [
    lambda d: d["records"][0]["g_pairs"].pop(),
    lambda d: d["records"][0]["g_pairs"][0].update(label="invalid"),
    lambda d: d["records"][0]["g_pairs"][0].update(evidence=["missing"]),
    lambda d: d["records"][0]["g_pairs"][0].update(reason=""),
    lambda d: d["records"][0]["g_pairs"][0].update(evidence=[]),
    lambda d: d["records"][0]["g_pairs"][0].update(c_message_ids=[]),
    lambda d: d["records"][0]["g_pairs"][0].update(r_message_ids=[]),
    lambda d: d["records"][0]["g_pairs"][0].update(x_message_ids=[]),
    lambda d: d["records"][0].update(s={}),
    lambda d: d["records"][0].update(f_step1={}),
    lambda d: d["records"][0].update(f_step2={}),
    lambda d: d["records"].pop(),
])
def test_g_only_export_rejects(mutation):
    data = valid_export("g_only")
    mutation(data)
    with pytest.raises(ValueError):
        check_export(data, "g_only")


def test_node_g_only_annotation_lock_export():
    output = run_core("""
const C=AnnotationCore, pkg=JSON.parse(process.argv[1]);
const state=C.createState(pkg,'tester-a','training');
state.batch_declaration={reply_seen:false,review_comments_read:false};
if(!C.completion(state).length)throw Error('blank annotations accepted');
for(const e of pkg.events){
  const r=state.records[e.event_id];
  if(r.g_pairs.some(p=>p.label!==null))throw Error('default label');
  for(const pair of r.g_pairs){
    pair.label='applicable';pair.reason='前提成立';pair.evidence=[pair.c_message_ids[0]];
  }
  if(C.gErrors(r,e,false).length)throw Error('valid G rejected');
  if(!C.gErrors(r,e).length)throw Error('unlocked G accepted');
  r._gLocked=true;
}
if(C.readyForStage2(state))throw Error('G-only allows stage2');
C.applyExposure(state);
const data=C.buildExport(state);
if(JSON.stringify(data).includes('permission'))throw Error('permission exported');
const restored=C.importExport(C.createState(pkg,'tester-a','training'),data);
const changed=C.clone(data);changed.records[0].g_pairs[0].label='uncertain';
let blocked=false;try{C.importExport(restored,changed);}catch(e){blocked=true;}
if(!blocked)throw Error('locked G changed by import');
state.batch_declaration.reply_seen=true;
state.event_exposure_exceptions=[{event_id:pkg.events[1].event_id,reply_seen:false,review_comments_read:false,note:'例外'}];
C.applyExposure(state);
const exposed=C.buildExport(state);
if(!exposed.records[0].g_reply_seen_before_judgment||
   exposed.records[1].g_reply_seen_before_judgment)throw Error('exposure flags differ');
console.log(JSON.stringify([data,exposed]));
""", read("g_only"))
    for data in json.loads(output):
        check_export(data, "g_only")


V12 = "rcpem-public-pilot-v1.2"
FEATURES_V12 = [
    "mitigation", "politeness", "emotion", "evaluation", "rapport", "humor", "irony",
    "intensification",
]


def v12_export(self_segmentation=False):
    data = self_export() if self_segmentation else valid_export()
    package = read("v12_self_seg_s_only" if self_segmentation else "v12_s_only")
    data.update({k: package[k] for k in ("protocol_version", "material_version", "batch_id")})
    for record in data["records"]:
        for segment in record["s"]["segments"]:
            segment["modifiers"] = {k: {"value": "absent"} for k in FEATURES_V12}
            segment["content_attributes"] = {"self_disclosure": {"value": "absent"}}
    return data, package


@pytest.mark.parametrize("name", ["s_only", "self_seg_s_only", "example_fg_stage1",
                                  "example_fg_stage2"])
def test_v12_package_cli(name):
    test_package_cli("v12_" + name)


@pytest.mark.parametrize("protocol", [None, "rcpem-public-pilot-v1.0", "rcpem-public-pilot-v1.3"])
def test_unknown_protocol_rejected(protocol):
    from scripts.annotation_package_check import validate_package

    package = read("s_only")
    package["protocol_version"] = protocol
    with pytest.raises(ValueError):
        validate_package(package)
    assert run_core("""
const p=JSON.parse(process.argv[1]);
try {AnnotationCore.validatePackage(p);} catch(e) {process.stdout.write('rejected');}
""", package) == "rejected"


@pytest.mark.parametrize("self_segmentation", [False, True])
def test_v12_valid_export(self_segmentation):
    from scripts.annotation_export_check import validate_export

    data, package = v12_export(self_segmentation)
    validate_export(data, [package])


@pytest.mark.parametrize("key", [*FEATURES_V12, "self_disclosure"])
@pytest.mark.parametrize("value", ["present", "absent", "unclear"])
def test_v12_valid_feature_states(key, value):
    from scripts.annotation_export_check import validate_export

    data, package = v12_export()
    segment = data["records"][0]["s"]["segments"][0]
    fields = segment["content_attributes"] if key == "self_disclosure" else segment["modifiers"]
    if key == "evaluation" and value == "present":
        value = "positive"
    fields[key] = {"value": value}
    if value in ("present", "positive"):
        fields[key]["evidence"] = copy.deepcopy(segment["spans"])
        if key == "irony":
            fields[key]["reason"] = "字面肯定但语境表示否定"
            m = package["events"][0]["partner"][0]
            fields[key]["evidence"].append(dict(message_id=m["message_id"], start=0,
                                               end=len(m["text"]), text=m["text"]))
    elif value == "unclear":
        fields[key]["reason"] = "语境不足"
    validate_export(data, [package])
    run_core("""
const [p,d]=JSON.parse(process.argv[1]), C=AnnotationCore;
const s=C.createState(p,d.annotator_id,d.phase);
s.batch_declaration=d.batch_declaration;
s.records=Object.fromEntries(d.records.map(r=>[r.event_id,r]));
if(C.completion(s).length)throw Error(C.completion(s).join('\\n'));
""", [package, data])


@pytest.mark.parametrize("value", ["positive", "negative", "mixed", "absent", "unclear"])
def test_v12_evaluation_values(value):
    from scripts.annotation_export_check import validate_export

    data, package = v12_export()
    segment = data["records"][0]["s"]["segments"][0]
    m = segment["modifiers"]["evaluation"] = {"value": value}
    if value in ("positive", "negative", "mixed"):
        m["evidence"] = copy.deepcopy(segment["spans"])
    if value == "unclear":
        m["reason"] = "褒贬不清"
    validate_export(data, [package])


@pytest.mark.parametrize("key", [*FEATURES_V12, "self_disclosure"])
@pytest.mark.parametrize("error", ["missing", "invalid", "no_evidence", "wrong_slice",
                                   "no_reason", "absent_evidence", "unselected"])
def test_v12_feature_rules_reject(key, error):
    from scripts.annotation_export_check import validate_export

    data, package = v12_export()
    segment = data["records"][0]["s"]["segments"][0]
    fields = segment["content_attributes"] if key == "self_disclosure" else segment["modifiers"]
    present = "positive" if key == "evaluation" else "present"
    if error == "unselected":
        fields[key] = {"value": None}
    elif error == "missing":
        fields.pop(key)
    elif error == "invalid":
        fields[key] = {"value": "present" if key == "evaluation" else "positive"}
    elif error == "no_evidence":
        fields[key] = {"value": present}
    elif error == "wrong_slice":
        fields[key] = {"value": present, "evidence": [dict(segment["spans"][0], text="错误")],
                       "reason": "反向立场"}
    elif error == "no_reason":
        fields[key] = {"value": "unclear", "reason": " "}
    else:
        fields[key] = {"value": "absent", "evidence": segment["spans"]}
    with pytest.raises(ValueError):
        validate_export(data, [package])
    assert run_core("""
const [p,d]=JSON.parse(process.argv[1]), C=AnnotationCore;
const s=C.createState(p,d.annotator_id,d.phase);
s.batch_declaration=d.batch_declaration;
s.records=Object.fromEntries(d.records.map(r=>[r.event_id,r]));
process.stdout.write(String(C.completion(s).length>0));
""", [package, data]) == "true"


@pytest.mark.parametrize("change", [
    lambda s: s["modifiers"].update(affection={"value": "absent"}),
    lambda s: s.pop("content_attributes"),
    lambda s: s["modifiers"].update(irony={"value": "present", "evidence": s["spans"]}),
    lambda s: s.update(actions=["REQUEST"], subtype="promise"),
    lambda s: s.update(actions=["REQUEST"], subtype={"REQUEST": "other"}),
    lambda s: s.update(actions=["INFORM"], subtype="seek_care"),
])
def test_v12_segment_rules_reject(change):
    from scripts.annotation_export_check import validate_export

    data, package = v12_export()
    change(data["records"][0]["s"]["segments"][0])
    with pytest.raises(ValueError):
        validate_export(data, [package])
    assert run_core("""
const [p,d]=JSON.parse(process.argv[1]),C=AnnotationCore;
const s=C.createState(p,d.annotator_id,d.phase);
s.batch_declaration=d.batch_declaration;
s.records=Object.fromEntries(d.records.map(r=>[r.event_id,r]));
process.stdout.write(String(C.completion(s).length>0));
""", [package, data]) == "true"


@pytest.mark.parametrize("subtype", [None, "seek_care", {"REQUEST": "seek_care"}])
def test_v12_request_subtype(subtype):
    from scripts.annotation_export_check import validate_export

    data, package = v12_export()
    data["records"][0]["s"]["segments"][0].update(actions=["REQUEST"], subtype=subtype)
    validate_export(data, [package])


@pytest.mark.parametrize("self_segmentation", [False, True])
def test_node_v12_annotation_export_cli(tmp_path, self_segmentation):
    name = "v12_self_seg_s_only" if self_segmentation else "v12_s_only"
    output = run_core("""
const C=AnnotationCore,p=JSON.parse(process.argv[1]);
const s=C.createState(p,'tester-a','training');
s.batch_declaration={reply_seen:false,review_comments_read:false};
for(const e of p.events) {
  const r=s.records[e.event_id];
  if(p.self_segmentation)C.lockSelfSegmentation(r,e,[],p.protocol_version);
  else if(e.segmentation_diagnosis)r.s.free_segmentation=[];
  for(const seg of r.s.segments) {
    if(Object.keys(seg.modifiers).length!==8 || Object.values(seg.modifiers).some(m=>m.value!==null)
       || seg.content_attributes.self_disclosure.value!==null)throw Error('blank defaults');
    seg.status='judged';seg.actions=['REQUEST'];seg.subtype='seek_care';
    for(const k of Object.keys(seg.modifiers))seg.modifiers[k]={value:'absent'};
    seg.modifiers.evaluation={value:'mixed',evidence:C.clone(seg.spans)};
    seg.content_attributes.self_disclosure={value:'present',evidence:C.clone(seg.spans)};
  }
}
const output=C.buildExport(s);
const restored=C.importExport(C.createState(p,'tester-a','training'),output);
if(!C.same(C.buildExport(restored),output))throw Error('roundtrip');
if(output.protocol_version!==p.protocol_version)throw Error('export protocol');
process.stdout.write(JSON.stringify(output));
""", read(name))
    path = tmp_path / "export.json"
    path.write_text(output)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/annotation_export_check.py"),
                             str(path), "--package", str(PACKAGES / f"{name}.json")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_v12_export_protocol_must_match_package():
    from scripts.annotation_export_check import validate_export

    data, package = v12_export()
    data["protocol_version"] = "rcpem-public-pilot-v1.1"
    with pytest.raises(ValueError):
        validate_export(data, [package])


@pytest.mark.parametrize("mode", ["g_only", "fg", "fg_self"])
def test_node_v12_g_and_fg_flows(mode):
    from scripts.annotation_export_check import validate_export

    fg = mode != "g_only"
    initial = read("fg_stage1" if fg else "g_only")
    initial["protocol_version"] = V12
    target = read("fg_stage2") if fg else None
    if target:
        target["protocol_version"] = V12
        if mode == "fg_self":
            target["self_segmentation"] = True
            for event in target["events"]:
                event.update(segments=[], segmentation_diagnosis=False)
    output = run_core("""
const C=AnnotationCore,[p,target,answers]=JSON.parse(process.argv[1]);
let state=C.createState(p,answers.annotator_id,answers.phase);
state.batch_declaration=answers.batch_declaration;
for(const a of answers.records) {
  const r=state.records[a.event_id];
  r.f_step1=a.f_step1;r.g_status=a.g_status;r.g_pairs=a.g_pairs;r._gLocked=true;
  if(a.g_reason)r.g_reason=a.g_reason;
}
if(target) {
  C.attachStage2(state,target);
  for(const e of target.events) {
    const r=state.records[e.event_id];
    r.f_step2=answers.records.find(a=>a.event_id===e.event_id).f_step2;
    if(target.self_segmentation)C.lockSelfSegmentation(r,e,[],p.protocol_version);
    else if(e.segmentation_diagnosis)r.s.free_segmentation=[];
    for(const seg of r.s.segments) {
      if(Object.keys(seg.modifiers).length!==8||seg.content_attributes.self_disclosure.value!==null)
        throw Error('stage2 version');
      seg.status='judged';seg.actions=['COMMIT','SOCIAL','REQUEST'];
      seg.reason_code='multi_function';
      seg.subtype={COMMIT:'offer',SOCIAL:'thank',REQUEST:'seek_care'};
      for(const k of Object.keys(seg.modifiers))seg.modifiers[k]={value:'absent'};
      seg.content_attributes.self_disclosure={value:'absent'};
    }
  }
}
const output=C.buildExport(state);
let restored=C.importExport(C.createState(p,answers.annotator_id,answers.phase),output);
if(target)C.attachStage2(restored,target);
if(!C.same(C.buildExport(restored),output))throw Error('resume');
process.stdout.write(JSON.stringify(output));
""", [initial, target, valid_export("fg_stage1" if fg else "g_only")])
    validate_export(json.loads(output), [initial, target] if fg else [initial])


def test_node_v12_fields_ui():
    ui = re.findall(r"<script>(.*?)</script>", PAGE.read_text(), re.S)[0]
    run_core("""
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.classList={add(){}};}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  get options(){return this.children;}
  contains(node){return this===node;}
}
const elements=new Map();
globalThis.document={createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text}),
  getElementById(id){
    if(!elements.has(id))elements.set(id,new Element('div'));return elements.get(id);
  },
  createRange(){return {
    selectNodeContents(){},
    setEnd(node,offset){this.prefix=node.textContent.slice(0,offset);},
    toString(){return this.prefix;}
  };}
};
globalThis.localStorage={setItem(){}};
globalThis.Option=function(text,value){this.textContent=text;this.value=value;};
""" + ui + "\n" + """
const p=JSON.parse(process.argv[1]);state=C.createState(p,'tester-a','training');
const e=p.events[0],r=state.records[e.event_id],seg=r.s.segments[0];
state.batch_declaration={reply_seen:false,review_comments_read:false};
r.s.free_segmentation=[];seg.status='judged';seg.actions=['REQUEST'];
const flatten=n=>[n,...(n.children||[]).flatMap(flatten)];
const feature=key=>flatten($('event-view')).find(
  n=>n.className==='feature'&&n.children[0].textContent.startsWith(key));
const radios=box=>flatten(box).filter(n=>n.type==='radio');
const choose=(key,value)=>radios(feature(key)).find(n=>n.value===value).onchange();
renderEvent();
if(flatten($('event-view')).filter(n=>n.className==='feature').length!==r.s.segments.length*9)
  throw Error('field count');
for(const k of [...Object.keys(C.FEATURES_V12),'self_disclosure']) {
  if(radios(feature(k)).some(n=>n.checked))
  throw Error('default selection');
  if(!feature(k).children[0].textContent.includes((C.FEATURES_V12[k]||C.CONTENT_V12[k])))
  throw Error('definition');
}
if(!C.same(radios(feature('evaluation')).map(n=>n.value),
  ['positive','negative','mixed','absent','unclear']))
  throw Error('evaluation choices');
if(!feature('intensification').children[0].textContent.includes('候选'))
  throw Error('candidate label');
const subtype=flatten($('event-view')).find(
  n=>n.tag==='select'&&n.options.some(o=>o.value==='seek_care'));
if(!subtype||!C.same(subtype.options.map(o=>o.value),['','seek_care']))
  throw Error('REQUEST dropdown');
subtype.value='seek_care';subtype.onchange();if(seg.subtype!=='seek_care')
  throw Error('REQUEST selection');
function addEvidence(key,text) {
  const box=feature(key),raw=flatten(box).find(
    n=>n.className==='evidence-text'&&n.textContent===text);
  const value=raw.textContent, start=value.indexOf('😀'), offset=start<0?0:start;
  const selected=start<0?value:'😀';
  globalThis.window={getSelection:()=>({rangeCount:1,isCollapsed:false,
    getRangeAt:()=>({startContainer:raw,endContainer:raw,startOffset:offset,toString:()=>selected}),removeAllRanges(){}})};
  flatten(box).find(n=>n.children?.includes(raw)).children.find(n=>n.tag==='button'&&n.textContent==='添加这段证据').onclick();
}
choose('evaluation','negative');addEvidence('evaluation',seg.spans[0].text);
const ev=seg.modifiers.evaluation.evidence[0];
if(ev.text!=='😀'||ev.start!==4||ev.end!==5)
  throw Error('unicode drag evidence');
choose('self_disclosure','present');addEvidence('self_disclosure',seg.spans[0].text);
if(seg.content_attributes.self_disclosure.evidence[0].text!=='😀')
  throw Error('content drag evidence');
choose('irony','present');
const reason=flatten(feature('irony')).find(n=>n.tag==='textarea');
if(!reason)
  throw Error('irony reason control');reason.value='字面肯定而语境否定';reason.oninput();
addEvidence('irony',e.partner[0].text);
if(seg.modifiers.irony.evidence[0].message_id!==e.partner[0].message_id)
  throw Error('irony context evidence');
choose('irony','absent');if(!C.same(seg.modifiers.irony,{value:'absent'}))
  throw Error('stale details');
choose('politeness','unclear');
const unclear=flatten(feature('politeness')).find(n=>n.tag==='textarea');
unclear.value='语境不明';unclear.oninput();if(seg.modifiers.politeness.reason!=='语境不明')
  throw Error('unclear reason');
const key=storageKey(state),legacy=C.createState(
  {...p,protocol_version:'rcpem-public-pilot-v1.1'},'tester-a','training');
if(key===storageKey(legacy)||!storageKey(legacy).startsWith('annotation-v1.1:'))
  throw Error('versioned drafts');
if($('notice').className==='error')
  throw Error($('notice').textContent);
""", read("v12_s_only"))
