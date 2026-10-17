# ruff: noqa: RUF001
import hashlib
import json
import random
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import companion.privacy.paired_snippets as paired_snippets
from companion.contracts import ChatRecord, SpeakerRole
from companion.contracts.style import StyleSnippet
from companion.privacy import (
    PairedSnippet,
    PrivacyValidationError,
    RiskCategory,
    build_paired_snippets,
    dataset_fingerprint,
    delete_snippets,
    read_snippets,
    sample_for_audit,
    split_holdout,
    write_snippets,
)

BASE_TIME = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)
CREATED_AT = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
OTHER = SpeakerRole.OTHER
OWNER = SpeakerRole.OWNER


def rec(
    message_id: str,
    text: str | None,
    *,
    minute: int,
    role: SpeakerRole,
    content_type: str = "text",
    conversation_id: str = "synthetic-conv-a",
) -> ChatRecord:
    return ChatRecord(
        conversation_id=conversation_id,
        message_id=message_id,
        timestamp=BASE_TIME + timedelta(minutes=minute),
        speaker_role=role,
        content_type=content_type,
        text=text,
    )


def build(records: list[ChatRecord], **kwargs: object) -> list[PairedSnippet]:
    return build_paired_snippets(records, created_at=CREATED_AT, **kwargs)


@pytest.fixture
def veiled():
    from companion.privacy.veiled import load_veiled_terms
    path = Path(__file__).resolve().parents[2] / "config/privacy/veiled_terms.json"
    return load_veiled_terms(path)


@pytest.mark.parametrize("position", ["context", "partner", "owner", "hidden_partner",
                                      "hidden_owner", "truncated_partner", "truncated_context"])
@pytest.mark.parametrize(("bait", "excluded", "flag"), [
    ("法*死*你", True, None), ("袜 子", False, "body:袜子"),
])
def test_veiled_scans_all_original_turns_before_redaction(
    veiled, position, bait, excluded, flag,
):
    from companion.privacy.paired_snippets import PseudonymEntity
    records = [rec("context", "前文", minute=0, role=OWNER)]
    records += [rec(f"p{i}", f"合成问题{i}", minute=i + 1, role=OTHER) for i in range(4)]
    records += [rec(f"o{i}", f"合成回答{i}", minute=i + 5, role=OWNER) for i in range(4)]
    index = {"context": 0, "partner": 2, "owner": 5, "hidden_partner": 1,
             "hidden_owner": 8, "truncated_partner": 4, "truncated_context": 0}[position]
    records[index].text = ("长" * 80 if position.startswith("truncated") else "") + bait
    options = {"context_turns": 1, "excluded_risks": frozenset(),
               "redaction_terms": {"name": [bait]}, "entities": [
        PseudonymEntity(id="synthetic", kind="person", aliases=[bait], pseudonym="小甲"),
    ]}
    baseline = build(records, **options)
    assert len(baseline) == 1
    result = build(records, veiled=veiled, **options)
    if excluded:
        assert result == []
    else:
        assert len(result) == 1
        assert result[0].privacy_flags == [flag]
        assert result[0].partner_turns == baseline[0].partner_turns


def test_veiled_is_per_pair_and_context_is_opt_in(veiled):
    records = [
        rec("c", "法死你", minute=0, role=OWNER),
        rec("p1", "袜子 袜 子", minute=1, role=OTHER),
        rec("o1", "胸", minute=2, role=OWNER),
        rec("p2", "法死你", minute=3, role=OTHER),
        rec("o2", "收到", minute=4, role=OWNER),
        rec("p3", "正常问题", minute=5, role=OTHER),
        rec("o3", "普通回答", minute=6, role=OWNER),
    ]
    results = build(records, veiled=veiled, excluded_risks=frozenset())
    assert len(results) == 2
    assert results[0].privacy_flags == ["body:胸", "body:袜子"]
    assert results[1].privacy_flags == []
    assert build(records, veiled=veiled, excluded_risks=frozenset(), context_turns=2) == []


@pytest.mark.parametrize(("turns", "excluded", "flags"), [
    (["法", "死", "你"], True, []),
    (["普通前文", "袜", "子"], False, ["body:袜子"]),
])
def test_veiled_separators_include_newlines_between_original_turns(
    veiled, turns, excluded, flags,
):
    records = [rec("c", turns[0], minute=0, role=OWNER),
               rec("p", turns[1], minute=1, role=OTHER),
               rec("o", turns[2], minute=2, role=OWNER)]
    results = build(records, veiled=veiled, context_turns=1, excluded_risks=frozenset())
    if excluded:
        assert results == []
    else:
        assert len(results) == 1
        assert results[0].privacy_flags == flags


def test_veiled_legacy_jsonl_is_byte_identical_and_flags_round_trip(tmp_path, veiled):
    records = [rec("p", "袜 子", minute=0, role=OTHER),
               rec("o", "收到", minute=1, role=OWNER)]
    legacy_payload = {
        "snippet_id": expected_id("synthetic-conv-a", "p"),
        "conversation_id": "synthetic-conv-a", "started_at": "2026-10-06T09:00:00Z",
        "partner_turns": ["袜 子"], "owner_reply": ["收到"],
        "source_fingerprint": dataset_fingerprint(records), "context_turns": [],
    }
    path = tmp_path / "pairs.jsonl"
    legacy_bytes = (json.dumps(legacy_payload, ensure_ascii=False) + "\n").encode("utf-8")
    write_snippets(path, build(records))
    assert path.read_bytes() == legacy_bytes
    assert read_snippets(path)[0].privacy_flags == []
    assert build(records, veiled=None) == read_snippets(path)
    flagged = build(records, veiled=veiled)
    write_snippets(path, flagged)
    assert json.loads(path.read_text())["privacy_flags"] == ["body:袜子"]
    assert read_snippets(path) == flagged
    assert flagged[0].to_style_snippet().model_dump() == {
        "partner_turns": ["袜 子"], "owner_reply": ["收到"],
    }


def pairs(snippets: list[PairedSnippet]) -> list[tuple[list[str], list[str]]]:
    return [(item.partner_turns, item.owner_reply) for item in snippets]


def expected_id(conversation_id: str, message_id: str) -> str:
    digest = hashlib.sha256(f"{conversation_id}\x1f{message_id}".encode()).hexdigest()
    return f"snip-{digest[:16]}"


def test_basic_pair_uses_consecutive_partner_then_owner_run() -> None:
    records = [
        rec("m1", "今天下班好晚", minute=0, role=OTHER),
        rec("m2", "累死了", minute=1, role=OTHER),
        rec("m3", "辛苦啦", minute=2, role=OWNER),
        rec("m4", "早点休息", minute=3, role=OWNER),
    ]

    snippets = build(records)

    assert pairs(snippets) == [(["今天下班好晚", "累死了"], ["辛苦啦", "早点休息"])]
    snippet = snippets[0]
    assert snippet.conversation_id == "synthetic-conv-a"
    assert snippet.started_at == BASE_TIME
    assert snippet.source_fingerprint == dataset_fingerprint(records)
    assert snippet.snippet_id == expected_id("synthetic-conv-a", "m1")


def test_owner_leading_and_partner_trailing_runs_are_not_paired() -> None:
    records = [
        rec("m1", "在吗", minute=0, role=OWNER),
        rec("m2", "在的", minute=1, role=OTHER),
        rec("m3", "晚上吃面吧", minute=2, role=OWNER),
        rec("m4", "好呀", minute=3, role=OTHER),
        rec("m5", "那我先去洗澡", minute=4, role=OTHER),
    ]

    assert pairs(build(records)) == [(["在的"], ["晚上吃面吧"])]


def test_multiple_pairs_in_one_segment() -> None:
    records = [
        rec("m1", "猫又跳上桌了", minute=0, role=OTHER),
        rec("m2", "它真会挑时候", minute=1, role=OWNER),
        rec("m3", "还打翻了水杯", minute=2, role=OTHER),
        rec("m4", "哈哈哈哈", minute=3, role=OWNER),
    ]

    assert pairs(build(records)) == [
        (["猫又跳上桌了"], ["它真会挑时候"]),
        (["还打翻了水杯"], ["哈哈哈哈"]),
    ]


def test_gap_over_thirty_minutes_breaks_pairing() -> None:
    records = [
        rec("m1", "出门了", minute=0, role=OTHER),
        rec("m2", "路上小心", minute=31, role=OWNER),
        rec("m3", "到了", minute=40, role=OTHER),
        rec("m4", "好的", minute=70, role=OWNER),
    ]

    assert pairs(build(records)) == [(["到了"], ["好的"])]


def test_partner_side_keeps_last_three_and_truncates_to_sixty_chars() -> None:
    long_text = "长" * 80
    records = [
        rec("m1", "第一句", minute=0, role=OTHER),
        rec("m2", "第二句", minute=1, role=OTHER),
        rec("m3", "第三句", minute=2, role=OTHER),
        rec("m4", "第四句", minute=3, role=OTHER),
        rec("m5", long_text, minute=4, role=OTHER),
        rec("m6", "收到", minute=5, role=OWNER),
    ]

    snippets = build(records)

    assert pairs(snippets) == [(["第三句", "第四句", "长" * 60], ["收到"])]
    assert snippets[0].started_at == BASE_TIME
    assert snippets[0].snippet_id == expected_id("synthetic-conv-a", "m1")


def test_owner_side_keeps_first_three_and_drops_overlong_pair() -> None:
    records = [
        rec("m1", "周末去哪", minute=0, role=OTHER),
        rec("m2", "去公园", minute=1, role=OWNER),
        rec("m3", "带上野餐垫", minute=2, role=OWNER),
        rec("m4", "再买点水果", minute=3, role=OWNER),
        rec("m5", "超" * 200, minute=4, role=OWNER),
        rec("m6", "晚饭呢", minute=5, role=OTHER),
        rec("m7", "好", minute=6, role=OWNER),
        rec("m8", "超" * 121, minute=7, role=OWNER),
    ]

    assert pairs(build(records)) == [(["周末去哪"], ["去公园", "带上野餐垫", "再买点水果"])]


def test_owner_length_limit_applies_after_redaction() -> None:
    name = "欧阳合成某某某"
    owner_text = name + "好" * 114
    assert len(owner_text) == 121
    records = [
        rec("m1", "谁来了", minute=0, role=OTHER),
        rec("m2", owner_text, minute=1, role=OWNER),
    ]

    snippets = build(records, redaction_terms={"name": [name]})

    assert snippets[0].owner_reply == ["[NAME]" + "好" * 114]


def test_unknown_and_non_text_records_are_skipped_and_neighbours_stay_adjacent() -> None:
    records = [
        rec("m1", "看这张图", minute=0, role=OTHER),
        rec("m2", None, minute=1, role=OTHER, content_type="image"),
        rec("m3", "谁也不知道这句是谁说的", minute=2, role=SpeakerRole.UNKNOWN),
        rec("m4", None, minute=3, role=OWNER, content_type="sticker"),
        rec("m5", "   ", minute=4, role=OWNER),
        rec("m6", "好看", minute=5, role=OWNER),
    ]

    assert pairs(build(records)) == [(["看这张图"], ["好看"])]


def test_risk_segments_are_excluded_whole() -> None:
    records = [
        rec("m1", "今天去医院复查", minute=0, role=OTHER),
        rec("m2", "结果还好吗", minute=1, role=OWNER),
        rec("m3", "晚上吃什么", minute=2, role=OTHER),
        rec("m4", "火锅", minute=3, role=OWNER),
        rec("m5", "下雨了", minute=120, role=OTHER),
        rec("m6", "带伞", minute=121, role=OWNER),
        rec("m7", "合成暗号来了", minute=300, role=OTHER),
        rec("m8", "收到暗号", minute=301, role=OWNER),
    ]

    snippets = build(
        records,
        risk_terms={RiskCategory.PRIVATE_EVENT: ["合成暗号"]},
        excluded_risks=frozenset(RiskCategory),
    )

    assert pairs(snippets) == [(["下雨了"], ["带伞"])]


def test_both_sides_are_redacted() -> None:
    name = "林合成"
    phone = "13800001234"
    records = [
        rec("m1", f"{name}今天来找你吗", minute=0, role=OTHER),
        rec("m2", f"对 {name}下午到", minute=1, role=OWNER),
        rec("m3", f"号码是{phone}", minute=120, role=OTHER),
        rec("m4", "记下了", minute=121, role=OWNER),
    ]

    snippets = build(records, redaction_terms={"name": [name]})

    assert pairs(snippets) == [(["[NAME]今天来找你吗"], ["对 [NAME]下午到"])]
    dumped = json.dumps([item.model_dump(mode="json") for item in snippets], ensure_ascii=False)
    assert name not in dumped
    assert phone not in dumped


def test_partner_text_only_appears_in_partner_turns() -> None:
    partner_texts = [f"伴侣原话{index}号" for index in range(6)]
    records: list[ChatRecord] = []
    for index, text in enumerate(partner_texts):
        records.append(rec(f"p{index}", text, minute=index * 2, role=OTHER))
        records.append(rec(f"o{index}", f"主人回应{index}", minute=index * 2 + 1, role=OWNER))

    snippets = build(records)

    assert len(snippets) == 6
    for snippet in snippets:
        for line in snippet.owner_reply:
            assert all(text not in line for text in partner_texts)
        assert all(turn in partner_texts for turn in snippet.partner_turns)


def test_snippet_ids_are_stable_and_output_is_time_ordered() -> None:
    records = [
        rec("b1", "后一段", minute=10, role=OTHER, conversation_id="synthetic-conv-b"),
        rec("b2", "嗯", minute=11, role=OWNER, conversation_id="synthetic-conv-b"),
        rec("a1", "先一段", minute=0, role=OTHER),
        rec("a2", "好", minute=1, role=OWNER),
    ]

    first = build(records)
    second = build(list(reversed(records)))

    assert [item.snippet_id for item in first] == [
        expected_id("synthetic-conv-a", "a1"),
        expected_id("synthetic-conv-b", "b1"),
    ]
    assert [item.snippet_id for item in second] == [item.snippet_id for item in first]


def test_to_style_snippet_round_trips() -> None:
    records = [
        rec("m1", "你好", minute=0, role=OTHER),
        rec("m2", "嗨", minute=1, role=OWNER),
        rec("m3", "吃了吗", minute=2, role=OWNER),
    ]

    style = build(records)[0].to_style_snippet()

    assert style == StyleSnippet(partner_turns=["你好"], owner_reply=["嗨", "吃了吗"])
    assert StyleSnippet.model_validate(style.model_dump(mode="json")) == style


def test_privacy_validation_errors_propagate() -> None:
    duplicate = [
        rec("m1", "你好", minute=0, role=OTHER),
        rec("m1", "嗨", minute=1, role=OWNER),
    ]
    with pytest.raises(PrivacyValidationError) as duplicate_error:
        build(duplicate)
    assert duplicate_error.value.code == "duplicate_message_id"

    records = [rec("m1", "你好", minute=0, role=OTHER), rec("m2", "嗨", minute=1, role=OWNER)]
    with pytest.raises(PrivacyValidationError) as terms_error:
        build(records, redaction_terms={"unknown_category": ["x"]})
    assert terms_error.value.code == "invalid_private_terms"


def snippet(index: int, partner: str | None = None, *, day: int | None = None) -> PairedSnippet:
    return PairedSnippet(
        snippet_id=f"snip-{index:016d}",
        conversation_id="synthetic-conv-a",
        started_at=BASE_TIME + timedelta(days=index if day is None else day),
        partner_turns=[partner if partner is not None else f"第{index}段伴侣话题编号{index * 7}"],
        owner_reply=[f"回复{index}"],
        source_fingerprint="0" * 64,
    )


def test_split_holdout_takes_latest_fraction_in_time_order() -> None:
    items = [snippet(index, f"话题{chr(0x4E00 + index * 37)}{index}甲乙") for index in range(11)]
    shuffled = items[:]
    random.Random(3).shuffle(shuffled)

    library, holdout = split_holdout(shuffled)

    assert library == items[:9]
    assert holdout == items[9:]


def test_split_holdout_drops_near_duplicate_holdout_candidates() -> None:
    repeated = "今天的晚饭你想吃什么呀我们一起去吃火锅好不好"
    items = [
        snippet(0, repeated),
        snippet(1, "周末要不要去爬山看看日出"),
        snippet(2, "猫咪今天又把花盆打翻了"),
        snippet(3, repeated + "呢"),
        snippet(4, "新买的耳机音质特别好"),
    ]

    library, holdout = split_holdout(items, fraction=0.4)

    assert library == items[:3]
    assert holdout == [items[4]]


def test_split_holdout_empty_input() -> None:
    assert split_holdout([]) == ([], [])


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    items = [snippet(0, "中文伴侣话"), snippet(1)]

    write_snippets(path, items)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "中文伴侣话" in lines[0]
    assert read_snippets(path) == items
    assert sorted(item.name for item in tmp_path.iterdir()) == ["pairs.jsonl"]


def test_active_snapshot_excludes_ids_and_inclusive_ranges(tmp_path: Path) -> None:
    library = tmp_path / "library.jsonl"
    exclusions = tmp_path / "deleted_ids.json"
    items = [snippet(index) for index in (6, 2, 0, 5, 1, 4, 3)]
    write_snippets(library, items)
    exclusions.write_text(json.dumps({
        "snippet_ids": [snippet(0).snippet_id, snippet(0).snippet_id, "missing"],
        "ranges": [
            [snippet(1).started_at.isoformat(), snippet(2).started_at.isoformat()],
            [snippet(2).started_at.isoformat(), snippet(3).started_at.isoformat()],
            ["2026-10-10T17:00:00+08:00", "2026-10-10T17:00:00+08:00"],
        ],
    }), encoding="utf-8")

    active, _ = paired_snippets.read_active_snippet_snapshot(library, exclusions)

    assert active == [items[0], items[3]]
    exclusions.write_text(json.dumps({
        "snippet_ids": [item.snippet_id for item in items], "ranges": [],
    }), encoding="utf-8")
    assert paired_snippets.read_active_snippet_snapshot(library, exclusions)[0] == []


def test_active_snapshot_filters_after_partial_delete(tmp_path: Path) -> None:
    library = tmp_path / "library.jsonl"
    exclusions = tmp_path / "deleted_ids.json"
    items = [snippet(0), snippet(1)]
    write_snippets(library, items)
    exclusions.write_text(json.dumps({
        "snippet_ids": [items[0].snippet_id], "ranges": [],
    }), encoding="utf-8")

    assert read_snippets(library) == items
    assert paired_snippets.read_active_snippet_snapshot(library, exclusions)[0] == items[1:]


def test_active_snapshot_revision_tracks_both_files(tmp_path: Path) -> None:
    library = tmp_path / "library.jsonl"
    exclusions = tmp_path / "deleted_ids.json"
    write_snippets(library, [snippet(0)])
    original = library.read_bytes()
    read = paired_snippets.read_active_snippet_snapshot
    assert read(library, exclusions)[1] == hashlib.sha256(original).hexdigest()

    revisions = []
    for library_bytes, exclusion_bytes in [
        (original, b'{"snippet_ids": [], "ranges": []}'),
        (original, b'{"snippet_ids": [], "ranges": []}\n'),
        (original + b"\n", b'{"snippet_ids": [], "ranges": []}\n'),
    ]:
        library.write_bytes(library_bytes)
        exclusions.write_bytes(exclusion_bytes)
        expected = hashlib.sha256(
            b"active-snippets-v1\0" + len(library_bytes).to_bytes(8, "big")
            + library_bytes + exclusion_bytes
        ).hexdigest()
        first = read(library, exclusions)
        assert first == read(library, exclusions)
        assert first == ([snippet(0)], expected)
        revisions.append(first[1])
    assert len(set(revisions)) == 3


@pytest.mark.parametrize("payload", [
    None, [], "invalid", 1,
    {}, {"snippet_ids": []}, {"ranges": []},
    {"snippet_ids": [], "ranges": [], "extra": []},
    {"snippet_ids": "id", "ranges": []},
    {"snippet_ids": [1], "ranges": []},
    {"snippet_ids": [None], "ranges": []},
    {"snippet_ids": {}, "ranges": []},
    {"snippet_ids": [], "ranges": None},
    {"snippet_ids": [], "ranges": {}},
    {"snippet_ids": [], "ranges": "invalid"},
    *[{"snippet_ids": [], "ranges": [item]} for item in [
        None, {}, "ab", [], ["2026-10-06T09:00:00Z"],
        ["2026-10-06T09:00:00Z"] * 3,
        [1, "2026-10-06T09:00:00Z"], ["2026-10-06T09:00:00Z", False],
        ["invalid", "2026-10-06T09:00:00Z"],
        ["2026-10-06T09:00:00Z", "invalid"],
        ["2026-10-06T09:00:00", "2026-10-06T10:00:00Z"],
        ["2026-10-06T09:00:00Z", "2026-10-06T10:00:00"],
        ["2026-10-06", "2026-10-07"],
        ["2026-10-07T09:00:00Z", "2026-10-06T09:00:00Z"],
    ]],
])
def test_active_snapshot_rejects_corrupt_exclusions(tmp_path: Path, payload) -> None:
    library = tmp_path / "library.jsonl"
    exclusions = tmp_path / "deleted_ids.json"
    write_snippets(library, [snippet(0)])
    exclusions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(paired_snippets.ActiveSnippetReadError,
                       match=r"^active snippet source unavailable$"):
        paired_snippets.read_active_snippet_snapshot(library, exclusions)


def test_active_snapshot_missing_exclusions_does_not_create_file(tmp_path: Path) -> None:
    library = tmp_path / "library.jsonl"
    exclusions = tmp_path / "absent" / "deleted_ids.json"
    write_snippets(library, [snippet(0)])

    assert paired_snippets.read_active_snippet_snapshot(library, exclusions) == (
        [snippet(0)], hashlib.sha256(library.read_bytes()).hexdigest(),
    )
    assert not exclusions.exists()
    assert not exclusions.parent.exists()
    assert list(tmp_path.iterdir()) == [library]


@pytest.mark.parametrize("with_exclusions", [False, True])
def test_active_snapshot_preserves_records_and_universal_newlines(
    tmp_path: Path, with_exclusions: bool,
) -> None:
    library = tmp_path / "library.jsonl"
    exclusions = tmp_path / "deleted_ids.json"
    items = [snippet(index, "长" * 80).model_copy(update={
        "context_turns": ["合成前文"], "privacy_flags": ["body:合成"],
    }) for index in (2, 0, 1)]
    library.write_bytes((" \r\n" + "\r \r\n".join(
        item.model_dump_json() for item in items
    ) + "\r\n\t").encode("utf-8"))
    if with_exclusions:
        exclusions.write_text('{"snippet_ids": [], "ranges": []}', encoding="utf-8")
    assert paired_snippets.read_active_snippet_snapshot(library, exclusions)[0] == items
    library.write_bytes(b" \r\n\t\r")
    assert paired_snippets.read_active_snippet_snapshot(library, exclusions)[0] == []
    library.write_bytes(b"")
    assert paired_snippets.read_active_snippet_snapshot(library, exclusions)[0] == []


@pytest.mark.parametrize("source", ["library", "exclusions"])
@pytest.mark.parametrize("failure", [
    "missing", "permission", "directory", "os_error", "runtime_error",
    "utf8", "json", "validation",
])
def test_active_snapshot_errors_hide_content_and_paths(
    tmp_path: Path, monkeypatch, capsys, caplog, source: str, failure: str,
) -> None:
    canary = "SYNTHETIC_ACTIVE_SNAPSHOT_CANARY"
    library = tmp_path / "library-path-canary.jsonl"
    exclusions = tmp_path / "exclusions-path-canary.json"
    target = library if source == "library" else exclusions
    write_snippets(library, [snippet(0, canary)])
    exclusions.write_text('{"snippet_ids": [], "ranges": []}', encoding="utf-8")
    if failure in {"utf8", "json", "validation"}:
        payload = {
            "utf8": canary.encode() + b"\xff",
            "json": ('{"' + canary).encode(),
            "validation": json.dumps({"snippet_ids": canary, "ranges": []}).encode(),
        }[failure]
        target.write_bytes(payload)

    real_read = Path.read_bytes
    reads = []
    failures = {
        "missing": FileNotFoundError, "permission": PermissionError,
        "directory": IsADirectoryError, "os_error": OSError, "runtime_error": RuntimeError,
    }

    def read_once(path):
        # Both real file reads are confined to the explicitly supplied synthetic paths.
        assert path in {library, exclusions}
        reads.append(path)
        if path == target and failure in failures:
            raise failures[failure](f"{canary}: {target}")
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    if source == "exclusions" and failure == "missing":
        assert paired_snippets.read_active_snippet_snapshot(library, exclusions)[0] == [
            snippet(0, canary),
        ]
    else:
        with pytest.raises(paired_snippets.ActiveSnippetReadError) as caught:
            paired_snippets.read_active_snippet_snapshot(library, exclusions)
        error = caught.value
        assert isinstance(error, RuntimeError)
        assert str(error) == "active snippet source unavailable"
        assert error.__cause__ is None
        assert error.__context__ is None
        formatted = "".join(traceback.format_exception(error))
        for secret in (canary, str(library), str(exclusions), library.name, exclusions.name):
            assert secret not in formatted

    assert reads == [library, exclusions]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert caplog.records == []


def test_write_snippets_cleans_temporary_on_serialization_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_bytes(b"original\n")
    original = path.read_bytes()
    dumps = paired_snippets.json.dumps
    calls = 0

    def fail_second_dump(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("serialization failed")
        return dumps(*args, **kwargs)

    monkeypatch.setattr(paired_snippets.json, "dumps", fail_second_dump)

    with pytest.raises(RuntimeError, match="serialization failed"):
        write_snippets(path, [snippet(0), snippet(1)])

    assert not list(tmp_path.glob(f".{path.name}.*"))
    assert path.read_bytes() == original


def test_write_snippets_cleans_temporary_on_replace_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_bytes(b"original\n")
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(paired_snippets.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_snippets(path, [snippet(0)])

    assert not list(tmp_path.glob(f".{path.name}.*"))
    assert path.read_bytes() == original


def test_write_snippets_empty_list_writes_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"

    write_snippets(path, [])

    assert path.read_bytes() == b""
    assert sorted(item.name for item in tmp_path.iterdir()) == ["pairs.jsonl"]


def test_delete_by_id_cascades_across_files(tmp_path: Path) -> None:
    library_path = tmp_path / "library.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    items = [snippet(index) for index in range(6)]
    write_snippets(library_path, items[:4])
    write_snippets(holdout_path, items[4:])

    deleted = delete_snippets(
        [library_path, holdout_path],
        snippet_ids=frozenset({items[1].snippet_id, items[5].snippet_id, "snip-missing"}),
    )

    assert deleted == 2
    assert read_snippets(library_path) == [items[0], items[2], items[3]]
    assert read_snippets(holdout_path) == [items[4]]


def test_delete_by_closed_time_range_cascades_across_files(tmp_path: Path) -> None:
    library_path = tmp_path / "library.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    items = [snippet(index) for index in range(6)]
    write_snippets(library_path, items[:4])
    write_snippets(holdout_path, items[4:])

    deleted = delete_snippets(
        [library_path, holdout_path],
        started_between=(items[3].started_at, items[4].started_at),
    )

    assert deleted == 2
    assert read_snippets(library_path) == items[:3]
    assert read_snippets(holdout_path) == [items[5]]


def test_sample_for_audit_is_reproducible() -> None:
    items = [snippet(index) for index in range(60)]

    first = sample_for_audit(items, seed=7)

    assert len(first) == 50
    assert first == sample_for_audit(items, seed=7)
    assert first == random.Random(7).sample(items, 50)


def test_sample_for_audit_returns_all_when_fewer_than_size() -> None:
    items = [snippet(index) for index in range(12)]

    assert sample_for_audit(items, seed=7) == items


@pytest.mark.parametrize(
    ("category", "kept"),
    [
        (RiskCategory.INTIMATE, True),
        (RiskCategory.CONFLICT, True),
        (RiskCategory.HEALTH, True),
        (RiskCategory.FAMILY, True),
        (RiskCategory.SEXUAL, False),
        (RiskCategory.CREDENTIAL, False),
        (RiskCategory.FINANCIAL, False),
        (RiskCategory.LOCATION, False),
        (RiskCategory.PRIVATE_EVENT, False),
    ],
)
def test_default_excluded_risk_categories(category: RiskCategory, kept: bool) -> None:
    records = [
        rec("m1", "合成标记", minute=0, role=OTHER),
        rec("m2", "收到", minute=1, role=OWNER),
    ]
    terms = {category: ["合成标记"]}
    assert bool(build(records, risk_terms=terms)) is kept
    assert build(records, risk_terms=terms, excluded_risks=frozenset(RiskCategory)) == []


def test_exclusion_uses_all_risk_windows_of_source_segment() -> None:
    records = [
        rec(f"m{i}", "普通话题", minute=i, role=OTHER if i % 2 == 0 else OWNER)
        for i in range(20)
    ]
    records[0].text = "想你"
    records[-1].text = "合成排除标记"
    assert build(records, risk_terms={RiskCategory.PRIVATE_EVENT: ["合成排除标记"]}) == []
    assert len(build(records, excluded_risks=frozenset())) == 10


def test_context_limit_order_labels_redaction_and_truncation() -> None:
    records = [
        rec("m0", "更早", minute=0, role=OWNER),
        rec("m1", "林合成" + "长" * 80, minute=1, role=OTHER),
        rec("m2", "收到林合成", minute=2, role=OWNER),
        rec("m3", "这次的问题", minute=3, role=OTHER),
        rec("m4", "这次的回答", minute=4, role=OWNER),
    ]
    result = build(records, context_turns=2, redaction_terms={"name": ["林合成"]})[-1]
    assert result.context_turns == ["她：" + ("[NAME]" + "长" * 80)[:60], "他：收到[NAME]"]
    assert result.to_style_snippet().model_dump() == {
        "partner_turns": ["这次的问题"], "owner_reply": ["这次的回答"]
    }
    assert build(records, context_turns=10)[-1].context_turns == [
        "他：更早", "她：" + ("林合成" + "长" * 80)[:60], "他：收到林合成"
    ]


@pytest.mark.parametrize("other_conversation", [False, True])
def test_context_never_crosses_segments(other_conversation: bool) -> None:
    records = [
        rec("m1", "上一段", minute=0, role=OWNER),
        rec("m2", "这段问题", minute=31, role=OTHER,
            conversation_id="synthetic-conv-b" if other_conversation else "synthetic-conv-a"),
        rec("m3", "这段回答", minute=32, role=OWNER,
            conversation_id="synthetic-conv-b" if other_conversation else "synthetic-conv-a"),
    ]
    assert build(records, context_turns=10)[0].context_turns == []


@pytest.mark.parametrize("position", [0, 2, 4, 6, 7, 8])
@pytest.mark.parametrize("bad_content", [("image", None), ("text", None), ("text", "   ")])
def test_nontext_exclusion_uses_unfiltered_window(
    position: int, bad_content: tuple[str, str | None]
) -> None:
    # Context media truncates; only media from m2 through m7 excludes the pair,
    # including the fourth owner turn that is omitted from owner_reply.
    records = [
        rec("m0", "窗口外前文", minute=1, role=OWNER),
        rec("m1", "窗口内前文", minute=3, role=OWNER),
        rec("m2", "问题一", minute=5, role=OTHER),
        rec("m3", "问题二", minute=7, role=OTHER),
        rec("m4", "回答一", minute=9, role=OWNER),
        rec("m5", "回答二", minute=11, role=OWNER),
        rec("m6", "回答三", minute=13, role=OWNER),
        rec("m7", "回答四", minute=15, role=OWNER),
    ]
    content_type, text = bad_content
    records.insert(position, rec("bad", text, minute=position * 2, role=OWNER,
                                 content_type=content_type))
    result = build(records, context_turns=1, exclude_near_nontext=True)
    assert bool(result) is (position in {0, 2, 8})
    if position == 2:
        assert result[0].context_turns == []
    assert len(build(records)) == 1


def test_nontext_exclusion_checks_unknown_media_in_pair_but_keeps_prior_media() -> None:
    records = [
        rec("before", None, minute=0, role=OWNER, content_type="image"),
        rec("p", "问题", minute=1, role=OTHER),
        rec("unknown", None, minute=2, role=SpeakerRole.UNKNOWN, content_type="image"),
        rec("o", "回答", minute=3, role=OWNER),
    ]
    assert build(records, exclude_near_nontext=True) == []
    records.pop(2)
    assert len(build(records, exclude_near_nontext=True)) == 1
    records.insert(3, rec("bad", None, minute=2, role=OTHER, content_type="image"))
    assert build(records, exclude_near_nontext=True) == []


@pytest.mark.parametrize("full_text", [False, True])
@pytest.mark.parametrize("position", [1, 2, 4, 5, 7])
@pytest.mark.parametrize("bad_content", [("image", "合成媒体"), ("text", None),
                                         ("text", " \n\t")])
def test_d6_media_checks_complete_raw_pair(full_text, position, bad_content):
    records = [rec(f"p{i}", f"问题{i}", minute=i * 2, role=OTHER) for i in range(4)]
    records += [rec(f"o{i}", f"回答{i}", minute=(i + 4) * 2, role=OWNER) for i in range(4)]
    content_type, text = bad_content
    records.append(rec("bad", text, minute=position * 2 - 1,
                       role=SpeakerRole.UNKNOWN, content_type=content_type))
    assert build(list(reversed(records)), exclude_near_nontext=True, full_text=full_text) == []
    assert len(build(records, full_text=full_text)) == 1


@pytest.mark.parametrize("bad_content", [("image", None), ("text", None), ("text", " ")])
def test_d6_context_truncates_at_last_media_without_refilling(bad_content):
    content_type, text = bad_content
    records = [
        rec("c0", "不补回", minute=0, role=OWNER),
        rec("c1", "旧前文", minute=1, role=OWNER),
        rec("bad1", text, minute=2, role=OWNER, content_type=content_type),
        rec("c2", "中间前文", minute=3, role=OWNER),
        rec("bad2", text, minute=4, role=SpeakerRole.UNKNOWN, content_type=content_type),
        rec("c3", "保留前文", minute=5, role=OWNER),
        rec("p", "问题", minute=6, role=OTHER),
        rec("o", "回答", minute=7, role=OWNER),
    ]
    counts = {}
    result = build(records, context_turns=3, exclude_near_nontext=True, media_counts=counts)
    assert result[0].context_turns == ["他：保留前文"]
    assert counts == {"excluded_by_media": 0, "excluded_by_deictic": 0,
                      "excluded_by_short_reaction": 0, "context_truncated_at_media": 1}
    assert build(records, context_turns=3)[0].context_turns == [
        "他：旧前文", "他：中间前文", "他：保留前文",
    ]


@pytest.mark.parametrize("context_turns", [0, 1, 5])
def test_d6_left_neighbour_media_marks_context_boundary(context_turns):
    records = [rec("bad", None, minute=0, role=OWNER, content_type="image")]
    if context_turns:
        records.append(rec("c", "普通前文", minute=1, role=OWNER))
    records += [rec("p", "问题", minute=2, role=OTHER), rec("o", "回答", minute=3, role=OWNER)]
    counts = {}
    result = build(records, context_turns=context_turns, exclude_near_nontext=True,
                   media_counts=counts)
    assert len(result) == 1
    assert result[0].context_turns == (["他：普通前文"] if context_turns else [])
    assert counts["context_truncated_at_media"] == 1


def test_d6_short_reaction_uses_first_turn_after_actual_context_cut():
    records = [rec("c1", "旧前文", minute=0, role=OWNER),
               rec("bad", None, minute=1, role=OWNER, content_type="image"),
               rec("c2", "你看看？", minute=2, role=OWNER),
               rec("p", "普通问题", minute=3, role=OTHER),
               rec("o", "普通回答", minute=4, role=OWNER)]
    counts = {}
    assert build(records, context_turns=2, exclude_near_nontext=True, media_counts=counts) == []
    assert counts["excluded_by_short_reaction"] == counts["context_truncated_at_media"] == 1
    records[2].text = "普通前文"
    records[3].text = "你看看？"
    result = build(records, context_turns=2, exclude_near_nontext=True)
    assert len(result) == 1
    assert result[0].context_turns == ["他：普通前文"]


def test_d6_left_neighbour_does_not_cross_conversations():
    records = [rec("bad", None, minute=0, role=OWNER, content_type="image"),
               rec("p", "看看", minute=1, role=OTHER, conversation_id="synthetic-conv-b"),
               rec("o", "回答", minute=2, role=OWNER, conversation_id="synthetic-conv-b")]
    counts = {}
    result = build(records, context_turns=10, exclude_near_nontext=True, media_counts=counts)
    assert len(result) == 1
    assert counts == {"excluded_by_media": 0, "excluded_by_deictic": 0,
                      "excluded_by_short_reaction": 0, "context_truncated_at_media": 0}


@pytest.mark.parametrize("deictic", ["这张", "那张", "哪张", "图里", "图中", "图片里",
                                    "照片里", "截图里", "视频里", "语音里", "上图", "下图"])
@pytest.mark.parametrize("position", ["context", "partner", "owner"])
def test_d6_deictic_scans_every_retained_side_before_redaction(deictic, position):
    records = [rec("c", "普通前文", minute=0, role=OWNER),
               rec("p", "普通问题", minute=1, role=OTHER),
               rec("o", "普通回答", minute=2, role=OWNER)]
    records[{"context": 0, "partner": 1, "owner": 2}[position]].text = deictic + "内容"
    counts = {}
    options = {"context_turns": 1, "redaction_terms": {"name": [deictic]}}
    assert build(records, exclude_near_nontext=True, media_counts=counts, **options) == []
    assert counts == {"excluded_by_media": 0, "excluded_by_deictic": 1,
                      "excluded_by_short_reaction": 0, "context_truncated_at_media": 0}
    assert len(build(records, **options)) == 1


@pytest.mark.parametrize("full_text", [False, True])
@pytest.mark.parametrize("position", ["hidden_partner", "hidden_owner", "long_partner",
                                      "long_context"])
def test_d6_deictic_scans_complete_text_before_display_limits(full_text, position):
    records = [rec("c", "前文", minute=0, role=OWNER)]
    records += [rec(f"p{i}", f"问题{i}", minute=i + 1, role=OTHER) for i in range(4)]
    records += [rec(f"o{i}", f"回答{i}", minute=i + 5, role=OWNER) for i in range(4)]
    index = {"hidden_partner": 1, "hidden_owner": 8, "long_partner": 4, "long_context": 0}[
        position
    ]
    records[index].text = "长" * 80 + "这张"
    assert build(records, context_turns=1, full_text=full_text, exclude_near_nontext=True) == []


def test_d6_deictic_does_not_scan_context_removed_by_media():
    records = [rec("c", "这张内容", minute=0, role=OWNER),
               rec("bad", None, minute=1, role=OWNER, content_type="image"),
               rec("p", "普通问题", minute=2, role=OTHER),
               rec("o", "普通回答", minute=3, role=OWNER)]
    result = build(records, context_turns=1, exclude_near_nontext=True)
    assert len(result) == 1
    assert result[0].context_turns == []


@pytest.mark.parametrize("reaction", ["这个呢", "那个呢", "你看看", "看看", "好看吗",
                                     "怎么样", "咋样", "好看", "不好看", "可以", "不错",
                                     "嗯", "对", "是的"])
@pytest.mark.parametrize("with_context", [False, True])
def test_d6_short_reaction_requires_media_boundary(reaction, with_context):
    records = [rec("bad", None, minute=0, role=OWNER, content_type="image")]
    text = " \n" + " \t".join(reaction) + "，。！？!?～~… \n"
    if with_context:
        records.append(rec("c", text, minute=1, role=OWNER))
    records += [rec("p", "问题" if with_context else text, minute=2, role=OTHER),
                rec("o", "回答", minute=3, role=OWNER)]
    counts = {}
    options = {"context_turns": 1 if with_context else 0, "exclude_near_nontext": True}
    assert build(records, media_counts=counts, **options) == []
    assert counts == {"excluded_by_media": 0, "excluded_by_deictic": 0,
                      "excluded_by_short_reaction": 1, "context_truncated_at_media": 1}
    assert len(build(records[1:], **options)) == 1
    assert len(build(records, context_turns=options["context_turns"])) == 1


@pytest.mark.parametrize("text", ["这个", "看看这个安排", "这样", "可以。明天再说",
                                 "看看，明天安排", "好看吗？我来改文字"])
def test_d6_ordinary_demonstratives_and_longer_reactions_are_kept(text):
    records = [rec("bad", None, minute=0, role=OWNER, content_type="image"),
               rec("p", text, minute=1, role=OTHER), rec("o", "回答", minute=2, role=OWNER)]
    assert len(build(records, exclude_near_nontext=True)) == 1


@pytest.mark.parametrize("bait,excluded,flag", [("法*死*你", True, None),
                                             ("袜 子", False, "body:袜子")])
def test_d6_veiled_still_scans_context_removed_by_media(veiled, bait, excluded, flag):
    records = [rec("c", bait, minute=0, role=OWNER),
               rec("bad", None, minute=1, role=OWNER, content_type="image"),
               rec("p", "问题", minute=2, role=OTHER), rec("o", "回答", minute=3, role=OWNER)]
    options = {"context_turns": 1, "exclude_near_nontext": True, "excluded_risks": frozenset()}
    assert len(build(records, **options)) == 1
    result = build(records, veiled=veiled, **options)
    if excluded:
        assert result == []
    else:
        assert result[0].context_turns == []
        assert result[0].privacy_flags == [flag]


def test_d6_risk_still_scans_context_removed_by_media():
    records = [rec("c", "合成排除标记", minute=0, role=OWNER),
               rec("bad", None, minute=1, role=OWNER, content_type="image"),
               rec("p", "问题", minute=2, role=OTHER), rec("o", "回答", minute=3, role=OWNER)]
    options = {"context_turns": 1, "exclude_near_nontext": True}
    assert len(build(records, **options)) == 1
    assert build(records, risk_terms={RiskCategory.PRIVATE_EVENT: ["合成排除标记"]},
                 **options) == []


def test_pseudonyms_keep_list_order_longest_match_and_all_redaction_rules() -> None:
    terms = {
        "name": ["合成人", "合成人名", "小甲"],
        "address": ["崇德", "崇德楼"],
        "account": ["synthetic_account"],
        "private_event": ["合成事件"],
    }
    text = "合成人名 合成人 小甲 崇德楼 崇德 synthetic_account 合成事件 13800001234"
    expected = "小乙 小甲 小丙 乙地 甲地 [ACCOUNT] [PRIVATE_EVENT] [PHONE]"
    records = [
        rec("m1", text, minute=0, role=OTHER),
        rec("m2", text, minute=1, role=OWNER),
        rec("m3", "再问合成人名", minute=2, role=OTHER),
        rec("m4", "回答崇德楼", minute=3, role=OWNER),
    ]
    result = build(records, context_turns=2, pseudonymize=True,
                   redaction_terms=terms, excluded_risks=frozenset())
    assert pairs(result) == [([expected], [expected]), (["再问小乙"], ["回答乙地"])]
    assert result[1].context_turns == ["她：" + expected, "他：" + expected]


def test_pseudonyms_cycle_after_twenty_two_terms() -> None:
    names = [f"合成人物{i:02d}" for i in range(45)]
    addresses = [f"合成地点{i:02d}" for i in range(45)]
    records = [
        rec("p", " ".join(names[i] for i in (0, 21, 22, 23, 44)), minute=0, role=OTHER),
        rec("o", " ".join(addresses[i] for i in (0, 21, 22, 23, 44)), minute=1, role=OWNER),
    ]
    assert pairs(build(records, pseudonymize=True,
                       redaction_terms={"name": names, "address": addresses})) == [
        (["小甲 小亥 小甲2 小乙2 小甲3"], ["甲地 亥地 甲地2 乙地2 甲地3"])
    ]


def test_default_options_preserve_legacy_text_and_metadata() -> None:
    records = [
        rec("m1", "林合成问好", minute=0, role=OTHER),
        rec("bad", None, minute=1, role=OWNER, content_type="image"),
        rec("m2", "回复林合成", minute=2, role=OWNER),
    ]
    result = build(records, redaction_terms={"name": ["林合成"]})[0]
    assert result.context_turns == []
    assert result.model_dump(mode="json", exclude={"context_turns"}) == {
        "snippet_id": expected_id("synthetic-conv-a", "m1"),
        "conversation_id": "synthetic-conv-a",
        "started_at": "2026-10-06T09:00:00Z",
        "partner_turns": ["[NAME]问好"],
        "owner_reply": ["回复[NAME]"],
        "source_fingerprint": dataset_fingerprint(records),
    }
    assert build(records, redaction_terms={"name": ["林合成"]}, context_turns=0,
                 exclude_near_nontext=False, pseudonymize=False) == [result]


def test_legacy_jsonl_defaults_to_independent_empty_context(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text("\n".join(
        json.dumps(snippet(i).model_dump(mode="json", exclude={"context_turns"}),
                   ensure_ascii=False)
        for i in range(2)
    ), encoding="utf-8")
    items = read_snippets(path)
    assert items[0].context_turns == items[1].context_turns == []
    items[0].context_turns.append("他：前文")
    assert items[1].context_turns == []
    write_snippets(path, items)
    assert read_snippets(path) == items


@pytest.mark.parametrize("terms", [{"name": "bad"}, {"address": [""]}, {"bad": ["x"]}])
def test_pseudonymization_validates_private_terms(terms: dict) -> None:
    records = [rec("p", "问题", minute=0, role=OTHER),
               rec("o", "回答", minute=1, role=OWNER)]
    with pytest.raises(PrivacyValidationError) as error:
        build(records, pseudonymize=True, redaction_terms=terms)
    assert error.value.code == "invalid_private_terms"


def entity(entity_id, aliases, pseudonym=None, kind="person"):
    from companion.privacy import PseudonymEntity
    return PseudonymEntity(id=entity_id, kind=kind, aliases=aliases, pseudonym=pseudonym)


def test_entities_preserve_owner_partner_and_group_third_parties_and_places() -> None:
    entities = [
        entity("owner", ["合成主人", "主昵称"]),
        entity("partner", ["合成伴侣", "伴昵称"]),
        entity("third", ["合成全名", "合成简称"], "小丙"),
        entity("place", ["合成校园", "合成校区"], "甲地", "place"),
    ]
    text = "合成主人 主昵称 合成伴侣 伴昵称 合成全名 合成简称 合成校园 合成校区 城市"
    expected = "合成主人 主昵称 合成伴侣 伴昵称 小丙 小丙 甲地 甲地 城市"
    records = [rec("p", text, minute=0, role=OTHER),
               rec("o", text, minute=1, role=OWNER),
               rec("p2", "合成全名", minute=2, role=OTHER),
               rec("o2", "合成校区", minute=3, role=OWNER)]
    result = build(records, entities=entities, context_turns=2, full_text=True)
    assert pairs(result) == [([expected], [expected]), (["小丙"], ["甲地"])]
    assert result[1].context_turns == ["她：" + expected, "他：" + expected]


def test_entities_longest_match_once_and_retained_aliases_are_protected() -> None:
    entities = [entity("short", ["合成"], "小甲"),
                entity("long", ["合成全名"], "小乙"),
                entity("replacement", ["小乙"], "小丙"),
                entity("owner", ["合成主人", "13800001234"])]
    text = "合成全名 合成 合成主人 小乙 13800001234 13900001234"
    expected = "小乙 小甲 合成主人 小丙 13800001234 [PHONE]"
    records = [rec("p", text, minute=0, role=OTHER),
               rec("o", text, minute=1, role=OWNER)]
    assert pairs(build(records, entities=entities, pseudonymize=True,
                       redaction_terms={"account": ["合成主人"]},
                       excluded_risks=frozenset())) == [([expected], [expected])]


@pytest.mark.parametrize("pseudonymize", [False, True])
def test_entities_redact_uncovered_name_address_and_keep_other_redaction(pseudonymize) -> None:
    text = "合成主人 未列姓名 未列地点 合成账号 合成事件 13800001234"
    records = [rec("p", text, minute=0, role=OTHER),
               rec("o", text, minute=1, role=OWNER)]
    terms = {"name": ["合成主人", "未列姓名"], "address": ["未列地点"],
             "account": ["合成账号"], "private_event": ["合成事件"]}
    for entities in ([], [entity("owner", ["合成主人"])]):
        expected = ("合成主人" if entities else "[NAME]") + (
            " [NAME] [ADDRESS] [ACCOUNT] [PRIVATE_EVENT] [PHONE]"
        )
        assert pairs(build(records, entities=entities, redaction_terms=terms,
                           pseudonymize=pseudonymize, excluded_risks=frozenset())) == [
            ([expected], [expected])
        ]


def test_entities_cover_alias_terms_before_fallback_redaction_on_all_turns() -> None:
    entities = [entity("owner", ["未列姓名主人"]),
                entity("partner", ["合成伴侣"]),
                entity("third", ["合成全名", "小丙"], "小丙"),
                entity("place", ["合成校园", "甲地"], "甲地", "place")]
    terms = {"name": ["未列姓名", "合成伴侣", "合成全名", "小丙"],
             "address": ["未列地点", "合成校园", "甲地"]}
    text = "未列姓名主人 合成伴侣 合成全名 小丙 合成校园 甲地 未列姓名 未列地点"
    expected = "未列姓名主人 合成伴侣 小丙 小丙 甲地 甲地 [NAME] [ADDRESS]"
    records = [rec("p", text, minute=0, role=OTHER),
               rec("o", text, minute=1, role=OWNER),
               rec("p2", text, minute=2, role=OTHER),
               rec("o2", text, minute=3, role=OWNER)]
    result = build(records, entities=entities, redaction_terms=terms,
                   full_text=True, context_turns=2)
    assert pairs(result) == [([expected], [expected])] * 2
    assert result[1].context_turns == ["她：" + expected, "他：" + expected]


def test_load_entities_validates_version_and_cross_entity_duplicates(tmp_path) -> None:
    from companion.privacy import load_entities
    path = tmp_path / "entities.json"
    valid = {"version": 1, "entities": [
        {"id": "owner", "kind": "person", "aliases": ["合成主人"], "pseudonym": None},
        {"id": "place", "kind": "place", "aliases": ["合成地点"], "pseudonym": "甲地"},
    ]}
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert load_entities(path) == [entity("owner", ["合成主人"]),
                                   entity("place", ["合成地点"], "甲地", "place")]
    for version in (0, 2, "1"):
        path.write_text(json.dumps({**valid, "version": version}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_entities(path)
    valid["entities"][1]["aliases"] = ["合成主人"]
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError):
        load_entities(path)
    with pytest.raises(ValueError):
        build([], entities=[entity("a", ["重复"]), entity("b", ["重复"], "小甲")])


@pytest.mark.parametrize("aliases", [[], [""], ["   "], ["字" * 41], [123]])
def test_entities_reject_invalid_aliases(aliases) -> None:
    with pytest.raises(ValueError):
        entity("invalid", aliases)


def test_entities_allow_alias_length_boundaries_and_literal_regex() -> None:
    entities = [entity("a", ["字", "名" * 40, "a.b+"], "小甲")]
    records = [rec("p", "字 " + "名" * 40 + " a.b+ axb", minute=0, role=OTHER),
               rec("o", "收到", minute=1, role=OWNER)]
    assert pairs(build(records, entities=entities)) == [(["小甲 小甲 小甲 axb"], ["收到"])]


def test_full_text_preserves_entire_context_and_partner_run_and_roundtrips(tmp_path) -> None:
    texts = ["前" * 81, "答" * 120, *[f"连发{i}" + "长" * 80 for i in range(5)]]
    records = [rec("p0", texts[0], minute=0, role=OTHER),
               rec("o0", texts[1], minute=1, role=OWNER)]
    records += [rec(f"p{i + 1}", text, minute=i + 2, role=OTHER)
                for i, text in enumerate(texts[2:])]
    records += [rec(f"o{i + 1}", "回复" if i < 3 else "超" * 121,
                    minute=i + 7, role=OWNER) for i in range(4)]
    result = build(records, full_text=True, context_turns=10)
    assert result[-1].context_turns == ["她：" + texts[0], "他：" + texts[1]]
    assert pairs(result)[-1] == (texts[2:], ["回复"] * 3)
    assert result[-1].as_style_snippet() is None
    assert pairs(build(records))[-1] == ([text[:60] for text in texts[-3:]], ["回复"] * 3)
    path = tmp_path / "full.jsonl"
    write_snippets(path, result)
    assert read_snippets(path) == result
    records[-3].text = "超" * 121
    assert len(build(records, full_text=True)) == 1


@pytest.mark.parametrize(("turns", "eligible"), [
    (["字" * 60] * 3, True), (["字" * 61], False), (["字"] * 4, False),
    ([], False), ([""], False),
])
def test_as_style_snippet_boundaries(turns, eligible) -> None:
    item = snippet(0).model_copy(update={"partner_turns": turns, "owner_reply": ["答" * 120]})
    if eligible:
        assert item.as_style_snippet() == item.to_style_snippet()
        assert item.as_style_snippet() == StyleSnippet(
            partner_turns=turns, owner_reply=["答" * 120]
        )
    else:
        assert item.as_style_snippet() is None
        with pytest.raises(ValueError):
            item.to_style_snippet()


@pytest.mark.parametrize("enabled", [False, True])
def test_blocked_context_truncates_without_refilling(enabled):
    records = [rec("old", "旧前文", minute=0, role=OWNER),
               rec("blocked", "删除内容", minute=1, role=OWNER),
               rec("kept", "保留前文", minute=2, role=OWNER),
               rec("p", "普通问题", minute=3, role=OTHER),
               rec("o", "普通回答", minute=4, role=OWNER)]
    result = build(records, context_turns=3, exclude_near_nontext=enabled,
                   blocked_message_ids=frozenset({"blocked"}))
    assert len(result) == 1
    assert result[0].context_turns == ["他：保留前文"]
    assert "删除内容" not in result[0].model_dump_json()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("blocked", ["p0", "p3", "o0", "o3", "unknown"])
def test_blocked_anywhere_in_full_pair_excludes(enabled, blocked):
    records = [rec(f"p{i}", "普通问题", minute=i, role=OTHER) for i in range(4)]
    records += [rec("unknown", "隐藏记录", minute=4, role=SpeakerRole.UNKNOWN)]
    records += [rec(f"o{i}", "普通回答", minute=i + 5, role=OWNER) for i in range(4)]
    assert build(records, exclude_near_nontext=enabled,
                 blocked_message_ids=frozenset({blocked})) == []


def test_blocked_left_neighbour_is_short_reaction_boundary():
    records = [rec("blocked", "删除内容", minute=0, role=SpeakerRole.UNKNOWN),
               rec("p", "你看看？", minute=1, role=OTHER),
               rec("o", "普通回答", minute=2, role=OWNER)]
    assert build(records, exclude_near_nontext=True,
                 blocked_message_ids=frozenset({"blocked"})) == []
    assert len(build(records, blocked_message_ids=frozenset({"blocked"}))) == 1


def test_blocked_context_cannot_return_as_veiled_flags(veiled):
    records = [rec("blocked", "袜子", minute=0, role=OWNER),
               rec("p", "普通问题", minute=1, role=OTHER),
               rec("o", "普通回答", minute=2, role=OWNER)]
    result = build(records, context_turns=1, veiled=veiled,
                   blocked_message_ids=frozenset({"blocked"}))
    assert result[0].context_turns == result[0].privacy_flags == []


def test_coverage_includes_full_runs_and_all_raw_interval_records():
    records = [rec("context", "前文", minute=0, role=OWNER)]
    records += [rec(f"p{i}", "普通问题", minute=i + 1, role=OTHER) for i in range(4)]
    records += [rec("media", None, minute=5, role=OTHER, content_type="image"),
                rec("unknown", "未知角色", minute=6, role=SpeakerRole.UNKNOWN)]
    records += [rec(f"o{i}", "普通回答", minute=i + 7, role=OWNER) for i in range(4)]
    records += [rec("following", "下个问题", minute=11, role=OTHER)]
    coverage = {}
    result = build(list(reversed(records)), context_turns=1, covered_message_ids=coverage)
    assert len(result) == 1
    assert len(result[0].partner_turns) == len(result[0].owner_reply) == 3
    assert coverage == {result[0].snippet_id: frozenset(
        [f"p{i}" for i in range(4)] + ["media", "unknown"] + [f"o{i}" for i in range(4)]
    )}
