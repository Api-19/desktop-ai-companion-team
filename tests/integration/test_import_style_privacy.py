"""Synthetic Importer-to-Privacy-to-Style owner-only system boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from companion.contracts import SpeakerRole
from companion.importer import (
    ConfirmedSpeakerMapping,
    ImportValidationError,
    StandardJsonAdapter,
    apply_confirmed_mapping,
    preview_speaker_mapping,
)
from companion.privacy import select_safe_candidates
from companion.style import build_style_pack, extract_style_features

PROCESSED_AT = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
OWNER = "synthetic-owner"
OTHER = "synthetic-contact"
OWNER_TEXTS = ("我准备吃饭啦", "今天好开心呀")
OTHER_TEXT = "我要睡觉啦"


def _source(tmp_path: Path, *, other_text_as_owner_control: bool = False) -> Path:
    rows = (
        ((OWNER, "outbound", OTHER_TEXT),)
        if other_text_as_owner_control
        else (
            (OWNER, "outbound", OWNER_TEXTS[0]),
            (OTHER, "inbound", OTHER_TEXT),
            (OWNER, "outbound", OWNER_TEXTS[1]),
        )
    )
    messages = [
        {
            "schema_version": 1,
            "conversation_id": "synthetic-style-conversation",
            "message_id": f"synthetic-message-{index}",
            "timestamp": f"2026-09-27T{10 + index:02d}:00:00+00:00",
            "source_speaker_id": speaker,
            "source_speaker_name": None,
            "direction": direction,
            "content_type": "text",
            "text": text,
        }
        for index, (speaker, direction, text) in enumerate(rows)
    ]
    filename = (
        "synthetic-control.json" if other_text_as_owner_control else "synthetic-standard-chat.json"
    )
    path = tmp_path / filename
    path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    return path


def _select(records):
    return select_safe_candidates(
        records,
        processed_at=PROCESSED_AT,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=("小团子",),
    )


def test_safe_other_text_stays_out_of_candidates_and_style_examples(tmp_path: Path) -> None:
    batch = StandardJsonAdapter().parse(_source(tmp_path))
    assert batch.issues == []
    preview = preview_speaker_mapping(batch.records)
    by_speaker = {item.source_speaker_id: item for item in preview.speakers}
    assert set(by_speaker) == {OWNER, OTHER}
    assert by_speaker[OWNER].outbound_count == 2
    assert by_speaker[OTHER].inbound_count == 1

    mapped = apply_confirmed_mapping(
        batch.records,
        ConfirmedSpeakerMapping(roles={OWNER: SpeakerRole.OWNER, OTHER: SpeakerRole.OTHER}),
    )
    assert [record.speaker_role for record in mapped] == [
        SpeakerRole.OWNER,
        SpeakerRole.OTHER,
        SpeakerRole.OWNER,
    ]

    control_batch = StandardJsonAdapter().parse(_source(tmp_path, other_text_as_owner_control=True))
    control_preview = preview_speaker_mapping(control_batch.records)
    assert len(control_preview.speakers) == 1
    control_records = apply_confirmed_mapping(
        control_batch.records,
        ConfirmedSpeakerMapping(roles={OWNER: SpeakerRole.OWNER}),
    )
    classification_control = _select(control_records)
    assert [candidate.text for candidate in classification_control.candidates] == [OTHER_TEXT]
    control_pack = build_style_pack(
        classification_control,
        extract_style_features(classification_control),
    )
    assert control_pack.examples == [OTHER_TEXT]

    package = _select(mapped)
    features = extract_style_features(package)
    pack = build_style_pack(package, features)

    assert package.audit.input_count == 3
    assert package.audit.owner_input_count == 2
    assert package.audit.dropped_other_count == 1
    assert package.audit.output_count == 2
    assert [candidate.text for candidate in package.candidates] == list(OWNER_TEXTS)
    assert OTHER_TEXT not in package.model_dump_json()
    assert features.statistics["candidate_count"] == 2
    assert len(pack.examples) == 2
    assert set(pack.examples) == set(OWNER_TEXTS)
    assert OTHER_TEXT not in pack.model_dump_json()


def test_direction_conflicting_owner_mapping_fails_before_candidates(tmp_path: Path) -> None:
    batch = StandardJsonAdapter().parse(_source(tmp_path))
    preview = preview_speaker_mapping(batch.records)
    assert len(preview.speakers) == 2

    with pytest.raises(ImportValidationError) as caught:
        mapped = apply_confirmed_mapping(
            batch.records,
            ConfirmedSpeakerMapping(roles={OWNER: SpeakerRole.OTHER, OTHER: SpeakerRole.OWNER}),
        )
        package = _select(mapped)
        build_style_pack(package, extract_style_features(package))

    assert caught.value.code == "invalid_speaker_mapping"
