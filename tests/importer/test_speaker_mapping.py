from pathlib import Path

import pytest

from companion.contracts import SpeakerRole
from companion.importer import (
    CipherTalkJsonAdapter,
    ConfirmedSpeakerMapping,
    ImportValidationError,
    apply_confirmed_mapping,
    preview_speaker_mapping,
)

FIXTURE = Path("fixtures/synthetic/ciphertalk_private_chat.json")


def test_preview_groups_speakers_and_limits_samples() -> None:
    batch = CipherTalkJsonAdapter().parse(FIXTURE)

    preview = preview_speaker_mapping(batch.records)

    assert len(preview.speakers) == 2
    assert [speaker.source_speaker_id for speaker in preview.speakers] == [
        "synthetic-contact",
        "synthetic-owner",
    ]
    owner = next(item for item in preview.speakers if item.source_speaker_id == "synthetic-owner")
    assert owner.message_count == 2
    assert owner.outbound_count == 2
    assert owner.inbound_count == 0
    assert owner.content_type_counts == {"other": 1, "text": 1}
    assert len(owner.sample_texts) <= 3
    assert all(len(text) <= 80 for text in owner.sample_texts)


def test_preview_truncates_text_and_keeps_only_three_samples() -> None:
    batch = CipherTalkJsonAdapter().parse(FIXTURE)
    template = batch.records[0]
    records = [
        template.model_copy(update={"message_id": f"synthetic-{index}", "text": "x" * (81 + index)})
        for index in range(4)
    ]

    preview = preview_speaker_mapping(records)

    assert len(preview.speakers[0].sample_texts) == 3
    assert all(len(text) == 80 for text in preview.speakers[0].sample_texts)


def test_complete_mapping_creates_shared_chat_records() -> None:
    batch = CipherTalkJsonAdapter().parse(FIXTURE)
    mapping = ConfirmedSpeakerMapping(
        roles={
            "synthetic-owner": SpeakerRole.OWNER,
            "synthetic-contact": SpeakerRole.OTHER,
        }
    )

    records = apply_confirmed_mapping(batch.records, mapping)

    assert len(records) == len(batch.records)
    assert records[0].speaker_role is SpeakerRole.OWNER
    assert records[1].speaker_role is SpeakerRole.OTHER
    assert records[2].speaker_role is SpeakerRole.OWNER
    assert records[0].message_id == batch.records[0].message_id
    assert records[0].timestamp == batch.records[0].timestamp


@pytest.mark.parametrize(
    "roles",
    [
        {},
        {"synthetic-owner": SpeakerRole.OWNER},
        {
            "synthetic-owner": SpeakerRole.UNKNOWN,
            "synthetic-contact": SpeakerRole.OTHER,
        },
        {
            "synthetic-owner": SpeakerRole.OWNER,
            "synthetic-contact": SpeakerRole.OWNER,
        },
        {
            "synthetic-owner": SpeakerRole.OWNER,
            "synthetic-contact": SpeakerRole.OTHER,
            "synthetic-extra": SpeakerRole.OTHER,
        },
    ],
)
def test_mapping_must_be_complete_and_match_message_directions(
    roles: dict[str, SpeakerRole],
) -> None:
    batch = CipherTalkJsonAdapter().parse(FIXTURE)

    with pytest.raises(ImportValidationError) as exc_info:
        apply_confirmed_mapping(batch.records, ConfirmedSpeakerMapping(roles=roles))

    assert exc_info.value.code == "invalid_speaker_mapping"
