from companion.contracts.chat import ChatRecord, SpeakerRole
from companion.importer.errors import ImportValidationError
from companion.importer.models import (
    ConfirmedSpeakerMapping,
    MessageDirection,
    SourceChatRecord,
    SpeakerMappingPreview,
    SpeakerPreview,
)

MAX_SAMPLE_COUNT = 3
MAX_SAMPLE_LENGTH = 80


def preview_speaker_mapping(records: list[SourceChatRecord]) -> SpeakerMappingPreview:
    grouped: dict[str, dict[str, object]] = {}
    for record in records:
        speaker = grouped.setdefault(
            record.source_speaker_id,
            {
                "source_speaker_name": None,
                "message_count": 0,
                "content_type_counts": {},
                "sample_texts": [],
                "outbound_count": 0,
                "inbound_count": 0,
            },
        )
        if speaker["source_speaker_name"] is None and record.source_speaker_name:
            speaker["source_speaker_name"] = record.source_speaker_name
        speaker["message_count"] = int(speaker["message_count"]) + 1

        content_type_counts = speaker["content_type_counts"]
        assert isinstance(content_type_counts, dict)
        content_type_counts[record.content_type] = (
            content_type_counts.get(record.content_type, 0) + 1
        )

        direction_key = (
            "outbound_count" if record.direction is MessageDirection.OUTBOUND else "inbound_count"
        )
        speaker[direction_key] = int(speaker[direction_key]) + 1

        sample_texts = speaker["sample_texts"]
        assert isinstance(sample_texts, list)
        if record.text and len(sample_texts) < MAX_SAMPLE_COUNT:
            sample_texts.append(record.text[:MAX_SAMPLE_LENGTH])

    previews = [
        SpeakerPreview(
            source_speaker_id=source_speaker_id,
            source_speaker_name=values["source_speaker_name"],
            message_count=values["message_count"],
            content_type_counts=values["content_type_counts"],
            sample_texts=values["sample_texts"],
            outbound_count=values["outbound_count"],
            inbound_count=values["inbound_count"],
        )
        for source_speaker_id, values in sorted(grouped.items())
    ]
    return SpeakerMappingPreview(speakers=previews)


def apply_confirmed_mapping(
    records: list[SourceChatRecord],
    mapping: ConfirmedSpeakerMapping,
) -> list[ChatRecord]:
    observed_speakers = {record.source_speaker_id for record in records}
    provided_speakers = set(mapping.roles)
    roles = list(mapping.roles.values())
    outbound_speakers = {
        record.source_speaker_id
        for record in records
        if record.direction is MessageDirection.OUTBOUND
    }
    role_conflict = any(
        mapping.roles.get(record.source_speaker_id)
        is not (
            SpeakerRole.OWNER
            if record.direction is MessageDirection.OUTBOUND
            else SpeakerRole.OTHER
        )
        for record in records
    )
    if (
        observed_speakers != provided_speakers
        or any(role is SpeakerRole.UNKNOWN for role in roles)
        or len(outbound_speakers) > 1
        or role_conflict
    ):
        raise ImportValidationError(
            "invalid_speaker_mapping",
            "mapping must cover every speaker and agree with message directions",
        )

    return [
        ChatRecord(
            conversation_id=record.conversation_id,
            message_id=record.message_id,
            timestamp=record.timestamp,
            speaker_role=mapping.roles[record.source_speaker_id],
            content_type=record.content_type,
            text=record.text,
        )
        for record in records
    ]
