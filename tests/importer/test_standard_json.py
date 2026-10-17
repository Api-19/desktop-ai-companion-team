import json
import traceback
from pathlib import Path

import pytest

from companion.contracts import SpeakerRole
from companion.importer import (
    ChatImportAdapter,
    ConfirmedSpeakerMapping,
    ImportValidationError,
    MessageDirection,
    StandardJsonAdapter,
    apply_confirmed_mapping,
)

FIXTURE = Path("fixtures/synthetic/standard_private_chat.json")


def load_fixture() -> list[dict[str, object]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def write_payload(tmp_path: Path, payload: object, *, encoding: str = "utf-8") -> Path:
    path = tmp_path / "synthetic-standard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding=encoding)
    return path


def standard_json_traceback_locals(
    error: ImportValidationError,
) -> list[tuple[str, str]]:
    captured: list[tuple[str, str]] = []
    current = error.__traceback__
    while current is not None:
        frame = current.tb_frame
        if frame.f_globals.get("__name__") == "companion.importer.standard_json":
            captured.append((frame.f_code.co_name, repr(frame.f_locals)))
        current = current.tb_next
    return captured


def test_parse_standard_json_to_stably_sorted_source_records() -> None:
    adapter: ChatImportAdapter = StandardJsonAdapter()

    batch = adapter.parse(FIXTURE)

    assert [record.message_id for record in batch.records] == [
        "synthetic-message-1",
        "synthetic-message-2",
        "synthetic-message-3",
    ]
    assert batch.records[0].direction is MessageDirection.INBOUND
    assert batch.records[0].content_type == "image"
    assert batch.records[0].text is None
    assert batch.records[2].source_speaker_name is None
    assert batch.records[2].text == ""
    assert batch.issues == []


def test_public_adapter_reuses_confirmed_mapping_boundary() -> None:
    batch = StandardJsonAdapter().parse(FIXTURE)

    records = apply_confirmed_mapping(
        batch.records,
        ConfirmedSpeakerMapping(
            roles={
                "synthetic-owner": SpeakerRole.OWNER,
                "synthetic-contact": SpeakerRole.OTHER,
            }
        ),
    )

    assert len(records) == 3
    assert [record.speaker_role for record in records] == [
        SpeakerRole.OTHER,
        SpeakerRole.OWNER,
        SpeakerRole.OWNER,
    ]


@pytest.mark.parametrize(
    "content_type",
    ["text", "image", "voice", "video", "emoji", "quote", "system", "other"],
)
def test_all_standard_content_types_round_trip(
    tmp_path: Path,
    content_type: str,
) -> None:
    payload = load_fixture()
    payload[0]["content_type"] = content_type

    batch = StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    record = next(item for item in batch.records if item.message_id == "synthetic-message-2")
    assert record.content_type == content_type


def test_initial_adapter_failure_never_contains_input_values(tmp_path: Path) -> None:
    path = write_payload(tmp_path, {"synthetic-secret-field": "synthetic-secret-value"})

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(path)

    assert "synthetic-secret" not in str(exc_info.value)


def test_invalid_json_uses_safe_error_code(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-standard.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(path)

    assert exc_info.value.code == "invalid_json"


@pytest.mark.parametrize("payload", [{}, "synthetic", 1, None])
def test_root_must_be_an_array(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_root"


def test_message_array_must_not_be_empty(tmp_path: Path) -> None:
    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, []))

    assert exc_info.value.code == "empty_messages"


def test_utf8_bom_is_supported(tmp_path: Path) -> None:
    batch = StandardJsonAdapter().parse(
        write_payload(tmp_path, load_fixture(), encoding="utf-8-sig")
    )

    assert len(batch.records) == 3


@pytest.mark.parametrize("message", [None, "synthetic", 1])
def test_each_message_must_be_an_object(tmp_path: Path, message: object) -> None:
    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, [message]))

    assert exc_info.value.code == "invalid_message"
    assert exc_info.value.message_index == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conversation_id", ""),
        ("conversation_id", 7),
        ("message_id", ""),
        ("timestamp", "2026-07-17T10:30:00"),
        ("timestamp", 1784246400),
        ("source_speaker_id", ""),
        ("direction", "sideways"),
        ("content_type", "sticker"),
        ("source_speaker_name", ""),
        ("text", 7),
        ("speaker_role", "owner"),
    ],
)
def test_invalid_message_fields_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = load_fixture()
    payload[0][field] = value

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_message"
    assert exc_info.value.message_index == 0


@pytest.mark.parametrize("version", [2, True, "1"])
def test_unsupported_or_non_integer_version_is_rejected(
    tmp_path: Path,
    version: object,
) -> None:
    payload = load_fixture()
    payload[0]["schema_version"] = version

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "unsupported_schema_version"


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    payload = load_fixture()
    del payload[0]["text"]

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_message"
    assert exc_info.value.field_path == "messages[0].text"


def test_validation_error_never_contains_input_values(tmp_path: Path) -> None:
    payload = load_fixture()
    payload[0].update(
        {
            "direction": "synthetic-secret-direction",
            "source_speaker_id": "synthetic-secret-account",
            "source_speaker_name": "Synthetic Secret Name",
            "text": "synthetic secret message body",
        }
    )

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    error_text = str(exc_info.value)
    assert exc_info.value.code == "invalid_message"
    assert "synthetic-secret" not in error_text
    assert "Synthetic Secret Name" not in error_text


def test_unknown_field_name_never_appears_in_validation_error(tmp_path: Path) -> None:
    payload = load_fixture()
    payload[0]["synthetic-secret-unknown-field"] = "synthetic-safe-value"

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_message"
    assert exc_info.value.field_path == "messages[0]"
    assert "synthetic-secret" not in str(exc_info.value)


def test_validation_traceback_never_contains_input_values(tmp_path: Path) -> None:
    payload = load_fixture()
    payload[0]["synthetic-secret-unknown-field"] = "synthetic-secret-value"

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    traceback_text = "".join(traceback.format_exception(exc_info.value))
    assert "synthetic-secret" not in traceback_text
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_invalid_json_error_does_not_retain_source_document(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-standard.json"
    path.write_text('{"synthetic-secret-json":', encoding="utf-8")

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(path)

    assert "synthetic-secret" not in "".join(traceback.format_exception(exc_info.value))
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    frame_locals = standard_json_traceback_locals(exc_info.value)
    assert [name for name, _ in frame_locals] == ["parse"]
    assert all("synthetic-secret" not in values for _, values in frame_locals)


def test_missing_file_error_does_not_retain_source_path() -> None:
    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(Path("synthetic-secret-account.json"))

    traceback_frame_locals: list[str] = []
    current = exc_info.value.__traceback__
    while current is not None:
        traceback_frame_locals.append(repr(current.tb_frame.f_locals))
        current = current.tb_next

    assert exc_info.value.code == "invalid_json"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert all("synthetic-secret-account.json" not in values for values in traceback_frame_locals)
    assert all("synthetic-secret-account" not in values for values in traceback_frame_locals)


def test_validation_traceback_frames_do_not_retain_input_values(tmp_path: Path) -> None:
    payload = load_fixture()
    payload[0].update(
        {
            "conversation_id": "synthetic-secret-conversation",
            "message_id": "synthetic-secret-message",
            "source_speaker_id": "synthetic-secret-account",
            "source_speaker_name": "Synthetic Secret Name",
            "text": "synthetic-secret-body",
            "direction": "synthetic-invalid-direction",
        }
    )

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    frame_locals = standard_json_traceback_locals(exc_info.value)
    assert [name for name, _ in frame_locals] == ["parse"]
    assert all("synthetic-secret" not in values for _, values in frame_locals)
    assert all("Synthetic Secret Name" not in values for _, values in frame_locals)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_excessively_nested_json_uses_safe_error_code(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-standard.json"
    source = "[" * 10_000 + '"synthetic-secret-deep-json"' + "]" * 10_000
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(path)

    assert exc_info.value.code == "invalid_json"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    frame_locals = standard_json_traceback_locals(exc_info.value)
    assert [name for name, _ in frame_locals] == ["parse"]
    assert all("synthetic-secret" not in values for _, values in frame_locals)


def test_mixed_conversations_are_rejected(tmp_path: Path) -> None:
    payload = load_fixture()
    payload[1]["conversation_id"] = "synthetic-second-conversation"

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "mixed_conversations"
    assert exc_info.value.message_index == 1


def test_duplicate_message_ids_are_rejected(tmp_path: Path) -> None:
    payload = load_fixture()
    payload[1]["message_id"] = "synthetic-message-2"

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "duplicate_message_id"


def test_multiple_outbound_speakers_are_rejected(tmp_path: Path) -> None:
    payload = load_fixture()
    payload[2]["source_speaker_id"] = "synthetic-second-owner"

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_private_chat"


def test_multiple_inbound_speakers_are_rejected(tmp_path: Path) -> None:
    payload = load_fixture()
    second_inbound = {
        **payload[1],
        "message_id": "synthetic-message-4",
        "source_speaker_id": "synthetic-second-contact",
    }
    payload.append(second_inbound)

    with pytest.raises(ImportValidationError) as exc_info:
        StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_private_chat"


def test_same_speaker_in_both_directions_cannot_be_confirmed(tmp_path: Path) -> None:
    payload = load_fixture()
    payload[1]["source_speaker_id"] = "synthetic-owner"
    batch = StandardJsonAdapter().parse(write_payload(tmp_path, payload))

    with pytest.raises(ImportValidationError) as exc_info:
        apply_confirmed_mapping(
            batch.records,
            ConfirmedSpeakerMapping(roles={"synthetic-owner": SpeakerRole.OWNER}),
        )

    assert exc_info.value.code == "invalid_speaker_mapping"
