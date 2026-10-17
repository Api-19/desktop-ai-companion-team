import json
from datetime import UTC
from pathlib import Path

import pytest

from companion.contracts import SpeakerRole
from companion.importer import (
    CipherTalkJsonAdapter,
    ConfirmedSpeakerMapping,
    ImportValidationError,
    MessageDirection,
    apply_confirmed_mapping,
)

FIXTURE = Path("fixtures/synthetic/ciphertalk_private_chat.json")


def load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def write_payload(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_parse_normalizes_private_chat_without_raw_content() -> None:
    batch = CipherTalkJsonAdapter().parse(FIXTURE)

    assert len(batch.records) == 3
    assert batch.records[0].conversation_id == "wechat:synthetic-contact"
    assert batch.records[0].direction is MessageDirection.OUTBOUND
    assert batch.records[0].timestamp.tzinfo is UTC
    assert batch.records[1].content_type == "image"
    assert "synthetic-private-raw-value" not in batch.model_dump_json()


def test_outbound_direction_overrides_conflicting_sender_metadata() -> None:
    batch = CipherTalkJsonAdapter().parse(FIXTURE)

    assert batch.records[2].source_speaker_id == "synthetic-owner"
    assert [issue.code for issue in batch.issues] == ["sender_direction_conflict"]


def test_inbound_direction_overrides_conflicting_owner_sender_metadata(tmp_path: Path) -> None:
    payload = load_fixture()
    template = payload["messages"][0]  # type: ignore[index]
    payload["messages"] = [  # type: ignore[index]
        {
            **template,
            "platformMessageId": f"synthetic-inbound-{index}",
            "createTime": 1784246400 + index,
            "isSend": 0,
            "senderUsername": "synthetic-owner",
            "senderDisplayName": "Synthetic Contact",
        }
        for index in range(3)
    ]

    batch = CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert [record.source_speaker_id for record in batch.records] == [
        "synthetic-contact",
        "synthetic-contact",
        "synthetic-contact",
    ]
    assert [issue.code for issue in batch.issues] == ["sender_direction_conflict"] * 3

    records = apply_confirmed_mapping(
        batch.records,
        ConfirmedSpeakerMapping(roles={"synthetic-contact": SpeakerRole.OTHER}),
    )
    assert [record.speaker_role for record in records] == [SpeakerRole.OTHER] * 3


def test_unknown_content_type_becomes_other_with_warning(tmp_path: Path) -> None:
    payload = load_fixture()
    payload["messages"][0]["type"] = "synthetic-unknown-type"  # type: ignore[index]

    batch = CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert batch.records[0].content_type == "other"
    assert {issue.code for issue in batch.issues} == {
        "sender_direction_conflict",
        "unknown_content_type",
    }


def test_invalid_json_uses_safe_error_code(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ImportValidationError) as exc_info:
        CipherTalkJsonAdapter().parse(path)

    assert exc_info.value.code == "invalid_json"


@pytest.mark.parametrize("payload", [[], "synthetic", 1, None])
def test_root_must_be_object(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ImportValidationError) as exc_info:
        CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_root"


def test_envelope_fields_are_required(tmp_path: Path) -> None:
    payload = load_fixture()
    del payload["session"]

    with pytest.raises(ImportValidationError) as exc_info:
        CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "missing_envelope"


def test_only_wechat_platform_is_supported(tmp_path: Path) -> None:
    payload = load_fixture()
    payload["session"]["platform"] = "synthetic-platform"  # type: ignore[index]

    with pytest.raises(ImportValidationError) as exc_info:
        CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "unsupported_platform"


def test_group_chat_is_rejected(tmp_path: Path) -> None:
    payload = load_fixture()
    payload["session"]["isGroup"] = True  # type: ignore[index]

    with pytest.raises(ImportValidationError) as exc_info:
        CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "group_chat_unsupported"


def test_invalid_session_is_rejected_without_input_values(tmp_path: Path) -> None:
    payload = load_fixture()
    del payload["session"]["ownerId"]  # type: ignore[index]

    with pytest.raises(ImportValidationError) as exc_info:
        CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_session"
    assert "Synthetic Contact" not in str(exc_info.value)


def test_invalid_message_is_rejected_without_input_values(tmp_path: Path) -> None:
    payload = load_fixture()
    del payload["messages"][0]["createTime"]  # type: ignore[index]

    with pytest.raises(ImportValidationError) as exc_info:
        CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "invalid_message"
    assert exc_info.value.message_index == 0
    assert "synthetic hello" not in str(exc_info.value)


def test_duplicate_message_ids_are_rejected(tmp_path: Path) -> None:
    payload = load_fixture()
    payload["messages"][1]["platformMessageId"] = "synthetic-message-1"  # type: ignore[index]

    with pytest.raises(ImportValidationError) as exc_info:
        CipherTalkJsonAdapter().parse(write_payload(tmp_path, payload))

    assert exc_info.value.code == "duplicate_message_id"
