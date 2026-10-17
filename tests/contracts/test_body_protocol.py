import json
import struct

import pytest
from pydantic import ValidationError

from companion.contracts import RobotEvent
from companion.contracts.body import (
    AUDIO_FRAME_HEADER_BYTES,
    AUDIO_FRAME_MAGIC,
    HEARTBEAT_INTERVAL_MS,
    INPUT_SAMPLE_RATE_HZ,
    LIVENESS_TIMEOUT_MS,
    MAX_AUDIO_FRAME_PAYLOAD_BYTES,
    MAX_CONTROL_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    AbortRecording,
    AudioFrame,
    BodyHeartbeat,
    BrainHeartbeat,
    ButtonPressed,
    Fault,
    Hello,
    MuteState,
    PlayAudio,
    PlaybackFinished,
    RecordingStopped,
    ShowState,
    StartRecording,
    StopPlayback,
    StopRecording,
    Welcome,
    decode_audio_frame,
    encode_audio_frame,
    encode_message,
    parse_body_message,
    parse_brain_message,
)

SESSION = "s-0001"
BASE = {"protocol_version": PROTOCOL_VERSION, "seq": 7}
SESSION_BASE = {**BASE, "session_id": SESSION}

BODY_MESSAGES = [
    Hello(**BASE, body_id="pi-01", input_sample_rate=16_000, output_sample_rates=(44_100, 48_000)),
    BodyHeartbeat(**SESSION_BASE),
    ButtonPressed(**SESSION_BASE, observed_at_ms=1234),
    MuteState(
        **SESSION_BASE,
        microphone_enabled=False,
        source="hardware_switch",
        observed_at_ms=99,
    ),
    RecordingStopped(**SESSION_BASE, turn_id=3, reason="stop_command", total_bytes=32_000),
    RecordingStopped(**SESSION_BASE, turn_id=3, reason="muted", total_bytes=0),
    PlaybackFinished(**SESSION_BASE, playback_token=5, result="completed"),
    Fault(**SESSION_BASE, kind="speaker_unavailable", detail="usb gone"),
    Fault(**SESSION_BASE, kind="internal_error"),
]

BRAIN_MESSAGES = [
    Welcome(
        **SESSION_BASE,
        heartbeat_interval_ms=HEARTBEAT_INTERVAL_MS,
        liveness_timeout_ms=LIVENESS_TIMEOUT_MS,
    ),
    BrainHeartbeat(**SESSION_BASE),
    StartRecording(**SESSION_BASE, turn_id=3, max_seconds=30),
    StopRecording(**SESSION_BASE, turn_id=3),
    AbortRecording(**SESSION_BASE, turn_id=3),
    PlayAudio(
        **SESSION_BASE,
        playback_token=5,
        sample_rate=44_100,
        channels=1,
        sample_format="pcm_s16le",
        total_bytes=88_200,
    ),
    StopPlayback(**SESSION_BASE, playback_token=5),
    ShowState(**SESSION_BASE, event=RobotEvent(state="speaking")),
]


def test_constants() -> None:
    assert PROTOCOL_VERSION == 1
    assert HEARTBEAT_INTERVAL_MS == 1000
    assert LIVENESS_TIMEOUT_MS == 3000
    assert INPUT_SAMPLE_RATE_HZ == 16_000
    assert MAX_AUDIO_FRAME_PAYLOAD_BYTES == 32_000


@pytest.mark.parametrize("message", BODY_MESSAGES, ids=lambda m: m.type)
def test_body_message_round_trip(message) -> None:
    text = encode_message(message)
    parsed = parse_body_message(text)
    assert type(parsed) is type(message)
    assert parsed == message
    assert json.loads(encode_message(parsed)) == json.loads(text)


@pytest.mark.parametrize("message", BRAIN_MESSAGES, ids=lambda m: m.type)
def test_brain_message_round_trip(message) -> None:
    text = encode_message(message)
    parsed = parse_brain_message(text)
    assert type(parsed) is type(message)
    assert parsed == message
    assert json.loads(encode_message(parsed)) == json.loads(text)


def test_every_message_type_is_covered_and_unique() -> None:
    body_types = {m.type for m in BODY_MESSAGES}
    brain_types = {m.type for m in BRAIN_MESSAGES}
    assert body_types == {
        "hello",
        "heartbeat",
        "button_pressed",
        "mute_state",
        "recording_stopped",
        "playback_finished",
        "fault",
    }
    assert brain_types == {
        "welcome",
        "heartbeat",
        "start_recording",
        "stop_recording",
        "abort_recording",
        "play_audio",
        "stop_playback",
        "show_state",
    }


def test_discriminator_selects_model_from_type() -> None:
    text = json.dumps({**SESSION_BASE, "type": "stop_recording", "turn_id": 9})
    parsed = parse_brain_message(text)
    assert isinstance(parsed, StopRecording)
    assert parsed.turn_id == 9


def test_direction_is_enforced() -> None:
    brain_only = encode_message(StopRecording(**SESSION_BASE, turn_id=1))
    body_only = encode_message(ButtonPressed(**SESSION_BASE, observed_at_ms=1))
    with pytest.raises(ValueError):
        parse_body_message(brain_only)
    with pytest.raises(ValueError):
        parse_brain_message(body_only)


def _valid_body_dict() -> dict:
    return json.loads(encode_message(ButtonPressed(**SESSION_BASE, observed_at_ms=1)))


@pytest.mark.parametrize(
    "mutation",
    [
        {"type": "unknown"},
        {"extra": 1},
        {"protocol_version": 2},
        {"protocol_version": 0},
        {"seq": -1},
        {"seq": 2**32},
        {"observed_at_ms": -5},
        {"observed_at_ms": "5"},
        {"seq": True},
        {"session_id": ""},
        {"session_id": "x" * 65},
        {"session_id": "has space"},
    ],
)
def test_invalid_fields_rejected(mutation) -> None:
    data = {**_valid_body_dict(), **mutation}
    with pytest.raises(ValueError):
        parse_body_message(json.dumps(data))


@pytest.mark.parametrize("field", ["protocol_version", "seq", "session_id", "type"])
def test_missing_envelope_fields_rejected(field) -> None:
    data = _valid_body_dict()
    del data[field]
    with pytest.raises(ValueError):
        parse_body_message(json.dumps(data))


def test_hello_has_no_session_id() -> None:
    data = json.loads(encode_message(BODY_MESSAGES[0]))
    assert "session_id" not in data
    data["session_id"] = SESSION
    with pytest.raises(ValueError):
        parse_body_message(json.dumps(data))


def test_hello_capabilities_bounds() -> None:
    with pytest.raises(ValidationError):
        Hello(**BASE, body_id="pi", input_sample_rate=48_000, output_sample_rates=(44_100,))
    with pytest.raises(ValidationError):
        Hello(**BASE, body_id="pi", input_sample_rate=16_000, output_sample_rates=())
    with pytest.raises(ValidationError):
        Hello(**BASE, body_id="pi", input_sample_rate=16_000, output_sample_rates=(1,))


def test_welcome_timeout_must_exceed_interval() -> None:
    with pytest.raises(ValidationError):
        Welcome(**SESSION_BASE, heartbeat_interval_ms=3000, liveness_timeout_ms=3000)


def test_enum_fields_reject_unknown_values() -> None:
    with pytest.raises(ValidationError):
        RecordingStopped(**SESSION_BASE, turn_id=1, reason="whatever", total_bytes=0)
    with pytest.raises(ValidationError):
        PlaybackFinished(**SESSION_BASE, playback_token=1, result="maybe")
    with pytest.raises(ValidationError):
        Fault(**SESSION_BASE, kind="anything")
    with pytest.raises(ValidationError):
        MuteState(**SESSION_BASE, microphone_enabled=True, source="software", observed_at_ms=0)


def test_mute_state_source_allows_development() -> None:
    message = MuteState(
        **SESSION_BASE, microphone_enabled=True, source="development", observed_at_ms=0
    )
    assert parse_body_message(encode_message(message)) == message


def test_fault_detail_is_short() -> None:
    with pytest.raises(ValidationError):
        Fault(**SESSION_BASE, kind="internal_error", detail="x" * 121)


def test_recording_and_playback_bounds() -> None:
    with pytest.raises(ValidationError):
        StartRecording(**SESSION_BASE, turn_id=1, max_seconds=0)
    with pytest.raises(ValidationError):
        StartRecording(**SESSION_BASE, turn_id=1, max_seconds=61)
    with pytest.raises(ValidationError):
        RecordingStopped(**SESSION_BASE, turn_id=1, reason="limit", total_bytes=3)
    with pytest.raises(ValidationError):
        RecordingStopped(**SESSION_BASE, turn_id=1, reason="limit", total_bytes=60 * 32_000 + 2)
    play = {
        **SESSION_BASE,
        "playback_token": 1,
        "sample_rate": 44_100,
        "channels": 1,
        "sample_format": "pcm_s16le",
        "total_bytes": 2,
    }
    PlayAudio(**play)
    for bad in (
        {"channels": 2},
        {"sample_format": "f32"},
        {"sample_rate": 7_999},
        {"sample_rate": 48_001},
        {"total_bytes": 0},
        {"total_bytes": 3},
        {"total_bytes": 120 * 48_000 * 2 + 2},
    ):
        with pytest.raises(ValidationError):
            PlayAudio(**{**play, **bad})


def test_oversized_control_message_rejected() -> None:
    message = ShowState(**SESSION_BASE, event=RobotEvent(state="error", detail="x" * 5000))
    with pytest.raises(ValueError):
        parse_brain_message(encode_message(message))
    assert MAX_CONTROL_MESSAGE_BYTES == 4096


def test_messages_are_immutable() -> None:
    message = StopRecording(**SESSION_BASE, turn_id=1)
    with pytest.raises(ValidationError):
        message.turn_id = 2


# ---- binary audio frames ----


def test_audio_frame_round_trip_uplink_and_last_flag() -> None:
    payload = b"\x01\x00\xff\x7f" * 10
    frame = AudioFrame(
        direction="uplink", stream_id=3, frame_index=0, is_last=False, payload=payload
    )
    data = encode_audio_frame(frame)
    assert data[:4] == AUDIO_FRAME_MAGIC
    assert len(data) == AUDIO_FRAME_HEADER_BYTES + len(payload)
    assert decode_audio_frame(data) == frame

    last = AudioFrame(
        direction="downlink", stream_id=2**32 - 1, frame_index=9, is_last=True, payload=b""
    )
    decoded = decode_audio_frame(encode_audio_frame(last))
    assert decoded == last
    assert decoded.is_last is True


def test_audio_frame_max_payload_round_trip() -> None:
    frame = AudioFrame(
        direction="downlink",
        stream_id=1,
        frame_index=1,
        is_last=True,
        payload=b"\x00" * MAX_AUDIO_FRAME_PAYLOAD_BYTES,
    )
    assert decode_audio_frame(encode_audio_frame(frame)) == frame


@pytest.mark.parametrize(
    "kwargs",
    [
        {"payload": b"\x00" * (MAX_AUDIO_FRAME_PAYLOAD_BYTES + 2)},
        {"payload": b"\x00\x00\x00"},
        {"stream_id": -1},
        {"stream_id": 2**32},
        {"frame_index": -1},
        {"frame_index": 2**32},
        {"direction": "sideways"},
    ],
)
def test_audio_frame_invalid_fields_raise_value_error(kwargs) -> None:
    fields = {
        "direction": "uplink",
        "stream_id": 1,
        "frame_index": 0,
        "is_last": False,
        "payload": b"",
    }
    fields.update(kwargs)
    with pytest.raises(ValueError):
        AudioFrame(**fields)


def _encoded() -> bytes:
    return encode_audio_frame(
        AudioFrame(
            direction="uplink", stream_id=1, frame_index=2, is_last=False, payload=b"\x01\x02" * 4
        )
    )


def _patch(data: bytes, offset: int, fmt: str, value: int) -> bytes:
    out = bytearray(data)
    struct.pack_into(fmt, out, offset, value)
    return bytes(out)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: b"",
        lambda d: d[: AUDIO_FRAME_HEADER_BYTES - 1],
        lambda d: d[:-1],
        lambda d: d[:-2],
        lambda d: d + b"\x00\x00",
        lambda d: b"XXXX" + d[4:],
        lambda d: _patch(d, 4, "<B", 2),  # version
        lambda d: _patch(d, 5, "<B", 3),  # direction
        lambda d: _patch(d, 6, "<B", 0b10),  # unknown flag bit
        lambda d: _patch(d, 7, "<B", 1),  # reserved
        lambda d: _patch(d, 16, "<I", 3)[: AUDIO_FRAME_HEADER_BYTES + 3],  # odd payload
        lambda d: _patch(d, 16, "<I", MAX_AUDIO_FRAME_PAYLOAD_BYTES + 2),
    ],
)
def test_corrupt_audio_frames_raise_value_error(corrupt) -> None:
    with pytest.raises(ValueError):
        decode_audio_frame(corrupt(_encoded()))
