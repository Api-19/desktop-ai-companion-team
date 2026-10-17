"""Body (Raspberry Pi) <-> brain (Mac) protocol: control messages and audio frames.

Pure data and codecs. Transport (WSS + mutual TLS) and timing live elsewhere.
See docs/superpowers/specs/2026-10-13-body-brain-protocol-design.md.
"""

import struct
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, TypeAdapter, model_validator

from companion.contracts.base import ContractModel
from companion.contracts.robot import RobotEvent

PROTOCOL_VERSION = 1
HEARTBEAT_INTERVAL_MS = 1000
LIVENESS_TIMEOUT_MS = 3000
INPUT_SAMPLE_RATE_HZ = 16_000
MIN_OUTPUT_SAMPLE_RATE_HZ = 8_000
MAX_OUTPUT_SAMPLE_RATE_HZ = 48_000
MAX_RECORDING_SECONDS = 60
MAX_PLAYBACK_SECONDS = 120
MAX_RECORDING_BYTES = MAX_RECORDING_SECONDS * INPUT_SAMPLE_RATE_HZ * 2
MAX_PLAYBACK_BYTES = MAX_PLAYBACK_SECONDS * MAX_OUTPUT_SAMPLE_RATE_HZ * 2
MAX_CONTROL_MESSAGE_BYTES = 4096
MAX_AUDIO_FRAME_PAYLOAD_BYTES = 32_000
MAX_FAULT_DETAIL_CHARS = 120
UINT32_MAX = 2**32 - 1

Uint32 = Annotated[int, Field(ge=0, le=UINT32_MAX)]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
OutputSampleRate = Annotated[int, Field(ge=MIN_OUTPUT_SAMPLE_RATE_HZ, le=MAX_OUTPUT_SAMPLE_RATE_HZ)]


class _Message(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    protocol_version: Literal[1]
    seq: Uint32


class _SessionMessage(_Message):
    session_id: Identifier


# ---- body -> brain ----


class Hello(_Message):
    type: Literal["hello"] = "hello"
    body_id: Identifier
    input_sample_rate: Literal[16000]
    output_sample_rates: tuple[OutputSampleRate, ...] = Field(min_length=1, max_length=8)


class BodyHeartbeat(_SessionMessage):
    type: Literal["heartbeat"] = "heartbeat"


class ButtonPressed(_SessionMessage):
    type: Literal["button_pressed"] = "button_pressed"
    observed_at_ms: Uint32


class MuteState(_SessionMessage):
    type: Literal["mute_state"] = "mute_state"
    microphone_enabled: bool
    source: Literal["hardware_switch", "development"]
    observed_at_ms: Uint32


class RecordingStopped(_SessionMessage):
    type: Literal["recording_stopped"] = "recording_stopped"
    turn_id: Uint32
    reason: Literal["stop_command", "muted", "local_abort", "limit"]
    total_bytes: Annotated[int, Field(ge=0, le=MAX_RECORDING_BYTES, multiple_of=2)]


class PlaybackFinished(_SessionMessage):
    type: Literal["playback_finished"] = "playback_finished"
    playback_token: Uint32
    result: Literal["completed", "stopped", "failed"]


class Fault(_SessionMessage):
    type: Literal["fault"] = "fault"
    kind: Literal[
        "microphone_unavailable",
        "speaker_unavailable",
        "display_unavailable",
        "mute_state_unknown",
        "audio_overrun",
        "protocol_error",
        "internal_error",
    ]
    detail: Annotated[str, Field(max_length=MAX_FAULT_DETAIL_CHARS)] | None = None


# ---- brain -> body ----


class Welcome(_SessionMessage):
    type: Literal["welcome"] = "welcome"
    heartbeat_interval_ms: Annotated[int, Field(ge=100, le=60_000)]
    liveness_timeout_ms: Annotated[int, Field(ge=100, le=60_000)]

    @model_validator(mode="after")
    def _timeout_exceeds_interval(self) -> "Welcome":
        if self.liveness_timeout_ms <= self.heartbeat_interval_ms:
            raise ValueError("liveness_timeout_ms must exceed heartbeat_interval_ms")
        return self


class BrainHeartbeat(_SessionMessage):
    type: Literal["heartbeat"] = "heartbeat"


class StartRecording(_SessionMessage):
    type: Literal["start_recording"] = "start_recording"
    turn_id: Uint32
    max_seconds: Annotated[int, Field(ge=1, le=MAX_RECORDING_SECONDS)]


class StopRecording(_SessionMessage):
    type: Literal["stop_recording"] = "stop_recording"
    turn_id: Uint32


class AbortRecording(_SessionMessage):
    type: Literal["abort_recording"] = "abort_recording"
    turn_id: Uint32


class PlayAudio(_SessionMessage):
    type: Literal["play_audio"] = "play_audio"
    playback_token: Uint32
    sample_rate: OutputSampleRate
    channels: Literal[1]
    sample_format: Literal["pcm_s16le"]
    total_bytes: Annotated[int, Field(ge=2, le=MAX_PLAYBACK_BYTES, multiple_of=2)]


class StopPlayback(_SessionMessage):
    type: Literal["stop_playback"] = "stop_playback"
    playback_token: Uint32


class ShowState(_SessionMessage):
    type: Literal["show_state"] = "show_state"
    event: RobotEvent


BodyMessage = Annotated[
    Hello | BodyHeartbeat | ButtonPressed | MuteState | RecordingStopped | PlaybackFinished | Fault,
    Field(discriminator="type"),
]
BrainMessage = Annotated[
    Welcome
    | BrainHeartbeat
    | StartRecording
    | StopRecording
    | AbortRecording
    | PlayAudio
    | StopPlayback
    | ShowState,
    Field(discriminator="type"),
]

_BODY_ADAPTER: TypeAdapter[BodyMessage] = TypeAdapter(BodyMessage)
_BRAIN_ADAPTER: TypeAdapter[BrainMessage] = TypeAdapter(BrainMessage)


def encode_message(message: _Message) -> str:
    return message.model_dump_json()


def _check_size(text: str) -> None:
    if len(text.encode("utf-8")) > MAX_CONTROL_MESSAGE_BYTES:
        raise ValueError("control message exceeds MAX_CONTROL_MESSAGE_BYTES")


def parse_body_message(text: str) -> BodyMessage:
    """Parse a body->brain JSON text frame. Raises ValueError on any violation."""
    _check_size(text)
    return _BODY_ADAPTER.validate_json(text)


def parse_brain_message(text: str) -> BrainMessage:
    """Parse a brain->body JSON text frame. Raises ValueError on any violation."""
    _check_size(text)
    return _BRAIN_ADAPTER.validate_json(text)


# ---- binary audio frames ----
#
# Header (little-endian, 20 bytes):
#   0  4s  magic b"CPAF"
#   4  B   frame format version (1)
#   5  B   direction (1 = uplink recording, 2 = downlink playback)
#   6  B   flags (bit 0 = last frame; other bits must be 0)
#   7  B   reserved (0)
#   8  I   stream_id (turn_id for uplink, playback_token for downlink)
#  12  I   frame_index
#  16  I   payload_length
# Payload: PCM16LE mono, even length, <= MAX_AUDIO_FRAME_PAYLOAD_BYTES.

AUDIO_FRAME_MAGIC = b"CPAF"
AUDIO_FRAME_VERSION = 1
_HEADER = struct.Struct("<4sBBBBIII")
AUDIO_FRAME_HEADER_BYTES = _HEADER.size
_FLAG_LAST = 0b1
_DIRECTION_CODES = {"uplink": 1, "downlink": 2}
_DIRECTION_NAMES = {code: name for name, code in _DIRECTION_CODES.items()}


@dataclass(frozen=True)
class AudioFrame:
    direction: Literal["uplink", "downlink"]
    stream_id: int
    frame_index: int
    is_last: bool
    payload: bytes

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTION_CODES:
            raise ValueError("unknown audio frame direction")
        for value in (self.stream_id, self.frame_index):
            if type(value) is not int or not 0 <= value <= UINT32_MAX:
                raise ValueError("audio frame ids must be uint32")
        if type(self.is_last) is not bool or type(self.payload) is not bytes:
            raise ValueError("invalid audio frame field type")
        if len(self.payload) > MAX_AUDIO_FRAME_PAYLOAD_BYTES:
            raise ValueError("audio frame payload too large")
        if len(self.payload) % 2:
            raise ValueError("PCM16 payload must have even length")


def encode_audio_frame(frame: AudioFrame) -> bytes:
    header = _HEADER.pack(
        AUDIO_FRAME_MAGIC,
        AUDIO_FRAME_VERSION,
        _DIRECTION_CODES[frame.direction],
        _FLAG_LAST if frame.is_last else 0,
        0,
        frame.stream_id,
        frame.frame_index,
        len(frame.payload),
    )
    return header + frame.payload


def decode_audio_frame(data: bytes) -> AudioFrame:
    if len(data) < AUDIO_FRAME_HEADER_BYTES:
        raise ValueError("audio frame truncated")
    magic, version, direction, flags, reserved, stream_id, frame_index, length = (
        _HEADER.unpack_from(data)
    )
    if magic != AUDIO_FRAME_MAGIC or version != AUDIO_FRAME_VERSION:
        raise ValueError("bad audio frame magic or version")
    if direction not in _DIRECTION_NAMES or flags & ~_FLAG_LAST or reserved:
        raise ValueError("bad audio frame header")
    if len(data) != AUDIO_FRAME_HEADER_BYTES + length:
        raise ValueError("audio frame length mismatch")
    return AudioFrame(
        direction=_DIRECTION_NAMES[direction],
        stream_id=stream_id,
        frame_index=frame_index,
        is_last=bool(flags & _FLAG_LAST),
        payload=bytes(data[AUDIO_FRAME_HEADER_BYTES:]),
    )
