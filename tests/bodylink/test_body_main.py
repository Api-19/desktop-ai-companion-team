import math
import os
import signal
import struct
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.bodylink import body_main
from companion.bodylink.body import BodyClient
from companion.bodylink.body_devices import (
    DeviceNotFoundError,
    Pcm16StreamResampler,
    SoundDeviceMicrophone,
    SoundDeviceSpeaker,
    resolve_body_devices,
)
from companion.bodylink.body_main import BodyCore, TerminalButtonSource, run_self_test
from companion.bodylink.brain import BrainServer
from companion.bodylink.brain_ports import BrainBodyLink
from companion.bodylink.pairing import PairingPaths, body_ssl_context, brain_ssl_context
from companion.contracts import DialogueResponse, TranscriptConfidenceStatus
from companion.contracts.body import (
    AbortRecording,
    AudioFrame,
    BrainHeartbeat,
    ButtonPressed,
    Fault,
    MuteState,
    PlayAudio,
    PlaybackFinished,
    RecordingStopped,
    ShowState,
    StartRecording,
    StopPlayback,
    StopRecording,
)
from companion.contracts.robot import RobotEvent
from companion.expression import Face
from companion.speech.controller import SpeechController
from companion.speech.controller_models import RecognitionInput, SpeechState
from companion.speech.local_wav import read_owned_pcm_wav
from companion.speech.pcm_audio import PcmAudio
from companion.speech.recognition_gate import AudioQualityEvidence

from .conftest import BODY_ID, Recorder

FAST_RECONNECT = {"reconnect_initial_s": 0.05, "reconnect_max_s": 0.2}


def wait_until(predicate: Callable[[], object], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def pcm_ramp(samples: int) -> bytes:
    return struct.pack(f"<{samples}h", *((i * 7) % 65536 - 32768 for i in range(samples)))


# ---- fake sounddevice runtime ----


class FakeInputStream:
    def __init__(self, runtime: "FakeRuntime", **kwargs: object) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.closed = False
        self._runtime = runtime
        self._feeder: threading.Thread | None = None
        runtime.inputs.append(self)

    def start(self) -> None:
        if self._runtime.auto_feed is not None:
            self._feeder = threading.Thread(target=self._auto_feed, daemon=True)
            self._feeder.start()

    def _auto_feed(self) -> None:
        pcm = self._runtime.auto_feed
        block = self.kwargs["blocksize"] * 2
        for offset in range(0, len(pcm), block):
            if self.closed:
                return
            self.feed(pcm[offset : offset + block])

    def feed(self, pcm: bytes) -> None:
        self.callback(pcm, len(pcm) // 2, None, None)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeOutputStream:
    def __init__(self, runtime: "FakeRuntime", **kwargs: object) -> None:
        self.kwargs = kwargs
        self.written: list[bytes] = []
        self.aborted = False
        self.closed = False
        self.abort_count = 0
        self.close_count = 0
        self._runtime = runtime
        runtime.outputs.append(self)

    def start(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        if self._runtime.fail_write:
            raise RuntimeError("device gone")
        time.sleep(self._runtime.write_delay_s)
        self.written.append(bytes(data))

    def stop(self) -> None:
        pass

    def abort(self) -> None:
        self.aborted = True
        self.abort_count += 1

    def close(self) -> None:
        self.closed = True
        self.close_count += 1
        if self._runtime.fail_close:
            raise RuntimeError("synthetic close error")

    def played(self) -> bytes:
        return b"".join(self.written)


class FakeRuntime:
    def __init__(self) -> None:
        self.inputs: list[FakeInputStream] = []
        self.outputs: list[FakeOutputStream] = []
        self.fail_input_open = False
        self.fail_write = False
        self.fail_close = False
        self.write_delay_s = 0.0
        self.auto_feed: bytes | None = None
        self.devices = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 0},
            {"name": "Spk", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 0},
        ]
        self.input_rates = (48000,)
        self.output_rates = (44100,)

    def RawInputStream(self, **kwargs: object) -> FakeInputStream:
        if self.fail_input_open:
            raise RuntimeError("no microphone")
        return FakeInputStream(self, **kwargs)

    def RawOutputStream(self, **kwargs: object) -> FakeOutputStream:
        return FakeOutputStream(self, **kwargs)

    def query_devices(self) -> list[dict]:
        return self.devices

    def query_hostapis(self) -> list[dict]:
        return [{"name": "Core Audio"}]

    def check_input_settings(self, *, samplerate: int, **_: object) -> None:
        if samplerate not in self.input_rates:
            raise ValueError("unsupported rate")

    def check_output_settings(self, *, samplerate: int, **_: object) -> None:
        if samplerate not in self.output_rates:
            raise ValueError("unsupported rate")


class SwitchMute:
    source = "hardware_switch"

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def microphone_enabled(self) -> bool:
        return self.enabled


class FaceLog:
    def __init__(self) -> None:
        self.faces: list[Face] = []

    def show(self, face: Face) -> None:
        self.faces.append(face)


class FailingDisplay(FaceLog):
    def show(self, face: Face) -> None:
        super().show(face)
        raise OSError("display disconnected")


class Body:
    def __init__(
        self,
        port: int,
        pairing: PairingPaths,
        *,
        input_rate: int,
        resend_s: float,
        mute: object = None,
        display: FaceLog | None = None,
    ):
        self.runtime = FakeRuntime()
        self.mute = SwitchMute() if mute is None else mute
        self.display = FaceLog() if display is None else display
        self.core = BodyCore(
            microphone=SoundDeviceMicrophone(self.runtime, 0, input_rate),
            speaker=SoundDeviceSpeaker(self.runtime, 1),
            mute_source=self.mute,
            display=self.display,
            mute_resend_interval_s=resend_s,
        )
        self.client = BodyClient(
            host="127.0.0.1",
            port=port,
            ssl_context=body_ssl_context(pairing.body_dir),
            body_id=BODY_ID,
            output_sample_rates=(44100,),
            on_session_started=self.core.on_session_started,
            on_session_ended=self.core.on_session_ended,
            on_message=self.core.on_message,
            on_frame=self.core.on_frame,
            **FAST_RECONNECT,
        )
        self.core.start(self.client)
        self.client.start()

    def close(self) -> None:
        self.client.close()
        self.core.close()

    def wait_input(self, count: int = 1) -> FakeInputStream:
        wait_until(lambda: len(self.runtime.inputs) >= count)
        return self.runtime.inputs[count - 1]


class FakeBrain:
    """A bare BrainServer that records everything the body sends."""

    def __init__(self, pairing: PairingPaths, port: int = 0) -> None:
        self.rec = Recorder()
        self.server = BrainServer(
            host="127.0.0.1",
            port=port,
            ssl_context=brain_ssl_context(pairing.brain_dir),
            body_id=BODY_ID,
            **self.rec.callbacks(),
        )
        self.server.start()

    @property
    def port(self) -> int:
        assert self.server.port is not None
        return self.server.port

    @property
    def session_id(self) -> str:
        return self.rec.started[-1][0]

    def send(self, cls: type, **fields: object) -> None:
        self.server.send_message(
            cls(protocol_version=1, seq=0, session_id=self.session_id, **fields)
        )

    def send_audio(self, token: int, pcm: bytes, *, step: int = 32000) -> None:
        offsets = list(range(0, len(pcm), step))
        for index, offset in enumerate(offsets):
            frame = AudioFrame(
                direction="downlink",
                stream_id=token,
                frame_index=index,
                is_last=index == len(offsets) - 1,
                payload=pcm[offset : offset + step],
            )
            self.server.send_frame(self.session_id, frame)

    def of(self, kind: type) -> list:
        with self.rec._cond:
            return [m for m in self.rec.messages if isinstance(m, kind)]

    def frames(self) -> list[AudioFrame]:
        with self.rec._cond:
            return list(self.rec.frames)

    def wait(self, kind: type, count: int = 1) -> list:
        self.rec.wait_for(
            lambda: len([m for m in self.rec.messages if isinstance(m, kind)]) >= count
        )
        return self.of(kind)

    def close(self) -> None:
        self.server.close()


@pytest.fixture
def rig(pairing, closer):
    def build(
        *, input_rate: int = 16000, resend_s: float = 1.0, display: FaceLog | None = None,
    ) -> tuple[FakeBrain, Body]:
        brain = FakeBrain(pairing)
        closer(brain)
        body = Body(brain.port, pairing, input_rate=input_rate, resend_s=resend_s, display=display)
        closer(body)
        brain.wait(MuteState)
        return brain, body

    return build


def uplink_bytes(frames: list[AudioFrame], turn_id: int) -> bytes:
    own = [f for f in frames if f.stream_id == turn_id]
    assert [f.frame_index for f in own] == list(range(len(own)))
    return b"".join(f.payload for f in own)


# ---- recording ----


def test_recording_streams_frames_and_reports_stop_command(rig) -> None:
    brain, body = rig()
    brain.send(StartRecording, turn_id=3, max_seconds=30)
    stream = body.wait_input()
    assert stream.kwargs["samplerate"] == 16000
    assert (stream.kwargs["channels"], stream.kwargs["dtype"]) == (1, "int16")
    pcm = pcm_ramp(24_000)
    for offset in range(0, len(pcm), 3200):
        stream.feed(pcm[offset : offset + 3200])
    brain.send(StopRecording, turn_id=3)
    stopped = brain.wait(RecordingStopped)[0]
    assert (stopped.turn_id, stopped.reason, stopped.total_bytes) == (3, "stop_command", len(pcm))
    frames = brain.frames()
    assert uplink_bytes(frames, 3) == pcm
    assert frames[-1].is_last and not any(f.is_last for f in frames[:-1])
    assert all(len(f.payload) <= 32000 for f in frames)
    assert stream.closed


def test_recording_round_trip_through_brain_link(pairing, closer, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    link = BrainBodyLink(
        host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=BODY_ID, run_root=run_root,
    )  # fmt: skip
    closer(link)
    link.start()
    body = Body(link.port, pairing, input_rate=16000, resend_s=1.0)
    closer(body)
    wait_until(link.mute_gate.microphone_enabled)
    session = link.recorder.start()
    pcm = pcm_ramp(20_000)
    body.wait_input().feed(pcm)
    audio = session.stop()
    wav = read_owned_pcm_wav(audio, expected_sample_rate_hz=16000)
    assert struct.pack(f"<{wav.frame_count}h", *wav.samples_int16) == pcm
    audio.close()
    session.close()


def test_48k_microphone_is_resampled_to_16k(rig) -> None:
    brain, body = rig(input_rate=48000)
    brain.send(StartRecording, turn_id=1, max_seconds=30)
    stream = body.wait_input()
    assert stream.kwargs["samplerate"] == 48000
    sine = [round(8000 * math.sin(2 * math.pi * 440 * i / 48000)) for i in range(48000)]
    pcm = struct.pack(f"<{len(sine)}h", *sine)
    for offset in range(0, len(pcm), 9600):
        stream.feed(pcm[offset : offset + 9600])
    brain.send(StopRecording, turn_id=1)
    stopped = brain.wait(RecordingStopped)[0]
    data = uplink_bytes(brain.frames(), 1)
    assert stopped.total_bytes == len(data) == 32000
    samples = struct.unpack(f"<{len(data) // 2}h", data)
    assert 7000 < max(samples) <= 8200


def test_resampler_output_length_is_exact() -> None:
    resampler = Pcm16StreamResampler(48000)
    out = b"".join(resampler.process(pcm_ramp(4801)) for _ in range(3)) + resampler.finish()
    assert len(out) == 2 * round(3 * 4801 / 3)
    passthrough = Pcm16StreamResampler(16000)
    assert passthrough.process(b"\x01\x02") == b"\x01\x02"
    assert passthrough.finish() == b""


def test_max_seconds_stops_with_limit(rig) -> None:
    brain, body = rig()
    brain.send(StartRecording, turn_id=4, max_seconds=1)
    stream = body.wait_input()
    pcm = pcm_ramp(40_000)
    for offset in range(0, len(pcm), 3200):
        stream.feed(pcm[offset : offset + 3200])
    stopped = brain.wait(RecordingStopped)[0]
    assert (stopped.reason, stopped.total_bytes) == ("limit", 32000)
    assert uplink_bytes(brain.frames(), 4) == pcm[:32000]
    assert brain.frames()[-1].is_last
    assert stream.closed


def test_abort_stops_and_discards(rig) -> None:
    brain, body = rig()
    brain.send(StartRecording, turn_id=5, max_seconds=30)
    stream = body.wait_input()
    stream.feed(pcm_ramp(1600))
    brain.send(AbortRecording, turn_id=5)
    stopped = brain.wait(RecordingStopped)[0]
    assert (stopped.turn_id, stopped.reason) == (5, "local_abort")
    assert stream.closed
    stream.feed(pcm_ramp(1600))  # a late audio callback after the abort
    time.sleep(0.2)
    assert not any(f.is_last for f in brain.frames())
    assert sum(len(f.payload) for f in brain.frames()) == stopped.total_bytes


def test_muting_during_recording_stops_with_muted(rig) -> None:
    brain, body = rig()
    brain.send(StartRecording, turn_id=6, max_seconds=30)
    stream = body.wait_input()
    stream.feed(pcm_ramp(1600))
    body.mute.enabled = False
    stopped = brain.wait(RecordingStopped)[0]
    assert (stopped.turn_id, stopped.reason) == (6, "muted")
    assert stream.closed
    brain.rec.wait_for(lambda: any(not m.microphone_enabled for m in brain.of(MuteState)))


def test_start_while_muted_reports_muted_without_opening_microphone(rig) -> None:
    brain, body = rig()
    body.mute.enabled = False
    brain.rec.wait_for(lambda: any(not m.microphone_enabled for m in brain.of(MuteState)))
    brain.send(StartRecording, turn_id=7, max_seconds=30)
    stopped = brain.wait(RecordingStopped)[0]
    assert (stopped.reason, stopped.total_bytes) == ("muted", 0)
    assert body.runtime.inputs == []


def test_microphone_failure_reports_fault(rig) -> None:
    brain, body = rig()
    body.runtime.fail_input_open = True
    brain.send(StartRecording, turn_id=8, max_seconds=30)
    assert brain.wait(Fault)[0].kind == "microphone_unavailable"
    assert brain.wait(RecordingStopped)[0].reason == "local_abort"


# ---- playback ----


@pytest.fixture
def deadline_body(monkeypatch, closer):
    from .test_body_expression import MemoryLink

    clock = SimpleNamespace(now=1000.0)
    runtime = FakeRuntime()
    link = MemoryLink()
    core = BodyCore(
        microphone=SoundDeviceMicrophone(runtime, 0, 16000),
        speaker=SoundDeviceSpeaker(runtime, 1), mute_source=SwitchMute(),
        display=FaceLog(), clock=lambda: clock.now,
    )
    body = SimpleNamespace(core=core, clock=clock, runtime=runtime, link=link, after_handle=None)
    handled = []
    handle = core._handle

    def observed_handle(item):
        handle(item)
        if body.after_handle is not None:
            body.after_handle(item)
        handled.append(item)

    monkeypatch.setattr(core, "_handle", observed_handle)

    def deliver(item):
        if isinstance(item, AudioFrame):
            core.on_frame(item)
        else:
            core.on_message(item)
        wait_until(lambda: any(seen is item for seen in handled))

    body.deliver = deliver
    core.start(link)
    closer(core)
    core.on_session_started("A")
    wait_until(lambda: link.of(MuteState))
    return body


def deadline_play(body, *, total_bytes=4):
    body.deliver(PlayAudio(
        protocol_version=1, seq=0, session_id="A", playback_token=2,
        sample_rate=44100, channels=1, sample_format="pcm_s16le", total_bytes=total_bytes,
    ))
    return body.core._playback


def downlink(index, payload=b"\x01\x00", *, last=False, token=2):
    return AudioFrame(direction="downlink", stream_id=token, frame_index=index,
                      is_last=last, payload=payload)


def heartbeat():
    return BrainHeartbeat(protocol_version=1, seq=0, session_id="A")


def test_missing_last_frame_expires_while_heartbeats_continue(deadline_body) -> None:
    body = deadline_body
    playback = deadline_play(body, total_bytes=2)
    body.deliver(downlink(0))  # all bytes arrived, but no last-frame marker
    for elapsed in (1.0, 2.0, 2.999):
        body.clock.now = 1000.0 + elapsed
        body.deliver(heartbeat())
        assert body.link.of(PlaybackFinished) == []
        assert body.core._playback is playback
    body.clock.now = 1003.0
    body.deliver(heartbeat())
    body.deliver(heartbeat())
    assert [m.result for m in body.link.of(PlaybackFinished)] == ["failed"]
    assert body.core._playback is None
    assert playback.chunks == []
    assert body.runtime.outputs == []


def test_valid_downlink_progress_renews_deadline(deadline_body) -> None:
    body = deadline_body
    playback = deadline_play(body, total_bytes=6)
    body.clock.now = 1002.0
    body.deliver(downlink(0))
    assert playback.receive_deadline == 1005.0
    body.clock.now = 1004.0
    body.deliver(downlink(1, token=99))  # unrelated stream is not progress
    body.deliver(heartbeat())
    assert playback.receive_deadline == 1005.0
    assert body.link.of(PlaybackFinished) == []
    body.clock.now = 1005.0
    # No queued input is needed for expiry on an idle worker round.
    wait_until(lambda: body.link.of(PlaybackFinished))
    assert playback.chunks == []
    assert body.runtime.outputs == []


def test_downlink_timeout_runs_under_continuous_queue_load(deadline_body) -> None:
    body = deadline_body
    playback = deadline_play(body)
    body.deliver(downlink(0))
    rounds = []

    def replenish(item):
        if not isinstance(item, BrainHeartbeat):
            return
        rounds.append(len(body.link.of(PlaybackFinished)))
        if len(rounds) == 5:
            body.clock.now = 1003.0
        if len(rounds) < 100:
            body.core.on_message(heartbeat())

    body.after_handle = replenish
    body.core.on_message(heartbeat())
    wait_until(lambda: len(rounds) == 100)
    assert rounds[:5] == [0] * 5
    assert rounds[5:] == [1] * 95
    assert playback.chunks == []
    assert body.runtime.outputs == []


def test_frame_at_deadline_cannot_revive_playback(deadline_body) -> None:
    body = deadline_body
    playback = deadline_play(body)
    body.deliver(downlink(0))
    body.clock.now = 1003.0
    body.deliver(downlink(1, last=True))
    assert [m.result for m in body.link.of(PlaybackFinished)] == ["failed"]
    assert body.core._playback is None
    assert playback.chunks == []
    assert body.runtime.outputs == []


def test_empty_nonfinal_frames_cannot_accumulate(deadline_body) -> None:
    body = deadline_body
    playback = deadline_play(body)
    body.deliver(downlink(0))
    for index in range(1, 101):
        body.deliver(downlink(index, b""))
    assert [m.result for m in body.link.of(PlaybackFinished)] == ["failed"]
    assert body.core._playback is None
    assert playback.chunks == []
    assert body.runtime.outputs == []


@pytest.mark.parametrize("total_bytes", [2, 4])
def test_empty_final_frame_preserves_completeness_rule(deadline_body, total_bytes) -> None:
    body = deadline_body
    playback = deadline_play(body, total_bytes=total_bytes)
    body.deliver(downlink(0))
    body.clock.now = 1002.0
    body.deliver(downlink(1, b"", last=True))
    wait_until(lambda: body.link.of(PlaybackFinished))
    expected = "completed" if total_bytes == 2 else "failed"
    assert [m.result for m in body.link.of(PlaybackFinished)] == [expected]
    assert playback.receive_deadline == 1003.0
    assert playback.chunks == []
    assert len(body.runtime.outputs) == (1 if total_bytes == 2 else 0)


def test_play_audio_plays_and_reports_completed(rig) -> None:
    brain, body = rig()
    pcm = pcm_ramp(40_000)
    brain.send(PlayAudio, playback_token=2, sample_rate=44100, channels=1,
               sample_format="pcm_s16le", total_bytes=len(pcm))  # fmt: skip
    brain.send_audio(2, pcm)
    finished = brain.wait(PlaybackFinished)[0]
    assert (finished.playback_token, finished.result) == (2, "completed")
    output = body.runtime.outputs[0]
    assert output.kwargs["samplerate"] == 44100
    assert output.played() == pcm
    assert output.closed
    assert (output.abort_count, output.close_count) == (0, 1)


def start_long_playback(brain: FakeBrain, body: Body, token: int) -> FakeOutputStream:
    body.runtime.write_delay_s = 0.01
    pcm = pcm_ramp(441_000)  # ~2 s of fake device writes
    brain.send(PlayAudio, playback_token=token, sample_rate=44100, channels=1,
               sample_format="pcm_s16le", total_bytes=len(pcm))  # fmt: skip
    brain.send_audio(token, pcm)
    wait_until(lambda: body.runtime.outputs and body.runtime.outputs[-1].written)
    return body.runtime.outputs[-1]


def test_stop_playback_reports_stopped(rig) -> None:
    brain, body = rig()
    output = start_long_playback(brain, body, 9)
    brain.send(StopPlayback, playback_token=9)
    finished = brain.wait(PlaybackFinished)[0]
    assert (finished.playback_token, finished.result) == (9, "stopped")
    wait_until(lambda: output.aborted and output.closed, timeout=1.0)
    time.sleep(0.1)
    assert len(brain.of(PlaybackFinished)) == 1
    assert (output.abort_count, output.close_count) == (1, 1)


def test_local_button_during_playback_stops_and_presses(rig) -> None:
    brain, body = rig()
    output = start_long_playback(brain, body, 10)
    body.core.button_pressed()
    brain.wait(ButtonPressed)
    kinds = [type(m) for m in brain.rec.messages if isinstance(m, PlaybackFinished | ButtonPressed)]
    assert kinds == [PlaybackFinished, ButtonPressed]
    assert brain.of(PlaybackFinished)[0].result == "stopped"
    wait_until(lambda: output.aborted, timeout=1.0)
    assert (output.abort_count, output.close_count) == (1, 1)


@pytest.mark.parametrize("invalidate", [None, "disconnect", "reconnect"])
def test_abort_error_sends_content_free_fault(monkeypatch, caplog, capsys, invalidate) -> None:
    from .test_body_devices import OutputRuntime
    from .test_body_expression import MemoryLink

    secret = "SYNTHETIC-ABORT-DEVICE-ERROR-CONTENT"
    runtime = OutputRuntime(block="write", error_at="abort")
    runtime.error = RuntimeError(secret)
    link = MemoryLink()
    core = BodyCore(
        microphone=SoundDeviceMicrophone(FakeRuntime(), 0, 16000),
        speaker=SoundDeviceSpeaker(runtime, 1),
        mute_source=SwitchMute(),
        display=FaceLog(),
    )
    core.start(link)
    core.on_session_started("A")
    core.on_message(PlayAudio(
        protocol_version=1, seq=0, session_id="A", playback_token=1,
        sample_rate=44100, channels=1, sample_format="pcm_s16le", total_bytes=4096,
    ))
    core.on_frame(AudioFrame(
        direction="downlink", stream_id=1, frame_index=0, is_last=True, payload=b"\0" * 4096,
    ))
    try:
        assert runtime.entered.wait(2)
        stream = runtime.streams[0]
        abort = stream.abort

        def abort_and_invalidate() -> None:
            if invalidate is not None:
                core.on_session_ended("A", "synthetic disconnect")
                if invalidate == "reconnect":
                    core.on_session_started("B")
            abort()

        monkeypatch.setattr(stream, "abort", abort_and_invalidate)
        core.on_message(StopPlayback(
            protocol_version=1, seq=0, session_id="A", playback_token=1,
        ))
        wait_until(lambda: stream.calls.count("close") == 1)
        core.button_pressed()
        if invalidate == "reconnect":
            wait_until(lambda: any(m.session_id == "B" for m in link.of(MuteState)))
        elif invalidate is None:
            wait_until(lambda: link.of(ButtonPressed))
        core.close()
        assert stream.calls.count("abort") == stream.calls.count("close") == 1
        faults = link.of(Fault)
        assert len(faults) == (1 if invalidate is None else 0)
        if faults:
            assert (faults[0].session_id, faults[0].kind, faults[0].detail) == (
                "A", "speaker_unavailable", None,
            )
        captured = capsys.readouterr()
        assert secret not in repr(link.messages) + caplog.text + captured.out + captured.err
    finally:
        runtime.released.set()
        core.close()


def test_button_when_idle_only_presses(rig) -> None:
    brain, body = rig()
    body.core.button_pressed()
    brain.wait(ButtonPressed)
    time.sleep(0.1)
    assert brain.of(PlaybackFinished) == []


@pytest.mark.parametrize("stage", ["write", "close"])
def test_speaker_failure_reports_failed_and_fault(rig, stage) -> None:
    brain, body = rig()
    body.runtime.fail_write = stage == "write"
    body.runtime.fail_close = stage == "close"
    pcm = pcm_ramp(100)
    brain.send(PlayAudio, playback_token=11, sample_rate=44100, channels=1,
               sample_format="pcm_s16le", total_bytes=len(pcm))  # fmt: skip
    brain.send_audio(11, pcm)
    assert brain.wait(PlaybackFinished)[0].result == "failed"
    assert brain.wait(Fault)[0].kind == "speaker_unavailable"


def test_stop_while_collecting_frames_does_not_abort(monkeypatch) -> None:
    from .test_body_expression import MemoryBody

    body = MemoryBody()

    def unexpected_abort() -> None:
        pytest.fail("collecting frames must not abort the speaker")

    monkeypatch.setattr(body.speaker, "abort", unexpected_abort)
    try:
        body.core.on_session_started("session-1")
        body.core.on_message(PlayAudio(
            protocol_version=1, seq=0, session_id="session-1", playback_token=1,
            sample_rate=44100, channels=1, sample_format="pcm_s16le", total_bytes=4,
        ))
        body.core.on_message(StopPlayback(
            protocol_version=1, seq=0, session_id="session-1", playback_token=1,
        ))
        body.flush()
        assert [m.result for m in body.link.of(PlaybackFinished)] == ["stopped"]
        assert not body.speaker.started.is_set()
    finally:
        body.close()


def test_short_playback_stream_fails(rig) -> None:
    brain, body = rig()
    pcm = pcm_ramp(100)
    brain.send(PlayAudio, playback_token=12, sample_rate=44100, channels=1,
               sample_format="pcm_s16le", total_bytes=len(pcm) + 2)  # fmt: skip
    brain.send_audio(12, pcm)
    assert brain.wait(PlaybackFinished)[0].result == "failed"
    assert body.runtime.outputs == []


# ---- mute state, display, disconnect ----


def test_mute_state_is_resent_every_interval(rig) -> None:
    brain, _body = rig(resend_s=0.1)
    started = time.monotonic()
    states = brain.wait(MuteState, 5)
    assert 0.3 < time.monotonic() - started < 1.5
    assert all(m.microphone_enabled and m.source == "hardware_switch" for m in states)


def test_development_mute_source_reports_development(pairing, closer) -> None:
    brain = FakeBrain(pairing)
    closer(brain)
    body = Body(
        brain.port, pairing, input_rate=16000, resend_s=1.0,
        mute=body_main.DevelopmentMuteSource(),
    )  # fmt: skip
    closer(body)
    states = brain.wait(MuteState)
    assert states[0].source == "development"
    assert states[0].microphone_enabled


def test_mute_change_is_sent_immediately(rig) -> None:
    brain, body = rig(resend_s=30.0)
    body.mute.enabled = False
    started = time.monotonic()
    brain.rec.wait_for(lambda: any(not m.microphone_enabled for m in brain.of(MuteState)))
    assert time.monotonic() - started < 0.5
    assert len(brain.of(MuteState)) == 2


def test_show_state_drives_terminal_face(rig) -> None:
    brain, body = rig()
    brain.send(ShowState, event=RobotEvent(state="listening"))
    brain.send(ShowState, event=RobotEvent(state="speaking"))
    wait_until(lambda: len(body.display.faces) >= 2)
    assert body.display.faces == [Face.ATTENTIVE, Face.TALKING]


def test_disconnect_stops_everything_and_reconnect_resumes_nothing(
    rig, pairing, closer, monkeypatch,
) -> None:
    brain, body = rig()
    brain.send(StartRecording, turn_id=13, max_seconds=30)
    stream = body.wait_input()
    stream.feed(pcm_ramp(1600))
    output = start_long_playback(brain, body, 14)
    entered = threading.Event()
    release = threading.Event()
    show = body.display.show

    def gated_show(face: Face) -> None:
        show(face)
        entered.set()
        assert release.wait(5), "test did not release display"

    monkeypatch.setattr(body.display, "show", gated_show)
    brain.send(ShowState, event=RobotEvent(state="thinking"))
    assert entered.wait(5)
    monkeypatch.setattr(body.display, "show", show)
    port = brain.port
    try:
        brain.send(StartRecording, turn_id=15, max_seconds=30)
        brain.send(PlayAudio, playback_token=16, sample_rate=44100, channels=1,
                   sample_format="pcm_s16le", total_bytes=20)  # fmt: skip
        brain.send_audio(16, pcm_ramp(10))
        stream.feed(pcm_ramp(1600))
        with body.core._condition:
            assert body.core._condition.wait_for(lambda: len(body.core._queue) >= 4, timeout=5)
        brain.close()
    finally:
        release.set()
    wait_until(lambda: stream.closed and output.aborted)
    wait_until(lambda: body.display.faces[-2:] == [Face.MUTED, Face.NEUTRAL])
    stream.feed(pcm_ramp(1600))  # late callback from the old recording
    second = FakeBrain(pairing, port)
    closer(second)
    second.wait(MuteState)
    second.send(StopRecording, turn_id=13)
    second.send(StopPlayback, playback_token=14)
    time.sleep(0.3)
    assert second.frames() == []
    assert second.of(RecordingStopped) == []
    assert second.of(PlaybackFinished) == []
    assert len(body.runtime.inputs) == 1
    assert len(body.runtime.outputs) == 1
    assert (output.abort_count, output.close_count) == (1, 1)


def test_disconnect_display_failure_keeps_worker_alive(rig, pairing, closer) -> None:
    brain, body = rig(display=FailingDisplay())
    worker = next(thread for thread in threading.enumerate() if thread.name == "body-core")
    port = brain.port
    brain.close()
    wait_until(lambda: body.display.faces)
    second = FakeBrain(pairing, port)
    closer(second)
    assert second.wait(MuteState)[0].microphone_enabled
    assert worker.is_alive()
    assert body.display.faces == [Face.MUTED, Face.NEUTRAL]

    second.send(StartRecording, turn_id=15, max_seconds=30)
    pcm = pcm_ramp(1600)
    body.wait_input().feed(pcm)
    second.send(StopRecording, turn_id=15)
    stopped = second.wait(RecordingStopped)[0]
    assert (stopped.turn_id, stopped.reason, stopped.total_bytes) == (15, "stop_command", len(pcm))
    assert uplink_bytes(second.frames(), 15) == pcm


def test_reconnect_after_display_failure_accepts_playback(rig, pairing, closer) -> None:
    brain, body = rig(display=FailingDisplay())
    port = brain.port
    brain.close()
    wait_until(lambda: body.display.faces)
    second = FakeBrain(pairing, port)
    closer(second)
    second.wait(MuteState)
    pcm = pcm_ramp(1600)
    second.send(PlayAudio, playback_token=16, sample_rate=44100, channels=1,
                sample_format="pcm_s16le", total_bytes=len(pcm))  # fmt: skip
    second.send_audio(16, pcm)
    finished = second.wait(PlaybackFinished)[0]
    assert (finished.playback_token, finished.result) == (16, "completed")
    assert body.runtime.outputs[0].played() == pcm
    assert body.runtime.outputs[0].closed


# ---- terminal button, devices, self-test ----


def test_terminal_button_source_reads_space_and_q() -> None:
    read_fd, write_fd = os.pipe()
    presses: list[int] = []
    quit_event = threading.Event()
    source = TerminalButtonSource(read_fd, on_quit=quit_event.set)
    source.start(lambda: presses.append(1))
    os.write(write_fd, b" x Q")
    assert quit_event.wait(2.0)
    source.close()
    assert presses == [1, 1]
    os.close(read_fd)
    os.close(write_fd)


def test_resolve_body_devices_by_name_and_host_api() -> None:
    runtime = FakeRuntime()
    pair = resolve_body_devices(runtime, "Mic", "Spk", "Core Audio")
    assert (pair.input_index, pair.output_index) == (0, 1)
    assert (pair.input_sample_rate_hz, pair.output_sample_rate_hz) == (48000, 44100)
    with pytest.raises(DeviceNotFoundError):
        resolve_body_devices(runtime, "Nope", "Spk", None)


def test_self_test_records_one_second_and_plays_tone() -> None:
    runtime = FakeRuntime()
    runtime.auto_feed = pcm_ramp(48000 * 2)
    lines: list[str] = []
    run_self_test(
        SoundDeviceMicrophone(runtime, 0, 48000), SoundDeviceSpeaker(runtime, 1), 44100,
        lines.append,
    )  # fmt: skip
    assert runtime.inputs[0].closed
    assert "32000 字节" in lines[0]
    played = runtime.outputs[0].played()
    assert len(played) == 2 * 22050
    assert runtime.outputs[0].kwargs["samplerate"] == 44100
    assert (runtime.outputs[0].abort_count, runtime.outputs[0].close_count) == (0, 1)


@pytest.mark.parametrize("button", ["terminal", "none"])
def test_main_self_test_uses_configured_devices(monkeypatch, capsys, button) -> None:
    runtime = FakeRuntime()
    runtime.auto_feed = pcm_ramp(48000 * 2)
    monkeypatch.setattr(body_main, "load_sounddevice_runtime", lambda: runtime)

    def unexpected(*args, **kwargs):
        pytest.fail("self-test must not connect, start buttons or wait for signals")

    for name in ("BodyClient", "TerminalButtonSource", "NoButtonSource", "_sigterm_quit"):
        monkeypatch.setattr(body_main, name, unexpected)
    assert body_main.main([
        "--input-device", "Mic", "--output-device", "Spk", "--self-test", "--button", button,
    ]) == 0
    assert "自检录音" in capsys.readouterr().out
    assert body_main.main(["--input-device", "Nope", "--output-device", "Spk", "--self-test"]) == 2


def test_main_requires_brain_settings_without_self_test(capsys) -> None:
    with pytest.raises(SystemExit):
        body_main.main(["--input-device", "Mic", "--output-device", "Spk"])


MAIN_ARGS = [
    "--input-device", "Mic", "--output-device", "Spk", "--brain-host", "synthetic",
    "--brain-port", "1234", "--body-id", "synthetic", "--cert-dir", "synthetic-certs",
]


@pytest.fixture
def main_rig(monkeypatch):
    calls = []
    state = SimpleNamespace(calls=calls, start_error=None, close_error=None, on_wait=None)

    class QuitEvent(threading.Event):
        def wait(self, timeout=None):
            calls.append("wait")
            assert calls[:3] == ["core.start", "client.start", "button.start"]
            if state.on_wait is not None:
                state.on_wait()
            else:
                self.set()
            assert self.is_set()
            return True

    quit_event = QuitEvent()

    class Core:
        def __init__(self, **kwargs):
            state.core = self

        def start(self, client):
            assert client is state.client
            calls.append("core.start")

        def close(self):
            calls.append("core.close")
            if state.close_error == "core":
                raise RuntimeError("core close failed")

        def button_pressed(self):
            pytest.fail("no automatic button press")

        def on_session_started(self, session_id):
            pass

        def on_session_ended(self, session_id):
            pass

        def on_message(self, message):
            pass

        def on_frame(self, frame):
            pass

    class Client:
        def __init__(self, **kwargs):
            state.client = self
            state.callbacks = kwargs

        def start(self):
            calls.append("client.start")
            if state.start_error is not None:
                raise state.start_error

        def close(self):
            calls.append("client.close")
            if state.close_error == "client":
                raise RuntimeError("client close failed")

    class Button:
        def __init__(self, fd, *, on_quit):
            assert fd == 123

        def start(self, on_press):
            assert on_press == state.core.button_pressed
            calls.append("button.start")

        def close(self):
            calls.append("button.close")
            if state.close_error == "button":
                raise RuntimeError("button close failed")

    @contextmanager
    def terminal_mode(fd):
        assert fd == 123
        state.terminal_active = True
        try:
            yield
        finally:
            state.terminal_active = False

    state.terminal_active = False
    monkeypatch.setattr(body_main, "load_sounddevice_runtime", FakeRuntime)
    monkeypatch.setattr(body_main, "BodyCore", Core)
    monkeypatch.setattr(body_main, "BodyClient", Client)
    monkeypatch.setattr(body_main, "TerminalButtonSource", Button)
    monkeypatch.setattr(body_main, "body_ssl_context", lambda path: "synthetic TLS")
    monkeypatch.setattr(body_main, "_cbreak", terminal_mode)
    monkeypatch.setattr(body_main, "threading", SimpleNamespace(Event=lambda: quit_event))
    monkeypatch.setattr(body_main.sys, "stdin", SimpleNamespace(
        isatty=lambda: False, fileno=lambda: 123,
    ))

    # Exercise the real no-button implementation; observe only its lifecycle calls.
    actual_no_button = body_main.NoButtonSource

    class ObservedNoButton(actual_no_button):
        def start(self, on_press):
            calls.append("button.start")
            super().start(on_press)

        def close(self):
            calls.append("button.close")
            super().close()

    monkeypatch.setattr(body_main, "NoButtonSource", ObservedNoButton)
    return state


def test_headless_main_runs_without_tty(main_rig, capsys) -> None:
    assert body_main.main([*MAIN_ARGS, "--button", "none"]) == 0
    assert main_rig.calls == [
        "core.start", "client.start", "button.start", "wait",
        "button.close", "client.close", "core.close",
    ]
    assert main_rig.callbacks["on_message"] == main_rig.core.on_message
    assert main_rig.callbacks["on_frame"] == main_rig.core.on_frame
    assert not main_rig.terminal_active
    output = capsys.readouterr().out
    assert "无按键模式：等待大脑连接；未接物理按键，不能用物理按键发起对话" in output  # noqa: RUF001
    assert "开发模式：未接物理断麦开关" in output  # noqa: RUF001


def test_headless_mode_never_reads_stdin(main_rig, monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("headless mode must not access stdin or terminal mode")

    monkeypatch.setattr(body_main.sys, "stdin", SimpleNamespace(
        isatty=unexpected, fileno=unexpected, read=unexpected,
    ))
    monkeypatch.setattr(body_main, "_cbreak", unexpected)
    monkeypatch.setattr(body_main, "TerminalButtonSource", unexpected)
    assert body_main.main([*MAIN_ARGS, "--button", "none"]) == 0


@pytest.mark.parametrize("button", ["terminal", "none"])
def test_sigterm_closes_body_resources(main_rig, monkeypatch, button) -> None:
    monkeypatch.setattr(body_main.sys.stdin, "isatty", lambda: True)
    original = object()
    handlers = []

    def register(signum, handler):
        assert signum == signal.SIGTERM
        handlers.append(handler)

    def on_wait():
        assert len(handlers) == 1
        assert main_rig.terminal_active == (button == "terminal")
        handlers[0](signal.SIGTERM, None)
        assert main_rig.calls[-1] == "wait"  # Handler only requests exit.

    monkeypatch.setattr(signal, "getsignal", lambda signum: original)
    monkeypatch.setattr(signal, "signal", register)
    main_rig.on_wait = on_wait
    assert body_main.main([*MAIN_ARGS, "--button", button]) == 0
    assert main_rig.calls == [
        "core.start", "client.start", "button.start", "wait",
        "button.close", "client.close", "core.close",
    ]
    assert handlers[1] is original
    assert len(handlers) == 2
    assert not main_rig.terminal_active


def test_terminal_mode_still_requires_tty(main_rig, capsys) -> None:
    assert body_main.main([*MAIN_ARGS, "--button", "terminal"]) == 2
    assert capsys.readouterr().err == "需要在终端里运行（空格键=按键，Q=退出）\n"  # noqa: RUF001
    assert main_rig.calls == []


def test_button_defaults_to_terminal() -> None:
    assert body_main._parse_args(MAIN_ARGS).button == "terminal"


def test_invalid_button_source_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        body_main._parse_args([*MAIN_ARGS, "--button", "gpio"])
    assert exc.value.code == 2


def test_headless_start_failure_still_closes_started_resources(main_rig) -> None:
    error = RuntimeError("synthetic client start failed")
    main_rig.start_error = error
    previous_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError) as exc:
        body_main.main([*MAIN_ARGS, "--button", "none"])
    assert exc.value is error
    assert main_rig.calls == [
        "core.start", "client.start", "button.close", "client.close", "core.close",
    ]
    assert signal.getsignal(signal.SIGTERM) is previous_handler


@pytest.mark.parametrize("resource", ["button", "client", "core"])
def test_main_close_failure_still_closes_remaining_resources(main_rig, monkeypatch, resource):
    monkeypatch.setattr(body_main.sys.stdin, "isatty", lambda: True)
    main_rig.close_error = resource
    previous_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError, match=f"{resource} close failed"):
        body_main.main([*MAIN_ARGS, "--button", "terminal"])
    assert main_rig.calls[-3:] == ["button.close", "client.close", "core.close"]
    assert not main_rig.terminal_active
    assert signal.getsignal(signal.SIGTERM) is previous_handler


@pytest.mark.parametrize("button", ["terminal", "none"])
def test_keyboard_interrupt_closes_body_resources(main_rig, monkeypatch, button):
    monkeypatch.setattr(body_main.sys.stdin, "isatty", lambda: True)

    def interrupt():
        raise KeyboardInterrupt

    main_rig.on_wait = interrupt
    assert body_main.main([*MAIN_ARGS, "--button", button]) == 0
    assert main_rig.calls[-3:] == ["button.close", "client.close", "core.close"]
    assert not main_rig.terminal_active


def test_main_terminal_wires_abortable_speaker(monkeypatch) -> None:
    runtime = FakeRuntime()
    constructed: list[BodyCore] = []
    clients: list[object] = []
    actual_core = BodyCore

    def make_core(**kwargs):
        assert isinstance(kwargs["speaker"], SoundDeviceSpeaker)
        kwargs["speaker"].abort()
        core = actual_core(**kwargs)
        constructed.append(core)
        return core

    class Client:
        def __init__(self, **kwargs):
            self.callbacks = kwargs
            self.started = False
            self.closed = False
            clients.append(self)

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

    class Button:
        def __init__(self, fd, *, on_quit):
            assert fd == 123
            self.on_quit = on_quit

        def start(self, on_press):
            assert on_press == constructed[0].button_pressed
            self.on_quit()

        def close(self):
            pass

    monkeypatch.setattr(body_main, "load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(body_main, "BodyCore", make_core)
    monkeypatch.setattr(body_main, "BodyClient", Client)
    monkeypatch.setattr(body_main, "body_ssl_context", lambda path: "synthetic TLS")
    monkeypatch.setattr(body_main, "TerminalButtonSource", Button)
    monkeypatch.setattr(body_main, "_cbreak", lambda fd: nullcontext())
    monkeypatch.setattr(body_main.sys, "stdin", SimpleNamespace(
        isatty=lambda: True, fileno=lambda: 123,
    ))
    assert body_main.main([
        "--input-device", "Mic", "--output-device", "Spk", "--brain-host", "synthetic",
        "--brain-port", "1234", "--body-id", "synthetic", "--cert-dir", "synthetic-certs",
    ]) == 0
    assert len(constructed) == len(clients) == 1
    assert clients[0].started and clients[0].closed
    assert clients[0].callbacks["on_message"] == constructed[0].on_message
    assert runtime.outputs == []


# ---- end to end with the real SpeechController ----


class CheckingRecognizer:
    def __init__(self, expected_pcm: bytes) -> None:
        self.expected_pcm = expected_pcm
        self.matched = False

    def recognize(self, audio: object) -> RecognitionInput:
        wav = read_owned_pcm_wav(audio, expected_sample_rate_hz=16000)  # type: ignore[arg-type]
        self.matched = struct.pack(f"<{wav.frame_count}h", *wav.samples_int16) == self.expected_pcm
        return RecognitionInput(
            audio=AudioQualityEvidence(
                sample_rate_hz=16_000, channels=1, sample_width_bytes=2, frame_count=4_800,
                rms=0.1, peak_abs=0.2, clipping_fraction=0.0, estimated_snr_db=20.0,
            ),
            vad_passed=True,
            text="synthetic transcript",
            confidence=None,
            confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
            confidence_threshold=None,
        )  # fmt: skip


class FakeDialogue:
    def reply(self, *, user_id: str, text: str) -> DialogueResponse:
        return DialogueResponse(
            text="synthetic safe response", disclosed_ai_identity=False, billable=False,
            blocked_reason=None,
        )  # fmt: skip


SYNTH_PCM = pcm_ramp(4410)


class FakeSynthesizer:
    def synthesize(self, text: str) -> PcmAudio:
        return PcmAudio(pcm=SYNTH_PCM, sample_rate_hz=44100)

    def close(self) -> None:
        pass


class MonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()


def test_full_turn_with_brain_link_controller_and_body_core(pairing, closer, tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    link = BrainBodyLink(
        host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=BODY_ID, run_root=run_root,
    )  # fmt: skip
    closer(link)
    link.start()
    body = Body(link.port, pairing, input_rate=16000, resend_s=1.0)
    closer(body)
    spoken = pcm_ramp(16_000)
    recognizer = CheckingRecognizer(spoken)
    controller = SpeechController(
        trusted_user_id="owner",
        recorder=link.recorder,
        recognizer=recognizer,
        dialogue=FakeDialogue(),
        synthesizer=FakeSynthesizer(),
        player=link.player,
        mute_gate=link.mute_gate,
        clock=MonotonicClock(),
        session_id_supplier=lambda: link.current_session_id,
        session_playback_starter=link.player.start_for_session,
    )

    def pump() -> None:
        for item in link.drain_body_inputs():
            if isinstance(item, ButtonPressed):
                controller.button_pressed()
            else:
                controller.mute_changed()
        for event in link.player.drain_events():
            if event.succeeded:
                controller.playback_finished(event.playback_token)
            else:
                controller.playback_failed(event.playback_token)
        controller.tick()

    wait_until(lambda: pump() or link.mute_gate.microphone_enabled())
    body.core.button_pressed()
    wait_until(lambda: pump() or controller.state is SpeechState.RECORDING)
    stream = body.wait_input()
    for offset in range(0, len(spoken), 3200):
        stream.feed(spoken[offset : offset + 3200])
    body.core.button_pressed()
    wait_until(lambda: pump() or controller.state is not SpeechState.RECORDING)
    wait_until(lambda: pump() or controller.state is SpeechState.IDLE)
    assert controller.last_issue is None
    assert recognizer.matched
    assert body.runtime.outputs[0].played() == SYNTH_PCM
    assert list(run_root.iterdir()) == []
    controller.shutdown()
