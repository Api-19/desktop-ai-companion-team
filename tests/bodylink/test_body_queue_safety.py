"""Synthetic device gates exercise the real worker and actual BodyLink send calls."""

import threading
from collections.abc import Callable

import pytest

from companion.bodylink.body_main import BodyCore
from companion.contracts.body import (
    AudioFrame,
    ButtonPressed,
    MuteState,
    PlayAudio,
    PlaybackFinished,
    RecordingStopped,
    ShowState,
    StartRecording,
    StopRecording,
)
from companion.contracts.robot import RobotEvent
from companion.expression import Face

from .test_body_expression import GatedSpeaker, MemoryLink
from .test_body_main import FaceLog, SwitchMute, pcm_ramp, wait_until


class Input:
    def __init__(self, callback: Callable[[bytes], None]) -> None:
        self.callback = callback
        self.closes = 0

    def close(self) -> None:
        self.closes += 1


class Microphone:
    sample_rate = 16000

    def __init__(self) -> None:
        self.inputs: list[Input] = []

    def open(self, on_audio: Callable[[bytes], None]) -> Input:
        stream = Input(on_audio)
        self.inputs.append(stream)
        return stream


class TrackingSpeaker(GatedSpeaker):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.exits = 0

    def play(self, pcm: bytes, sample_rate: int, stop: threading.Event) -> bool:
        self.calls += 1
        result = super().play(pcm, sample_rate, stop)
        self.exits += 1
        return result


class QueueBody:
    def __init__(self) -> None:
        self.microphone = Microphone()
        self.speaker = TrackingSpeaker()
        self.display = FaceLog()
        self.link = MemoryLink()
        self.core = BodyCore(
            microphone=self.microphone,
            speaker=self.speaker,
            mute_source=SwitchMute(),
            display=self.display,
        )
        self.release = threading.Event()
        self.core.start(self.link)
        self.core.on_session_started("A")
        wait_until(lambda: self.link.of(MuteState))

    def send(self, kind: type, *, session_id: str = "A", **fields: object) -> None:
        self.core.on_message(kind(protocol_version=1, seq=0, session_id=session_id, **fields))

    def record(self) -> Input:
        self.send(StartRecording, turn_id=1, max_seconds=30)
        wait_until(lambda: self.microphone.inputs)
        return self.microphone.inputs[-1]

    def play(self, *, session_id: str = "A", token: int = 2) -> None:
        self.send(
            PlayAudio, session_id=session_id, playback_token=token, sample_rate=44100,
            channels=1, sample_format="pcm_s16le", total_bytes=20,
        )
        self.core.on_frame(self.frame(token))

    @staticmethod
    def frame(token: int = 2) -> AudioFrame:
        return AudioFrame(
            direction="downlink", stream_id=token, frame_index=0, is_last=True,
            payload=pcm_ramp(10),
        )

    def pause(self, monkeypatch) -> None:
        """Block inside a display call, outside the queue/session condition."""
        entered = threading.Event()
        show = self.display.show

        def gated_show(face) -> None:
            show(face)
            entered.set()
            assert self.release.wait(5), "test did not release worker"

        monkeypatch.setattr(self.display, "show", gated_show)
        self.send(ShowState, event=RobotEvent(state="thinking"))
        assert entered.wait(5)
        monkeypatch.setattr(self.display, "show", show)

    def flush(self) -> None:
        count = len(self.link.of(ButtonPressed))
        self.core.button_pressed()
        wait_until(lambda: len(self.link.of(ButtonPressed)) == count + 1)

    def close(self) -> None:
        self.release.set()
        self.speaker.finish.set()
        self.core.close()


@pytest.fixture
def body(closer) -> QueueBody:
    body = QueueBody()
    closer(body)
    return body


def test_disconnect_preempts_queued_audio(body, monkeypatch) -> None:
    stream = body.record()
    body.pause(monkeypatch)
    body.play()
    stream.callback(pcm_ramp(20))
    body.core.on_session_ended("A", "disconnected")
    body.release.set()
    wait_until(lambda: stream.closes == 1)
    body.core.close()
    assert not body.speaker.started.is_set()
    assert body.link.frames == []
    assert body.link.of(PlaybackFinished) == []
    assert body.link.of(RecordingStopped) == []


def test_queue_overflow_aborts_without_growing(body, monkeypatch) -> None:
    stream = body.record()
    pcm = pcm_ramp(20)
    stream.callback(pcm)
    wait_until(lambda: body.link.frames)
    body.play()
    assert body.speaker.started.wait(5)
    body.pause(monkeypatch)
    sizes = []
    produced = threading.Event()

    def flood() -> None:
        for _ in range(100):
            body.core.on_frame(body.frame())
            with body.core._condition:
                sizes.append(len(body.core._queue))
        produced.set()

    producer = threading.Thread(target=flood)
    producer.start()
    try:
        assert produced.wait(1), "producer waited for queue space"
        assert body.speaker.stop.is_set()
        stream.callback(pcm)
        assert body.link.of(RecordingStopped) == []
        assert body.link.of(PlaybackFinished) == []
    finally:
        body.release.set()
        producer.join(5)
    assert max(sizes) == 64
    wait_until(lambda: body.link.of(RecordingStopped) and body.link.of(PlaybackFinished))
    with body.core._condition:
        assert body.core._condition.wait_for(
            lambda: body.core._overflow_generation is None, timeout=5
        )
    body.flush()
    stopped, = body.link.of(RecordingStopped)
    finished, = body.link.of(PlaybackFinished)
    assert (stopped.session_id, stopped.reason, stopped.total_bytes) == (
        "A", "local_abort", len(pcm)
    )
    assert (finished.session_id, finished.result) == ("A", "failed")
    assert all(not frame.is_last for _, frame in body.link.frames)
    assert b"".join(frame.payload for _, frame in body.link.frames) == pcm
    assert stream.closes == 1
    body.play(token=3)
    wait_until(lambda: body.speaker.stop is not None and not body.speaker.stop.is_set())
    body.speaker.finish.set()
    wait_until(lambda: len(body.link.of(PlaybackFinished)) == 2)
    assert body.link.of(PlaybackFinished)[-1].result == "completed"


def test_old_session_commands_never_open_devices(body, monkeypatch) -> None:
    body.pause(monkeypatch)
    body.send(StartRecording, turn_id=1, max_seconds=30)
    body.play()
    body.core.button_pressed()
    body.core.on_session_started("B")
    body.send(StartRecording, session_id="A", turn_id=9, max_seconds=30)
    body.send(ShowState, session_id="B", event=RobotEvent(state="listening"))
    body.release.set()
    wait_until(lambda: any(m.session_id == "B" for m in body.link.of(MuteState)))
    # B's queued work survives the generation cleanup; A's work has no device/result effect.
    wait_until(lambda: body.display.faces[-1] == Face.ATTENTIVE)
    assert body.microphone.inputs == []
    assert not body.speaker.started.is_set()
    assert body.link.of(ButtonPressed) == []
    assert body.link.of(RecordingStopped) == []
    assert body.link.of(PlaybackFinished) == []


@pytest.mark.parametrize("next_session", ["B", "A"])
def test_late_device_callbacks_keep_original_session(body, monkeypatch, next_session) -> None:
    stream = body.record()
    entered = threading.Event()
    release_callback = threading.Event()
    enqueue = body.core._enqueue

    def gated_enqueue(payload, *, session_id, generation) -> None:
        if isinstance(payload, tuple) and payload[0] == "played":
            entered.set()
            assert release_callback.wait(5)
        enqueue(payload, session_id=session_id, generation=generation)

    monkeypatch.setattr(body.core, "_enqueue", gated_enqueue)
    body.play()
    assert body.speaker.started.wait(5)
    body.speaker.finish.set()
    assert entered.wait(5)
    previous_mutes = len(body.link.of(MuteState))
    try:
        body.core.on_session_ended("A", "disconnected")
        body.core.on_session_started(next_session)
        stream.callback(pcm_ramp(20))
    finally:
        release_callback.set()
    wait_until(lambda: len(body.link.of(MuteState)) == previous_mutes + 1)
    assert body.link.of(MuteState)[-1].session_id == next_session
    body.flush()
    assert body.link.frames == []
    assert body.link.of(PlaybackFinished) == []
    assert body.link.of(RecordingStopped) == []
    assert stream.closes == 1


def test_stale_disconnect_does_not_invalidate_current_session(body) -> None:
    body.core.on_session_started("B")
    wait_until(lambda: any(m.session_id == "B" for m in body.link.of(MuteState)))
    stream_count = len(body.microphone.inputs)
    body.send(StartRecording, session_id="B", turn_id=3, max_seconds=30)
    wait_until(lambda: len(body.microphone.inputs) == stream_count + 1)
    stream = body.microphone.inputs[-1]
    body.core.on_session_ended("A", "late disconnect")
    body.core.on_session_started("B")  # duplicate start must preserve B's resources/queue
    stream.callback(pcm_ramp(10))
    body.flush()
    assert stream.closes == 0
    assert [session_id for session_id, _ in body.link.frames] == ["B"]
    body.core.on_session_ended("B", "disconnect")
    body.core.on_session_ended("B", "duplicate")
    wait_until(lambda: stream.closes == 1)


def test_collapsed_disconnect_and_reconnect_still_clear_old_resources(body, monkeypatch) -> None:
    stream = body.record()
    body.play()
    assert body.speaker.started.wait(5)
    old_stop = body.speaker.stop
    body.pause(monkeypatch)
    body.core.on_session_ended("A", "disconnected")
    body.core.on_session_started("B")
    assert old_stop.is_set()
    body.send(StartRecording, session_id="B", turn_id=3, max_seconds=30)
    body.release.set()
    wait_until(lambda: len(body.microphone.inputs) == 2)
    body.flush()
    assert stream.closes == 1
    assert body.microphone.inputs[-1].closes == 0
    assert body.link.of(PlaybackFinished) == []
    assert body.link.of(RecordingStopped) == []


def test_close_preempts_full_queue(body, monkeypatch) -> None:
    stream = body.record()
    body.play()
    assert body.speaker.started.wait(5)
    body.pause(monkeypatch)
    for _ in range(64):
        body.send(StartRecording, turn_id=4, max_seconds=30)
    worker = body.core._thread
    closed = threading.Event()

    def close() -> None:
        body.core.close()
        closed.set()

    closing = threading.Thread(target=close)
    closing.start()
    try:
        wait_until(lambda: body.speaker.stop.is_set())
        assert not closed.is_set()  # still waiting for the in-flight display operation
        with body.core._condition:
            assert len(body.core._queue) == 64
    finally:
        body.release.set()
        closing.join(5)
    assert closed.is_set()
    assert not worker.is_alive()
    assert body.core._thread is None
    assert stream.closes == 1
    assert len(body.microphone.inputs) == 1
    assert (body.speaker.calls, body.speaker.exits) == (1, 1)
    assert body.link.of(PlaybackFinished) == []
    assert body.link.of(RecordingStopped) == []
    body.core.close()
    assert stream.closes == 1


@pytest.mark.parametrize("path", ["control", "pcm", "recording_stop"])
def test_invalidated_body_events_send_no_old_audio(body, monkeypatch, path) -> None:
    entered = threading.Event()
    release = threading.Event()
    stream = body.record() if path != "control" else None
    method = "_send_frame" if path == "pcm" else "_send"
    send = getattr(body.core, method)

    def gated_send(*args, **kwargs) -> None:
        if path == "pcm" or args[0] in (ButtonPressed, RecordingStopped):
            entered.set()
            assert release.wait(5)
        send(*args, **kwargs)

    monkeypatch.setattr(body.core, method, gated_send)
    if path == "pcm":
        stream.callback(pcm_ramp(20))
    elif path == "recording_stop":
        body.send(StopRecording, turn_id=1)
    else:
        body.core.button_pressed()
    assert entered.wait(5)
    messages_before = list(body.link.messages)
    frames_before = list(body.link.frames)
    body.core.on_session_ended("A", "disconnected")
    body.core.on_session_started("B")
    release.set()
    wait_until(lambda: any(m.session_id == "B" for m in body.link.of(MuteState)))
    body.core.close()
    assert body.link.frames == frames_before
    assert body.link.messages[len(messages_before):] == [body.link.of(MuteState)[-1]]
