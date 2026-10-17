import threading

import pytest

from companion.bodylink.body_devices import SoundDeviceMicrophone
from companion.bodylink.body_main import BodyCore
from companion.contracts.body import (
    AudioFrame,
    ButtonPressed,
    MuteState,
    PlayAudio,
    PlaybackFinished,
    ShowState,
)
from companion.contracts.robot import RobotEvent
from companion.expression import Face

from .test_body_main import FaceLog, FakeRuntime, SwitchMute, pcm_ramp, wait_until


class MemoryLink:
    def __init__(self, *, hold_mute: bool = False) -> None:
        self.messages: list[object] = []
        self.frames: list[tuple[str, AudioFrame]] = []
        self.hold_mute = hold_mute
        self.mute_blocked = threading.Event()
        self.release_mute = threading.Event()

    def send_message(self, message: object) -> None:
        self.messages.append(message)
        if isinstance(message, MuteState) and self.hold_mute:
            self.hold_mute = False
            self.mute_blocked.set()
            assert self.release_mute.wait(5.0), "test did not release mute send"

    def send_frame(self, session_id: str, frame: AudioFrame) -> None:
        self.frames.append((session_id, frame))

    def of(self, kind: type) -> list:
        return [message for message in self.messages if isinstance(message, kind)]


class GatedSpeaker:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.finish = threading.Event()
        self.completed = threading.Event()
        self.pcm = b""
        self.sample_rate: int | None = None
        self.stop: threading.Event | None = None

    def abort(self) -> None:
        if self.stop is not None:
            self.stop.set()

    def play(self, pcm: bytes, sample_rate: int, stop: threading.Event) -> bool:
        self.pcm = pcm
        self.sample_rate = sample_rate
        self.stop = stop
        self.started.set()
        while not self.finish.wait(0.01):
            if stop.is_set():
                return False
        if stop.is_set():
            return False
        self.completed.set()
        return True


class MemoryBody:
    def __init__(self, *, enabled: bool = True, hold_mute: bool = False) -> None:
        self.mute = SwitchMute(enabled)
        self.display = FaceLog()
        self.speaker = GatedSpeaker()
        self.link = MemoryLink(hold_mute=hold_mute)
        self.core = BodyCore(
            microphone=SoundDeviceMicrophone(FakeRuntime(), 0, 16000),
            speaker=self.speaker,
            mute_source=self.mute,
            display=self.display,
        )
        self.core.start(self.link)

    def show(self, state: str, *, session_id: str = "session-1") -> None:
        self.core.on_message(
            ShowState(
                protocol_version=1, seq=0, session_id=session_id, event=RobotEvent(state=state)
            )
        )

    def flush(self) -> None:
        """A button acknowledgement follows all previously queued inputs in an active session."""
        count = len(self.link.of(ButtonPressed))
        self.core.button_pressed()
        wait_until(lambda: len(self.link.of(ButtonPressed)) == count + 1)

    def close(self) -> None:
        self.link.release_mute.set()
        self.speaker.finish.set()
        self.core.close()


@pytest.fixture
def memory_body(closer):
    def build(**kwargs) -> MemoryBody:
        body = MemoryBody(**kwargs)
        closer(body)
        return body

    return build


def test_initially_closed_microphone_shows_muted_without_session(memory_body) -> None:
    body = memory_body(enabled=False)
    wait_until(lambda: body.display.faces)
    body.core.close()
    assert body.display.faces == [Face.MUTED]
    assert body.link.messages == []


@pytest.mark.parametrize("state", ["idle", "listening", "thinking", "speaking", "muted", "error"])
def test_local_mute_overrides_remote_state(memory_body, state: str) -> None:
    body = memory_body(hold_mute=True)
    body.core.on_session_started("session-1")
    assert body.link.mute_blocked.wait(5.0)
    # The worker has cached True and is finishing its poll. Queue ShowState immediately
    # after switching off, before another periodic poll can refresh that cache.
    body.mute.enabled = False
    body.show(state)
    body.show(state)
    body.link.release_mute.set()
    body.flush()
    assert body.display.faces == [Face.MUTED]
    assert [message.microphone_enabled for message in body.link.of(MuteState)] == [True, False]


def test_unmute_does_not_replay_old_expression(memory_body) -> None:
    body = memory_body()
    body.core.on_session_started("session-1")
    body.show("listening")
    body.flush()
    assert body.display.faces == [Face.ATTENTIVE]

    body.mute.enabled = False
    wait_until(lambda: body.display.faces[-1] == Face.MUTED)
    body.show("speaking")
    body.flush()
    assert body.display.faces == [Face.ATTENTIVE, Face.MUTED]

    body.mute.enabled = True
    wait_until(lambda: body.display.faces[-1] == Face.NEUTRAL)
    body.flush()
    assert body.display.faces == [Face.ATTENTIVE, Face.MUTED, Face.NEUTRAL]

    body.show("speaking")
    body.flush()
    assert body.display.faces == [Face.ATTENTIVE, Face.MUTED, Face.NEUTRAL, Face.TALKING]


def test_disconnect_while_muted_stays_muted(memory_body, capsys) -> None:
    body = memory_body()
    body.core.on_session_started("session-1")
    body.show("speaking")
    body.flush()
    assert body.display.faces == [Face.TALKING]
    body.mute.enabled = False
    wait_until(lambda: body.display.faces[-1] == Face.MUTED)

    body.core.on_session_ended("session-1", "synthetic disconnect")
    wait_until(lambda: body.display.faces.count(Face.MUTED) == 2)
    assert "与大脑断开，等待重连" in capsys.readouterr().out  # noqa: RUF001
    assert body.display.faces == [Face.TALKING, Face.MUTED, Face.MUTED]

    body.core.on_session_started("session-2")
    for state in ("idle", "listening", "thinking", "speaking", "error"):
        body.show(state, session_id="session-2")
    body.flush()
    assert body.display.faces == [Face.TALKING, Face.MUTED, Face.MUTED]
    assert body.link.of(MuteState)[-1].session_id == "session-2"
    assert body.link.of(MuteState)[-1].microphone_enabled is False


def test_muting_during_playback_changes_only_expression(memory_body) -> None:
    body = memory_body()
    body.core.on_session_started("session-1")
    body.show("speaking")
    pcm = pcm_ramp(1600)
    body.core.on_message(
        PlayAudio(
            protocol_version=1, seq=0, session_id="session-1", playback_token=1,
            sample_rate=44100, channels=1, sample_format="pcm_s16le", total_bytes=len(pcm),
        )
    )
    body.core.on_frame(
        AudioFrame(
            direction="downlink", stream_id=1, frame_index=0, is_last=True, payload=pcm
        )
    )
    assert body.speaker.started.wait(5.0)
    body.mute.enabled = False
    wait_until(lambda: body.display.faces[-1] == Face.MUTED)
    assert body.display.faces == [Face.TALKING, Face.MUTED]
    assert body.speaker.stop is not None and not body.speaker.stop.is_set()
    assert not body.speaker.completed.is_set()
    assert body.link.of(PlaybackFinished) == []

    body.speaker.finish.set()
    wait_until(lambda: body.link.of(PlaybackFinished))
    finished = body.link.of(PlaybackFinished)[0]
    assert (finished.playback_token, finished.result) == (1, "completed")
    assert body.speaker.completed.is_set()
    assert (body.speaker.pcm, body.speaker.sample_rate) == (pcm, 44100)
    assert body.display.faces == [Face.TALKING, Face.MUTED]
