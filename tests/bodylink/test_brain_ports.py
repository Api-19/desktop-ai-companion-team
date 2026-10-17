import queue
import ssl
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.bodylink import brain_ports
from companion.bodylink.body import BodyClient
from companion.bodylink.brain_ports import BrainBodyLink
from companion.bodylink.pairing import PairingPaths, body_ssl_context, brain_ssl_context
from companion.bodylink.transport import NotConnectedError
from companion.contracts import DialogueResponse, TranscriptConfidenceStatus
from companion.contracts.body import (
    AbortRecording,
    AudioFrame,
    ButtonPressed,
    Hello,
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
from companion.speech.controller import SpeechController
from companion.speech.controller_models import RecognitionInput, SpeechControllerIssue, SpeechState
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_wav import OwnedWavFile, read_owned_pcm_wav
from companion.speech.pcm_audio import PcmAudio
from companion.speech.recognition_gate import AudioQualityEvidence
from companion.speech.runtime_assembly import LocalSpeechRuntimeAssembly
from companion.speech.sounddevice_player import PlaybackDeviceEvent

from .conftest import BODY_ID

FAST_RECONNECT = {"reconnect_initial_s": 0.05, "reconnect_max_s": 0.2}


def wait_until(predicate: Callable[[], object], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def pcm_ramp(samples: int) -> bytes:
    return struct.pack(f"<{samples}h", *((i * 7) % 65536 - 32768 for i in range(samples)))


class FakeBody:
    """BodyClient plus a worker thread that reacts to brain commands (callbacks must not send).

    record_mode: "normal" | "no_stop" (never reports RecordingStopped) | "wrong_total" |
    "disconnect_on_stop". play_result: result sent after the last downlink frame, or None."""

    def __init__(self, pairing: PairingPaths, port: int, *, rates: tuple[int, ...] = (44100,)):
        self.record_pcm = pcm_ramp(20_000)  # 40000 bytes: two uplink frames
        self.record_mode = "normal"
        self.play_result: str | None = "completed"
        self.received: list[object] = []
        self.downlink: list[AudioFrame] = []
        self.session_id: str | None = None
        self.ended = threading.Event()
        self._work: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self.client = BodyClient(
            host="127.0.0.1",
            port=port,
            ssl_context=body_ssl_context(pairing.body_dir),
            body_id=BODY_ID,
            output_sample_rates=rates,
            on_session_started=self._started,
            on_session_ended=lambda _sid, _reason: self.ended.set(),
            on_message=self._work.put,
            on_frame=self._work.put,
            **FAST_RECONNECT,
        )
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self.client.start()

    def _started(self, session_id: str) -> None:
        self.session_id = session_id

    def _run(self) -> None:
        while (item := self._work.get()) is not None:
            if isinstance(item, AudioFrame):
                with self._lock:
                    self.downlink.append(item)
                if item.is_last and self.play_result is not None:
                    self.send(PlaybackFinished(**self._common(), playback_token=item.stream_id,
                                               result=self.play_result))  # fmt: skip
                continue
            with self._lock:
                self.received.append(item)
            if isinstance(item, StartRecording):
                pcm = self.record_pcm
                chunks = [pcm[i : i + 32000] for i in range(0, len(pcm), 32000)]
                for index, chunk in enumerate(chunks):
                    self.client.send_frame(self.session_id, AudioFrame(
                        direction="uplink", stream_id=item.turn_id, frame_index=index,
                        is_last=False, payload=chunk))  # fmt: skip
                self._turn = (item.turn_id, len(chunks))
            elif isinstance(item, StopRecording):
                if self.record_mode == "disconnect_on_stop":
                    self.client.close()
                    continue
                turn_id, next_index = self._turn
                self.client.send_frame(self.session_id, AudioFrame(
                    direction="uplink", stream_id=turn_id, frame_index=next_index,
                    is_last=True, payload=b""))  # fmt: skip
                total = len(self.record_pcm) + (2 if self.record_mode == "wrong_total" else 0)
                if self.record_mode != "no_stop":
                    self.send(self.stopped(turn_id, "stop_command", total))

    def stopped(self, turn_id: int, reason: str, total: int) -> RecordingStopped:
        return RecordingStopped(**self._common(), turn_id=turn_id, reason=reason, total_bytes=total)

    def _common(self) -> dict:
        return {"protocol_version": 1, "seq": 0, "session_id": self.session_id}

    def send(self, message: object) -> None:
        self.client.send_message(message)

    def mute(self, enabled: bool) -> None:
        self.send(MuteState(**self._common(), microphone_enabled=enabled,
                            source="hardware_switch", observed_at_ms=1))  # fmt: skip

    def press(self) -> None:
        self.send(ButtonPressed(**self._common(), observed_at_ms=2))

    def messages_of(self, kind: type) -> list:
        with self._lock:
            return [m for m in self.received if isinstance(m, kind)]

    def frames(self) -> list[AudioFrame]:
        with self._lock:
            return list(self.downlink)

    def close(self) -> None:
        self._work.put(None)
        self._worker.join()
        self.client.close()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    return root


@pytest.fixture
def setup(pairing, closer, run_root):
    def build(*, rates=(44100,), clock=time.monotonic, **link_kw):
        link = BrainBodyLink(
            host="127.0.0.1",
            port=0,
            ssl_context=brain_ssl_context(pairing.brain_dir),
            body_id=BODY_ID,
            run_root=run_root,
            clock=clock,
            **link_kw,
        )
        closer(link)
        link.start()
        body = FakeBody(pairing, link.port, rates=rates)
        closer(body)
        wait_until(lambda: body.session_id is not None)
        return link, body

    return build


def enable_mic(link: BrainBodyLink, body: FakeBody) -> None:
    body.mute(True)
    wait_until(link.mute_gate.microphone_enabled)


def assert_recording_failed(call: Callable[[], object]) -> None:
    with pytest.raises(LocalSpeechAdapterError) as info:
        call()
    assert info.value.code is LocalSpeechAdapterErrorCode.RECORDING_FAILED


@pytest.fixture
def session_link(monkeypatch, run_root):
    server = SimpleNamespace(messages=[], frames=[], clock=FakeClock())

    def make_server(**callbacks):
        server.callbacks = callbacks
        server.send_message = server.messages.append
        server.send_frame = lambda session_id, frame: server.frames.append((session_id, frame))
        return server

    monkeypatch.setattr(brain_ports, "BrainServer", make_server)
    link = BrainBodyLink(
        host="127.0.0.1", port=0, ssl_context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        body_id=BODY_ID, run_root=run_root, clock=server.clock,
    )  # fmt: skip
    return link, server


def start_session(server, session_id: str) -> None:
    server.callbacks["on_session_started"](
        session_id,
        Hello(
            protocol_version=1, seq=0, body_id=BODY_ID,
            input_sample_rate=16000, output_sample_rates=(44100,),
        ),
    )


def session_button(session_id: str) -> ButtonPressed:
    return ButtonPressed(protocol_version=1, seq=0, session_id=session_id, observed_at_ms=0)


def session_mute(session_id: str, enabled: bool = True) -> MuteState:
    return MuteState(protocol_version=1, seq=0, session_id=session_id,
                     microphone_enabled=enabled, source="hardware_switch",
                     observed_at_ms=0)  # fmt: skip


@pytest.mark.parametrize("end_old", [True, False])
def test_queued_old_button_is_not_replayed(session_link, end_old: bool) -> None:
    link, server = session_link
    start_session(server, "A")
    server.callbacks["on_message"](session_button("A"))
    if end_old:
        server.callbacks["on_session_ended"]("A", "peer_closed")
    start_session(server, "B")
    new_button = session_button("B")
    server.callbacks["on_message"](new_button)

    assert link.drain_body_inputs() == (new_button,)
    assert link.current_session_id == "B"


def test_session_end_discards_pending_inputs(session_link) -> None:
    link, server = session_link
    start_session(server, "A")
    server.callbacks["on_message"](session_button("A"))

    server.callbacks["on_session_ended"]("A", "peer_closed")

    assert link.drain_body_inputs() == ()
    assert link.current_session_id is None


def test_stale_session_end_does_not_clear_new_session(session_link) -> None:
    link, server = session_link
    start_session(server, "A")
    server.callbacks["on_session_ended"]("A", "peer_closed")
    start_session(server, "B")
    mute = session_mute("B")
    server.callbacks["on_message"](mute)
    recording = link.recorder.start()
    link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 7)
    link.show_state(RobotEvent(state="speaking"))
    button = session_button("B")
    server.callbacks["on_message"](button)

    server.callbacks["on_session_ended"]("A", "peer_closed")

    assert link.mute_gate.microphone_enabled()
    assert link.current_session_id == "B"
    assert link.drain_body_inputs() == (mute, button)
    assert link.player.drain_events() == ()
    assert_recording_failed(link.recorder.start)
    with pytest.raises(LocalSpeechAdapterError) as caught:
        link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 8)
    assert caught.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED
    link.show_state(RobotEvent(state="speaking"))
    assert len([m for m in server.messages if isinstance(m, ShowState)]) == 1
    server.callbacks["on_frame"](AudioFrame(
        direction="uplink", stream_id=server.messages[0].turn_id,
        frame_index=0, is_last=True, payload=pcm_ramp(100)))  # fmt: skip
    server.callbacks["on_message"](RecordingStopped(
        protocol_version=1, seq=0, session_id="B", turn_id=server.messages[0].turn_id,
        reason="stop_command", total_bytes=200))  # fmt: skip
    recording.stop().close()
    recording.close()
    server.callbacks["on_message"](PlaybackFinished(
        protocol_version=1, seq=0, session_id="B", playback_token=7,
        result="completed"))  # fmt: skip
    assert link.player.drain_events() == (PlaybackDeviceEvent(7, True),)


def test_start_for_session_rejects_old_session_before_any_send(session_link) -> None:
    link, server = session_link
    start_session(server, "A")
    server.callbacks["on_session_ended"]("A", "peer_closed")
    start_session(server, "B")

    with pytest.raises(LocalSpeechAdapterError) as caught:
        link.player.start_for_session(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 1, "A")

    assert caught.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED
    assert server.messages == []
    assert server.frames == []


@pytest.mark.parametrize("transport_rejects_old", [False, True])
def test_reconnect_during_bound_playback_send_never_targets_new_session(
    session_link, monkeypatch, transport_rejects_old: bool,
) -> None:
    link, server = session_link
    start_session(server, "A")

    def send_message(message):
        server.messages.append(message)
        server.callbacks["on_session_ended"]("A", "peer_closed")
        start_session(server, "B")

    def send_frame(session_id, frame):
        server.frames.append((session_id, frame))
        if transport_rejects_old:
            raise NotConnectedError

    monkeypatch.setattr(server, "send_message", send_message)
    monkeypatch.setattr(server, "send_frame", send_frame)
    audio = PcmAudio(pcm=pcm_ramp(40_000), sample_rate_hz=44100)
    if transport_rejects_old:
        with pytest.raises(LocalSpeechAdapterError) as caught:
            link.player.start_for_session(audio, 1, "A")
        assert caught.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED
    else:
        link.player.start_for_session(audio, 1, "A")

    assert [message.session_id for message in server.messages] == ["A"]
    assert [session_id for session_id, _ in server.frames] == (
        ["A"] if transport_rejects_old else ["A", "A", "A"]
    )
    assert link.current_session_id == "B"


@pytest.mark.parametrize("kind", ["button", "mute", "recording", "playback"])
def test_stale_session_messages_do_not_change_current_ports(session_link, kind: str) -> None:
    link, server = session_link
    start_session(server, "A")
    server.callbacks["on_session_ended"]("A", "peer_closed")
    start_session(server, "B")
    server.callbacks["on_message"](session_mute("B"))
    link.drain_body_inputs()
    recording = link.recorder.start()
    link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 7)
    messages = {
        "button": session_button("A"),
        "mute": session_mute("A", False),
        "recording": RecordingStopped(
            protocol_version=1, seq=0, session_id="A", turn_id=server.messages[0].turn_id,
            reason="muted", total_bytes=0,
        ),
        "playback": PlaybackFinished(
            protocol_version=1, seq=0, session_id="A", playback_token=7, result="completed",
        ),
    }

    server.callbacks["on_message"](messages[kind])

    assert link.drain_body_inputs() == ()
    assert link.mute_gate.microphone_enabled()
    assert link.player.drain_events() == ()
    recording.abort()
    assert isinstance(server.messages[-1], AbortRecording)
    recording.close()
    link.player.stop()
    assert isinstance(server.messages[-1], StopPlayback)


# ---- mute gate ----


def test_gate_is_closed_without_session(pairing, closer, run_root) -> None:
    link = BrainBodyLink(
        host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=BODY_ID, run_root=run_root,
    )  # fmt: skip
    closer(link)
    link.start()
    assert link.mute_gate.microphone_enabled() is False


def test_gate_is_closed_until_mute_state_arrives_and_when_muted(setup) -> None:
    link, body = setup()
    assert link.mute_gate.microphone_enabled() is False
    enable_mic(link, body)
    body.mute(False)
    wait_until(lambda: not link.mute_gate.microphone_enabled())


def test_gate_closes_when_mute_state_is_stale(setup) -> None:
    clock = FakeClock()
    link, body = setup(clock=clock)
    enable_mic(link, body)
    clock.now += 3.0
    assert link.mute_gate.microphone_enabled() is True
    clock.now += 0.001
    assert link.mute_gate.microphone_enabled() is False


def test_gate_closes_when_session_ends(setup) -> None:
    link, body = setup()
    enable_mic(link, body)
    body.client.close()
    wait_until(lambda: not link.mute_gate.microphone_enabled())


# ---- recorder ----


def test_start_refused_when_gate_closed(setup) -> None:
    link, body = setup()
    assert_recording_failed(link.recorder.start)
    time.sleep(0.1)
    assert body.messages_of(StartRecording) == []


def test_recording_round_trip_yields_owned_wav_with_body_pcm(setup, run_root) -> None:
    link, body = setup()
    enable_mic(link, body)
    session = link.recorder.start()
    wait_until(lambda: body.messages_of(StartRecording))
    start = body.messages_of(StartRecording)[0]
    assert start.max_seconds == 30
    audio = session.stop()
    session.close()
    assert type(audio) is OwnedWavFile
    assert body.messages_of(StopRecording)[0].turn_id == start.turn_id
    wav = read_owned_pcm_wav(audio, expected_sample_rate_hz=16000)
    assert struct.pack(f"<{wav.frame_count}h", *wav.samples_int16) == body.record_pcm
    assert [p.suffix for p in run_root.iterdir()] == [".wav"]
    audio.close()
    assert list(run_root.iterdir()) == []
    assert body.messages_of(AbortRecording) == []


def test_turn_ids_increase_per_recording(setup) -> None:
    link, body = setup()
    enable_mic(link, body)
    for _ in range(2):
        link.recorder.start().stop().close()
    turn_ids = [m.turn_id for m in body.messages_of(StartRecording)]
    assert turn_ids[1] > turn_ids[0] >= 1


def test_stop_times_out_without_recording_stopped(setup, run_root) -> None:
    link, body = setup(stop_timeout_s=0.3)
    body.record_mode = "no_stop"
    enable_mic(link, body)
    session = link.recorder.start()
    started = time.monotonic()
    assert_recording_failed(session.stop)
    assert 0.3 <= time.monotonic() - started < 2.0
    assert list(run_root.iterdir()) == []
    link.recorder.start().abort()  # recorder is free again


def test_stop_fails_when_total_bytes_mismatch(setup, run_root) -> None:
    link, body = setup()
    body.record_mode = "wrong_total"
    enable_mic(link, body)
    session = link.recorder.start()
    assert_recording_failed(session.stop)
    assert list(run_root.iterdir()) == []


def test_stop_fails_when_session_drops_during_stop(setup, run_root) -> None:
    link, body = setup()
    body.record_mode = "disconnect_on_stop"
    enable_mic(link, body)
    session = link.recorder.start()
    started = time.monotonic()
    assert_recording_failed(session.stop)
    assert time.monotonic() - started < 1.5
    assert list(run_root.iterdir()) == []


def test_body_muted_recording_stopped_fails_stop(setup, run_root) -> None:
    link, body = setup()
    enable_mic(link, body)
    session = link.recorder.start()
    wait_until(lambda: body.messages_of(StartRecording))
    turn_id = body.messages_of(StartRecording)[0].turn_id
    body.send(body.stopped(turn_id, "muted", 0))
    time.sleep(0.1)
    assert_recording_failed(session.stop)
    session.close()
    assert body.messages_of(StopRecording) == []
    assert body.messages_of(AbortRecording) == []
    assert list(run_root.iterdir()) == []


def test_session_drop_during_recording_fails_stop(setup) -> None:
    link, body = setup()
    enable_mic(link, body)
    session = link.recorder.start()
    wait_until(lambda: body.messages_of(StartRecording))
    # Disconnect an active recording after the fake body's synchronous sends have finished.
    wait_until(lambda: link._recording.received_bytes == len(body.record_pcm))
    body.client.close()
    wait_until(lambda: not link.mute_gate.microphone_enabled())
    assert_recording_failed(session.stop)
    session.abort()


def test_abort_sends_abort_recording_and_frees_recorder(setup) -> None:
    link, body = setup()
    enable_mic(link, body)
    session = link.recorder.start()
    session.abort()
    session.close()
    wait_until(lambda: body.messages_of(AbortRecording))
    start = body.messages_of(StartRecording)[0]
    assert [m.turn_id for m in body.messages_of(AbortRecording)] == [start.turn_id]
    assert_recording_failed(session.stop)
    link.recorder.start().abort()


def test_second_start_while_recording_fails(setup) -> None:
    link, body = setup()
    enable_mic(link, body)
    session = link.recorder.start()
    assert_recording_failed(link.recorder.start)
    session.abort()


# ---- player ----


def test_playback_frames_and_finished_event(setup) -> None:
    link, body = setup()
    pcm = pcm_ramp(40_000)  # 80000 bytes at 44100 -> 32000 + 32000 + 16000
    link.player.start(PcmAudio(pcm=pcm, sample_rate_hz=44100), 1)
    wait_until(lambda: link.player.drain_events() == (PlaybackDeviceEvent(1, True),) or None)
    play = body.messages_of(PlayAudio)[0]
    assert (play.playback_token, play.sample_rate, play.total_bytes) == (1, 44100, len(pcm))
    frames = body.frames()
    assert [len(f.payload) for f in frames] == [32000, 32000, 16000]
    assert [f.frame_index for f in frames] == [0, 1, 2]
    assert [f.is_last for f in frames] == [False, False, True]
    assert {f.stream_id for f in frames} == {1}
    assert b"".join(f.payload for f in frames) == pcm


def test_playback_resamples_to_a_declared_rate(setup) -> None:
    link, body = setup(rates=(44100,))
    pcm = pcm_ramp(2205)  # 0.1 s at 22050
    link.player.start(PcmAudio(pcm=pcm, sample_rate_hz=22050), 1)
    wait_until(lambda: body.messages_of(PlayAudio))
    play = body.messages_of(PlayAudio)[0]
    assert play.sample_rate == 44100
    assert abs(play.total_bytes - 2 * len(pcm)) <= 2
    wait_until(lambda: link.player.drain_events())
    assert sum(len(f.payload) for f in body.frames()) == play.total_bytes


def test_playback_picks_nearest_declared_rate(setup) -> None:
    link, body = setup(rates=(48000, 16000))
    link.player.start(PcmAudio(pcm=pcm_ramp(2205), sample_rate_hz=22050), 1)
    wait_until(lambda: body.messages_of(PlayAudio))
    assert body.messages_of(PlayAudio)[0].sample_rate == 16000


@pytest.mark.parametrize(("result", "expected"), [("completed", True), ("failed", False)])
def test_playback_finished_result_maps_to_event(setup, result, expected) -> None:
    link, body = setup()
    body.play_result = result
    link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 7)
    events: list = []
    wait_until(lambda: events.extend(link.player.drain_events()) or events)
    assert events == [PlaybackDeviceEvent(7, expected)]


def test_body_side_stop_ends_playback_without_failure(setup) -> None:
    link, body = setup()
    body.play_result = "stopped"
    link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 3)
    events: list = []
    wait_until(lambda: events.extend(link.player.drain_events()) or events)
    assert events == [PlaybackDeviceEvent(3, True)]


def test_brain_stop_sends_stop_playback_and_suppresses_event(setup) -> None:
    link, body = setup()
    body.play_result = None
    link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 4)
    link.player.stop()
    wait_until(lambda: body.messages_of(StopPlayback))
    assert body.messages_of(StopPlayback)[0].playback_token == 4
    body.send(PlaybackFinished(**body._common(), playback_token=4, result="stopped"))
    time.sleep(0.2)
    assert link.player.drain_events() == ()
    link.player.stop()  # nothing active: no-op


def test_stale_playback_finished_is_ignored(setup) -> None:
    link, body = setup()
    body.play_result = None
    link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 5)
    body.send(PlaybackFinished(**body._common(), playback_token=99, result="completed"))
    time.sleep(0.2)
    assert link.player.drain_events() == ()


def test_session_drop_fails_active_playback(setup) -> None:
    link, body = setup()
    body.play_result = None
    link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 6)
    wait_until(lambda: body.frames())
    body.client.close()
    events: list = []
    wait_until(lambda: events.extend(link.player.drain_events()) or events)
    assert events == [PlaybackDeviceEvent(6, False)]


def test_start_without_session_fails(pairing, closer, run_root) -> None:
    link = BrainBodyLink(
        host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=BODY_ID, run_root=run_root,
    )  # fmt: skip
    closer(link)
    link.start()
    with pytest.raises(LocalSpeechAdapterError) as info:
        link.player.start(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 1)
    assert info.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED


# ---- body inputs ----


def test_button_and_mute_changes_reach_drain_body_inputs(setup) -> None:
    link, body = setup()
    body.mute(True)
    body.mute(True)  # periodic resend, not a change
    body.press()
    body.mute(False)
    inputs: list = []
    wait_until(lambda: inputs.extend(link.drain_body_inputs()) or len(inputs) >= 3)
    time.sleep(0.1)
    inputs.extend(link.drain_body_inputs())
    assert [type(m) for m in inputs] == [MuteState, ButtonPressed, MuteState]
    assert [m.microphone_enabled for m in inputs if isinstance(m, MuteState)] == [True, False]
    assert link.drain_body_inputs() == ()


# ---- show state ----


def test_show_state_is_sent_once_per_state_and_resent_after_reconnect(
    setup, pairing, closer
) -> None:
    link, body = setup()
    link.show_state(RobotEvent(state="listening", detail="synthetic-listening-first"))
    link.show_state(RobotEvent(state="listening", detail="synthetic-listening-second"))
    link.show_state(RobotEvent(state="thinking", detail="synthetic-thinking-first"))
    wait_until(lambda: len(body.messages_of(ShowState)) >= 2)
    time.sleep(0.1)
    assert [m.event.state for m in body.messages_of(ShowState)] == ["listening", "thinking"]
    assert all(m.event.detail is None for m in body.messages_of(ShowState))
    body.close()
    wait_until(lambda: not link.mute_gate.microphone_enabled() and link._session_id is None)
    link.show_state(RobotEvent(state="idle"))  # no session: dropped, not remembered
    second = FakeBody(pairing, link.port)
    closer(second)
    wait_until(lambda: second.session_id is not None and link._session_id is not None)
    link.show_state(RobotEvent(state="thinking", detail="synthetic-thinking-reconnected"))
    wait_until(lambda: second.messages_of(ShowState))
    assert [m.event.state for m in second.messages_of(ShowState)] == ["thinking"]
    assert all(m.event.detail is None for m in second.messages_of(ShowState))


def test_show_state_without_session_is_dropped(pairing, closer, run_root) -> None:
    link = BrainBodyLink(
        host="127.0.0.1", port=0, ssl_context=brain_ssl_context(pairing.brain_dir),
        body_id=BODY_ID, run_root=run_root,
    )  # fmt: skip
    closer(link)
    link.show_state(RobotEvent(state="idle"))


# ---- integration with the real SpeechController ----


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


class FakeSynthesizer:
    def synthesize(self, text: str) -> PcmAudio:
        return PcmAudio(pcm=pcm_ramp(4410), sample_rate_hz=44100)

    def close(self) -> None:
        pass


class MonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()


def test_controller_full_turn_over_bodylink(setup, run_root) -> None:
    link, body = setup()
    recognizer = CheckingRecognizer(body.record_pcm)
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

    enable_mic(link, body)
    pump()
    body.press()
    wait_until(lambda: pump() or controller.state is SpeechState.RECORDING)
    wait_until(lambda: body.messages_of(StartRecording))
    body.press()
    wait_until(lambda: pump() or controller.state is not SpeechState.RECORDING)
    wait_until(lambda: pump() or controller.state is SpeechState.IDLE)
    assert recognizer.matched
    assert controller.last_issue is None
    assert body.messages_of(PlayAudio)[0].total_bytes == 8820
    assert body.frames()[-1].is_last
    assert list(run_root.iterdir()) == []
    controller.shutdown()


# ---- progress and playback receipt deadlines ----


def finished(session_id="A", token=7):
    return PlaybackFinished(protocol_version=1, seq=0, session_id=session_id,
                            playback_token=token, result="completed")


@pytest.fixture
def pending_playback(session_link):
    link, server = session_link
    start_session(server, "A")
    link.player.start_for_session(PcmAudio(pcm=pcm_ramp(4410), sample_rate_hz=44100), 7, "A")
    return link, server


def assert_no_active_playback(link):
    assert link._playback_token is None
    assert link._playback_session_id is None
    assert link._playback_receipt_deadline is None


def test_empty_nonfinal_uplink_frame_fails_recording(session_link, monkeypatch, run_root) -> None:
    link, server = session_link
    start_session(server, "A")
    server.callbacks["on_message"](session_mute("A"))
    recording = link.recorder.start()
    turn = server.messages[-1].turn_id
    calls = []
    writer = brain_ports.write_validated_pcm_wav

    def observed_writer(*args):
        calls.append(args)
        return writer(*args)

    monkeypatch.setattr(brain_ports, "write_validated_pcm_wav", observed_writer)
    for index, payload, last in [(0, pcm_ramp(100), False), (1, b"", False), (2, b"", True)]:
        server.callbacks["on_frame"](AudioFrame(
            direction="uplink", stream_id=turn, frame_index=index, is_last=last, payload=payload,
        ))
    server.callbacks["on_message"](RecordingStopped(
        protocol_version=1, seq=0, session_id="A", turn_id=turn,
        reason="stop_command", total_bytes=200,
    ))
    assert_recording_failed(recording.stop)
    assert calls == []
    assert list(run_root.rglob("*.wav")) == []
    assert link._recording is None


def test_missing_playback_receipt_expires(pending_playback) -> None:
    link, server = pending_playback
    deadline = 1000.0 + 0.1 + 5.0
    assert link._playback_receipt_deadline == deadline
    server.clock.now = deadline - 0.001
    assert link.player.drain_events() == ()
    assert link._playback_token == 7
    server.clock.now = deadline
    assert link.player.drain_events() == (PlaybackDeviceEvent(7, False),)
    assert_no_active_playback(link)
    assert link.player.drain_events() == ()
    assert [(m.session_id, m.playback_token) for m in server.messages
            if isinstance(m, StopPlayback)] == [("A", 7)]


def test_receipt_deadline_uses_resampled_payload_duration(session_link) -> None:
    link, server = session_link
    start_session(server, "A")
    audio = PcmAudio(pcm=pcm_ramp(101), sample_rate_hz=16000)
    link.player.start_for_session(audio, 7, "A")
    play = server.messages[-1]
    sent = b"".join(frame.payload for _, frame in server.frames)
    assert play.sample_rate == 44100
    assert len(sent) == play.total_bytes
    expected = server.clock.now + len(sent) / (play.sample_rate * 2) + 5.0
    assert link._playback_receipt_deadline == expected
    assert expected != server.clock.now + len(audio.pcm) / (audio.sample_rate_hz * 2) + 5.0
    server.clock.now = expected - 0.000001
    assert link.player.drain_events() == ()
    server.clock.now = expected
    assert link.player.drain_events() == (PlaybackDeviceEvent(7, False),)


def test_receipt_deadline_starts_after_all_frames_are_sent(session_link, monkeypatch) -> None:
    link, server = session_link
    start_session(server, "A")
    send_frame = server.send_frame

    def timed_send(session_id, frame):
        assert link._playback_receipt_deadline is None
        server.clock.now += 1.0
        send_frame(session_id, frame)

    monkeypatch.setattr(server, "send_frame", timed_send)
    link.player.start_for_session(PcmAudio(pcm=pcm_ramp(40000), sample_rate_hz=44100), 7, "A")
    assert len(server.frames) == 3
    assert link._playback_receipt_deadline == 1003.0 + 40000 / 44100 + 5.0


def test_late_receipt_after_timeout_is_ignored(pending_playback) -> None:
    link, server = pending_playback
    server.clock.now = 1005.1
    assert link.player.drain_events() == (PlaybackDeviceEvent(7, False),)
    for _ in range(2):
        server.callbacks["on_message"](finished())
    assert link.player.drain_events() == ()
    assert_no_active_playback(link)


def test_same_token_receipt_from_another_session_is_ignored(pending_playback) -> None:
    link, server = pending_playback
    start_session(server, "B")
    server.callbacks["on_message"](finished("B"))
    assert link.player.drain_events() == ()
    assert link._playback_token == 7
    assert link._playback_session_id == "A"
    link.player.stop()


@pytest.mark.parametrize("action", ["stop", "disconnect", "finished", "send_failure"])
def test_stop_and_disconnect_clear_receipt_deadline(pending_playback, monkeypatch, action) -> None:
    link, server = pending_playback
    if action == "stop":
        # The stored playback source owns the stop even if the current session changed.
        start_session(server, "B")
        link.player.stop()
        assert server.messages[-1].session_id == "A"
        expected = ()
    elif action == "disconnect":
        server.callbacks["on_session_ended"]("A", "peer_closed")
        expected = (PlaybackDeviceEvent(7, False),)
    elif action == "finished":
        server.callbacks["on_message"](finished())
        server.callbacks["on_message"](finished())
        expected = (PlaybackDeviceEvent(7, True),)
    else:
        link.player.stop()

        def failed_send(_session, _frame):
            raise NotConnectedError

        monkeypatch.setattr(server, "send_frame", failed_send)
        with pytest.raises(LocalSpeechAdapterError) as caught:
            link.player.start_for_session(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 8, "A")
        assert caught.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED
        expected = ()
    assert_no_active_playback(link)
    server.clock.now = 2000.0
    assert link.player.drain_events() == expected
    assert link.player.drain_events() == ()


def test_old_send_failure_does_not_clear_new_session_same_token(session_link, monkeypatch) -> None:
    link, server = session_link
    start_session(server, "A")
    audio = PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100)
    send_frame = server.send_frame

    def reconnect(session_id, frame):
        assert link._playback_receipt_deadline is None
        server.callbacks["on_session_ended"]("A", "peer_closed")
        start_session(server, "B")
        monkeypatch.setattr(server, "send_frame", send_frame)
        link.player.start_for_session(audio, 7, "B")
        raise NotConnectedError

    monkeypatch.setattr(server, "send_frame", reconnect)
    with pytest.raises(LocalSpeechAdapterError) as caught:
        link.player.start_for_session(audio, 7, "A")
    assert caught.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED
    assert link._playback_token == 7
    assert link._playback_session_id == "B"
    assert link._playback_receipt_deadline == 1000.0 + 100 / 44100 + 5.0
    assert link.player.drain_events() == (PlaybackDeviceEvent(7, False),)
    server.callbacks["on_message"](finished("A"))
    assert link.player.drain_events() == ()
    assert link._playback_session_id == "B"
    server.callbacks["on_message"](finished("B"))
    assert link.player.drain_events() == (PlaybackDeviceEvent(7, True),)
    assert_no_active_playback(link)


def test_finished_during_send_does_not_rearm_deadline(session_link, monkeypatch) -> None:
    link, server = session_link
    start_session(server, "A")

    def finish_during_send(session_id, frame):
        assert link._playback_receipt_deadline is None
        server.frames.append((session_id, frame))
        if frame.is_last:
            server.callbacks["on_message"](finished())

    monkeypatch.setattr(server, "send_frame", finish_during_send)
    link.player.start_for_session(PcmAudio(pcm=pcm_ramp(100), sample_rate_hz=44100), 7, "A")
    assert_no_active_playback(link)
    assert link.player.drain_events() == (PlaybackDeviceEvent(7, True),)
    server.clock.now = 2000.0
    assert link.player.drain_events() == ()


@pytest.mark.parametrize("send_error", [None, NotConnectedError, OSError])
def test_receipt_timeout_sends_only_source_stop(
    pending_playback, monkeypatch, caplog, capsys, send_error,
) -> None:
    link, server = pending_playback
    secret = "synthetic-timeout-transport-body"
    server.messages.clear()
    server.frames.clear()

    def send_stop(message):
        assert_no_active_playback(link)
        start_session(server, "B")  # a new session cannot retarget this send
        server.messages.append(message)
        if send_error is not None:
            raise send_error(secret)

    def unexpected_pcm(*args):
        pytest.fail("receipt expiry must not resend PCM")

    monkeypatch.setattr(server, "send_message", send_stop)
    monkeypatch.setattr(server, "send_frame", unexpected_pcm)
    server.clock.now = 1005.1
    if send_error is OSError:
        with pytest.raises(OSError):
            link.player.drain_events()
    assert link.player.drain_events() == (PlaybackDeviceEvent(7, False),)
    assert server.messages == [StopPlayback(
        protocol_version=1, seq=0, session_id="A", playback_token=7,
    )]
    assert server.frames == []
    assert link.player.drain_events() == ()
    captured = capsys.readouterr()
    assert secret not in repr(server.messages) + caplog.text + captured.out + captured.err


def test_receipt_timeout_returns_real_controller_to_idle(setup, run_root) -> None:
    clock = FakeClock()
    link, body = setup(clock=clock)
    body.play_result = None
    controller = SpeechController(
        trusted_user_id="owner", recorder=link.recorder,
        recognizer=CheckingRecognizer(body.record_pcm), dialogue=FakeDialogue(),
        synthesizer=FakeSynthesizer(), player=link.player, mute_gate=link.mute_gate,
        clock=MonotonicClock(), session_id_supplier=lambda: link.current_session_id,
        session_playback_starter=link.player.start_for_session,
    )
    # Use the existing production playback-event consumer without loading models/devices.
    assembly = LocalSpeechRuntimeAssembly.__new__(LocalSpeechRuntimeAssembly)
    assembly._closed = False
    assembly._controller = controller
    assembly._player = link.player
    try:
        enable_mic(link, body)
        controller.button_pressed()
        wait_until(lambda: body.messages_of(StartRecording))
        controller.button_pressed()
        assert controller.state is SpeechState.PLAYING
        clock.now = link._playback_receipt_deadline - 0.001
        assembly.drain_playback_events()
        assert controller.state is SpeechState.PLAYING
        clock.now += 0.001
        assembly.drain_playback_events()
        assert controller.state is SpeechState.IDLE
        assert controller.last_issue is SpeechControllerIssue.PLAYBACK_FAILED
        assert list(run_root.iterdir()) == []
        wait_until(lambda: body.messages_of(StopPlayback))
        assembly.drain_playback_events()
        assert controller.state is SpeechState.IDLE
    finally:
        controller.shutdown()
