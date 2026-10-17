"""Session lifecycle evidence over loopback WSS, without models or audio devices."""

import logging
import struct
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import pytest

from companion.bodylink.body import BodyClient
from companion.bodylink.body_main import BodyCore, DevelopmentMuteSource
from companion.bodylink.brain_ports import BrainBodyLink
from companion.bodylink.pairing import body_ssl_context, brain_ssl_context, generate_pairing
from companion.contracts import DialogueResponse, TranscriptConfidenceStatus
from companion.contracts.body import AudioFrame, PlayAudio, ShowState, StartRecording
from companion.expression import Face
from companion.runtime.speech_chat_body import (
    FixedReplyDialogue,
    _run_body_loop,
    _write_console_status,
)
from companion.runtime.speech_expression import BodyDisplayPort, SpeechExpressionDriver
from companion.speech.controller import SpeechController
from companion.speech.controller_models import RecognitionInput, SpeechControllerIssue, SpeechState
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_wav import OwnedWavFile, read_owned_pcm_wav
from companion.speech.pcm_audio import PcmAudio
from companion.speech.recognition_gate import AudioQualityEvidence

TIMEOUT = 10.0
BODY_ID = "synthetic-session-lifecycle-body"
USER = "synthetic-owner-payload-sentinel"
TRANSCRIPT = "synthetic-transcript-payload-sentinel"
REPLY = "synthetic-reply-payload-sentinel"
ERROR = "synthetic-exception-payload-sentinel"
PCM_SENTINEL = "synthetic-pcm-payload-sentinel"
INPUT_PCM = PCM_SENTINEL.encode("ascii") * 480
OUTPUT_PCM = struct.pack("<4410h", *(index % 1000 for index in range(4410)))
STAGES = ("recognize", "dialogue", "synthesize")


class StageGate:
    """Only the first call at the selected boundary waits for the test's release."""

    def __init__(self, stage: str, outcome: str) -> None:
        self.stage = stage
        self.outcome = outcome
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []

    def checkpoint(self, stage: str) -> None:
        self.calls.append(stage)
        if stage == self.stage and not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(TIMEOUT), "stage result was not released"
            if self.outcome == "error":
                raise RuntimeError(ERROR)


class SyntheticInput:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class SyntheticMicrophone:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.handles: list[SyntheticInput] = []
        self.opened = threading.Event()

    def open(self, on_audio: Callable[[bytes], None]) -> SyntheticInput:
        handle = SyntheticInput()
        self.handles.append(handle)
        on_audio(INPUT_PCM)
        self.opened.set()
        return handle


class SyntheticSpeaker:
    def __init__(self) -> None:
        self.played: list[tuple[bytes, int]] = []
        self.finished = threading.Event()

    def play(self, pcm: bytes, sample_rate: int, stop: threading.Event) -> bool:
        self.played.append((pcm, sample_rate))
        self.finished.set()
        return not stop.is_set()

    def abort(self) -> None:
        return None


class SyntheticDisplay:
    def show(self, face: Face) -> None:
        pass


class SyntheticRecognizer:
    def __init__(self, gate: StageGate, run_root: Path) -> None:
        self.gate = gate
        self.run_root = run_root
        self.wav_reads = 0

    def recognize(self, audio: OwnedWavFile) -> RecognitionInput:
        wav = read_owned_pcm_wav(audio, expected_sample_rate_hz=16_000)
        assert struct.pack(f"<{wav.frame_count}h", *wav.samples_int16) == INPUT_PCM
        assert len(list(self.run_root.glob("*.wav"))) == 1
        self.wav_reads += 1
        self.gate.checkpoint("recognize")
        return RecognitionInput(
            audio=AudioQualityEvidence(
                sample_rate_hz=16_000,
                channels=1,
                sample_width_bytes=2,
                frame_count=wav.frame_count,
                rms=0.1,
                peak_abs=0.2,
                clipping_fraction=0.0,
                estimated_snr_db=20.0,
            ),
            vad_passed=True,
            text=TRANSCRIPT,
            confidence=None,
            confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
            confidence_threshold=None,
        )


class SyntheticDialogue:
    def __init__(self, gate: StageGate) -> None:
        self.gate = gate
        self.fixed = FixedReplyDialogue(REPLY)
        self.inputs: list[tuple[str, str]] = []

    def reply(self, *, user_id: str, text: str) -> DialogueResponse:
        self.inputs.append((user_id, text))
        self.gate.checkpoint("dialogue")
        return self.fixed.reply(user_id=user_id, text=text)


class SyntheticSynthesizer:
    def __init__(self, gate: StageGate) -> None:
        self.gate = gate
        self.texts: list[str] = []
        self.results: list[PcmAudio] = []

    def synthesize(self, text: str) -> PcmAudio:
        self.texts.append(text)
        self.gate.checkpoint("synthesize")
        audio = PcmAudio(pcm=OUTPUT_PCM, sample_rate_hz=44_100)
        self.results.append(audio)
        return audio

    def close(self) -> None:
        pass


class MonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()


class ConnectedBody:
    def __init__(self, rig: "Lifecycle") -> None:
        self.microphone = SyntheticMicrophone()
        self.speaker = SyntheticSpeaker()
        self.started = threading.Event()
        self.ended = threading.Event()
        self.idle = threading.Event()
        self.session_id: str | None = None
        self.messages: list[object] = []
        self.frames: list[AudioFrame] = []
        self.core = BodyCore(
            microphone=self.microphone,
            speaker=self.speaker,
            mute_source=DevelopmentMuteSource(),
            display=SyntheticDisplay(),
        )
        rig.resources.callback(self.core.close)
        self.client = BodyClient(
            host="127.0.0.1",
            port=rig.link.port,
            ssl_context=body_ssl_context(rig.pairing.body_dir),
            body_id=BODY_ID,
            output_sample_rates=(44_100,),
            on_session_started=self.on_started,
            on_session_ended=self.on_ended,
            on_message=self.on_message,
            on_frame=self.on_frame,
            reconnect_initial_s=0.01,
            reconnect_max_s=0.05,
        )
        rig.resources.callback(self.client.close)
        self.core.start(self.client)
        self.client.start()
        assert self.started.wait(TIMEOUT), "body handshake did not complete"
        assert self.session_id == rig.link.current_session_id

    def on_started(self, session_id: str) -> None:
        self.session_id = session_id
        self.core.on_session_started(session_id)
        self.started.set()

    def on_ended(self, session_id: str, reason: str) -> None:
        self.core.on_session_ended(session_id, reason)
        self.ended.set()

    def on_message(self, message: object) -> None:
        self.messages.append(message)
        self.core.on_message(message)
        if isinstance(message, ShowState) and message.event.state == "idle":
            self.idle.set()

    def on_frame(self, frame: AudioFrame) -> None:
        self.frames.append(frame)
        self.core.on_frame(frame)

    def count(self, kind: type) -> int:
        return sum(isinstance(message, kind) for message in self.messages)


class Lifecycle:
    """Small assembly for the real entry loop; no model assembly or production main."""

    def __init__(self, tmp_path: Path, stage: str, outcome: str) -> None:
        self.resources = ExitStack()
        self.gate = StageGate(stage, outcome)
        self.run_root = tmp_path / "synthetic-wav"
        self.run_root.mkdir()
        self.pairing = generate_pairing(tmp_path / "pairing", BODY_ID)
        self.link = BrainBodyLink(
            host="127.0.0.1",
            port=0,
            ssl_context=brain_ssl_context(self.pairing.brain_dir),
            body_id=BODY_ID,
            run_root=self.run_root,
        )
        self.resources.callback(self.link.close)
        self.recognizer = SyntheticRecognizer(self.gate, self.run_root)
        self.dialogue = SyntheticDialogue(self.gate)
        self.synthesizer = SyntheticSynthesizer(self.gate)
        self.resources.callback(self.synthesizer.close)
        self.expression = SpeechExpressionDriver(BodyDisplayPort(self.link), self.link.mute_gate)
        self.start_attempts: list[str] = []
        self.start_errors: list[LocalSpeechAdapterErrorCode] = []
        self.controller = SpeechController(
            trusted_user_id=USER,
            recorder=self.link.recorder,
            recognizer=self.recognizer,
            dialogue=self.dialogue,
            synthesizer=self.synthesizer,
            player=self.link.player,
            mute_gate=self.link.mute_gate,
            clock=MonotonicClock(),
            session_id_supplier=lambda: self.link.current_session_id,
            session_playback_starter=self.start_for_session,
            processing_observer=self.expression.processing,
        )
        self.resources.callback(self.controller.shutdown)
        self.quit = threading.Event()
        self.cycles = threading.Condition()
        self.cycle_count = 0
        self.failures: list[Exception] = []
        self.thread = threading.Thread(target=self.run, name="session-lifecycle-loop")

    def start(self) -> ConnectedBody:
        self.link.start()
        body = ConnectedBody(self)
        self.thread.start()
        assert body.idle.wait(TIMEOUT), "initial session did not become ready"
        return body

    def run(self) -> None:
        try:
            _run_body_loop(
                assembly=self,
                link=self.link,
                expression=self.expression,
                read_key=lambda: "q" if self.quit.is_set() else None,
                write_status=_write_console_status,
                sleep=self.after_cycle,
            )
        except Exception as error:
            self.failures.append(error)

    def after_cycle(self, seconds: float) -> None:
        with self.cycles:
            self.cycle_count += 1
            self.cycles.notify_all()
        self.quit.wait(seconds)

    def wait_state(self, state: SpeechState, *, after: int = -1) -> None:
        with self.cycles:
            assert self.cycles.wait_for(
                lambda: self.cycle_count > after and self.controller.state is state,
                timeout=TIMEOUT,
            ), f"entry loop did not reach {state}; failures={self.failures!r}"

    def drain_playback_events(self) -> None:
        for event in self.link.player.drain_events():
            if event.succeeded:
                self.controller.playback_finished(event.playback_token)
            else:
                self.controller.playback_failed(event.playback_token)

    def start_for_session(self, audio: PcmAudio, token: int, session_id: str) -> None:
        self.start_attempts.append(session_id)
        # The controller has validated its source session before this callback. The
        # real network player has not yet accepted it: this is the race boundary.
        self.gate.checkpoint("playback")
        try:
            self.link.player.start_for_session(audio, token, session_id)
        except LocalSpeechAdapterError as error:
            self.start_errors.append(error.code)
            raise

    def begin_processing(self, body: ConnectedBody) -> None:
        body.idle.clear()
        body.core.button_pressed()
        assert body.microphone.opened.wait(TIMEOUT), "new button did not start recording"
        self.wait_state(SpeechState.RECORDING)
        body.core.button_pressed()
        assert self.gate.entered.wait(TIMEOUT), "processing stage was not entered"

    def reconnect(self, body: ConnectedBody) -> ConnectedBody:
        body.client.close()
        assert body.ended.wait(TIMEOUT), "old client did not end its session"
        body.core.close()
        replacement = ConnectedBody(self)
        assert replacement.session_id != body.session_id
        assert replacement.session_id == self.link.current_session_id
        assert not self.gate.release.is_set()
        return replacement

    def release_result(self) -> None:
        cycle = self.cycle_count
        self.gate.release.set()
        self.wait_state(SpeechState.IDLE, after=cycle)

    def assert_quiet(self, body: ConnectedBody) -> None:
        # ShowState follows the completed old controller call on the same WSS
        # stream. Its receipt fences every earlier PlayAudio/audio/record command.
        assert body.idle.wait(TIMEOUT), "new session did not receive the idle fence"
        assert body.count(StartRecording) == 0
        assert body.microphone.handles == []
        assert body.count(PlayAudio) == 0
        assert body.frames == []
        assert body.speaker.played == []
        assert self.controller.last_issue is None
        assert list(self.run_root.iterdir()) == []

    def complete_new_turn(self, body: ConnectedBody) -> None:
        body.idle.clear()
        body.core.button_pressed()
        assert body.microphone.opened.wait(TIMEOUT)
        self.wait_state(SpeechState.RECORDING)
        cycle = self.cycle_count
        body.core.button_pressed()
        assert body.speaker.finished.wait(TIMEOUT)
        self.wait_state(SpeechState.IDLE, after=cycle)
        assert body.idle.wait(TIMEOUT)
        assert body.count(StartRecording) == 1
        assert len(body.microphone.handles) == 1
        assert body.microphone.handles[0].close_count == 1
        assert body.count(PlayAudio) == 1
        assert body.speaker.played == [(OUTPUT_PCM, 44_100)]
        assert b"".join(frame.payload for frame in body.frames) == OUTPUT_PCM
        assert body.frames[-1].is_last
        assert self.dialogue.inputs[-1] == (USER, TRANSCRIPT)
        assert self.synthesizer.texts[-1] == REPLY
        assert self.controller.last_issue is None
        assert list(self.run_root.iterdir()) == []

    def close(self) -> None:
        self.gate.release.set()
        self.quit.set()
        try:
            if self.thread.ident is not None:
                self.thread.join(TIMEOUT)
                assert not self.thread.is_alive(), "entry loop thread leaked"
            assert self.failures == []
        finally:
            self.resources.close()
        assert list(self.run_root.iterdir()) == []


@pytest.fixture
def make_lifecycle(tmp_path: Path, request: pytest.FixtureRequest) -> Callable:
    def make(stage: str, outcome: str = "success") -> tuple[Lifecycle, ConnectedBody]:
        rig = Lifecycle(tmp_path, stage, outcome)
        request.addfinalizer(rig.close)
        return rig, rig.start()

    return make


def _discard_old_turn(rig: Lifecycle, body: ConnectedBody) -> ConnectedBody:
    rig.begin_processing(body)
    replacement = rig.reconnect(body)
    rig.release_result()
    rig.assert_quiet(replacement)
    assert body.count(PlayAudio) == 0
    assert body.frames == []
    assert body.speaker.played == []
    assert [handle.close_count for handle in body.microphone.handles] == [1]
    return replacement


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("outcome", ("success", "error"))
def test_reconnect_during_each_processing_stage_requires_new_button(
    make_lifecycle: Callable, stage: str, outcome: str
) -> None:
    rig, body = make_lifecycle(stage, outcome)
    replacement = _discard_old_turn(rig, body)
    assert rig.gate.calls == list(STAGES[: STAGES.index(stage) + 1])
    rig.complete_new_turn(replacement)
    assert rig.start_attempts == [replacement.session_id]


def test_reconnect_between_controller_check_and_player_start_discards_old_audio(
    make_lifecycle: Callable,
) -> None:
    rig, body = make_lifecycle("playback")
    replacement = _discard_old_turn(rig, body)
    assert rig.start_attempts == [body.session_id]
    assert rig.start_errors == [LocalSpeechAdapterErrorCode.PLAYBACK_FAILED]
    rig.complete_new_turn(replacement)
    assert rig.start_attempts == [body.session_id, replacement.session_id]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("outcome", ("success", "error"))
@pytest.mark.parametrize("reconnect", (False, True), ids=("same-session", "reconnect"))
def test_late_result_cleanup_leaves_no_synthetic_wav(
    make_lifecycle: Callable, stage: str, outcome: str, reconnect: bool
) -> None:
    rig, body = make_lifecycle(stage, outcome)
    rig.begin_processing(body)
    assert rig.recognizer.wav_reads == 1
    assert len(list(rig.run_root.glob("*.wav"))) == (1 if stage == "recognize" else 0)
    if reconnect:
        body = rig.reconnect(body)
    rig.release_result()
    if reconnect:
        rig.assert_quiet(body)
        rig.complete_new_turn(body)
    else:
        expected_plays = 0 if stage == "synthesize" and outcome == "error" else 1
        assert body.count(PlayAudio) == expected_plays
        assert len(body.speaker.played) == expected_plays
        expected_issue = {
            "recognize": SpeechControllerIssue.RECOGNITION_FAILED,
            "dialogue": SpeechControllerIssue.DIALOGUE_FAILED,
            "synthesize": SpeechControllerIssue.SYNTHESIS_FAILED,
        }[stage]
        assert rig.controller.last_issue is (expected_issue if outcome == "error" else None)
    assert list(rig.run_root.iterdir()) == []
    for audio in rig.synthesizer.results:
        with pytest.raises(LocalSpeechAdapterError) as closed:
            _ = audio.pcm
        assert closed.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("outcome", ("success", "error"))
def test_session_lifecycle_logs_contain_no_payload(
    make_lifecycle: Callable,
    stage: str,
    outcome: str,
    capsys: pytest.CaptureFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    rig, body = make_lifecycle(stage, outcome)
    replacement = _discard_old_turn(rig, body)
    rig.complete_new_turn(replacement)
    rig.close()
    captured = capsys.readouterr()
    assert captured.out  # Exercise real entry/core diagnostics, including teardown.
    for payload in (USER, TRANSCRIPT, REPLY, ERROR, PCM_SENTINEL):
        assert payload not in captured.out
        assert payload not in captured.err
        assert payload not in caplog.text
