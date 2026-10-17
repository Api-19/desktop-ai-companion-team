from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import companion.speech.sensevoice_silero_recognizer as recognizer_module
import companion.speech.sherpa_vits_synthesizer as synthesizer_module
from companion.contracts import DialogueResponse, TranscriptConfidenceStatus
from companion.speech.controller import SpeechController
from companion.speech.controller_models import RecognitionInput, SpeechControllerIssue, SpeechState
from companion.speech.local_errors import LocalSpeechAdapterError
from companion.speech.local_wav import OwnedWavFile, write_validated_pcm_wav
from companion.speech.pcm_audio import PcmAudio
from companion.speech.sensevoice_silero_recognizer import SenseVoiceSileroRecognizer
from companion.speech.sherpa_vits_synthesizer import SherpaVitsSynthesizer
from tests.speech.test_local_model_config import make_config
from tests.speech.test_sensevoice_silero_recognizer import _FakeRuntime as FakeAsrRuntime
from tests.speech.test_sensevoice_silero_recognizer import _paths as asr_paths
from tests.speech.test_sherpa_vits_synthesizer import _paths as tts_paths
from tests.speech.test_sherpa_vits_synthesizer import _Runtime as FakeTtsRuntime

_TRANSCRIPT = "synthetic transcript"
_REPLY = "公开合成回复"


def _accepted_samples() -> tuple[int, ...]:
    return (0,) * 4_000 + (8_192,) * 8_000 + (0,) * 4_001


class _RecordingSession:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def stop(self) -> OwnedWavFile:
        self._recorder.events.append("session.stop")
        return write_validated_pcm_wav(self._recorder.run_root, self._recorder.samples, 16_000)

    def abort(self) -> None:
        self._recorder.events.append("session.abort")

    def close(self) -> None:
        self._recorder.events.append("session.close")


class _Recorder:
    def __init__(self, run_root: Path, samples: tuple[int, ...]) -> None:
        self.run_root = run_root
        self.samples = samples
        self.events: list[str] = []
        self.starts = 0
        self.device_open_calls = 0

    def start(self) -> _RecordingSession:
        self.starts += 1
        self.events.append("recorder.start")
        return _RecordingSession(self)


class _Dialogue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.provider_calls = 0
        self.on_reply: object | None = None
        self.run_root: Path | None = None

    def reply(self, *, user_id: str, text: str) -> DialogueResponse:
        self.calls.append((user_id, text))
        assert self.run_root is not None
        assert list(self.run_root.iterdir()) == []
        if callable(self.on_reply):
            self.on_reply()
        return DialogueResponse(
            text=_REPLY,
            disclosed_ai_identity=False,
            billable=False,
            blocked_reason=None,
        )


class _Player:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.audios: list[PcmAudio] = []
        self.stop_calls = 0
        self.device_open_calls = 0

    def start(self, audio: PcmAudio, playback_token: int) -> None:
        assert type(audio) is PcmAudio
        self.audios.append(audio)
        self.tokens.append(playback_token)

    def stop(self) -> None:
        self.stop_calls += 1


class _MuteGate:
    def microphone_enabled(self) -> bool:
        return True


class _ObservingRecognizer:
    def __init__(self, delegate: SenseVoiceSileroRecognizer) -> None:
        self.delegate = delegate
        self.results: list[RecognitionInput] = []

    def recognize(self, audio: OwnedWavFile) -> RecognitionInput:
        result = self.delegate.recognize(audio)
        self.results.append(result)
        return result


class _Clock:
    def monotonic(self) -> float:
        return 0.0


@dataclass
class _Stack:
    controller: SpeechController
    recognizer: SenseVoiceSileroRecognizer
    observed_recognition: _ObservingRecognizer
    synthesizer: SherpaVitsSynthesizer
    recorder: _Recorder
    dialogue: _Dialogue
    player: _Player
    fake_asr: FakeAsrRuntime
    fake_tts: FakeTtsRuntime
    run_root: Path

    def run_files(self) -> list[Path]:
        return list(self.run_root.iterdir())

    def finish(self) -> None:
        self.controller.playback_finished(1)


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Stack:
    config = make_config(tmp_path)
    fake_asr = FakeAsrRuntime()
    fake_asr.result.text = _TRANSCRIPT
    fake_tts = FakeTtsRuntime()
    monkeypatch.setattr(recognizer_module, "load_sherpa_runtime", lambda _config: fake_asr)
    monkeypatch.setattr(recognizer_module, "verify_local_speech_config", asr_paths)
    monkeypatch.setattr(synthesizer_module, "load_sherpa_runtime", lambda _config: fake_tts)
    monkeypatch.setattr(synthesizer_module, "verify_local_speech_config", tts_paths)
    recognizer = SenseVoiceSileroRecognizer(config)
    observed_recognition = _ObservingRecognizer(recognizer)
    fake_asr.vads[0]._configured_segments = [(4_000, (0.0,) * 8_000)]
    synthesizer = SherpaVitsSynthesizer.from_config(config)
    recorder = _Recorder(config.run_root, _accepted_samples())
    dialogue = _Dialogue()
    dialogue.run_root = config.run_root
    player = _Player()
    controller = SpeechController(
        trusted_user_id="trusted-owner",
        recorder=recorder,
        recognizer=observed_recognition,
        dialogue=dialogue,
        synthesizer=synthesizer,
        player=player,
        mute_gate=_MuteGate(),
        clock=_Clock(),
    )
    return _Stack(
        controller=controller,
        recognizer=recognizer,
        observed_recognition=observed_recognition,
        synthesizer=synthesizer,
        recorder=recorder,
        dialogue=dialogue,
        player=player,
        fake_asr=fake_asr,
        fake_tts=fake_tts,
        run_root=config.run_root,
    )


def _record_once(stack: _Stack) -> None:
    stack.controller.button_pressed()
    stack.controller.button_pressed()


def test_controller_accepts_file_audio_and_plays_in_memory_tts_pcm(stack: _Stack) -> None:
    stack.dialogue.on_reply = stack.controller.button_pressed

    _record_once(stack)

    assert stack.fake_asr.decode_calls == 1
    assert len(stack.observed_recognition.results) == 1
    recognition = stack.observed_recognition.results[0]
    assert recognition.confidence_status is TranscriptConfidenceStatus.UNAVAILABLE
    assert recognition.confidence is None
    assert recognition.confidence_threshold is None
    assert stack.dialogue.calls == [("trusted-owner", _TRANSCRIPT)]
    assert stack.fake_tts.calls == [(_REPLY, 0, 1.0)]
    assert stack.player.tokens == [1]
    assert stack.recorder.starts == 1
    assert stack.recorder.events == ["recorder.start", "session.stop", "session.close"]
    assert stack.run_files() == []
    assert stack.player.audios[0].sample_rate_hz == 44_100
    assert stack.controller.state is SpeechState.PLAYING

    stack.finish()

    assert stack.run_files() == []
    with pytest.raises(LocalSpeechAdapterError):
        _ = stack.player.audios[0].pcm
    assert stack.controller.state is SpeechState.IDLE
    assert stack.controller.last_issue is None
    assert stack.recorder.device_open_calls == 0
    assert stack.player.device_open_calls == 0
    assert stack.dialogue.provider_calls == 0
    stack.recognizer.close()
    stack.recognizer.close()
    stack.synthesizer.close()
    stack.synthesizer.close()


@pytest.mark.parametrize(
    ("samples", "segments"),
    [
        (_accepted_samples(), []),
        ((32_767,) * 16_001, [(4_000, (0.0,) * 8_000)]),
    ],
)
def test_pre_asr_rejections_never_call_asr_or_dialogue(
    stack: _Stack, samples: tuple[int, ...], segments: list[tuple[int, tuple[float, ...]]]
) -> None:
    stack.recorder.samples = samples
    stack.fake_asr.vads[0]._configured_segments = segments

    _record_once(stack)

    assert stack.fake_asr.decode_calls == 0
    assert len(stack.observed_recognition.results) == 1
    recognition = stack.observed_recognition.results[0]
    assert recognition.confidence_status is TranscriptConfidenceStatus.UNAVAILABLE
    assert recognition.confidence is None
    assert recognition.confidence_threshold is None
    assert stack.dialogue.calls == []
    assert stack.fake_tts.calls == [("我没听清，再说一次吧", 0, 1.0)]  # noqa: RUF001
    assert stack.player.tokens == [1]
    stack.finish()
    assert stack.run_files() == []


def test_blank_or_failed_asr_stays_before_dialogue_and_closes_owned_files(stack: _Stack) -> None:
    stack.fake_asr.result.text = "  "

    _record_once(stack)

    assert stack.fake_asr.decode_calls == 1
    assert stack.dialogue.calls == []
    assert stack.controller.last_issue is SpeechControllerIssue.RECOGNITION_REJECTED
    stack.finish()
    assert stack.run_files() == []

    stack.fake_asr.decode_error = RuntimeError("fake runtime failure")
    _record_once(stack)

    assert stack.fake_asr.decode_calls == 2
    assert stack.dialogue.calls == []
    assert stack.controller.last_issue is SpeechControllerIssue.RECOGNITION_FAILED
    stack.controller.playback_finished(2)
    assert stack.run_files() == []


def test_tts_failure_and_playback_interruption_leave_no_owned_files(stack: _Stack) -> None:
    stack.fake_tts.error = RuntimeError("fake tts failure")

    _record_once(stack)

    assert stack.dialogue.calls == [("trusted-owner", _TRANSCRIPT)]
    assert stack.fake_tts.calls == [(_REPLY, 0, 1.0)]
    assert stack.player.tokens == []
    assert stack.controller.state is SpeechState.IDLE
    assert stack.controller.last_issue is SpeechControllerIssue.SYNTHESIS_FAILED
    assert stack.run_files() == []

    stack.fake_tts.error = None
    _record_once(stack)
    assert stack.run_files() == []
    assert len(stack.player.audios[0].pcm) == 8
    stack.controller.button_pressed()

    assert stack.player.stop_calls == 1
    assert stack.run_files() == []
    with pytest.raises(LocalSpeechAdapterError):
        _ = stack.player.audios[0].pcm
    assert stack.controller.state is SpeechState.RECORDING


def test_stale_token_keeps_playing_and_finish_releases_audio(stack: _Stack) -> None:
    _record_once(stack)

    stack.controller.playback_finished(999)
    assert stack.controller.state is SpeechState.PLAYING
    assert len(stack.player.audios[0].pcm) == 8

    stack.finish()

    assert stack.controller.state is SpeechState.IDLE
    assert stack.controller.last_issue is None
    assert stack.run_files() == []
    with pytest.raises(LocalSpeechAdapterError):
        _ = stack.player.audios[0].pcm
