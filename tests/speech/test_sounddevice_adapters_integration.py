from __future__ import annotations

import struct
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from inspect import getmodule
from pathlib import Path

import pytest

import companion.speech.local_audio_devices as devices_module
import companion.speech.sensevoice_silero_recognizer as recognizer_module
import companion.speech.sherpa_vits_synthesizer as synthesizer_module
import companion.speech.sounddevice_player as player_module
import companion.speech.sounddevice_recorder as recorder_module
from companion.speech.local_audio_devices import LocalAudioDeviceConfig
from companion.speech.local_wav import OwnedWavFile
from companion.speech.pcm_audio import PcmAudio
from companion.speech.recognition_gate import gate_recognition
from companion.speech.sensevoice_silero_recognizer import SenseVoiceSileroRecognizer
from companion.speech.sherpa_vits_synthesizer import SherpaVitsSynthesizer
from companion.speech.sounddevice_player import PlaybackDeviceEvent, SoundDeviceAudioPlayer
from companion.speech.sounddevice_recorder import SoundDeviceAudioRecorder
from tests.speech.test_local_model_config import make_config
from tests.speech.test_sensevoice_silero_recognizer import _FakeRuntime as FakeAsrRuntime
from tests.speech.test_sensevoice_silero_recognizer import _paths as asr_paths
from tests.speech.test_sherpa_vits_synthesizer import _paths as tts_paths
from tests.speech.test_sherpa_vits_synthesizer import _Runtime as FakeTtsRuntime

_FIXED_REPLY = "公开合成回复"


def _accepted_pcm() -> bytes:
    samples = (0,) * 4_000 + (8_192,) * 8_000 + (0,) * 4_001
    return b"".join(struct.pack("<h", sample) for sample in samples)


class _InputStream:
    def __init__(self, callback: object) -> None:
        self._callback = callback
        self.calls: list[str] = []

    def start(self) -> _InputStream:
        self.calls.append("start")
        return self

    def emit(self, payload: bytes) -> None:
        self._callback(payload, len(payload) // 2, object(), False)  # type: ignore[operator]

    def stop(self) -> _InputStream:
        self.calls.append("stop")
        return self

    def close(self) -> _InputStream:
        self.calls.append("close")
        return self


class _OutputStream:
    def __init__(self, write_gate: threading.Event | None) -> None:
        self._write_gate = write_gate
        self._aborted = threading.Event()
        self.closed = threading.Event()
        self.write_entered = threading.Event()
        self.calls: list[str] = []
        self.writes: list[bytes] = []

    def start(self) -> _OutputStream:
        self.calls.append("start")
        return self

    def write(self, payload: bytes) -> None:
        self.calls.append("write")
        self.write_entered.set()
        if self._write_gate is not None:
            while not self._write_gate.wait(timeout=0.01):
                if self._aborted.is_set():
                    raise RuntimeError("interrupted")
        self.writes.append(bytes(payload))

    def abort(self) -> _OutputStream:
        self.calls.append("abort")
        self._aborted.set()
        return self

    def stop(self) -> _OutputStream:
        self.calls.append("stop")
        return self

    def close(self) -> _OutputStream:
        self.calls.append("close")
        self.closed.set()
        return self


class _SoundDeviceRuntime:
    def __init__(self) -> None:
        self.input_streams: list[_InputStream] = []
        self.output_streams: list[_OutputStream] = []
        self.output_write_gate: threading.Event | None = None
        self.query_calls = 0
        self.input_checks: list[dict[str, object]] = []
        self.output_checks: list[dict[str, object]] = []
        self.input_open_calls = 0
        self.output_open_calls = 0

    def query_devices(self) -> list[dict[str, object]]:
        self.query_calls += 1
        return [
            {"name": "public-input", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "public-output", "max_input_channels": 0, "max_output_channels": 1},
        ]

    def query_hostapis(self) -> object:
        raise AssertionError("hostapi must not be queried without a selector")

    def check_input_settings(self, **_kwargs: object) -> None:
        self.input_checks.append(_kwargs)
        return None

    def check_output_settings(self, **_kwargs: object) -> None:
        self.output_checks.append(_kwargs)
        return None

    def RawInputStream(self, **kwargs: object) -> _InputStream:
        self.input_open_calls += 1
        stream = _InputStream(kwargs["callback"])
        self.input_streams.append(stream)
        return stream

    def RawOutputStream(self, **_kwargs: object) -> _OutputStream:
        self.output_open_calls += 1
        stream = _OutputStream(self.output_write_gate)
        self.output_streams.append(stream)
        return stream


class _Stack:
    def __init__(
        self,
        *,
        run_root: Path,
        devices: _SoundDeviceRuntime,
        asr: FakeAsrRuntime,
        tts: FakeTtsRuntime,
        recorder: SoundDeviceAudioRecorder,
        recognizer: SenseVoiceSileroRecognizer,
        synthesizer: SherpaVitsSynthesizer,
        player: SoundDeviceAudioPlayer,
    ) -> None:
        self.run_root = run_root
        self.devices = devices
        self.asr = asr
        self.tts = tts
        self.recorder = recorder
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self.player = player
        self.owned_audio: list[OwnedWavFile | PcmAudio] = []

    def record(self) -> OwnedWavFile:
        session = self.recorder.start()
        self.devices.input_streams[-1].emit(_accepted_pcm())
        audio = session.stop()
        session.close()
        self.owned_audio.append(audio)
        return audio

    def synthesize(self) -> PcmAudio:
        audio = self.synthesizer.synthesize(_FIXED_REPLY)
        self.owned_audio.append(audio)
        return audio

    def release(self, audio: OwnedWavFile | PcmAudio) -> None:
        audio.close()
        self.owned_audio.remove(audio)

    def close(self) -> None:
        with suppress(Exception):
            self.player.stop()
        for audio in tuple(self.owned_audio):
            with suppress(Exception):
                audio.close()
        self.owned_audio.clear()
        self.recognizer.close()
        self.synthesizer.close()


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Stack]:
    model_config = make_config(tmp_path)
    device_config = LocalAudioDeviceConfig(
        run_root=model_config.run_root,
        input_selector="public-input",
        output_selector="public-output",
    )
    devices = _SoundDeviceRuntime()
    asr = FakeAsrRuntime()
    tts = FakeTtsRuntime()
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: devices)
    monkeypatch.setattr(recorder_module, "_load_sounddevice_runtime", lambda: devices)
    monkeypatch.setattr(player_module, "_load_sounddevice_runtime", lambda: devices)
    monkeypatch.setattr(recognizer_module, "load_sherpa_runtime", lambda _config: asr)
    monkeypatch.setattr(recognizer_module, "verify_local_speech_config", asr_paths)
    monkeypatch.setattr(synthesizer_module, "load_sherpa_runtime", lambda _config: tts)
    monkeypatch.setattr(synthesizer_module, "verify_local_speech_config", tts_paths)
    recognizer = SenseVoiceSileroRecognizer(model_config)
    asr.vads[0]._configured_segments = [(4_000, (0.0,) * 8_000)]
    result = _Stack(
        run_root=model_config.run_root,
        devices=devices,
        asr=asr,
        tts=tts,
        recorder=SoundDeviceAudioRecorder.from_config(device_config),
        recognizer=recognizer,
        synthesizer=SherpaVitsSynthesizer.from_config(model_config),
        player=SoundDeviceAudioPlayer.from_config(device_config),
    )
    try:
        yield result
    finally:
        result.close()
        assert list(result.run_root.iterdir()) == []


def _wait_for_event(stack: _Stack) -> tuple[PlaybackDeviceEvent, ...]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        events = stack.player.drain_events()
        if events:
            return events
        time.sleep(0.001)
    return ()


def test_successful_file_pipeline_releases_each_owned_wav(stack: _Stack) -> None:
    recorded = stack.record()
    result = stack.recognizer.recognize(recorded)
    stack.release(recorded)
    transcript = gate_recognition(
        audio=result.audio,
        vad_passed=result.vad_passed,
        text=result.text,
        confidence=result.confidence,
        confidence_status=result.confidence_status,
        confidence_threshold=result.confidence_threshold,
        recognizer_failed=result.recognizer_failed,
    ).transcript
    assert transcript is not None
    assert list(stack.run_root.iterdir()) == []

    synthesized = stack.synthesize()
    stack.player.start(synthesized, 17)
    events = _wait_for_event(stack)
    stack.release(synthesized)

    assert events == (PlaybackDeviceEvent(playback_token=17, succeeded=True),)
    assert len(stack.devices.input_streams) == 1
    assert len(stack.devices.output_streams) == 1
    assert stack.asr.decode_calls == 1
    assert stack.tts.calls == [(_FIXED_REPLY, 0, 1.0)]
    assert list(stack.run_root.iterdir()) == []


def test_rejected_recognition_stops_before_asr_tts_and_external_calls(stack: _Stack) -> None:
    stack.asr.vads[0]._configured_segments = []
    recorded = stack.record()

    result = stack.recognizer.recognize(recorded)
    stack.release(recorded)
    gated = gate_recognition(
        audio=result.audio,
        vad_passed=result.vad_passed,
        text=result.text,
        confidence=result.confidence,
        confidence_status=result.confidence_status,
        confidence_threshold=result.confidence_threshold,
        recognizer_failed=result.recognizer_failed,
    )

    assert gated.transcript is None
    assert stack.asr.decode_calls == 0
    assert stack.tts.calls == []
    assert len(stack.devices.input_streams) == 1
    assert stack.devices.output_streams == []
    assert list(stack.run_root.iterdir()) == []


def test_interrupted_playback_discards_completion_and_releases_owned_wavs(stack: _Stack) -> None:
    recorded = stack.record()
    result = stack.recognizer.recognize(recorded)
    stack.release(recorded)
    assert result.text is not None
    synthesized = stack.synthesize()
    gate = threading.Event()
    stack.devices.output_write_gate = gate

    stack.player.start(synthesized, 23)
    deadline = time.monotonic() + 2
    while not stack.devices.output_streams and time.monotonic() < deadline:
        time.sleep(0.001)
    assert stack.devices.output_streams[0].write_entered.wait(timeout=2)
    stack.player.stop()
    stack.release(synthesized)

    assert stack.player.drain_events() == ()
    assert len(stack.devices.input_streams) == 1
    assert len(stack.devices.output_streams) == 1
    assert stack.asr.decode_calls == 1
    assert stack.tts.calls == [(_FIXED_REPLY, 0, 1.0)]
    assert stack.devices.output_streams[0].calls.count("abort") == 1
    assert stack.devices.output_streams[0].calls.count("close") == 1
    assert list(stack.run_root.iterdir()) == []


def test_device_adapter_composition_excludes_dialogue_and_provider_dependencies() -> None:
    adapter_modules = (devices_module, recorder_module, player_module)
    forbidden_prefixes = ("companion.dialogue", "companion.providers")

    for module in adapter_modules:
        dependencies = {
            dependency.__name__
            for value in vars(module).values()
            if (dependency := getmodule(value)) is not None
        }
        assert not any(dependency.startswith(forbidden_prefixes) for dependency in dependencies)


def test_single_resolution_feeds_both_adapters_without_second_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="public-input",
        output_selector="public-output",
    )
    devices = _SoundDeviceRuntime()
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: devices)

    resolved = devices_module._resolve_audio_runtime(config)
    recorder = SoundDeviceAudioRecorder.from_resolved(config, resolved)
    player = SoundDeviceAudioPlayer.from_resolved(config, resolved)

    assert devices.query_calls == 1
    assert devices.input_open_calls == 0
    assert devices.output_open_calls == 0
    assert devices.input_checks == [
        {"device": 0, "channels": 1, "dtype": "int16", "samplerate": 16_000}
    ]
    assert devices.output_checks == [
        {"device": 1, "channels": 1, "dtype": "int16", "samplerate": 44_100}
    ]
    assert type(recorder) is SoundDeviceAudioRecorder
    assert type(player) is SoundDeviceAudioPlayer
