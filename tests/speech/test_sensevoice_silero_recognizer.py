from __future__ import annotations

import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

import companion.speech.sensevoice_silero_recognizer as recognizer_module
from companion.contracts import TranscriptConfidenceStatus
from companion.speech.controller_models import RecognitionInput
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_model_config import LocalSpeechModelConfig, VerifiedLocalSpeechPaths
from companion.speech.local_wav import OwnedWavFile, write_validated_pcm_wav
from tests.speech.test_local_model_config import make_config


class _FakeVadModelConfig:
    def __init__(self) -> None:
        self.silero_vad = SimpleNamespace()


class _FakeVad:
    def __init__(
        self,
        config: _FakeVadModelConfig,
        *,
        buffer_size_in_seconds: float,
    ) -> None:
        self.config = config
        self.buffer_size_in_seconds = buffer_size_in_seconds
        self.accepted_chunks: list[tuple[float, ...]] = []
        self.reset_calls = 0
        self.flush_calls = 0
        self.pop_calls = 0
        self._fronts: list[SimpleNamespace] = []
        self._configured_segments: list[tuple[object, object]] = []
        self.on_accept: object | None = None
        self.sticky_nonempty = False
        self.empty_calls = 0
        self.sticky_nonempty_limit: int | None = None

    def reset(self) -> None:
        self.reset_calls += 1
        self.accepted_chunks = []
        self._fronts = []

    def accept_waveform(self, samples: tuple[float, ...]) -> None:
        self.accepted_chunks.append(samples)
        if callable(self.on_accept):
            self.on_accept()

    def flush(self) -> None:
        self.flush_calls += 1
        self._fronts = [
            SimpleNamespace(start=start, samples=samples)
            for start, samples in self._configured_segments
        ]

    def empty(self) -> bool:
        self.empty_calls += 1
        if self.sticky_nonempty:
            if (
                self.sticky_nonempty_limit is not None
                and self.empty_calls > self.sticky_nonempty_limit
            ):
                raise RuntimeError("STICKY_VAD_SENTINEL")
            return False
        return not self._fronts

    @property
    def front(self) -> SimpleNamespace:
        return self._fronts[0]

    def pop(self) -> None:
        self.pop_calls += 1
        if not self.sticky_nonempty:
            self._fronts.pop(0)


class _FakeStream:
    def __init__(self, result: object) -> None:
        self.result = result
        self.accepted: list[tuple[int, tuple[float, ...]]] = []

    def accept_waveform(self, sample_rate: int, samples: tuple[float, ...]) -> None:
        self.accepted.append((sample_rate, samples))


class _FakeRecognizer:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self._runtime = runtime

    def create_stream(self) -> _FakeStream:
        self._runtime.stream_calls += 1
        stream = _FakeStream(self._runtime.result)
        self._runtime.streams.append(stream)
        return stream

    def decode_stream(self, stream: _FakeStream) -> None:
        self._runtime.decode_calls += 1
        if self._runtime.decode_error is not None:
            raise self._runtime.decode_error


class _FakeOfflineRecognizerFactory:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self._runtime = runtime

    def from_sense_voice(
        self,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16_000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        language: str = "",
        use_itn: bool = False,
        rule_fsts: str = "",
        rule_fars: str = "",
        hr_dict_dir: str = "",
        hr_rule_fsts: str = "",
        hr_lexicon: str = "",
    ) -> _FakeRecognizer:
        if self._runtime.raise_asr_initialization:
            raise RuntimeError("ASR_FACTORY_SENTINEL")
        self._runtime.asr_initializations += 1
        self._runtime.events.append("SenseVoice factory")
        self._runtime.asr_options = {
            "model": model,
            "tokens": tokens,
            "num_threads": num_threads,
            "sample_rate": sample_rate,
            "feature_dim": feature_dim,
            "decoding_method": decoding_method,
            "debug": debug,
            "provider": provider,
            "language": language,
            "use_itn": use_itn,
            "rule_fsts": rule_fsts,
            "rule_fars": rule_fars,
            "hr_dict_dir": hr_dict_dir,
            "hr_rule_fsts": hr_rule_fsts,
            "hr_lexicon": hr_lexicon,
        }
        return _FakeRecognizer(self._runtime)


class _FakeRuntime:
    def __init__(self) -> None:
        self.vad_configurations: list[_FakeVadModelConfig] = []
        self.vads: list[_FakeVad] = []
        self.vad_initializations = 0
        self.asr_initializations = 0
        self.asr_options: dict[str, object] | None = None
        self.stream_calls = 0
        self.decode_calls = 0
        self.streams: list[_FakeStream] = []
        self.result: object = SimpleNamespace(text="synthetic transcript")
        self.decode_error: BaseException | None = None
        self.raise_asr_initialization = False
        self.events: list[str] = []
        self.OfflineRecognizer = _FakeOfflineRecognizerFactory(self)

    def VadModelConfig(self) -> _FakeVadModelConfig:
        self.events.append("VadModelConfig")
        config = _FakeVadModelConfig()
        self.vad_configurations.append(config)
        return config

    def VoiceActivityDetector(
        self, config: _FakeVadModelConfig, *, buffer_size_in_seconds: float
    ) -> _FakeVad:
        self.events.append("VAD factory")
        self.vad_initializations += 1
        vad = _FakeVad(config, buffer_size_in_seconds=buffer_size_in_seconds)
        self.vads.append(vad)
        return vad


def _paths(config: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
    root = config.model_root
    return VerifiedLocalSpeechPaths(
        sensevoice_model=root / config.sensevoice_model_relative_path,
        sensevoice_tokens=root / config.sensevoice_tokens_relative_path,
        silero_model=root / config.silero_model_relative_path,
        vits_model=root / config.vits_model_relative_path,
        vits_lexicon=root / config.vits_lexicon_relative_path,
        vits_tokens=root / config.vits_tokens_relative_path,
        vits_rule_fsts=(
            root / config.vits_date_fst_relative_path,
            root / config.vits_number_fst_relative_path,
        ),
        run_root=config.run_root,
    )


def _samples(*, count: int = 16_001, clipped: bool = False) -> tuple[int, ...]:
    if clipped:
        return (32_767,) * count
    return (0,) * 4_000 + (8_192,) * 8_000 + (0,) * (count - 12_000)


@pytest.fixture
def adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime]:
    config = make_config(tmp_path)
    runtime = _FakeRuntime()

    def load(received: LocalSpeechModelConfig) -> _FakeRuntime:
        assert received is config
        runtime.events.append("load")
        return runtime

    def verify(received: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
        assert received is config
        runtime.events.append("verify")
        return _paths(received)

    monkeypatch.setattr(recognizer_module, "load_sherpa_runtime", load)
    monkeypatch.setattr(recognizer_module, "verify_local_speech_config", verify)
    recognizer = recognizer_module.SenseVoiceSileroRecognizer(config)
    runtime.vads[0]._configured_segments = [(4_000, (0.0,) * 8_000)]
    return recognizer, runtime


def _owned_audio(tmp_path: Path, samples: tuple[int, ...]) -> OwnedWavFile:
    return write_validated_pcm_wav(tmp_path, samples, 16_000)


def _assert_unavailable(result: RecognitionInput) -> None:
    assert result.confidence_status is TranscriptConfidenceStatus.UNAVAILABLE
    assert result.confidence is None
    assert result.confidence_threshold is None
    assert result.recognizer_failed is False


def test_construction_uses_the_frozen_factories_once(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
) -> None:
    _, runtime = adapter
    vad = runtime.vads[0]
    config = runtime.vad_configurations[0]

    assert runtime.vad_initializations == 1
    assert runtime.asr_initializations == 1
    assert runtime.events == [
        "verify",
        "load",
        "VadModelConfig",
        "verify",
        "VAD factory",
        "verify",
        "SenseVoice factory",
    ]
    assert vad.buffer_size_in_seconds == 30.0
    assert vars(config.silero_vad) == {
        "model": str(vad.config.silero_vad.model),
        "threshold": 0.25,
        "min_silence_duration": 0.5,
        "min_speech_duration": 0.5,
        "max_speech_duration": 10.0,
        "window_size": 512,
    }
    assert config.sample_rate == 16_000
    assert config.num_threads == 1
    assert config.provider == "cpu"
    assert config.debug is False
    assert runtime.asr_options == {
        "model": str(
            Path(vad.config.silero_vad.model).parent / "sensevoice-2024-07-17-int8/model.int8.onnx"
        ),
        "tokens": str(
            Path(vad.config.silero_vad.model).parent / "sensevoice-2024-07-17-int8/tokens.txt"
        ),
        "num_threads": 4,
        "sample_rate": 16_000,
        "feature_dim": 80,
        "decoding_method": "greedy_search",
        "debug": False,
        "provider": "cpu",
        "language": "zh",
        "use_itn": False,
        "rule_fsts": "",
        "rule_fars": "",
        "hr_dict_dir": "",
        "hr_rule_fsts": "",
        "hr_lexicon": "",
    }


def test_runtime_load_or_digest_verification_failure_stops_before_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    runtime = _FakeRuntime()

    def load(received: LocalSpeechModelConfig) -> _FakeRuntime:
        assert received is config
        runtime.events.append("load")
        return runtime

    def verify(received: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
        assert received is config
        runtime.events.append("verify")
        raise RuntimeError("DIGEST_SENTINEL")

    monkeypatch.setattr(recognizer_module, "load_sherpa_runtime", load)
    monkeypatch.setattr(recognizer_module, "verify_local_speech_config", verify)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer_module.SenseVoiceSileroRecognizer(config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert runtime.events == ["verify"]
    assert runtime.vad_initializations == 0
    assert runtime.asr_initializations == 0


def test_runtime_load_failure_stops_before_verification_and_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    runtime = _FakeRuntime()

    def load(received: LocalSpeechModelConfig) -> _FakeRuntime:
        assert received is config
        runtime.events.append("load")
        raise RuntimeError("RUNTIME_SENTINEL")

    def verify(_received: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
        runtime.events.append("verify")
        return _paths(config)

    monkeypatch.setattr(recognizer_module, "load_sherpa_runtime", load)
    monkeypatch.setattr(recognizer_module, "verify_local_speech_config", verify)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer_module.SenseVoiceSileroRecognizer(config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert runtime.events == ["verify", "load"]
    assert runtime.vad_initializations == 0
    assert runtime.asr_initializations == 0


@pytest.mark.parametrize(("failed_verification", "vad_calls", "asr_calls"), [(2, 0, 0), (3, 1, 0)])
def test_each_engine_construction_has_an_immediate_digest_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_verification: int,
    vad_calls: int,
    asr_calls: int,
) -> None:
    config = make_config(tmp_path)
    runtime = _FakeRuntime()
    verification_calls = 0

    def verify(received: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
        nonlocal verification_calls
        assert received is config
        verification_calls += 1
        runtime.events.append("verify")
        if verification_calls == failed_verification:
            raise RuntimeError("DIGEST_DRIFT_SENTINEL")
        return _paths(received)

    monkeypatch.setattr(recognizer_module, "verify_local_speech_config", verify)
    monkeypatch.setattr(recognizer_module, "load_sherpa_runtime", lambda _config: runtime)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer_module.SenseVoiceSileroRecognizer(config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert runtime.vad_initializations == vad_calls
    assert runtime.asr_initializations == asr_calls


def test_partial_construction_failure_does_not_retain_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    runtime = _FakeRuntime()
    runtime.raise_asr_initialization = True
    monkeypatch.setattr(recognizer_module, "load_sherpa_runtime", lambda _config: runtime)
    monkeypatch.setattr(recognizer_module, "verify_local_speech_config", _paths)
    recognizer = object.__new__(recognizer_module.SenseVoiceSileroRecognizer)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer_module.SenseVoiceSileroRecognizer.__init__(recognizer, config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert runtime.vad_initializations == 1
    assert runtime.asr_initializations == 0
    assert recognizer._SenseVoiceSileroRecognizer__vad is None
    assert recognizer._SenseVoiceSileroRecognizer__recognizer is None
    assert recognizer._SenseVoiceSileroRecognizer__closed is True


def test_stuck_vad_drain_is_bounded_without_asr(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter
    vad = runtime.vads[0]
    vad.sticky_nonempty = True
    vad.sticky_nonempty_limit = 40

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECOGNITION_FAILED
    assert vad.pop_calls == 33
    assert runtime.decode_calls == 0


def test_preflight_rejection_never_calls_asr(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter
    runtime.vads[0]._configured_segments = []

    result = recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert runtime.vads[0].reset_calls == 1
    assert runtime.vads[0].flush_calls == 1
    assert runtime.decode_calls == 0
    assert result.text is None
    assert result.vad_passed is False
    _assert_unavailable(result)


def test_invalid_duration_preflight_rejection_never_calls_asr(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter
    runtime.vads[0]._configured_segments = [(0, (0.0,) * 400)]

    result = recognizer.recognize(_owned_audio(tmp_path, (8_192,) * 400))

    assert result.audio.frame_count == 400
    assert result.text is None
    assert runtime.decode_calls == 0
    _assert_unavailable(result)


def test_no_snr_quality_preflight_rejection_never_calls_asr(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter
    runtime.vads[0]._configured_segments = [(0, (0.0,) * 16_001)]

    result = recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert result.audio.estimated_snr_db is None
    assert result.vad_passed is True
    assert result.text is None
    assert runtime.decode_calls == 0
    _assert_unavailable(result)


def test_admitted_audio_reaches_sensevoice_once_and_pads_vad_chunks(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter

    result = recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert runtime.decode_calls == 1
    assert runtime.stream_calls == 1
    assert result.text == "synthetic transcript"
    assert result.vad_passed is True
    _assert_unavailable(result)
    accepted = runtime.vads[0].accepted_chunks
    assert len(accepted) == 32
    assert all(len(chunk) == 512 for chunk in accepted)
    assert accepted[-1][-1] == 0.0
    assert runtime.streams[0].accepted == [
        (16_000, tuple(sample / 32_768 for sample in _samples()))
    ]


def test_consecutive_admitted_calls_reuse_the_initialized_engines(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter

    first = recognizer.recognize(_owned_audio(tmp_path, _samples()))
    second = recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert first.text == second.text == "synthetic transcript"
    assert runtime.vad_initializations == 1
    assert runtime.asr_initializations == 1
    assert runtime.vads[0].reset_calls == 2
    assert runtime.decode_calls == 2


def test_vad_failure_then_next_call_recovers_with_a_new_reset(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter
    runtime.vads[0]._configured_segments = [(True, (0.0,) * 8_000)]

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECOGNITION_FAILED
    runtime.vads[0]._configured_segments = [(4_000, (0.0,) * 8_000)]

    result = recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert result.text == "synthetic transcript"
    assert runtime.vads[0].reset_calls == 2
    assert runtime.decode_calls == 1


def test_recognizer_captures_the_owned_wav_once(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer, _ = adapter
    real_reader = recognizer_module.read_owned_pcm_wav
    calls = 0

    def recording_reader(audio: OwnedWavFile, *, expected_sample_rate_hz: int) -> object:
        nonlocal calls
        calls += 1
        return real_reader(audio, expected_sample_rate_hz=expected_sample_rate_hz)

    monkeypatch.setattr(recognizer_module, "read_owned_pcm_wav", recording_reader)

    result = recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert calls == 1
    assert result.audio.frame_count == 16_001


@pytest.mark.parametrize("segments", [[], [(4_000, (0.0,) * 8_000)]])
def test_each_preflight_rejection_keeps_asr_at_zero(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
    segments: list[tuple[object, object]],
) -> None:
    recognizer, runtime = adapter
    runtime.vads[0]._configured_segments = segments

    result = recognizer.recognize(_owned_audio(tmp_path, _samples(clipped=True)))

    assert result.text is None
    assert runtime.decode_calls == 0
    assert result.audio.clipping_fraction == 1.0
    _assert_unavailable(result)


def test_blank_result_is_left_for_the_existing_full_gate(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter
    runtime.result = SimpleNamespace(text="  ")

    result = recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert runtime.decode_calls == 1
    assert result.text == "  "
    _assert_unavailable(result)


@pytest.mark.parametrize(
    "configure",
    [
        lambda runtime: setattr(runtime, "result", SimpleNamespace(text=object())),
        lambda runtime: setattr(runtime, "decode_error", RuntimeError("THIRD_PARTY_SENTINEL")),
        lambda runtime: setattr(runtime.vads[0], "_configured_segments", [(True, (0.0,) * 8_000)]),
    ],
)
def test_malformed_or_runtime_failures_use_only_fixed_recognition_error(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
    configure: object,
) -> None:
    recognizer, runtime = adapter
    configure(runtime)  # type: ignore[operator]

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer.recognize(_owned_audio(tmp_path, _samples()))

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.code is LocalSpeechAdapterErrorCode.RECOGNITION_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "THIRD_PARTY_SENTINEL" not in formatted


def test_only_exact_owned_wav_is_accepted(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
) -> None:
    recognizer, runtime = adapter

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer.recognize(object())  # type: ignore[arg-type]

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    assert runtime.decode_calls == 0


def test_close_is_idempotent_and_recognition_after_close_fails_fixed(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, _ = adapter

    recognizer.close()
    recognizer.close()

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECOGNITION_FAILED


def test_synchronous_runtime_reentry_is_rejected_without_second_asr_call(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter
    audio = _owned_audio(tmp_path, _samples())
    errors: list[LocalSpeechAdapterError] = []

    def reenter() -> None:
        try:
            recognizer.recognize(audio)
        except LocalSpeechAdapterError as error:
            errors.append(error)

    runtime.vads[0].on_accept = reenter

    result = recognizer.recognize(audio)

    assert result.text == "synthetic transcript"
    assert runtime.decode_calls == 1
    assert [error.code for error in errors] == [LocalSpeechAdapterErrorCode.RECOGNITION_FAILED] * 32


def test_adapter_repr_and_fixed_failure_hide_runtime_and_model_paths(
    adapter: tuple[recognizer_module.SenseVoiceSileroRecognizer, _FakeRuntime],
    tmp_path: Path,
) -> None:
    recognizer, runtime = adapter
    runtime.decode_error = RuntimeError(str(tmp_path / "MODEL_PATH_SENTINEL"))

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recognizer.recognize(_owned_audio(tmp_path, _samples()))

    assert str(tmp_path) not in repr(recognizer)
    assert str(tmp_path) not in "".join(traceback.format_exception(caught.value))
