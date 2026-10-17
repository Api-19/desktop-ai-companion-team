from __future__ import annotations

import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

import companion.speech.sherpa_vits_synthesizer as vits_module
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_model_config import LocalSpeechModelConfig, VerifiedLocalSpeechPaths
from companion.speech.pcm_audio import PcmAudio, pcm16_samples
from companion.speech.sherpa_vits_synthesizer import SherpaVitsSynthesizer
from tests.speech.test_local_model_config import make_config

PUBLIC_TEXT = "合成公开测试文本"


class _VitsConfig:
    def __init__(self, *, model: str, lexicon: str, tokens: str, **scales: object) -> None:
        self.model = model
        self.lexicon = lexicon
        self.tokens = tokens
        self.scales = scales


class _ModelConfig:
    def __init__(self, *, vits: _VitsConfig, **options: object) -> None:
        self.vits = vits
        self.options = options


class _TtsConfig:
    def __init__(self, *, model: _ModelConfig, **options: object) -> None:
        self.model = model
        self.options = options
        self.valid = True

    def validate(self) -> bool:
        return self.valid


class _Engine:
    def __init__(self, runtime: _Runtime) -> None:
        self._runtime = runtime

    def generate(self, *, text: str, sid: int, speed: float) -> object:
        self._runtime.calls.append((text, sid, speed))
        if self._runtime.error is not None:
            raise self._runtime.error
        return self._runtime.output


class _Runtime:
    def __init__(self) -> None:
        self.configs: list[_TtsConfig] = []
        self.initializations = 0
        self.calls: list[tuple[str, int, float]] = []
        self.output: object = SimpleNamespace(samples=(0.25, -0.25, 0.5, -0.5), sample_rate=44_100)
        self.error: BaseException | None = None

    OfflineTtsVitsModelConfig = _VitsConfig
    OfflineTtsModelConfig = _ModelConfig
    OfflineTtsConfig = _TtsConfig

    def OfflineTts(self, config: _TtsConfig) -> _Engine:
        self.initializations += 1
        self.configs.append(config)
        return _Engine(self)


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


@pytest.fixture
def adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SherpaVitsSynthesizer, _Runtime, Path]:
    config = make_config(tmp_path)
    runtime = _Runtime()
    monkeypatch.setattr(vits_module, "load_sherpa_runtime", lambda received: runtime)
    monkeypatch.setattr(vits_module, "verify_local_speech_config", _paths)
    return SherpaVitsSynthesizer.from_config(config), runtime, config.run_root


def test_vits_initializes_once_with_frozen_options(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path],
) -> None:
    synthesizer, runtime, _ = adapter

    first = synthesizer.synthesize(PUBLIC_TEXT)
    second = synthesizer.synthesize("第二条公开测试文本")

    assert runtime.initializations == 1
    assert runtime.calls == [(PUBLIC_TEXT, 0, 1.0), ("第二条公开测试文本", 0, 1.0)]
    config = runtime.configs[0]
    assert Path(config.model.vits.model).parts[-2:] == ("vits-melo-tts-zh_en", "model.onnx")
    assert Path(config.model.vits.lexicon).parts[-2:] == ("vits-melo-tts-zh_en", "lexicon.txt")
    assert Path(config.model.vits.tokens).parts[-2:] == ("vits-melo-tts-zh_en", "tokens.txt")
    assert config.model.vits.scales == {
        "noise_scale": 0.667,
        "noise_scale_w": 0.8,
        "length_scale": 1.0,
    }
    assert config.model.options == {"num_threads": 4, "debug": False, "provider": "cpu"}
    assert config.options == {
        "rule_fsts": ",".join(
            (
                str(config.model.vits.model).replace("model.onnx", "date.fst"),
                str(config.model.vits.model).replace("model.onnx", "number.fst"),
            )
        ),
        "max_num_sentences": 1,
    }
    assert type(first) is PcmAudio
    assert type(second) is PcmAudio
    first.close()
    second.close()


def test_config_verification_precedes_runtime_loading_and_blocks_it_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    events: list[str] = []

    def verify(received: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
        assert received is config
        events.append("verify")
        raise RuntimeError("VERIFY_SENTINEL")

    def load(received: LocalSpeechModelConfig) -> _Runtime:
        assert received is config
        events.append("load")
        return _Runtime()

    monkeypatch.setattr(vits_module, "verify_local_speech_config", verify)
    monkeypatch.setattr(vits_module, "load_sherpa_runtime", load)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        SherpaVitsSynthesizer.from_config(config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert events == ["verify"]


def test_config_verification_precedes_runtime_loading_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    runtime = _Runtime()
    events: list[str] = []

    def verify(received: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
        assert received is config
        events.append("verify")
        return _paths(received)

    def load(received: LocalSpeechModelConfig) -> _Runtime:
        assert received is config
        events.append("load")
        return runtime

    monkeypatch.setattr(vits_module, "verify_local_speech_config", verify)
    monkeypatch.setattr(vits_module, "load_sherpa_runtime", load)

    synthesizer = SherpaVitsSynthesizer.from_config(config)

    assert events == ["verify", "load", "verify"]
    synthesizer.close()


def test_engine_construction_stops_when_immediate_digest_recheck_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    runtime = _Runtime()
    verification_calls = 0

    def verify(received: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
        nonlocal verification_calls
        assert received is config
        verification_calls += 1
        if verification_calls == 2:
            raise RuntimeError("DIGEST_DRIFT_SENTINEL")
        return _paths(received)

    monkeypatch.setattr(vits_module, "verify_local_speech_config", verify)
    monkeypatch.setattr(vits_module, "load_sherpa_runtime", lambda _config: runtime)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        SherpaVitsSynthesizer.from_config(config)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert runtime.initializations == 0


def test_float_samples_use_full_int16_range_and_half_away_from_zero_ties(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path],
) -> None:
    synthesizer, runtime, _ = adapter
    unit = 32_767.0
    runtime.output = SimpleNamespace(
        samples=(-1.0, 1.0, 0.49 / unit, -0.49 / unit, 0.5 / unit, -0.5 / unit),
        sample_rate=44_100,
    )

    audio = synthesizer.synthesize(PUBLIC_TEXT)

    assert audio.sample_rate_hz == 44_100
    assert pcm16_samples(audio) == (
        -32_768,
        32_767,
        0,
        0,
        1,
        -1,
    )
    audio.close()


def test_synthesis_returns_unique_in_memory_44khz_pcm_without_files(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path],
) -> None:
    synthesizer, _, run_root = adapter

    first = synthesizer.synthesize(PUBLIC_TEXT)
    second = synthesizer.synthesize(PUBLIC_TEXT)

    assert first is not second
    assert list(run_root.iterdir()) == []
    assert first.sample_rate_hz == 44_100
    assert pcm16_samples(first) == (
        8192,
        -8192,
        16384,
        -16384,
    )
    first.close()
    second.close()


@pytest.mark.parametrize("text", ["", "  ", 42, b"text"])
def test_invalid_text_is_rejected_before_the_engine(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path], text: object
) -> None:
    synthesizer, runtime, _ = adapter

    with pytest.raises(LocalSpeechAdapterError) as caught:
        synthesizer.synthesize(text)  # type: ignore[arg-type]

    assert caught.value.code is LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED
    assert runtime.calls == []


def test_string_subclass_is_rejected_before_the_engine(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path],
) -> None:
    class TextSubclass(str):
        pass

    synthesizer, runtime, _ = adapter

    with pytest.raises(LocalSpeechAdapterError) as caught:
        synthesizer.synthesize(TextSubclass(PUBLIC_TEXT))

    assert caught.value.code is LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED
    assert runtime.calls == []


@pytest.mark.parametrize(
    "output",
    [
        object(),
        SimpleNamespace(samples=(), sample_rate=44_100),
        SimpleNamespace(samples=(0.0, 0.0), sample_rate=44_100),
        SimpleNamespace(samples=(True,), sample_rate=44_100),
        SimpleNamespace(samples=(float("nan"),), sample_rate=44_100),
        SimpleNamespace(samples=(1.01,), sample_rate=44_100),
        SimpleNamespace(samples=(0.25,), sample_rate=True),
        SimpleNamespace(samples=(0.25,), sample_rate=16_000),
    ],
)
def test_malformed_output_uses_fixed_error_and_writes_no_files(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path], output: object
) -> None:
    synthesizer, runtime, run_root = adapter
    runtime.output = output

    with pytest.raises(LocalSpeechAdapterError) as caught:
        synthesizer.synthesize(PUBLIC_TEXT)

    assert caught.value.code is LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED
    assert runtime.calls == [(PUBLIC_TEXT, 0, 1.0)]
    assert list(run_root.iterdir()) == []


def test_close_is_idempotent_and_blocks_later_engine_calls(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path],
) -> None:
    synthesizer, runtime, _ = adapter

    synthesizer.close()
    synthesizer.close()

    with pytest.raises(LocalSpeechAdapterError) as caught:
        synthesizer.synthesize(PUBLIC_TEXT)

    assert caught.value.code is LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED
    assert runtime.calls == []


def test_reentrant_generation_is_rejected_without_a_second_engine_call(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    synthesizer, runtime, _ = adapter
    engine = synthesizer._SherpaVitsSynthesizer__engine
    errors: list[LocalSpeechAdapterError] = []
    original_generate = engine.generate

    def reenter(*, text: str, sid: int, speed: float) -> object:
        try:
            synthesizer.synthesize(PUBLIC_TEXT)
        except LocalSpeechAdapterError as error:
            errors.append(error)
        return original_generate(text=text, sid=sid, speed=speed)

    monkeypatch.setattr(engine, "generate", reenter)

    audio = synthesizer.synthesize(PUBLIC_TEXT)

    assert type(audio) is PcmAudio
    assert runtime.calls == [(PUBLIC_TEXT, 0, 1.0)]
    assert [error.code for error in errors] == [LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED]
    audio.close()


def test_failure_does_not_leak_text_or_runtime_exception(
    adapter: tuple[SherpaVitsSynthesizer, _Runtime, Path],
) -> None:
    synthesizer, runtime, run_root = adapter
    marker = "PRIVATE_TEXT_SENTINEL"
    runtime.error = RuntimeError(f"ENGINE_SENTINEL:{run_root}")

    with pytest.raises(LocalSpeechAdapterError) as caught:
        synthesizer.synthesize(marker)

    formatted = "".join(traceback.format_exception(caught.value))

    assert caught.value.code is LocalSpeechAdapterErrorCode.SYNTHESIS_FAILED
    assert marker not in str(caught.value)
    assert str(run_root) not in str(caught.value)
    assert "ENGINE_SENTINEL" not in str(caught.value)
    assert marker not in formatted
    assert str(run_root) not in formatted
    assert "ENGINE_SENTINEL" not in formatted
    assert marker not in repr(synthesizer)
    assert str(run_root) not in repr(synthesizer)
    assert "ENGINE_SENTINEL" not in repr(synthesizer)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None

    module_locals: list[str] = []
    current = caught.value.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == vits_module.__name__:
            module_locals.extend(
                f"{name}={value!r}" for name, value in current.tb_frame.f_locals.items()
            )
        current = current.tb_next
    local_rendering = "\n".join(module_locals)
    assert marker not in local_rendering
    assert str(run_root) not in local_rendering
    assert "8192" not in local_rendering
