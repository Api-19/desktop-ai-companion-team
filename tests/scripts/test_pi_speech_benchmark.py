import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pi_speech_benchmark.py"
TEXTS = [
    "可以可以。",
    "那我们等会去吃点啥呀",
    "今天好累，感觉身体被掏空了，想早点睡觉。",  # noqa: RUF001
    "你刚才说的那家店我知道，周末人很多，我们早点过去排队吧，不然又要等很久。",  # noqa: RUF001
]


@pytest.fixture
def benchmark(monkeypatch):
    calls = SimpleNamespace(tts=[], asr=[], vad=[], generated=[], decoded=[], chunks=[])
    fake = ModuleType("sherpa_onnx")
    fake.__version__ = "synthetic"
    fake.OfflineTtsVitsModelConfig = lambda **kw: SimpleNamespace(**kw)
    fake.OfflineTtsModelConfig = lambda **kw: SimpleNamespace(**kw)
    fake.OfflineTtsConfig = lambda **kw: SimpleNamespace(**kw, validate=lambda: True)

    class Tts:
        def __init__(self, config):
            calls.tts.append(config)

        def generate(self, *, text, sid, speed):
            calls.generated.append((text, sid, speed))
            return SimpleNamespace(samples=np.ones(24000, dtype=np.float32), sample_rate=24000)

    class Recognizer:
        @classmethod
        def from_sense_voice(cls, **kw):
            calls.asr.append(kw)
            return cls()

        def create_stream(self):
            return SimpleNamespace(
                accept_waveform=lambda rate, samples: calls.decoded.append((rate, samples)),
                result=SimpleNamespace(text="合成识别文本"),
            )

        def decode_stream(self, stream):
            assert stream.result.text == "合成识别文本"

    class Vad:
        def __init__(self, config, *, buffer_size_in_seconds):
            calls.vad.append((config, buffer_size_in_seconds))
            self.pending = False

        def reset(self):
            self.pending = False

        def accept_waveform(self, samples):
            calls.chunks.append(samples)

        def flush(self):
            self.pending = True

        def empty(self):
            return not self.pending

        def pop(self):
            self.pending = False

    fake.OfflineTts = Tts
    fake.OfflineRecognizer = Recognizer
    fake.VadModelConfig = lambda: SimpleNamespace(silero_vad=SimpleNamespace())
    fake.VoiceActivityDetector = Vad
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    spec = importlib.util.spec_from_file_location("pi_speech_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_cli_benchmark(benchmark, tmp_path, capsys):
    module, calls = benchmark
    out = tmp_path / "result.json"
    assert module.main([
        "--models", str(tmp_path), "--out", str(out), "--threads", "1,4", "--repeats", "2",
    ]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["results"]) == 2
    for result, threads in zip(data["results"], [1, 4], strict=True):
        assert result["threads"] == threads
        assert set(result["load"]) == {"tts_seconds", "recognizer_seconds", "vad_seconds"}
        assert all(isinstance(value, (float, int)) for value in result["load"].values())
        assert [row["text"] for row in result["synthesis"]] == TEXTS
        assert [row["text"] for row in result["recognition"]] == TEXTS[1:]
        for row in result["synthesis"]:
            assert row["audio_seconds"] == 1
            assert row["rtf"] == row["synthesis_seconds"]
        for row in result["recognition"]:
            assert row["audio_seconds"] == 1
            assert row["recognized_text"] == "合成识别文本"
            assert row["vad_seconds"] >= 0
            assert row["recognition_seconds"] >= 0
        assert result["e2e_estimate"] == (
            result["synthesis"][2]["synthesis_seconds"]
            + result["recognition"][1]["recognition_seconds"]
        )
    assert [c.model.num_threads for c in calls.tts] == [1, 4]
    assert [c["num_threads"] for c in calls.asr] == [1, 4]
    assert [c.num_threads for c, _ in calls.vad] == [1, 4]
    assert calls.tts[0].model.vits.model == str(tmp_path / "vits-melo-tts-zh_en/model.onnx")
    assert calls.asr[0]["model"] == str(tmp_path / "sensevoice-2024-07-17-int8/model.int8.onnx")
    assert calls.vad[0][0].silero_vad.model == str(tmp_path / "silero_vad.onnx")
    assert calls.generated == [(text, 0, 1.0) for _ in range(2) for text in TEXTS for _ in range(2)]
    assert len(calls.decoded) == 12
    assert all(rate == 16000 and len(samples) == 16000 for rate, samples in calls.decoded)
    assert all(len(chunk) == 512 for chunk in calls.chunks)
    assert data["environment"]["sherpa_onnx_version"] == "synthetic"
    assert data["environment"]["peak_rss_mb"] > 0
    assert len(capsys.readouterr().out.splitlines()) == 2
    assert list(tmp_path.iterdir()) == [out]


@pytest.mark.parametrize("extra", [["--unknown"], ["--threads", "0"], ["--repeats", "0"]])
def test_invalid_arguments(benchmark, tmp_path, extra):
    module, _ = benchmark
    with pytest.raises(SystemExit) as exc:
        module.main(["--models", str(tmp_path), "--out", str(tmp_path / "out.json"), *extra])
    assert exc.value.code != 0


def test_standalone_source():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import companion" not in source
    assert "from companion" not in source


def test_resampling_and_environment_helpers(benchmark, monkeypatch, tmp_path):
    module, _ = benchmark
    samples = np.array([0, 1, 0, -1], dtype=np.float32)
    assert module.resample(samples, 16000) is samples
    np.testing.assert_allclose(module.resample(samples, 8000), [0, .5, 1, .5, 0, -.5, -1, -1])
    temperature = tmp_path / "temperature"
    assert module.cpu_temperature(temperature) is None
    temperature.write_text("42000\n", encoding="utf-8")
    assert module.cpu_temperature(temperature) == 42
    temperature.write_text("invalid", encoding="utf-8")
    assert module.cpu_temperature(temperature) is None
    monkeypatch.setattr(module.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1048576))
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    assert module.peak_rss_mb() == 1
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    assert module.peak_rss_mb() == 1024
