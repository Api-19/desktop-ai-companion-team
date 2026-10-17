"""Standalone CPU speech benchmark; requires only sherpa_onnx and numpy.

Copy this file to the target machine and run:
python3 pi_speech_benchmark.py --models /path/to/models --out result.json
Times include each measured call without warmup; recognition uses the complete
utterance after VAD, matching the application's local speech adapter.
"""

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path
from statistics import median

import numpy as np

TEXTS = (
    "可以可以。",
    "那我们等会去吃点啥呀",
    "今天好累，感觉身体被掏空了，想早点睡觉。",  # noqa: RUF001
    "你刚才说的那家店我知道，周末人很多，我们早点过去排队吧，不然又要等很久。",  # noqa: RUF001
)
SAMPLE_RATE = 16000
VAD_WINDOW = 512


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def thread_counts(value):
    try:
        return [positive_int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("线程数必须是逗号分隔的正整数") from exc


def cpu_temperature(path=Path("/sys/class/thermal/thermal_zone0/temp")):
    try:
        return float(path.read_text(encoding="utf-8").strip()) / 1000
    except (OSError, ValueError):
        return None


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if platform.system() == "Darwin" else 1024)


def resample(samples, sample_rate):
    if sample_rate == SAMPLE_RATE:
        return samples
    length = round(len(samples) * SAMPLE_RATE / sample_rate)
    return np.interp(
        np.arange(length) * sample_rate / SAMPLE_RATE,
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)


def load_tts(runtime, root, threads):
    directory = root / "vits-melo-tts-zh_en"
    vits = runtime.OfflineTtsVitsModelConfig(
        model=str(directory / "model.onnx"),
        lexicon=str(directory / "lexicon.txt"),
        tokens=str(directory / "tokens.txt"),
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1.0,
    )
    model = runtime.OfflineTtsModelConfig(
        vits=vits, num_threads=threads, debug=False, provider="cpu",
    )
    config = runtime.OfflineTtsConfig(
        model=model,
        rule_fsts=",".join(str(directory / name) for name in ("date.fst", "number.fst")),
        max_num_sentences=1,
    )
    if not config.validate():
        raise ValueError("invalid TTS config")
    return runtime.OfflineTts(config)


def load_recognizer(runtime, root, threads):
    directory = root / "sensevoice-2024-07-17-int8"
    return runtime.OfflineRecognizer.from_sense_voice(
        model=str(directory / "model.int8.onnx"),
        tokens=str(directory / "tokens.txt"),
        num_threads=threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
        provider="cpu",
        language="zh",
        use_itn=False,
    )


def load_vad(runtime, root, threads):
    config = runtime.VadModelConfig()
    config.silero_vad.model = str(root / "silero_vad.onnx")
    config.silero_vad.threshold = 0.25
    config.silero_vad.min_silence_duration = 0.5
    config.silero_vad.min_speech_duration = 0.5
    config.silero_vad.max_speech_duration = 10.0
    config.silero_vad.window_size = VAD_WINDOW
    config.sample_rate = SAMPLE_RATE
    config.num_threads = threads
    config.provider = "cpu"
    config.debug = False
    return runtime.VoiceActivityDetector(config, buffer_size_in_seconds=30.0)


def run_vad(vad, samples):
    vad.reset()
    for offset in range(0, len(samples), VAD_WINDOW):
        chunk = samples[offset:offset + VAD_WINDOW]
        if len(chunk) < VAD_WINDOW:
            chunk = np.pad(chunk, (0, VAD_WINDOW - len(chunk)))
        vad.accept_waveform(chunk)
    vad.flush()
    while not vad.empty():
        vad.pop()


def benchmark_round(runtime, root, threads, repeats):
    load = {}
    start = time.perf_counter()
    tts = load_tts(runtime, root, threads)
    load["tts_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    recognizer = load_recognizer(runtime, root, threads)
    load["recognizer_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    vad = load_vad(runtime, root, threads)
    load["vad_seconds"] = time.perf_counter() - start

    synthesis = []
    audio_inputs = []
    for text in TEXTS:
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            audio = tts.generate(text=text, sid=0, speed=1.0)
            timings.append(time.perf_counter() - start)
        samples = np.asarray(audio.samples, dtype=np.float32)
        duration = len(samples) / audio.sample_rate
        elapsed = median(timings)
        synthesis.append({
            "text": text, "synthesis_seconds": elapsed,
            "audio_seconds": duration, "rtf": elapsed / duration,
        })
        # Retain the final generated waveform for each utterance's ASR trials.
        audio_inputs.append(resample(samples, audio.sample_rate))

    recognition = []
    for text, samples in zip(TEXTS[1:], audio_inputs[1:], strict=True):
        vad_timings = []
        recognition_timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            run_vad(vad, samples)
            vad_timings.append(time.perf_counter() - start)
            start = time.perf_counter()
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            recognizer.decode_stream(stream)
            recognition_timings.append(time.perf_counter() - start)
        recognition.append({
            "text": text,
            "vad_seconds": median(vad_timings),
            "recognition_seconds": median(recognition_timings),
            "audio_seconds": len(samples) / SAMPLE_RATE,
            "recognized_text": stream.result.text,
        })

    return {
        "threads": threads,
        "load": load,
        "synthesis": synthesis,
        "recognition": recognition,
        "e2e_estimate": recognition[1]["recognition_seconds"] + synthesis[2]["synthesis_seconds"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="树莓派本地语音性能基准")
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=thread_counts, default="1,4")
    parser.add_argument("--repeats", type=positive_int, default=3)
    args = parser.parse_args(argv)

    import sherpa_onnx

    environment = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "sherpa_onnx_version": sherpa_onnx.__version__,
        "cpu_temperature_start_c": cpu_temperature(),
    }
    results = []
    for threads in args.threads:
        result = benchmark_round(sherpa_onnx, args.models, threads, args.repeats)
        results.append(result)
        print(
            f"{threads} 线程: 加载共 {sum(result['load'].values()):.3f} 秒, "
            f"第 3 句识别 {result['recognition'][1]['recognition_seconds']:.3f} 秒, "
            f"合成 {result['synthesis'][2]['synthesis_seconds']:.3f} 秒, "
            f"端到端估计 {result['e2e_estimate']:.3f} 秒"
        )
    environment["cpu_temperature_end_c"] = cpu_temperature()
    environment["peak_rss_mb"] = peak_rss_mb()
    args.out.write_text(
        json.dumps(
            {"environment": environment, "repeats": args.repeats, "results": results},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
