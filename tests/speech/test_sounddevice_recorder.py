from __future__ import annotations

import struct
import threading
import traceback
from contextlib import suppress
from pathlib import Path

import pytest

import companion.speech.local_audio_devices as devices_module
import companion.speech.sounddevice_recorder as recorder_module
from companion.speech.controller_ports import AudioRecorder, RecordingSession
from companion.speech.local_audio_devices import LocalAudioDeviceConfig
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_wav import OwnedWavFile, read_owned_pcm_wav
from companion.speech.sounddevice_recorder import SoundDeviceAudioRecorder


class _FakeInputStream:
    def __init__(self, callback: object) -> None:
        self.callback = callback
        self.calls: list[str] = []
        self.start_failure: Exception | None = None
        self.stop_failure: Exception | None = None
        self.close_failure: Exception | None = None

    def start(self) -> object:
        self.calls.append("start")
        if self.start_failure is not None:
            raise self.start_failure
        return self

    def stop(self) -> object:
        self.calls.append("stop")
        if self.stop_failure is not None:
            raise self.stop_failure
        return self

    def close(self) -> object:
        self.calls.append("close")
        if self.close_failure is not None:
            raise self.close_failure
        return self

    def emit(self, payload: bytes, *, status: object = False, frames: object | None = None) -> None:
        frame_count = len(payload) // 2 if frames is None else frames
        self.callback(payload, frame_count, object(), status)  # type: ignore[operator]


class _FakeSoundDevice:
    def __init__(self) -> None:
        self.streams: list[_FakeInputStream] = []
        self.stream_kwargs: list[dict[str, object]] = []
        self.constructor_failure: Exception | None = None
        self.query_calls = 0
        self.input_checks: list[dict[str, object]] = []
        self.output_checks: list[dict[str, object]] = []
        self.input_open_calls = 0
        self.output_open_calls = 0
        self.unsupported_input_rates: set[int] = set()

    def query_devices(self) -> list[dict[str, object]]:
        self.query_calls += 1
        return [
            {"name": "synthetic-input", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "synthetic-output", "max_input_channels": 0, "max_output_channels": 1},
        ]

    def check_input_settings(self, **_kwargs: object) -> None:
        self.input_checks.append(_kwargs)
        if _kwargs.get("samplerate") in self.unsupported_input_rates:
            raise ValueError("unsupported")
        return None

    def check_output_settings(self, **_kwargs: object) -> None:
        self.output_checks.append(_kwargs)
        return None

    def RawInputStream(self, **kwargs: object) -> _FakeInputStream:
        self.input_open_calls += 1
        self.stream_kwargs.append(kwargs)
        if self.constructor_failure is not None:
            raise self.constructor_failure
        stream = _FakeInputStream(kwargs["callback"])
        self.streams.append(stream)
        return stream

    def RawOutputStream(self, **_kwargs: object) -> object:
        self.output_open_calls += 1
        return object()


def _pcm_bytes(frame_count: int = 3200) -> bytes:
    return b"".join(struct.pack("<h", 1200 if index % 2 else -1200) for index in range(frame_count))


def _make_recorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SoundDeviceAudioRecorder, _FakeSoundDevice]:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )
    runtime = _FakeSoundDevice()
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(recorder_module, "_load_sounddevice_runtime", lambda: runtime)
    return SoundDeviceAudioRecorder.from_config(config), runtime


def test_stop_returns_owned_pcm_wav_and_releases_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()
    stream = runtime.streams[0]
    stream.emit(_pcm_bytes())
    assert list((tmp_path / "run").iterdir()) == []

    audio = session.stop()
    session.close()

    assert type(audio) is OwnedWavFile
    pcm = read_owned_pcm_wav(audio, expected_sample_rate_hz=16_000)
    assert (pcm.channels, pcm.sample_width_bytes, pcm.frame_count) == (1, 2, 3200)
    assert pcm.samples_int16[:4] == (-1200, 1200, -1200, 1200)
    assert stream.calls == ["start", "stop", "close"]
    audio.close()
    assert list((tmp_path / "run").iterdir()) == []


def test_stream_uses_exact_capture_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)

    session = recorder.start()

    assert runtime.stream_kwargs == [
        {
            "device": 0,
            "samplerate": 16_000,
            "channels": 1,
            "dtype": "int16",
            "callback": runtime.stream_kwargs[0]["callback"],
        }
    ]
    session.abort()
    session.close()


def test_native_48khz_capture_is_resampled_to_canonical_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )
    runtime = _FakeSoundDevice()
    runtime.unsupported_input_rates.add(16_000)
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(recorder_module, "_load_sounddevice_runtime", lambda: runtime)
    calls: list[tuple[tuple[int, ...], int, int]] = []

    def resample(samples: tuple[int, ...], source: int, target: int) -> tuple[int, ...]:
        calls.append((samples, source, target))
        return samples[::3]

    monkeypatch.setattr(recorder_module, "resample_pcm16", resample)
    recorder = SoundDeviceAudioRecorder.from_config(config)

    session = recorder.start()
    runtime.streams[0].emit(_pcm_bytes(4800))
    audio = session.stop()
    pcm = read_owned_pcm_wav(audio, expected_sample_rate_hz=16_000)

    assert runtime.stream_kwargs[0]["samplerate"] == 48_000
    assert len(calls) == 1
    assert calls[0][1:] == (48_000, 16_000)
    assert pcm.frame_count == 1600
    audio.close()


def test_native_capture_resample_failure_creates_no_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )
    runtime = _FakeSoundDevice()
    runtime.unsupported_input_rates.add(16_000)
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(recorder_module, "_load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(
        recorder_module,
        "resample_pcm16",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("SENSITIVE_PCM")),
    )
    recorder = SoundDeviceAudioRecorder.from_config(config)
    session = recorder.start()
    runtime.streams[0].emit(_pcm_bytes(4800))

    with pytest.raises(LocalSpeechAdapterError) as caught:
        session.stop()

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECORDING_FAILED
    assert caught.value.__cause__ is None
    assert list(run_root.iterdir()) == []


def test_abort_is_idempotent_and_never_creates_a_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()
    runtime.streams[0].emit(_pcm_bytes())

    session.abort()
    session.abort()
    session.close()

    assert runtime.streams[0].calls == ["start", "stop", "close"]
    assert list((tmp_path / "run").iterdir()) == []


@pytest.mark.parametrize("payload", [b"", b"\x00"])
def test_empty_or_odd_capture_fails_without_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()
    runtime.streams[0].emit(payload)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        session.stop()

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECORDING_FAILED
    assert list((tmp_path / "run").iterdir()) == []
    session.close()


def test_callback_status_fails_closed_without_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()
    runtime.streams[0].emit(_pcm_bytes(), status=object())

    with pytest.raises(LocalSpeechAdapterError) as caught:
        session.stop()

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECORDING_FAILED
    assert list((tmp_path / "run").iterdir()) == []


@pytest.mark.parametrize("frames", [True, 3199, 3201, -1, "3200"])
def test_callback_frame_count_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: object
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()
    runtime.streams[0].emit(_pcm_bytes(), frames=frames)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        session.stop()

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECORDING_FAILED
    assert list((tmp_path / "run").iterdir()) == []


@pytest.mark.parametrize("failure_stage", ["constructor", "start"])
def test_stream_start_failure_closes_once_and_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    marker = "SENSITIVE_DEVICE_RUNTIME"
    if failure_stage == "constructor":
        runtime.constructor_failure = RuntimeError(marker)
    else:
        original_factory = runtime.RawInputStream

        def failing_factory(**kwargs: object) -> _FakeInputStream:
            stream = original_factory(**kwargs)
            stream.start_failure = RuntimeError(marker)
            return stream

        runtime.RawInputStream = failing_factory  # type: ignore[method-assign]

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recorder.start()

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECORDING_FAILED
    assert marker not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    if runtime.streams:
        assert runtime.streams[0].calls == ["start", "close"]


@pytest.mark.parametrize("failure_stage", ["stop", "close"])
def test_release_failure_clears_session_and_uses_fixed_cleanup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()
    stream = runtime.streams[0]
    stream.emit(_pcm_bytes())
    setattr(stream, f"{failure_stage}_failure", RuntimeError("SENSITIVE_RELEASE"))

    with pytest.raises(LocalSpeechAdapterError) as caught:
        session.stop()

    assert caught.value.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
    assert "SENSITIVE_RELEASE" not in str(caught.value)
    assert list((tmp_path / "run").iterdir()) == []
    session.close()
    assert stream.calls.count("stop") == 1
    assert stream.calls.count("close") == 1


def test_callback_after_stop_is_ignored_and_second_stop_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()
    stream = runtime.streams[0]
    stream.emit(_pcm_bytes())
    audio = session.stop()

    stream.emit(_pcm_bytes(10))
    with pytest.raises(LocalSpeechAdapterError):
        session.stop()

    audio.close()
    assert list((tmp_path / "run").iterdir()) == []


def test_wav_write_failure_is_fresh_and_leaves_no_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()
    runtime.streams[0].emit(_pcm_bytes())
    marker = "SENSITIVE_WAV_WRITER"

    def fail_write(*_args: object, **_kwargs: object) -> OwnedWavFile:
        raise RuntimeError(marker)

    monkeypatch.setattr(recorder_module, "write_validated_pcm_wav", fail_write)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        session.stop()

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECORDING_FAILED
    assert marker not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert list((tmp_path / "run").iterdir()) == []


def test_failure_traceback_does_not_retain_run_root_or_runtime_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "RECORDER_" + "SENSITIVE_MARKER"
    recorder, runtime = _make_recorder(tmp_path / marker, monkeypatch)
    session = recorder.start()
    runtime.streams[0].emit(_pcm_bytes())
    monkeypatch.setattr(
        recorder_module,
        "write_validated_pcm_wav",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(marker)),
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        session.stop()

    production_locals: list[str] = []
    for frame, _lineno in traceback.walk_tb(caught.value.__traceback__):
        if Path(frame.f_code.co_filename).name == "sounddevice_recorder.py":
            production_locals.extend(repr(value) for value in frame.f_locals.values())
    assert marker not in "".join(production_locals)
    assert caught.value.__context__ is None


def test_recorder_uses_the_same_runtime_that_resolved_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )
    first = _FakeSoundDevice()
    second = _FakeSoundDevice()
    runtimes = iter((first, second))
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: next(runtimes))
    monkeypatch.setattr(recorder_module, "_load_sounddevice_runtime", lambda: next(runtimes))

    recorder = SoundDeviceAudioRecorder.from_config(config)
    session = recorder.start()

    assert len(first.streams) == 1
    assert second.streams == []
    session.abort()


def test_concurrent_start_opens_only_one_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, runtime = _make_recorder(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    original_factory = runtime.RawInputStream

    def synchronized_factory(**kwargs: object) -> _FakeInputStream:
        with suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=0.5)
        return original_factory(**kwargs)

    runtime.RawInputStream = synchronized_factory  # type: ignore[method-assign]
    results: list[object] = []

    def invoke_start() -> None:
        try:
            results.append(recorder.start())
        except Exception as error:
            results.append(error)

    threads = [threading.Thread(target=invoke_start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert len(runtime.streams) == 1
    assert sum(type(item) is recorder_module.SoundDeviceRecordingSession for item in results) == 1
    assert sum(type(item) is LocalSpeechAdapterError for item in results) == 1
    session = next(
        item for item in results if type(item) is recorder_module.SoundDeviceRecordingSession
    )
    session.abort()  # type: ignore[union-attr]


def test_recorder_and_session_satisfy_existing_runtime_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, _runtime = _make_recorder(tmp_path, monkeypatch)
    session = recorder.start()

    assert isinstance(recorder, AudioRecorder)
    assert isinstance(session, RecordingSession)
    assert callable(session.stop)
    assert callable(session.abort)
    assert callable(session.close)
    session.abort()


def test_recorder_rejects_a_second_active_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, _runtime = _make_recorder(tmp_path, monkeypatch)
    first = recorder.start()

    with pytest.raises(LocalSpeechAdapterError) as caught:
        recorder.start()

    assert caught.value.code is LocalSpeechAdapterErrorCode.RECORDING_FAILED
    first.abort()
    first.close()
    second = recorder.start()
    second.abort()
    second.close()


def test_from_resolved_constructs_without_re_resolution_or_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )
    runtime = _FakeSoundDevice()
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)

    resolved = devices_module._resolve_audio_runtime(config)
    assert runtime.query_calls == 1

    recorder = SoundDeviceAudioRecorder.from_resolved(config, resolved)

    assert type(recorder) is SoundDeviceAudioRecorder
    assert runtime.query_calls == 1
    assert runtime.input_open_calls == 0
    assert runtime.output_open_calls == 0

    session = recorder.start()
    assert runtime.stream_kwargs == [
        {
            "device": 0,
            "samplerate": 16_000,
            "channels": 1,
            "dtype": "int16",
            "callback": runtime.stream_kwargs[0]["callback"],
        }
    ]
    assert runtime.input_open_calls == 1
    session.abort()
    session.close()


def test_from_resolved_rejects_mismatched_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )
    runtime = _FakeSoundDevice()
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)
    resolved = devices_module._resolve_audio_runtime(config)

    other_root = tmp_path / "other"
    other_root.mkdir()
    other = LocalAudioDeviceConfig(
        run_root=other_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        SoundDeviceAudioRecorder.from_resolved(other, resolved)

    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED
    assert runtime.query_calls == 1
    assert runtime.input_open_calls == 0


def test_from_resolved_scrubs_hostile_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "HOSTILE_FROM_RESOLVED"
    run_root = tmp_path / "run"
    run_root.mkdir()
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )

    class HostileResolved:
        def __repr__(self) -> str:
            return marker

    with pytest.raises(LocalSpeechAdapterError) as caught:
        SoundDeviceAudioRecorder.from_resolved(config, HostileResolved())

    production_locals: list[str] = []
    for frame, _lineno in traceback.walk_tb(caught.value.__traceback__):
        if Path(frame.f_code.co_filename).name == "sounddevice_recorder.py":
            production_locals.extend(repr(value) for value in frame.f_locals.values())
    assert marker not in str(caught.value)
    assert marker not in "".join(production_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
