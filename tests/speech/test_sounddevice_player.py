from __future__ import annotations

import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import companion.speech.local_audio_devices as devices_module
import companion.speech.sounddevice_player as player_module
from companion.speech.controller_ports import AudioPlayer
from companion.speech.local_audio_devices import LocalAudioDeviceConfig
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.pcm_audio import PcmAudio
from companion.speech.sounddevice_player import PlaybackDeviceEvent, SoundDeviceAudioPlayer


class _FakeOutputStream:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.writes: list[bytes] = []
        self.start_failure: Exception | None = None
        self.write_failure: Exception | None = None
        self.stop_failure: Exception | None = None
        self.close_failure: Exception | None = None
        self.write_gate: threading.Event | None = None
        self.start_gate: threading.Event | None = None
        self.stop_gate: threading.Event | None = None
        self.close_gate: threading.Event | None = None
        self.start_entered = threading.Event()
        self.stop_entered = threading.Event()
        self.close_entered = threading.Event()
        self.write_entered = threading.Event()
        self.closed = threading.Event()
        self.aborted = threading.Event()

    def start(self) -> object:
        self.calls.append("start")
        self.start_entered.set()
        if self.start_gate is not None:
            self.start_gate.wait(timeout=2)
        if self.start_failure is not None:
            raise self.start_failure
        return self

    def write(self, payload: bytes) -> object:
        self.calls.append("write")
        self.write_entered.set()
        if self.write_gate is not None:
            while not self.write_gate.wait(timeout=0.01):
                if self.aborted.is_set():
                    raise RuntimeError("aborted")
        if self.write_failure is not None:
            raise self.write_failure
        self.writes.append(bytes(payload))
        return None

    def abort(self) -> object:
        self.calls.append("abort")
        self.aborted.set()
        return self

    def stop(self) -> object:
        self.calls.append("stop")
        self.stop_entered.set()
        if self.stop_gate is not None:
            self.stop_gate.wait(timeout=2)
        if self.stop_failure is not None:
            raise self.stop_failure
        return self

    def close(self) -> object:
        self.calls.append("close")
        self.close_entered.set()
        if self.close_gate is not None:
            self.close_gate.wait(timeout=2)
        self.closed.set()
        if self.close_failure is not None:
            raise self.close_failure
        return self


class _FakeSoundDevice:
    def __init__(self) -> None:
        self.streams: list[_FakeOutputStream] = []
        self.stream_kwargs: list[dict[str, object]] = []
        self.next_stream: _FakeOutputStream | None = None
        self.factory_failure: Exception | None = None
        self.query_calls = 0
        self.input_checks: list[dict[str, object]] = []
        self.output_checks: list[dict[str, object]] = []
        self.input_open_calls = 0
        self.output_open_calls = 0
        self.unsupported_output_rates: set[int] = set()

    def query_devices(self) -> list[dict[str, object]]:
        self.query_calls += 1
        return [
            {"name": "synthetic-input", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "synthetic-output", "max_input_channels": 0, "max_output_channels": 1},
        ]

    def check_input_settings(self, **_kwargs: object) -> None:
        self.input_checks.append(_kwargs)
        return None

    def check_output_settings(self, **_kwargs: object) -> None:
        self.output_checks.append(_kwargs)
        if _kwargs.get("samplerate") in self.unsupported_output_rates:
            raise ValueError("unsupported")
        return None

    def RawInputStream(self, **_kwargs: object) -> object:
        self.input_open_calls += 1
        return object()

    def RawOutputStream(self, **kwargs: object) -> _FakeOutputStream:
        self.output_open_calls += 1
        if self.factory_failure is not None:
            raise self.factory_failure
        self.stream_kwargs.append(kwargs)
        stream = self.next_stream or _FakeOutputStream()
        self.next_stream = None
        self.streams.append(stream)
        return stream


def _pcm_audio(samples: tuple[int, ...], sample_rate_hz: int) -> PcmAudio:
    return PcmAudio(
        pcm=struct.pack(f"<{len(samples)}h", *samples),
        sample_rate_hz=sample_rate_hz,
    )


def _make_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, frames: int = 2500
) -> tuple[SoundDeviceAudioPlayer, _FakeSoundDevice, PcmAudio, tuple[int, ...]]:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    config = LocalAudioDeviceConfig(
        run_root=run_root,
        input_selector="synthetic-input",
        output_selector="synthetic-output",
    )
    runtime = _FakeSoundDevice()
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(player_module, "_load_sounddevice_runtime", lambda: runtime)
    player = SoundDeviceAudioPlayer.from_config(config)
    samples = tuple(-2000 if index % 2 == 0 else 2000 for index in range(frames))
    audio = _pcm_audio(samples, 44_100)
    return player, runtime, audio, samples


def _wait(
    runtime: _FakeSoundDevice, player: SoundDeviceAudioPlayer
) -> tuple[PlaybackDeviceEvent, ...]:
    deadline = time.monotonic() + 2
    while not runtime.streams and time.monotonic() < deadline:
        time.sleep(0.001)
    assert runtime.streams
    assert runtime.streams[-1].closed.wait(timeout=2)
    events: tuple[PlaybackDeviceEvent, ...] = ()
    deadline = time.monotonic() + 2
    while not events and time.monotonic() < deadline:
        events = player.drain_events()
        if not events:
            time.sleep(0.001)
    return events


def test_start_writes_all_pcm_in_order_and_queues_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, samples = _make_stack(tmp_path, monkeypatch)

    assert player.start(audio, 9) is None
    events = _wait(runtime, player)

    expected = b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples)
    assert b"".join(runtime.streams[0].writes) == expected
    assert events == (PlaybackDeviceEvent(playback_token=9, succeeded=True),)
    assert runtime.streams[0].calls[0] == "start"
    assert runtime.streams[0].calls[-2:] == ["stop", "close"]
    assert runtime.stream_kwargs == [
        {"device": 1, "samplerate": 44_100, "channels": 1, "dtype": "int16"}
    ]
    audio.close()


def test_player_resamples_to_resolved_native_48khz_before_open(
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
    runtime.unsupported_output_rates.add(44_100)
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(player_module, "_load_sounddevice_runtime", lambda: runtime)
    calls: list[tuple[tuple[int, ...], int, int]] = []

    def resample(samples: tuple[int, ...], source: int, target: int) -> tuple[int, ...]:
        calls.append((samples, source, target))
        return (-1000, 1000, -1000, 1000)

    monkeypatch.setattr(player_module, "resample_pcm16", resample)
    player = SoundDeviceAudioPlayer.from_config(config)
    audio = _pcm_audio((-2000, 2000), 44_100)

    player.start(audio, 1)
    assert _wait(runtime, player) == (PlaybackDeviceEvent(1, True),)

    assert calls == [((-2000, 2000), 44_100, 48_000)]
    assert runtime.stream_kwargs[0]["samplerate"] == 48_000
    expected = struct.pack("<4h", -1000, 1000, -1000, 1000)
    assert b"".join(runtime.streams[0].writes) == expected
    audio.close()


def test_player_resample_failure_happens_before_output_open(
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
    runtime.unsupported_output_rates.add(44_100)
    monkeypatch.setattr(devices_module, "_load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(player_module, "_load_sounddevice_runtime", lambda: runtime)
    monkeypatch.setattr(
        player_module,
        "resample_pcm16",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("SENSITIVE_PCM")),
    )
    player = SoundDeviceAudioPlayer.from_config(config)
    audio = _pcm_audio((-2000, 2000), 44_100)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        player.start(audio, 1)

    assert caught.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED
    assert caught.value.__cause__ is None
    assert runtime.output_open_calls == 0
    audio.close()


def test_player_is_nonblocking_and_stop_discards_late_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    gate = threading.Event()
    stream = _FakeOutputStream()
    stream.write_gate = gate
    runtime.next_stream = stream

    player.start(audio, 1)
    deadline = time.monotonic() + 2
    while "write" not in stream.calls and time.monotonic() < deadline:
        time.sleep(0.001)
    assert player.drain_events() == ()
    gate.set()
    player.stop()

    assert stream.closed.wait(timeout=2)
    assert player.drain_events() == ()
    assert stream.calls.count("abort") == 1
    assert stream.calls.count("stop") == 0
    assert stream.calls.count("close") == 1
    audio.close()


def test_stop_interrupts_a_blocked_write_and_releases_stream_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    stream = _FakeOutputStream()
    stream.write_gate = threading.Event()
    runtime.next_stream = stream
    player.start(audio, 1)
    assert stream.write_entered.wait(timeout=2)

    started = time.monotonic()
    player.stop()

    assert time.monotonic() - started < 1
    assert stream.calls.count("abort") == 1
    assert stream.calls.count("stop") == 0
    assert stream.calls.count("close") == 1
    assert stream.closed.is_set()
    assert player.drain_events() == ()
    audio.close()


def test_cancelled_start_cannot_take_a_later_start_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    original = player_module.pcm16_samples
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def delayed_read(candidate: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=2)
        return original(candidate)  # type: ignore[arg-type]

    monkeypatch.setattr(player_module, "pcm16_samples", delayed_read)
    failures: list[Exception] = []

    def start_first() -> None:
        try:
            player.start(audio, 1)
        except Exception as error:
            failures.append(error)

    first = threading.Thread(target=start_first)
    first.start()
    assert entered.wait(timeout=2)
    player.stop()
    player.start(audio, 2)
    release.set()
    first.join(timeout=2)
    events = _wait(runtime, player)

    assert not first.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], LocalSpeechAdapterError)
    assert len(runtime.streams) == 1
    assert events == (PlaybackDeviceEvent(2, True),)
    audio.close()


def test_stop_cannot_join_a_thread_before_it_is_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    stream = _FakeOutputStream()
    stream.write_gate = threading.Event()
    runtime.next_stream = stream
    real_thread = threading.Thread
    publish_entered = threading.Event()
    allow_start = threading.Event()

    class DelayedThread(real_thread):
        def start(self) -> None:
            publish_entered.set()
            assert allow_start.wait(timeout=2)
            super().start()

    monkeypatch.setattr(
        player_module,
        "threading",
        SimpleNamespace(Thread=DelayedThread, Event=threading.Event, Lock=threading.Lock),
    )
    failures: list[Exception] = []
    starter = real_thread(target=lambda: _capture_failure(lambda: player.start(audio, 1), failures))
    starter.start()
    assert publish_entered.wait(timeout=2)
    stopper = real_thread(target=lambda: _capture_failure(player.stop, failures))
    stopper.start()
    time.sleep(0.02)
    allow_start.set()
    starter.join(timeout=2)
    stopper.join(timeout=2)

    assert not starter.is_alive()
    assert not stopper.is_alive()
    assert failures == []
    assert all(stream.closed.is_set() for stream in runtime.streams)
    assert all(stream.writes == [] for stream in runtime.streams)
    assert player.drain_events() == ()
    audio.close()


def _capture_failure(action: object, failures: list[Exception]) -> None:
    try:
        action()  # type: ignore[operator]
    except Exception as error:
        failures.append(error)


@pytest.mark.parametrize("token", [True, 0, -1, 1.0, "1"])
def test_invalid_token_fails_before_worker_or_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: object
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        player.start(audio, token)  # type: ignore[arg-type]

    assert caught.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED
    assert runtime.streams == []
    audio.close()


def test_int_subclass_token_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Token(int):
        pass

    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)

    with pytest.raises(LocalSpeechAdapterError):
        player.start(audio, Token(1))

    assert runtime.streams == []
    audio.close()


def test_wrong_or_closed_audio_fails_before_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)

    with pytest.raises(LocalSpeechAdapterError):
        player.start(object(), 1)
    audio.close()
    with pytest.raises(LocalSpeechAdapterError):
        player.start(audio, 1)

    assert runtime.streams == []


def test_stream_factory_failure_queues_fixed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    runtime.factory_failure = RuntimeError("SENSITIVE_STREAM_FACTORY")

    player.start(audio, 1)
    deadline = time.monotonic() + 2
    events: tuple[PlaybackDeviceEvent, ...] = ()
    while not events and time.monotonic() < deadline:
        events = player.drain_events()
        time.sleep(0.001)

    assert events == (PlaybackDeviceEvent(1, False),)
    assert "SENSITIVE" not in repr(events)
    audio.close()


def test_thread_start_failure_scrubs_payload_and_requires_a_new_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    real_thread = threading.Thread

    class FailingThread(real_thread):
        def start(self) -> None:
            raise RuntimeError("SENSITIVE_THREAD_START")

    monkeypatch.setattr(
        player_module,
        "threading",
        SimpleNamespace(Thread=FailingThread, Event=threading.Event, Lock=threading.Lock),
    )
    with pytest.raises(LocalSpeechAdapterError) as caught:
        player.start(audio, 1)

    assert runtime.streams == []
    assert str(caught.value) == LocalSpeechAdapterErrorCode.PLAYBACK_FAILED.value
    monkeypatch.setattr(player_module, "threading", threading)
    with pytest.raises(LocalSpeechAdapterError):
        player.start(audio, 1)
    player.start(audio, 2)
    assert _wait(runtime, player) == (PlaybackDeviceEvent(2, True),)
    audio.close()


def test_output_chunks_are_pcm16_aligned_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch, frames=2501)

    player.start(audio, 1)
    assert _wait(runtime, player) == (PlaybackDeviceEvent(1, True),)

    assert len(runtime.streams[0].writes) == 3
    assert all(0 < len(chunk) <= 2048 for chunk in runtime.streams[0].writes)
    assert all(len(chunk) % 2 == 0 for chunk in runtime.streams[0].writes)
    audio.close()


def _production_traceback_retains_pcm(error: BaseException, pcm_marker: bytes) -> bool:
    traceback = error.__traceback__
    while traceback is not None:
        if Path(traceback.tb_frame.f_code.co_filename).name == "sounddevice_player.py":
            for value in traceback.tb_frame.f_locals.values():
                if type(value) is bytes and pcm_marker in value:
                    return True
                if type(value) is player_module._PlaybackRun and pcm_marker in value.payload:
                    return True
        traceback = traceback.tb_next
    return False


def test_start_failure_traceback_does_not_retain_pcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    marker = "SENSITIVE_PCM_MARKER"

    def fail_settings(**_kwargs: object) -> None:
        raise RuntimeError(marker)

    runtime.check_output_settings = fail_settings  # type: ignore[method-assign]
    pcm_marker = (2000).to_bytes(2, "little", signed=True) * 8

    with pytest.raises(LocalSpeechAdapterError) as caught:
        player.start(audio, 1)

    assert not _production_traceback_retains_pcm(caught.value, pcm_marker)
    assert runtime.streams == []
    audio.close()


def test_join_timeout_traceback_scrubs_run_payload_and_later_stop_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)

    class UninterruptibleStream(_FakeOutputStream):
        def write(self, payload: bytes) -> object:
            self.calls.append("write")
            self.write_entered.set()
            assert self.write_gate is not None
            assert self.write_gate.wait(timeout=2)
            return None

    stream = UninterruptibleStream()
    stream.write_gate = threading.Event()
    runtime.next_stream = stream
    monkeypatch.setattr(player_module, "_JOIN_TIMEOUT_SECONDS", 0.01)
    player.start(audio, 1)
    assert stream.write_entered.wait(timeout=2)
    pcm_marker = (2000).to_bytes(2, "little", signed=True) * 8

    with pytest.raises(LocalSpeechAdapterError) as caught:
        player.stop()

    assert not _production_traceback_retains_pcm(caught.value, pcm_marker)
    assert stream.calls.count("abort") == 1
    assert not stream.closed.is_set()
    stream.write_gate.set()
    player.stop()
    assert stream.closed.wait(timeout=2)
    assert stream.calls.count("abort") == 1
    assert stream.calls.count("close") == 1
    audio.close()


def test_stop_remains_bounded_while_stream_start_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    stream = _FakeOutputStream()
    stream.start_gate = threading.Event()
    runtime.next_stream = stream
    monkeypatch.setattr(player_module, "_JOIN_TIMEOUT_SECONDS", 0.01)
    player.start(audio, 1)
    assert stream.start_entered.wait(timeout=2)

    started = time.monotonic()
    with pytest.raises(LocalSpeechAdapterError):
        player.stop()

    assert time.monotonic() - started < 0.2
    assert stream.calls.count("abort") == 1
    stream.start_gate.set()
    player.stop()
    assert stream.closed.wait(timeout=2)
    assert stream.writes == []
    assert player.drain_events() == ()
    audio.close()


def test_stop_remains_bounded_while_worker_cleanup_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    stream = _FakeOutputStream()
    stream.stop_gate = threading.Event()
    runtime.next_stream = stream
    monkeypatch.setattr(player_module, "_JOIN_TIMEOUT_SECONDS", 0.01)
    player.start(audio, 1)
    assert stream.stop_entered.wait(timeout=2)

    started = time.monotonic()
    with pytest.raises(LocalSpeechAdapterError):
        player.stop()

    assert time.monotonic() - started < 0.2
    stream.stop_gate.set()
    player.stop()
    assert stream.closed.wait(timeout=2)
    assert stream.calls.count("stop") == 1
    assert stream.calls.count("close") == 1
    assert player.drain_events() == ()
    audio.close()


def test_interrupt_wait_timeout_defers_close_until_abort_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    entered = threading.Event()

    class SlowAbortStream(_FakeOutputStream):
        def abort(self) -> object:
            self.calls.append("abort")
            entered.set()
            assert gate.wait(timeout=2)
            return self

    stream = SlowAbortStream()
    run = player_module._PlaybackRun(
        playback_token=1,
        payload=b"",
        sample_rate_hz=16000,
        stop_event=threading.Event(),
        stream=stream,
        stream_started=True,
    )
    monkeypatch.setattr(player_module, "_JOIN_TIMEOUT_SECONDS", 0.01)
    thread = threading.Thread(target=player_module._interrupt_output_stream, args=(run,))
    thread.start()
    try:
        assert entered.wait(timeout=2)
        assert player_module._release_output_stream(run) is False
        assert not stream.closed.is_set()
    finally:
        gate.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert stream.closed.is_set()
    assert player_module._release_output_stream(run) is True
    assert stream.calls == ["abort", "close"]


def test_token_must_be_strictly_increasing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    player.start(audio, 2)
    assert _wait(runtime, player) == (PlaybackDeviceEvent(2, True),)

    with pytest.raises(LocalSpeechAdapterError):
        player.start(audio, 2)
    with pytest.raises(LocalSpeechAdapterError):
        player.start(audio, 1)

    assert len(runtime.streams) == 1
    audio.close()


def test_second_active_playback_fails_without_second_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    gate = threading.Event()
    stream = _FakeOutputStream()
    stream.write_gate = gate
    runtime.next_stream = stream
    player.start(audio, 1)

    with pytest.raises(LocalSpeechAdapterError):
        player.start(audio, 2)

    assert len(runtime.streams) == 1
    gate.set()
    player.stop()
    assert stream.closed.wait(timeout=2)
    audio.close()


@pytest.mark.parametrize("stage", ["start", "write", "stop", "close"])
def test_worker_failure_queues_only_a_fixed_failure_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    stream = _FakeOutputStream()
    setattr(stream, f"{stage}_failure", RuntimeError("SENSITIVE_PLAYER_RUNTIME"))
    runtime.next_stream = stream

    player.start(audio, 5)
    events = _wait(runtime, player)

    assert events == (PlaybackDeviceEvent(5, False),)
    assert "SENSITIVE" not in repr(events)
    assert stream.calls.count("close") == 1
    audio.close()


def test_stop_is_idempotent_and_never_closes_caller_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)
    player.start(audio, 1)
    _wait(runtime, player)

    player.stop()
    player.stop()

    assert len(runtime.streams) == 1
    player.start(audio, 2)
    assert _wait(runtime, player) == (PlaybackDeviceEvent(2, True),)
    audio.close()


def test_player_uses_same_runtime_that_resolved_output(
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
    monkeypatch.setattr(player_module, "_load_sounddevice_runtime", lambda: next(runtimes))
    player = SoundDeviceAudioPlayer.from_config(config)
    audio = _pcm_audio((1000, -1000) * 1000, 44_100)

    player.start(audio, 1)
    _wait(first, player)

    assert len(first.streams) == 1
    assert second.streams == []
    audio.close()


def test_player_satisfies_existing_runtime_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player, _runtime, audio, _samples = _make_stack(tmp_path, monkeypatch)

    assert isinstance(player, AudioPlayer)
    assert callable(player.start)
    assert callable(player.stop)
    player.start(audio, 1)
    _wait(_runtime, player)
    audio.close()


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

    player = SoundDeviceAudioPlayer.from_resolved(config, resolved)

    assert type(player) is SoundDeviceAudioPlayer
    assert runtime.query_calls == 1
    assert runtime.input_open_calls == 0
    assert runtime.output_open_calls == 0

    audio = _pcm_audio((1000, -1000) * 1000, 44_100)
    player.start(audio, 1)
    assert _wait(runtime, player) == (PlaybackDeviceEvent(1, True),)
    assert runtime.stream_kwargs == [
        {"device": 1, "samplerate": 44_100, "channels": 1, "dtype": "int16"}
    ]
    assert runtime.output_open_calls == 1
    audio.close()


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
        SoundDeviceAudioPlayer.from_resolved(other, resolved)

    assert caught.value.code is LocalSpeechAdapterErrorCode.DEVICE_CONFIGURATION_FAILED
    assert runtime.query_calls == 1
    assert runtime.output_open_calls == 0
