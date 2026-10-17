import threading

import pytest

from companion.bodylink.body_devices import SoundDeviceSpeaker


class OutputStream:
    def __init__(self, runtime: "OutputRuntime") -> None:
        self.runtime = runtime
        self.calls: list[str] = []
        self.close_threads: list[int] = []

    def start(self) -> None:
        self.calls.append("start")
        self.runtime.at("start")

    def write(self, data: bytes) -> None:
        self.calls.append("write")
        self.runtime.at("write")

    def stop(self) -> None:
        self.calls.append("stop")
        self.runtime.at("stop")

    def abort(self) -> None:
        self.calls.append("abort")
        self.runtime.released.set()
        self.runtime.at("abort")

    def close(self) -> None:
        self.calls.append("close")
        self.close_threads.append(threading.get_ident())
        self.runtime.at("close")


class OutputRuntime:
    def __init__(self, *, block: str | None = None, error_at: str | None = None) -> None:
        self.block = block
        self.error_at = error_at
        self.error = RuntimeError("synthetic output error")
        self.stop_on_error: threading.Event | None = None
        self.entered = threading.Event()
        self.released = threading.Event()
        self.streams: list[OutputStream] = []

    def at(self, stage: str) -> None:
        if stage == self.block:
            self.entered.set()
            assert self.released.wait(5), "test did not release output"
        if stage == self.error_at:
            if self.stop_on_error is not None:
                self.stop_on_error.set()
            raise self.error

    def RawOutputStream(self, **kwargs: object) -> OutputStream:
        stream = OutputStream(self)
        self.streams.append(stream)
        self.at("create")
        return stream


def start_playback(speaker: SoundDeviceSpeaker, stop: threading.Event):
    results: list[bool] = []
    errors: list[Exception] = []

    def play() -> None:
        try:
            results.append(speaker.play(b"\0\0" * 2048, 44100, stop))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=play, daemon=True)
    thread.start()
    return thread, results, errors


@pytest.mark.parametrize("device_error", [False, True])
def test_abort_interrupts_blocked_write(device_error: bool) -> None:
    _check_abort_interrupts("write", device_error)


@pytest.mark.parametrize("device_error", [False, True])
def test_abort_interrupts_drain(device_error: bool) -> None:
    _check_abort_interrupts("stop", device_error)


def _check_abort_interrupts(stage: str, device_error: bool) -> None:
    runtime = OutputRuntime(block=stage, error_at=stage if device_error else None)
    speaker = SoundDeviceSpeaker(runtime, 1)
    stop = threading.Event()
    thread, results, errors = start_playback(speaker, stop)
    try:
        assert runtime.entered.wait(2)
        stop.set()  # the fake remains blocked until stream.abort is called
        speaker.abort()
        thread.join(2)
        assert not thread.is_alive()
        assert results == [False]
        assert errors == []
        stream = runtime.streams[0]
        assert stream.calls.count("abort") == stream.calls.count("close") == 1
        assert stream.close_threads == [thread.ident]
    finally:
        runtime.released.set()
        thread.join(2)


def test_stop_before_stream_start_opens_no_output() -> None:
    runtime = OutputRuntime()
    stop = threading.Event()
    stop.set()
    assert SoundDeviceSpeaker(runtime, 1).play(b"\0\0", 44100, stop) is False
    assert runtime.streams == []


def test_stop_during_stream_creation_never_starts_output() -> None:
    runtime = OutputRuntime(block="create")
    speaker = SoundDeviceSpeaker(runtime, 1)
    stop = threading.Event()
    thread, results, errors = start_playback(speaker, stop)
    try:
        assert runtime.entered.wait(2)
        stop.set()
        speaker.abort()  # not yet published
        runtime.released.set()
        thread.join(2)
        assert not thread.is_alive()
        assert results == [False]
        assert errors == []
        assert runtime.streams[0].calls == ["close"]
        assert runtime.streams[0].close_threads == [thread.ident]
    finally:
        runtime.released.set()
        thread.join(2)


def test_repeated_abort_closes_once() -> None:
    runtime = OutputRuntime(block="write")
    speaker = SoundDeviceSpeaker(runtime, 1)
    stop = threading.Event()
    thread, results, errors = start_playback(speaker, stop)
    try:
        assert runtime.entered.wait(2)
        stop.set()
        for _ in range(3):
            speaker.abort()
        thread.join(2)
        assert not thread.is_alive()
        assert results == [False]
        assert errors == []
        assert runtime.streams[0].calls.count("abort") == 1
        assert runtime.streams[0].calls.count("close") == 1
    finally:
        runtime.released.set()
        thread.join(2)


def test_abort_after_completion_is_noop() -> None:
    runtime = OutputRuntime()
    speaker = SoundDeviceSpeaker(runtime, 1)
    assert speaker.play(b"\0\0", 44100, threading.Event()) is True
    speaker.abort()
    assert runtime.streams[0].calls == ["start", "write", "stop", "close"]


def test_normal_playback_does_not_abort() -> None:
    runtime = OutputRuntime()
    assert SoundDeviceSpeaker(runtime, 1).play(b"\0\0", 44100, threading.Event()) is True
    assert runtime.streams[0].calls == ["start", "write", "stop", "close"]


def test_stop_between_writes_requests_abort(monkeypatch) -> None:
    runtime = OutputRuntime()
    stop = threading.Event()

    def stop_after_write(stage: str) -> None:
        if stage == "write":
            stop.set()

    monkeypatch.setattr(runtime, "at", stop_after_write)
    assert SoundDeviceSpeaker(runtime, 1).play(b"\0" * 4096, 44100, stop) is False
    assert runtime.streams[0].calls == ["start", "write", "abort", "close"]


@pytest.mark.parametrize("stage", ["create", "start", "write", "stop"])
def test_unrequested_device_error_propagates(stage: str) -> None:
    runtime = OutputRuntime(error_at=stage)
    with pytest.raises(RuntimeError) as caught:
        SoundDeviceSpeaker(runtime, 1).play(b"\0\0", 44100, threading.Event())
    assert caught.value is runtime.error
    assert runtime.streams[0].calls.count("close") == (0 if stage == "create" else 1)


@pytest.mark.parametrize("stage", ["create", "start", "write", "stop"])
def test_requested_abort_device_error_returns_false(stage: str) -> None:
    runtime = OutputRuntime(error_at=stage)
    stop = threading.Event()
    runtime.stop_on_error = stop
    assert SoundDeviceSpeaker(runtime, 1).play(b"\0\0", 44100, stop) is False
    assert runtime.streams[0].calls.count("close") == (0 if stage == "create" else 1)


@pytest.mark.parametrize("requested", [False, True])
def test_close_error_propagates(requested: bool) -> None:
    runtime = OutputRuntime(error_at="close")
    if requested:
        runtime.stop_on_error = threading.Event()
    stop = runtime.stop_on_error if requested else threading.Event()
    with pytest.raises(RuntimeError) as caught:
        SoundDeviceSpeaker(runtime, 1).play(b"\0\0", 44100, stop)
    assert caught.value is runtime.error
    assert runtime.streams[0].calls.count("close") == 1


def test_abort_error_propagates_without_retry_or_close() -> None:
    runtime = OutputRuntime(block="write", error_at="abort")
    speaker = SoundDeviceSpeaker(runtime, 1)
    stop = threading.Event()
    thread, results, errors = start_playback(speaker, stop)
    try:
        assert runtime.entered.wait(2)
        stop.set()
        with pytest.raises(RuntimeError) as caught:
            speaker.abort()
        assert caught.value is runtime.error
        speaker.abort()
        thread.join(2)
        assert not thread.is_alive()
        assert results == [False]
        assert errors == []
        assert runtime.streams[0].calls.count("abort") == 1
        assert runtime.streams[0].calls.count("close") == 1
        assert runtime.streams[0].close_threads == [thread.ident]
    finally:
        runtime.released.set()
        thread.join(2)
