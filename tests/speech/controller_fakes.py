from companion.contracts import DialogueResponse, TranscriptConfidenceStatus
from companion.speech.controller_models import RecognitionInput
from companion.speech.recognition_gate import AudioQualityEvidence

_DEFAULT_DIALOGUE_RESPONSE = object()
_DEFAULT_RESOURCE_RESULT = object()


class RecordedAudioFake:
    def __init__(
        self,
        calls: list[str],
        callbacks: dict[str, object],
        failures: dict[str, Exception],
        results: dict[str, object],
    ) -> None:
        self._calls = calls
        self._callbacks = callbacks
        self._failures = failures
        self._results = results
        self.live_count = 1
        self.close_calls = 0

    def close(self) -> None:
        self._calls.append("audio.close")
        self.close_calls += 1
        callback = self._callbacks.get("audio.close")
        if callable(callback):
            callback()
        failure = self._failures.get("audio.close")
        if failure is not None:
            raise failure
        self.live_count -= 1
        return self._results.get("audio.close")  # type: ignore[return-value]

    def __repr__(self) -> str:
        return "RecordedAudioFake()"


class RecordingSessionFake:
    def __init__(
        self,
        calls: list[str],
        recorded_audios: list[RecordedAudioFake],
        callbacks: dict[str, object],
        failures: dict[str, Exception],
        results: dict[str, object],
    ) -> None:
        self._calls = calls
        self._recorded_audios = recorded_audios
        self._callbacks = callbacks
        self._failures = failures
        self._results = results
        self.live_count = 1
        self.close_calls = 0

    def stop(self) -> RecordedAudioFake:
        self._calls.append("stop")
        callback = self._callbacks.get("session.stop")
        if callable(callback):
            callback()
        failure = self._failures.get("session.stop")
        if failure is not None:
            raise failure
        if "session.stop" in self._results:
            return self._results["session.stop"]  # type: ignore[return-value]
        recorded_audio = RecordedAudioFake(
            self._calls,
            self._callbacks,
            self._failures,
            self._results,
        )
        self._recorded_audios.append(recorded_audio)
        return recorded_audio

    def abort(self) -> None:
        self._calls.append("abort")
        callback = self._callbacks.get("session.abort")
        if callable(callback):
            callback()
        failure = self._failures.get("session.abort")
        if failure is not None:
            raise failure
        return self._results.get("session.abort")  # type: ignore[return-value]

    def close(self) -> None:
        self._calls.append("session.close")
        self.close_calls += 1
        callback = self._callbacks.get("session.close")
        if callable(callback):
            callback()
        failure = self._failures.get("session.close")
        if failure is not None:
            raise failure
        self.live_count -= 1
        return self._results.get("session.close")  # type: ignore[return-value]

    def __repr__(self) -> str:
        return "RecordingSessionFake()"


class AudioRecorderFake:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.recorded_audios: list[RecordedAudioFake] = []
        self.sessions: list[RecordingSessionFake] = []
        self._callbacks: dict[str, object] = {}
        self._failures: dict[str, Exception] = {}
        self._results: dict[str, object] = {}

    def configure_callback(self, method: str, callback: object) -> None:
        self._callbacks[method] = callback

    def configure_failure(self, method: str, failure: Exception) -> None:
        self._failures[method] = failure

    def configure_result(self, method: str, result: object) -> None:
        self._results[method] = result

    @property
    def live_session_count(self) -> int:
        return sum(session.live_count for session in self.sessions)

    @property
    def live_recorded_audio_count(self) -> int:
        return sum(audio.live_count for audio in self.recorded_audios)

    @property
    def session_close_calls(self) -> int:
        return sum(session.close_calls for session in self.sessions)

    @property
    def recorded_audio_close_calls(self) -> int:
        return sum(audio.close_calls for audio in self.recorded_audios)

    def start(self) -> RecordingSessionFake:
        self.calls.append("start")
        callback = self._callbacks.get("recorder.start")
        if callable(callback):
            callback()
        failure = self._failures.get("recorder.start")
        if failure is not None:
            raise failure
        if "recorder.start" in self._results:
            return self._results["recorder.start"]  # type: ignore[return-value]
        session = RecordingSessionFake(
            self.calls,
            self.recorded_audios,
            self._callbacks,
            self._failures,
            self._results,
        )
        self.sessions.append(session)
        return session

    def __repr__(self) -> str:
        return "AudioRecorderFake()"


class SpeechRecognizerFake:
    def __init__(
        self,
        *,
        result: RecognitionInput | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.calls = 0
        self._callbacks: dict[str, object] = {}
        self.result = result
        self.failure = failure

    def configure_callback(self, method: str, callback: object) -> None:
        self._callbacks[method] = callback

    def recognize(self, audio: RecordedAudioFake) -> RecognitionInput:
        self.calls += 1
        callback = self._callbacks.get("recognizer.recognize")
        if callable(callback):
            callback()
        if self.failure is not None:
            raise self.failure
        if self.result is not None:
            return self.result
        return RecognitionInput(
            audio=AudioQualityEvidence(
                sample_rate_hz=16_000,
                channels=1,
                sample_width_bytes=2,
                frame_count=4_800,
                rms=0.1,
                peak_abs=0.2,
                clipping_fraction=0.0,
                estimated_snr_db=20.0,
            ),
            vad_passed=True,
            text="synthetic transcript",
            confidence=None,
            confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
            confidence_threshold=None,
        )

    def __repr__(self) -> str:
        return "SpeechRecognizerFake()"


class DialogueFake:
    def __init__(
        self,
        *,
        result: object = _DEFAULT_DIALOGUE_RESPONSE,
        failure: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._callbacks: dict[str, object] = {}
        self.result = result
        self.failure = failure

    def configure_callback(self, method: str, callback: object) -> None:
        self._callbacks[method] = callback

    def reply(self, *, user_id: str, text: str) -> DialogueResponse:
        self.calls.append((user_id, text))
        callback = self._callbacks.get("dialogue.reply")
        if callable(callback):
            callback()
        if self.failure is not None:
            raise self.failure
        if self.result is not _DEFAULT_DIALOGUE_RESPONSE:
            return self.result  # type: ignore[return-value]
        return DialogueResponse(
            text="synthetic safe response",
            disclosed_ai_identity=False,
            billable=False,
            blocked_reason=None,
        )

    def __repr__(self) -> str:
        return "DialogueFake()"


class SynthesizedAudioFake:
    def __init__(
        self,
        callbacks: dict[str, object],
        close_failure: Exception | None = None,
        close_result: object = None,
    ) -> None:
        self.live_count = 1
        self.close_calls = 0
        self._callbacks = callbacks
        self._close_failure = close_failure
        self._close_result = close_result

    def close(self) -> None:
        self.close_calls += 1
        callback = self._callbacks.get("synthesized_audio.close")
        if callable(callback):
            callback()
        if self._close_failure is not None:
            raise self._close_failure
        self.live_count -= 1
        return self._close_result  # type: ignore[return-value]

    def __repr__(self) -> str:
        return "SynthesizedAudioFake()"


class SpeechSynthesizerFake:
    def __init__(
        self,
        *,
        failure: Exception | None = None,
        audio_close_failure: Exception | None = None,
        result: object = _DEFAULT_RESOURCE_RESULT,
        audio_close_result: object = None,
    ) -> None:
        self.inputs: list[str] = []
        self.audios: list[SynthesizedAudioFake] = []
        self._callbacks: dict[str, object] = {}
        self.failure = failure
        self.audio_close_failure = audio_close_failure
        self.result = result
        self.audio_close_result = audio_close_result

    def configure_callback(self, method: str, callback: object) -> None:
        self._callbacks[method] = callback

    @property
    def live_audio_count(self) -> int:
        return sum(audio.live_count for audio in self.audios)

    @property
    def audio_close_calls(self) -> int:
        return sum(audio.close_calls for audio in self.audios)

    def synthesize(self, text: str) -> SynthesizedAudioFake:
        self.inputs.append(text)
        callback = self._callbacks.get("synthesizer.synthesize")
        if callable(callback):
            callback()
        if self.failure is not None:
            raise self.failure
        if self.result is not _DEFAULT_RESOURCE_RESULT:
            return self.result  # type: ignore[return-value]
        audio = SynthesizedAudioFake(
            self._callbacks,
            close_failure=self.audio_close_failure,
            close_result=self.audio_close_result,
        )
        self.audios.append(audio)
        return audio

    def __repr__(self) -> str:
        return "SpeechSynthesizerFake()"


class AudioPlayerFake:
    def __init__(
        self,
        *,
        start_failure: Exception | None = None,
        stop_failure: Exception | None = None,
        start_result: object = None,
        stop_result: object = None,
    ) -> None:
        self.tokens: list[int] = []
        self.stop_calls = 0
        self._callbacks: dict[str, object] = {}
        self.start_failure = start_failure
        self.stop_failure = stop_failure
        self.start_result = start_result
        self.stop_result = stop_result

    def configure_callback(self, method: str, callback: object) -> None:
        self._callbacks[method] = callback

    def start(self, audio: SynthesizedAudioFake, playback_token: int) -> None:
        self.tokens.append(playback_token)
        callback = self._callbacks.get("player.start")
        if callable(callback):
            callback()
        if self.start_failure is not None:
            raise self.start_failure
        return self.start_result  # type: ignore[return-value]

    def stop(self) -> None:
        self.stop_calls += 1
        callback = self._callbacks.get("player.stop")
        if callable(callback):
            callback()
        if self.stop_failure is not None:
            raise self.stop_failure
        return self.stop_result  # type: ignore[return-value]

    def __repr__(self) -> str:
        return "AudioPlayerFake()"


class MuteGateFake:
    def __init__(self, *, enabled: object = True) -> None:
        self.enabled = enabled
        self.raise_on_read: Exception | None = None

    def microphone_enabled(self) -> object:
        if self.raise_on_read is not None:
            raise self.raise_on_read
        return self.enabled


class ClockFake:
    def __init__(self, *, now: object = 0.0) -> None:
        self.now = now
        self.raise_on_read: Exception | None = None

    def monotonic(self) -> object:
        if self.raise_on_read is not None:
            raise self.raise_on_read
        return self.now
