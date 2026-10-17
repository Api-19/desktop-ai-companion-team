import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from decimal import Decimal, localcontext
from fractions import Fraction

from companion.contracts import DialogueResponse
from companion.speech.controller_models import (
    CONNECTION_RETRY_PROMPT,
    EMPTY_REPLY_PROMPT,
    RECOGNITION_RETRY_PROMPT,
    RecognitionInput,
    SpeechControllerIssue,
    SpeechState,
    SpeechTurnFailure,
    TurnTiming,
)
from companion.speech.controller_ports import (
    AudioPlayer,
    AudioRecorder,
    Clock,
    DialogueCallable,
    MuteGate,
    RecordedAudio,
    RecordingSession,
    SpeechRecognizer,
    SpeechSynthesizer,
    SynthesizedAudio,
)
from companion.speech.recognition_gate import RecognitionRejectionReason, gate_recognition
from companion.speech.tts_text import spoken_text


class SpeechController:
    def __init__(
        self,
        *,
        trusted_user_id: str,
        recorder: AudioRecorder,
        recognizer: SpeechRecognizer,
        dialogue: DialogueCallable,
        synthesizer: SpeechSynthesizer,
        player: AudioPlayer,
        mute_gate: MuteGate,
        clock: Clock,
        timing_observer: Callable[[TurnTiming], None] | None = None,
        session_id_supplier: Callable[[], str | None] | None = None,
        session_playback_starter: Callable[[SynthesizedAudio, int, str], None] | None = None,
        processing_observer: Callable[[], None] | None = None,
    ) -> None:
        if (session_id_supplier is None) != (session_playback_starter is None):
            raise ValueError("session callbacks must be supplied together")
        if type(trusted_user_id) is not str:
            raise TypeError("trusted_user_id must be a built-in str")
        if not trusted_user_id.strip() or len(trusted_user_id) > 120:
            raise ValueError("trusted_user_id is invalid")

        self._trusted_user_id = trusted_user_id
        self._recorder = recorder
        self._recognizer = recognizer
        self._dialogue = dialogue
        self._synthesizer = synthesizer
        self._player = player
        self._mute_gate = mute_gate
        self._clock = clock
        self._timing_observer = timing_observer
        self._processing_observer = processing_observer
        self._session_id_supplier = session_id_supplier
        self._session_playback_starter = session_playback_starter
        self._source_session_id: str | None = None
        self._state = SpeechState.IDLE
        self._last_issue: SpeechControllerIssue | None = None
        self._turn_failure: SpeechTurnFailure | None = None
        self._recording_session: RecordingSession | None = None
        self._recording_started_at: int | float | Decimal | Fraction | None = None
        self._synthesized_audio: SynthesizedAudio | None = None
        self._next_playback_token = 0
        self._playback_token: int | None = None
        self._handling_event = False
        self._shutdown = False

    @property
    def state(self) -> SpeechState:
        return self._state

    @property
    def last_issue(self) -> SpeechControllerIssue | None:
        return self._last_issue

    @property
    def turn_failure(self) -> SpeechTurnFailure | None:
        return self._turn_failure

    def button_pressed(self) -> None:
        self._run_event(self._handle_button_pressed)

    def playback_finished(self, playback_token: object) -> None:
        self._run_event(lambda: self._handle_playback_finished(playback_token))

    def playback_failed(self, playback_token: object) -> None:
        self._run_event(lambda: self._handle_playback_failed(playback_token))

    def tick(self) -> None:
        self._run_event(self._handle_tick)

    def mute_changed(self) -> None:
        self._run_event(self._handle_mute_changed)

    def shutdown(self) -> None:
        if self._shutdown or self._handling_event:
            return
        self._handling_event = True
        self._shutdown = True
        cleanup_failed = False
        try:
            try:
                cleanup_failed = not self._shutdown_current_resources()
            except Exception:
                cleanup_failed = True
        finally:
            self._trusted_user_id = None  # type: ignore[assignment]
            self._recorder = None  # type: ignore[assignment]
            self._recognizer = None  # type: ignore[assignment]
            self._dialogue = None  # type: ignore[assignment]
            self._synthesizer = None  # type: ignore[assignment]
            self._player = None  # type: ignore[assignment]
            self._mute_gate = None  # type: ignore[assignment]
            self._clock = None  # type: ignore[assignment]
            self._timing_observer = None
            self._processing_observer = None
            self._session_id_supplier = None
            self._session_playback_starter = None
            self._source_session_id = None
            self._recording_session = None
            self._recording_started_at = None
            self._synthesized_audio = None
            self._playback_token = None
            self._state = SpeechState.IDLE
            self._turn_failure = None
            if cleanup_failed:
                self._last_issue = SpeechControllerIssue.CLEANUP_FAILED
            self._handling_event = False

    def _run_event(self, operation: Callable[[], None]) -> None:
        if self._shutdown or self._handling_event:
            return
        self._handling_event = True
        try:
            operation()
        finally:
            self._handling_event = False

    def _handle_button_pressed(self) -> None:
        if self._state is SpeechState.IDLE:
            self._start_recording()
        elif self._state is SpeechState.RECORDING:
            self._finish_recording()
        elif self._state is SpeechState.PLAYING:
            self._interrupt_playback()

    def _handle_playback_finished(self, playback_token: object) -> None:
        if (
            type(playback_token) is not int
            or self._state is not SpeechState.PLAYING
            or playback_token != self._playback_token
        ):
            return

        self._playback_token = None
        audio = self._detach_synthesized_audio()
        self._state = SpeechState.IDLE
        if audio is not None:
            self._close_synthesized_audio(audio)

    def _handle_playback_failed(self, playback_token: object) -> None:
        if (
            type(playback_token) is not int
            or self._state is not SpeechState.PLAYING
            or playback_token != self._playback_token
        ):
            return

        audio = self._detach_synthesized_audio()
        self._state = SpeechState.IDLE
        self._last_issue = SpeechControllerIssue.PLAYBACK_FAILED
        if audio is not None:
            self._close_synthesized_audio(audio)

    def _handle_tick(self) -> None:
        if (
            self._state in {SpeechState.RECORDING, SpeechState.PLAYING}
            and self._discard_if_session_invalid()
        ):
            return
        if self._state is not SpeechState.RECORDING:
            return
        if not self._microphone_enabled():
            self._abort_recording()
            return

        now = self._validated_now()
        if now is None or self._recording_started_at is None:
            self._last_issue = SpeechControllerIssue.RECORDING_FAILED
            self._abort_recording()
            return
        timed_out = self._timed_out(now, self._recording_started_at)
        if timed_out is None:
            self._last_issue = SpeechControllerIssue.RECORDING_FAILED
            self._abort_recording()
            return
        if timed_out:
            self._finish_recording()

    def _handle_mute_changed(self) -> None:
        if self._state is SpeechState.RECORDING and not self._microphone_enabled():
            self._abort_recording()

    def _start_recording(self) -> None:
        if self._session_id_supplier is not None:
            self._source_session_id = self._session_id_supplier()
            if self._source_session_id is None:
                return
        if not self._microphone_enabled():
            return

        self._turn_failure = None
        started_at = self._validated_now()
        if started_at is None:
            self._last_issue = SpeechControllerIssue.RECORDING_FAILED
            return
        self._last_issue = None
        if self._discard_if_session_invalid():
            return
        try:
            session = self._recorder.start()
        except Exception:
            if not self._discard_if_session_invalid():
                self._fail(SpeechControllerIssue.RECORDING_FAILED)
            return
        if not self._is_recording_session(session):
            if not self._discard_if_session_invalid():
                self._fail(SpeechControllerIssue.RECORDING_FAILED)
            return
        self._recording_session = session
        self._recording_started_at = started_at
        self._state = SpeechState.RECORDING
        self._discard_if_session_invalid()

    def _finish_recording(self) -> None:
        if self._discard_if_session_invalid():
            return
        session = self._detach_recording_session()
        if session is None:
            return

        if self._timing_observer is None:
            self._process_recording(session, None)
            return

        timings: dict[str, int] = {}
        started_at = self._clock.monotonic()
        try:
            self._process_recording(session, timings)
        finally:
            total_ms = round(float(self._clock.monotonic() - started_at) * 1000)
            timing = TurnTiming(
                played=self._state is SpeechState.PLAYING,
                issue=self._last_issue.value if self._last_issue is not None else None,
                stop_ms=timings["stop_ms"],
                recognize_ms=timings.get("recognize_ms"),
                dialogue_ms=timings.get("dialogue_ms"),
                synthesize_ms=timings.get("synthesize_ms"),
                playback_start_ms=timings.get("playback_start_ms"),
                total_ms=total_ms,
            )
            with suppress(Exception):
                self._timing_observer(timing)

    @contextmanager
    def _time_stage(self, timings: dict[str, int] | None, stage: str) -> Iterator[None]:
        if timings is None:
            yield
            return
        started_at = self._clock.monotonic()
        try:
            yield
        finally:
            timings[stage] = round(float(self._clock.monotonic() - started_at) * 1000)

    def _process_recording(
        self, session: RecordingSession, timings: dict[str, int] | None
    ) -> None:
        recorded_audio: RecordedAudio | None = None
        stop_failed = False
        try:
            with self._time_stage(timings, "stop_ms"):
                candidate_audio = session.stop()
            if self._is_recorded_audio(candidate_audio):
                recorded_audio = candidate_audio
            else:
                stop_failed = True
        except Exception:
            stop_failed = True

        discarded = self._discard_if_session_invalid()
        session_closed = self._close_recording_session(session)
        if discarded or not session_closed or self._discard_if_session_invalid():
            if recorded_audio is not None:
                self._close_recorded_audio(recorded_audio)
            return
        if stop_failed or recorded_audio is None:
            self._fail(SpeechControllerIssue.RECORDING_FAILED)
            return

        self._state = SpeechState.RECOGNIZING
        if self._processing_observer is not None:
            with suppress(Exception):
                self._processing_observer()
        if self._discard_if_session_invalid():
            self._close_recorded_audio(recorded_audio)
            return
        try:
            with self._time_stage(timings, "recognize_ms"):
                recognition_input = self._recognizer.recognize(recorded_audio)
        except Exception:
            recognition_input = None

        discarded = self._discard_if_session_invalid()
        audio_closed = self._close_recorded_audio(recorded_audio)
        if discarded or not audio_closed or self._discard_if_session_invalid():
            return

        if type(recognition_input) is not RecognitionInput:
            self._synthesize_and_play(
                RECOGNITION_RETRY_PROMPT,
                SpeechControllerIssue.RECOGNITION_FAILED,
                timings,
            )
            return

        try:
            recognition_result = gate_recognition(
                audio=recognition_input.audio,
                vad_passed=recognition_input.vad_passed,
                text=recognition_input.text,
                confidence=recognition_input.confidence,
                confidence_status=recognition_input.confidence_status,
                confidence_threshold=recognition_input.confidence_threshold,
                recognizer_failed=recognition_input.recognizer_failed,
            )
        except Exception:
            self._synthesize_and_play(
                RECOGNITION_RETRY_PROMPT,
                SpeechControllerIssue.RECOGNITION_FAILED,
                timings,
            )
            return
        if recognition_result.transcript is None:
            issue = (
                SpeechControllerIssue.RECOGNITION_FAILED
                if recognition_result.rejection_reason
                is RecognitionRejectionReason.RECOGNIZER_FAILED
                else SpeechControllerIssue.RECOGNITION_REJECTED
            )
            self._synthesize_and_play(RECOGNITION_RETRY_PROMPT, issue, timings)
            return

        self._state = SpeechState.THINKING
        if self._discard_if_session_invalid():
            return
        try:
            with self._time_stage(timings, "dialogue_ms"):
                raw_response = self._dialogue.reply(
                    user_id=self._trusted_user_id,
                    text=recognition_result.transcript.text,
                )
        except Exception:
            if self._discard_if_session_invalid():
                return
            self._synthesize_and_play(
                CONNECTION_RETRY_PROMPT,
                SpeechControllerIssue.DIALOGUE_FAILED,
                timings,
            )
            return
        if self._discard_if_session_invalid():
            return
        response = self._validated_dialogue_response(raw_response)
        if response is None:
            self._synthesize_and_play(
                CONNECTION_RETRY_PROMPT,
                SpeechControllerIssue.DIALOGUE_FAILED,
                timings,
            )
            return
        self._turn_failure = {
            "provider_timeout": SpeechTurnFailure.PROVIDER_TIMEOUT,
            "provider_unavailable": SpeechTurnFailure.PROVIDER_UNAVAILABLE,
            "invalid_provider_response": SpeechTurnFailure.INVALID_PROVIDER_RESPONSE,
        }.get(response.blocked_reason)
        self._synthesize_and_play(response.text, None, timings)

    def _validated_dialogue_response(self, value: object) -> DialogueResponse | None:
        try:
            if type(value) is not DialogueResponse:
                return None
            return DialogueResponse.model_validate(
                value.model_dump(mode="python", warnings="error"), strict=True
            )
        except Exception:
            return None

    def _abort_recording(self) -> None:
        session = self._detach_recording_session()
        self._state = SpeechState.IDLE
        if session is not None:
            try:
                abort_failed = session.abort() is not None
            except Exception:
                abort_failed = True
            if abort_failed:
                self._last_issue = SpeechControllerIssue.RECORDING_FAILED
            self._close_recording_session(session)

    def _microphone_enabled(self) -> bool:
        try:
            enabled = self._mute_gate.microphone_enabled()
        except Exception:
            return False
        return type(enabled) is bool and enabled

    def _validated_now(self) -> int | float | Decimal | Fraction | None:
        try:
            value = self._clock.monotonic()
        except Exception:
            return None
        if type(value) is bool:
            return None
        if isinstance(value, int | Fraction):
            return value
        if isinstance(value, Decimal):
            return value if value.is_finite() else None
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return None

    def _timed_out(
        self,
        now: int | float | Decimal | Fraction,
        started_at: int | float | Decimal | Fraction,
    ) -> bool | None:
        try:
            if isinstance(now, Decimal) and isinstance(started_at, Decimal):
                precision = (
                    max(now.adjusted(), started_at.adjusted(), 1)
                    - min(now.as_tuple().exponent, started_at.as_tuple().exponent, 0)
                    + 2
                )
                with localcontext() as context:
                    context.prec = precision
                    return now - started_at >= Decimal(30)
            return now - started_at >= 30
        except (ArithmeticError, TypeError, ValueError):
            return None

    def _synthesize_and_play(
        self,
        text: str,
        issue: SpeechControllerIssue | None,
        timings: dict[str, int] | None,
    ) -> None:
        if self._discard_if_session_invalid():
            return
        text = spoken_text(text)
        if not text:
            text = EMPTY_REPLY_PROMPT
        self._state = SpeechState.SYNTHESIZING
        self._last_issue = issue
        try:
            with self._time_stage(timings, "synthesize_ms"):
                audio = self._synthesizer.synthesize(text)
        except Exception:
            if not self._discard_if_session_invalid():
                self._fail(SpeechControllerIssue.SYNTHESIS_FAILED)
            return
        if not self._is_synthesized_audio(audio):
            if not self._discard_if_session_invalid():
                self._fail(SpeechControllerIssue.SYNTHESIS_FAILED)
            return
        self._synthesized_audio = audio
        self._next_playback_token += 1
        token = self._next_playback_token
        self._playback_token = token
        if self._discard_if_session_invalid():
            return
        try:
            with self._time_stage(timings, "playback_start_ms"):
                if self._session_playback_starter is None:
                    start_failed = self._player.start(audio, token) is not None
                else:
                    start_failed = self._session_playback_starter(
                        audio, token, self._source_session_id
                    ) is not None
        except Exception:
            start_failed = True
        self._state = SpeechState.PLAYING
        if self._discard_if_session_invalid():
            return
        if start_failed:
            detached_audio = self._detach_synthesized_audio()
            self._fail(SpeechControllerIssue.PLAYBACK_FAILED)
            self._stop_player()
            if detached_audio is not None:
                self._close_synthesized_audio(detached_audio)

    def _interrupt_playback(self) -> None:
        if self._discard_if_session_invalid():
            return
        self._playback_token = None
        audio = self._detach_synthesized_audio()
        self._state = SpeechState.IDLE
        stop_failed = not self._stop_player()
        if stop_failed:
            self._last_issue = SpeechControllerIssue.PLAYBACK_FAILED
        if audio is not None and not self._close_synthesized_audio(audio):
            return

        if not stop_failed and not self._discard_if_session_invalid():
            self._start_recording()

    def _discard_if_session_invalid(self) -> bool:
        if self._session_id_supplier is None:
            return False
        if (
            self._source_session_id is not None
            and self._session_id_supplier() == self._source_session_id
        ):
            return False
        cleanup_failed = self._last_issue is SpeechControllerIssue.CLEANUP_FAILED
        if not self._shutdown_current_resources():
            cleanup_failed = True
        self._last_issue = SpeechControllerIssue.CLEANUP_FAILED if cleanup_failed else None
        return True

    def _shutdown_current_resources(self) -> bool:
        session = self._detach_recording_session()
        audio = self._detach_synthesized_audio()
        was_playing = self._state is SpeechState.PLAYING
        self._state = SpeechState.IDLE
        self._source_session_id = None
        cleanup_ok = True

        if session is not None:
            try:
                if session.abort() is not None:
                    cleanup_ok = False
            except Exception:
                cleanup_ok = False
            if not self._close_recording_session(session):
                cleanup_ok = False

        if was_playing and not self._stop_player():
            cleanup_ok = False
        if audio is not None and not self._close_synthesized_audio(audio):
            cleanup_ok = False
        return cleanup_ok

    def _detach_recording_session(self) -> RecordingSession | None:
        session = self._recording_session
        self._recording_session = None
        self._recording_started_at = None
        return session

    def _detach_synthesized_audio(self) -> SynthesizedAudio | None:
        audio = self._synthesized_audio
        self._synthesized_audio = None
        self._playback_token = None
        return audio

    @staticmethod
    def _is_recording_session(value: object) -> bool:
        try:
            return isinstance(value, RecordingSession) and all(
                callable(getattr(value, method_name)) for method_name in ("stop", "abort", "close")
            )
        except Exception:
            return False

    @staticmethod
    def _is_recorded_audio(value: object) -> bool:
        try:
            return isinstance(value, RecordedAudio) and callable(value.close)
        except Exception:
            return False

    @staticmethod
    def _is_synthesized_audio(value: object) -> bool:
        try:
            return isinstance(value, SynthesizedAudio) and callable(value.close)
        except Exception:
            return False

    def _close_recording_session(self, session: RecordingSession) -> bool:
        try:
            close_failed = session.close() is not None
        except Exception:
            close_failed = True
        if close_failed:
            self._fail(SpeechControllerIssue.CLEANUP_FAILED)
            return False
        return True

    def _close_recorded_audio(self, audio: RecordedAudio) -> bool:
        try:
            close_failed = audio.close() is not None
        except Exception:
            close_failed = True
        if close_failed:
            self._fail(SpeechControllerIssue.CLEANUP_FAILED)
            return False
        return True

    def _close_synthesized_audio(self, audio: SynthesizedAudio) -> bool:
        try:
            close_failed = audio.close() is not None
        except Exception:
            close_failed = True
        if close_failed:
            self._fail(SpeechControllerIssue.CLEANUP_FAILED)
            return False
        return True

    def _stop_player(self) -> bool:
        try:
            return self._player.stop() is None
        except Exception:
            return False

    def _fail(self, issue: SpeechControllerIssue) -> None:
        self._state = SpeechState.IDLE
        self._last_issue = issue
