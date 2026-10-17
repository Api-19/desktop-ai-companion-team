from typing import Protocol, runtime_checkable

from companion.contracts import DialogueResponse
from companion.speech.controller_models import RecognitionInput
from companion.speech.pcm_audio import PcmAudio


@runtime_checkable
class RecordedAudio(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class RecordingSession(Protocol):
    def stop(self) -> RecordedAudio: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class AudioRecorder(Protocol):
    def start(self) -> RecordingSession: ...


@runtime_checkable
class SpeechRecognizer(Protocol):
    def recognize(self, audio: RecordedAudio) -> RecognitionInput: ...


@runtime_checkable
class DialogueCallable(Protocol):
    def reply(self, *, user_id: str, text: str) -> DialogueResponse: ...


@runtime_checkable
class SynthesizedAudio(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str) -> PcmAudio: ...

    def close(self) -> None: ...


@runtime_checkable
class AudioPlayer(Protocol):
    def start(self, audio: PcmAudio, playback_token: int) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class MuteGate(Protocol):
    def microphone_enabled(self) -> bool: ...


@runtime_checkable
class Clock(Protocol):
    def monotonic(self) -> float: ...
