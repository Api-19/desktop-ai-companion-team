from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.runtime.settings import Settings
from companion.runtime.speech_dialogue import SpeechDialogueBridge
from companion.speech import sherpa_vits_synthesizer
from companion.speech.controller import SpeechController
from companion.speech.controller_models import TurnTiming
from companion.speech.controller_ports import (
    AudioPlayer,
    AudioRecorder,
    Clock,
    DialogueCallable,
    MuteGate,
    SpeechSynthesizer,
    SynthesizedAudio,
)
from companion.speech.local_audio_devices import (
    _resolve_audio_runtime,
)
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_model_config import verify_local_speech_config
from companion.speech.runtime_config import (
    _load_validated_speech_runtime_config,
    _local_audio_config_from_runtime,
    _local_model_config_from_runtime,
)
from companion.speech.sensevoice_silero_recognizer import SenseVoiceSileroRecognizer
from companion.speech.sounddevice_player import (
    PlaybackDeviceEvent,
    SoundDeviceAudioPlayer,
)
from companion.speech.sounddevice_recorder import SoundDeviceAudioRecorder


def _raise_configuration_failed() -> NoReturn:
    raise LocalSpeechAdapterError(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED) from None


def _raise_cleanup_failed() -> NoReturn:
    raise LocalSpeechAdapterError(LocalSpeechAdapterErrorCode.CLEANUP_FAILED) from None


def _raise_playback_failed() -> NoReturn:
    raise LocalSpeechAdapterError(LocalSpeechAdapterErrorCode.PLAYBACK_FAILED) from None


_load_runtime_config = _load_validated_speech_runtime_config
_make_model_config = _local_model_config_from_runtime
_make_audio_config = _local_audio_config_from_runtime
_verify_model_tree = verify_local_speech_config
_resolve_device_pair = _resolve_audio_runtime


def _construct_recognizer(config: object) -> object:
    return SenseVoiceSileroRecognizer(config)  # type: ignore[arg-type]


def _construct_synthesizer(config: object) -> SpeechSynthesizer:
    """Default synthesizer factory; swap the engine here, nothing else binds to it."""
    return sherpa_vits_synthesizer.SherpaVitsSynthesizer.from_config(config)  # type: ignore[arg-type]


def _construct_recorder(config: object, resolved: object) -> object:
    return SoundDeviceAudioRecorder.from_resolved(config, resolved)  # type: ignore[arg-type]


def _construct_player(config: object, resolved: object) -> object:
    return SoundDeviceAudioPlayer.from_resolved(config, resolved)  # type: ignore[arg-type]


def _construct_bridge(
    orchestrator: object, *, budget_notice: Callable[[], None] | None = None
) -> object:
    return SpeechDialogueBridge(
        orchestrator=orchestrator,  # type: ignore[arg-type]
        privacy_revision=orchestrator.privacy_revision,  # type: ignore[attr-defined]
        budget_notice=budget_notice,
    )


def _construct_controller(**values: object) -> object:
    return SpeechController(**values)  # type: ignore[arg-type]


def _shutdown_controller(owner: object) -> object:
    if type(owner) is not SpeechController:
        raise TypeError
    return SpeechController.shutdown(owner)


def _close_bridge(owner: object) -> object:
    if type(owner) is not SpeechDialogueBridge:
        raise TypeError
    return SpeechDialogueBridge.close(owner)


def _close_synthesizer(owner: SpeechSynthesizer) -> object:
    return owner.close()


def _close_recognizer(owner: object) -> object:
    if type(owner) is not SenseVoiceSileroRecognizer:
        raise TypeError
    return SenseVoiceSileroRecognizer.close(owner)


def _close_owners(
    controller: object,
    bridge: object,
    synthesizer: object,
    recognizer: object,
) -> bool:
    failed = False
    for owner, cleanup in (
        (controller, _shutdown_controller),
        (bridge, _close_bridge),
        (synthesizer, _close_synthesizer),
        (recognizer, _close_recognizer),
    ):
        if owner is None:
            continue
        try:
            if cleanup(owner) is not None:
                failed = True
        except Exception:
            failed = True
    owner = None
    return failed


@dataclass(slots=True, repr=False, init=False)
class LocalSpeechRuntimeAssembly:
    _controller: object | None
    _bridge: object | None
    _recorder: object | None
    _player: object | None
    _recognizer: object | None
    _synthesizer: object | None
    _closed: bool

    @classmethod
    def _create(
        cls,
        *,
        controller: object,
        bridge: object,
        recorder: object,
        player: object,
        recognizer: object,
        synthesizer: object,
    ) -> LocalSpeechRuntimeAssembly:
        if cls is not LocalSpeechRuntimeAssembly:
            _raise_configuration_failed()
        result = object.__new__(cls)
        result._controller = controller
        result._bridge = bridge
        result._recorder = recorder
        result._player = player
        result._recognizer = recognizer
        result._synthesizer = synthesizer
        result._closed = False
        return result

    @property
    def controller(self) -> SpeechController:
        controller = self._controller
        if self._closed or controller is None:
            controller = None
            self = None  # type: ignore[assignment]
            _raise_configuration_failed()
        return controller  # type: ignore[return-value]

    def drain_playback_events(self) -> None:
        if self._closed:
            self = None  # type: ignore[assignment]
            _raise_playback_failed()
        controller = self._controller
        player = self._player
        event: object = None
        failed = False
        try:
            if controller is None or player is None:
                raise TypeError
            events = player.drain_events()  # type: ignore[attr-defined]
            if type(events) is not tuple:
                raise TypeError
            for event in events:
                if type(event) is not PlaybackDeviceEvent:
                    raise TypeError
                if event.succeeded:
                    controller.playback_finished(event.playback_token)  # type: ignore[attr-defined]
                else:
                    controller.playback_failed(event.playback_token)  # type: ignore[attr-defined]
        except Exception:
            failed = True
        controller = None
        player = None
        event = None
        events = None  # type: ignore[assignment]
        if failed:
            self = None  # type: ignore[assignment]
            _raise_playback_failed()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        controller, self._controller = self._controller, None
        bridge, self._bridge = self._bridge, None
        recorder, self._recorder = self._recorder, None
        player, self._player = self._player, None
        synthesizer, self._synthesizer = self._synthesizer, None
        recognizer, self._recognizer = self._recognizer, None
        failed = _close_owners(controller, bridge, synthesizer, recognizer)
        controller = None
        bridge = None
        del recorder
        del player
        synthesizer = None
        recognizer = None
        if failed:
            self = None  # type: ignore[assignment]
            _raise_cleanup_failed()


def assemble_local_speech_runtime(
    *,
    settings: Settings,
    orchestrator: DialogueOrchestrator,
    trusted_user_id: str,
    mute_gate: MuteGate,
    clock: Clock,
    timing_observer: Callable[[TurnTiming], None] | None = None,
    processing_observer: Callable[[], None] | None = None,
    budget_notice: Callable[[], None] | None = None,
) -> LocalSpeechRuntimeAssembly:
    validated: object = None
    model_config: object = None
    audio_config: object = None
    resolved: object = None
    recognizer: object = None
    synthesizer: object = None
    recorder: object = None
    player: object = None
    bridge: object = None
    controller: object = None
    result: LocalSpeechRuntimeAssembly | None = None
    failed = False
    try:
        if type(settings) is not Settings or type(orchestrator) is not DialogueOrchestrator:
            raise TypeError
        validated = _load_runtime_config(settings)
        model_config = _make_model_config(settings, validated)
        audio_config = _make_audio_config(settings, validated, model_config)
        if audio_config.run_root != model_config.run_root:  # type: ignore[attr-defined]
            raise ValueError
        _verify_model_tree(model_config)
        resolved = _resolve_device_pair(audio_config)
        recognizer = _construct_recognizer(model_config)
        synthesizer = _construct_synthesizer(model_config)
        recorder = _construct_recorder(audio_config, resolved)
        player = _construct_player(audio_config, resolved)
        bridge = _construct_bridge(orchestrator, budget_notice=budget_notice)
        controller = _construct_controller(
            trusted_user_id=trusted_user_id,
            recorder=recorder,
            recognizer=recognizer,
            dialogue=bridge,
            synthesizer=synthesizer,
            player=player,
            mute_gate=mute_gate,
            clock=clock,
            timing_observer=timing_observer,
            processing_observer=processing_observer,
        )
        result = LocalSpeechRuntimeAssembly._create(
            controller=controller,
            bridge=bridge,
            recorder=recorder,
            player=player,
            recognizer=recognizer,
            synthesizer=synthesizer,
        )
    except Exception:
        failed = True

    cleanup_failed = False
    if failed:
        cleanup_failed = _close_owners(controller, bridge, synthesizer, recognizer)
        result = None
    settings = None  # type: ignore[assignment]
    orchestrator = None  # type: ignore[assignment]
    trusted_user_id = None  # type: ignore[assignment]
    mute_gate = None  # type: ignore[assignment]
    clock = None  # type: ignore[assignment]
    validated = None
    model_config = None
    audio_config = None
    resolved = None
    recognizer = None
    synthesizer = None
    recorder = None
    player = None
    bridge = None
    controller = None
    if cleanup_failed:
        _raise_cleanup_failed()
    if failed or result is None:
        result = None
        _raise_configuration_failed()
    return result


def speech_run_root(settings: Settings) -> Path:
    """The validated speech run directory, for recorders built outside this module."""
    return _make_model_config(settings, _load_runtime_config(settings)).run_root  # type: ignore[attr-defined]


def assemble_body_speech_runtime(
    *,
    settings: Settings,
    dialogue: DialogueCallable,
    trusted_user_id: str,
    recorder: AudioRecorder,
    player: AudioPlayer,
    mute_gate: MuteGate,
    clock: Clock,
    timing_observer: Callable[[TurnTiming], None] | None = None,
    session_id_supplier: Callable[[], str | None] | None = None,
    session_playback_starter: Callable[[SynthesizedAudio, int, str], None] | None = None,
    processing_observer: Callable[[], None] | None = None,
) -> LocalSpeechRuntimeAssembly:
    """Like ``assemble_local_speech_runtime`` but audio and mute come from the caller (the
    robot body over bodylink); only the local models are loaded here. ``player`` must also
    provide ``drain_events()``. The assembly owns ``dialogue`` when it is a
    ``SpeechDialogueBridge`` and closes it, also when assembly fails."""
    bridge = dialogue if type(dialogue) is SpeechDialogueBridge else None
    recognizer: object = None
    synthesizer: object = None
    controller: object = None
    try:
        model_config = _make_model_config(settings, _load_runtime_config(settings))
        _verify_model_tree(model_config)
        recognizer = _construct_recognizer(model_config)
        synthesizer = _construct_synthesizer(model_config)
        controller = _construct_controller(
            trusted_user_id=trusted_user_id,
            recorder=recorder,
            recognizer=recognizer,
            dialogue=dialogue,
            synthesizer=synthesizer,
            player=player,
            mute_gate=mute_gate,
            clock=clock,
            timing_observer=timing_observer,
            session_id_supplier=session_id_supplier,
            session_playback_starter=session_playback_starter,
            processing_observer=processing_observer,
        )
    except Exception:
        if _close_owners(controller, bridge, synthesizer, recognizer):
            _raise_cleanup_failed()
        _raise_configuration_failed()
    return LocalSpeechRuntimeAssembly._create(
        controller=controller,
        bridge=bridge,
        recorder=recorder,
        player=player,
        recognizer=recognizer,
        synthesizer=synthesizer,
    )
