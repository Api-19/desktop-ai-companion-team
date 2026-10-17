from __future__ import annotations

import traceback
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

import companion.speech.runtime_assembly as assembly_module
import companion.speech.sherpa_vits_synthesizer as vits_module
from companion.contracts import StyleReplyGuidance, TranscriptConfidenceStatus
from companion.dialogue.config import DialogueConfig
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import (
    BudgetConfig,
    ProviderConfig,
    ProviderResult,
    ProviderUsage,
    StyleRuntimeSources,
)
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime.settings import Settings
from companion.runtime.speech_dialogue import SpeechDialogueBridge
from companion.speech.controller import SpeechController
from companion.speech.controller_models import RecognitionInput, SpeechState
from companion.speech.controller_ports import Clock, MuteGate
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.recognition_gate import AudioQualityEvidence
from companion.speech.runtime_assembly import (
    LocalSpeechRuntimeAssembly,
    assemble_body_speech_runtime,
    assemble_local_speech_runtime,
    speech_run_root,
)
from companion.speech.sherpa_vits_synthesizer import SherpaVitsSynthesizer
from companion.speech.sounddevice_player import PlaybackDeviceEvent
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)
from tests.speech.test_local_model_config import make_config
from tests.speech.test_sherpa_vits_synthesizer import _paths as tts_paths
from tests.speech.test_sherpa_vits_synthesizer import _Runtime as FakeTtsRuntime


class MuteGateFake:
    def microphone_enabled(self) -> bool:
        return True


class ClockFake:
    def monotonic(self) -> float:
        return 0.0


class RuntimeStyleFake:
    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def guidance_for(self, *args: object, **kwargs: object) -> StyleReplyGuidance:
        del args, kwargs
        return StyleReplyGuidance(status="unavailable")


class RuntimeProviderFake:
    def __init__(self) -> None:
        self.requests: list[tuple[PromptMessage, ...]] = []

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        return ProviderResult(
            text="合成公开回答",
            usage=ProviderUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
        )


class RuntimeRecognizerFake(SpeechRecognizerFake):
    def __init__(self, *, result: RecognitionInput) -> None:
        super().__init__(result=result)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class RuntimeSynthesizerFake(SpeechSynthesizerFake):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class RuntimePlayerFake(AudioPlayerFake):
    def __init__(self) -> None:
        super().__init__()
        self.events: tuple[PlaybackDeviceEvent, ...] = ()

    def drain_events(self) -> tuple[PlaybackDeviceEvent, ...]:
        events = self.events
        self.events = ()
        return events


class OwnerFake:
    def __init__(
        self,
        name: str,
        close_calls: list[str],
        *,
        close_failure: Exception | None = None,
    ) -> None:
        self.name = name
        self.close_calls = close_calls
        self.close_failure = close_failure

    def close(self) -> None:
        self.close_calls.append(self.name)
        if self.close_failure is not None:
            raise self.close_failure


class ControllerFake:
    def __init__(
        self,
        close_calls: list[str],
        *,
        close_failure: Exception | None = None,
    ) -> None:
        self.close_calls = close_calls
        self.close_failure = close_failure
        self.finished_tokens: list[int] = []
        self.failed_tokens: list[int] = []
        self.state = SpeechState.IDLE

    def shutdown(self) -> None:
        self.close_calls.append("controller")
        if self.close_failure is not None:
            raise self.close_failure

    def playback_finished(self, playback_token: object) -> None:
        assert type(playback_token) is int
        self.finished_tokens.append(playback_token)

    def playback_failed(self, playback_token: object) -> None:
        assert type(playback_token) is int
        self.failed_tokens.append(playback_token)


class PlayerFake:
    def __init__(self) -> None:
        self.events: tuple[PlaybackDeviceEvent, ...] = ()

    def drain_events(self) -> tuple[PlaybackDeviceEvent, ...]:
        result = self.events
        self.events = ()
        return result


class _ConfigFake:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root


def _settings(tmp_path: Path) -> Settings:
    return Settings.model_construct(
        environment="test",
        log_level="INFO",
        private_data_dir=tmp_path / "private",
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )


def _orchestrator() -> DialogueOrchestrator:
    return object.__new__(DialogueOrchestrator)


def _install_success_seams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    failure_stage: str | None = None,
    close_failure_owner: str | None = None,
) -> tuple[list[str], list[str], ControllerFake, PlayerFake]:
    calls: list[str] = []
    close_calls: list[str] = []
    validated = object()
    model_config = _ConfigFake(tmp_path / "run")
    audio_config = _ConfigFake(tmp_path / "run")
    resolved = object()
    controller = ControllerFake(
        close_calls,
        close_failure=RuntimeError("PRIVATE_CLOSE_SENTINEL")
        if close_failure_owner == "controller"
        else None,
    )
    player = PlayerFake()

    def step(name: str, result: object) -> object:
        calls.append(name)
        if failure_stage == name:
            raise RuntimeError("PRIVATE_STAGE_SENTINEL")
        return result

    monkeypatch.setattr(
        assembly_module,
        "_load_runtime_config",
        lambda settings: step("load-config", validated),
    )
    monkeypatch.setattr(
        assembly_module,
        "_make_model_config",
        lambda settings, value: model_config,
    )
    monkeypatch.setattr(
        assembly_module,
        "_make_audio_config",
        lambda settings, value, model: audio_config,
    )
    monkeypatch.setattr(
        assembly_module,
        "_verify_model_tree",
        lambda config: step("verify-model-tree", object()),
    )
    monkeypatch.setattr(
        assembly_module,
        "_resolve_device_pair",
        lambda config: step("resolve-device-pair", resolved),
    )

    def owner(name: str) -> OwnerFake:
        return OwnerFake(
            name,
            close_calls,
            close_failure=RuntimeError("PRIVATE_CLOSE_SENTINEL")
            if close_failure_owner == name
            else None,
        )

    monkeypatch.setattr(
        assembly_module,
        "_construct_recognizer",
        lambda config: step("construct-recognizer", owner("recognizer")),
    )
    monkeypatch.setattr(
        assembly_module,
        "_construct_synthesizer",
        lambda config: step("construct-synthesizer", owner("synthesizer")),
    )
    monkeypatch.setattr(
        assembly_module,
        "_construct_recorder",
        lambda config, value: step("construct-recorder", object()),
    )
    monkeypatch.setattr(
        assembly_module,
        "_construct_player",
        lambda config, value: step("construct-player", player),
    )
    monkeypatch.setattr(
        assembly_module,
        "_construct_bridge",
        lambda value, *, budget_notice=None: step("construct-bridge", owner("bridge")),
    )
    monkeypatch.setattr(
        assembly_module,
        "_construct_controller",
        lambda **kwargs: step("construct-controller", controller),
    )
    monkeypatch.setattr(
        assembly_module,
        "_shutdown_controller",
        lambda value: value.shutdown(),
    )
    monkeypatch.setattr(
        assembly_module,
        "_close_bridge",
        lambda value: value.close(),
    )
    monkeypatch.setattr(
        assembly_module,
        "_close_synthesizer",
        lambda value: value.close(),
    )
    monkeypatch.setattr(
        assembly_module,
        "_close_recognizer",
        lambda value: value.close(),
    )
    return calls, close_calls, controller, player


def _assemble(tmp_path: Path) -> LocalSpeechRuntimeAssembly:
    return assemble_local_speech_runtime(
        settings=_settings(tmp_path),
        orchestrator=_orchestrator(),
        trusted_user_id="synthetic-trusted-owner",
        mute_gate=MuteGateFake(),
        clock=ClockFake(),
    )


@pytest.mark.parametrize("body", [False, True])
@pytest.mark.parametrize("with_observer", [False, True])
def test_assemblies_forward_timing_observer_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bool, with_observer: bool
) -> None:
    _, _, controller, _ = _install_success_seams(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    def construct(**kwargs: object) -> object:
        captured.update(kwargs)
        return controller

    monkeypatch.setattr(assembly_module, "_construct_controller", construct)
    observer = (lambda timing: None) if with_observer else None
    common = dict(settings=_settings(tmp_path), trusted_user_id="synthetic-owner",
                  mute_gate=MuteGateFake(), clock=ClockFake(), timing_observer=observer)
    if body:
        assembly = assemble_body_speech_runtime(
            **common, dialogue=FixedDialogueFake(), recorder=AudioRecorderFake(),
            player=PlayerFake(),
        )
    else:
        assembly = assemble_local_speech_runtime(**common, orchestrator=_orchestrator())
    assert captured["timing_observer"] is observer
    assembly.close()


@pytest.mark.parametrize("body", [False, True])
@pytest.mark.parametrize("with_observer", [False, True])
def test_assemblies_forward_processing_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bool, with_observer: bool
) -> None:
    _, _, controller, _ = _install_success_seams(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    def construct(**kwargs: object) -> object:
        captured.update(kwargs)
        return controller

    monkeypatch.setattr(assembly_module, "_construct_controller", construct)
    observer = (lambda: None) if with_observer else None
    common = dict(
        settings=_settings(tmp_path), trusted_user_id="synthetic-owner",
        mute_gate=MuteGateFake(), clock=ClockFake(), processing_observer=observer,
    )
    if body:
        assembly = assemble_body_speech_runtime(
            **common, dialogue=FixedDialogueFake(), recorder=AudioRecorderFake(),
            player=PlayerFake(),
        )
    else:
        assembly = assemble_local_speech_runtime(**common, orchestrator=_orchestrator())

    assert captured["processing_observer"] is observer
    assembly.close()


def _real_fake_orchestrator(root: Path, provider: RuntimeProviderFake) -> DialogueOrchestrator:
    budget = BudgetConfig(
        monthly_cap_cny=Decimal("100.00"),
        worst_case_request_cny=Decimal("1.00"),
        input_cny_per_million=Decimal("1.00"),
        output_cny_per_million=Decimal("2.00"),
        max_output_tokens=256,
    )
    provider_config = ProviderConfig(
        base_url="https://example.invalid/v1",
        model="synthetic-provider",
        timeout_seconds=10,
    )
    style = StyleRuntimeSources(
        candidate_source=root / "synthetic-candidates.json",
        expected_count=1,
        active_style_source=root / "synthetic-style.json",
        active_manifest_source=root / "synthetic-manifest.json",
        expected_active_manifest_digest="a" * 64,
        calibration_source=root / "synthetic-calibration.json",
        expected_calibration_fingerprint="b" * 64,
        private_data_dir=root,
        true_name_terms=("synthetic-owner",),
        approved_nicknames=("synthetic-nickname",),
    )
    return DialogueOrchestrator(
        config=DialogueConfig(provider=provider_config, budget=budget, style=style),
        style_adapter=RuntimeStyleFake(),
        provider=provider,
        ledger=MonthlyUsageLedger(root / "ledger", budget),
        clock=lambda: datetime(2026, 8, 25, 8, 0, tzinfo=UTC),
    )


def _accepted_recognition() -> RecognitionInput:
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
        text="合成公开问题",
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
        confidence_threshold=None,
    )


def test_local_assembly_binds_public_privacy_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_bridge = assembly_module._construct_bridge
    _install_success_seams(monkeypatch, tmp_path)
    monkeypatch.setattr(assembly_module, "_construct_bridge", construct_bridge)
    orchestrator = _real_fake_orchestrator(tmp_path, RuntimeProviderFake())

    assembly = assemble_local_speech_runtime(
        settings=_settings(tmp_path),
        orchestrator=orchestrator,
        trusted_user_id="synthetic-owner",
        mute_gate=MuteGateFake(),
        clock=ClockFake(),
    )

    bridge = assembly._bridge
    assert type(bridge) is SpeechDialogueBridge
    assert bridge._privacy_revision == orchestrator.privacy_revision
    assert bridge._privacy_revision.__self__ is orchestrator
    assert bridge._privacy_revision("synthetic-owner") == ()
    assembly.close()
    assert bridge._privacy_revision is None


def test_local_assembly_forwards_budget_notice_to_real_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_bridge = assembly_module._construct_bridge
    _install_success_seams(monkeypatch, tmp_path)
    monkeypatch.setattr(assembly_module, "_construct_bridge", construct_bridge)
    orchestrator = _real_fake_orchestrator(tmp_path, RuntimeProviderFake())
    notices: list[None] = []
    def callback() -> None:
        notices.append(None)

    assembly = assemble_local_speech_runtime(
        settings=_settings(tmp_path), orchestrator=orchestrator,
        trusted_user_id="synthetic-owner", mute_gate=MuteGateFake(), clock=ClockFake(),
        budget_notice=callback,
    )

    bridge = assembly._bridge
    assert type(bridge) is SpeechDialogueBridge
    assert bridge._budget_notice is callback
    assert bridge._privacy_revision == orchestrator.privacy_revision
    assert bridge._privacy_revision.__self__ is orchestrator
    assert notices == []
    assembly.close()
    assert bridge._budget_notice is None
    assert bridge._privacy_revision is None
    assert notices == []


def _assemble_real_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    recognition: RecognitionInput | None = None,
) -> tuple[
    LocalSpeechRuntimeAssembly,
    RuntimeProviderFake,
    AudioRecorderFake,
    RuntimeRecognizerFake,
    RuntimeSynthesizerFake,
    RuntimePlayerFake,
]:
    provider = RuntimeProviderFake()
    orchestrator = _real_fake_orchestrator(tmp_path, provider)
    recorder = AudioRecorderFake()
    recognizer = RuntimeRecognizerFake(result=recognition or _accepted_recognition())
    synthesizer = RuntimeSynthesizerFake()
    player = RuntimePlayerFake()
    run_root = tmp_path / "run"
    model_config = _ConfigFake(run_root)
    audio_config = _ConfigFake(run_root)

    monkeypatch.setattr(assembly_module, "_load_runtime_config", lambda settings: object())
    monkeypatch.setattr(
        assembly_module,
        "_make_model_config",
        lambda settings, validated: model_config,
    )
    monkeypatch.setattr(
        assembly_module,
        "_make_audio_config",
        lambda settings, validated, model: audio_config,
    )
    monkeypatch.setattr(assembly_module, "_verify_model_tree", lambda config: object())
    monkeypatch.setattr(assembly_module, "_resolve_device_pair", lambda config: object())
    monkeypatch.setattr(assembly_module, "_construct_recognizer", lambda config: recognizer)
    monkeypatch.setattr(assembly_module, "_construct_synthesizer", lambda config: synthesizer)
    monkeypatch.setattr(
        assembly_module,
        "_construct_recorder",
        lambda config, resolved: recorder,
    )
    monkeypatch.setattr(
        assembly_module,
        "_construct_player",
        lambda config, resolved: player,
    )
    monkeypatch.setattr(
        assembly_module,
        "_close_recognizer",
        lambda value: value.close(),
    )
    assembly = assemble_local_speech_runtime(
        settings=_settings(tmp_path),
        orchestrator=orchestrator,
        trusted_user_id="synthetic-owner",
        mute_gate=MuteGateFake(),
        clock=ClockFake(),
    )
    return assembly, provider, recorder, recognizer, synthesizer, player


def test_real_controller_bridge_orchestrator_composition_is_software_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly, provider, recorder, recognizer, synthesizer, player = _assemble_real_runtime(
        tmp_path,
        monkeypatch,
    )
    controller = assembly.controller
    assert type(controller) is SpeechController

    controller.button_pressed()
    controller.button_pressed()
    assert controller.state is SpeechState.PLAYING
    player.events = (PlaybackDeviceEvent(playback_token=player.tokens[0], succeeded=True),)
    assembly.drain_playback_events()

    assert controller.state is SpeechState.IDLE
    assert len(provider.requests) == 1
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0

    assembly.close()

    assert controller._trusted_user_id is None
    assert controller._dialogue is None
    assert recognizer.close_calls == 1
    assert synthesizer.close_calls == 1


def test_real_composition_recognition_rejection_never_calls_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected = replace(_accepted_recognition(), vad_passed=False)
    assembly, provider, recorder, _, synthesizer, player = _assemble_real_runtime(
        tmp_path,
        monkeypatch,
        recognition=rejected,
    )
    controller = assembly.controller

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.PLAYING
    assert provider.requests == []
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert len(synthesizer.inputs) == 1
    assert player.tokens == [1]
    assembly.close()


@pytest.mark.parametrize(
    "failure_stage",
    ["verify-model-tree", "resolve-device-pair"],
    ids=["invalid-model-contract", "invalid-exact-selector-resolution"],
)
def test_model_or_selector_failure_stops_before_runtime_components(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, close_calls, _, _ = _install_success_seams(
        monkeypatch,
        tmp_path,
        failure_stage=failure_stage,
    )

    with pytest.raises(LocalSpeechAdapterError):
        _assemble(tmp_path)

    assert calls[-1] == failure_stage
    assert all(not name.startswith("construct-") for name in calls)
    assert close_calls == []


def test_real_bridge_structural_failure_uses_no_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly, provider, recorder, _, synthesizer, player = _assemble_real_runtime(
        tmp_path,
        monkeypatch,
    )

    def malformed_reply(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    monkeypatch.setattr(DialogueOrchestrator, "reply", malformed_reply)
    controller = assembly.controller
    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.PLAYING
    assert provider.requests == []
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert len(synthesizer.inputs) == 1
    assert player.tokens == [1]
    assembly.close()


@pytest.mark.parametrize("state", ["recording", "playing"])
def test_assembly_close_during_active_audio_is_terminal(
    state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly, provider, recorder, recognizer, synthesizer, player = _assemble_real_runtime(
        tmp_path,
        monkeypatch,
    )
    controller = assembly.controller
    controller.button_pressed()
    if state == "playing":
        controller.button_pressed()

    assembly.close()

    assert controller.state is SpeechState.IDLE
    assert controller._trusted_user_id is None
    assert recognizer.close_calls == 1
    assert synthesizer.close_calls == 1
    if state == "recording":
        assert provider.requests == []
        assert recorder.calls == ["start", "abort", "session.close"]
        assert player.stop_calls == 0
    else:
        assert len(provider.requests) == 1
        assert player.stop_calls == 1
        assert synthesizer.audio_close_calls == 1


def test_assembly_uses_fixed_order_and_distinguishes_playback_event_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _, controller, player = _install_success_seams(monkeypatch, tmp_path)
    assembly = _assemble(tmp_path)
    player.events = (
        PlaybackDeviceEvent(playback_token=7, succeeded=True),
        PlaybackDeviceEvent(playback_token=8, succeeded=False),
    )

    assembly.drain_playback_events()

    assert calls == [
        "load-config",
        "verify-model-tree",
        "resolve-device-pair",
        "construct-recognizer",
        "construct-synthesizer",
        "construct-recorder",
        "construct-player",
        "construct-bridge",
        "construct-controller",
    ]
    assert assembly.controller is controller
    assert controller.finished_tokens == [7]
    assert controller.failed_tokens == [8]
    assert "selector" not in repr(assembly)


def test_invalid_playback_event_is_absent_from_failure_traceback_locals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "PRIVATE_PLAYBACK_EVENT_SENTINEL"

    class HostileEvent:
        def __repr__(self) -> str:
            print(marker)
            raise RuntimeError(marker)

    _, _, _, player = _install_success_seams(monkeypatch, tmp_path)
    assembly = _assemble(tmp_path)
    hostile_event = HostileEvent()
    player.events = (hostile_event,)  # type: ignore[assignment]

    with pytest.raises(LocalSpeechAdapterError) as caught:
        assembly.drain_playback_events()

    reachable: list[object] = []
    for frame, _line_number in traceback.walk_tb(caught.value.__traceback__):
        if frame.f_code.co_filename.replace("\\", "/").endswith(
            "/src/companion/speech/runtime_assembly.py"
        ):
            reachable.extend(frame.f_locals.values())
    captured = capsys.readouterr()
    assert all(value is not hostile_event for value in reachable)
    assert caught.value.code is LocalSpeechAdapterErrorCode.PLAYBACK_FAILED
    assert marker not in captured.out
    assert marker not in captured.err


@pytest.mark.parametrize(
    "failure_stage",
    [
        "load-config",
        "verify-model-tree",
        "resolve-device-pair",
        "construct-recognizer",
        "construct-synthesizer",
        "construct-recorder",
        "construct-player",
        "construct-bridge",
        "construct-controller",
    ],
)
def test_each_stage_failure_stops_and_reverse_closes_created_owners(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, close_calls, _, _ = _install_success_seams(
        monkeypatch, tmp_path, failure_stage=failure_stage
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        _assemble(tmp_path)

    error = caught.value
    rendered = "\n".join(traceback.format_exception(error))
    assert calls[-1] == failure_stage
    assert close_calls == [
        name
        for name in ("bridge", "synthesizer", "recognizer")
        if f"construct-{name}" in calls and f"construct-{name}" != failure_stage
    ]
    assert type(error) is LocalSpeechAdapterError
    assert error.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert error.__cause__ is None
    assert error.__context__ is None
    forbidden = ("PRIVATE_STAGE_SENTINEL", "synthetic-trusted-owner")
    captured = capsys.readouterr()
    assert all(value not in rendered for value in forbidden)
    assert all(value not in caplog.text for value in forbidden)
    assert all(value not in captured.out for value in forbidden)
    assert all(value not in captured.err for value in forbidden)


@pytest.mark.parametrize("failure_owner", ["controller", "bridge", "synthesizer", "recognizer"])
def test_close_continues_in_fixed_order_and_clears_every_field(
    failure_owner: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, close_calls, _, _ = _install_success_seams(
        monkeypatch, tmp_path, close_failure_owner=failure_owner
    )
    assembly = _assemble(tmp_path)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        assembly.close()

    captured = capsys.readouterr()
    assert close_calls == ["controller", "bridge", "synthesizer", "recognizer"]
    assert caught.value.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert all(
        getattr(assembly, name) is None
        for name in (
            "_controller",
            "_bridge",
            "_recorder",
            "_player",
            "_recognizer",
            "_synthesizer",
        )
    )
    assert "synthetic-trusted-owner" not in repr(assembly)
    assert "PRIVATE_CLOSE_SENTINEL" not in caplog.text
    assert "PRIVATE_CLOSE_SENTINEL" not in captured.out
    assert "PRIVATE_CLOSE_SENTINEL" not in captured.err
    assembly.close()
    assert close_calls == ["controller", "bridge", "synthesizer", "recognizer"]
    with pytest.raises(LocalSpeechAdapterError):
        _ = assembly.controller


def test_successful_close_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, close_calls, _, _ = _install_success_seams(monkeypatch, tmp_path)
    assembly = _assemble(tmp_path)

    assembly.close()
    assembly.close()

    assert close_calls == ["controller", "bridge", "synthesizer", "recognizer"]


def test_close_never_invokes_hostile_cleanup_descriptor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "PRIVATE_CLEANUP_DESCRIPTOR_SENTINEL"

    class HostileController:
        @property
        def shutdown(self) -> object:
            print(marker)
            raise RuntimeError(marker)

    assembly = LocalSpeechRuntimeAssembly._create(
        controller=HostileController(),
        bridge=None,
        recorder=None,
        player=None,
        recognizer=None,
        synthesizer=None,
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        assembly.close()

    captured = capsys.readouterr()
    assert caught.value.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert marker not in captured.out
    assert marker not in captured.err


def test_public_entry_rejects_non_exact_settings_and_orchestrator_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _, _, _ = _install_success_seams(monkeypatch, tmp_path)

    with pytest.raises(LocalSpeechAdapterError):
        assemble_local_speech_runtime(
            settings=object(),  # type: ignore[arg-type]
            orchestrator=_orchestrator(),
            trusted_user_id="synthetic-owner",
            mute_gate=MuteGateFake(),
            clock=ClockFake(),
        )
    with pytest.raises(LocalSpeechAdapterError):
        assemble_local_speech_runtime(
            settings=_settings(tmp_path),
            orchestrator=object(),  # type: ignore[arg-type]
            trusted_user_id="synthetic-owner",
            mute_gate=MuteGateFake(),
            clock=ClockFake(),
        )

    assert calls == []


def test_settings_drift_stops_before_model_verification_or_device_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    original_root = settings.private_data_dir
    calls: list[str] = []

    def load(received: Settings) -> object:
        assert received is settings
        calls.append("load-config")
        received.private_data_dir = tmp_path / "changed-private"
        return object()

    def reject_stale(received: Settings, validated: object) -> object:
        del validated
        calls.append("model-config")
        assert received is settings
        assert received.private_data_dir != original_root
        raise LocalSpeechAdapterError(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)

    monkeypatch.setattr(assembly_module, "_load_runtime_config", load)
    monkeypatch.setattr(assembly_module, "_make_model_config", reject_stale)
    monkeypatch.setattr(
        assembly_module,
        "_verify_model_tree",
        lambda config: calls.append("verify-model-tree"),
    )
    monkeypatch.setattr(
        assembly_module,
        "_resolve_device_pair",
        lambda config: calls.append("resolve-device-pair"),
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        assemble_local_speech_runtime(
            settings=settings,
            orchestrator=_orchestrator(),
            trusted_user_id="synthetic-owner",
            mute_gate=MuteGateFake(),
            clock=ClockFake(),
        )

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert calls == ["load-config", "model-config"]


def test_model_audio_run_root_mismatch_stops_before_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _, _, _ = _install_success_seams(monkeypatch, tmp_path)
    monkeypatch.setattr(
        assembly_module,
        "_make_audio_config",
        lambda settings, value, model: _ConfigFake(tmp_path / "other-run"),
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        _assemble(tmp_path)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert calls == ["load-config"]


def test_default_synthesizer_factory_builds_sherpa_vits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FakeTtsRuntime()
    monkeypatch.setattr(vits_module, "load_sherpa_runtime", lambda _config: runtime)
    monkeypatch.setattr(vits_module, "verify_local_speech_config", tts_paths)

    synthesizer = assembly_module._construct_synthesizer(make_config(tmp_path))

    assert type(synthesizer) is SherpaVitsSynthesizer
    assert assembly_module._close_synthesizer(synthesizer) is None
    with pytest.raises(LocalSpeechAdapterError):
        synthesizer.synthesize("合成公开文本")


def test_any_synthesizer_is_closed_through_its_own_close() -> None:
    synthesizer = RuntimeSynthesizerFake()
    assembly = LocalSpeechRuntimeAssembly._create(
        controller=None,
        bridge=None,
        recorder=None,
        player=None,
        recognizer=None,
        synthesizer=synthesizer,
    )

    assembly.close()

    assert synthesizer.close_calls == 1


def test_protocol_fakes_match_assembly_inputs() -> None:
    assert isinstance(MuteGateFake(), MuteGate)
    assert isinstance(ClockFake(), Clock)


# ---- body (bodylink) assembly: injected recorder/player/mute gate ----


class FixedDialogueFake:
    def reply(self, *, user_id: str, text: str) -> object:
        raise AssertionError("not called during assembly")


def _install_body_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, failure_stage: str | None = None
) -> tuple[list[str], list[str], dict[str, object]]:
    calls, close_calls, controller, _ = _install_success_seams(
        monkeypatch, tmp_path, failure_stage=failure_stage
    )
    controller_kwargs: dict[str, object] = {}

    def construct_controller(**kwargs: object) -> object:
        calls.append("construct-controller")
        if failure_stage == "construct-controller":
            raise RuntimeError("PRIVATE_STAGE_SENTINEL")
        controller_kwargs.update(kwargs)
        return controller

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("local audio device work in body assembly")

    monkeypatch.setattr(assembly_module, "_construct_controller", construct_controller)
    for name in ("_make_audio_config", "_resolve_device_pair", "_construct_recorder",
                 "_construct_player", "_construct_bridge"):  # fmt: skip
        monkeypatch.setattr(assembly_module, name, forbidden)
    return calls, close_calls, controller_kwargs


def _assemble_body(tmp_path: Path, dialogue: object, player: object) -> LocalSpeechRuntimeAssembly:
    return assemble_body_speech_runtime(
        settings=_settings(tmp_path),
        dialogue=dialogue,  # type: ignore[arg-type]
        trusted_user_id="synthetic-trusted-owner",
        recorder=AudioRecorderFake(),
        player=player,  # type: ignore[arg-type]
        mute_gate=MuteGateFake(),
        clock=ClockFake(),
    )


def test_speech_run_root_is_the_validated_model_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_body_seams(monkeypatch, tmp_path)

    assert speech_run_root(_settings(tmp_path)) == tmp_path / "run"


def test_body_assembly_wires_injected_ports_and_loads_models_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, close_calls, kwargs = _install_body_seams(monkeypatch, tmp_path)
    dialogue = FixedDialogueFake()
    player = PlayerFake()

    assembly = _assemble_body(tmp_path, dialogue, player)

    assert calls == [
        "load-config",
        "verify-model-tree",
        "construct-recognizer",
        "construct-synthesizer",
        "construct-controller",
    ]
    assert kwargs["dialogue"] is dialogue
    assert kwargs["player"] is player
    assert type(kwargs["recorder"]) is AudioRecorderFake
    assert type(kwargs["mute_gate"]) is MuteGateFake
    assert kwargs["trusted_user_id"] == "synthetic-trusted-owner"
    assert kwargs["session_id_supplier"] is None
    assert kwargs["session_playback_starter"] is None
    player.events = (PlaybackDeviceEvent(3, True), PlaybackDeviceEvent(4, False))
    assembly.drain_playback_events()
    assert assembly.controller.finished_tokens == [3]  # type: ignore[attr-defined]
    assert assembly.controller.failed_tokens == [4]  # type: ignore[attr-defined]
    assembly.close()
    assert close_calls == ["controller", "synthesizer", "recognizer"]


def test_body_assembly_forwards_session_callbacks_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, kwargs = _install_body_seams(monkeypatch, tmp_path)

    def supplier() -> str:
        return "s"

    def starter(audio, token, session_id) -> None:
        raise AssertionError("not called during assembly")

    assembly = assemble_body_speech_runtime(
        settings=_settings(tmp_path),
        dialogue=FixedDialogueFake(),
        trusted_user_id="synthetic-trusted-owner",
        recorder=AudioRecorderFake(),
        player=PlayerFake(),
        mute_gate=MuteGateFake(),
        clock=ClockFake(),
        session_id_supplier=supplier,
        session_playback_starter=starter,
    )

    assert kwargs["session_id_supplier"] is supplier
    assert kwargs["session_playback_starter"] is starter
    assembly.close()


def test_body_assembly_owns_and_closes_a_dialogue_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, close_calls, _ = _install_body_seams(monkeypatch, tmp_path)
    bridge = SpeechDialogueBridge(orchestrator=_orchestrator())

    _assemble_body(tmp_path, bridge, PlayerFake()).close()

    assert close_calls == ["controller", "synthesizer", "recognizer"]
    assert bridge._closed is True


@pytest.mark.parametrize(
    "failure_stage",
    ["load-config", "verify-model-tree", "construct-recognizer", "construct-synthesizer",
     "construct-controller"],
)  # fmt: skip
def test_body_assembly_stage_failure_closes_created_owners(
    failure_stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, close_calls, _ = _install_body_seams(monkeypatch, tmp_path, failure_stage=failure_stage)
    bridge = SpeechDialogueBridge(orchestrator=_orchestrator())

    with pytest.raises(LocalSpeechAdapterError) as caught:
        _assemble_body(tmp_path, bridge, PlayerFake())

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert caught.value.__cause__ is None
    assert calls[-1] == failure_stage
    assert close_calls == [
        name
        for name in ("synthesizer", "recognizer")
        if f"construct-{name}" in calls and f"construct-{name}" != failure_stage
    ]
    assert bridge._closed is True
