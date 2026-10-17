"""Synthetic Speech-to-Dialogue conversation through real local services."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from companion.contracts import ConsentRecord, StyleReplyGuidance, TranscriptConfidenceStatus
from companion.dialogue.config import DialogueConfig
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import (
    BudgetConfig,
    ProviderConfig,
    ProviderResult,
    StyleRuntimeSources,
)
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.memory import MemoryField, MemoryService
from companion.runtime.speech_dialogue import SpeechDialogueBridge
from companion.speech.controller import SpeechController
from companion.speech.controller_models import RecognitionInput, SpeechState
from companion.speech.recognition_gate import AudioQualityEvidence
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
USER = "synthetic-full-conversation-user"


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[PromptMessage, ...]] = []

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        return ProviderResult(text="这是合成安全回答")


class UnavailableStyle:
    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def guidance_for(self, *args: object, **kwargs: object) -> StyleReplyGuidance:
        return StyleReplyGuidance(status="unavailable")


def _recognition(text: str) -> RecognitionInput:
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
        text=text,
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
        confidence_threshold=None,
    )


def _config(root: Path) -> DialogueConfig:
    return DialogueConfig(
        provider=ProviderConfig(
            base_url="https://example.invalid/v1",
            model="synthetic-conversation-provider",
            timeout_seconds=10,
        ),
        budget=BudgetConfig(
            monthly_cap_cny=Decimal("100.00"),
            worst_case_request_cny=Decimal("1.00"),
            input_cny_per_million=Decimal("1.00"),
            output_cny_per_million=Decimal("2.00"),
            max_output_tokens=256,
        ),
        style=StyleRuntimeSources(
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
        ),
    )


def _system(tmp_path: Path, transcript: str):
    memory = MemoryService(tmp_path / "synthetic-memory.sqlite3", clock=lambda: NOW)
    provider = RecordingProvider()
    config = _config(tmp_path)
    dialogue = DialogueOrchestrator(
        config=config,
        style_adapter=UnavailableStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "usage", config.budget),
        clock=lambda: NOW,
        memory_gateway=memory,
    )
    bridge = SpeechDialogueBridge(orchestrator=dialogue)
    recorder = AudioRecorderFake()
    recognizer = SpeechRecognizerFake(result=_recognition(transcript))
    synthesizer = SpeechSynthesizerFake()
    player = AudioPlayerFake()
    controller = SpeechController(
        trusted_user_id=USER,
        recorder=recorder,
        recognizer=recognizer,
        dialogue=bridge,
        synthesizer=synthesizer,
        player=player,
        mute_gate=MuteGateFake(),
        clock=ClockFake(),
    )
    return memory, provider, controller, recorder, recognizer, synthesizer, player


def _complete_turn(controller: SpeechController, player: AudioPlayerFake) -> None:
    controller.button_pressed()
    controller.button_pressed()
    assert controller.state is SpeechState.PLAYING
    controller.playback_finished(player.tokens[-1])
    assert controller.state is SpeechState.IDLE


def test_authorized_memory_to_provider_answer_and_synthetic_playback(tmp_path: Path) -> None:
    memory, provider, controller, recorder, recognizer, synthesizer, player = _system(
        tmp_path, "记住我喜欢合成蓝色"
    )
    memory.grant_consent(
        ConsentRecord(
            user_id=USER,
            granted=True,
            granted_at=NOW,
            allowed_fields=[field.value for field in MemoryField],
        )
    )

    _complete_turn(controller, player)
    assert provider.requests == []
    saved = memory.list_memories(USER)
    assert len(saved) == 1
    assert saved[0].value == "喜欢合成蓝色"
    assert len(synthesizer.inputs) == len(player.tokens) == 1

    recognizer.result = _recognition("合成蓝色")
    _complete_turn(controller, player)

    assert recognizer.calls == 2
    assert len(provider.requests) == 1
    assert provider.requests[0][1].content == "合成蓝色"
    assert "memory[1]" in provider.requests[0][0].content
    assert "value=喜欢合成蓝色" in provider.requests[0][0].content
    assert "这是合成安全回答" in synthesizer.inputs[1]
    assert len(synthesizer.inputs) == len(player.tokens) == 2
    assert recorder.live_session_count == recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0


def test_unconsented_memory_command_stops_before_provider_and_write(tmp_path: Path) -> None:
    memory, provider, controller, recorder, recognizer, synthesizer, player = _system(
        tmp_path, "记住我喜欢合成红色"
    )

    _complete_turn(controller, player)

    assert recognizer.calls == 1
    assert memory.list_memories(USER) == ()
    assert provider.requests == []
    assert len(synthesizer.inputs) == len(player.tokens) == 1
    assert "合成红色" not in synthesizer.inputs[0]
    assert recorder.live_session_count == recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0
