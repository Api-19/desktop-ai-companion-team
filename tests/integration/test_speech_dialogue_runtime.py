from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from companion.contracts import (
    DialogueRequest,
    DialogueResponse,
    MemoryEntry,
    StyleReplyGuidance,
    TranscriptConfidenceStatus,
)
from companion.dialogue.config import DialogueConfig
from companion.dialogue.context import PromptMessage
from companion.dialogue.identity import AI_DISCLOSURE
from companion.dialogue.models import (
    BudgetConfig,
    DialogueMode,
    DialogueRun,
    ProviderConfig,
    ProviderResult,
    ProviderUsage,
    StyleRuntimeSources,
)
from companion.dialogue.observer import TurnObservation, TurnObserver
from companion.dialogue.orchestrator import (
    BUDGET_FALLBACK,
    INVALID_PROVIDER_RESPONSE_FALLBACK,
    PROVIDER_TIMEOUT_FALLBACK,
    PROVIDER_UNAVAILABLE_FALLBACK,
    UNSAFE_OUTPUT_FALLBACK,
    DialogueOrchestrator,
)
from companion.dialogue.provider import (
    OpenAIProviderUnavailable,
    ProviderTimeout,
    ProviderUnavailableKind,
    ProviderUnavailableStage,
)
from companion.dialogue.usage import MonthlyUsageLedger
from companion.memory import CreateMemoryRequest
from companion.runtime.speech_dialogue import SpeechDialogueBridge
from companion.speech.controller import SpeechController
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
from companion.speech.recognition_gate import AudioQualityEvidence
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)

NOW = datetime(2026, 8, 24, 8, 0, tzinfo=UTC)
USER_A = "synthetic-user-a"
USER_B = "synthetic-user-b"


class SyntheticStyle:
    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def guidance_for(self, *args: object, **kwargs: object) -> StyleReplyGuidance:
        del args, kwargs
        return StyleReplyGuidance(status="unavailable")


class CaptureProvider:
    def __init__(self, text: str = "合成安全回答") -> None:
        self.text = text
        self.requests: list[tuple[PromptMessage, ...]] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        return ProviderResult(
            text=self.text,
            usage=ProviderUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
        )


class TimeoutProvider(CaptureProvider):
    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        raise ProviderTimeout("synthetic-provider-timeout")


class MalformedProvider(CaptureProvider):
    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        return ProviderResult.model_construct(text="", usage=None)


class SequenceProvider(CaptureProvider):
    def __init__(self, *outcomes: Exception | str) -> None:
        super().__init__()
        self.outcomes = outcomes

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        result = super().complete(messages)
        outcome = self.outcomes[self.call_count - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return result.model_copy(update={"text": outcome})


class RecordingRuntime:
    def __init__(self, orchestrator: DialogueOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.requests: list[DialogueRequest] = []
        self.runs: list[DialogueRun] = []

    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        self.requests.append(request)
        run = self.orchestrator.reply(request, mode, trusted_user_id=trusted_user_id)
        self.runs.append(run)
        return run


class EmptyMemoryGateway:
    def __init__(self) -> None:
        self.created: list[CreateMemoryRequest] = []
        self.retrievals: list[tuple[str, str, int]] = []

    def create_memory(self, request: CreateMemoryRequest) -> MemoryEntry:
        self.created.append(request)
        return MemoryEntry(
            memory_id="a" * 32,
            user_id=request.user_id,
            field=request.field.value,
            value=request.value,
            source_timestamp=request.source_timestamp,
            deleted_at=None,
        )

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> tuple[MemoryEntry, ...]:
        self.retrievals.append((user_id, query, limit))
        return ()


class CountingBridge:
    def __init__(self, bridge: SpeechDialogueBridge) -> None:
        self.bridge = bridge
        self.calls: list[tuple[str, str]] = []

    def reply(self, *, user_id: str, text: str) -> DialogueResponse:
        self.calls.append((user_id, text))
        return self.bridge.reply(user_id=user_id, text=text)


def _config(root: Path) -> tuple[BudgetConfig, ProviderConfig, StyleRuntimeSources]:
    budget = BudgetConfig(
        monthly_cap_cny=Decimal("100.00"),
        worst_case_request_cny=Decimal("1.00"),
        input_cny_per_million=Decimal("1.00"),
        output_cny_per_million=Decimal("2.00"),
        max_output_tokens=256,
    )
    provider = ProviderConfig(
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
    return budget, provider, style


def _recognition(text: str = "第一条合成问题") -> RecognitionInput:
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


def _make_orchestrator(
    root: Path,
    provider: object,
    *,
    memory: EmptyMemoryGateway | None = None,
    exhaust_budget: bool = False,
    ledger: MonthlyUsageLedger | None = None,
    turn_observer: TurnObserver | None = None,
) -> DialogueOrchestrator:
    budget, provider_config, style = _config(root)
    if ledger is None:
        ledger = MonthlyUsageLedger(root / "ledger", budget)
    if exhaust_budget:
        ledger.reserve(count=100, now=NOW)
    return DialogueOrchestrator(
        config=DialogueConfig(provider=provider_config, budget=budget, style=style),
        style_adapter=SyntheticStyle(),
        provider=provider,  # type: ignore[arg-type]
        ledger=ledger,
        clock=lambda: NOW,
        memory_gateway=memory,
        turn_observer=turn_observer,
    )


def _controller(
    *,
    user_id: str,
    dialogue: object,
    recognition: RecognitionInput,
    clock: ClockFake | None = None,
    timing_observer: Callable[[TurnTiming], None] | None = None,
) -> tuple[
    SpeechController,
    AudioRecorderFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
    AudioPlayerFake,
]:
    recorder = AudioRecorderFake()
    recognizer = SpeechRecognizerFake(result=recognition)
    synthesizer = SpeechSynthesizerFake()
    player = AudioPlayerFake()
    controller = SpeechController(
        trusted_user_id=user_id,
        recorder=recorder,
        recognizer=recognizer,
        dialogue=dialogue,  # type: ignore[arg-type]
        synthesizer=synthesizer,
        player=player,
        mute_gate=MuteGateFake(),
        clock=ClockFake() if clock is None else clock,
        timing_observer=timing_observer,
    )
    return controller, recorder, recognizer, synthesizer, player


def _complete_turn(controller: SpeechController, player: AudioPlayerFake) -> None:
    controller.button_pressed()
    controller.button_pressed()
    assert controller.state is SpeechState.PLAYING
    controller.playback_finished(player.tokens[-1])
    assert controller.state is SpeechState.IDLE


def _assert_resources_released(
    recorder: AudioRecorderFake,
    synthesizer: SpeechSynthesizerFake,
) -> None:
    assert recorder.live_session_count == 0
    assert recorder.live_recorded_audio_count == 0
    assert synthesizer.live_audio_count == 0


def test_two_accepted_turns_use_real_orchestrator_and_bounded_bridge_context(
    tmp_path: Path,
) -> None:
    provider = CaptureProvider()
    budget, provider_config, style = _config(tmp_path)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(provider=provider_config, budget=budget, style=style),
        style_adapter=SyntheticStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "ledger", budget),
        clock=lambda: NOW,
    )
    assert orchestrator._style_personalization is None
    port = CountingBridge(SpeechDialogueBridge(orchestrator=orchestrator))
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=port,
        recognition=_recognition("第一条合成问题"),
    )

    _complete_turn(controller, player)
    recognizer.result = _recognition("第二条合成问题")
    _complete_turn(controller, player)

    assert provider.call_count == 2
    assert port.calls == [(USER_A, "第一条合成问题"), (USER_A, "第二条合成问题")]
    assert "第一条合成问题" in provider.requests[1][0].content
    assert "合成安全回答" in synthesizer.inputs[0]
    assert synthesizer.audio_close_calls == 2
    _assert_resources_released(recorder, synthesizer)


def test_real_composition_accepts_exact_total_context_limit_and_evicts_oldest(
    tmp_path: Path,
) -> None:
    provider = CaptureProvider()
    budget, provider_config, style = _config(tmp_path)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(provider=provider_config, budget=budget, style=style),
        style_adapter=SyntheticStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "ledger", budget),
        clock=lambda: NOW,
    )
    bridge = SpeechDialogueBridge(orchestrator=orchestrator)
    utterance_length = 2_000 - len("对方\uff1a") - len("你\uff1a合成安全回答")
    first_three = [character * utterance_length for character in "甲乙丙"]
    # The first response also includes the AI disclosure and its newline.
    first_three[0] = first_three[0][: -(len(AI_DISCLOSURE) + 1)]
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=bridge,
        recognition=_recognition(first_three[0]),
    )

    for utterance in [*first_three, "第四条", "第五条"]:
        recognizer.result = _recognition(utterance)
        _complete_turn(controller, player)

    fourth_prompt = provider.requests[3][0].content
    assert all(utterance in fourth_prompt for utterance in first_three)
    fifth_prompt = provider.requests[4][0].content
    assert first_three[0] not in fifth_prompt
    assert all(utterance in fifth_prompt for utterance in [*first_three[1:], "第四条"])
    _assert_resources_released(recorder, synthesizer)


@pytest.mark.parametrize("rejection", ["vad", "blank", "recognizer", "quality"])
def test_recognition_rejection_stays_before_bridge_and_provider(
    tmp_path: Path,
    rejection: str,
) -> None:
    evidence = _recognition()
    if rejection == "vad":
        evidence = replace(evidence, vad_passed=False)
    elif rejection == "blank":
        evidence = replace(evidence, text=" ")
    elif rejection == "recognizer":
        evidence = replace(evidence, recognizer_failed=True)
    else:
        evidence = replace(
            evidence,
            audio=replace(evidence.audio, estimated_snr_db=None),
        )
    provider = CaptureProvider()
    budget, provider_config, style = _config(tmp_path)
    bridge = CountingBridge(
        SpeechDialogueBridge(
            orchestrator=DialogueOrchestrator(
                config=DialogueConfig(provider=provider_config, budget=budget, style=style),
                style_adapter=SyntheticStyle(),
                provider=provider,
                ledger=MonthlyUsageLedger(tmp_path / "ledger", budget),
                clock=lambda: NOW,
            )
        )
    )
    controller, recorder, _, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=bridge,
        recognition=evidence,
    )

    _complete_turn(controller, player)

    assert bridge.calls == []
    assert provider.call_count == 0
    assert synthesizer.inputs == [RECOGNITION_RETRY_PROMPT]
    _assert_resources_released(recorder, synthesizer)


def test_local_memory_command_never_enters_bridge_context_or_next_provider_prompt(
    tmp_path: Path,
) -> None:
    provider = CaptureProvider()
    memory = EmptyMemoryGateway()
    budget, provider_config, style = _config(tmp_path)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(provider=provider_config, budget=budget, style=style),
        style_adapter=SyntheticStyle(),
        provider=provider,
        ledger=MonthlyUsageLedger(tmp_path / "ledger", budget),
        clock=lambda: NOW,
        memory_gateway=memory,
    )
    bridge = SpeechDialogueBridge(orchestrator=orchestrator)
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=bridge,
        recognition=_recognition("记住我喜欢合成蓝色"),
    )

    _complete_turn(controller, player)
    recognizer.result = _recognition("下一条普通问题")
    _complete_turn(controller, player)

    assert len(memory.created) == 1
    assert provider.call_count == 1
    assert "记住我喜欢合成蓝色" not in provider.requests[0][0].content
    assert "记住我喜欢合成蓝色" not in provider.requests[0][1].content
    _assert_resources_released(recorder, synthesizer)


def test_disabled_memory_command_never_enters_next_bridge_prompt(tmp_path: Path) -> None:
    provider = CaptureProvider()
    bridge = SpeechDialogueBridge(orchestrator=_make_orchestrator(tmp_path, provider))
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=bridge,
        recognition=_recognition("记住我喜欢合成蓝色"),
    )

    _complete_turn(controller, player)

    assert provider.call_count == 0
    assert synthesizer.inputs[0].endswith("记忆功能还没开启，这句我没有记下来。")  # noqa: RUF001
    recognizer.result = _recognition("下一条普通问题")
    _complete_turn(controller, player)

    assert provider.call_count == 1
    assert all("合成蓝色" not in message.content for message in provider.requests[0])
    _assert_resources_released(recorder, synthesizer)


class MalformedDialogueRuntime:
    def reply(
        self,
        request: DialogueRequest,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        del request, trusted_user_id
        return object()  # type: ignore[return-value]


def test_bridge_structural_failure_uses_only_existing_connection_prompt() -> None:
    bridge = SpeechDialogueBridge(orchestrator=MalformedDialogueRuntime())
    controller, recorder, _, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=bridge,
        recognition=_recognition(),
    )

    _complete_turn(controller, player)

    assert synthesizer.inputs == [CONNECTION_RETRY_PROMPT]
    _assert_resources_released(recorder, synthesizer)


@pytest.mark.parametrize("failure_kind", ["timeout", "malformed", "unsafe", "budget"])
def test_real_dialogue_safe_failures_flow_to_tts_without_bridge_fabrication(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    if failure_kind == "timeout":
        provider: CaptureProvider = TimeoutProvider()
    elif failure_kind == "malformed":
        provider = MalformedProvider()
    elif failure_kind == "unsafe":
        provider = CaptureProvider("我代表主人承诺这笔合成交易。")
    else:
        provider = CaptureProvider()
    bridge = SpeechDialogueBridge(
        orchestrator=_make_orchestrator(
            tmp_path,
            provider,
            exhaust_budget=failure_kind == "budget",
        )
    )
    controller, recorder, _, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=bridge,
        recognition=_recognition(),
    )

    _complete_turn(controller, player)

    expected_provider_calls = (
        0 if failure_kind == "budget" else 2 if failure_kind == "timeout" else 1
    )
    assert provider.call_count == expected_provider_calls
    expected_fallback, expected_failure = {
        "timeout": (PROVIDER_TIMEOUT_FALLBACK, SpeechTurnFailure.PROVIDER_TIMEOUT),
        "malformed": (
            INVALID_PROVIDER_RESPONSE_FALLBACK,
            SpeechTurnFailure.INVALID_PROVIDER_RESPONSE,
        ),
        "unsafe": (UNSAFE_OUTPUT_FALLBACK, None),
        "budget": (BUDGET_FALLBACK, None),
    }[failure_kind]
    assert synthesizer.inputs == [AI_DISCLOSURE + "\n" + expected_fallback]
    assert controller.turn_failure is expected_failure
    if failure_kind == "budget":
        assert synthesizer.inputs[0].endswith(BUDGET_FALLBACK)
    assert synthesizer.inputs[0] != CONNECTION_RETRY_PROMPT
    assert "这笔合成交易" not in synthesizer.inputs[0]
    _assert_resources_released(recorder, synthesizer)


def test_shared_runtime_keeps_bridge_context_and_dialogue_identity_per_user(
    tmp_path: Path,
) -> None:
    provider = CaptureProvider()
    bridge = SpeechDialogueBridge(orchestrator=_make_orchestrator(tmp_path, provider))
    controller_a, recorder_a, _, synthesizer_a, player_a = _controller(
        user_id=USER_A,
        dialogue=bridge,
        recognition=_recognition("用户甲合成问题"),
    )
    controller_b, recorder_b, _, synthesizer_b, player_b = _controller(
        user_id=USER_B,
        dialogue=bridge,
        recognition=_recognition("用户乙合成问题"),
    )

    _complete_turn(controller_a, player_a)
    _complete_turn(controller_b, player_b)

    assert provider.call_count == 2
    assert "用户甲合成问题" not in provider.requests[1][0].content
    assert synthesizer_a.inputs[0].startswith(AI_DISCLOSURE)
    assert synthesizer_b.inputs[0].startswith(AI_DISCLOSURE)
    _assert_resources_released(recorder_a, synthesizer_a)
    _assert_resources_released(recorder_b, synthesizer_b)
    assert list(tmp_path.rglob("*.wav")) == []


@pytest.mark.parametrize("failure_kind", ["timeout", "HTTP_5XX", "CONNECT_UNAVAILABLE"])
def test_retry_exhausted_then_next_turn_succeeds(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    failures: list[Exception] = []
    for _ in range(2):
        if failure_kind == "timeout":
            failures.append(ProviderTimeout("R04_EXCEPTION_BODY"))
        else:
            failure = OpenAIProviderUnavailable(
                ProviderUnavailableKind[failure_kind],
                ProviderUnavailableStage.TRANSPORT_CALL,
            )
            failure.__cause__ = OSError("R04_EXCEPTION_BODY")
            failures.append(failure)
    provider = SequenceProvider(*failures, "合成恢复回答")
    budget, _, _ = _config(tmp_path)
    ledger = MonthlyUsageLedger(tmp_path / "ledger", budget)
    runtime = RecordingRuntime(_make_orchestrator(tmp_path, provider, ledger=ledger))
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=SpeechDialogueBridge(orchestrator=runtime),
        recognition=_recognition("第一轮合成输入"),
    )

    _complete_turn(controller, player)

    fallback = (
        PROVIDER_TIMEOUT_FALLBACK
        if failure_kind == "timeout"
        else PROVIDER_UNAVAILABLE_FALLBACK
    )
    expected_failure = (
        SpeechTurnFailure.PROVIDER_TIMEOUT
        if failure_kind == "timeout"
        else SpeechTurnFailure.PROVIDER_UNAVAILABLE
    )
    assert provider.call_count == 2
    assert provider.requests[0] == provider.requests[1]
    assert len(runtime.requests) == len(runtime.runs) == 1
    assert runtime.requests[0].recent_context == []
    assert runtime.runs[0].trace.provider_call_count == 2
    assert synthesizer.inputs == [AI_DISCLOSURE + "\n" + fallback]
    assert len(player.tokens) == 1
    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is None
    assert controller.turn_failure is expected_failure
    _assert_resources_released(recorder, synthesizer)
    assert ledger.snapshot(NOW).reserved_cny == Decimal("0")

    recognizer.result = _recognition("第二轮合成输入")
    _complete_turn(controller, player)

    assert provider.call_count == 3
    assert len(runtime.requests) == 2
    assert [run.trace.provider_call_count for run in runtime.runs] == [2, 1]
    assert runtime.requests[1].recent_context == ["对方\uff1a第一轮合成输入"]
    assert synthesizer.inputs == [AI_DISCLOSURE + "\n" + fallback, "合成恢复回答"]
    assert len(player.tokens) == 2
    assert controller.last_issue is None
    assert controller.turn_failure is None
    _assert_resources_released(recorder, synthesizer)
    assert ledger.snapshot(NOW).reserved_cny == Decimal("0")
    for messages in provider.requests:
        for message in messages:
            for forbidden in (
                PROVIDER_TIMEOUT_FALLBACK,
                PROVIDER_UNAVAILABLE_FALLBACK,
                "R04_EXCEPTION_BODY",
            ):
                assert forbidden not in message.content
    assert any("第一轮合成输入" in message.content for message in provider.requests[2])


def test_identical_text_in_later_turn_is_new_utterance(tmp_path: Path) -> None:
    provider = SequenceProvider(
        ProviderTimeout("R04_EXCEPTION_BODY"),
        ProviderTimeout("R04_EXCEPTION_BODY"),
        "合成恢复回答",
        "合成后续回答",
    )
    runtime = RecordingRuntime(_make_orchestrator(tmp_path, provider))
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=SpeechDialogueBridge(orchestrator=runtime),
        recognition=_recognition("相同合成输入"),
    )

    _complete_turn(controller, player)
    assert provider.call_count == 2
    _complete_turn(controller, player)

    assert provider.call_count == 3
    assert runtime.requests[1].recent_context == ["对方\uff1a相同合成输入"]
    recognizer.result = _recognition("第三轮合成输入")
    _complete_turn(controller, player)

    assert provider.call_count == 4
    assert [run.trace.provider_call_count for run in runtime.runs] == [2, 1, 1]
    assert runtime.requests[2].recent_context.count("对方\uff1a相同合成输入") == 2
    assert synthesizer.inputs == [
        AI_DISCLOSURE + "\n" + PROVIDER_TIMEOUT_FALLBACK,
        "合成恢复回答",
        "合成后续回答",
    ]
    assert len(player.tokens) == 3
    assert controller.turn_failure is None
    _assert_resources_released(recorder, synthesizer)


def test_retry_exhausted_then_tts_failure_still_allows_next_turn(tmp_path: Path) -> None:
    provider = SequenceProvider(
        ProviderTimeout("R04_EXCEPTION_BODY"),
        ProviderTimeout("R04_EXCEPTION_BODY"),
        "合成恢复回答",
    )
    runtime = RecordingRuntime(_make_orchestrator(tmp_path, provider))
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=SpeechDialogueBridge(orchestrator=runtime),
        recognition=_recognition("第一轮合成输入"),
    )
    synthesizer.failure = RuntimeError("R04_TTS_EXCEPTION")

    controller.button_pressed()
    controller.button_pressed()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.SYNTHESIS_FAILED
    assert controller.turn_failure is SpeechTurnFailure.PROVIDER_TIMEOUT
    assert provider.call_count == 2
    assert synthesizer.inputs == [AI_DISCLOSURE + "\n" + PROVIDER_TIMEOUT_FALLBACK]
    assert player.tokens == []
    _assert_resources_released(recorder, synthesizer)

    synthesizer.failure = None
    recognizer.result = _recognition("第二轮合成输入")
    _complete_turn(controller, player)

    assert provider.call_count == 3
    assert [run.trace.provider_call_count for run in runtime.runs] == [2, 1]
    assert synthesizer.inputs == [
        AI_DISCLOSURE + "\n" + PROVIDER_TIMEOUT_FALLBACK,
        "合成恢复回答",
    ]
    assert len(player.tokens) == 1
    assert controller.last_issue is None
    assert controller.turn_failure is None
    _assert_resources_released(recorder, synthesizer)


def test_dialogue_timing_includes_both_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ClockFake()

    class TimedSequenceProvider(SequenceProvider):
        def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
            clock.now += (0.25, 0.75)[self.call_count]
            return super().complete(messages)

    provider = TimedSequenceProvider(
        ProviderTimeout("R04_EXCEPTION_BODY"),
        ProviderTimeout("R04_EXCEPTION_BODY"),
    )
    timings: list[TurnTiming] = []
    observations: list[TurnObservation] = []
    runtime = RecordingRuntime(
        _make_orchestrator(tmp_path, provider, turn_observer=observations.append)
    )
    controller, recorder, _, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=SpeechDialogueBridge(orchestrator=runtime),
        recognition=_recognition(),
        clock=clock,
        timing_observer=timings.append,
    )
    monkeypatch.setattr("companion.dialogue.orchestrator.time.monotonic", clock.monotonic)

    _complete_turn(controller, player)

    assert len(timings) == 1
    assert len(observations) == 1
    assert timings[0].dialogue_ms == 1000
    assert timings[0].total_ms == 1000
    assert observations[0].provider_elapsed_ms == 750
    assert runtime.runs[0].trace.provider_call_count == provider.call_count == 2
    _assert_resources_released(recorder, synthesizer)


def test_retry_ledger_contains_no_turn_content(tmp_path: Path) -> None:
    provider = SequenceProvider(
        ProviderTimeout("R04_EXCEPTION_BODY"),
        ProviderTimeout("R04_EXCEPTION_BODY"),
        "合成回复_R04_REPLY_BODY",
    )
    budget, _, _ = _config(tmp_path)
    ledger = MonthlyUsageLedger(tmp_path / "ledger", budget)
    runtime = RecordingRuntime(_make_orchestrator(tmp_path, provider, ledger=ledger))
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=SpeechDialogueBridge(orchestrator=runtime),
        recognition=_recognition("合成输入_R04_INPUT_BODY"),
    )

    _complete_turn(controller, player)
    recognizer.result = _recognition("合成下一轮_R04_NEXT_BODY")
    _complete_turn(controller, player)

    assert provider.call_count == 3
    assert [run.trace.provider_call_count for run in runtime.runs] == [2, 1]
    assert runtime.requests[1].recent_context == ["对方\uff1a合成输入_R04_INPUT_BODY"]
    assert synthesizer.inputs[-1] == "合成回复_R04_REPLY_BODY"
    assert ledger.snapshot(NOW).reserved_cny == Decimal("0")
    ledger_path = tmp_path / "ledger" / "dialogue-usage-2026-08.json"
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == {ledger_path}
    contents = ledger_path.read_bytes()
    for sentinel in (
        "R04_INPUT_BODY",
        "R04_NEXT_BODY",
        "R04_REPLY_BODY",
        "R04_EXCEPTION_BODY",
    ):
        assert sentinel.encode("utf-8") not in contents
    _assert_resources_released(recorder, synthesizer)


@pytest.mark.parametrize("empty_reply", ["👍", "[憨笑]", "。\uff1f"])
def test_empty_reply_notice_never_enters_next_provider_prompt(
    tmp_path: Path,
    empty_reply: str,
) -> None:
    provider = SequenceProvider("合成开场回答", empty_reply, "合成后续回答")
    budget, _, _ = _config(tmp_path)
    ledger = MonthlyUsageLedger(tmp_path / "ledger", budget)
    runtime = RecordingRuntime(_make_orchestrator(tmp_path, provider, ledger=ledger))
    controller, recorder, recognizer, synthesizer, player = _controller(
        user_id=USER_A,
        dialogue=SpeechDialogueBridge(orchestrator=runtime),
        recognition=_recognition("合成开场问题"),
    )
    expected_tts = [
        AI_DISCLOSURE + "\n合成开场回答",
        EMPTY_REPLY_PROMPT,
        "合成后续回答",
    ]

    for turn_number, question in enumerate(
        ("合成开场问题", "合成空回复问题", "合成后续问题"), start=1
    ):
        recognizer.result = _recognition(question)
        _complete_turn(controller, player)

        assert provider.call_count == turn_number
        assert len(runtime.runs) == turn_number
        assert runtime.runs[-1].trace.provider_call_count == 1
        assert synthesizer.inputs == expected_tts[:turn_number]
        assert len(player.tokens) == turn_number
        assert controller.state is SpeechState.IDLE
        assert controller.last_issue is None
        assert controller.turn_failure is None
        _assert_resources_released(recorder, synthesizer)
        assert ledger.snapshot(NOW).reserved_cny == Decimal("0")

    assert runtime.runs[1].response.text == empty_reply
    assert runtime.runs[1].response.blocked_reason is None
    assert runtime.runs[1].response.billable is True
    assert runtime.requests[2].recent_context == [
        "对方\uff1a合成开场问题",
        "你\uff1a" + AI_DISCLOSURE + "\n合成开场回答",
        "对方\uff1a合成空回复问题",
        "你\uff1a" + empty_reply,
    ]
    assert all(
        EMPTY_REPLY_PROMPT not in message.content
        for messages in provider.requests
        for message in messages
    )
    expected_provider_reply = {
        "👍": "你\uff1a👍",
        "[憨笑]": "你\uff1a" + r"\[憨笑\]",
        "。\uff1f": "你\uff1a。\uff1f",
    }[empty_reply]
    assert any(
        expected_provider_reply in message.content for message in provider.requests[2]
    )
