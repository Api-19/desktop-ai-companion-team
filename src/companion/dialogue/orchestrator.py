# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from companion.contracts import (
    AddresseeRegister,
    DialogueRequest,
    DialogueResponse,
    MemoryEntry,
    StyleEvidence,
    StyleGuidanceStatus,
    StyleReplyGuidance,
)
from companion.dialogue.config import DialogueConfig
from companion.dialogue.context import (
    MAX_MEMORY_ITEMS,
    PERSONALIZATION_START,
    STYLE_END,
    STYLE_START,
    PreparedDialogueInput,
    PromptBundle,
    build_prompt,
    build_snippet_prompt,
    prepare_dialogue_input,
)
from companion.dialogue.identity import ensure_ai_disclosure
from companion.dialogue.memory import MemoryGateway, ParsedMemoryCommand, parse_memory_command
from companion.dialogue.models import (
    DialogueMode,
    DialogueRun,
    DialogueSessionState,
    DialogueTrace,
    ProviderResult,
)
from companion.dialogue.observer import TurnObservation, TurnObserver
from companion.dialogue.output_guard import (
    OutputBlockReason,
    UnsafeDialogueOutput,
    validate_dialogue_output,
)
from companion.dialogue.personalization import PersonalizationSpec, RetrievalStatus
from companion.dialogue.privacy_revision import (
    PRIVACY_REVISION_FAILURE,
    ContentRevisionSource,
    PrivacyRevision,
    PrivacyRevisionUnavailable,
)
from companion.dialogue.profile import PartnerProfileProvider
from companion.dialogue.provider import (
    DialogueProvider,
    InvalidProviderResponse,
    OpenAIProviderUnavailable,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailableKind,
)
from companion.dialogue.shared_terms import SharedTerm, match_shared_terms, shared_term_query_texts
from companion.dialogue.style_personalization import ProductionStylePersonalization
from companion.dialogue.usage import (
    BudgetExceeded,
    MonthlyUsageLedger,
    UsageReservation,
)
from companion.memory.profile import ProfileView
from companion.privacy.veiled import load_veiled_terms
from companion.style.evidence_retrieval import StyleEvidenceResult
from companion.style.verifier import (
    StyleVerificationAction,
    StyleVerificationReport,
)

if TYPE_CHECKING:
    from companion.dialogue.ab import ABPair, ABScenario

INVALID_REQUEST_FALLBACK = "这条消息我没能处理好，能重新发一下吗？"
CONFIGURATION_FALLBACK = "我这边的设置出了点问题，现在没法正常回复。让主人检查一下吧。"
BUDGET_FALLBACK = "这个月的云端聊天额度用完啦，云端回复先停一下，让主人看看额度吧。"
PROVIDER_TIMEOUT_FALLBACK = "这次回复等太久了，没能回来，过会儿再试吧。"
PROVIDER_UNAVAILABLE_FALLBACK = "我这边暂时连不上聊天服务，过会儿再试吧。"
INVALID_PROVIDER_RESPONSE_FALLBACK = "这次收到的回复出了点问题，没法正常给你，过会儿再试吧。"
UNSAFE_OUTPUT_FALLBACK = "刚才那句不合适，我先不说啦。"
MECHANICAL_REPEAT_FALLBACK = "刚才又重复了，我先停一下，别让你听一遍又一遍。"
STYLE_VERIFICATION_FALLBACK = "刚才那句说得不太对，我先不发啦。"
UNTRUSTED_USER_FALLBACK = "我还没确认你是谁，这次没有读取或保存记忆。"
MEMORY_DISABLED_FALLBACK = "记忆功能还没开启，这句我没有记下来。"
MEMORY_COMMAND_INVALID_FALLBACK = "想让我记住哪件事呀？请明确说“记住……”和具体内容。"
MEMORY_WRITE_FAILED_FALLBACK = "这次没能保存记忆，还没记下来，过会儿再试吧。"

_FALLBACK_TEXT = {
    "invalid_request": INVALID_REQUEST_FALLBACK,
    "configuration_unavailable": CONFIGURATION_FALLBACK,
    "budget_exceeded": BUDGET_FALLBACK,
    "provider_timeout": PROVIDER_TIMEOUT_FALLBACK,
    "provider_unavailable": PROVIDER_UNAVAILABLE_FALLBACK,
    "invalid_provider_response": INVALID_PROVIDER_RESPONSE_FALLBACK,
    "unsafe_output": UNSAFE_OUTPUT_FALLBACK,
    "profile_assertion": UNSAFE_OUTPUT_FALLBACK,
    "mechanical_style_repeat": MECHANICAL_REPEAT_FALLBACK,
    "style_verification_rejected": STYLE_VERIFICATION_FALLBACK,
}

_UNBILLED_PROVIDER_FAILURE_KINDS = frozenset(
    {
        ProviderUnavailableKind.DNS_UNAVAILABLE,
        ProviderUnavailableKind.CONNECT_UNAVAILABLE,
        ProviderUnavailableKind.TLS_UNAVAILABLE,
        ProviderUnavailableKind.REDIRECT_REFUSED,
        ProviderUnavailableKind.HTTP_400_BAD_REQUEST,
        ProviderUnavailableKind.HTTP_401_UNAUTHORIZED,
        ProviderUnavailableKind.HTTP_403_FORBIDDEN,
        ProviderUnavailableKind.HTTP_404_NOT_FOUND,
        ProviderUnavailableKind.HTTP_429_RATE_LIMITED,
    }
)

_RETRYABLE_PROVIDER_FAILURES: frozenset[str] = frozenset(
    {
        "timeout",
        "http_408_timeout",
        "http_5xx",
        "connect_unavailable",
        "dns_unavailable",
        "tls_unavailable",
        "transport_unknown",
    }
)


class StyleAdapter(Protocol):
    """Protocol for Style adapters that enforce user identity."""

    def accepts_user(self, user_id: str) -> bool: ...

    def guidance_for(
        self,
        user_id: str,
        text: str,
        recent_context: Sequence[str],
        idle_seconds: int | None,
    ) -> StyleReplyGuidance: ...


class SnippetStyleSource(Protocol):
    """Protocol for paired owner-snippet sources that enforce user identity."""

    def accepts_user(self, user_id: str) -> bool: ...

    def evidence_for(
        self, user_id: str, text: str, recent_context: Sequence[str]
    ) -> StyleEvidence: ...


class StyleResponseVerifier(Protocol):
    """Request-local verifier; implementations must not perform provider I/O."""

    def verify(
        self,
        *,
        user_id: str,
        request_text: str,
        response_text: str,
        guidance: StyleReplyGuidance,
        mode: str,
        previous_soft_lock: str | None,
        retrieval_result: StyleEvidenceResult | None = None,
    ) -> StyleVerificationReport: ...


@dataclass
class _AttemptOutcome:
    provider_failure: str | None = None


@dataclass
class _TurnObservationContext:
    input_chars: int = 0
    context_count: int = 0
    context_chars: int = 0
    similar_count: int = 0
    general_count: int = 0
    profile_count: int = 0
    provider_failure: str | None = None
    provider_elapsed_ms: int | None = None
    system_prompt_sha256: str | None = None


class DialogueOrchestrator:
    """One-shot dialogue orchestration with conservative local degradation."""

    def __init__(
        self,
        *,
        config: DialogueConfig,
        style_adapter: StyleAdapter | None = None,
        snippet_style: SnippetStyleSource | None = None,
        snippet_habit_card: str | None = None,
        snippet_shared_terms: Sequence[SharedTerm] | None = None,
        snippet_session_check: bool = False,
        veiled_terms_path: Path | None = None,
        provider: DialogueProvider,
        ledger: MonthlyUsageLedger,
        clock: Callable[[], datetime],
        test_results_dir: Path | None = None,
        memory_gateway: MemoryGateway | None = None,
        profile_source: PartnerProfileProvider | None = None,
        style_verifier: StyleResponseVerifier | None = None,
        style_personalization: ProductionStylePersonalization | None = None,
        turn_observer: TurnObserver | None = None,
        privacy_revision_sources: Sequence[ContentRevisionSource] = (),
    ) -> None:
        if snippet_style is None and snippet_habit_card is not None:
            raise ValueError("habit card requires snippet style")
        if snippet_style is None and snippet_shared_terms is not None:
            raise ValueError("shared terms require snippet style")
        if (style_adapter is None) == (snippet_style is None):
            raise ValueError("exactly one of style adapter or snippet style is required")
        if snippet_style is not None and (
            style_verifier is not None or style_personalization is not None
        ):
            raise ValueError("snippet style excludes style verifier and personalization")
        if style_personalization is not None and style_verifier is None:
            raise ValueError("production personalization requires style verifier")
        self._config = config
        self._style = style_adapter
        self._snippet_style = snippet_style
        self._snippet_habit_card = snippet_habit_card
        self._snippet_shared_terms = tuple(snippet_shared_terms or ())
        self._snippet_session_check = snippet_session_check
        self._veiled = (
            load_veiled_terms(veiled_terms_path)
            if snippet_style is not None and veiled_terms_path is not None
            else None
        )
        self._provider = provider
        self._ledger = ledger
        self._clock = clock
        self._test_results_dir = (
            test_results_dir.resolve() if test_results_dir is not None else None
        )
        self._memory = memory_gateway
        self._profile_source = profile_source
        self._style_verifier = style_verifier
        self._style_personalization = style_personalization
        self._turn_observer = turn_observer
        self._privacy_revision_sources = tuple(privacy_revision_sources)
        self._sessions: dict[str, DialogueSessionState] = {}

    def privacy_revision(self, user_id: str) -> PrivacyRevision:
        """Read every explicit source in order, failing without partial results."""
        try:
            return tuple(
                source.content_revision(user_id) for source in self._privacy_revision_sources
            )
        except Exception:
            pass
        raise PrivacyRevisionUnavailable(PRIVACY_REVISION_FAILURE) from None

    def ensure_paid_capacity(self, count: int) -> None:
        """Check capacity without creating a reservation or exposing accounting."""
        self._ledger.ensure_available(count=count, now=self._clock())

    def resolve_test_results_dir(self, supplied: Path) -> Path | None:
        """Return the bound evidence root without raising across the public boundary."""
        try:
            resolved = supplied.resolve()
        except Exception:
            return None
        if self._test_results_dir is None or resolved != self._test_results_dir:
            return None
        return resolved

    def preflight_framework20(
        self,
        scenarios: Sequence[ABScenario],
        *,
        trusted_user_id: str,
        results_dir: Path,
    ) -> None:
        """Validate an owner-bound framework suite without provider calls or answer files."""
        from companion.dialogue.ab import validate_framework20_scenarios

        try:
            if (
                self._style_personalization is None
                or self._style_verifier is None
                or self.resolve_test_results_dir(results_dir) is None
                or type(trusted_user_id) is not str
                or not self._style_accepts_user(trusted_user_id)
            ):
                raise ValueError
            validated = validate_framework20_scenarios(scenarios)
            now = self._aware_now()
            for scenario in validated:
                prepared = prepare_dialogue_input(
                    DialogueRequest(
                        user_id=trusted_user_id,
                        text=scenario.text,
                        recent_context=list(scenario.recent_context),
                    )
                )
                if _validated_trusted_user_id(trusted_user_id, prepared) is None:
                    raise ValueError
                spec = self._compile_personalization(
                    user_id=trusted_user_id,
                    prepared=prepared,
                    idle_seconds=scenario.idle_seconds,
                    now=now,
                    mode=DialogueMode.STYLED,
                )
                if spec is None or spec.retrieval_status is RetrievalStatus.CORRUPT:
                    raise ValueError
                guidance_candidate = self._style.guidance_for(
                    trusted_user_id,
                    prepared.text,
                    prepared.recent_context,
                    scenario.idle_seconds,
                )
                guidance = StyleReplyGuidance.model_validate(
                    guidance_candidate.model_dump(mode="python", warnings=False)
                )
                baseline = build_prompt(prepared, guidance, DialogueMode.BASELINE, None)
                styled = build_prompt(
                    prepared,
                    guidance,
                    DialogueMode.STYLED,
                    None,
                    personalization=spec,
                )
                if (
                    PERSONALIZATION_START in baseline.messages[0].content
                    or PERSONALIZATION_START not in styled.messages[0].content
                    or baseline.messages[0].content == styled.messages[0].content
                    or baseline.messages[1] != styled.messages[1]
                ):
                    raise ValueError
        except Exception as error:
            _detach_exception(error)
            raise ValueError("framework preflight unavailable") from None

    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        observation_context = (
            _TurnObservationContext() if self._turn_observer is not None else None
        )
        run = self._reply(
            request, mode, trusted_user_id=trusted_user_id,
            observation_context=observation_context,
        )
        request = None  # type: ignore[assignment]
        if observation_context is not None:
            observation = TurnObservation(
                blocked_reason=run.response.blocked_reason,
                input_chars=observation_context.input_chars,
                context_count=observation_context.context_count,
                context_chars=observation_context.context_chars,
                reply_chars=len(run.response.text),
                similar_count=observation_context.similar_count,
                general_count=observation_context.general_count,
                profile_count=observation_context.profile_count,
                provider_attempted=run.trace.provider_attempted,
                provider_failure=observation_context.provider_failure,
                provider_elapsed_ms=observation_context.provider_elapsed_ms,
                system_prompt_sha256=observation_context.system_prompt_sha256,
                observed_at=datetime.now(UTC),
            )
            try:
                self._turn_observer(observation)  # type: ignore[misc]
            except Exception as error:
                _detach_exception(error)
        return run

    def _reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode,
        *,
        trusted_user_id: str | None,
        observation_context: _TurnObservationContext | None,
    ) -> DialogueRun:
        prepared: PreparedDialogueInput | None = None
        try:
            validated_mode = DialogueMode(mode)
            prepared = prepare_dialogue_input(request)
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            request = None  # type: ignore[assignment]
            prepared = None
            return self._local_run(
                blocked_reason="invalid_request",
                billable=False,
                guidance_status=StyleGuidanceStatus.UNAVAILABLE,
                mode=DialogueMode.STYLED,
                verifier_enabled=self._style_verifier is not None,
            )
        request = None  # type: ignore[assignment]
        if observation_context is not None:
            observation_context.input_chars = len(prepared.text)
            observation_context.context_count = len(prepared.recent_context)
            observation_context.context_chars = sum(len(item) for item in prepared.recent_context)

        # Validate identity regardless of whether Memory is enabled
        validated_user_id = _validated_trusted_user_id(trusted_user_id, prepared)
        if validated_user_id is None or not self._style_accepts_user(validated_user_id):
            return self._memory_local_run(
                text=UNTRUSTED_USER_FALLBACK,
                blocked_reason="untrusted_user",
                mode=validated_mode,
                session=DialogueSessionState(),
                now=None,
            )

        if self._memory is None:
            if prepared.text.startswith("记住"):
                return self._memory_local_run(
                    text=MEMORY_DISABLED_FALLBACK,
                    blocked_reason="memory_disabled",
                    mode=validated_mode,
                    session=DialogueSessionState(),
                    now=None,
                )
            return self._reply_prepared(
                prepared,
                validated_mode,
                verified_user_id=validated_user_id,
                observation_context=observation_context,
            )

        session = self._sessions.get(prepared.user_id, DialogueSessionState())
        try:
            now = self._aware_now()
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            return self._memory_local_run(
                text=CONFIGURATION_FALLBACK,
                blocked_reason="configuration_unavailable",
                mode=validated_mode,
                session=session,
                now=None,
            )

        command = parse_memory_command(
            prepared.text,
            user_id=validated_user_id,
            source_timestamp=now,
        )
        if command is not None:
            return self._create_memory_locally(
                command,
                user_id=validated_user_id,
                mode=validated_mode,
                session=session,
                now=now,
            )
        if prepared.text.startswith("记住"):
            return self._memory_local_run(
                text=MEMORY_COMMAND_INVALID_FALLBACK,
                blocked_reason="memory_command_invalid",
                mode=validated_mode,
                session=session,
                now=now,
                user_id=prepared.user_id,
            )

        retrieved_memories: object | None = None
        try:
            retrieved_memories = self._memory.retrieve_memories(
                validated_user_id,
                prepared.text,
                limit=5,
            )
            memories = _validated_memory_batch(
                retrieved_memories,
                user_id=validated_user_id,
            )
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            retrieved_memories = None
            memories = ()
        retrieved_memories = None
        return self._reply_prepared(
            prepared,
            validated_mode,
            verified_user_id=validated_user_id,
            memories=memories,
            observation_context=observation_context,
        )

    def reply_ab(
        self,
        scenario_id: str,
        request: DialogueRequest,
        evaluation_idle_seconds: int | None = None,
        *,
        trusted_user_id: str | None = None,
    ) -> ABPair:
        from companion.dialogue.ab import ABPair, ABStatus, _blind_order

        try:
            if self._snippet_style is not None:
                raise ValueError("snippet style has no A/B comparison")
            if re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", scenario_id) is None:
                raise ValueError
            if evaluation_idle_seconds is not None and (
                isinstance(evaluation_idle_seconds, bool)
                or not isinstance(evaluation_idle_seconds, int)
                or evaluation_idle_seconds < 0
            ):
                raise ValueError
            prepared = prepare_dialogue_input(request)

            # Validate identity regardless of whether Memory is enabled
            validated_user_id = _validated_trusted_user_id(trusted_user_id, prepared)
            if validated_user_id is None or not self._style_accepts_user(validated_user_id):
                return ABPair(
                    scenario_id=scenario_id, status=ABStatus.FAILED, request_attempt_count=0
                )

            user_id = validated_user_id
            now = self._aware_now()
            session = self._sessions.get(
                prepared.user_id,
                DialogueSessionState(),
            ).model_copy(deep=True)
            idle_seconds = (
                evaluation_idle_seconds
                if evaluation_idle_seconds is not None
                else self._idle_seconds(session, now)
            )
            personalization = self._compile_personalization(
                user_id=user_id,
                prepared=prepared,
                idle_seconds=idle_seconds,
                now=now,
                mode=DialogueMode.STYLED,
            )
            _reject_corrupt_retrieval(personalization)
            guidance_candidate = (
                self._style.guidance_for(
                    user_id, prepared.text, prepared.recent_context, idle_seconds
                )
                if self._style_personalization is None or personalization is not None
                else StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)
            )
            guidance = StyleReplyGuidance.model_validate(
                guidance_candidate.model_dump(mode="python", warnings=False)
            )
            prompts = {
                mode: build_prompt(
                    prepared,
                    guidance,
                    mode,
                    session.previous_soft_lock,
                    personalization=(personalization if mode is DialogueMode.STYLED else None),
                )
                for mode in (DialogueMode.BASELINE, DialogueMode.STYLED)
            }
            baseline = prompts[DialogueMode.BASELINE]
            baseline_system = baseline.messages[0].model_copy(
                update={
                    "content": (f"{baseline.messages[0].content}\n\n{STYLE_START}\n{STYLE_END}")
                }
            )
            prompts[DialogueMode.BASELINE] = baseline.model_copy(
                update={"messages": (baseline_system, baseline.messages[1])}
            )
            verification_request_text = prepared.text
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            request = None  # type: ignore[assignment]
            prepared = None  # type: ignore[assignment]
            guidance_candidate = None  # type: ignore[assignment]
            return ABPair(scenario_id=scenario_id, status=ABStatus.FAILED, request_attempt_count=0)

        request = None  # type: ignore[assignment]
        prepared = None  # type: ignore[assignment]
        guidance_candidate = None  # type: ignore[assignment]
        if personalization is None and guidance.status is not StyleGuidanceStatus.MATCHED:
            status = (
                ABStatus.NOT_MATCHED
                if guidance.status is StyleGuidanceStatus.NO_MATCH
                else ABStatus.FAILED
            )
            return ABPair(scenario_id=scenario_id, status=status, request_attempt_count=0)

        try:
            reservations = self._ledger.reserve(count=2, now=now)
        except BudgetExceeded as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            return ABPair(
                scenario_id=scenario_id, status=ABStatus.BUDGET_EXCEEDED, request_attempt_count=0
            )
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            return ABPair(scenario_id=scenario_id, status=ABStatus.FAILED, request_attempt_count=0)

        modes = _blind_order(scenario_id)
        answer_key = {"A": modes[0], "B": modes[1]}
        answers: dict[str, DialogueResponse] = {}
        attempted = 0
        for label, mode, reservation in zip(
            ("A", "B"),
            modes,
            reservations,
            strict=True,
        ):
            run = self._complete_once(
                user_id=user_id,
                request_text=verification_request_text,
                session=session.model_copy(deep=True),
                guidance=guidance,
                prompt=prompts[mode],
                reservation=reservation,
                mode=mode,
                now=now,
                persist_session=False,
                personalization=(personalization if mode is DialogueMode.STYLED else None),
                observation_context=None,
            )
            attempted += run.trace.provider_call_count
            if run.response.blocked_reason is None:
                answers[label] = run.response

        status = (
            ABStatus.COMPLETE
            if len(answers) == 2
            else ABStatus.PARTIAL
            if answers
            else ABStatus.FAILED
        )
        return ABPair(
            scenario_id=scenario_id,
            status=status,
            answers=answers,
            answer_key=answer_key,
            request_attempt_count=attempted,
        )

    def _style_accepts_user(self, user_id: str) -> bool:
        """Fail closed if the configured Style boundary cannot prove ownership."""
        try:
            source = self._style if self._snippet_style is None else self._snippet_style
            return source.accepts_user(user_id) is True
        except Exception as error:
            _detach_exception(error)
            return False

    def _compile_personalization(
        self,
        *,
        user_id: str,
        prepared: PreparedDialogueInput,
        idle_seconds: int | None,
        now: datetime,
        mode: DialogueMode,
    ) -> PersonalizationSpec | None:
        if self._style_personalization is None or mode is DialogueMode.BASELINE:
            return None
        return self._style_personalization.compile(
            user_id=user_id,
            prepared=prepared,
            idle_seconds=idle_seconds,
            observed_at=now,
        )

    def _reply_prepared(
        self,
        prepared: PreparedDialogueInput,
        mode: DialogueMode,
        *,
        verified_user_id: str,
        memories: Sequence[MemoryEntry] = (),
        observation_context: _TurnObservationContext | None = None,
    ) -> DialogueRun:
        user_id = prepared.user_id
        session = self._sessions.get(user_id, DialogueSessionState())
        now: datetime | None = None
        extra_examples: tuple[str, ...] = ()
        style_evidence_count: int | None = None
        profile_topics: tuple[str, ...] = ()
        try:
            now = self._aware_now()
            if self._snippet_style is not None:
                profile = self._retrieve_profile(prepared, user_id=verified_user_id)
                profile_topics = tuple(item.topic for item in profile)
                personalization = None
                (
                    guidance, prompt, extra_examples, style_evidence_count, evidence
                ) = self._snippet_prompt(
                    prepared, mode, user_id=verified_user_id, memories=memories, profile=profile
                )
                if observation_context is not None:
                    observation_context.similar_count = len(evidence.similar)
                    observation_context.general_count = len(evidence.general)
                    observation_context.profile_count = len(profile)
                del profile
            else:
                idle_seconds = self._idle_seconds(session, now)
                personalization = self._compile_personalization(
                    user_id=verified_user_id,
                    prepared=prepared,
                    idle_seconds=idle_seconds,
                    now=now,
                    mode=mode,
                )
                _reject_corrupt_retrieval(personalization)
                guidance_candidate = (
                    self._style.guidance_for(
                        verified_user_id, prepared.text, prepared.recent_context, idle_seconds
                    )
                    if (
                        self._style_personalization is None
                        or mode is DialogueMode.BASELINE
                        or personalization is not None
                    )
                    else StyleReplyGuidance(status=StyleGuidanceStatus.UNAVAILABLE)
                )
                guidance = StyleReplyGuidance.model_validate(
                    guidance_candidate.model_dump(mode="python", warnings=False)
                )
                prompt = build_prompt(
                    prepared,
                    guidance,
                    mode,
                    session.previous_soft_lock,
                    memories=memories,
                    personalization=personalization,
                )
            verification_request_text = prepared.text
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            prepared = None  # type: ignore[assignment]
            guidance_candidate = None  # type: ignore[assignment]
            guidance = None  # type: ignore[assignment]
            prompt = None  # type: ignore[assignment]
            if now is None:
                return self._local_run(
                    blocked_reason="configuration_unavailable",
                    billable=False,
                    guidance_status=StyleGuidanceStatus.UNAVAILABLE,
                    mode=mode,
                    already_disclosed=session.identity_disclosed,
                )
            return self._local_session_run(
                user_id=user_id,
                session=session,
                now=now,
                blocked_reason="configuration_unavailable",
                billable=False,
                guidance_status=StyleGuidanceStatus.UNAVAILABLE,
                mode=mode,
            )

        try:
            reservation = self._ledger.reserve(count=1, now=now)[0]
        except BudgetExceeded as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            prepared = None  # type: ignore[assignment]
            prompt = None  # type: ignore[assignment]
            return self._local_session_run(
                user_id=user_id,
                session=session,
                now=now,
                blocked_reason="budget_exceeded",
                billable=False,
                guidance_status=guidance.status,
                style_version=guidance.style_version,
                mode=mode,
                style_evidence_count=style_evidence_count,
            )
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            prepared = None  # type: ignore[assignment]
            prompt = None  # type: ignore[assignment]
            return self._local_session_run(
                user_id=user_id,
                session=session,
                now=now,
                blocked_reason="configuration_unavailable",
                billable=False,
                guidance_status=guidance.status,
                style_version=guidance.style_version,
                mode=mode,
                style_evidence_count=style_evidence_count,
            )

        prepared = None  # type: ignore[assignment]
        attempt_outcome = _AttemptOutcome()
        run = self._complete_once(
            user_id=user_id,
            request_text=verification_request_text,
            session=session,
            guidance=guidance,
            prompt=prompt,
            reservation=reservation,
            mode=mode,
            now=now,
            personalization=personalization,
            extra_examples=extra_examples,
            style_evidence_count=style_evidence_count,
            profile_topics=profile_topics,
            observation_context=observation_context,
            attempt_outcome=attempt_outcome,
        )
        if (
            not run.trace.provider_attempted
            or attempt_outcome.provider_failure not in _RETRYABLE_PROVIDER_FAILURES
        ):
            return run

        try:
            reservation = self._ledger.reserve(count=1, now=now)[0]
        except Exception as error:
            _detach_exception(error)
            return run

        run = self._complete_once(
            user_id=user_id,
            request_text=verification_request_text,
            session=session,
            guidance=guidance,
            prompt=prompt,
            reservation=reservation,
            mode=mode,
            now=now,
            personalization=personalization,
            extra_examples=extra_examples,
            style_evidence_count=style_evidence_count,
            profile_topics=profile_topics,
            observation_context=observation_context,
            attempt_outcome=_AttemptOutcome(),
        )
        return run.model_copy(update={
            "trace": run.trace.model_copy(update={"provider_call_count": 2}),
        })

    def _retrieve_profile(
        self, prepared: PreparedDialogueInput, *, user_id: str
    ) -> tuple[ProfileView, ...]:
        if self._profile_source is None:
            return ()
        try:
            profile = tuple(self._profile_source.retrieve_profile(
                user_id=user_id,
                query_texts=[prepared.text, *_partner_query_context(prepared.recent_context)[-2:]],
                limit=3,
            ))
            return profile if all(isinstance(item, ProfileView) for item in profile) else ()
        except Exception as error:
            _detach_exception(error)
            return ()

    def _snippet_prompt(
        self,
        prepared: PreparedDialogueInput,
        mode: DialogueMode,
        *,
        user_id: str,
        memories: Sequence[MemoryEntry],
        profile: Sequence[ProfileView],
    ) -> tuple[StyleReplyGuidance, PromptBundle, tuple[str, ...], int, StyleEvidence]:
        if mode is DialogueMode.STYLED:
            evidence = StyleEvidence.model_validate(
                self._snippet_style.evidence_for(  # type: ignore[union-attr]
                    user_id, prepared.text, _partner_query_context(prepared.recent_context)
                ).model_dump(mode="python", warnings=False)
            )
        else:
            evidence = StyleEvidence(
                style_version="baseline", addressee_register=AddresseeRegister.NEUTRAL
            )
        snippets = (*evidence.similar, *evidence.general)
        return (
            StyleReplyGuidance(
                status=StyleGuidanceStatus.NO_MATCH, style_version=evidence.style_version
            ),
            build_snippet_prompt(
                prepared,
                evidence,
                memories=memories,
                profile=profile,
                habit_card=self._snippet_habit_card,
                shared_terms=match_shared_terms(
                    self._snippet_shared_terms,
                    shared_term_query_texts(prepared.text, prepared.recent_context),
                ),
                character_name=self._config.character_name,
                session_check=self._snippet_session_check,
            ),
            tuple(line for snippet in snippets for line in snippet.owner_reply),
            len(snippets),
            evidence,
        )

    def _create_memory_locally(
        self,
        command: ParsedMemoryCommand,
        *,
        user_id: str,
        mode: DialogueMode,
        session: DialogueSessionState,
        now: datetime,
    ) -> DialogueRun:
        try:
            self._memory.create_memory(command.request)  # type: ignore[union-attr]
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
            return self._memory_local_run(
                text=MEMORY_WRITE_FAILED_FALLBACK,
                blocked_reason="memory_write_failed",
                mode=mode,
                session=session,
                now=now,
                user_id=user_id,
            )
        return self._memory_local_run(
            text="记住了。",
            blocked_reason=None,
            mode=mode,
            session=session,
            now=now,
            user_id=user_id,
        )

    def _memory_local_run(
        self,
        *,
        text: str,
        blocked_reason: str | None,
        mode: DialogueMode,
        session: DialogueSessionState,
        now: datetime | None,
        user_id: str | None = None,
    ) -> DialogueRun:
        response_text, disclosed = ensure_ai_disclosure(
            text,
            already_disclosed=session.identity_disclosed,
            character_name=self._config.character_name,
        )
        if user_id is not None and now is not None:
            self._sessions[user_id] = DialogueSessionState(
                last_completed_at=now,
                identity_disclosed=session.identity_disclosed or disclosed,
                previous_soft_lock=None,
            )
        return DialogueRun(
            response=DialogueResponse(
                text=response_text,
                disclosed_ai_identity=disclosed,
                billable=False,
                blocked_reason=blocked_reason,
            ),
            trace=DialogueTrace(
                guidance_status=StyleGuidanceStatus.UNAVAILABLE,
                mode=mode,
                provider_attempted=False,
                usage_reliable=False,
                used_soft_lock=False,
                verifier_enabled=self._style_verifier is not None,
            ),
        )

    def _complete_once(
        self,
        *,
        user_id: str,
        request_text: str,
        session: DialogueSessionState,
        guidance: StyleReplyGuidance,
        prompt: PromptBundle,
        reservation: UsageReservation,
        mode: DialogueMode,
        now: datetime,
        persist_session: bool = True,
        personalization: PersonalizationSpec | None = None,
        extra_examples: Sequence[str] = (),
        style_evidence_count: int | None = None,
        profile_topics: Sequence[str] = (),
        observation_context: _TurnObservationContext | None = None,
        attempt_outcome: _AttemptOutcome | None = None,
    ) -> DialogueRun:
        result: ProviderResult | None = None
        safe_text: str | None = None
        blocked_reason: str | None = None
        output_block_reason: OutputBlockReason | None = None
        usage_reliable = False
        settlement_failed = False
        provider_request_unbilled = False
        verification_action = "not_run"
        if personalization is not None and not self._source_still_current(
            user_id=user_id, personalization=personalization, reservation=reservation, now=now
        ):
            personalization = None
            reservation = None  # type: ignore[assignment]
            return self._local_session_run(
                user_id=user_id,
                session=session,
                now=now,
                blocked_reason="configuration_unavailable",
                billable=False,
                guidance_status=guidance.status,
                style_version=guidance.style_version,
                mode=mode,
                persist_session=persist_session,
            )
        try:
            try:
                if observation_context is None:
                    candidate = self._provider.complete(prompt.messages)
                else:
                    observation_context.provider_failure = None
                    observation_context.system_prompt_sha256 = hashlib.sha256(
                        prompt.messages[0].content.encode("utf-8")
                    ).hexdigest()
                    started_at = time.monotonic()
                    try:
                        candidate = self._provider.complete(prompt.messages)
                    finally:
                        observation_context.provider_elapsed_ms = round(
                            (time.monotonic() - started_at) * 1000
                        )
            except ProviderTimeout as error:
                if observation_context is not None:
                    observation_context.provider_failure = "timeout"
                if attempt_outcome is not None:
                    attempt_outcome.provider_failure = "timeout"
                _detach_exception(error)
                error = None  # type: ignore[assignment]
                blocked_reason = "provider_timeout"
            except InvalidProviderResponse as error:
                if observation_context is not None:
                    observation_context.provider_failure = "invalid_response"
                if attempt_outcome is not None:
                    attempt_outcome.provider_failure = "invalid_response"
                _detach_exception(error)
                error = None  # type: ignore[assignment]
                blocked_reason = "invalid_provider_response"
            except ProviderError as error:
                provider_failure = (
                    error.kind.value
                    if isinstance(error, OpenAIProviderUnavailable)
                    else "provider_error"
                )
                if observation_context is not None:
                    observation_context.provider_failure = provider_failure
                if attempt_outcome is not None:
                    attempt_outcome.provider_failure = provider_failure
                provider_request_unbilled = (
                    isinstance(error, OpenAIProviderUnavailable)
                    and error.kind in _UNBILLED_PROVIDER_FAILURE_KINDS
                )
                _detach_exception(error)
                error = None  # type: ignore[assignment]
                blocked_reason = "provider_unavailable"
            except Exception as error:
                if observation_context is not None:
                    observation_context.provider_failure = "unexpected_error"
                if attempt_outcome is not None:
                    attempt_outcome.provider_failure = "unexpected_error"
                _detach_exception(error)
                error = None  # type: ignore[assignment]
                blocked_reason = "provider_unavailable"
            else:
                try:
                    result = ProviderResult.model_validate(
                        candidate.model_dump(mode="python", warnings=False)
                    )
                except Exception as error:
                    if observation_context is not None:
                        observation_context.provider_failure = "invalid_response"
                    if attempt_outcome is not None:
                        attempt_outcome.provider_failure = "invalid_response"
                    _detach_exception(error)
                    error = None  # type: ignore[assignment]
                    blocked_reason = "invalid_provider_response"

            if result is not None:
                try:
                    safe_text = validate_dialogue_output(
                        result.text,
                        guidance,
                        session.previous_soft_lock,
                        extra_examples=extra_examples,
                        veiled=self._veiled,
                        profile_topics=profile_topics,
                    )
                except UnsafeDialogueOutput as error:
                    output_block_reason = error.reason
                    blocked_reason = (
                        error.reason.value
                        if error.reason in (
                            OutputBlockReason.MECHANICAL_STYLE_REPEAT,
                            OutputBlockReason.PROFILE_ASSERTION,
                        )
                        else "unsafe_output"
                    )
                    _detach_exception(error)
                    error = None  # type: ignore[assignment]
                except Exception as error:
                    _detach_exception(error)
                    error = None  # type: ignore[assignment]
                    safe_text = None
                    blocked_reason = "unsafe_output"
                    output_block_reason = OutputBlockReason.UNKNOWN

                if safe_text is not None and self._style_verifier is not None:
                    try:
                        verification = self._style_verifier.verify(
                            user_id=user_id,
                            request_text=request_text,
                            response_text=safe_text,
                            guidance=guidance,
                            mode=mode.value,
                            previous_soft_lock=session.previous_soft_lock,
                            retrieval_result=(
                                personalization.source_retrieval
                                if personalization is not None
                                else None
                            ),
                        )
                        if type(verification) is not StyleVerificationReport:
                            raise ValueError("invalid style verifier result")
                        validated_verification = StyleVerificationReport.model_validate(
                            verification.model_dump(mode="python", warnings=False)
                        )
                        if validated_verification.action is StyleVerificationAction.ACCEPT:
                            verification_action = "accepted"
                        else:
                            safe_text = None
                            blocked_reason = "style_verification_rejected"
                            verification_action = "rejected"
                    except Exception as error:
                        _detach_exception(error)
                        error = None  # type: ignore[assignment]
                        safe_text = None
                        blocked_reason = "style_verification_rejected"
                        verification_action = "rejected"
        finally:
            usage = result.usage if result is not None else None
            try:
                if provider_request_unbilled:
                    self._ledger.cancel_unused(reservation, now=now)
                else:
                    self._ledger.settle(reservation, usage=usage, now=now)
                    usage_reliable = usage is not None
            except Exception as error:
                _detach_exception(error)
                error = None  # type: ignore[assignment]
                settlement_failed = True
            candidate = None  # type: ignore[assignment]
            result = None
            usage = None
            reservation = None  # type: ignore[assignment]

        if settlement_failed:
            safe_text = None
            blocked_reason = "configuration_unavailable"

        if blocked_reason is not None or safe_text is None:
            safe_text = None
            return self._local_session_run(
                user_id=user_id,
                session=session,
                now=now,
                blocked_reason=blocked_reason or "provider_unavailable",
                billable=True,
                guidance_status=guidance.status,
                style_version=guidance.style_version,
                mode=mode,
                provider_attempted=True,
                usage_reliable=usage_reliable,
                used_soft_lock=prompt.used_soft_lock is not None,
                verification_action=verification_action,
                persist_session=persist_session,
                style_evidence_count=style_evidence_count,
                output_block_reason=output_block_reason,
            )

        response_text, disclosed = ensure_ai_disclosure(
            safe_text,
            already_disclosed=session.identity_disclosed,
            character_name=self._config.character_name,
        )
        if persist_session:
            self._sessions[user_id] = DialogueSessionState(
                last_completed_at=now,
                identity_disclosed=session.identity_disclosed or disclosed,
                previous_soft_lock=prompt.used_soft_lock,
            )
        return DialogueRun(
            response=DialogueResponse(
                text=response_text,
                disclosed_ai_identity=disclosed,
                billable=True,
                blocked_reason=None,
            ),
            trace=DialogueTrace(
                guidance_status=guidance.status,
                style_version=guidance.style_version,
                mode=mode,
                provider_attempted=True,
                usage_reliable=usage_reliable,
                used_soft_lock=prompt.used_soft_lock is not None,
                verifier_enabled=self._style_verifier is not None,
                verification_action=verification_action,
                provider_call_count=1,
                style_evidence_count=style_evidence_count,
            ),
        )

    def _source_still_current(
        self,
        *,
        user_id: str,
        personalization: PersonalizationSpec,
        reservation: UsageReservation,
        now: datetime,
    ) -> bool:
        """Refuse one unsent styled request whose compiled source is no longer current."""
        try:
            self._style_personalization.verify_current(user_id=user_id, spec=personalization)
        except Exception as error:
            _detach_exception(error)
            error = None  # type: ignore[assignment]
        else:
            return True
        try:
            # The provider has not been attempted for this reservation, so release it unused.
            self._ledger.cancel_unused(reservation, now=now)
        except Exception as error:
            # The reservation stays held; capacity is never reported as restored.
            _detach_exception(error)
            error = None  # type: ignore[assignment]
        return False

    def _local_session_run(
        self,
        *,
        user_id: str,
        session: DialogueSessionState,
        now: datetime,
        blocked_reason: str,
        billable: bool,
        guidance_status: StyleGuidanceStatus,
        mode: DialogueMode,
        style_version: str | None = None,
        provider_attempted: bool = False,
        usage_reliable: bool = False,
        used_soft_lock: bool = False,
        verification_action: str = "not_run",
        persist_session: bool = True,
        style_evidence_count: int | None = None,
        output_block_reason: OutputBlockReason | None = None,
    ) -> DialogueRun:
        run = self._local_run(
            blocked_reason=blocked_reason,
            billable=billable,
            guidance_status=guidance_status,
            style_version=style_version,
            mode=mode,
            already_disclosed=session.identity_disclosed,
            provider_attempted=provider_attempted,
            usage_reliable=usage_reliable,
            used_soft_lock=used_soft_lock,
            verifier_enabled=self._style_verifier is not None,
            verification_action=verification_action,
            provider_call_count=1 if provider_attempted else 0,
            style_evidence_count=style_evidence_count,
            output_block_reason=output_block_reason,
        )
        if persist_session:
            self._sessions[user_id] = DialogueSessionState(
                last_completed_at=now,
                identity_disclosed=session.identity_disclosed or run.response.disclosed_ai_identity,
                previous_soft_lock=None,
            )
        return run

    def _local_run(
        self,
        *,
        blocked_reason: str,
        billable: bool,
        guidance_status: StyleGuidanceStatus,
        mode: DialogueMode,
        style_version: str | None = None,
        already_disclosed: bool = False,
        provider_attempted: bool = False,
        usage_reliable: bool = False,
        used_soft_lock: bool = False,
        verifier_enabled: bool = False,
        verification_action: str = "not_run",
        provider_call_count: int = 0,
        style_evidence_count: int | None = None,
        output_block_reason: OutputBlockReason | None = None,
    ) -> DialogueRun:
        text, disclosed = ensure_ai_disclosure(
            _FALLBACK_TEXT[blocked_reason],
            already_disclosed=already_disclosed,
            character_name=self._config.character_name,
        )
        return DialogueRun(
            response=DialogueResponse(
                text=text,
                disclosed_ai_identity=disclosed,
                billable=billable,
                blocked_reason=blocked_reason,
            ),
            trace=DialogueTrace(
                guidance_status=guidance_status,
                style_version=style_version,
                mode=mode,
                provider_attempted=provider_attempted,
                usage_reliable=usage_reliable,
                used_soft_lock=used_soft_lock,
                verifier_enabled=verifier_enabled,
                verification_action=verification_action,
                provider_call_count=provider_call_count,
                style_evidence_count=style_evidence_count,
                output_block_reason=output_block_reason,
            ),
        )

    def _aware_now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("dialogue clock must be timezone-aware")
        return now

    @staticmethod
    def _idle_seconds(session: DialogueSessionState, now: datetime) -> int | None:
        if session.last_completed_at is None:
            return None
        elapsed = int((now - session.last_completed_at).total_seconds())
        return max(0, elapsed)


def _partner_query_context(recent_context: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        item.removeprefix("对方：") for item in recent_context if not item.startswith("你：")
    )


def _reject_corrupt_retrieval(personalization: PersonalizationSpec | None) -> None:
    if personalization is not None and personalization.retrieval_status is RetrievalStatus.CORRUPT:
        raise ValueError("style evidence unavailable")


def _detach_exception(error: BaseException) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        cause = current.__cause__
        context = current.__context__
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)


def _validated_trusted_user_id(
    trusted_user_id: str | None,
    prepared: PreparedDialogueInput,
) -> str | None:
    prepared_user_id = prepared.user_id
    if (
        type(trusted_user_id) is not str
        or type(prepared_user_id) is not str
        or trusted_user_id != prepared_user_id
    ):
        return None
    return prepared_user_id


def _validated_memory_batch(
    retrieved_memories: object,
    *,
    user_id: str,
) -> tuple[MemoryEntry, ...]:
    if type(user_id) is not str:
        raise ValueError("invalid trusted user")
    validated_entries: list[MemoryEntry] = []
    for entry in islice(iter(retrieved_memories), MAX_MEMORY_ITEMS):  # type: ignore[arg-type]
        if type(entry) is not MemoryEntry:
            raise ValueError("invalid memory entry")
        entry_user_id = entry.user_id
        if type(entry_user_id) is not str or entry_user_id != user_id:
            raise ValueError("invalid memory user")
        validated_entry = MemoryEntry.model_validate(
            {
                "memory_id": entry.memory_id,
                "user_id": entry_user_id,
                "field": entry.field,
                "value": entry.value,
                "source_timestamp": entry.source_timestamp,
                "deleted_at": entry.deleted_at,
            }
        )
        if type(validated_entry.user_id) is not str or validated_entry.user_id != user_id:
            raise ValueError("invalid memory user")
        validated_entries.append(validated_entry)
    return tuple(validated_entries)
