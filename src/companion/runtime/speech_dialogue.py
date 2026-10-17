from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from companion.contracts import DialogueRequest, DialogueResponse
from companion.dialogue.context import (
    MAX_CONTEXT_ITEM_CHARS,
    MAX_CONTEXT_ITEMS,
    MAX_CONTEXT_TOTAL_CHARS,
    MAX_CURRENT_CHARS,
)
from companion.dialogue.models import DialogueMode, DialogueRun, DialogueTrace
from companion.dialogue.privacy_revision import PrivacyRevision

_BRIDGE_FAILURE = "speech-dialogue-bridge-failed"


class _DialogueRuntime(Protocol):
    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun: ...


class SpeechDialogueBridgeError(RuntimeError):
    """Fixed, content-free failure at the Speech/Dialogue runtime boundary."""


class _BridgeBoundaryFailure(RuntimeError):
    pass


def _raise_bridge_failure() -> None:
    raise SpeechDialogueBridgeError(_BRIDGE_FAILURE) from None


def _require_exact_model_storage(model: BaseModel) -> None:
    if set(model.__dict__) != set(type(model).model_fields) or model.__pydantic_extra__ is not None:
        raise _BridgeBoundaryFailure


class SpeechDialogueBridge:
    def __init__(
        self,
        *,
        orchestrator: _DialogueRuntime,
        privacy_revision: Callable[[str], PrivacyRevision] | None = None,
        budget_notice: Callable[[], None] | None = None,
    ) -> None:
        self._orchestrator: _DialogueRuntime | None = orchestrator
        self._privacy_revision = privacy_revision
        self._budget_notice = budget_notice
        self._privacy_revisions: dict[str, PrivacyRevision] = {}
        self._recent_context: dict[str, tuple[str, ...]] = {}
        self._call_in_progress = False
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._recent_context = {}
        self._privacy_revisions = {}
        self._privacy_revision = None
        self._budget_notice = None
        self._orchestrator = None

    def reply(self, *, user_id: str, text: str) -> DialogueResponse:
        if self._closed or self._call_in_progress:
            self = None  # type: ignore[assignment]
            user_id = None  # type: ignore[assignment]
            text = None  # type: ignore[assignment]
            _raise_bridge_failure()

        self._call_in_progress = True
        failed = False
        response: DialogueResponse | None = None
        try:
            response = self._reply_once(user_id=user_id, text=text)
        except Exception:
            failed = True
        finally:
            self._call_in_progress = False
            user_id = None  # type: ignore[assignment]
            text = None  # type: ignore[assignment]

        if failed or type(response) is not DialogueResponse:
            response = None
            self = None  # type: ignore[assignment]
            _raise_bridge_failure()
        return response

    def _reply_once(self, *, user_id: object, text: object) -> DialogueResponse:
        if (
            type(user_id) is not str
            or not user_id.strip()
            or len(user_id) > 120
            or type(text) is not str
            or not text.strip()
            or len(text) > MAX_CURRENT_CHARS
        ):
            raise _BridgeBoundaryFailure

        validated_user_id = (" " + user_id)[1:]
        validated_text = (" " + text)[1:]
        if self._privacy_revision is not None:
            try:
                revision = self._privacy_revision(validated_user_id)
            except Exception:
                self._recent_context.pop(validated_user_id, None)
                self._privacy_revisions.pop(validated_user_id, None)
                raise
            if self._closed:
                raise _BridgeBoundaryFailure
            if self._privacy_revisions.get(validated_user_id) != revision:
                self._recent_context.pop(validated_user_id, None)
                self._privacy_revisions[validated_user_id] = revision
        recent_context = list(self._recent_context.get(validated_user_id, ()))
        request = DialogueRequest(
            user_id=validated_user_id,
            text=validated_text,
            recent_context=recent_context,
        )
        recent_context = None  # type: ignore[assignment]
        orchestrator = self._orchestrator
        if orchestrator is None:
            raise _BridgeBoundaryFailure
        try:
            candidate = orchestrator.reply(
                request,
                trusted_user_id=validated_user_id,
            )
        finally:
            orchestrator = None
        request = None  # type: ignore[assignment]
        if self._closed:
            candidate = None
            raise _BridgeBoundaryFailure

        if type(candidate) is not DialogueRun:
            raise _BridgeBoundaryFailure
        if type(candidate.response) is not DialogueResponse:
            raise _BridgeBoundaryFailure
        if type(candidate.trace) is not DialogueTrace:
            raise _BridgeBoundaryFailure

        _require_exact_model_storage(candidate)
        _require_exact_model_storage(candidate.response)
        _require_exact_model_storage(candidate.trace)

        primitive = candidate.model_dump(mode="python", warnings="error")
        candidate = None  # type: ignore[assignment]
        validated_run = DialogueRun.model_validate(primitive, strict=True)
        primitive = None
        if type(validated_run.response) is not DialogueResponse:
            raise _BridgeBoundaryFailure
        if type(validated_run.trace) is not DialogueTrace:
            raise _BridgeBoundaryFailure
        provider_attempted = validated_run.trace.provider_attempted is True
        cache_eligible = (
            validated_run.response.blocked_reason is None
            and validated_run.response.billable is True
            and provider_attempted
        )
        response_data = validated_run.response.model_dump(mode="python")
        validated_run = None  # type: ignore[assignment]
        response = DialogueResponse.model_validate(response_data, strict=True)
        response_data = None
        if type(response) is not DialogueResponse:
            raise _BridgeBoundaryFailure
        if provider_attempted:
            self._store_context(validated_user_id, f"对方\uff1a{validated_text}")
        if cache_eligible and response.text.strip():
            self._store_context(validated_user_id, f"你\uff1a{response.text}")
        if self._budget_notice is not None:
            self._budget_notice()
        validated_user_id = None  # type: ignore[assignment]
        validated_text = None  # type: ignore[assignment]
        return response

    def _store_context(self, user_id: str, utterance: str) -> None:
        if len(utterance) > MAX_CONTEXT_ITEM_CHARS:
            return
        items = [*self._recent_context.get(user_id, ()), utterance]
        while (
            len(items) > MAX_CONTEXT_ITEMS
            or sum(len(item) for item in items) > MAX_CONTEXT_TOTAL_CHARS
        ):
            items.pop(0)
        self._recent_context[user_id] = tuple(items)
