from __future__ import annotations

import sys
import traceback
import warnings
from collections.abc import Callable

import pytest

from companion.contracts import DialogueRequest, DialogueResponse, StyleGuidanceStatus
from companion.dialogue.context import (
    MAX_CONTEXT_ITEM_CHARS,
    MAX_CONTEXT_ITEMS,
    MAX_CONTEXT_TOTAL_CHARS,
    MAX_CURRENT_CHARS,
)
from companion.dialogue.models import DialogueMode, DialogueRun, DialogueTrace
from companion.dialogue.privacy_revision import PrivacyRevision
from companion.runtime.speech_dialogue import (
    SpeechDialogueBridge,
    SpeechDialogueBridgeError,
)

SYNTHETIC_USER_ID = "synthetic-owner"
SYNTHETIC_UTTERANCE = "合成公开问题"
SYNTHETIC_RESPONSE = "合成公开回答"


def _run(
    *,
    text: str = SYNTHETIC_RESPONSE,
    billable: bool = True,
    blocked_reason: str | None = None,
    provider_attempted: bool = True,
) -> DialogueRun:
    return DialogueRun(
        response=DialogueResponse(
            text=text,
            disclosed_ai_identity=True,
            billable=billable,
            blocked_reason=blocked_reason,
        ),
        trace=DialogueTrace(
            guidance_status=StyleGuidanceStatus.UNAVAILABLE,
            provider_attempted=provider_attempted,
        ),
    )


class RecordingDialogueRuntime:
    def __init__(self, run: object) -> None:
        self.run = run
        self.calls: list[tuple[DialogueRequest, str | None]] = []

    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        assert mode is DialogueMode.STYLED
        self.calls.append((request, trusted_user_id))
        return self.run  # type: ignore[return-value]


class RecordingPrivacyRevision:
    def __init__(self) -> None:
        self.value: PrivacyRevision = (1, "synthetic-snippets-v1")
        self.calls: list[str] = []
        self.failure: Exception | None = None

    def __call__(self, user_id: str) -> PrivacyRevision:
        self.calls.append(user_id)
        if self.failure is not None:
            raise self.failure
        return self.value


@pytest.mark.parametrize("kind", ["provider", "budget", "local"])
def test_budget_notice_runs_once_after_validated_reply(kind: str) -> None:
    events: list[str] = []
    run = _run(
        billable=kind == "provider",
        blocked_reason="budget_exceeded" if kind == "budget" else None,
        provider_attempted=kind == "provider",
    )

    class Runtime(RecordingDialogueRuntime):
        def reply(self, *args, **kwargs):
            result = super().reply(*args, **kwargs)
            events.append("runtime-returned")
            return result

    runtime = Runtime(run)

    def notice() -> None:
        assert events[-1] == "runtime-returned"
        expected = (
            (f"对方\uff1a{SYNTHETIC_UTTERANCE}", f"你\uff1a{SYNTHETIC_RESPONSE}")
            * len(runtime.calls)
            if kind == "provider" else ()
        )
        assert bridge._recent_context.get(SYNTHETIC_USER_ID, ()) == expected
        events.append("notice")

    bridge = SpeechDialogueBridge(orchestrator=runtime, budget_notice=notice)
    response = bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)
    assert response == run.response
    assert events == ["runtime-returned", "notice"]
    bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)
    assert runtime.calls[-1][0].recent_context == (
        [f"对方\uff1a{SYNTHETIC_UTTERANCE}", f"你\uff1a{SYNTHETIC_RESPONSE}"]
        if kind == "provider" else []
    )
    assert events == ["runtime-returned", "notice", "runtime-returned", "notice"]


@pytest.mark.parametrize("failure", [
    "invalid-input", "runtime", "invalid-run", "revision", "closed", "close-during-reply",
])
def test_budget_notice_skipped_when_bridge_has_no_validated_response(failure: str) -> None:
    notices: list[None] = []
    runtime = (
        ClosingRuntime(_run()) if failure == "close-during-reply" else ToggleFailureRuntime(_run())
    )
    revision = RecordingPrivacyRevision()
    bridge = SpeechDialogueBridge(
        orchestrator=runtime, privacy_revision=revision,
        budget_notice=lambda: notices.append(None),
    )
    if failure == "runtime":
        runtime.fail = True
    elif failure == "invalid-run":
        runtime.run = object()
    elif failure == "revision":
        revision.failure = RuntimeError("synthetic revision failure")
    elif failure == "closed":
        bridge.close()
    elif failure == "close-during-reply":
        runtime.bridge = bridge

    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(
            user_id=" " if failure == "invalid-input" else SYNTHETIC_USER_ID,
            text=SYNTHETIC_UTTERANCE,
        )
    assert notices == []


def test_close_releases_budget_notice_callback() -> None:
    notices: list[None] = []
    def callback() -> None:
        notices.append(None)
    bridge = SpeechDialogueBridge(
        orchestrator=RecordingDialogueRuntime(_run()), budget_notice=callback,
    )
    assert bridge._budget_notice is callback
    bridge.close()
    assert bridge._budget_notice is None
    assert notices == []


def test_revision_is_checked_before_recent_context_is_built() -> None:
    runtime = RecordingDialogueRuntime(_run())
    revision = RecordingPrivacyRevision()
    bridge = SpeechDialogueBridge(orchestrator=runtime, privacy_revision=revision)
    bridge._recent_context[SYNTHETIC_USER_ID] = ("对方\uff1asynthetic-unversioned-history",)

    bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)

    assert revision.calls == [SYNTHETIC_USER_ID]
    assert runtime.calls[0][0].recent_context == []
    assert bridge._privacy_revisions == {SYNTHETIC_USER_ID: revision.value}


def test_changed_revision_clears_only_that_user_context() -> None:
    runtime = RecordingDialogueRuntime(_run())
    revision = RecordingPrivacyRevision()
    bridge = SpeechDialogueBridge(orchestrator=runtime, privacy_revision=revision)
    other_user = SYNTHETIC_USER_ID + " "
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-old-history")
    bridge.reply(user_id=other_user, text="synthetic-other-history")
    other_context = bridge._recent_context[other_user]
    other_revision = bridge._privacy_revisions[other_user]
    revision.value = (2, "synthetic-snippets-v2")

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-new-history")

    assert revision.calls == [SYNTHETIC_USER_ID, other_user, SYNTHETIC_USER_ID]
    assert runtime.calls[-1][0].recent_context == []
    assert bridge._recent_context[other_user] == other_context
    assert bridge._privacy_revisions[other_user] == other_revision
    assert bridge._privacy_revisions[SYNTHETIC_USER_ID] == revision.value
    assert bridge._recent_context[SYNTHETIC_USER_ID] == (
        "对方\uff1asynthetic-new-history", f"你\uff1a{SYNTHETIC_RESPONSE}"
    )


def test_unchanged_revision_preserves_both_speakers() -> None:
    runtime = RecordingDialogueRuntime(_run())
    revision = RecordingPrivacyRevision()
    bridge = SpeechDialogueBridge(orchestrator=runtime, privacy_revision=revision)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)
    revision.value = tuple([1, "synthetic-snippets-v1"])

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-next-question")

    assert revision.calls == [SYNTHETIC_USER_ID, SYNTHETIC_USER_ID]
    assert runtime.calls[-1][0].recent_context == [
        f"对方\uff1a{SYNTHETIC_UTTERANCE}", f"你\uff1a{SYNTHETIC_RESPONSE}"
    ]


def test_revision_failure_clears_user_state_and_skips_runtime() -> None:
    runtime = RecordingDialogueRuntime(_run())
    revision = RecordingPrivacyRevision()
    bridge = SpeechDialogueBridge(orchestrator=runtime, privacy_revision=revision)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)
    bridge.reply(user_id="synthetic-other", text="synthetic-other-history")
    other_context = bridge._recent_context["synthetic-other"]
    other_revision = bridge._privacy_revisions["synthetic-other"]
    runtime.calls.clear()
    revision.failure = RuntimeError("synthetic-revision-failure")

    with pytest.raises(SpeechDialogueBridgeError, match=r"^speech-dialogue-bridge-failed$"):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-failing-question")

    assert runtime.calls == []
    assert bridge._recent_context == {"synthetic-other": other_context}
    assert bridge._privacy_revisions == {"synthetic-other": other_revision}


def test_revision_recovery_starts_without_old_history() -> None:
    runtime = RecordingDialogueRuntime(_run())
    revision = RecordingPrivacyRevision()
    bridge = SpeechDialogueBridge(orchestrator=runtime, privacy_revision=revision)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-old-history")
    revision.failure = RuntimeError("synthetic-revision-failure")
    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-failed-question")
    revision.failure = None

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-recovered-question")
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-next-question")

    assert revision.calls == [SYNTHETIC_USER_ID] * 4
    assert len(runtime.calls) == 3
    assert runtime.calls[1][0].recent_context == []
    assert runtime.calls[2][0].recent_context == [
        "对方\uff1asynthetic-recovered-question", f"你\uff1a{SYNTHETIC_RESPONSE}"
    ]


def test_close_releases_revision_state_and_callback() -> None:
    runtime = RecordingDialogueRuntime(_run())
    revision = RecordingPrivacyRevision()
    bridge = SpeechDialogueBridge(orchestrator=runtime, privacy_revision=revision)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)
    assert bridge._privacy_revisions

    bridge.close()
    bridge.close()

    assert bridge._privacy_revisions == {}
    assert bridge._privacy_revision is None
    assert bridge._recent_context == {}
    assert bridge._orchestrator is None
    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text="synthetic-after-close")
    assert revision.calls == [SYNTHETIC_USER_ID]
    assert len(runtime.calls) == 1


@pytest.mark.parametrize(
    ("user_id", "text"),
    [(" ", SYNTHETIC_UTTERANCE), (SYNTHETIC_USER_ID, " "), (1, SYNTHETIC_UTTERANCE)],
)
def test_invalid_input_skips_revision_callback(user_id: object, text: object) -> None:
    runtime = RecordingDialogueRuntime(_run())
    revision = RecordingPrivacyRevision()
    bridge = SpeechDialogueBridge(orchestrator=runtime, privacy_revision=revision)

    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(user_id=user_id, text=text)  # type: ignore[arg-type]

    assert revision.calls == []
    assert runtime.calls == []


def test_close_during_revision_check_keeps_terminal_state() -> None:
    runtime = RecordingDialogueRuntime(_run())

    def revision(user_id: str) -> PrivacyRevision:
        bridge.close()
        return (1,)

    bridge = SpeechDialogueBridge(orchestrator=runtime, privacy_revision=revision)

    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)

    assert bridge._privacy_revisions == {}
    assert bridge._privacy_revision is None
    assert bridge._recent_context == {}
    assert runtime.calls == []


def test_reply_builds_strict_initial_request_and_returns_fresh_response() -> None:
    original_run = _run()
    runtime = RecordingDialogueRuntime(original_run)
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    response = bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)

    assert len(runtime.calls) == 1
    request, trusted_user_id = runtime.calls[0]
    assert type(request) is DialogueRequest
    assert request == DialogueRequest(
        user_id=SYNTHETIC_USER_ID,
        text=SYNTHETIC_UTTERANCE,
        recent_context=[],
    )
    assert type(request.recent_context) is list
    assert trusted_user_id == SYNTHETIC_USER_ID
    assert type(trusted_user_id) is str
    assert type(response) is DialogueResponse
    assert response == original_run.response
    assert response is not original_run.response


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    ("user_id", "text"),
    [
        (StringSubclass(SYNTHETIC_USER_ID), SYNTHETIC_UTTERANCE),
        (True, SYNTHETIC_UTTERANCE),
        (" ", SYNTHETIC_UTTERANCE),
        ("u" * 121, SYNTHETIC_UTTERANCE),
        (SYNTHETIC_USER_ID, StringSubclass(SYNTHETIC_UTTERANCE)),
        (SYNTHETIC_USER_ID, False),
        (SYNTHETIC_USER_ID, "\t"),
        (SYNTHETIC_USER_ID, "x" * (MAX_CURRENT_CHARS + 1)),
    ],
)
def test_reply_rejects_invalid_exact_inputs_before_runtime_call(
    user_id: object,
    text: object,
) -> None:
    runtime = RecordingDialogueRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    with pytest.raises(SpeechDialogueBridgeError, match=r"^speech-dialogue-bridge-failed$"):
        bridge.reply(user_id=user_id, text=text)  # type: ignore[arg-type]

    assert runtime.calls == []


class DialogueRunSubclass(DialogueRun):
    pass


class DialogueResponseSubclass(DialogueResponse):
    pass


class DialogueTraceSubclass(DialogueTrace):
    pass


def _wrong_run_subclass() -> object:
    return DialogueRunSubclass.model_validate(_run().model_dump(mode="python"))


def _wrong_response_subclass() -> object:
    valid = _run()
    response = DialogueResponseSubclass.model_validate(valid.response.model_dump(mode="python"))
    return DialogueRun.model_construct(response=response, trace=valid.trace)


def _wrong_trace_subclass() -> object:
    valid = _run()
    trace = DialogueTraceSubclass.model_validate(valid.trace.model_dump(mode="python"))
    return DialogueRun.model_construct(response=valid.response, trace=trace)


@pytest.mark.parametrize(
    "candidate_factory",
    [
        lambda: object(),
        _wrong_run_subclass,
        _wrong_response_subclass,
        _wrong_trace_subclass,
        lambda: DialogueRun.model_construct(response="bad", trace=_run().trace),
        lambda: DialogueRun.model_construct(response=_run().response, trace="bad"),
    ],
)
def test_reply_rejects_wrong_or_malformed_run_without_fabricating_response(
    candidate_factory: Callable[[], object],
) -> None:
    runtime = RecordingDialogueRuntime(candidate_factory())
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    with pytest.raises(SpeechDialogueBridgeError, match=r"^speech-dialogue-bridge-failed$"):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)

    assert len(runtime.calls) == 1


@pytest.mark.parametrize(
    "malformed_run",
    [
        DialogueRun.model_construct(
            response=DialogueResponse.model_construct(
                text="synthetic-response",
                disclosed_ai_identity=True,
                billable=1,
                blocked_reason=None,
            ),
            trace=_run().trace,
        ),
        DialogueRun.model_construct(
            response=_run().response,
            trace=DialogueTrace.model_construct(
                guidance_status=StyleGuidanceStatus.MATCHED,
                style_version="synthetic-style",
                mode=DialogueMode.STYLED,
                provider_attempted="true",
                usage_reliable=False,
                used_soft_lock=False,
            ),
        ),
    ],
)
def test_reply_rejects_coercible_non_exact_boolean_fields(
    malformed_run: DialogueRun,
) -> None:
    runtime = RecordingDialogueRuntime(malformed_run)
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text="must-not-cache")

    runtime.run = _run()
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="inspect-cache")
    assert runtime.calls[1][0].recent_context == []


def test_reply_rejects_unknown_model_storage_fields() -> None:
    malformed = _run()
    malformed.__dict__["unexpected"] = "synthetic-unknown"
    runtime = RecordingDialogueRuntime(malformed)
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)


def test_reply_suppresses_serializer_warnings_with_malformed_response(
    capsys: pytest.CaptureFixture[str],
) -> None:
    malformed = DialogueRun.model_construct(
        response=DialogueResponse.model_construct(
            text=["sensitive-response-sentinel"],
            disclosed_ai_identity=True,
            billable=True,
            blocked_reason=None,
        ),
        trace=_run().trace,
    )
    bridge = SpeechDialogueBridge(orchestrator=RecordingDialogueRuntime(malformed))

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(SpeechDialogueBridgeError):
            bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)

    captured = capsys.readouterr()
    assert caught_warnings == []
    assert "sensitive-response-sentinel" not in captured.out
    assert "sensitive-response-sentinel" not in captured.err


class RaisingRuntime:
    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        del request, mode, trusted_user_id
        raise RuntimeError("third-party-sensitive-detail")


class EvilSetattrError(Exception):
    def __setattr__(self, name: str, value: object) -> None:
        if name in {"__traceback__", "__cause__", "__context__"}:
            raise RuntimeError("PRIVATE_SENTINEL")
        super().__setattr__(name, value)


class RecoveringEvilRuntime(RecordingDialogueRuntime):
    def __init__(self, run: DialogueRun) -> None:
        super().__init__(run)
        self.raise_evil = False

    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        if self.raise_evil:
            raise EvilSetattrError("private body")
        return super().reply(
            request,
            mode,
            trusted_user_id=trusted_user_id,
        )


class PrintingFailingDescriptor:
    def __init__(self, sentinel: str, *, stderr: bool = False) -> None:
        self._sentinel = sentinel
        self._stderr = stderr

    def __set__(self, instance: object, value: object) -> None:
        del instance, value
        print(self._sentinel, file=sys.stderr if self._stderr else sys.stdout)
        raise RuntimeError(self._sentinel)


class EvilDescriptorError(Exception):
    __traceback__ = PrintingFailingDescriptor("TRACE_DESCRIPTOR_PRIVATE_SENTINEL")
    __cause__ = PrintingFailingDescriptor("CAUSE_DESCRIPTOR_PRIVATE_SENTINEL", stderr=True)
    __context__ = PrintingFailingDescriptor("CONTEXT_DESCRIPTOR_PRIVATE_SENTINEL")


class RecoveringDescriptorRuntime(RecordingDialogueRuntime):
    def __init__(self, run: DialogueRun) -> None:
        super().__init__(run)
        self.raise_evil = False

    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        if self.raise_evil:
            raise EvilDescriptorError("descriptor private body")
        return super().reply(
            request,
            mode,
            trusted_user_id=trusted_user_id,
        )


def test_reply_replaces_runtime_exception_with_content_free_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bridge = SpeechDialogueBridge(orchestrator=RaisingRuntime())

    with pytest.raises(SpeechDialogueBridgeError) as caught:
        bridge.reply(user_id=SYNTHETIC_USER_ID, text=SYNTHETIC_UTTERANCE)

    error = caught.value
    rendered = "\n".join(traceback.format_exception(error))
    local_values: list[str] = []
    for frame, _line_number in traceback.walk_tb(error.__traceback__):
        local_values.extend(repr(value) for value in frame.f_locals.values())
    captured = capsys.readouterr()
    forbidden = (
        SYNTHETIC_USER_ID,
        SYNTHETIC_UTTERANCE,
        SYNTHETIC_RESPONSE,
        "third-party-sensitive-detail",
    )
    assert str(error) == "speech-dialogue-bridge-failed"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert all(value not in rendered for value in forbidden)
    assert all(value not in "\n".join(local_values) for value in forbidden)
    assert all(value not in caplog.text for value in forbidden)
    assert all(value not in captured.out for value in forbidden)
    assert all(value not in captured.err for value in forbidden)


def test_reply_contains_malicious_exception_detachment_failure_and_recovers(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cached_utterance = "cached-private-utterance"
    failing_user_id = "private-user-id"
    failing_text = "private-question"
    runtime = RecoveringEvilRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text=cached_utterance)
    runtime.raise_evil = True

    with pytest.raises(SpeechDialogueBridgeError) as caught:
        bridge.reply(user_id=failing_user_id, text=failing_text)

    error = caught.value
    rendered = "\n".join(traceback.format_exception(error))
    production_locals: list[str] = []
    cached_values_reachable_from_bridge: list[str] = []
    for frame, _line_number in traceback.walk_tb(error.__traceback__):
        if not frame.f_code.co_filename.replace("\\", "/").endswith(
            "/src/companion/runtime/speech_dialogue.py"
        ):
            continue
        production_locals.extend(repr(value) for value in frame.f_locals.values())
        for value in frame.f_locals.values():
            if isinstance(value, SpeechDialogueBridge):
                cached_values_reachable_from_bridge.extend(
                    item for items in value._recent_context.values() for item in items
                )
    captured = capsys.readouterr()
    forbidden = (
        "private body",
        "PRIVATE_SENTINEL",
        failing_user_id,
        failing_text,
        cached_utterance,
    )
    assert type(error) is SpeechDialogueBridgeError
    assert str(error) == "speech-dialogue-bridge-failed"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert all(value not in rendered for value in forbidden)
    assert all(value not in "\n".join(production_locals) for value in forbidden)
    assert cached_utterance not in cached_values_reachable_from_bridge
    assert all(value not in caplog.text for value in forbidden)
    assert all(value not in captured.out for value in forbidden)
    assert all(value not in captured.err for value in forbidden)

    runtime.raise_evil = False
    response = bridge.reply(user_id=SYNTHETIC_USER_ID, text="ordinary-follow-up")
    assert response.text == SYNTHETIC_RESPONSE


def test_reply_never_invokes_malicious_exception_data_descriptors(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cached_utterance = "descriptor-cached-private-utterance"
    failing_user_id = "descriptor-private-user-id"
    failing_text = "descriptor-private-question"
    runtime = RecoveringDescriptorRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text=cached_utterance)
    runtime.raise_evil = True

    with pytest.raises(SpeechDialogueBridgeError) as caught:
        bridge.reply(user_id=failing_user_id, text=failing_text)

    error = caught.value
    rendered = "\n".join(traceback.format_exception(error))
    production_locals: list[str] = []
    cached_values_reachable_from_bridge: list[str] = []
    for frame, _line_number in traceback.walk_tb(error.__traceback__):
        if not frame.f_code.co_filename.replace("\\", "/").endswith(
            "/src/companion/runtime/speech_dialogue.py"
        ):
            continue
        production_locals.extend(repr(value) for value in frame.f_locals.values())
        for value in frame.f_locals.values():
            if isinstance(value, SpeechDialogueBridge):
                cached_values_reachable_from_bridge.extend(
                    item for items in value._recent_context.values() for item in items
                )
    captured = capsys.readouterr()
    forbidden = (
        "descriptor private body",
        "TRACE_DESCRIPTOR_PRIVATE_SENTINEL",
        "CAUSE_DESCRIPTOR_PRIVATE_SENTINEL",
        "CONTEXT_DESCRIPTOR_PRIVATE_SENTINEL",
        failing_user_id,
        failing_text,
        cached_utterance,
    )
    assert type(error) is SpeechDialogueBridgeError
    assert str(error) == "speech-dialogue-bridge-failed"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert all(value not in rendered for value in forbidden)
    assert all(value not in "\n".join(production_locals) for value in forbidden)
    assert cached_utterance not in cached_values_reachable_from_bridge
    assert all(value not in caplog.text for value in forbidden)
    assert all(value not in captured.out for value in forbidden)
    assert all(value not in captured.err for value in forbidden)

    runtime.raise_evil = False
    response = bridge.reply(user_id=SYNTHETIC_USER_ID, text="descriptor-ordinary-follow-up")
    assert response.text == SYNTHETIC_RESPONSE


def test_reply_failure_traceback_cannot_reach_cached_utterances() -> None:
    cached_utterance = "cached-sensitive-utterance"
    runtime = RecordingDialogueRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text=cached_utterance)
    runtime.run = object()

    with pytest.raises(SpeechDialogueBridgeError) as caught:
        bridge.reply(user_id=SYNTHETIC_USER_ID, text="failing-utterance")

    reachable: list[str] = []
    bridge_locations: list[tuple[str, str]] = []
    for frame, _line_number in traceback.walk_tb(caught.value.__traceback__):
        if not frame.f_code.co_filename.replace("\\", "/").endswith(
            "/src/companion/runtime/speech_dialogue.py"
        ):
            continue
        for name, value in frame.f_locals.items():
            if isinstance(value, SpeechDialogueBridge):
                bridge_locations.append((frame.f_code.co_name, name))
                reachable.extend(item for items in value._recent_context.values() for item in items)
    assert cached_utterance not in reachable, bridge_locations


def test_reply_caches_both_speakers_for_next_request() -> None:
    runtime = RecordingDialogueRuntime(_run(text="回复一"))
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="第一句")
    runtime.run = _run(text="second-model-response")
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="second-user-utterance")

    assert runtime.calls[1][0].recent_context == ["对方\uff1a第一句", "你\uff1a回复一"]


@pytest.mark.parametrize("blocked_reason", ["provider_timeout", "synthetic-block"])
def test_reply_retains_other_speaker_after_failure_and_both_after_success(
    blocked_reason: str,
) -> None:
    runtime = RecordingDialogueRuntime(
        _run(text="失败轮回复", billable=False, blocked_reason=blocked_reason)
    )
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="合成更正")
    runtime.run = _run(text="成功轮回复")
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="合成下一句")

    assert runtime.calls[1][0].recent_context == ["对方\uff1a合成更正"]

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="检查前文")

    assert runtime.calls[2][0].recent_context == [
        "对方\uff1a合成更正", "对方\uff1a合成下一句", "你\uff1a成功轮回复"
    ]


def test_reply_retains_newest_suffix_within_total_character_limit() -> None:
    runtime = RecordingDialogueRuntime(_run(text=""))
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    utterances = [character * (MAX_CONTEXT_ITEM_CHARS - len("对方\uff1a")) for character in "abcd"]
    expected = [f"对方\uff1a{utterance}" for utterance in utterances]

    for utterance in utterances:
        bridge.reply(user_id=SYNTHETIC_USER_ID, text=utterance)

    fourth_context = runtime.calls[3][0].recent_context
    assert fourth_context == expected[:3]
    assert sum(map(len, fourth_context)) == MAX_CONTEXT_TOTAL_CHARS

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="inspect-newest-suffix")

    assert runtime.calls[4][0].recent_context == expected[1:]
    assert sum(map(len, runtime.calls[4][0].recent_context)) <= MAX_CONTEXT_TOTAL_CHARS


def test_reply_retains_at_most_context_item_limit_for_short_items() -> None:
    runtime = RecordingDialogueRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    utterances = [f"short-{index}" for index in range(6)]
    expected = []

    for index, utterance in enumerate(utterances):
        reply = f"reply-{index}"
        runtime.run = _run(text=reply)
        bridge.reply(user_id=SYNTHETIC_USER_ID, text=utterance)
        expected.extend([f"对方\uff1a{utterance}", f"你\uff1a{reply}"])
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="inspect-short-suffix")

    assert runtime.calls[-1][0].recent_context == expected[-MAX_CONTEXT_ITEMS:]


def test_reply_isolates_recent_context_by_exact_user_id() -> None:
    runtime = RecordingDialogueRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    bridge.reply(user_id="synthetic-user-a", text="utterance-a")
    bridge.reply(user_id="synthetic-user-b", text="utterance-b")
    bridge.reply(user_id="synthetic-user-a", text="utterance-a-2")
    bridge.reply(user_id="synthetic-user-b", text="utterance-b-2")

    assert runtime.calls[1][0].recent_context == []
    assert runtime.calls[2][0].recent_context == [
        "对方\uff1autterance-a", f"你\uff1a{SYNTHETIC_RESPONSE}"
    ]
    assert runtime.calls[3][0].recent_context == [
        "对方\uff1autterance-b", f"你\uff1a{SYNTHETIC_RESPONSE}"
    ]


@pytest.mark.parametrize(
    ("not_blocked", "billable", "provider_attempted"),
    [
        (not_blocked, billable, provider_attempted)
        for not_blocked in (False, True)
        for billable in (False, True)
        for provider_attempted in (False, True)
    ],
)
def test_reply_assistant_cache_requires_all_three_dialogue_run_signals(
    not_blocked: bool,
    billable: bool,
    provider_attempted: bool,
) -> None:
    first_run = _run(
        text="authoritative-safe-response",
        billable=billable,
        blocked_reason=None if not_blocked else "synthetic-block",
        provider_attempted=provider_attempted,
    )
    runtime = RecordingDialogueRuntime(first_run)
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    response = bridge.reply(user_id=SYNTHETIC_USER_ID, text="cache-candidate")
    runtime.run = _run()
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="inspect-cache")

    expected = ["对方\uff1acache-candidate"] if provider_attempted else []
    if all((not_blocked, billable, provider_attempted)):
        expected.append("你\uff1aauthoritative-safe-response")
    assert response == first_run.response
    assert runtime.calls[1][0].recent_context == expected


def test_reply_without_provider_attempt_caches_neither_speaker() -> None:
    runtime = RecordingDialogueRuntime(_run(provider_attempted=False, blocked_reason=None))
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="本地处理的合成原话")
    runtime.run = _run()
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="检查前文")

    assert runtime.calls[1][0].recent_context == []


@pytest.mark.parametrize("reply", ["", " ", "\t\n"])
def test_reply_caches_only_other_speaker_when_response_is_blank(reply: str) -> None:
    runtime = RecordingDialogueRuntime(_run(text=reply))
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="  原话  ")
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="inspect")

    assert runtime.calls[1][0].recent_context == ["对方\uff1a  原话  "]


@pytest.mark.parametrize("oversized_speaker", ["other", "assistant"])
def test_reply_skips_only_item_exceeding_limit_including_label(oversized_speaker: str) -> None:
    utterance = "话" * (MAX_CONTEXT_ITEM_CHARS - len("对方\uff1a") + 1)
    reply = "答" * (MAX_CONTEXT_ITEM_CHARS - len("你\uff1a") + 1)
    if oversized_speaker == "other":
        reply = "short-reply"
        expected = [f"你\uff1a{reply}"]
    else:
        utterance = "short-utterance"
        expected = [f"对方\uff1a{utterance}"]
    runtime = RecordingDialogueRuntime(_run(text=reply))
    bridge = SpeechDialogueBridge(orchestrator=runtime)

    bridge.reply(user_id=SYNTHETIC_USER_ID, text=utterance)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="inspect")

    assert runtime.calls[1][0].recent_context == expected


@pytest.mark.parametrize(
    "ineligible_run",
    [_run(blocked_reason="synthetic-block"), _run(billable=False), _run(provider_attempted=False)],
)
def test_ineligible_reply_preserves_context_and_adds_other_speaker_only_after_provider_attempt(
    ineligible_run: DialogueRun,
) -> None:
    runtime = RecordingDialogueRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="stable")
    runtime.run = ineligible_run

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="ineligible-utterance")
    runtime.run = _run()
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="inspect")

    expected = ["对方\uff1astable", f"你\uff1a{SYNTHETIC_RESPONSE}"]
    if ineligible_run.trace.provider_attempted:
        expected.append("对方\uff1aineligible-utterance")
    assert runtime.calls[-1][0].recent_context == expected


class ToggleFailureRuntime(RecordingDialogueRuntime):
    def __init__(self, run: object) -> None:
        super().__init__(run)
        self.fail = False

    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        if self.fail:
            raise RuntimeError("synthetic-runtime-failure")
        return super().reply(
            request,
            mode,
            trusted_user_id=trusted_user_id,
        )


@pytest.mark.parametrize("failure_kind", ["invalid-input", "runtime", "invalid-run"])
def test_failed_reply_does_not_mutate_existing_context(failure_kind: str) -> None:
    runtime = ToggleFailureRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="stable-context")

    if failure_kind == "invalid-input":
        failing_user_id: object = " "
    else:
        failing_user_id = SYNTHETIC_USER_ID
        runtime.fail = failure_kind == "runtime"
        if failure_kind == "invalid-run":
            runtime.run = object()

    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(user_id=failing_user_id, text="must-not-cache")  # type: ignore[arg-type]

    runtime.fail = False
    runtime.run = _run()
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="inspect-after-failure")

    assert runtime.calls[-1][0].recent_context == [
        "对方\uff1astable-context", f"你\uff1a{SYNTHETIC_RESPONSE}"
    ]


class ReentrantRuntime:
    def __init__(self) -> None:
        self.bridge: SpeechDialogueBridge | None = None
        self.calls: list[DialogueRequest] = []
        self.inner_errors: list[SpeechDialogueBridgeError] = []
        self.reenter = True

    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        del mode, trusted_user_id
        self.calls.append(request)
        if self.reenter:
            self.reenter = False
            assert self.bridge is not None
            try:
                self.bridge.reply(user_id=SYNTHETIC_USER_ID, text="inner-reentry")
            except SpeechDialogueBridgeError as error:
                self.inner_errors.append(error)
        return _run()


def test_synchronous_reentry_fails_before_second_call_or_context_mutation() -> None:
    runtime = ReentrantRuntime()
    notices: list[None] = []
    bridge = SpeechDialogueBridge(orchestrator=runtime, budget_notice=lambda: notices.append(None))
    runtime.bridge = bridge

    bridge.reply(user_id=SYNTHETIC_USER_ID, text="outer-utterance")
    bridge.reply(user_id=SYNTHETIC_USER_ID, text="later-utterance")

    assert len(runtime.inner_errors) == 1
    assert len(notices) == 2
    assert str(runtime.inner_errors[0]) == "speech-dialogue-bridge-failed"
    assert len(runtime.calls) == 2
    assert runtime.calls[0].recent_context == []
    assert runtime.calls[1].recent_context == [
        "对方\uff1aouter-utterance", f"你\uff1a{SYNTHETIC_RESPONSE}"
    ]


def test_close_is_terminal_idempotent_and_releases_owned_state() -> None:
    cached_utterance = "close-cached-private-sentinel"
    orchestrator_secret = "close-orchestrator-private-sentinel"
    runtime = RecordingDialogueRuntime(_run())
    runtime.secret = orchestrator_secret
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    bridge.reply(user_id=SYNTHETIC_USER_ID, text=cached_utterance)
    calls_before_close = len(runtime.calls)

    bridge.close()

    assert bridge._recent_context == {}
    assert bridge._orchestrator is None
    assert cached_utterance not in repr(bridge.__dict__)
    assert orchestrator_secret not in repr(bridge.__dict__)
    with pytest.raises(SpeechDialogueBridgeError, match=r"^speech-dialogue-bridge-failed$"):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text="post-close")
    assert len(runtime.calls) == calls_before_close
    bridge.close()
    assert bridge._recent_context == {}
    assert bridge._orchestrator is None


class ClosingRuntime(RecordingDialogueRuntime):
    def __init__(self, run: DialogueRun) -> None:
        super().__init__(run)
        self.bridge: SpeechDialogueBridge | None = None

    def reply(
        self,
        request: DialogueRequest,
        mode: DialogueMode = DialogueMode.STYLED,
        *,
        trusted_user_id: str | None = None,
    ) -> DialogueRun:
        assert self.bridge is not None
        self.bridge.close()
        return super().reply(request, mode, trusted_user_id=trusted_user_id)


def test_close_during_reply_wins_without_caching_or_later_runtime_calls() -> None:
    runtime = ClosingRuntime(_run())
    bridge = SpeechDialogueBridge(orchestrator=runtime)
    runtime.bridge = bridge

    with pytest.raises(SpeechDialogueBridgeError, match=r"^speech-dialogue-bridge-failed$"):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text="must-not-cache-after-close")

    assert len(runtime.calls) == 1
    assert bridge._recent_context == {}
    assert bridge._orchestrator is None
    with pytest.raises(SpeechDialogueBridgeError):
        bridge.reply(user_id=SYNTHETIC_USER_ID, text="must-not-call-runtime")
    assert len(runtime.calls) == 1


def test_close_replaces_hostile_cache_without_interaction_or_output(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "HOSTILE_CLOSE_PRIVATE_SENTINEL"

    class HostileCache:
        def __str__(self) -> str:
            print(marker)
            raise RuntimeError(marker)

        def __repr__(self) -> str:
            print(marker, file=sys.stderr)
            raise RuntimeError(marker)

        def clear(self) -> None:
            raise RuntimeError(marker)

    bridge = SpeechDialogueBridge(orchestrator=RecordingDialogueRuntime(_run()))
    bridge._recent_context = HostileCache()  # type: ignore[assignment]

    bridge.close()

    captured = capsys.readouterr()
    assert bridge._recent_context == {}
    assert bridge._orchestrator is None
    assert marker not in caplog.text
    assert marker not in captured.out
    assert marker not in captured.err
