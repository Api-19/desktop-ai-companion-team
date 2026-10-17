"""Explicit privacy revision aggregation using in-memory sources."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from dialogue_support import FIXED_NOW, FakeProvider, FakeStyleProvider, synthetic_dialogue_config

from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.privacy_revision import (
    PRIVACY_REVISION_FAILURE,
    PrivacyRevisionUnavailable,
)
from companion.dialogue.usage import MonthlyUsageLedger


class FakeRevisionSource:
    def __init__(
        self, name: str, revision: int | str, calls: list[tuple[str, str]]
    ) -> None:
        self.name = name
        self.revision = revision
        self.calls = calls
        self.error: BaseException | None = None

    def content_revision(self, user_id: str) -> int | str:
        self.calls.append((self.name, user_id))
        if self.error is not None:
            raise self.error
        return self.revision


def build(
    root: Path, sources: Sequence[FakeRevisionSource] | None = None
) -> DialogueOrchestrator:
    config = synthetic_dialogue_config(root)
    kwargs = {} if sources is None else {"privacy_revision_sources": sources}
    return DialogueOrchestrator(
        config=config,
        style_adapter=FakeStyleProvider(),
        provider=FakeProvider(),
        ledger=MonthlyUsageLedger(root / "synthetic-ledger", config.budget),
        clock=lambda: FIXED_NOW,
        **kwargs,
    )


def test_privacy_revision_aggregates_sources_in_order(tmp_path: Path) -> None:
    calls = []
    memory = FakeRevisionSource("memory", 7, calls)
    snippets = FakeRevisionSource("snippets", "synthetic-digest", calls)
    sources = [memory, snippets]
    runtime = build(tmp_path, sources)
    assert calls == []
    sources.clear()

    assert runtime.privacy_revision("synthetic-user") == (7, "synthetic-digest")
    assert calls == [("memory", "synthetic-user"), ("snippets", "synthetic-user")]
    memory.revision = 8
    snippets.revision = "synthetic-next-digest"
    assert runtime.privacy_revision("synthetic-other") == (8, "synthetic-next-digest")
    assert calls[2:] == [("memory", "synthetic-other"), ("snippets", "synthetic-other")]


def test_privacy_revision_empty_sources_is_stable(tmp_path: Path) -> None:
    runtime = build(tmp_path)
    assert runtime.privacy_revision("synthetic-user") == ()
    assert runtime.privacy_revision("synthetic-user") == ()
    assert runtime.privacy_revision("synthetic-other") == ()


@pytest.mark.parametrize("previous_success", [False, True])
def test_privacy_revision_failure_does_not_return_partial_or_cached_value(
    tmp_path: Path, previous_success: bool
) -> None:
    calls = []
    memory = FakeRevisionSource("memory", 7, calls)
    snippets = FakeRevisionSource("snippets", "synthetic-digest", calls)
    runtime = build(tmp_path, [memory, snippets])

    if previous_success:
        assert runtime.privacy_revision("synthetic-user") == (7, "synthetic-digest")
    calls.clear()
    memory.revision = 8
    snippets.error = ValueError("synthetic source failure")

    with pytest.raises(PrivacyRevisionUnavailable) as raised:
        runtime.privacy_revision("synthetic-user")

    assert isinstance(raised.value, RuntimeError)
    assert str(raised.value) == PRIVACY_REVISION_FAILURE == "privacy revision unavailable"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert calls == [("memory", "synthetic-user"), ("snippets", "synthetic-user")]


def test_privacy_revision_recovery_reads_all_sources_again(tmp_path: Path) -> None:
    calls = []
    first = FakeRevisionSource("first", 1, calls)
    second = FakeRevisionSource("second", 2, calls)
    third = FakeRevisionSource("third", "synthetic-digest", calls)
    second.error = OSError("synthetic source failure")
    runtime = build(tmp_path, [first, second, third])

    with pytest.raises(PrivacyRevisionUnavailable):
        runtime.privacy_revision("synthetic-user")
    assert calls == [("first", "synthetic-user"), ("second", "synthetic-user")]
    calls.clear()
    first.revision = 3
    second.revision = 4
    second.error = None

    assert runtime.privacy_revision("synthetic-user") == (3, 4, "synthetic-digest")
    assert calls == [
        ("first", "synthetic-user"),
        ("second", "synthetic-user"),
        ("third", "synthetic-user"),
    ]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit()])
def test_privacy_revision_does_not_catch_base_exception(
    tmp_path: Path, error: BaseException
) -> None:
    source = FakeRevisionSource("memory", 1, [])
    source.error = error
    runtime = build(tmp_path, [source])
    with pytest.raises(type(error)) as raised:
        runtime.privacy_revision("synthetic-user")
    assert raised.value is error
