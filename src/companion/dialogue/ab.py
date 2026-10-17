from __future__ import annotations

import hashlib
import json
import os
import secrets
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter

from companion.contracts import DialogueRequest, DialogueResponse
from companion.dialogue.composition import build_real_orchestrator as build_real_orchestrator
from companion.dialogue.models import DialogueMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from companion.dialogue.orchestrator import DialogueOrchestrator


class ABStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_MATCHED = "not_matched"
    BUDGET_EXCEEDED = "budget_exceeded"


class ABScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"[a-z][a-z0-9_-]{0,39}")
    text: str = Field(min_length=1, max_length=2_000)
    recent_context: tuple[str, ...] = Field(max_length=4)
    idle_seconds: int = Field(ge=0)


class ABPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    status: ABStatus
    answers: dict[Literal["A", "B"], DialogueResponse] = Field(default_factory=dict)
    answer_key: dict[Literal["A", "B"], DialogueMode] = Field(default_factory=dict)
    request_attempt_count: StrictInt | None = Field(default=None, ge=0, le=2)


class SmokeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "failed", "blocked"]
    succeeded: bool
    request_attempt_count: int = Field(ge=0, le=1)


class ABSuiteSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    scenario_count: int = Field(ge=0)
    request_attempt_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    paired_count: int = Field(ge=0)


_SCENARIO_ADAPTER = TypeAdapter(tuple[ABScenario, ...])
DIALOGUE_EVALUATION_USER_ID = "synthetic-dialogue-evaluation-user"
SMOKE_REQUEST = DialogueRequest(
    user_id=DIALOGUE_EVALUATION_USER_ID,
    text="最近读了一本书，想随便聊聊",  # noqa: RUF001
    recent_context=[],
)
STYLE_REFRESH_SCENARIO_IDS = (
    "meals-2",
    "sleep-1",
    "sleep-2",
    "weather-2",
    "household-1",
    "shopping-2",
    "gentle_affection-2",
)
FRAMEWORK20_SCENARIO_IDS = (
    "greeting-1",
    "greeting-2",
    "courtesy-1",
    "courtesy-2",
    "meals-1",
    "meals-2",
    "sleep-1",
    "sleep-2",
    "weather-1",
    "weather-2",
    "household-1",
    "household-2",
    "shopping-1",
    "shopping-2",
    "work_study-1",
    "work_study-2",
    "general_emotion-1",
    "general_emotion-2",
    "gentle_affection-1",
    "gentle_affection-2",
)


def load_ab_scenarios(path: Path) -> tuple[ABScenario, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenarios = _SCENARIO_ADAPTER.validate_python(payload)
        if not scenarios or len({item.scenario_id for item in scenarios}) != len(scenarios):
            raise ValueError
        return scenarios
    except Exception:
        raise ValueError("invalid A/B scenarios") from None


def validate_framework20_scenarios(scenarios: Sequence[ABScenario]) -> tuple[ABScenario, ...]:
    try:
        validated = _SCENARIO_ADAPTER.validate_python(tuple(scenarios))
        if tuple(item.scenario_id for item in validated) != FRAMEWORK20_SCENARIO_IDS:
            raise ValueError
        return validated
    except Exception:
        raise ValueError("invalid framework20 scenarios") from None


def load_framework20_scenarios(path: Path) -> tuple[ABScenario, ...]:
    try:
        return validate_framework20_scenarios(load_ab_scenarios(path))
    except Exception:
        raise ValueError("invalid framework20 scenarios") from None


def load_style_refresh_scenarios(
    full_source: Path,
    refresh_source: Path,
) -> tuple[ABScenario, ...]:
    try:
        full = load_ab_scenarios(full_source)
        refresh = load_ab_scenarios(refresh_source)
        expected = tuple(
            scenario for scenario in full if scenario.scenario_id in STYLE_REFRESH_SCENARIO_IDS
        )
        if (
            tuple(scenario.scenario_id for scenario in refresh) != STYLE_REFRESH_SCENARIO_IDS
            or len(refresh) != 7
            or refresh != expected
        ):
            raise ValueError
        return refresh
    except Exception:
        raise ValueError("invalid style refresh scenarios") from None


def run_baseline_smoke(
    orchestrator: DialogueOrchestrator,
    results_dir: Path,
    *,
    trusted_user_id: str = DIALOGUE_EVALUATION_USER_ID,
) -> SmokeSummary:
    trusted_results_dir = orchestrator.resolve_test_results_dir(results_dir)
    if trusted_results_dir is None:
        orchestrator = None  # type: ignore[assignment]
        results_dir = None  # type: ignore[assignment]
        _raise_invalid_results_dir()
    request = DialogueRequest(
        user_id=trusted_user_id,
        text=SMOKE_REQUEST.text,
        recent_context=[],
    )
    run = orchestrator.reply(request, DialogueMode.BASELINE, trusted_user_id=trusted_user_id)
    attempted = int(run.trace.provider_attempted)
    if run.response.blocked_reason is None:
        run_dir = _make_private_run_dir(trusted_results_dir, "dialogue-smoke")
        _write_private_json(
            run_dir / "answer.json",
            run.response.model_dump(mode="json"),
        )
        return SmokeSummary(
            status="complete",
            succeeded=True,
            request_attempt_count=attempted,
        )
    return SmokeSummary(
        status="failed" if attempted else "blocked",
        succeeded=False,
        request_attempt_count=attempted,
    )


def run_ab_suite(
    orchestrator: DialogueOrchestrator,
    scenarios: Sequence[ABScenario],
    results_dir: Path,
    *,
    trusted_user_id: str = DIALOGUE_EVALUATION_USER_ID,
) -> ABSuiteSummary:
    trusted_results_dir = orchestrator.resolve_test_results_dir(results_dir)
    if trusted_results_dir is None:
        orchestrator = None  # type: ignore[assignment]
        scenarios = None  # type: ignore[assignment]
        results_dir = None  # type: ignore[assignment]
        _raise_invalid_results_dir()
    validated = _SCENARIO_ADAPTER.validate_python(tuple(scenarios))
    run_dir = _make_private_run_dir(trusted_results_dir, "dialogue-ab")
    run_id = run_dir.name

    answer_records: list[dict[str, object]] = []
    key_records: list[dict[str, object]] = []
    succeeded_count = 0
    failed_count = 0
    blocked_count = 0
    paired_count = 0
    request_attempt_count = 0

    for scenario in validated:
        pair = orchestrator.reply_ab(
            scenario.scenario_id,
            DialogueRequest(
                user_id=trusted_user_id,
                text=scenario.text,
                recent_context=list(scenario.recent_context),
            ),
            evaluation_idle_seconds=scenario.idle_seconds,
            trusted_user_id=trusted_user_id,
        )
        if pair.request_attempt_count is not None:
            request_attempt_count += pair.request_attempt_count
        elif pair.answer_key:
            request_attempt_count += 2
        if pair.status is ABStatus.COMPLETE:
            succeeded_count += 1
            paired_count += 1
        elif pair.status in {ABStatus.PARTIAL, ABStatus.FAILED}:
            failed_count += 1
        else:
            blocked_count += 1
        answer_records.append(
            {
                "scenario_id": pair.scenario_id,
                "status": pair.status.value,
                "answers": {
                    label: answer.model_dump(mode="json") for label, answer in pair.answers.items()
                },
            }
        )
        key_records.append(
            {
                "scenario_id": pair.scenario_id,
                "answer_key": {label: mode.value for label, mode in pair.answer_key.items()},
            }
        )

    _write_private_json(run_dir / "answers.json", answer_records)
    _write_private_json(run_dir / "answer-key.json", key_records)
    return ABSuiteSummary(
        run_id=run_id,
        scenario_count=len(validated),
        request_attempt_count=request_attempt_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        blocked_count=blocked_count,
        paired_count=paired_count,
    )


def _blind_order(scenario_id: str) -> tuple[DialogueMode, DialogueMode]:
    digest = hashlib.sha256(f"{scenario_id}:dialogue-ab-v1".encode()).digest()
    if digest[0] % 2 == 0:
        return DialogueMode.BASELINE, DialogueMode.STYLED
    return DialogueMode.STYLED, DialogueMode.BASELINE


def _make_private_run_dir(results_dir: Path, prefix: str) -> Path:
    root = results_dir.resolve()
    run_dir = root / f"{prefix}-{secrets.token_hex(12)}"
    if run_dir.parent != root:
        raise ValueError("invalid dialogue results directory")
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    return run_dir


def _write_private_json(path: Path, payload: object) -> None:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(content)


def _raise_invalid_results_dir() -> None:
    raise ValueError("invalid dialogue results directory") from None
