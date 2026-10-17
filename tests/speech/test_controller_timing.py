from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from companion.contracts import DialogueResponse
from companion.speech import controller_models
from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechState
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    DialogueFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)


class StepClock:
    def __init__(self, step: int | float | Decimal | Fraction = Fraction(1, 100)) -> None:
        self.step = step
        self.reads = 0

    def monotonic(self) -> int | float | Decimal | Fraction:
        self.reads += 1
        return self.step * self.reads


def make_turn(*, observer: object, step: object = Fraction(1, 100)) -> SimpleNamespace:
    ports = SimpleNamespace(
        recorder=AudioRecorderFake(), recognizer=SpeechRecognizerFake(),
        dialogue=DialogueFake(), synthesizer=SpeechSynthesizerFake(),
        player=AudioPlayerFake(), mute_gate=MuteGateFake(), clock=StepClock(step),
    )
    ports.controller = SpeechController(
        trusted_user_id="synthetic-owner", **vars(ports), timing_observer=observer
    )
    return ports


def finish(ports: SimpleNamespace) -> None:
    ports.controller.button_pressed()
    ports.controller.button_pressed()


@pytest.mark.parametrize("step", [1, 0.01, Decimal("0.01"), Fraction(1, 100)])
def test_normal_turn_has_exact_stage_and_total_timings(step: object) -> None:
    rows = []
    ports = make_turn(observer=rows.append, step=step)

    finish(ports)

    assert len(rows) == 1
    ms = round(float(step) * 1000)
    assert rows[0].model_dump() == {
        "played": True, "issue": None, "stop_ms": ms, "recognize_ms": ms,
        "dialogue_ms": ms, "synthesize_ms": ms, "playback_start_ms": ms,
        "total_ms": ms * 11,
    }
    assert ports.controller.state is SpeechState.PLAYING


@pytest.mark.parametrize(
    ("failure", "issue", "stages", "played"),
    [
        ("stop", "recording-failed", (), False),
        ("session-close", "cleanup-failed", (), False),
        ("recognize", "recognition-failed", ("recognize", "synthesize", "playback_start"), True),
        ("audio-close", "cleanup-failed", ("recognize",), False),
        ("dialogue", "dialogue-failed",
         ("recognize", "dialogue", "synthesize", "playback_start"), True),
        ("synthesize", "synthesis-failed", ("recognize", "dialogue", "synthesize"), False),
        ("playback", "playback-failed",
         ("recognize", "dialogue", "synthesize", "playback_start"), False),
    ],
)
def test_failed_stages_are_timed_and_skipped_stages_are_none(
    failure: str, issue: str, stages: tuple[str, ...], played: bool
) -> None:
    rows = []
    ports = make_turn(observer=rows.append)
    error = RuntimeError("synthetic failure")
    if failure in {"stop", "session-close", "audio-close"}:
        method = {
            "stop": "session.stop", "session-close": "session.close", "audio-close": "audio.close"
        }
        ports.recorder.configure_failure(method[failure], error)
    elif failure == "recognize":
        ports.recognizer.failure = error
    elif failure == "dialogue":
        ports.dialogue.failure = error
    elif failure == "synthesize":
        ports.synthesizer.failure = error
    else:
        ports.player.start_failure = error

    finish(ports)

    assert len(rows) == 1
    row = rows[0].model_dump()
    assert row["played"] is played
    assert row["issue"] == issue
    assert row["stop_ms"] == 10
    for stage in ("recognize", "dialogue", "synthesize", "playback_start"):
        assert row[f"{stage}_ms"] == (10 if stage in stages else None)
    assert row["total_ms"] == (3 + 2 * len(stages)) * 10


def test_empty_spoken_text_times_local_notice_synthesis_and_playback() -> None:
    rows = []
    ports = make_turn(observer=rows.append)
    ports.dialogue.result = DialogueResponse(
        text="[合成动作]", disclosed_ai_identity=False, billable=False, blocked_reason=None
    )

    finish(ports)

    assert len(rows) == 1
    assert rows[0].synthesize_ms == 10
    assert rows[0].playback_start_ms == 10
    assert rows[0].played is True
    assert rows[0].total_ms == 110
    assert ports.controller.state is SpeechState.PLAYING


def test_observer_exception_preserves_state_and_resource_behavior() -> None:
    observer = Mock(side_effect=RuntimeError("synthetic observer failure"))
    observed = make_turn(observer=observer)
    baseline = make_turn(observer=None)

    finish(observed)
    finish(baseline)

    observer.assert_called_once()
    assert observed.controller.state is baseline.controller.state
    assert observed.controller.last_issue is baseline.controller.last_issue
    assert observed.recorder.calls == baseline.recorder.calls
    assert observed.player.tokens == baseline.player.tokens
    assert baseline.clock.reads == 1


def test_no_session_does_not_emit_and_each_later_turn_emits_once() -> None:
    rows = []
    ports = make_turn(observer=rows.append)
    ports.controller._finish_recording()
    assert rows == []
    finish(ports)
    ports.controller.playback_finished(ports.player.tokens[-1])
    finish(ports)
    assert len(rows) == 2
    assert rows[0] == rows[1]
    ports.controller.shutdown()
    assert len(rows) == 2


def test_turn_timing_is_frozen_and_forbids_extra_fields() -> None:
    data = dict(played=False, issue=None, stop_ms=0, recognize_ms=None, dialogue_ms=None,
                synthesize_ms=None, playback_start_ms=None, total_ms=0)
    model = controller_models.TurnTiming(**data)
    with pytest.raises(ValidationError):
        model.played = True
    with pytest.raises(ValidationError):
        controller_models.TurnTiming(**data, text="synthetic private text")
    for field in ("stop_ms", "recognize_ms", "dialogue_ms", "synthesize_ms",
                  "playback_start_ms", "total_ms"):
        with pytest.raises(ValidationError):
            controller_models.TurnTiming(**{**data, field: -1})


def test_submillisecond_timings_use_round() -> None:
    rows = []
    ports = make_turn(observer=rows.append, step=Fraction(3, 2000))
    finish(ports)
    assert rows[0].stop_ms == round(float(Fraction(3, 2000)) * 1000)
    assert rows[0].total_ms == round(float(Fraction(33, 2000)) * 1000)
