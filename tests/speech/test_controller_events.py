import math
from collections.abc import Callable
from decimal import Decimal

import pytest

from companion.speech.controller import SpeechController
from companion.speech.controller_models import SpeechControllerIssue, SpeechState
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    DialogueFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)


def make_controller(
    *,
    recorder: AudioRecorderFake | None = None,
    recognizer: SpeechRecognizerFake | None = None,
    dialogue: DialogueFake | None = None,
    synthesizer: SpeechSynthesizerFake | None = None,
    player: AudioPlayerFake | None = None,
    mute_gate: MuteGateFake | None = None,
    clock: ClockFake | None = None,
    processing_observer: Callable[[], None] | None = None,
) -> tuple[
    SpeechController,
    AudioRecorderFake,
    SpeechRecognizerFake,
    DialogueFake,
    SpeechSynthesizerFake,
    AudioPlayerFake,
    MuteGateFake,
    ClockFake,
]:
    recorder = recorder or AudioRecorderFake()
    recognizer = recognizer or SpeechRecognizerFake()
    dialogue = dialogue or DialogueFake()
    synthesizer = synthesizer or SpeechSynthesizerFake()
    player = player or AudioPlayerFake()
    mute_gate = mute_gate or MuteGateFake()
    clock = clock or ClockFake()
    controller = SpeechController(
        trusted_user_id="trusted-owner",
        recorder=recorder,
        recognizer=recognizer,
        dialogue=dialogue,
        synthesizer=synthesizer,
        player=player,
        mute_gate=mute_gate,
        clock=clock,
        processing_observer=processing_observer,
    )
    return (
        controller,
        recorder,
        recognizer,
        dialogue,
        synthesizer,
        player,
        mute_gate,
        clock,
    )


def test_tick_before_thirty_seconds_keeps_recording() -> None:
    clock = ClockFake(now=100.0)
    controller, recorder, recognizer, _, _, _, _, _ = make_controller(clock=clock)

    controller.button_pressed()
    clock.now = 129.999
    controller.tick()

    assert controller.state is SpeechState.RECORDING
    assert recorder.calls == ["start"]
    assert recognizer.calls == 0


def test_tick_at_thirty_seconds_runs_second_button_pipeline() -> None:
    clock = ClockFake(now=100.0)
    controller, recorder, recognizer, dialogue, synthesizer, player, _, _ = make_controller(
        clock=clock
    )

    controller.button_pressed()
    clock.now = 130.0
    controller.tick()

    assert controller.state is SpeechState.PLAYING
    assert recorder.calls == ["start", "stop", "session.close", "audio.close"]
    assert recognizer.calls == 1
    assert dialogue.calls == [("trusted-owner", "synthetic transcript")]
    assert synthesizer.inputs == ["synthetic safe response"]
    assert player.tokens == [1]


def test_tick_accepts_a_non_bool_finite_numeric_clock_value() -> None:
    clock = ClockFake(now=Decimal("100.0"))
    controller, _, recognizer, _, _, _, _, _ = make_controller(clock=clock)

    controller.button_pressed()
    clock.now = Decimal("130.0")
    controller.tick()

    assert controller.state is SpeechState.PLAYING
    assert recognizer.calls == 1


@pytest.mark.parametrize(
    ("started_at", "before_timeout", "at_timeout"),
    [
        (
            Decimal("100.000000000000000000000000000"),
            Decimal("129.999999999999999999999999999"),
            Decimal("130.000000000000000000000000000"),
        ),
        (10**20, 10**20 + 29, 10**20 + 30),
    ],
)
def test_tick_preserves_exact_numeric_timeout_boundaries(
    started_at: object,
    before_timeout: object,
    at_timeout: object,
) -> None:
    clock = ClockFake(now=started_at)
    controller, recorder, recognizer, _, _, _, _, _ = make_controller(clock=clock)

    controller.button_pressed()
    clock.now = before_timeout
    controller.tick()

    assert controller.state is SpeechState.RECORDING
    assert recorder.calls == ["start"]
    assert recognizer.calls == 0

    clock.now = at_timeout
    controller.tick()

    assert controller.state is SpeechState.PLAYING
    assert recorder.calls == ["start", "stop", "session.close", "audio.close"]
    assert recognizer.calls == 1


@pytest.mark.parametrize("now", [True, False, math.nan, math.inf, -math.inf, "30"])
def test_invalid_tick_clock_value_fails_closed_without_downstream_calls(
    now: object,
) -> None:
    clock = ClockFake(now=100.0)
    controller, recorder, recognizer, dialogue, synthesizer, player, _, _ = make_controller(
        clock=clock
    )
    controller.button_pressed()
    clock.now = now

    controller.tick()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert recorder.calls == ["start", "abort", "session.close"]
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


def test_tick_clock_exception_fails_closed_without_downstream_calls() -> None:
    clock = ClockFake(now=100.0)
    controller, recorder, recognizer, dialogue, synthesizer, player, _, _ = make_controller(
        clock=clock
    )
    controller.button_pressed()
    clock.raise_on_read = RuntimeError("synthetic clock failure")

    controller.tick()

    assert controller.state is SpeechState.IDLE
    assert controller.last_issue is SpeechControllerIssue.RECORDING_FAILED
    assert recorder.calls == ["start", "abort", "session.close"]
    assert recognizer.calls == 0
    assert dialogue.calls == []
    assert synthesizer.inputs == []
    assert player.tokens == []


def test_mute_changed_aborts_only_recording() -> None:
    mute_gate = MuteGateFake(enabled=True)
    controller, recorder, recognizer, _, _, player, _, _ = make_controller(mute_gate=mute_gate)
    controller.button_pressed()
    mute_gate.enabled = False

    controller.mute_changed()

    assert controller.state is SpeechState.IDLE
    assert recorder.calls == ["start", "abort", "session.close"]
    assert recognizer.calls == 0
    assert player.stop_calls == 0


def test_mute_wins_over_timeout_and_never_recognizes() -> None:
    clock = ClockFake(now=100.0)
    mute_gate = MuteGateFake(enabled=True)
    controller, recorder, recognizer, _, _, _, _, _ = make_controller(
        clock=clock, mute_gate=mute_gate
    )
    controller.button_pressed()
    clock.now = 130.0
    mute_gate.enabled = False

    controller.tick()

    assert controller.state is SpeechState.IDLE
    assert recorder.calls == ["start", "abort", "session.close"]
    assert recognizer.calls == 0


@pytest.mark.parametrize(
    "configure_gate",
    [
        lambda gate: setattr(gate, "enabled", 1),
        lambda gate: setattr(gate, "raise_on_read", RuntimeError("synthetic mute failure")),
    ],
)
def test_invalid_or_exception_mute_read_fails_closed_while_recording(
    configure_gate: Callable[[MuteGateFake], None],
) -> None:
    mute_gate = MuteGateFake(enabled=True)
    controller, recorder, recognizer, _, _, player, _, _ = make_controller(mute_gate=mute_gate)
    controller.button_pressed()
    configure_gate(mute_gate)

    controller.mute_changed()

    assert controller.state is SpeechState.IDLE
    assert recorder.calls == ["start", "abort", "session.close"]
    assert recognizer.calls == 0
    assert player.stop_calls == 0


@pytest.mark.parametrize(
    "configure_gate",
    [
        lambda gate: setattr(gate, "enabled", False),
        lambda gate: setattr(gate, "enabled", 1),
        lambda gate: setattr(gate, "raise_on_read", RuntimeError("synthetic mute failure")),
    ],
)
def test_mute_changed_while_playing_does_not_stop_or_mutate_playback(
    configure_gate: Callable[[MuteGateFake], None],
) -> None:
    mute_gate = MuteGateFake(enabled=True)
    controller, _, _, _, synthesizer, player, _, _ = make_controller(mute_gate=mute_gate)
    controller.button_pressed()
    controller.button_pressed()
    playing_audio = synthesizer.audios[0]
    configure_gate(mute_gate)

    controller.mute_changed()

    assert controller.state is SpeechState.PLAYING
    assert player.stop_calls == 0
    assert player.tokens == [1]
    assert playing_audio.live_count == 1
    assert playing_audio.close_calls == 0


@pytest.mark.parametrize(
    "advance",
    [
        lambda controller: None,
        lambda controller: controller.button_pressed(),
        lambda controller: controller.playback_finished(1),
    ],
)
def test_mute_changed_leaves_non_recording_states_untouched(
    advance: Callable[[SpeechController], None],
) -> None:
    mute_gate = MuteGateFake(enabled=False)
    controller, recorder, _, _, _, player, _, _ = make_controller(mute_gate=mute_gate)
    advance(controller)

    before_state = controller.state
    before_recorder_calls = list(recorder.calls)
    controller.mute_changed()

    assert controller.state is before_state
    assert recorder.calls == before_recorder_calls
    assert player.stop_calls == 0


@pytest.mark.parametrize(
    ("trigger_name", "button_presses"),
    [
        ("recorder.start", 1),
        ("session.stop", 2),
        ("session.close", 2),
        ("audio.close", 2),
        ("recognizer.recognize", 2),
        ("dialogue.reply", 2),
        ("synthesizer.synthesize", 2),
        ("player.start", 2),
        ("player.stop", 3),
        ("synthesized_audio.close", 3),
    ],
)
@pytest.mark.parametrize(
    "event_name",
    ["button_pressed", "tick", "mute_changed", "playback_finished"],
)
def test_synchronous_adapter_reentry_is_a_noop(
    trigger_name: str,
    button_presses: int,
    event_name: str,
) -> None:
    recorder = AudioRecorderFake()
    recognizer = SpeechRecognizerFake()
    dialogue = DialogueFake()
    synthesizer = SpeechSynthesizerFake()
    player = AudioPlayerFake()
    controller, _, _, _, _, _, _, _ = make_controller(
        recorder=recorder,
        recognizer=recognizer,
        dialogue=dialogue,
        synthesizer=synthesizer,
        player=player,
    )

    callback_calls: list[str] = []

    def reenter() -> None:
        callback_calls.append(trigger_name)
        if event_name == "playback_finished":
            controller.playback_finished(1)
        else:
            getattr(controller, event_name)()

    adapters = {
        "recorder": recorder,
        "session": recorder,
        "audio": recorder,
        "recognizer": recognizer,
        "dialogue": dialogue,
        "synthesizer": synthesizer,
        "player": player,
        "synthesized_audio": synthesizer,
    }
    adapters[trigger_name.split(".")[0]].configure_callback(trigger_name, reenter)

    for _ in range(button_presses):
        controller.button_pressed()

    completed_flow = button_presses >= 2
    assert callback_calls == [trigger_name]
    assert recorder.calls.count("start") == (2 if button_presses == 3 else 1)
    assert recorder.calls.count("stop") == int(completed_flow)
    assert recorder.calls.count("session.close") == int(completed_flow)
    assert recorder.calls.count("audio.close") == int(completed_flow)
    assert recognizer.calls == int(completed_flow)
    assert len(dialogue.calls) == int(completed_flow)
    assert len(synthesizer.inputs) == int(completed_flow)
    assert player.tokens == ([1] if completed_flow else [])
    assert player.stop_calls == (1 if button_presses == 3 else 0)
    assert controller.state is (
        SpeechState.RECORDING if button_presses in {1, 3} else SpeechState.PLAYING
    )


def _assert_controller_ports_released(controller: SpeechController) -> None:
    assert controller._trusted_user_id is None
    assert controller._recorder is None
    assert controller._recognizer is None
    assert controller._dialogue is None
    assert controller._synthesizer is None
    assert controller._player is None
    assert controller._mute_gate is None
    assert controller._clock is None
    assert controller._processing_observer is None
    assert controller._recording_session is None
    assert controller._recording_started_at is None
    assert controller._synthesized_audio is None
    assert controller._playback_token is None


@pytest.mark.parametrize("initial_state", ["idle", "recording", "playing"])
def test_shutdown_is_terminal_idempotent_and_releases_all_ports(
    initial_state: str,
) -> None:
    controller, recorder, recognizer, dialogue, synthesizer, player, _, _ = make_controller(
        processing_observer=lambda: None
    )
    if initial_state in {"recording", "playing"}:
        controller.button_pressed()
    if initial_state == "playing":
        controller.button_pressed()

    controller.shutdown()

    assert controller.state is SpeechState.IDLE
    _assert_controller_ports_released(controller)
    if initial_state == "recording":
        assert recorder.calls == ["start", "abort", "session.close"]
        assert recorder.live_session_count == 0
    if initial_state == "playing":
        assert recorder.calls == ["start", "stop", "session.close", "audio.close"]
        assert player.stop_calls == 1
        assert synthesizer.audio_close_calls == 1
        assert synthesizer.live_audio_count == 0

    counts = (
        tuple(recorder.calls),
        recognizer.calls,
        tuple(dialogue.calls),
        tuple(synthesizer.inputs),
        player.stop_calls,
        synthesizer.audio_close_calls,
    )
    controller.button_pressed()
    controller.tick()
    controller.mute_changed()
    controller.playback_finished(1)
    controller.shutdown()
    assert counts == (
        tuple(recorder.calls),
        recognizer.calls,
        tuple(dialogue.calls),
        tuple(synthesizer.inputs),
        player.stop_calls,
        synthesizer.audio_close_calls,
    )
