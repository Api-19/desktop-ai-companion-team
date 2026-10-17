import pytest

from companion.contracts import DialogueResponse
from companion.speech.controller import SpeechController
from companion.speech.controller_models import EMPTY_REPLY_PROMPT, SpeechState
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
    *, trusted_user_id: str = "trusted-owner"
) -> tuple[
    SpeechController,
    AudioRecorderFake,
    DialogueFake,
    SpeechSynthesizerFake,
    AudioPlayerFake,
]:
    recorder = AudioRecorderFake()
    dialogue = DialogueFake()
    synthesizer = SpeechSynthesizerFake()
    player = AudioPlayerFake()
    controller = SpeechController(
        trusted_user_id=trusted_user_id,
        recorder=recorder,
        recognizer=SpeechRecognizerFake(),
        dialogue=dialogue,
        synthesizer=synthesizer,
        player=player,
        mute_gate=MuteGateFake(),
        clock=ClockFake(),
    )
    return controller, recorder, dialogue, synthesizer, player


@pytest.mark.parametrize("trusted_user_id", ["x", "x" * 120])
def test_constructor_accepts_exact_nonblank_trusted_user_id(
    trusted_user_id: str,
) -> None:
    controller, recorder, _, _, _ = make_controller(trusted_user_id=trusted_user_id)

    assert controller.state is SpeechState.IDLE
    assert recorder.calls == []


@pytest.mark.parametrize("trusted_user_id", ["", "   ", "x" * 121])
def test_constructor_rejects_invalid_builtin_trusted_user_id_before_port_calls(
    trusted_user_id: str,
) -> None:
    recorder = AudioRecorderFake()
    with pytest.raises(ValueError, match="trusted_user_id is invalid"):
        SpeechController(
            trusted_user_id=trusted_user_id,
            recorder=recorder,
            recognizer=SpeechRecognizerFake(),
            dialogue=DialogueFake(),
            synthesizer=SpeechSynthesizerFake(),
            player=AudioPlayerFake(),
            mute_gate=MuteGateFake(),
            clock=ClockFake(),
        )

    assert recorder.calls == []


def test_constructor_rejects_trusted_user_id_subclass_before_port_calls() -> None:
    class UserIdSubclass(str):
        pass

    recorder = AudioRecorderFake()
    with pytest.raises(TypeError, match="trusted_user_id must be a built-in str"):
        SpeechController(
            trusted_user_id=UserIdSubclass("trusted-owner"),
            recorder=recorder,
            recognizer=SpeechRecognizerFake(),
            dialogue=DialogueFake(),
            synthesizer=SpeechSynthesizerFake(),
            player=AudioPlayerFake(),
            mute_gate=MuteGateFake(),
            clock=ClockFake(),
        )

    assert recorder.calls == []


def test_second_press_runs_accepted_unavailable_flow_and_releases_recording_audio() -> None:
    controller, recorder, dialogue, synthesizer, player = make_controller()

    controller.button_pressed()

    assert controller.state is SpeechState.RECORDING

    controller.button_pressed()

    recorded_audio = recorder.recorded_audios[0]
    assert controller.state is SpeechState.PLAYING
    assert recorder.calls == ["start", "stop", "session.close", "audio.close"]
    assert dialogue.calls == [("trusted-owner", "synthetic transcript")]
    assert synthesizer.inputs == ["synthetic safe response"]
    assert player.tokens == [1]
    assert recorded_audio.live_count == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("可以可以。\n饿着上桌啊。[憨笑]😋👍🏻", "可以可以。\n饿着上桌啊。"),
        ("[猪头][猪头]😋👍🏻", EMPTY_REPLY_PROMPT),
        ("。？", EMPTY_REPLY_PROMPT),  # noqa: RUF001
    ],
)
def test_reply_is_cleaned_only_for_synthesis(text: str, expected: str) -> None:
    controller, recorder, dialogue, synthesizer, player = make_controller()
    response = DialogueResponse(
        text=text,
        disclosed_ai_identity=False,
        billable=False,
        blocked_reason=None,
    )
    dialogue.result = response

    controller.button_pressed()
    controller.button_pressed()

    assert response.text == text
    assert controller.last_issue is None
    assert recorder.live_recorded_audio_count == 0
    assert synthesizer.inputs == [expected]
    assert player.tokens == [1]
    assert controller.state is SpeechState.PLAYING

    controller.playback_finished(1)

    assert controller.state is SpeechState.IDLE
    assert synthesizer.audios[0].close_calls == 1
    assert synthesizer.audios[0].live_count == 0
    controller.button_pressed()
    assert controller.state is SpeechState.RECORDING


@pytest.mark.parametrize("text", ["😋👍🏻", "[猪头][猪头]", "。？"])  # noqa: RUF001
def test_empty_spoken_reply_plays_local_notice_once(text: str) -> None:
    controller, _, dialogue, synthesizer, player = make_controller()
    response = DialogueResponse(
        text=text, disclosed_ai_identity=False, billable=False, blocked_reason=None
    )
    dialogue.result = response

    controller.button_pressed()
    controller.button_pressed()

    assert synthesizer.inputs == ["这次没有可朗读的回复，换个说法再试吧。"]  # noqa: RUF001
    assert player.tokens == [1]
    assert len(dialogue.calls) == 1
    assert response.text == text
    assert controller.state is SpeechState.PLAYING
    assert controller.last_issue is None

    controller.playback_finished(1)
    controller.playback_finished(1)

    assert controller.state is SpeechState.IDLE
    assert synthesizer.audios[0].close_calls == 1
    assert synthesizer.audios[0].live_count == 0
    assert len(synthesizer.inputs) == len(player.tokens) == 1


def test_repeated_empty_replies_each_play_one_notice() -> None:
    controller, _, dialogue, synthesizer, player = make_controller()
    texts = ["😋👍🏻", "[猪头][猪头]", "。？"]  # noqa: RUF001
    for turn, text in enumerate(texts, start=1):
        response = DialogueResponse(
            text=text, disclosed_ai_identity=False, billable=False, blocked_reason=None
        )
        dialogue.result = response

        controller.button_pressed()
        controller.button_pressed()

        assert dialogue.calls == [("trusted-owner", "synthetic transcript")] * turn
        assert synthesizer.inputs == [EMPTY_REPLY_PROMPT] * turn
        assert player.tokens == list(range(1, turn + 1))
        assert response.text == text
        assert controller.last_issue is None
        assert controller.state is SpeechState.PLAYING

        controller.playback_finished(turn)

        assert controller.state is SpeechState.IDLE
        assert synthesizer.audios[-1].close_calls == 1
        assert synthesizer.live_audio_count == 0


def test_stale_duplicate_and_invalid_playback_tokens_cannot_close_new_audio() -> None:
    controller, _, _, synthesizer, player = make_controller()

    controller.button_pressed()
    controller.button_pressed()
    old_audio = synthesizer.audios[0]

    controller.button_pressed()
    controller.button_pressed()
    current_audio = synthesizer.audios[1]

    controller.playback_finished(1)

    assert player.tokens == [1, 2]
    assert player.stop_calls == 1
    assert old_audio.live_count == 0
    assert current_audio.live_count == 1
    assert controller.state is SpeechState.PLAYING

    class IntSubclass(int):
        pass

    for token in (True, IntSubclass(2), 3):
        controller.playback_finished(token)

    assert current_audio.live_count == 1
    assert controller.state is SpeechState.PLAYING

    controller.playback_finished(2)

    assert current_audio.live_count == 0
    assert current_audio.close_calls == 1
    assert controller.state is SpeechState.IDLE

    controller.playback_finished(2)
    controller.playback_finished(1)
    assert current_audio.close_calls == 1

    controller.button_pressed()
    controller.playback_finished(2)
    assert controller.state is SpeechState.RECORDING
