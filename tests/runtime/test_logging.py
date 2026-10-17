import io
import json
from datetime import UTC, datetime

import pytest

from companion.contracts import (
    ChatRecord,
    DialogueRequest,
    DialogueResponse,
    MemoryEntry,
    SpeakerRole,
    StylePack,
    Transcript,
    TranscriptConfidenceStatus,
)
from companion.runtime.logging import configure_logging, sanitize_fields
from companion.speech.controller import SpeechController
from companion.speech.controller_models import (
    CONNECTION_RETRY_PROMPT,
    RECOGNITION_RETRY_PROMPT,
    RecognitionInput,
)
from companion.speech.recognition_gate import AudioQualityEvidence
from tests.speech.controller_fakes import (
    AudioPlayerFake,
    AudioRecorderFake,
    ClockFake,
    DialogueFake,
    MuteGateFake,
    SpeechRecognizerFake,
    SpeechSynthesizerFake,
)


def test_sanitize_fields_recursively_redacts_private_values() -> None:
    fields = {
        "request": {
            "chat-text": "synthetic private sentence",
            "items": [{"memory_value": "synthetic memory"}],
        },
        "count": 2,
    }
    assert sanitize_fields(fields) == {
        "request": {
            "chat-text": "[REDACTED]",
            "items": [{"memory_value": "[REDACTED]"}],
        },
        "count": 2,
    }


def test_safe_logger_emits_single_json_line_without_secret() -> None:
    stream = io.StringIO()
    logger = configure_logging("INFO", stream)
    logger.event("synthetic.event", api_key="sk-" + "synthetic-value-123456", status="ok")
    output = stream.getvalue()
    payload = json.loads(output)
    assert payload["event"] == "synthetic.event"
    assert payload["fields"]["api_key"] == "[REDACTED]"
    assert "synthetic-value" not in output
    assert output.count("\n") == 1


def test_contract_model_dumps_never_emit_sensitive_content() -> None:
    timestamp = datetime(2026, 7, 17, tzinfo=UTC)
    sensitive_values = {
        "chat body synthetic secret",
        "request body synthetic secret",
        "recent context synthetic secret",
        "response body synthetic secret",
        "transcript synthetic secret",
        "memory synthetic secret",
        "style example synthetic secret",
    }
    fields = {
        "record": ChatRecord(
            conversation_id="synthetic-conversation",
            message_id="synthetic-message",
            timestamp=timestamp,
            speaker_role=SpeakerRole.OWNER,
            content_type="text",
            text="chat body synthetic secret",
        ).model_dump(mode="json"),
        "request": DialogueRequest(
            user_id="synthetic-user",
            text="request body synthetic secret",
            recent_context=["recent context synthetic secret"],
        ).model_dump(mode="json"),
        "reply": DialogueResponse(
            text="response body synthetic secret",
            disclosed_ai_identity=True,
            billable=False,
            blocked_reason=None,
        ).model_dump(mode="json"),
        "recognition": Transcript(
            text="transcript synthetic secret",
            confidence=0.9,
            confidence_status=TranscriptConfidenceStatus.AVAILABLE,
        ).model_dump(mode="json"),
        "entry": MemoryEntry(
            memory_id="synthetic-memory",
            user_id="synthetic-user",
            field="preference",
            value="memory synthetic secret",
            source_timestamp=timestamp,
            deleted_at=None,
        ).model_dump(mode="json"),
        "style": StylePack(
            version="synthetic-version",
            statistics={},
            scene_tags=[],
            forbidden_traits=[],
            examples=["style example synthetic secret"],
        ).model_dump(mode="json"),
    }
    output = json.dumps(sanitize_fields(fields), ensure_ascii=False)
    assert all(value not in output for value in sensitive_values)


def test_unavailable_transcript_dump_keeps_nested_text_redacted() -> None:
    transcript = Transcript(
        text="unavailable transcript synthetic secret",
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
    )

    output = json.dumps(
        sanitize_fields({"recognition": transcript.model_dump(mode="json")}),
        ensure_ascii=False,
    )

    assert "unavailable transcript synthetic secret" not in output
    assert '"text": "[REDACTED]"' in output
    assert '"confidence": null' in output
    assert '"confidence_status": "unavailable"' in output


def test_secret_pattern_is_redacted_under_ordinary_key() -> None:
    secret = "sk-" + "synthetic-value-1234567890"
    assert sanitize_fields({"detail": secret}) == {"detail": "[REDACTED]"}


def test_unsupported_object_never_calls_repr() -> None:
    class ExplosiveRepr:
        def __repr__(self) -> str:
            raise AssertionError("repr must not be called")

    assert sanitize_fields({"object": ExplosiveRepr()}) == {"object": "<ExplosiveRepr>"}


def test_reconfiguring_logger_does_not_duplicate_output() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream)
    logger = configure_logging("INFO", stream)
    logger.event("synthetic.event", status="ok")
    assert stream.getvalue().count("\n") == 1


def test_speech_controller_logs_only_state_and_issue_aggregates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript_text = "unique synthetic transcript privacy sentinel"
    dialogue_text = "unique synthetic Dialogue privacy sentinel"
    audio_path = "synthetic-private/audio-capture.wav"
    model_path = "synthetic-private/model-file.onnx"
    input_device = "Synthetic Private Microphone Device"
    output_device = "Synthetic Private Speaker Device"
    failure_detail = f"{audio_path}|{model_path}|{input_device}|{output_device}"
    sensitive_values = {
        transcript_text,
        dialogue_text,
        RECOGNITION_RETRY_PROMPT,
        CONNECTION_RETRY_PROMPT,
        audio_path,
        model_path,
        input_device,
        output_device,
        failure_detail,
    }

    accepted_input = RecognitionInput(
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
        text=transcript_text,
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
        confidence_threshold=None,
    )
    rejected_input = RecognitionInput(
        audio=accepted_input.audio,
        vad_passed=False,
        text=transcript_text,
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
        confidence_threshold=None,
    )

    fixtures = [
        (
            AudioRecorderFake(),
            SpeechRecognizerFake(result=accepted_input),
            DialogueFake(
                result=DialogueResponse(
                    text=dialogue_text,
                    disclosed_ai_identity=False,
                    billable=False,
                    blocked_reason=None,
                )
            ),
            SpeechSynthesizerFake(),
            AudioPlayerFake(),
        ),
        (
            AudioRecorderFake(),
            SpeechRecognizerFake(result=rejected_input),
            DialogueFake(),
            SpeechSynthesizerFake(),
            AudioPlayerFake(),
        ),
        (
            AudioRecorderFake(),
            SpeechRecognizerFake(result=accepted_input),
            DialogueFake(failure=RuntimeError(failure_detail)),
            SpeechSynthesizerFake(),
            AudioPlayerFake(),
        ),
        (
            AudioRecorderFake(),
            SpeechRecognizerFake(result=accepted_input),
            DialogueFake(),
            SpeechSynthesizerFake(),
            AudioPlayerFake(),
        ),
    ]
    fixtures[-1][0].configure_failure("audio.close", RuntimeError(failure_detail))

    controllers: list[SpeechController] = []
    resource_reprs: list[str] = []
    escaped_exceptions: list[Exception] = []
    aggregates: list[dict[str, object]] = []
    for recorder, recognizer, dialogue, synthesizer, player in fixtures:
        recorder.private_audio_path = audio_path
        recognizer.private_model_path = model_path
        recorder.private_device_name = input_device
        player.private_device_name = output_device
        controller = SpeechController(
            trusted_user_id="trusted-owner",
            recorder=recorder,
            recognizer=recognizer,
            dialogue=dialogue,
            synthesizer=synthesizer,
            player=player,
            mute_gate=MuteGateFake(),
            clock=ClockFake(),
        )
        controllers.append(controller)
        try:
            controller.button_pressed()
            controller.button_pressed()
            if controller.state.value == "playing":
                controller.playback_finished(player.tokens[0])
        except Exception as exc:  # pragma: no cover - assertion target
            escaped_exceptions.append(exc)

        aggregates.append(
            sanitize_fields(
                {
                    "state": controller.state.value,
                    "issue": (
                        controller.last_issue.value if controller.last_issue is not None else None
                    ),
                }
            )
        )
        resource_reprs.extend(
            repr(resource)
            for resource in (
                recorder,
                recognizer,
                dialogue,
                synthesizer,
                player,
                *recorder.sessions,
                *recorder.recorded_audios,
                *synthesizer.audios,
            )
        )

    stream = io.StringIO()
    logger = configure_logging("INFO", stream)
    for aggregate in aggregates:
        logger.event("speech.controller.aggregate", **aggregate)

    captured = capsys.readouterr()
    observable_output = "\n".join(
        [
            stream.getvalue(),
            captured.out,
            captured.err,
            *(repr(controller) for controller in controllers),
            *resource_reprs,
            *(str(exc) for exc in escaped_exceptions),
        ]
    )
    assert escaped_exceptions == []
    assert all(value not in observable_output for value in sensitive_values)
