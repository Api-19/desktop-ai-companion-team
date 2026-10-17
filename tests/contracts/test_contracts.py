from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from companion.contracts import (
    AddresseeRegister,
    ChatRecord,
    ConsentRecord,
    DialogueRequest,
    DialogueResponse,
    MemoryEntry,
    RedactedRecord,
    RobotEvent,
    SpeakerRole,
    StyleEvidence,
    StyleGuidanceStatus,
    StyleIntent,
    StylePack,
    StyleReplyGuidance,
    StyleSnippet,
    Transcript,
    TranscriptConfidenceStatus,
    UsageDecision,
)
from companion.contracts.style import StylePreference

TIMESTAMP = datetime(2026, 7, 17, tzinfo=UTC)


def test_style_intent_contract_includes_approved_precise_runtime_values() -> None:
    expected = {
        "morning_greeting",
        "missed_meal_care",
        "unfinished_household_reassurance",
        "shopping_comparison_request",
        "work_fatigue_care",
        "positive_emotion_share",
        "negative_emotion_check_in",
        "missed_contact_affection",
    }

    assert expected <= {intent.value for intent in StyleIntent}


def sample_contracts() -> list[object]:
    chat = ChatRecord(
        conversation_id="synthetic-conversation",
        message_id="synthetic-message",
        timestamp=TIMESTAMP,
        speaker_role=SpeakerRole.OWNER,
        content_type="text",
        text="synthetic text",
    )
    return [
        chat,
        RedactedRecord(
            record=chat,
            redacted_text="synthetic redacted text",
            redaction_labels=["synthetic-label"],
        ),
        StylePack(
            version="synthetic-version",
            statistics={"average_length": 12.5},
            scene_tags=["synthetic-scene"],
            forbidden_traits=["synthetic-trait"],
            examples=["synthetic example"],
        ),
        StyleReplyGuidance(
            status=StyleGuidanceStatus.MATCHED,
            style_version="dev-style-runtime-synthetic",
            category="sleep",
            intent=StyleIntent.SLEEP_FAREWELL,
            examples=["synthetic safe long example"],
            soft_lock_expression="synthetic-safe-lock",
        ),
        StyleEvidence(
            style_version="synthetic-version",
            addressee_register=AddresseeRegister.INTIMATE,
            similar=[StyleSnippet(partner_turns=["synthetic turn"], owner_reply=["reply"])],
            general=[StyleSnippet(partner_turns=["other turn"], owner_reply=["other reply"])],
        ),
        ConsentRecord(
            user_id="synthetic-user",
            granted=True,
            granted_at=TIMESTAMP,
            allowed_fields=["preference"],
        ),
        MemoryEntry(
            memory_id="synthetic-memory",
            user_id="synthetic-user",
            field="preference",
            value="synthetic value",
            source_timestamp=TIMESTAMP,
            deleted_at=None,
        ),
        DialogueRequest(
            user_id="synthetic-user",
            text="synthetic request",
            recent_context=["synthetic context"],
        ),
        DialogueResponse(
            text="synthetic response",
            disclosed_ai_identity=True,
            billable=False,
            blocked_reason=None,
        ),
        Transcript(
            text="synthetic transcript",
            confidence=0.9,
            confidence_status=TranscriptConfidenceStatus.AVAILABLE,
        ),
        RobotEvent(state="idle", detail=None),
        UsageDecision(allowed=True, month_total_cny=Decimal("99.99"), reason=None),
    ]


@pytest.mark.parametrize("contract", sample_contracts())
def test_every_contract_round_trips_json(contract: object) -> None:
    contract_type = type(contract)
    assert contract_type.model_validate_json(contract.model_dump_json()) == contract


def test_chat_record_rejects_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        ChatRecord(
            conversation_id="synthetic-conversation",
            message_id="synthetic-message",
            timestamp=TIMESTAMP,
            speaker_role=SpeakerRole.UNKNOWN,
            content_type="text",
        )


def test_chat_record_rejects_unknown_speaker_role() -> None:
    with pytest.raises(ValidationError):
        ChatRecord(
            conversation_id="synthetic-conversation",
            message_id="synthetic-message",
            timestamp=TIMESTAMP,
            speaker_role="unmapped",
            content_type="text",
            text=None,
        )


@pytest.mark.parametrize("state", ["moving", "offline"])
def test_robot_event_rejects_unknown_state(state: str) -> None:
    with pytest.raises(ValidationError):
        RobotEvent(state=state)


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Transcript(
            text="synthetic",
            confidence=0.5,
            confidence_status=TranscriptConfidenceStatus.AVAILABLE,
            unknown=True,
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_transcript_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Transcript(
            text="synthetic",
            confidence=confidence,
            confidence_status=TranscriptConfidenceStatus.AVAILABLE,
        )


def test_transcript_serializes_available_confidence_explicitly() -> None:
    transcript = Transcript(
        text="synthetic",
        confidence=1,
        confidence_status=TranscriptConfidenceStatus.AVAILABLE,
    )

    assert transcript.model_dump(mode="json") == {
        "text": "synthetic",
        "confidence": 1.0,
        "confidence_status": "available",
    }


def test_transcript_serializes_unavailable_confidence_explicitly() -> None:
    transcript = Transcript(
        text="synthetic",
        confidence=None,
        confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
    )

    assert transcript.model_dump(mode="json") == {
        "text": "synthetic",
        "confidence": None,
        "confidence_status": "unavailable",
    }


@pytest.mark.parametrize(
    "values",
    [
        {"text": "synthetic", "confidence": 0.5},
        {"text": "synthetic", "confidence_status": "available"},
        {
            "text": "synthetic",
            "confidence": None,
            "confidence_status": "available",
        },
        {
            "text": "synthetic",
            "confidence": 0.5,
            "confidence_status": "unavailable",
        },
        {
            "text": "synthetic",
            "confidence": 0.5,
            "confidence_status": "unknown",
        },
    ],
)
def test_transcript_rejects_missing_or_inconsistent_confidence_state(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Transcript(**values)


@pytest.mark.parametrize(
    "confidence",
    [True, False, float("nan"), float("inf"), float("-inf"), "0.5", Decimal("0.5")],
)
def test_transcript_rejects_untrusted_confidence_types_and_nonfinite_values(
    confidence: object,
) -> None:
    with pytest.raises(ValidationError):
        Transcript(
            text="synthetic",
            confidence=confidence,
            confidence_status=TranscriptConfidenceStatus.AVAILABLE,
        )


def test_usage_decision_preserves_decimal_precision() -> None:
    decision = UsageDecision(allowed=True, month_total_cny=Decimal("99.99"), reason=None)
    restored = UsageDecision.model_validate_json(decision.model_dump_json())
    assert restored.month_total_cny == Decimal("99.99")


def test_legacy_style_pack_defaults_to_no_preferences() -> None:
    pack = StylePack(
        version="synthetic-v1",
        statistics={"candidate_count": 3},
        scene_tags=["sleep"],
        forbidden_traits=["do_not_repeat"],
        examples=["synthetic safe example"],
    )

    assert pack.preferences == []


def test_style_preference_round_trips() -> None:
    preference = StylePreference(
        rule_id="synthetic-rule-sleep",
        category="sleep",
        expression="synthetic-safe-short-form",
        trigger_tags=["farewell", "sleep"],
        max_consecutive_uses=1,
    )

    assert StylePreference.model_validate_json(preference.model_dump_json()) == preference


def test_style_reply_guidance_distinguishes_no_match_and_unavailable() -> None:
    no_match = StyleReplyGuidance(
        status="no_match",
        style_version="dev-style-runtime-synthetic",
    )
    unavailable = StyleReplyGuidance(status="unavailable")

    assert no_match.style_version == "dev-style-runtime-synthetic"
    assert no_match.examples == []
    assert unavailable.style_version is None
    assert unavailable.examples == []


@pytest.mark.parametrize(
    "values",
    [
        {
            "status": "matched",
            "style_version": None,
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": ["safe"],
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": [],
        },
        {"status": "no_match", "style_version": None},
        {"status": "no_match", "style_version": "v", "category": "sleep"},
        {"status": "unavailable", "style_version": "v"},
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": ["same"],
            "soft_lock_expression": "same",
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": ["duplicate", "duplicate"],
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": [""],
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": ["x" * 161],
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "soft_lock_expression": "x" * 33,
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": ["safe", "other", "third"],
        },
        {
            "status": "matched",
            "style_version": " ",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": ["safe"],
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": " ",
            "intent": "sleep_farewell",
            "examples": ["safe"],
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": ["safe"],
            "soft_lock_expression": " ",
        },
        {
            "status": "matched",
            "style_version": "v",
            "category": "sleep",
            "intent": "sleep_farewell",
            "examples": ["safe"],
            "unknown": True,
        },
    ],
)
def test_style_reply_guidance_rejects_invalid_state(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StyleReplyGuidance(**values)


@pytest.mark.parametrize(
    ("contract_type", "values"),
    [
        (
            ChatRecord,
            {
                "conversation_id": "synthetic-conversation",
                "message_id": "synthetic-message",
                "timestamp": datetime(2026, 7, 17),
                "speaker_role": SpeakerRole.UNKNOWN,
                "content_type": "text",
                "text": None,
            },
        ),
        (
            ConsentRecord,
            {
                "user_id": "synthetic-user",
                "granted": True,
                "granted_at": datetime(2026, 7, 17),
                "allowed_fields": [],
            },
        ),
        (
            MemoryEntry,
            {
                "memory_id": "synthetic-memory",
                "user_id": "synthetic-user",
                "field": "preference",
                "value": "synthetic",
                "source_timestamp": datetime(2026, 7, 17),
                "deleted_at": None,
            },
        ),
    ],
)
def test_datetime_fields_require_timezone(contract_type: type, values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        contract_type(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"partner_turns": ["字" * 61], "owner_reply": ["reply"]},
        {"partner_turns": [], "owner_reply": ["reply"]},
        {"partner_turns": ["turn"] * 4, "owner_reply": ["reply"]},
        {"partner_turns": ["turn"], "owner_reply": []},
        {"partner_turns": ["turn"], "owner_reply": ["reply"] * 4},
        {"partner_turns": ["turn"], "owner_reply": ["字" * 121]},
        {"partner_turns": [" "], "owner_reply": ["reply"]},
        {"partner_turns": ["turn"], "owner_reply": [" "]},
        {"partner_turns": ["turn"], "owner_reply": ["reply"], "unknown": True},
    ],
)
def test_style_snippet_rejects_invalid_state(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StyleSnippet(**values)


@pytest.mark.parametrize("field", ["similar", "general"])
def test_style_evidence_rejects_too_many_snippets(field: str) -> None:
    snippets = [
        StyleSnippet(partner_turns=[f"turn {index}"], owner_reply=["reply"])
        for index in range(7)
    ]
    with pytest.raises(ValidationError):
        StyleEvidence(style_version="v", addressee_register="neutral", **{field: snippets})


@pytest.mark.parametrize(
    ("similar", "general"),
    [(2, 0), (0, 2), (1, 1)],
)
def test_style_evidence_rejects_duplicate_snippets(similar: int, general: int) -> None:
    def snippet() -> StyleSnippet:
        return StyleSnippet(partner_turns=["turn"], owner_reply=["reply"])

    with pytest.raises(ValidationError):
        StyleEvidence(
            style_version="v",
            addressee_register="neutral",
            similar=[snippet() for _ in range(similar)],
            general=[snippet() for _ in range(general)],
        )


def test_style_evidence_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        StyleEvidence(style_version="v", addressee_register="neutral", unknown=True)


@pytest.mark.parametrize("register", list(AddresseeRegister))
def test_style_evidence_allows_empty_lists(register: AddresseeRegister) -> None:
    evidence = StyleEvidence(style_version="v", addressee_register=register)
    assert evidence.similar == []
    assert evidence.general == []
    assert StyleEvidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_style_snippet_strips_whitespace_and_accepts_length_limits() -> None:
    snippet = StyleSnippet(partner_turns=[" " + "字" * 60 + " "], owner_reply=["字" * 120])
    assert snippet.partner_turns == ["字" * 60]
    assert StyleSnippet.model_validate_json(snippet.model_dump_json()) == snippet
