from datetime import UTC, datetime, timedelta

import pytest

from companion.contracts import ChatRecord, SpeakerRole
from companion.privacy import (
    PrivacyValidationError,
    ReviewDecision,
    ReviewDecisionValue,
    RiskCategory,
    finalize_privacy_batch,
    scan_for_review,
)

BASE_TIME = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
PROCESSED_AT = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def privacy_traceback_retains(error: BaseException, marker: str) -> bool:
    current = error.__traceback__
    while current is not None:
        filename = current.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename and marker in repr(current.tb_frame.f_locals):
            return True
        current = current.tb_next
    return False


def record(
    message_id: str,
    text: str | None,
    *,
    minute: int = 0,
    role: SpeakerRole = SpeakerRole.OWNER,
) -> ChatRecord:
    return ChatRecord(
        conversation_id="synthetic-conversation",
        message_id=message_id,
        timestamp=BASE_TIME + timedelta(minutes=minute),
        speaker_role=role,
        content_type="text",
        text=text,
    )


def mixed_records() -> list[ChatRecord]:
    return [
        record("m1", "普通聊天", role=SpeakerRole.OTHER),
        record("m2", "联系 synthetic.owner@example.test", minute=1),
        record("m3", "银行卡密码别发在这里", minute=60, role=SpeakerRole.OTHER),
        record("m4", "我知道了", minute=61),
    ]


def decision(segment_id: str, value: ReviewDecisionValue) -> ReviewDecision:
    return ReviewDecision(segment_id=segment_id, decision=value)


def test_unflagged_segments_need_no_decision_and_emit_owner_only_output() -> None:
    records = [record("m1", "普通聊天", role=SpeakerRole.OTHER), record("m2", "好的", minute=1)]
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [],
        processed_at=PROCESSED_AT,
    )

    assert [item.record.message_id for item in result.records] == ["m2"]
    assert result.records[0].record.text is None
    assert result.records[0].redacted_text == "好的"
    assert result.audit.scanned_segment_count == 1
    assert result.audit.risk_segment_count == 0
    assert result.audit.auto_cleared_segment_count == 1
    assert result.audit.dropped_other_count == 1


def test_approved_risk_segment_emits_only_redacted_owner_messages() -> None:
    records = mixed_records()
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [
            decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE),
            decision("segment-000002-risk-001", ReviewDecisionValue.APPROVE),
        ],
        processed_at=PROCESSED_AT,
    )

    assert [item.record.message_id for item in result.records] == ["m2", "m4"]
    assert all(item.record.speaker_role is SpeakerRole.OWNER for item in result.records)
    assert all(item.record.text is None for item in result.records)
    assert result.records[0].redacted_text == "联系 [EMAIL]"
    assert result.audit.approved_risk_segment_count == 2
    assert result.audit.excluded_risk_segment_count == 0
    assert result.audit.redaction_counts == {"EMAIL": 1}


def test_excluded_risk_segment_emits_no_candidates_from_that_segment() -> None:
    records = mixed_records()
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [
            decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE),
            decision("segment-000002-risk-001", ReviewDecisionValue.EXCLUDE),
        ],
        processed_at=PROCESSED_AT,
    )

    assert [item.record.message_id for item in result.records] == ["m2"]
    assert result.audit.approved_risk_segment_count == 1
    assert result.audit.excluded_risk_segment_count == 1


def test_missing_risk_decision_fails_atomically() -> None:
    records = mixed_records()
    package = scan_for_review(records, created_at=BASE_TIME)

    with pytest.raises(PrivacyValidationError) as exc_info:
        finalize_privacy_batch(
            records,
            package,
            [decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE)],
            processed_at=PROCESSED_AT,
        )

    assert exc_info.value.code == "incomplete_review"
    assert exc_info.value.__context__ is None
    assert not privacy_traceback_retains(exc_info.value, "synthetic.owner")


@pytest.mark.parametrize(
    "decisions",
    [
        [
            decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE),
            decision("segment-000002-risk-001", ReviewDecisionValue.APPROVE),
            decision("segment-999999", ReviewDecisionValue.EXCLUDE),
        ],
        [
            decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE),
            decision("segment-000001-risk-001", ReviewDecisionValue.EXCLUDE),
            decision("segment-000002-risk-001", ReviewDecisionValue.APPROVE),
        ],
    ],
)
def test_extra_or_duplicate_decisions_fail_atomically(
    decisions: list[ReviewDecision],
) -> None:
    records = mixed_records()
    package = scan_for_review(records, created_at=BASE_TIME)

    with pytest.raises(PrivacyValidationError) as exc_info:
        finalize_privacy_batch(
            records,
            package,
            decisions,
            processed_at=PROCESSED_AT,
        )

    assert exc_info.value.code == "invalid_review_decisions"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda items: [items[0].model_copy(update={"text": "changed"}), *items[1:]],
        lambda items: [items[0].model_copy(update={"message_id": "changed"}), *items[1:]],
        lambda items: [
            items[0].model_copy(update={"timestamp": BASE_TIME + timedelta(seconds=1)}),
            *items[1:],
        ],
        lambda items: [items[0].model_copy(update={"speaker_role": SpeakerRole.OWNER}), *items[1:]],
        lambda items: [*reversed(items)],
        lambda items: [*items, record("m5", "extra", minute=62)],
        lambda items: items[:-1],
    ],
)
def test_changed_dataset_invalidates_review_package(mutate) -> None:
    records = mixed_records()
    package = scan_for_review(records, created_at=BASE_TIME)

    with pytest.raises(PrivacyValidationError) as exc_info:
        finalize_privacy_batch(
            mutate(records),
            package,
            [],
            processed_at=PROCESSED_AT,
        )

    assert exc_info.value.code == "stale_review_package"


def test_changed_risk_segments_invalidate_review_package() -> None:
    records = [record("m1", "银行卡密码")]
    package = scan_for_review(records, created_at=BASE_TIME)
    tampered = package.model_copy(update={"risk_segments": []})

    with pytest.raises(PrivacyValidationError) as exc_info:
        finalize_privacy_batch(
            records,
            tampered,
            [],
            processed_at=PROCESSED_AT,
        )

    assert exc_info.value.code == "stale_review_package"


def test_custom_risk_terms_must_be_reused_during_finalization() -> None:
    records = [record("m1", "虚构特殊私事甲")]
    terms = {RiskCategory.PRIVATE_EVENT: ("虚构特殊私事甲",)}
    package = scan_for_review(records, created_at=BASE_TIME, private_terms=terms)
    decisions = [decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE)]

    with pytest.raises(PrivacyValidationError) as exc_info:
        finalize_privacy_batch(
            records,
            package,
            decisions,
            processed_at=PROCESSED_AT,
        )

    assert exc_info.value.code == "stale_review_package"

    result = finalize_privacy_batch(
        records,
        package,
        decisions,
        processed_at=PROCESSED_AT,
        risk_terms=terms,
    )
    assert len(result.records) == 1


def test_audit_is_aggregate_only() -> None:
    records = mixed_records()
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [
            decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE),
            decision("segment-000002-risk-001", ReviewDecisionValue.EXCLUDE),
        ],
        processed_at=PROCESSED_AT,
    )

    audit_json = result.audit.model_dump_json()
    assert result.audit.input_count == 4
    assert result.audit.owner_output_count == 1
    assert "synthetic-conversation" not in audit_json
    assert "synthetic.owner" not in audit_json
    assert package.dataset_fingerprint not in audit_json
    assert all(item not in audit_json for item in ("m1", "m2", "m3", "m4"))


def test_fifty_approved_canaries_leave_no_original_values() -> None:
    canaries = [f"synthetic.canary{index:02d}@example.test" for index in range(50)]
    records = [
        record(f"m{index:02d}", canary, minute=index) for index, canary in enumerate(canaries)
    ]
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE)],
        processed_at=PROCESSED_AT,
    )

    output = " ".join(item.redacted_text or "" for item in result.records)
    assert len(result.records) == 50
    assert all(canary not in output for canary in canaries)
    assert output.count("[EMAIL]") == 50


def test_pipeline_requires_timezone_for_audit() -> None:
    records = [record("synthetic-secret-message", "synthetic-secret-text")]

    with pytest.raises(PrivacyValidationError) as exc_info:
        finalize_privacy_batch(
            records,
            scan_for_review(records, created_at=BASE_TIME),
            [],
            processed_at=datetime(2026, 7, 17, 12, 0),
        )

    assert exc_info.value.code == "invalid_timestamp"
    assert not privacy_traceback_retains(exc_info.value, "synthetic-secret")


def test_excluded_risk_window_drops_only_records_inside_that_window() -> None:
    records = [record(f"m{index:02d}", "普通消息", minute=index) for index in range(20)]
    records[10] = record("m10", "医院复查", minute=10)
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [decision("segment-000001-risk-001", ReviewDecisionValue.EXCLUDE)],
        processed_at=PROCESSED_AT,
    )

    assert [item.record.message_id for item in result.records] == [
        *[f"m{index:02d}" for index in range(5)],
        *[f"m{index:02d}" for index in range(16, 20)],
    ]


def test_approved_risk_window_keeps_owner_messages_inside_and_outside_window() -> None:
    records = [record(f"m{index:02d}", "普通消息", minute=index) for index in range(20)]
    records[10] = record("m10", "医院复查", minute=10)
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE)],
        processed_at=PROCESSED_AT,
    )

    assert [item.record.message_id for item in result.records] == [
        f"m{index:02d}" for index in range(20)
    ]


def test_partner_messages_never_enter_output_even_when_window_is_approved() -> None:
    records = [
        record(
            f"m{index:02d}",
            "医院复查" if index == 10 else "普通消息",
            minute=index,
            role=SpeakerRole.OTHER,
        )
        for index in range(20)
    ]
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE)],
        processed_at=PROCESSED_AT,
    )

    assert result.records == []
    assert result.audit.dropped_other_count == 20


def test_multiple_risk_windows_in_one_source_segment_do_not_reduce_auto_clear_count() -> None:
    records = [record(f"m{index:02d}", "普通消息", minute=index) for index in range(30)]
    records[5] = record("m05", "银行卡密码", minute=5)
    records[20] = record("m20", "医院复查", minute=20)
    package = scan_for_review(records, created_at=BASE_TIME)

    result = finalize_privacy_batch(
        records,
        package,
        [
            decision("segment-000001-risk-001", ReviewDecisionValue.APPROVE),
            decision("segment-000001-risk-002", ReviewDecisionValue.EXCLUDE),
        ],
        processed_at=PROCESSED_AT,
    )

    assert result.audit.risk_segment_count == 2
    assert result.audit.approved_risk_segment_count == 1
    assert result.audit.excluded_risk_segment_count == 1
    assert result.audit.auto_cleared_segment_count == 0
