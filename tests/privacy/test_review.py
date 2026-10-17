from datetime import UTC, datetime, timedelta

import pytest

from companion.contracts import ChatRecord, SpeakerRole
from companion.privacy import (
    PrivacyValidationError,
    ReviewDecision,
    ReviewDecisionValue,
    RiskCategory,
    read_review_decisions,
    scan_for_review,
    write_review_package,
)

BASE_TIME = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)


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
    conversation_id: str = "synthetic-conversation",
) -> ChatRecord:
    return ChatRecord(
        conversation_id=conversation_id,
        message_id=message_id,
        timestamp=BASE_TIME + timedelta(minutes=minute),
        speaker_role=role,
        content_type="text",
        text=text,
    )


def test_scan_checks_every_segment_but_returns_only_risky_segments() -> None:
    package = scan_for_review(
        [
            record("m1", "普通聊天", role=SpeakerRole.OTHER),
            record("m2", "好的呀", minute=1),
            record("m3", "银行卡密码别发在这里", minute=60, role=SpeakerRole.OTHER),
            record("m4", "我知道了", minute=61),
        ],
        created_at=BASE_TIME,
    )

    assert package.scanned_segment_count == 2
    assert len(package.risk_segments) == 1
    segment = package.risk_segments[0]
    assert segment.segment_id == "segment-000002-risk-001"
    assert segment.source_segment_id == "segment-000002"
    assert segment.risk_categories == [RiskCategory.CREDENTIAL, RiskCategory.FINANCIAL]
    assert segment.reason_labels == ["credential_keyword", "financial_keyword"]
    assert segment.recommended_decision is ReviewDecisionValue.EXCLUDE
    assert [item.message_id for item in segment.records] == ["m3", "m4"]
    assert len(package.dataset_fingerprint) == 64


def test_exactly_thirty_minute_gap_stays_together_and_larger_gap_splits() -> None:
    package = scan_for_review(
        [
            record("m1", "普通消息"),
            record("m2", "普通回复", minute=30),
            record("m3", "家庭矛盾", minute=61),
        ],
        created_at=BASE_TIME,
    )

    assert package.scanned_segment_count == 2
    assert [item.segment_id for item in package.risk_segments] == ["segment-000002-risk-001"]


def test_conversations_never_merge() -> None:
    package = scan_for_review(
        [
            record("m1", "家庭矛盾", conversation_id="synthetic-a"),
            record("m2", "医院复查", minute=1, conversation_id="synthetic-b"),
        ],
        created_at=BASE_TIME,
    )

    assert package.scanned_segment_count == 2
    assert len(package.risk_segments) == 2
    assert [len(item.records) for item in package.risk_segments] == [1, 1]


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("这是亲密关系话题", RiskCategory.INTIMATE),
        ("涉及性生活", RiskCategory.SEXUAL),
        ("明天医院复查", RiskCategory.HEALTH),
        ("家庭矛盾需要处理", RiskCategory.FAMILY),
        ("银行卡余额不足", RiskCategory.FINANCIAL),
        ("不要发送登录密码", RiskCategory.CREDENTIAL),
        ("我们刚刚吵架了", RiskCategory.CONFLICT),
        ("把详细地址发给我", RiskCategory.LOCATION),
        ("这是一个私人事件", RiskCategory.PRIVATE_EVENT),
    ],
)
def test_high_recall_topic_rules_cover_all_categories(
    text: str,
    category: RiskCategory,
) -> None:
    package = scan_for_review([record("m1", text)], created_at=BASE_TIME)

    assert category in package.risk_segments[0].risk_categories


@pytest.mark.parametrize(
    ("text", "category", "reason"),
    [
        ("联系 synthetic.user@example.test", RiskCategory.CREDENTIAL, "email_pattern"),
        ("手机 13800138000", RiskCategory.CREDENTIAL, "phone_pattern"),
        ("证件 11010519491231002X", RiskCategory.CREDENTIAL, "national_id_pattern"),
        ("卡号 6222021234567890123", RiskCategory.FINANCIAL, "bank_card_pattern"),
        ("地址 192.0.2.10", RiskCategory.LOCATION, "ip_address_pattern"),
    ],
)
def test_structured_sensitive_values_flag_segments(
    text: str,
    category: RiskCategory,
    reason: str,
) -> None:
    package = scan_for_review([record("m1", text)], created_at=BASE_TIME)

    segment = package.risk_segments[0]
    assert category in segment.risk_categories
    assert reason in segment.reason_labels


def test_unflagged_segments_do_not_enter_review_package() -> None:
    package = scan_for_review(
        [record("m1", "今天天气不错"), record("m2", "是呀", minute=1)],
        created_at=BASE_TIME,
    )

    assert package.scanned_segment_count == 1
    assert package.risk_segments == []


def test_custom_private_term_flags_whole_segment() -> None:
    private_value = "虚构私密事件甲"
    package = scan_for_review(
        [record("m1", f"提到了{private_value}"), record("m2", "知道了", minute=1)],
        created_at=BASE_TIME,
        private_terms={RiskCategory.PRIVATE_EVENT: (private_value,)},
    )

    assert len(package.risk_segments) == 1
    assert package.risk_segments[0].reason_labels == ["private_term_match"]
    assert private_value in package.model_dump_json()


@pytest.mark.parametrize(
    "private_terms",
    [
        {RiskCategory.PRIVATE_EVENT: ()},
        {RiskCategory.PRIVATE_EVENT: ("",)},
        {RiskCategory.PRIVATE_EVENT: ("   ",)},
    ],
)
def test_invalid_private_terms_fail_without_values(
    private_terms: dict[RiskCategory, tuple[str, ...]],
) -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        scan_for_review(
            [record("m1", "普通消息")],
            created_at=BASE_TIME,
            private_terms=private_terms,
        )

    assert exc_info.value.code == "invalid_private_terms"
    assert "普通消息" not in str(exc_info.value)


def test_empty_records_fail_closed() -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        scan_for_review([], created_at=BASE_TIME)

    assert exc_info.value.code == "empty_records"


def test_duplicate_message_ids_fail_closed() -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        scan_for_review(
            [record("duplicate", "普通消息"), record("duplicate", "另一条", minute=1)],
            created_at=BASE_TIME,
        )

    assert exc_info.value.code == "duplicate_message_id"


def test_unknown_role_fails_without_retaining_input_values() -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        scan_for_review(
            [record("synthetic-secret-message", "synthetic-secret-text", role=SpeakerRole.UNKNOWN)],
            created_at=BASE_TIME,
        )

    assert exc_info.value.code == "unconfirmed_speaker_role"
    assert "synthetic-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert not privacy_traceback_retains(exc_info.value, "synthetic-secret")


def test_review_package_requires_timezone() -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        scan_for_review(
            [record("synthetic-secret-message", "synthetic-secret-text")],
            created_at=datetime(2026, 7, 17, 10, 0),
        )

    assert exc_info.value.code == "invalid_timestamp"
    assert not privacy_traceback_retains(exc_info.value, "synthetic-secret")


def test_single_hit_returns_only_five_messages_on_each_side() -> None:
    records = [record(f"m{index:02d}", "普通消息", minute=index) for index in range(20)]
    records[10] = record("m10", "银行卡密码", minute=10)

    package = scan_for_review(records, created_at=BASE_TIME)

    assert package.scanned_segment_count == 1
    assert len(package.risk_segments) == 1
    window = package.risk_segments[0]
    assert window.source_segment_id == "segment-000001"
    assert [item.message_id for item in window.records] == [
        f"m{index:02d}" for index in range(5, 16)
    ]


@pytest.mark.parametrize(
    ("hit_index", "expected_indexes"),
    [
        (0, range(0, 6)),
        (19, range(14, 20)),
    ],
)
def test_hit_window_clamps_at_source_segment_boundaries(
    hit_index: int,
    expected_indexes: range,
) -> None:
    records = [record(f"m{index:02d}", "普通消息", minute=index) for index in range(20)]
    records[hit_index] = record(f"m{hit_index:02d}", "医院复查", minute=hit_index)

    window = scan_for_review(records, created_at=BASE_TIME).risk_segments[0]

    assert [item.message_id for item in window.records] == [
        f"m{index:02d}" for index in expected_indexes
    ]


def test_overlapping_or_touching_hit_windows_merge() -> None:
    records = [record(f"m{index:02d}", "普通消息", minute=index) for index in range(22)]
    records[5] = record("m05", "银行卡密码", minute=5)
    records[16] = record("m16", "医院复查", minute=16)

    package = scan_for_review(records, created_at=BASE_TIME)

    assert len(package.risk_segments) == 1
    window = package.risk_segments[0]
    assert [item.message_id for item in window.records] == [f"m{index:02d}" for index in range(22)]
    assert RiskCategory.CREDENTIAL in window.risk_categories
    assert RiskCategory.FINANCIAL in window.risk_categories
    assert RiskCategory.HEALTH in window.risk_categories


def test_distant_hits_in_one_source_segment_remain_separate_without_duplicates() -> None:
    records = [record(f"m{index:02d}", "普通消息", minute=index) for index in range(30)]
    records[5] = record("m05", "银行卡密码", minute=5)
    records[20] = record("m20", "医院复查", minute=20)

    package = scan_for_review(records, created_at=BASE_TIME)

    assert package.scanned_segment_count == 1
    assert [item.segment_id for item in package.risk_segments] == [
        "segment-000001-risk-001",
        "segment-000001-risk-002",
    ]
    assert {item.source_segment_id for item in package.risk_segments} == {"segment-000001"}
    review_ids = [
        record.message_id for window in package.risk_segments for record in window.records
    ]
    assert len(review_ids) == len(set(review_ids))
    assert review_ids == [
        *[f"m{index:02d}" for index in range(0, 11)],
        *[f"m{index:02d}" for index in range(15, 26)],
    ]


def test_review_package_writes_only_inside_private_directory(tmp_path) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "reviews" / "synthetic-review.json"
    package = scan_for_review([record("m1", "家庭矛盾")], created_at=BASE_TIME)

    written = write_review_package(
        package,
        destination=destination,
        private_data_dir=private_root,
    )

    assert written == destination.resolve()
    restored = type(package).model_validate_json(destination.read_text(encoding="utf-8"))
    assert restored == package


def test_review_package_rejects_destination_outside_private_directory(tmp_path) -> None:
    package = scan_for_review([record("m1", "家庭矛盾")], created_at=BASE_TIME)
    unsafe_path = tmp_path / "synthetic-secret-review.json"

    with pytest.raises(PrivacyValidationError) as exc_info:
        write_review_package(
            package,
            destination=unsafe_path,
            private_data_dir=tmp_path / "private",
        )

    assert exc_info.value.code == "unsafe_review_path"
    assert "synthetic-secret" not in str(exc_info.value)


def test_review_package_write_failure_does_not_fall_back(tmp_path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    package = scan_for_review([record("m1", "家庭矛盾")], created_at=BASE_TIME)

    with pytest.raises(PrivacyValidationError) as exc_info:
        write_review_package(
            package,
            destination=private_root,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "review_write_failed"
    assert list(tmp_path.rglob("*.json")) == []


def test_review_package_is_owner_only_and_atomic(tmp_path, monkeypatch) -> None:
    import os
    import stat
    from pathlib import Path

    if os.name != "posix":
        pytest.skip("POSIX file mode contract")
    package = scan_for_review([record("m1", "合成家庭矛盾")], created_at=BASE_TIME)
    destination = tmp_path / "synthetic-review.json"
    previous = os.umask(0)
    try:
        write_review_package(package, destination=destination, private_data_dir=tmp_path)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    before = destination.read_bytes()

    def fail_replace(self, target):
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PrivacyValidationError):
        write_review_package(package, destination=destination, private_data_dir=tmp_path)
    assert destination.read_bytes() == before
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("parent_link", [False, True])
def test_review_package_rejects_internal_write_symlinks(tmp_path, parent_link) -> None:
    package = scan_for_review([record("m1", "合成家庭矛盾")], created_at=BASE_TIME)
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "synthetic.json"
    sentinel.write_text("synthetic unchanged", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(target if parent_link else sentinel, target_is_directory=parent_link)
    destination = alias / "synthetic.json" if parent_link else alias
    with pytest.raises(PrivacyValidationError) as caught:
        write_review_package(package, destination=destination, private_data_dir=tmp_path)
    assert caught.value.code == "unsafe_review_path"
    assert sentinel.read_text(encoding="utf-8") == "synthetic unchanged"


def test_review_package_cleanup_failure_is_sanitized(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    package = scan_for_review([record("m1", "合成家庭矛盾")], created_at=BASE_TIME)
    destination = tmp_path / "synthetic-review.json"

    def fail_unlink(self, **kwargs):
        raise OSError("SYNTHETIC_CLEANUP_PATH_SENTINEL")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(PrivacyValidationError) as caught:
        write_review_package(package, destination=destination, private_data_dir=tmp_path)
    assert caught.value.code == "review_write_failed"
    assert "SYNTHETIC_CLEANUP_PATH_SENTINEL" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_read_review_decisions_from_private_directory(tmp_path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    path = private_root / "synthetic-decisions.json"
    path.write_text(
        '{"decisions":[{"segment_id":"segment-000001","decision":"exclude"}]}',
        encoding="utf-8",
    )

    decisions = read_review_decisions(path, private_data_dir=private_root)

    assert decisions == [
        ReviewDecision(
            segment_id="segment-000001",
            decision=ReviewDecisionValue.EXCLUDE,
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        '{"decisions":[],"unknown":true}',
        '{"decisions":[{"segment_id":"segment-000001","decision":"exclude"},'
        '{"segment_id":"segment-000001","decision":"approve"}]}',
    ],
)
def test_invalid_review_decisions_fail_safely(tmp_path, payload: str) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    path = private_root / "synthetic-decisions.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(PrivacyValidationError) as exc_info:
        read_review_decisions(path, private_data_dir=private_root)

    assert exc_info.value.code == "invalid_review_decisions"
    assert payload not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_review_decisions_reject_path_outside_private_directory(tmp_path) -> None:
    path = tmp_path / "synthetic-secret-decisions.json"
    path.write_text('{"decisions":[]}', encoding="utf-8")

    with pytest.raises(PrivacyValidationError) as exc_info:
        read_review_decisions(path, private_data_dir=tmp_path / "private")

    assert exc_info.value.code == "unsafe_review_path"
    assert "synthetic-secret" not in str(exc_info.value)
