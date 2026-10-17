from datetime import UTC, datetime, timedelta

import pytest

from companion.contracts import ChatRecord, SpeakerRole
from companion.privacy.ambiguous_models import CalibrationProfile, CalibrationRule
from companion.privacy.candidate_models import CandidateCategory
from companion.privacy.candidate_selection import _feature_group, select_safe_candidates
from companion.privacy.errors import PrivacyValidationError

BASE_TIME = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
PROCESSED_AT = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
TRUE_NAMES = ("示例姓名甲",)
NICKNAMES = ("小团子",)

REQUIRED_ADVERSARIAL_CLASSES = {
    "conflict",
    "contact",
    "finance",
    "health",
    "identity",
    "location",
    "obfuscation",
    "partner_only",
    "private_event",
    "safe_private_mixture",
    "schedule",
    "sexual_or_suggestive",
    "unknown_prose",
}

PRIVATE_OR_UNKNOWN_CANARY_CASES = (
    ("sexual_or_suggestive", "合成成人暗示甲", SpeakerRole.OWNER),
    ("sexual_or_suggestive", "合成暧昧话题乙", SpeakerRole.OWNER),
    ("sexual_or_suggestive", "合成裸聊话题丙", SpeakerRole.OWNER),
    ("sexual_or_suggestive", "合成想亲亲的内容丁", SpeakerRole.OWNER),
    ("sexual_or_suggestive", "合成要摸摸的内容戊", SpeakerRole.OWNER),
    ("sexual_or_suggestive", "合成性-生活内容己", SpeakerRole.OWNER),
    ("sexual_or_suggestive", "合成杏_行为内容庚", SpeakerRole.OWNER),
    ("obfuscation", "合成性 生活变体甲", SpeakerRole.OWNER),
    ("obfuscation", "合成杏.行为变体乙", SpeakerRole.OWNER),
    ("obfuscation", "合成性\uff0e生活变体丙", SpeakerRole.OWNER),
    ("obfuscation", "合成瑟瑟变体丁", SpeakerRole.OWNER),
    ("obfuscation", "合成涩涩变体戊", SpeakerRole.OWNER),
    ("conflict", "合成争吵内容甲", SpeakerRole.OWNER),
    ("conflict", "合成分手内容乙", SpeakerRole.OWNER),
    ("conflict", "合成冷战内容丙", SpeakerRole.OWNER),
    ("conflict", "合成嫉妒内容丁", SpeakerRole.OWNER),
    ("conflict", "合成胁迫内容戊", SpeakerRole.OWNER),
    ("health", "合成医院内容甲", SpeakerRole.OWNER),
    ("health", "合成诊断内容乙", SpeakerRole.OWNER),
    ("health", "合成怀孕内容丙", SpeakerRole.OWNER),
    ("health", "合成心理健康内容丁", SpeakerRole.OWNER),
    ("health", "合成吃药内容戊", SpeakerRole.OWNER),
    ("finance", "合成银行卡内容甲", SpeakerRole.OWNER),
    ("finance", "合成转账内容乙", SpeakerRole.OWNER),
    ("finance", "合成密码内容丙", SpeakerRole.OWNER),
    ("finance", "合成工资内容丁", SpeakerRole.OWNER),
    ("finance", "合成债务内容戊", SpeakerRole.OWNER),
    ("contact", "联系 synthetic-canary@example.test", SpeakerRole.OWNER),
    ("contact", "手机 13800138001", SpeakerRole.OWNER),
    ("contact", "合成电话内容丙", SpeakerRole.OWNER),
    ("contact", "合成联系方式内容丁", SpeakerRole.OWNER),
    ("identity", "合成身份证内容甲", SpeakerRole.OWNER),
    ("identity", "合成证件内容乙", SpeakerRole.OWNER),
    ("identity", "地址 192.0.2.44", SpeakerRole.OWNER),
    ("identity", "合成手机号内容丁", SpeakerRole.OWNER),
    ("location", "合成酒店房间甲", SpeakerRole.OWNER),
    ("location", "合成具体地点乙", SpeakerRole.OWNER),
    ("location", "合成实时位置丙", SpeakerRole.OWNER),
    ("location", "合成地址内容丁", SpeakerRole.OWNER),
    ("schedule", "合成明天下午三点半安排甲", SpeakerRole.OWNER),
    ("schedule", "合成周五晚上八点十五分安排乙", SpeakerRole.OWNER),
    ("schedule", "合成行程安排丙", SpeakerRole.OWNER),
    ("schedule", "合成具体时间安排丁", SpeakerRole.OWNER),
    ("private_event", "合成私人事件甲", SpeakerRole.OWNER),
    ("private_event", "合成秘密内容乙", SpeakerRole.OWNER),
    ("private_event", "合成不要告诉内容丙", SpeakerRole.OWNER),
    ("private_event", "合成保密内容丁", SpeakerRole.OWNER),
    ("safe_private_mixture", "吃饭后聊合成成人暗示内容甲", SpeakerRole.OWNER),
    ("safe_private_mixture", "晚安后聊合成医院内容乙", SpeakerRole.OWNER),
    ("safe_private_mixture", "谢谢你但有合成银行卡内容丙", SpeakerRole.OWNER),
    ("safe_private_mixture", "今天天气真好但有合成实时位置丁", SpeakerRole.OWNER),
    ("unknown_prose", "合成蓝色纸船慢慢转圈", SpeakerRole.OWNER),
    ("unknown_prose", "合成一段没有分类标签的话", SpeakerRole.OWNER),
    ("unknown_prose", "合成窗边有三颗石头", SpeakerRole.OWNER),
    ("unknown_prose", "合成句子沿着方格移动", SpeakerRole.OWNER),
    ("unknown_prose", "合成陌生词组轻轻排列", SpeakerRole.OWNER),
    ("partner_only", "早安", SpeakerRole.OTHER),
    ("partner_only", "谢谢你", SpeakerRole.OTHER),
    ("partner_only", "我准备吃饭啦", SpeakerRole.OTHER),
    ("partner_only", "我要睡觉啦", SpeakerRole.OTHER),
    ("partner_only", "今天天气真好", SpeakerRole.OTHER),
)

SAFE_CANARY_CASES = (
    ("早安", CandidateCategory.GREETING),
    ("晚安小团子", CandidateCategory.GREETING),
    ("谢谢你", CandidateCategory.COURTESY),
    ("辛苦了", CandidateCategory.COURTESY),
    ("我准备吃饭啦", CandidateCategory.MEALS),
    ("我要吃晚饭了", CandidateCategory.MEALS),
    ("我要睡觉啦", CandidateCategory.SLEEP),
    ("我去睡觉了", CandidateCategory.SLEEP),
    ("今天天气真好", CandidateCategory.WEATHER),
    ("今天天气不错", CandidateCategory.WEATHER),
    ("我去收拾房间啦", CandidateCategory.HOUSEHOLD),
    ("我准备洗衣服了", CandidateCategory.HOUSEHOLD),
    ("我去买日用品", CandidateCategory.SHOPPING),
    ("我准备买菜了", CandidateCategory.SHOPPING),
    ("我准备学习啦", CandidateCategory.WORK_STUDY),
    ("我去写作业了", CandidateCategory.WORK_STUDY),
    ("今天好开心呀", CandidateCategory.GENERAL_EMOTION),
    ("我很想你呀", CandidateCategory.GENTLE_AFFECTION),
    ("嗯嗯", CandidateCategory.SHORT_REPLY),
    ("😊", CandidateCategory.EMOJI),
)


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
    content_type: str = "text",
    conversation_id: str = "synthetic-conversation",
) -> ChatRecord:
    return ChatRecord(
        conversation_id=conversation_id,
        message_id=message_id,
        timestamp=BASE_TIME + timedelta(minutes=minute),
        speaker_role=role,
        content_type=content_type,
        text=text,
    )


def synthetic_profile(
    *,
    manual_approved_texts: tuple[str, ...] = (),
    rules: tuple[CalibrationRule, ...] = (),
    enabled: bool = True,
) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id="synthetic-profile",
        activated_at=PROCESSED_AT,
        manual_approved_texts=list(manual_approved_texts),
        rules=list(rules),
        first_round_keep_count=max(2, len(manual_approved_texts)),
        first_round_discard_count=0,
        first_round_uncertain_count=0,
        second_round_keep_count=1,
        second_round_discard_count=0,
        second_round_uncertain_count=0,
        user_confirmed=True,
        enabled=enabled,
    )


def select(records: list[ChatRecord]):
    return select_safe_candidates(
        records,
        processed_at=PROCESSED_AT,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
    )


def synthetic_private_and_unknown_canaries() -> list[ChatRecord]:
    return [
        record(
            f"adversarial-{index}",
            text,
            minute=index * 40,
            role=role,
            conversation_id=f"synthetic-canary-conversation-{index}",
        )
        for index, (_, text, role) in enumerate(PRIVATE_OR_UNKNOWN_CANARY_CASES)
    ]


def synthetic_safe_canaries() -> list[ChatRecord]:
    return [
        record(f"safe-{index}", text, minute=index)
        for index, (text, _) in enumerate(SAFE_CANARY_CASES)
    ]


def test_private_unknown_and_partner_canaries_produce_zero_original_output() -> None:
    records = synthetic_private_and_unknown_canaries()

    package = select(records)
    serialized = package.model_dump_json()

    assert len(records) >= 50
    assert {case_class for case_class, _, _ in PRIVATE_OR_UNKNOWN_CANARY_CASES} == (
        REQUIRED_ADVERSARIAL_CLASSES
    )
    assert package.candidates == []
    assert package.audit.owner_input_count == 56
    assert package.audit.dropped_other_count == 5
    assert sum(package.audit.rejection_reason_counts.values()) == 51
    assert package.audit.unknown_count == 5
    assert all(record.text is None or record.text not in serialized for record in records)


def test_twenty_safe_canaries_cover_all_allow_categories() -> None:
    records = synthetic_safe_canaries()

    package = select(records)

    assert len(records) == len(SAFE_CANARY_CASES) == 20
    assert len(package.candidates) == 20
    assert [(candidate.text, candidate.category) for candidate in package.candidates] == list(
        SAFE_CANARY_CASES
    )
    assert set(package.audit.allowed_category_counts) == {
        category.value
        for category in CandidateCategory
        if category is not CandidateCategory.CALIBRATED_DAILY
    }
    assert all(
        candidate.scene_tags == [candidate.category.value] for candidate in package.candidates
    )


def test_selector_keeps_only_explicitly_safe_owner_text() -> None:
    records = [
        record("m1", "晚安小团子"),
        record("m2", "我准备吃饭啦", minute=40),
        record("m3", "合成无法分类长句", minute=80),
        record("m4", "晚安呀", minute=120, role=SpeakerRole.OTHER),
    ]

    package = select(records)

    assert [candidate.text for candidate in package.candidates] == [
        "晚安小团子",
        "我准备吃饭啦",
    ]
    assert package.audit.owner_input_count == 3
    assert package.audit.dropped_other_count == 1
    assert package.audit.unknown_count == 1


def test_hard_reject_wins_over_safe_word_and_partner_context_is_never_output() -> None:
    records = [
        record("m1", "合成成人暗示内容", role=SpeakerRole.OTHER),
        record("m2", "嗯嗯", minute=1),
        record("m3", "吃饭后聊合成成人暗示内容", minute=40),
    ]

    package = select(records)

    assert package.candidates == []


def test_hard_risk_context_rejects_self_contained_allow_category() -> None:
    package = select(
        [
            record("m1", "合成医院诊断内容", role=SpeakerRole.OTHER),
            record("m2", "我准备吃饭啦", minute=1),
        ]
    )

    assert package.candidates == []
    assert package.audit.rejection_reason_counts == {"sensitive_context": 1}


def test_short_reply_requires_positively_safe_context() -> None:
    records = [
        record("m1", "合成无法分类长句", role=SpeakerRole.OTHER),
        record("m2", "嗯嗯", minute=1),
        record("m3", "我准备吃饭啦", minute=2),
    ]

    package = select(records)

    assert [item.text for item in package.candidates] == ["我准备吃饭啦"]


@pytest.mark.parametrize(
    "unsafe_context",
    [
        record("m1", "合成无法分类长句", role=SpeakerRole.OTHER),
        record("m1", None, role=SpeakerRole.OTHER),
        record("m1", "合成附件", role=SpeakerRole.OTHER, content_type="image"),
    ],
)
def test_short_reply_rejects_unknown_or_non_text_context(
    unsafe_context: ChatRecord,
) -> None:
    package = select([unsafe_context, record("m2", "嗯嗯", minute=1)])

    assert package.candidates == []
    assert package.audit.rejection_reason_counts == {"sensitive_context": 1}


def test_isolated_short_reply_is_rejected() -> None:
    package = select([record("m1", "嗯嗯")])

    assert package.candidates == []
    assert package.audit.rejection_reason_counts == {"sensitive_context": 1}


@pytest.mark.parametrize(
    "context",
    [
        pytest.param([], id="empty"),
        pytest.param(
            [record("m1", "合成无法分类长句", role=SpeakerRole.OTHER)],
            id="unknown",
        ),
        pytest.param(
            [record("m1", None, role=SpeakerRole.OTHER)],
            id="missing-text",
        ),
        pytest.param(
            [
                record(
                    "m1",
                    "合成附件",
                    role=SpeakerRole.OTHER,
                    content_type="image",
                )
            ],
            id="non-text",
        ),
    ],
)
def test_emoji_requires_non_empty_positively_safe_context(
    context: list[ChatRecord],
) -> None:
    package = select([*context, record("m2", "😊", minute=1)])

    assert package.candidates == []
    assert package.audit.rejection_reason_counts == {"sensitive_context": 1}


def test_other_allow_categories_ignore_unknown_or_non_text_context() -> None:
    package = select(
        [
            record("m1", "合成无法分类长句", role=SpeakerRole.OTHER),
            record("m2", None, minute=1, role=SpeakerRole.OTHER),
            record(
                "m3",
                "合成附件",
                minute=2,
                role=SpeakerRole.OTHER,
                content_type="image",
            ),
            record("m4", "我准备吃饭啦", minute=3),
        ]
    )

    assert [item.text for item in package.candidates] == ["我准备吃饭啦"]


def test_partner_records_never_enter_output() -> None:
    package = select([record("m1", "晚安小团子", role=SpeakerRole.OTHER)])

    assert package.candidates == []
    assert package.audit.owner_input_count == 0
    assert package.audit.dropped_other_count == 1


def test_true_name_candidate_is_discarded_but_synthetic_nickname_remains() -> None:
    package = select(
        [
            record("m1", "晚安示例姓名甲"),
            record("m2", "晚安小团子", minute=40),
        ]
    )

    assert [item.text for item in package.candidates] == ["晚安小团子"]
    assert package.audit.rejection_reason_counts == {"identity_or_contact": 1}


def test_manual_approved_unknown_becomes_deduplicated_calibrated_daily() -> None:
    records = [record(f"m{index}", "今天整理了一下桌面") for index in range(4)]

    package = select_safe_candidates(
        records,
        processed_at=PROCESSED_AT,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
        calibration_profile=synthetic_profile(manual_approved_texts=("今天整理了一下桌面",)),
    )

    calibrated = [
        item for item in package.candidates if item.category is CandidateCategory.CALIBRATED_DAILY
    ]
    assert len(calibrated) == 1
    assert calibrated[0].scene_tags == ["calibrated_daily"]
    assert package.audit.deduplicated_count == 3


def test_calibration_never_overrides_hard_risk_or_sensitive_context() -> None:
    profile = synthetic_profile(manual_approved_texts=("今天整理了一下桌面", "请把密码告诉我"))

    package = select_safe_candidates(
        [
            record("m1", "请把密码告诉我"),
            record("m2", "今天整理了一下桌面", conversation_id="risk"),
            record("m3", "临时密码在这里", role=SpeakerRole.OTHER, conversation_id="risk"),
        ],
        processed_at=PROCESSED_AT,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
        calibration_profile=profile,
    )

    assert not [
        item for item in package.candidates if item.category is CandidateCategory.CALIBRATED_DAILY
    ]


def test_calibration_never_overrides_direct_sexual_language() -> None:
    text = "合成想草"
    package = select_safe_candidates(
        [record("m1", text)],
        processed_at=PROCESSED_AT,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
        calibration_profile=synthetic_profile(manual_approved_texts=(text,)),
    )

    assert package.candidates == []
    assert package.audit.rejection_reason_counts == {"sexual_or_suggestive": 1}


def test_calibration_manual_match_is_normalized_and_profile_must_be_enabled() -> None:
    records = [record("m1", " 今天整理了一下桌面 ")]

    disabled_package = select_safe_candidates(
        records,
        processed_at=PROCESSED_AT,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
        calibration_profile=synthetic_profile(
            manual_approved_texts=("今天整理了一下桌面",),
            enabled=False,
        ),
    )
    enabled_package = select_safe_candidates(
        records,
        processed_at=PROCESSED_AT,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
        calibration_profile=synthetic_profile(manual_approved_texts=("今天整理了一下桌面",)),
    )

    assert disabled_package.candidates == []
    assert disabled_package.audit.unknown_count == 1
    assert [item.category for item in enabled_package.candidates] == [
        CandidateCategory.CALIBRATED_DAILY
    ]


def test_enabled_calibration_rule_matches_unknown_text() -> None:
    text = "今天整理了一下桌面"
    rule = CalibrationRule(
        rule_id="synthetic-rule-selection",
        category=CandidateCategory.CALIBRATED_DAILY,
        similarity_group=_feature_group(text),
        normalized_template=text,
        similarity_threshold=1.0,
        first_round_approved=2,
        second_round_approved=1,
        second_round_discarded=0,
        second_round_uncertain=0,
        enabled=True,
    )

    package = select_safe_candidates(
        [record("m1", text)],
        processed_at=PROCESSED_AT,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=NICKNAMES,
        calibration_profile=synthetic_profile(rules=(rule,)),
    )

    assert [item.category for item in package.candidates] == [CandidateCategory.CALIBRATED_DAILY]


def test_short_reply_duplicates_are_capped_at_three() -> None:
    records = [
        item
        for index in range(7)
        for item in (
            record(
                f"context-{index}",
                "我准备吃饭啦",
                minute=index * 80,
                role=SpeakerRole.OTHER,
            ),
            record(f"reply-{index}", "嗯嗯", minute=index * 80 + 1),
        )
    ]

    package = select_safe_candidates(
        records,
        processed_at=PROCESSED_AT,
        true_name_terms=TRUE_NAMES,
        approved_nicknames=(),
    )

    assert len(package.candidates) == 3
    assert all(
        candidate.category is CandidateCategory.SHORT_REPLY for candidate in package.candidates
    )
    assert package.audit.deduplicated_count == 4


def test_audit_is_consistent_sorted_and_contains_no_source_identifiers() -> None:
    package = select(
        [
            record("synthetic-message-1", "我准备吃饭啦"),
            record("synthetic-message-2", "晚安小团子", minute=40),
            record("synthetic-message-3", "合成秘密内容", minute=80),
            record("synthetic-message-4", "合成未知内容", minute=120),
        ]
    )

    assert package.created_at == PROCESSED_AT
    assert package.audit.input_count == 4
    assert package.audit.output_count == len(package.candidates) == 2
    assert package.audit.allowed_category_counts == {"greeting": 1, "meals": 1}
    assert package.audit.rejection_reason_counts == {"private_event": 1}
    assert package.audit.unknown_count == 1
    assert list(package.audit.allowed_category_counts) == ["greeting", "meals"]
    payload = package.model_dump_json()
    assert "synthetic-conversation" not in payload
    assert "synthetic-message" not in payload
    assert all(
        set(candidate.model_dump()) == {"text", "category", "scene_tags"}
        for candidate in package.candidates
    )
    assert [candidate.category for candidate in package.candidates] == [
        CandidateCategory.MEALS,
        CandidateCategory.GREETING,
    ]


@pytest.mark.parametrize(
    ("records", "processed_at", "expected_code"),
    [
        ([], PROCESSED_AT, "empty_records"),
        (
            [record("m1", "晚安"), record("m1", "我准备吃饭啦", minute=40)],
            PROCESSED_AT,
            "duplicate_message_id",
        ),
        (
            [record("m1", "晚安", role=SpeakerRole.UNKNOWN)],
            PROCESSED_AT,
            "unconfirmed_speaker_role",
        ),
        (
            [record("m1", "晚安")],
            datetime(2026, 7, 17, 12, 0),
            "invalid_timestamp",
        ),
        (
            [record("m1", "晚安").model_copy(update={"timestamp": datetime(2026, 7, 17, 10, 0)})],
            PROCESSED_AT,
            "invalid_timestamp",
        ),
    ],
)
def test_record_and_timestamp_validation(
    records: list[ChatRecord],
    processed_at: datetime,
    expected_code: str,
) -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        select_safe_candidates(
            records,
            processed_at=processed_at,
            true_name_terms=TRUE_NAMES,
            approved_nicknames=NICKNAMES,
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("true_names", "nicknames"),
    [
        ((), NICKNAMES),
        ([], NICKNAMES),
        (("",), NICKNAMES),
        (TRUE_NAMES, ("   ",)),
        (("synthetic.owner@example.test",), NICKNAMES),
        (TRUE_NAMES, ("合成密码内容",)),
        ("示例姓名甲", NICKNAMES),
        (TRUE_NAMES, "小团子"),
        (("示例姓名甲",), ("示例姓名甲",)),
        ((123,), NICKNAMES),
    ],
)
def test_candidate_term_validation_is_strict(
    true_names,
    nicknames,
) -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        select_safe_candidates(
            [record("m1", "晚安小团子")],
            processed_at=PROCESSED_AT,
            true_name_terms=true_names,
            approved_nicknames=nicknames,
        )

    assert exc_info.value.code == "invalid_candidate_terms"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("records", "true_names", "marker"),
    [
        (
            [
                record("m1", "synthetic-private-text"),
                record("m1", "晚安", minute=40),
            ],
            TRUE_NAMES,
            "synthetic-private-text",
        ),
        (
            [record("m1", "晚安")],
            ("synthetic-private-term@example.test",),
            "synthetic-private-term",
        ),
    ],
)
def test_public_errors_do_not_retain_input_values_in_privacy_traceback_locals(
    records: list[ChatRecord],
    true_names,
    marker: str,
) -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        select_safe_candidates(
            records,
            processed_at=PROCESSED_AT,
            true_name_terms=true_names,
            approved_nicknames=NICKNAMES,
        )

    assert marker not in str(exc_info.value)
    assert not privacy_traceback_retains(exc_info.value, marker)
