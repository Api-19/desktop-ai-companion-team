import pytest

from companion.privacy import redaction
from companion.privacy.candidate_models import CandidateCategory, CandidateRejectReason
from companion.privacy.candidate_rules import (
    allowed_category,
    classification_text,
    hard_rejection_reason,
    replace_private_names,
    short_repeat_key,
)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("合成成人暗示内容", CandidateRejectReason.SEXUAL_OR_SUGGESTIVE),
        ("合成争吵分手内容", CandidateRejectReason.RELATIONSHIP_CONFLICT),
        ("合成医院诊断内容", CandidateRejectReason.HEALTH_PRIVATE),
        ("合成家庭秘密内容", CandidateRejectReason.FAMILY_OR_THIRD_PARTY_PRIVATE),
        ("合成银行卡密码内容", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("联系 synthetic@example.test", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("合成酒店房间与精确时间", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
        ("吃饭后聊合成成人暗示内容", CandidateRejectReason.SEXUAL_OR_SUGGESTIVE),
        ("合成私人纪念事件", CandidateRejectReason.PRIVATE_EVENT),
        ("手机 13800138000", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("证件 11010519491231002X", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("卡号 6222021234567890123", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("地址 192.0.2.10", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("合成想要来点暧昧的事", CandidateRejectReason.SEXUAL_OR_SUGGESTIVE),
    ],
)
def test_hard_rejection_wins_for_private_or_mixed_text(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("合成嫉妒关系内容", CandidateRejectReason.RELATIONSHIP_CONFLICT),
        ("合成威胁关系内容", CandidateRejectReason.RELATIONSHIP_CONFLICT),
        ("合成控制关系内容", CandidateRejectReason.RELATIONSHIP_CONFLICT),
        ("合成胁迫关系内容", CandidateRejectReason.RELATIONSHIP_CONFLICT),
        ("合成生育隐私内容", CandidateRejectReason.HEALTH_PRIVATE),
        ("合成心理健康内容", CandidateRejectReason.HEALTH_PRIVATE),
        ("合成工资金额内容", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("合成单位安排内容", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
        ("合成实时位置内容", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
        ("合成行程安排内容", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
        ("合成明天下午三点半安排", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
        ("合成周五晚上八点十五分安排", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
    ],
)
def test_hard_rejection_covers_reviewed_private_topics(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


@pytest.mark.parametrize(
    "text",
    [
        "合成我们涩涩一下",
        "合成想聊瑟瑟",
        "合成想聊杏生活",
        "合成想做那件事",
    ],
)
def test_hard_rejection_covers_colloquial_obfuscated_sexual_constructions(text: str) -> None:
    assert hard_rejection_reason(text) is CandidateRejectReason.SEXUAL_OR_SUGGESTIVE


@pytest.mark.parametrize(
    "text",
    [
        "合成性 生活变体",
        "合成杏-行为变体",
        "合成性\uff0e生活变体",
        "合成想亲亲的变体",
        "合成要开车的变体",
        "合成去做那种事的变体",
        "合成来点摸摸的变体",
        "合成试试车车的变体",
    ],
)
def test_hard_rejection_normalizes_obfuscation_and_colloquial_variants(text: str) -> None:
    assert hard_rejection_reason(text) is CandidateRejectReason.SEXUAL_OR_SUGGESTIVE


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("合成编号 622202123456789012", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("合成身份证 110105194912310021", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("合成编号 11010519491231002X", CandidateRejectReason.IDENTITY_OR_CONTACT),
    ],
)
def test_eighteen_digit_values_have_deterministic_identity_or_card_reasons(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("合成秘密内容", CandidateRejectReason.PRIVATE_EVENT),
        ("合成隐私内容", CandidateRejectReason.PRIVATE_EVENT),
        ("合成不要告诉内容", CandidateRejectReason.PRIVATE_EVENT),
        ("合成保密内容", CandidateRejectReason.PRIVATE_EVENT),
        ("合成生病内容", CandidateRejectReason.HEALTH_PRIVATE),
        ("合成症状内容", CandidateRejectReason.HEALTH_PRIVATE),
        ("合成吃药内容", CandidateRejectReason.HEALTH_PRIVATE),
        ("合成账号内容", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("合成地址内容", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
        ("合成定位内容", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
    ],
)
def test_hard_rejection_covers_sensitive_context_terms(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("合成编号 999999200001010011", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("合成编号 99999920000101002X", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("合成编号 444444444444444444", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("合成编号 999999200002300011", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("合成身份证 999999200002300011", CandidateRejectReason.IDENTITY_OR_CONTACT),
    ],
)
def test_eighteen_digit_values_validate_identity_before_card_fallback(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("合成密码内容", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("合成凭证内容", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("合成疾病内容", CandidateRejectReason.HEALTH_PRIVATE),
    ],
)
def test_hard_rejection_covers_independent_sensitive_words(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("合成编号 999999200001010011中文后缀", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("合成编号 999999200001010012", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
    ],
)
def test_identity_tokens_allow_adjacent_chinese_but_require_a_valid_checksum(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


def test_runtime_name_is_redacted_and_synthetic_nickname_is_preserved() -> None:
    assert replace_private_names("晚安示例姓名甲", ("示例姓名甲",)) == "晚安[NAME]"
    assert classification_text("晚安小团子", ("小团子",)) == "晚安[NICKNAME]"


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("晚安[NICKNAME]", CandidateCategory.GREETING),
        ("谢谢你", CandidateCategory.COURTESY),
        ("我准备吃饭啦", CandidateCategory.MEALS),
        ("我要睡觉啦", CandidateCategory.SLEEP),
        ("安安", CandidateCategory.SLEEP),
        ("安安\uff01", CandidateCategory.SLEEP),
        ("今天天气真好", CandidateCategory.WEATHER),
        ("我去收拾房间啦", CandidateCategory.HOUSEHOLD),
        ("我去买日用品", CandidateCategory.SHOPPING),
        ("我准备学习啦", CandidateCategory.WORK_STUDY),
        ("今天好开心呀", CandidateCategory.GENERAL_EMOTION),
        ("我很想你呀", CandidateCategory.GENTLE_AFFECTION),
        ("嗯嗯", CandidateCategory.SHORT_REPLY),
        ("😊", CandidateCategory.EMOJI),
    ],
)
def test_approved_categories_use_complete_safe_phrases(
    text: str,
    category: CandidateCategory,
) -> None:
    assert allowed_category(text) is category


def test_unknown_or_extra_content_is_not_allowed() -> None:
    assert allowed_category("合成无法分类长句") is None
    assert allowed_category("吃饭以及无法解释的附加内容") is None
    assert allowed_category("我准备吃饭啦合成成人暗示内容") is None


def test_only_short_replies_and_emoji_receive_repeat_keys() -> None:
    assert short_repeat_key("嗯嗯\uff01", CandidateCategory.SHORT_REPLY) == "嗯嗯"
    assert short_repeat_key("😊\uff01", CandidateCategory.EMOJI) == "😊"
    assert short_repeat_key("我准备吃饭啦", CandidateCategory.MEALS) is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "要不要我帮你再写几句更具体的聊天回复\uff1f",
            CandidateRejectReason.NON_ORIGINAL_OR_UNSUITABLE,
        ),
        (
            "@示例用户: 这是一段转发的长文",
            CandidateRejectReason.NON_ORIGINAL_OR_UNSUITABLE,
        ),
        (
            "https://example.test/private",
            CandidateRejectReason.NON_ORIGINAL_OR_UNSUITABLE,
        ),
        ("[NAME]不是垃圾", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("19号或者20号\uff1f", CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE),
        ("100块的饭", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("把他弄死", CandidateRejectReason.VIOLENT_OR_DEGRADING),
        ("叫爸爸", CandidateRejectReason.SEXUAL_OR_SUGGESTIVE),
        ("夹着一肚子屎", CandidateRejectReason.BODILY_PRIVATE),
    ],
)
def test_confirmed_review_policy_rejects_unfit_style_material(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("\uff01", CandidateCategory.SHORT_REPLY),
        ("..", CandidateCategory.SHORT_REPLY),
        ("just asking\uff01", CandidateCategory.SHORT_REPLY),
        ("do u love me\uff1f", CandidateCategory.GENTLE_AFFECTION),
        ("mua\uff01", CandidateCategory.GENTLE_AFFECTION),
        ("不吃葱", CandidateCategory.MEALS),
        ("不吃谢谢\uff01", CandidateCategory.MEALS),
        ("我他妈紧张死了", CandidateCategory.GENERAL_EMOTION),
    ],
)
def test_confirmed_review_policy_allows_only_tightly_scoped_daily_phrases(
    text: str,
    category: CandidateCategory,
) -> None:
    assert allowed_category(text) is category


def test_punctuation_replies_receive_a_repeat_key_for_limited_sampling() -> None:
    assert short_repeat_key("\uff01", CandidateCategory.SHORT_REPLY) == "!"
    assert short_repeat_key("..", CandidateCategory.SHORT_REPLY) == ".."


def test_violent_rule_does_not_reject_unrelated_compound_words() -> None:
    assert hard_rejection_reason("打开杀菌灯") is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("爱人在我这里永远最重要", CandidateRejectReason.PERSONAL_BELIEF_OR_COMMITMENT),
        ("你要做贤妻良母", CandidateRejectReason.GENDERED_OR_ROLE_LABEL),
        ("让同学帮忙代签", CandidateRejectReason.MISCONDUCT),
        ("感觉自己长得像某个人", CandidateRejectReason.APPEARANCE_OR_BODY_PRIVATE),
        ("你没和别人说吧", CandidateRejectReason.PRIVATE_EVENT),
        ("一直在冷暴力我", CandidateRejectReason.RELATIONSHIP_CONFLICT),
        ("今天发烧了", CandidateRejectReason.HEALTH_PRIVATE),
        ("账号 1799992492", CandidateRejectReason.IDENTITY_OR_CONTACT),
        ("还有红包", CandidateRejectReason.FINANCIAL_OR_CREDENTIAL),
        ("adult video", CandidateRejectReason.SEXUAL_OR_SUGGESTIVE),
    ],
)
def test_second_review_policy_rejects_new_private_or_unsuitable_categories(
    text: str,
    reason: CandidateRejectReason,
) -> None:
    assert hard_rejection_reason(text) is reason


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("gpt牛逼", CandidateCategory.GENERAL_EMOTION),
        ("记得早点睡", CandidateCategory.SLEEP),
        ("少喝点吧", CandidateCategory.MEALS),
        ("我想吃披萨", CandidateCategory.MEALS),
        ("厉害👍", CandidateCategory.GENERAL_EMOTION),
        ("hello", CandidateCategory.SHORT_REPLY),
        ("maybe\uff1f", CandidateCategory.SHORT_REPLY),
        ("just joking", CandidateCategory.SHORT_REPLY),
        ("叫哥哥\uff01", CandidateCategory.GENTLE_AFFECTION),
    ],
)
def test_second_review_policy_allows_tightly_scoped_safe_daily_phrases(
    text: str,
    category: CandidateCategory,
) -> None:
    assert allowed_category(text) is category


@pytest.mark.parametrize(
    "text",
    [
        "合成想草",
        "合成想操你",
        "合成想艹你",
        "合成日你",
        "合成和你上床",
        "合成想法你",
    ],
)
def test_direct_sexual_variants_are_always_hard_rejected(text: str) -> None:
    assert hard_rejection_reason(text) is CandidateRejectReason.SEXUAL_OR_SUGGESTIVE


@pytest.mark.parametrize(
    "placeholder",
    [f"[{label}]" for label in redaction._LABEL_ORDER],
)
def test_all_redaction_placeholders_are_hard_rejected(placeholder: str) -> None:
    reason = hard_rejection_reason(f"合成内容 {placeholder}")
    assert reason is CandidateRejectReason.IDENTITY_OR_CONTACT


def test_redaction_label_set_has_twelve_labels() -> None:
    assert len(redaction._LABEL_ORDER) == 12


def test_health_terms_do_not_reject_safe_work_checks() -> None:
    assert hard_rejection_reason("检查作业") is None
