from __future__ import annotations

import re
import unicodedata
from datetime import date

from companion.privacy.candidate_models import CandidateCategory, CandidateRejectReason
from companion.privacy.redaction import _LABEL_ORDER, _STRUCTURED_PATTERNS


def normalize_text(text: str) -> str:
    """Normalize compatibility characters and collapse whitespace for matching."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


_STRUCTURED_BY_LABEL = dict(_STRUCTURED_PATTERNS)
_NATIONAL_ID_TOKEN = re.compile(r"(?<!\d)(\d{17}[\dXx])(?![0-9A-Za-z_])")
_NATIONAL_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_NATIONAL_ID_CHECK_VALUES = "10X98765432"


def _is_valid_mainland_national_id(value: str) -> bool:
    if not re.fullmatch(r"\d{17}[\dXx]", value):
        return False

    try:
        birth_date = date(int(value[6:10]), int(value[10:12]), int(value[12:14]))
    except ValueError:
        return False
    if birth_date > date.today():
        return False

    checksum = sum(
        int(digit) * weight for digit, weight in zip(value[:17], _NATIONAL_ID_WEIGHTS, strict=True)
    )
    return _NATIONAL_ID_CHECK_VALUES[checksum % 11] == value[-1].upper()


def _contains_valid_mainland_national_id(text: str) -> bool:
    return any(
        _is_valid_mainland_national_id(match.group(1))
        for match in _NATIONAL_ID_TOKEN.finditer(text)
    )


_HARD_REJECT_PATTERNS: tuple[tuple[CandidateRejectReason, tuple[re.Pattern[str], ...]], ...] = (
    (
        CandidateRejectReason.NON_ORIGINAL_OR_UNSUITABLE,
        (
            re.compile(r"^@[^:\uff1a\s]+\s*[:\uff1a]"),
            re.compile(r"https?://|www\."),
            re.compile(r"要不要我帮你(?:再)?写.{0,20}(?:回复|文案)"),
            re.compile(r"你随便挑一个领域.{0,40}研究生层级的概念"),
        ),
    ),
    (
        CandidateRejectReason.PERSONAL_BELIEF_OR_COMMITMENT,
        (
            re.compile(r"(?:爱人|对象|伴侣).{0,12}永远.{0,8}最重要"),
            re.compile(r"(?:我自己|爱情|友情|亲情).{0,12}(?:第一位|最重要)"),
        ),
    ),
    (
        CandidateRejectReason.GENDERED_OR_ROLE_LABEL,
        (re.compile(r"贤妻良母|good\s+girl|坏女人|好女人", re.IGNORECASE),),
    ),
    (
        CandidateRejectReason.MISCONDUCT,
        (re.compile(r"逃课|代签|作弊|盗版|刷校园跑|规避规则"),),
    ),
    (
        CandidateRejectReason.APPEARANCE_OR_BODY_PRIVATE,
        (
            re.compile(r"(?:长得|看着).{0,8}像.{1,12}"),
            re.compile(r"修图|P图|p图|身材|露腿|外貌比较"),
        ),
    ),
    (
        CandidateRejectReason.SEXUAL_OR_SUGGESTIVE,
        (
            re.compile(
                r"adult\s+video|成人暗示|性生活|性行为|色情|裸聊|暧昧|涩涩|瑟瑟|(?:性|杏)[\s._-]?(?:生活|行为|行為)",
                re.IGNORECASE,
            ),
            re.compile(r"想(?:草|操|艹)(?:你)?|(?:草|操|艹|日)你|(?:和|跟)你上床|想法你"),
            re.compile(r"(?:想|要|去|来|做|玩|搞|试).{0,8}(?:亲亲|摸摸|开车|车车|那种事|那件事)"),
            re.compile(r"(?:叫|喊)(?:我)?爸爸|主人|奴隶"),
        ),
    ),
    (
        CandidateRejectReason.VIOLENT_OR_DEGRADING,
        (
            re.compile(r"杀(?=$|[!\uff01。…]|你|他|她|它|人|掉|了|死)"),
            re.compile(r"(?:弄死|砍死|打死|暗杀|死定了|砸死)(?:你|他|她|它|人|掉)?"),
        ),
    ),
    (
        CandidateRejectReason.BODILY_PRIVATE,
        (re.compile(r"排泄|拉屎|拉尿|尿尿|屎|粪|生理反应|性器官"),),
    ),
    (
        CandidateRejectReason.RELATIONSHIP_CONFLICT,
        (re.compile(r"争吵|吵架|分手|冷战|冷暴力|矛盾|闹别扭|拉黑|嫉妒|威胁|控制|胁迫"),),
    ),
    (
        CandidateRejectReason.HEALTH_PRIVATE,
        (
            re.compile(
                r"医院|诊断|病历|复查|(?:身体|健康|去|做)检查|(?:牙齿|视力|去)矫正|手术|病情|用药|怀孕|生育|产检|心理健康|心理咨询|抑郁|焦虑|生病|发烧|发热|体温|口腔溃疡|皮肤病|症状|吃药|疾病"
            ),
        ),
    ),
    (
        CandidateRejectReason.FAMILY_OR_THIRD_PARTY_PRIVATE,
        (re.compile(r"家庭秘密|家人隐私|父母隐私|亲属隐私|孩子隐私|第三方隐私"),),
    ),
    (
        CandidateRejectReason.IDENTITY_OR_CONTACT,
        (
            _STRUCTURED_BY_LABEL["EMAIL"],
            _STRUCTURED_BY_LABEL["PHONE"],
            _STRUCTURED_BY_LABEL["IP_ADDRESS"],
            re.compile(r"(?<!\d)\d{10,17}(?!\d)"),
            re.compile(
                r"\[(?:" + "|".join(map(re.escape, _LABEL_ORDER)) + r")\]"
            ),
            re.compile(r"身份证|证件|手机号|电话|邮箱|联系方式|住址"),
        ),
    ),
    (
        CandidateRejectReason.FINANCIAL_OR_CREDENTIAL,
        (
            _STRUCTURED_BY_LABEL["BANK_CARD"],
            re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:块|元|rmb|RMB|人民币)"),
            re.compile(
                r"银行卡|卡号|余额|转账|红包|密码|凭证|支付密码|登录密码|验证码|口令|密钥|账号|钱|金额|工资|收入|借钱|还钱|债务|房贷|存款|资产"
            ),
        ),
    ),
    (
        CandidateRejectReason.PRECISE_LOCATION_OR_SCHEDULE,
        (
            re.compile(
                r"酒店(?:房间|房号)?|(?:精确|具体|详细)(?:时间|位置|地点)|房间号|单位|实时位置|行程|地址|定位"
            ),
            re.compile(r"(?:今天|明天|今晚|周[一二三四五六日天])?\d{1,2}:\d{2}"),
            re.compile(
                r"(?:今天|明天|后天|周[一二三四五六日天])?"
                r"(?:凌晨|早上|上午|中午|下午|傍晚|晚上)?"
                r"(?:[一二三四五六七八九十两]|\d{1,2})点"
                r"(?:(?:[一二三四五六七八九十两]|\d{1,2})分|半)?"
            ),
            re.compile(r"(?<!\d)\d{1,2}号(?!\d)"),
        ),
    ),
    (
        CandidateRejectReason.PRIVATE_EVENT,
        (
            re.compile(r"私人(?:事件|纪念|经历)|秘密事件|私密经历|秘密|隐私|不要告诉|保密"),
            re.compile(r"(?:你)?没和别人说吧"),
        ),
    ),
)

_FULL_MESSAGE_ALLOW_PATTERNS: tuple[tuple[CandidateCategory, tuple[re.Pattern[str], ...]], ...] = (
    (
        CandidateCategory.GREETING,
        (re.compile(r"(?:早安|早上好|午安|下午好|晚安|你好|嗨)(?:\[NAME\]|\[NICKNAME\])?[!。]?"),),
    ),
    (
        CandidateCategory.COURTESY,
        (re.compile(r"(?:谢谢(?:你)?|不客气|辛苦了|麻烦你了)[!。]?"),),
    ),
    (
        CandidateCategory.MEALS,
        (
            re.compile(r"(?:我)?(?:准备|要|去)?吃(?:饭|早餐|午饭|晚饭)(?:啦|了|呀)?[!。]?"),
            re.compile(r"(?:我)?不吃葱[!\uff01。]?"),
            re.compile(r"不吃谢谢[!\uff01。]?"),
            re.compile(
                r"(?:我)?想吃(?:披萨|面|饭|蛋糕|水果|排骨|番茄炒蛋)[\U0001F300-\U0001FAFF]?[!\uff01。]?"
            ),
            re.compile(r"(?:少喝点吧|一天最多喝一杯)[!\uff01。]?"),
        ),
    ),
    (
        CandidateCategory.SLEEP,
        (
            re.compile(r"(?:我)?(?:要|去|准备)?睡觉(?:啦|了|呀)?[!。]?"),
            re.compile(r"记得早点睡[!\uff01。]?"),
            re.compile(r"安安[!\uff01。]?"),
        ),
    ),
    (
        CandidateCategory.WEATHER,
        (re.compile(r"今天天气(?:真)?(?:好|不错|晴朗|凉快|暖和)[!。]?"),),
    ),
    (
        CandidateCategory.HOUSEHOLD,
        (re.compile(r"(?:我)?(?:去|准备)?(?:收拾房间|做家务|洗衣服|打扫卫生)(?:啦|了|呀)?[!。]?"),),
    ),
    (
        CandidateCategory.SHOPPING,
        (re.compile(r"(?:我)?(?:去|准备)?买(?:日用品|菜|东西)(?:啦|了|呀)?[!。]?"),),
    ),
    (
        CandidateCategory.WORK_STUDY,
        (re.compile(r"(?:我)?(?:去|准备)?(?:学习|工作|写作业)(?:啦|了|呀)?[!。]?"),),
    ),
    (
        CandidateCategory.GENERAL_EMOTION,
        (
            re.compile(r"今天(?:好|真)?(?:开心|高兴|快乐)(?:呀|啊|呢)?[!。]?"),
            re.compile(r"我(?:他妈|特么)?(?:紧张|累|冷|困)死了[!\uff01。]?"),
            re.compile(r"(?:ai|gpt)真?牛逼[!\uff01。]?", re.IGNORECASE),
            re.compile(r"厉害[\U0001F300-\U0001FAFF]?[!\uff01。]?"),
        ),
    ),
    (
        CandidateCategory.GENTLE_AFFECTION,
        (
            re.compile(
                r"(?:\[NAME\]|\[NICKNAME\])?(?:我)?(?:很|好)?(?:想你|喜欢你|爱你)(?:呀|啊|呢)?[!。]?"
            ),
            re.compile(r"do\s+u\s+love\s+me[?\uff1f!\uff01]?", re.IGNORECASE),
            re.compile(r"mua[!\uff01。]?", re.IGNORECASE),
            re.compile(r"叫(?:哥哥|姐姐)[!\uff01。]?"),
        ),
    ),
    (
        CandidateCategory.SHORT_REPLY,
        (
            re.compile(r"(?:嗯+|好(?:的)?|是(?:的)?|行|知道了|收到)[!。]?"),
            re.compile(r"just\s+asking[!\uff01。]?", re.IGNORECASE),
            re.compile(
                r"(?:hello|hi|maybe|just\s+joking|no\s+why)[?\uff1f!\uff01。]?", re.IGNORECASE
            ),
            re.compile(r"[!\uff01?\uff1f。.…]+"),
        ),
    ),
    (
        CandidateCategory.EMOJI,
        (re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]+"),),
    ),
)

_REPEAT_KEY_TRIM = re.compile(r"[\s!?,.。…~]+")


def hard_rejection_reason(text: str) -> CandidateRejectReason | None:
    normalized = normalize_text(text)
    for reason, patterns in _HARD_REJECT_PATTERNS:
        if (
            reason is CandidateRejectReason.IDENTITY_OR_CONTACT
            and _contains_valid_mainland_national_id(normalized)
        ):
            return reason
        if any(pattern.search(normalized) for pattern in patterns):
            return reason
    return None


def replace_private_names(text: str, true_name_terms: tuple[str, ...]) -> str:
    result = text
    for term in sorted(true_name_terms, key=len, reverse=True):
        result = result.replace(term, "[NAME]")
    return result


def classification_text(text: str, approved_nicknames: tuple[str, ...]) -> str:
    result = text
    for nickname in sorted(approved_nicknames, key=len, reverse=True):
        result = result.replace(nickname, "[NICKNAME]")
    return result


def allowed_category(text: str) -> CandidateCategory | None:
    normalized = normalize_text(text)
    for category, patterns in _FULL_MESSAGE_ALLOW_PATTERNS:
        if any(pattern.fullmatch(normalized) for pattern in patterns):
            return category
    return None


def short_repeat_key(text: str, category: CandidateCategory) -> str | None:
    if category not in {CandidateCategory.SHORT_REPLY, CandidateCategory.EMOJI}:
        return None
    normalized = normalize_text(text)
    if category is CandidateCategory.SHORT_REPLY and re.fullmatch(
        r"[!\uff01?\uff1f。.…]+", normalized
    ):
        return normalized
    key = _REPEAT_KEY_TRIM.sub("", normalized)
    return key or None
