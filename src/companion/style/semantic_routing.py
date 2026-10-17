import unicodedata

from companion.privacy import CandidateCategory

_PHRASES = {
    CandidateCategory.GREETING: ("早上好", "早安", "晚上好", "你好", "哈喽", "嗨", "在吗"),
    CandidateCategory.COURTESY: ("谢谢", "感谢", "辛苦", "不客气", "对不起", "抱歉", "麻烦"),
    CandidateCategory.WEATHER: ("天气", "下雨", "降温", "带伞", "晴天", "风大", "好冷", "好热"),
    CandidateCategory.HOUSEHOLD: ("洗澡", "洗衣", "收拾", "打扫", "做饭", "洗碗", "家务", "拖地"),
    CandidateCategory.SHOPPING: (
        "下单",
        "快递",
        "外卖",
        "淘宝",
        "拼多多",
        "多少钱",
        "便宜",
        "好贵",
        "购物",
    ),
    CandidateCategory.WORK_STUDY: (
        "上课",
        "下课",
        "上班",
        "下班",
        "作业",
        "学习",
        "考试",
        "开会",
        "学校",
        "老师",
    ),
}


def _normalized(text: str) -> str:
    return "".join(
        character
        for character in text
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def derive_calibrated_category(text: str) -> CandidateCategory | None:
    compact = _normalized(text)
    matches = [
        category
        for category, phrases in _PHRASES.items()
        if any(phrase in compact for phrase in phrases)
    ]
    return matches[0] if len(matches) == 1 else None
