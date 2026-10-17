from collections.abc import Sequence
from typing import NamedTuple

from companion.contracts import StyleIntent
from companion.privacy import CandidateCategory
from companion.style.errors import StyleValidationError

REOPEN_SECONDS = 21_600


class StyleMatch(NamedTuple):
    category: CandidateCategory
    intent: StyleIntent


_GREETING_UTTERANCES = ("hello", "你好", "哈喽", "嗨", "在吗", "早上好", "早安", "晚上好")
_EXACT_INTENT_UTTERANCES = {
    StyleMatch(CandidateCategory.GREETING, StyleIntent.MORNING_GREETING): (
        "早上好",
        "早安",
    ),
    StyleMatch(CandidateCategory.COURTESY, StyleIntent.COURTESY_THANKS): (
        "我帮你弄好了",
        "给你弄好了",
        "我给你发了",
        "帮你做好了",
    ),
    StyleMatch(CandidateCategory.COURTESY, StyleIntent.COURTESY_APOLOGY): (
        "你打断我了",
        "别打断我",
        "刚才打断我",
        "你刚才打断我了",
    ),
    StyleMatch(CandidateCategory.WEATHER, StyleIntent.WEATHER_COLD_REMINDER): (
        "今天降温",
        "今天降温了",
        "外面好冷",
        "变冷了",
        "天气转冷",
    ),
    StyleMatch(CandidateCategory.WORK_STUDY, StyleIntent.WORK_STUDY_COMPLETED): (
        "我下课了",
        "我下班了",
        "下班了",
        "作业写完了",
        "学习完成了",
        "工作做完了",
    ),
    StyleMatch(CandidateCategory.MEALS, StyleIntent.MISSED_MEAL_CARE): (
        "中午忙得忘记吃饭了",
        "忙得忘记吃饭了",
    ),
    StyleMatch(CandidateCategory.MEALS, StyleIntent.NOT_EATEN_YET_CARE): (
        "我还没吃饭",
        "还没吃饭",
    ),
    StyleMatch(
        CandidateCategory.HOUSEHOLD,
        StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE,
    ): (
        "房间还没整理",
        "房间还没收拾",
    ),
    StyleMatch(CandidateCategory.SHOPPING, StyleIntent.SHOPPING_COMPARISON_REQUEST): (
        "这两件我不知道选哪个",
        "这两个不知道选哪个",
    ),
    StyleMatch(CandidateCategory.WEATHER, StyleIntent.RAIN_REMINDER): (
        "外面好像要下雨",
        "外面要下雨了",
    ),
    StyleMatch(
        CandidateCategory.HOUSEHOLD,
        StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
    ): (
        "家务刚做完",
        "家务做完了",
    ),
    StyleMatch(
        CandidateCategory.SHOPPING,
        StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
    ): (
        "我又想买东西了",
        "又想买东西了",
    ),
    StyleMatch(CandidateCategory.GENTLE_AFFECTION, StyleIntent.GOING_OUT_CARE): (
        "我准备出门了",
        "马上出门了",
    ),
    StyleMatch(CandidateCategory.WORK_STUDY, StyleIntent.WORK_FATIGUE_CARE): (
        "今天工作有点累",
        "工作有点累",
    ),
    StyleMatch(CandidateCategory.GENERAL_EMOTION, StyleIntent.POSITIVE_EMOTION_SHARE): (
        "我今天心情挺好的",
        "今天心情很好",
    ),
    StyleMatch(CandidateCategory.GENERAL_EMOTION, StyleIntent.NEGATIVE_EMOTION_CHECK_IN): (
        "我有点失落",
        "有点失落",
    ),
    StyleMatch(CandidateCategory.GENTLE_AFFECTION, StyleIntent.MISSED_CONTACT_AFFECTION): (
        "今天都没怎么见到你",
        "今天没怎么见到你",
    ),
    StyleMatch(CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL): (
        "我要睡觉",
        "我要睡觉了",
        "我睡了",
        "准备睡觉",
        "我准备休息了",
        "准备休息了",
        "困困",
        "晚安",
    ),
    StyleMatch(CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER): (
        "我还不睡",
        "我又熬夜",
        "我又熬夜了",
        "我在熬夜",
        "今晚又熬夜了",
        "今晚熬夜了",
    ),
}

_GENERAL_CATEGORY_UTTERANCES = {
    CandidateCategory.MEALS: (
        "准备吃饭",
        "去吃饭",
        "中午忙得忘记吃饭了",
        "忙得忘记吃饭了",
    ),
    CandidateCategory.GENTLE_AFFECTION: (
        "想你了",
        "我想你",
        "今天都没怎么见到你",
        "今天没怎么见到你",
    ),
    CandidateCategory.GENERAL_EMOTION: (
        "有点难过",
        "我有点难过",
        "心情不好",
        "有点伤心",
        "我今天心情挺好的",
        "今天心情很好",
        "我有点失落",
        "有点失落",
    ),
    CandidateCategory.SHOPPING: (
        "快递到了",
        "这个快递到了",
        "快递来了",
        "准备下单",
        "这两件我不知道选哪个",
        "这两个不知道选哪个",
    ),
    CandidateCategory.HOUSEHOLD: (
        "打扫完房间",
        "刚打扫完房间",
        "收拾完房间",
        "洗完衣服",
        "房间还没整理",
        "房间还没收拾",
    ),
    CandidateCategory.WORK_STUDY: ("今天工作有点累", "工作有点累"),
}

_CONTEXT_HELP_REQUEST = "可以帮我整理一下吗"
_INCOMPLETE_HELP_COMPLETION = "弄好了"
_APPROVED_ENDING_MARKS = "~～"  # noqa: RUF001


def _compact(text: str) -> str:
    return text.strip().rstrip(_APPROVED_ENDING_MARKS)


def _has_valid_input(text: object, recent_context: object, idle_seconds: object) -> bool:
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2_000:
        return False
    if (
        not isinstance(recent_context, Sequence)
        or isinstance(recent_context, str)
        or len(recent_context) > 4
    ):
        return False
    if any(
        not isinstance(context_item, str) or not 1 <= len(context_item.strip()) <= 2_000
        for context_item in recent_context
    ):
        return False
    return idle_seconds is None or (
        isinstance(idle_seconds, int) and not isinstance(idle_seconds, bool) and idle_seconds >= 0
    )


def _current_matches(text: str, idle_seconds: int | None) -> list[StyleMatch]:
    matches: list[StyleMatch] = []
    reopens_conversation = False
    if text in _GREETING_UTTERANCES:
        reopens_conversation = idle_seconds is not None and idle_seconds >= REOPEN_SECONDS
        greeting_intent = (
            StyleIntent.CONVERSATION_REOPEN
            if reopens_conversation
            else StyleIntent.CATEGORY_DEFAULT
        )
        matches.append(StyleMatch(CandidateCategory.GREETING, greeting_intent))
    for match, utterances in _EXACT_INTENT_UTTERANCES.items():
        if text in utterances and not (
            reopens_conversation and match.intent is StyleIntent.MORNING_GREETING
        ):
            matches.append(match)
    for category, utterances in _GENERAL_CATEGORY_UTTERANCES.items():
        if text in utterances:
            matches.append(StyleMatch(category, StyleIntent.CATEGORY_DEFAULT))
    return matches


def _context_assisted_match(text: str, recent_context: Sequence[str]) -> StyleMatch | None:
    if text != _INCOMPLETE_HELP_COMPLETION:
        return None
    if any(_compact(context_item) == _CONTEXT_HELP_REQUEST for context_item in recent_context):
        return StyleMatch(CandidateCategory.COURTESY, StyleIntent.COURTESY_THANKS)
    return None


def _resolve_matches(matches: list[StyleMatch]) -> StyleMatch | None:
    categories = {match.category for match in matches}
    exact_matches = [match for match in matches if match.intent is not StyleIntent.CATEGORY_DEFAULT]
    if len(categories) != 1 or len(exact_matches) > 1:
        return None
    if exact_matches:
        return exact_matches[0]
    return matches[0] if matches else None


def classify_style_reply(
    text: str,
    recent_context: Sequence[str],
    idle_seconds: int | None,
) -> StyleMatch | None:
    """Classify only high-confidence, approved runtime style intents."""
    if not _has_valid_input(text, recent_context, idle_seconds):
        del text, recent_context, idle_seconds
        raise StyleValidationError(
            "invalid_style_guidance_input", "style guidance input is invalid"
        ) from None

    compact_text = _compact(text)
    matches = _current_matches(compact_text, idle_seconds)
    context_match = _context_assisted_match(compact_text, recent_context)
    if context_match is not None:
        matches.append(context_match)
    return _resolve_matches(matches)
