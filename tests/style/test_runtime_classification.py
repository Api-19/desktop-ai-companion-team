# ruff: noqa: RUF001

import pytest

from companion.contracts import StyleIntent
from companion.privacy import CandidateCategory
from companion.style.errors import StyleValidationError
from companion.style.runtime_classification import StyleMatch, classify_style_reply


@pytest.mark.parametrize(
    ("text", "context", "idle_seconds", "category", "intent"),
    [
        ("早上好", [], 0, CandidateCategory.GREETING, StyleIntent.MORNING_GREETING),
        ("早安", [], 0, CandidateCategory.GREETING, StyleIntent.MORNING_GREETING),
        ("hello", [], 21600, CandidateCategory.GREETING, StyleIntent.CONVERSATION_REOPEN),
        ("我帮你弄好了", [], 0, CandidateCategory.COURTESY, StyleIntent.COURTESY_THANKS),
        ("你刚才打断我了", [], 0, CandidateCategory.COURTESY, StyleIntent.COURTESY_APOLOGY),
        (
            "今天降温了",
            [],
            0,
            CandidateCategory.WEATHER,
            StyleIntent.WEATHER_COLD_REMINDER,
        ),
        ("我下课了", [], 0, CandidateCategory.WORK_STUDY, StyleIntent.WORK_STUDY_COMPLETED),
        ("我要睡觉了", [], 0, CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL),
        ("我又熬夜了", [], 0, CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER),
    ],
)
def test_classifies_only_approved_precise_intents(
    text: str,
    context: list[str],
    idle_seconds: int,
    category: CandidateCategory,
    intent: StyleIntent,
) -> None:
    assert classify_style_reply(text, context, idle_seconds) == StyleMatch(category, intent)


def test_reopen_requires_six_hours() -> None:
    assert classify_style_reply("hello", [], 21599) == StyleMatch(
        CandidateCategory.GREETING,
        StyleIntent.CATEGORY_DEFAULT,
    )


@pytest.mark.parametrize("text", ("早上好", "早安"))
def test_reopen_threshold_overrides_morning_greeting(text: str) -> None:
    assert classify_style_reply(text, [], 21_600) == StyleMatch(
        CandidateCategory.GREETING,
        StyleIntent.CONVERSATION_REOPEN,
    )
    assert classify_style_reply(text, [], 21_599) == StyleMatch(
        CandidateCategory.GREETING,
        StyleIntent.MORNING_GREETING,
    )


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("想你了", CandidateCategory.GENTLE_AFFECTION),
        ("我有点难过", CandidateCategory.GENERAL_EMOTION),
        ("这个快递到了", CandidateCategory.SHOPPING),
        ("刚打扫完房间", CandidateCategory.HOUSEHOLD),
    ],
)
def test_reliable_general_category_uses_default_intent(
    text: str, category: CandidateCategory
) -> None:
    assert classify_style_reply(text, [], None) == StyleMatch(
        category,
        StyleIntent.CATEGORY_DEFAULT,
    )


@pytest.mark.parametrize(
    ("text", "category", "intent"),
    [
        (
            "中午忙得忘记吃饭了",
            CandidateCategory.MEALS,
            StyleIntent.MISSED_MEAL_CARE,
        ),
        (
            "忙得忘记吃饭了",
            CandidateCategory.MEALS,
            StyleIntent.MISSED_MEAL_CARE,
        ),
        ("我还没吃饭", CandidateCategory.MEALS, StyleIntent.NOT_EATEN_YET_CARE),
        ("还没吃饭", CandidateCategory.MEALS, StyleIntent.NOT_EATEN_YET_CARE),
        ("我准备休息了", CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL),
        ("准备休息了", CandidateCategory.SLEEP, StyleIntent.SLEEP_FAREWELL),
        ("今晚又熬夜了", CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER),
        ("今晚熬夜了", CandidateCategory.SLEEP, StyleIntent.SLEEP_REMINDER),
        (
            "外面好像要下雨",
            CandidateCategory.WEATHER,
            StyleIntent.RAIN_REMINDER,
        ),
        (
            "外面要下雨了",
            CandidateCategory.WEATHER,
            StyleIntent.RAIN_REMINDER,
        ),
        (
            "家务刚做完",
            CandidateCategory.HOUSEHOLD,
            StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
        ),
        (
            "家务做完了",
            CandidateCategory.HOUSEHOLD,
            StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
        ),
        (
            "房间还没整理",
            CandidateCategory.HOUSEHOLD,
            StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE,
        ),
        (
            "房间还没收拾",
            CandidateCategory.HOUSEHOLD,
            StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE,
        ),
        (
            "这两件我不知道选哪个",
            CandidateCategory.SHOPPING,
            StyleIntent.SHOPPING_COMPARISON_REQUEST,
        ),
        (
            "这两个不知道选哪个",
            CandidateCategory.SHOPPING,
            StyleIntent.SHOPPING_COMPARISON_REQUEST,
        ),
        (
            "我又想买东西了",
            CandidateCategory.SHOPPING,
            StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
        ),
        (
            "又想买东西了",
            CandidateCategory.SHOPPING,
            StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
        ),
        (
            "今天工作有点累",
            CandidateCategory.WORK_STUDY,
            StyleIntent.WORK_FATIGUE_CARE,
        ),
        (
            "工作有点累",
            CandidateCategory.WORK_STUDY,
            StyleIntent.WORK_FATIGUE_CARE,
        ),
        (
            "我今天心情挺好的",
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.POSITIVE_EMOTION_SHARE,
        ),
        (
            "今天心情很好",
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.POSITIVE_EMOTION_SHARE,
        ),
        (
            "我有点失落",
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.NEGATIVE_EMOTION_CHECK_IN,
        ),
        (
            "有点失落",
            CandidateCategory.GENERAL_EMOTION,
            StyleIntent.NEGATIVE_EMOTION_CHECK_IN,
        ),
        (
            "今天都没怎么见到你",
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.MISSED_CONTACT_AFFECTION,
        ),
        (
            "今天没怎么见到你",
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.MISSED_CONTACT_AFFECTION,
        ),
        (
            "我准备出门了",
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.GOING_OUT_CARE,
        ),
        (
            "马上出门了",
            CandidateCategory.GENTLE_AFFECTION,
            StyleIntent.GOING_OUT_CARE,
        ),
    ],
)
def test_fixed_runtime_semantics_use_approved_category_and_intent(
    text: str,
    category: CandidateCategory,
    intent: StyleIntent,
) -> None:
    assert classify_style_reply(text, [], None) == StyleMatch(category, intent)


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("早上好", StyleIntent.MORNING_GREETING),
        ("中午忙得忘记吃饭了", StyleIntent.MISSED_MEAL_CARE),
        ("房间还没整理", StyleIntent.UNFINISHED_HOUSEHOLD_REASSURANCE),
        ("这两件我不知道选哪个", StyleIntent.SHOPPING_COMPARISON_REQUEST),
        ("今天工作有点累", StyleIntent.WORK_FATIGUE_CARE),
        ("我今天心情挺好的", StyleIntent.POSITIVE_EMOTION_SHARE),
        ("我有点失落", StyleIntent.NEGATIVE_EMOTION_CHECK_IN),
        ("今天都没怎么见到你", StyleIntent.MISSED_CONTACT_AFFECTION),
        ("我还没吃饭", StyleIntent.NOT_EATEN_YET_CARE),
        ("今晚又熬夜了", StyleIntent.SLEEP_REMINDER),
        ("天气转冷", StyleIntent.WEATHER_COLD_REMINDER),
        ("外面好像要下雨", StyleIntent.RAIN_REMINDER),
        ("家务刚做完", StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT),
        ("我又想买东西了", StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT),
        ("我准备出门了", StyleIntent.GOING_OUT_CARE),
    ],
)
@pytest.mark.parametrize("ending_mark", ("~", "～"))
def test_fixed_precise_intents_allow_only_approved_ending_display_marks(
    text: str,
    intent: StyleIntent,
    ending_mark: str,
) -> None:
    match = classify_style_reply(f"{text}{ending_mark}", [], 0)

    assert match is not None
    assert match.intent is intent


@pytest.mark.parametrize(
    "text",
    [
        "早上，好",
        "中午忙得，忘记吃饭了",
        "房间还没 整理",
        "这两件我不知道？选哪个",
        "今天工作，有点累",
        "我今天心情挺好的？",
        "我有点失落？",
        "今天都没怎么 见到你",
        "我还没吃饭？",
        "我还没 吃饭",
        "今晚又熬夜了？",
        "天气，转冷",
        "外面好像，要下雨",
        "家务，刚做完",
        "我又想买东西了？",
        "我准备 出门了",
    ],
)
def test_fixed_precise_intents_reject_questions_internal_punctuation_and_spaces(
    text: str,
) -> None:
    assert classify_style_reply(text, [], 0) is None


@pytest.mark.parametrize(
    "text",
    [
        "中午忙吗",
        "你准备休息了吗",
        "昨晚又熬夜了",
        "带伞了吗",
        "家务好多",
        "房间真漂亮",
        "我买完了",
        "你想买东西吗",
        "我运动有点累",
        "你开心吗",
        "你失落吗",
        "你今天见到她了吗",
        "你出门了吗",
    ],
)
def test_adjacent_fixed_semantics_remain_no_match(text: str) -> None:
    assert classify_style_reply(text, [], None) is None


@pytest.mark.parametrize(
    "text",
    [
        "合成无关内容",
        "下班了但是外面好冷",
        "我要睡觉了，我又熬夜了",
    ],
)
def test_unknown_or_conflicting_categories_return_no_match(text: str) -> None:
    assert classify_style_reply(text, [], None) is None


def test_prior_context_category_does_not_override_explicit_current_category() -> None:
    assert classify_style_reply("我下课了", ["外面好冷"], None) == StyleMatch(
        CandidateCategory.WORK_STUDY,
        StyleIntent.WORK_STUDY_COMPLETED,
    )


def test_context_can_complete_only_the_approved_help_completion_case() -> None:
    assert classify_style_reply("弄好了", ["可以帮我整理一下吗"], None) == StyleMatch(
        CandidateCategory.COURTESY,
        StyleIntent.COURTESY_THANKS,
    )


@pytest.mark.parametrize(
    ("text", "idle_seconds"),
    [
        ("你好像感冒了", 21600),
        ("现在吗", 21600),
        ("我下课了吗", 0),
        ("工作做完了吗", 0),
        ("今晚安慰你", 0),
    ],
)
def test_short_phrase_substrings_do_not_create_precise_intents(
    text: str, idle_seconds: int
) -> None:
    assert classify_style_reply(text, [], idle_seconds) is None


def test_context_help_request_must_be_the_complete_approved_utterance() -> None:
    assert classify_style_reply("弄好了", ["不要再问我可以帮我整理一下吗"], None) is None


@pytest.mark.parametrize("ending_mark", ("~", "～"))
def test_approved_ending_marks_preserve_precise_intent(ending_mark: str) -> None:
    assert classify_style_reply(f"我要睡觉了{ending_mark}", [], 0) == StyleMatch(
        CandidateCategory.SLEEP,
        StyleIntent.SLEEP_FAREWELL,
    )


@pytest.mark.parametrize("ending_mark", ("~", "～"))
def test_approved_ending_marks_preserve_default_category(ending_mark: str) -> None:
    assert classify_style_reply(f"想你了{ending_mark}", [], None) == StyleMatch(
        CandidateCategory.GENTLE_AFFECTION,
        StyleIntent.CATEGORY_DEFAULT,
    )


@pytest.mark.parametrize("ending_mark", ("~", "～"))
def test_approved_ending_marks_preserve_greeting_reopen(ending_mark: str) -> None:
    assert classify_style_reply(f"hello{ending_mark}", [], 21600) == StyleMatch(
        CandidateCategory.GREETING,
        StyleIntent.CONVERSATION_REOPEN,
    )


@pytest.mark.parametrize(
    ("text", "recent_context"),
    [
        ("弄好了~", ["可以帮我整理一下吗"]),
        ("弄好了～", ["可以帮我整理一下吗"]),
        ("弄好了", ["可以帮我整理一下吗~"]),
        ("弄好了", ["可以帮我整理一下吗～"]),
    ],
)
def test_approved_ending_marks_preserve_context_assisted_completion(
    text: str, recent_context: list[str]
) -> None:
    assert classify_style_reply(text, recent_context, None) == StyleMatch(
        CandidateCategory.COURTESY,
        StyleIntent.COURTESY_THANKS,
    )


@pytest.mark.parametrize(
    ("text", "recent_context", "idle_seconds"),
    [
        (" ", [], None),
        ("合" * 2001, [], None),
        ("hello", ["一", "二", "三", "四", "五"], None),
        ("hello", [" "], None),
        ("hello", ["合" * 2001], None),
        ("hello", [], -1),
        ("hello", [], 1.5),
        ("hello", [], True),
    ],
)
def test_invalid_input_raises_sanitized_style_validation_error(
    text: str, recent_context: list[str], idle_seconds: object
) -> None:
    with pytest.raises(StyleValidationError) as caught:
        classify_style_reply(text, recent_context, idle_seconds)  # type: ignore[arg-type]

    assert caught.value.code == "invalid_style_guidance_input"
    assert caught.value.detail == "style guidance input is invalid"
