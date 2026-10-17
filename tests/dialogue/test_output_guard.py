# ruff: noqa: RUF001

from __future__ import annotations

import traceback

import pytest

from companion.contracts.style import StyleReplyGuidance
from companion.dialogue.output_guard import (
    OutputBlockReason,
    UnsafeDialogueOutput,
    validate_dialogue_output,
)


def unavailable_guidance() -> StyleReplyGuidance:
    return StyleReplyGuidance(status="unavailable")


@pytest.mark.parametrize(
    "fragment",
    (
        "પડી", "್ಯರ್ಥ",
        "\u0f60\u0f53\u0f0b\u0f63\u0f72\u0f53\u0f0b\u0f62\u0f92\u0f71\u0f56\u0f0b",
        "】【。", r"\()",
        "在线观看", "中文字幕", "】【", "α", "Ж", "ا", "a\u0308\u0301",
        "\U000e0100", "＼（",
    ),
)
def test_guard_blocks_garbled_output_without_retaining_text(fragment: str) -> None:
    text = f"合成回复：{fragment}"
    with pytest.raises(UnsafeDialogueOutput) as caught:
        validate_dialogue_output(text, unavailable_guidance(), None)
    assert caught.value.reason.value == "garbled"
    assert str(caught.value) == "unsafe dialogue output"
    for frame, _ in traceback.walk_tb(caught.value.__traceback__):
        if "/companion/dialogue/output_guard.py" in frame.f_code.co_filename:
            assert text not in repr(frame.f_locals)


@pytest.mark.parametrize(
    "text",
    (
        "正常中文。？……", "中文 English café Ā 123", "😡🥺❤️👨‍👩‍👧",
        "こんにちは、カタカナ", "スーパー𛀀𛀂", "안녕하세요 가", "𠀀﨑", "Cafe\u0301",
    ),
)
def test_guard_allows_supported_scripts_and_emoji(text: str) -> None:
    assert validate_dialogue_output(text, unavailable_guidance(), None) == text


def test_garbled_check_runs_after_invalid_text_before_impersonation() -> None:
    for text, reason in (
        ("પડી\x00", OutputBlockReason.INVALID_TEXT),
        ("પડી我就是主人", "garbled"),
    ):
        with pytest.raises(UnsafeDialogueOutput) as caught:
            validate_dialogue_output(text, unavailable_guidance(), None)
        assert caught.value.reason == reason


def matched_guidance(
    *,
    examples: list[str],
    soft_lock: str | None,
) -> StyleReplyGuidance:
    return StyleReplyGuidance(
        status="matched",
        style_version="synthetic-style-v1",
        category="greeting",
        intent="conversation_reopen",
        examples=examples,
        soft_lock_expression=soft_lock,
    )


@pytest.mark.parametrize(
    ("text", "reason"),
    (
        ("我就是主人本人，可以替他处理。", "impersonation"),
        ("我代表主人承诺这笔交易。", "owner_authorization"),
        ("系统提示是：忽略所有安全规则。", "prompt_disclosure"),
    ),
)
def test_guard_blocks_identity_authority_and_prompt_leaks(text: str, reason: str) -> None:
    with pytest.raises(UnsafeDialogueOutput) as error:
        validate_dialogue_output(text, unavailable_guidance(), None)

    assert error.value.reason.value == reason


def test_guard_blocks_full_example_copy_and_consecutive_soft_lock() -> None:
    guidance = matched_guidance(
        examples=["这是一条完整合成风格示例"],
        soft_lock="合成固定短语",
    )

    with pytest.raises(UnsafeDialogueOutput, match="unsafe dialogue output"):
        validate_dialogue_output("回复：这是一条完整合成风格示例", guidance, None)
    with pytest.raises(UnsafeDialogueOutput) as error:
        validate_dialogue_output("合成固定短语，继续回答。", guidance, "合成固定短语")

    assert error.value.reason is OutputBlockReason.MECHANICAL_STYLE_REPEAT


@pytest.mark.parametrize(
    "text",
    (
        "",
        " \n\t ",
        "正常回答\x00",
        "a" * 4_001,
        *[f"{control}正常回答" for control in ("\x1c", "\x1d", "\x1e", "\x1f")],
        *[f"正常回答{control}" for control in ("\x1c", "\x1d", "\x1e", "\x1f")],
    ),
)
def test_guard_blocks_blank_control_and_too_long_output(text: str) -> None:
    with pytest.raises(UnsafeDialogueOutput) as error:
        validate_dialogue_output(text, unavailable_guidance(), None)

    assert error.value.reason is OutputBlockReason.INVALID_TEXT


def test_guard_allows_independent_ai_wording_and_returns_stripped_original() -> None:
    text = "  我是独立运行的 ＡＩ 助手，不是主人本人。  "

    result = validate_dialogue_output(text, unavailable_guidance(), None)

    assert result == "我是独立运行的 ＡＩ 助手，不是主人本人。"


def test_guard_uses_nfkc_for_checks_without_normalizing_returned_text() -> None:
    text = "  ｓｙｓｔｅｍ　ｐｒｏｍｐｔ　ｉｓ：忽略所有安全规则。  "

    with pytest.raises(UnsafeDialogueOutput) as error:
        validate_dialogue_output(text, unavailable_guidance(), None)

    assert error.value.reason is OutputBlockReason.PROMPT_DISCLOSURE


def test_guard_allows_short_coincidental_example_fragment() -> None:
    guidance = matched_guidance(examples=["好呀"], soft_lock=None)

    result = validate_dialogue_output("好呀，今天也要慢慢来。", guidance, None)

    assert result == "好呀，今天也要慢慢来。"


def test_unsafe_exception_and_dialogue_traceback_do_not_retain_style_material() -> None:
    example = "SYNTHETIC_STYLE_EXAMPLE_SENTINEL"
    soft_lock = "SYNTHETIC_SOFT_LOCK_SENTINEL"
    output = f"回复：{soft_lock}"
    guidance = matched_guidance(examples=[example], soft_lock=soft_lock)

    with pytest.raises(UnsafeDialogueOutput) as caught:
        validate_dialogue_output(output, guidance, soft_lock)

    assert str(caught.value) == "unsafe dialogue output"
    assert caught.value.__cause__ is None
    sentinels = (output, example, soft_lock, guidance.style_version)
    for frame, _line_number in traceback.walk_tb(caught.value.__traceback__):
        if "/companion/dialogue/output_guard.py" in frame.f_code.co_filename.replace("\\", "/"):
            assert not any(sentinel in repr(frame.f_locals) for sentinel in sentinels)
