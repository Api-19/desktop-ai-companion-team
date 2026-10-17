import pytest


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("你好你好。[憨笑]", "你好你好。"),
        ("可以可以。\n饿着上桌啊。[憨笑]", "可以可以。\n饿着上桌啊。"),
        ("[猪头][猪头]", ""),
        ("好呀😋👍🏻", "好呀"),
        ("。？", ""),  # noqa: RUF001
        ("正文[一二三四五六七]", "正文[一二三四五六七]"),
        ("正文[普通 内容]", "正文[普通 内容]"),
        ("  你好。[拥抱]  \n  \n 世界！ \t", "你好。\n世界！"),  # noqa: RUF001
        ("好呀👩‍👩‍👧‍👦☀️^", "好呀"),
        ("1️⃣ #️⃣ *️⃣", "1 # *"),
        ("Hello, 123!", "Hello, 123!"),
        (" \n\t", ""),
    ],
)
def test_spoken_text(text: str, expected: str) -> None:
    from companion.speech.tts_text import spoken_text

    assert spoken_text(text) == expected
