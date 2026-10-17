import pytest

from companion.privacy import PrivacyValidationError, redact_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1234567890", "[LONG_NUMBER]"),
        ("QQ 1234567890 Steam 7654321", "QQ [LONG_NUMBER] Steam [LONG_NUMBER]"),
        (
            "http://example.test\uff0chttps://example.test。www.example.test\uff01",
            "[URL]\uff0c[URL]。[URL]\uff01",
        ),
        ("HTTPS://example.test\uff1fWWW.example.test、", "[URL]\uff1f[URL]、"),
        ("尾号3233", "[PLATE]"),
        ("车牌 ABC123 车号 123", "[PLATE] [PLATE]"),
        ("浙A12345 粤B123456", "[PLATE] [PLATE]"),
        ("13800138000", "[PHONE]"),
        ("110105194912310021", "[NATIONAL_ID]"),
        ("6222021234567890123", "[BANK_CARD]"),
        ("192.0.2.10", "[IP_ADDRESS]"),
        ("2026 3.14 100块 6666", "2026 3.14 100块 6666"),
        ("3.1234567 1234567.8", "3.1234567 1234567.8"),
        ("https://example.test/13800138000", "https://example.test/[PHONE]"),
        ("https://synthetic.user@example.test", "https://[EMAIL]"),
    ],
)
def test_new_structured_redactions_and_existing_priority(source: str, expected: str) -> None:
    assert redact_text(source).text == expected


def privacy_traceback_retains(error: BaseException, marker: str) -> bool:
    current = error.__traceback__
    while current is not None:
        filename = current.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename and marker in repr(current.tb_frame.f_locals):
            return True
        current = current.tb_next
    return False


def test_none_and_unmatched_text_are_preserved() -> None:
    assert redact_text(None).text is None

    result = redact_text("普通合成文本")

    assert result.text == "普通合成文本"
    assert result.labels == ()
    assert result.counts == {}


def test_structured_values_are_replaced_with_fixed_labels() -> None:
    source = (
        "邮箱 synthetic.user@example.test、手机 13800138000、"
        "证件 11010519491231002X、卡号 6222021234567890123、IP 192.0.2.10"
    )

    result = redact_text(source)

    assert result.text == (
        "邮箱 [EMAIL]、手机 [PHONE]、证件 [NATIONAL_ID]、卡号 [BANK_CARD]、IP [IP_ADDRESS]"
    )
    assert result.labels == (
        "EMAIL",
        "PHONE",
        "NATIONAL_ID",
        "BANK_CARD",
        "IP_ADDRESS",
    )
    assert result.counts == {
        "EMAIL": 1,
        "PHONE": 1,
        "NATIONAL_ID": 1,
        "BANK_CARD": 1,
        "IP_ADDRESS": 1,
    }


def test_literal_private_terms_use_fixed_labels_and_regex_characters_are_literal() -> None:
    result = redact_text(
        "虚构姓名甲住在虚构地址[A]、账号是 user+test、发生过虚构私密事件。",
        private_terms={
            "name": ("虚构姓名甲",),
            "address": ("虚构地址[A]",),
            "account": ("user+test",),
            "private_event": ("虚构私密事件",),
        },
    )

    assert result.text == ("[NAME]住在[ADDRESS]、账号是 [ACCOUNT]、发生过[PRIVATE_EVENT]。")
    assert result.labels == ("NAME", "ADDRESS", "ACCOUNT", "PRIVATE_EVENT")


def test_longer_overlapping_literal_wins_and_generated_label_is_not_reprocessed() -> None:
    result = redact_text(
        "虚构地址甲乙和[NAME]",
        private_terms={
            "address": ("虚构地址甲", "虚构地址甲乙"),
            "name": ("[NAME]",),
        },
    )

    assert result.text == "[ADDRESS]和[NAME]"
    assert result.counts == {"ADDRESS": 1, "NAME": 1}


def test_repeated_matches_have_deterministic_counts_and_label_order() -> None:
    result = redact_text(
        "13800138000 与 13900139000、邮箱 synthetic.a@example.test",
    )

    assert result.labels == ("EMAIL", "PHONE")
    assert result.counts == {"EMAIL": 1, "PHONE": 2}


@pytest.mark.parametrize(
    "private_terms",
    [
        {"unknown": ("synthetic",)},
        {"name": ()},
        {"name": ("",)},
        {"name": ("   ",)},
        {"name": "synthetic"},
    ],
)
def test_invalid_private_terms_fail_without_exposing_values(private_terms) -> None:
    with pytest.raises(PrivacyValidationError) as exc_info:
        redact_text("synthetic secret body", private_terms=private_terms)

    assert exc_info.value.code == "invalid_private_terms"
    assert "synthetic" not in str(exc_info.value)
    assert exc_info.value.__context__ is None
    assert not privacy_traceback_retains(exc_info.value, "synthetic")


def test_fifty_synthetic_canaries_leave_no_original_values() -> None:
    canaries = [f"synthetic.canary{index:02d}@example.test" for index in range(50)]
    source = " ".join(canaries)

    result = redact_text(source)

    assert result.counts == {"EMAIL": 50}
    assert all(canary not in result.text for canary in canaries)
    assert result.text.count("[EMAIL]") == 50
