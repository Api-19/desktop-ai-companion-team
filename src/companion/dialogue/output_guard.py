from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum

from companion.contracts.style import StyleReplyGuidance
from companion.privacy.veiled import VeiledLexicon

IMPERSONATION_PATTERNS = ("我是主人本人", "我就是主人", "我代表主人本人")
OWNER_AUTHORITY_PATTERNS = (
    "我替主人同意",
    "我代表主人承诺",
    "主人已经授权",
    "我替主人授权",
)
_QUOTED_TEXT = re.compile(r"「[^」]*」|“[^”]*”")
_CLAUSE_BOUNDARY = re.compile(r"[，,。.!！？?；;：:\n]")  # noqa: RUF001
_NEGATION = re.compile(r"[不没别未]")
_OWNER_FACT = re.compile(
    r"(?<![其吉])他(?:明天|今天|今晚|待会|马上|一会|会|还|在|已经|刚|正在|一直)[\u3400-\u9fff]"
)
_PROFILE_ASSERTION = re.compile(r"你又|你肯定|你总是|你还是|你老是")


def _owner_assertion_reason(text: str) -> OutputBlockReason | None:
    # Remove quoted data before splitting, so its punctuation/negation cannot affect assertions.
    clauses = _CLAUSE_BOUNDARY.split(_QUOTED_TEXT.sub(" ", text))
    for clause in clauses:
        for phrase in OWNER_AUTHORITY_PATTERNS:
            for match in re.finditer(phrase, clause):
                if not _NEGATION.search(clause[: match.start()]):
                    return OutputBlockReason.OWNER_AUTHORIZATION
    for clause in clauses:
        if _OWNER_FACT.search(clause) and not _NEGATION.search(clause):
            return OutputBlockReason.OWNER_FACT
    return None


PROMPT_DISCLOSURE_PATTERNS = (
    "系统提示是",
    "system prompt is",
    "STYLE_ACTIVE_MANIFEST_DIGEST",
    "OPENAI_API_KEY",
)
GARBLED_PATTERNS = ("在线观看", "中文字幕", "】【", "\\(")
ALLOWED_LETTER_NAME_PREFIXES = (
    "CJK UNIFIED IDEOGRAPH-",
    "CJK COMPATIBILITY IDEOGRAPH-",
    "LATIN ",
    "HIRAGANA ",
    "KATAKANA ",
    "KATAKANA-HIRAGANA ",
    "HENTAIGANA ",
    "HANGUL ",
)


def _is_garbled(text: str) -> bool:
    if any(pattern in text for pattern in GARBLED_PATTERNS):
        return True
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("M") and character not in "\ufe0e\ufe0f":
            return True
        if category.startswith("L") and not unicodedata.name(character, "").startswith(
            ALLOWED_LETTER_NAME_PREFIXES
        ):
            return True
    return False


class OutputBlockReason(StrEnum):
    IMPERSONATION = "impersonation"
    OWNER_AUTHORIZATION = "owner_authorization"
    OWNER_FACT = "owner_fact"
    VEILED = "veiled"
    UNKNOWN = "unknown"
    PROMPT_DISCLOSURE = "prompt_disclosure"
    EXAMPLE_COPY = "example_copy"
    MECHANICAL_STYLE_REPEAT = "mechanical_style_repeat"
    INVALID_TEXT = "invalid_text"
    GARBLED = "garbled"
    PROFILE_ASSERTION = "profile_assertion"


class UnsafeDialogueOutput(RuntimeError):
    def __init__(self, message: str, reason: OutputBlockReason) -> None:
        super().__init__(message)
        self.reason = reason


def _raise_unsafe(reason: OutputBlockReason) -> None:
    raise UnsafeDialogueOutput("unsafe dialogue output", reason) from None


def validate_dialogue_output(
    text: str,
    guidance: StyleReplyGuidance,
    previous_soft_lock: str | None,
    *,
    extra_examples: Sequence[str] = (),
    veiled: VeiledLexicon | None = None,
    profile_topics: Sequence[str] = (),
) -> str:
    """Return safe provider output or raise a fixed, non-sensitive error."""
    reason: OutputBlockReason | None = None
    example: str | None = None
    normalized_example: str | None = None
    normalized_soft_lock: str | None = None

    if not isinstance(text, str) or any(
        ord(character) < 32 and character not in "\n\t" for character in text
    ):
        stripped_text = ""
        normalized_text = ""
        reason = OutputBlockReason.INVALID_TEXT
    else:
        stripped_text = str.strip(text)
        normalized_text = unicodedata.normalize("NFKC", stripped_text)
        if not 1 <= len(normalized_text) <= 4_000 or any(
            ord(character) < 32 and character not in "\n\t" for character in normalized_text
        ):
            reason = OutputBlockReason.INVALID_TEXT
        elif _is_garbled(normalized_text):
            reason = OutputBlockReason.GARBLED
        elif any(pattern in normalized_text for pattern in IMPERSONATION_PATTERNS):
            reason = OutputBlockReason.IMPERSONATION
        elif (owner_reason := _owner_assertion_reason(normalized_text)) is not None:
            reason = owner_reason
        elif any(pattern in normalized_text for pattern in PROMPT_DISCLOSURE_PATTERNS):
            reason = OutputBlockReason.PROMPT_DISCLOSURE
        elif veiled is not None and veiled.scan(stripped_text).high:
            reason = OutputBlockReason.VEILED
        elif profile_topics and any(
            topic in normalized_text[match.end() : match.end() + 8]
            for match in _PROFILE_ASSERTION.finditer(normalized_text)
            for topic in profile_topics
        ):
            reason = OutputBlockReason.PROFILE_ASSERTION
        else:
            for example in (*guidance.examples, *extra_examples):
                normalized_example = unicodedata.normalize("NFKC", str.strip(example))
                if (
                    len("".join(normalized_example.split())) >= 8
                    and normalized_example in normalized_text
                ):
                    reason = OutputBlockReason.EXAMPLE_COPY
                    break
            if reason is None and previous_soft_lock:
                normalized_soft_lock = unicodedata.normalize(
                    "NFKC",
                    str.strip(previous_soft_lock),
                )
                if normalized_soft_lock and normalized_soft_lock in normalized_text:
                    reason = OutputBlockReason.MECHANICAL_STYLE_REPEAT

    if reason is None:
        return stripped_text

    del text, guidance, previous_soft_lock, extra_examples, veiled, stripped_text, normalized_text
    del example, normalized_example, normalized_soft_lock
    del profile_topics
    _raise_unsafe(reason)
