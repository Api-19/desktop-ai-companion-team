import re
import unicodedata

_BRACKET_EMOTICON = re.compile(r"\[[^\[\]\s]{1,6}\]")


def spoken_text(text: str) -> str:
    """Clean symbols for synthesis without changing the dialogue response."""
    text = _BRACKET_EMOTICON.sub("", text)
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"So", "Sk"}
        and char not in "\ufe0f\u200d\u20e3"
        and not "\U0001f3fb" <= char <= "\U0001f3ff"
    )
    text = "\n".join(line for raw_line in text.splitlines() if (line := raw_line.strip()))
    return text if any(char.isalnum() for char in text) else ""
