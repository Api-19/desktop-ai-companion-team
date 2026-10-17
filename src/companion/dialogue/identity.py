AI_DISCLOSURE_TEMPLATE = "我是 {name}，一个 AI，学的是他说话的方式，不是他本人哦。"  # noqa: RUF001
AI_DISCLOSURE = AI_DISCLOSURE_TEMPLATE.format(name="kiki")


def ensure_ai_disclosure(
    text: str, already_disclosed: bool, *, character_name: str = "kiki"
) -> tuple[str, bool]:
    clean = text.strip()
    if already_disclosed:
        return clean, False
    disclosure = AI_DISCLOSURE_TEMPLATE.format(name=character_name)
    return f"{disclosure}\n{clean}", True
