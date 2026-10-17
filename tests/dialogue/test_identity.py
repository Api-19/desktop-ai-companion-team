from companion.dialogue.identity import AI_DISCLOSURE, ensure_ai_disclosure
from companion.dialogue.models import DialogueSessionState


def test_first_contact_prepends_local_ai_disclosure() -> None:
    text, disclosed = ensure_ai_disclosure("  synthetic reply  ", already_disclosed=False)

    assert text == f"{AI_DISCLOSURE}\nsynthetic reply"
    assert disclosed is True


def test_later_reply_does_not_mechanically_prepend_disclosure() -> None:
    text, disclosed = ensure_ai_disclosure("  synthetic reply  ", already_disclosed=True)

    assert text == "synthetic reply"
    assert disclosed is False


def test_fresh_session_requires_disclosure_again() -> None:
    state = DialogueSessionState()

    text, disclosed = ensure_ai_disclosure("synthetic reply", state.identity_disclosed)

    assert text.startswith(AI_DISCLOSURE)
    assert disclosed is True
