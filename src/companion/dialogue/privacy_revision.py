"""Public revision sources for explicitly assembled privacy checks."""

from typing import Protocol

PrivacyRevision = tuple[int | str, ...]
PRIVACY_REVISION_FAILURE = "privacy revision unavailable"


class ContentRevisionSource(Protocol):
    def content_revision(self, user_id: str) -> int | str: ...


class PrivacyRevisionUnavailable(RuntimeError):
    """At least one explicitly supplied revision source could not be read."""
