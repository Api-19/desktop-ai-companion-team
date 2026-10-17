"""Read-only partner profile source for dialogue."""

from collections.abc import Sequence
from typing import Protocol

from companion.memory.profile import ProfileView


class PartnerProfileProvider(Protocol):
    def retrieve_profile(
        self, *, user_id: str, query_texts: Sequence[str], limit: int = 3
    ) -> tuple[ProfileView, ...]: ...
