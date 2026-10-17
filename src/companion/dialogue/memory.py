from __future__ import annotations

import re
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from companion.contracts import MemoryEntry
from companion.memory import CreateMemoryRequest, MemoryField


class MemoryGateway(Protocol):
    def create_memory(self, request: CreateMemoryRequest) -> MemoryEntry: ...

    def retrieve_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> tuple[MemoryEntry, ...]: ...


class ParsedMemoryCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: CreateMemoryRequest


_MEMORY_COMMAND_PATTERNS: tuple[tuple[MemoryField, re.Pattern[str]], ...] = (
    (MemoryField.PREFERRED_NAME, re.compile(r"记住以后叫我(?P<value>.*)")),
    (MemoryField.PREFERENCE, re.compile(r"记住我(?P<value>喜欢.*|不喜欢.*)")),
    (MemoryField.EXPLICIT_NOTE, re.compile(r"记住这件事\uFF1A(?P<value>.*)")),
)
_MEMORY_COMMAND_PREFIXES = (
    "记住以后叫我",
    "记住我喜欢",
    "记住我不喜欢",
    "记住这件事\uff1a",
)


def parse_memory_command(
    text: str, *, user_id: str, source_timestamp: datetime
) -> ParsedMemoryCommand | None:
    if not isinstance(text, str):
        return None

    for field, pattern in _MEMORY_COMMAND_PATTERNS:
        match = pattern.fullmatch(text)
        if match is None:
            continue

        value = match.group("value").strip()
        if (
            not value
            or value in {"喜欢", "不喜欢"}
            or len(value) > 500
            or any(prefix in value for prefix in _MEMORY_COMMAND_PREFIXES)
        ):
            return None

        return ParsedMemoryCommand(
            request=CreateMemoryRequest(
                user_id=user_id,
                field=field,
                value=value,
                source_timestamp=source_timestamp,
                explicitly_requested=True,
            )
        )

    return None
