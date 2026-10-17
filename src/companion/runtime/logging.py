from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TextIO

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = (
    "apikey",
    "token",
    "password",
    "authorization",
    "text",
    "value",
    "recentcontext",
    "examples",
    "redactedtext",
    "chattext",
    "prompt",
    "transcript",
    "response",
    "memoryvalue",
    "redactionmap",
    "identitymap",
    "rawaudio",
)
SECRET_PATTERN = re.compile(r"(?:sk-[A-Za-z0-9_-]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def _normalise_key(key: object) -> str:
    return str(key).lower().replace("_", "").replace("-", "")


def _sanitize(value: object, key: object | None = None) -> object:
    if key is not None and any(part in _normalise_key(key) for part in SENSITIVE_KEYS):
        return REDACTED
    if isinstance(value, str):
        return REDACTED if SECRET_PATTERN.search(value) else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(item_key): _sanitize(item, item_key) for item_key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize(item) for item in value]
    return f"<{type(value).__name__}>"


def sanitize_fields(fields: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _sanitize(value, key) for key, value in fields.items()}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "fields": sanitize_fields(getattr(record, "event_fields", {})),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class SafeLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def event(self, event: str, level: int = logging.INFO, **fields: object) -> None:
        self._logger.log(level, event, extra={"event_fields": fields})


def configure_logging(level: str, stream: TextIO | None = None) -> SafeLogger:
    logger = logging.getLogger("companion")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return SafeLogger(logger)
