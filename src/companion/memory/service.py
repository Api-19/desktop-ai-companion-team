from __future__ import annotations

import secrets
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from functools import wraps
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar, cast

from pydantic import ValidationError

from companion.contracts import ConsentRecord, MemoryEntry
from companion.memory.errors import (
    MemoryAuthorizationError,
    MemoryServiceError,
    MemoryStoreError,
    MemoryValidationError,
)
from companion.memory.models import (
    CorrectMemoryRequest,
    CreateMemoryRequest,
    DeletionReceipt,
    MemoryField,
)
from companion.memory.profile import (
    PROFILE_PERMISSION,
    ProfileCategory,
    ProfileItem,
    ProfileSource,
    ProfileStatus,
    ProfileView,
    bigrams,
    profile_overlap,
)
from companion.memory.retrieval import select_matching_memories, validate_memory_query
from companion.memory.store import _MemoryStore

_ALLOWED_FIELDS = frozenset(field.value for field in MemoryField) | {PROFILE_PERMISSION}
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _sanitize_public_errors(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        result: _R | None = None
        error_type: type[MemoryServiceError] | None = None
        error_message: str | None = None
        try:
            result = function(*args, **kwargs)
        except MemoryServiceError as error:
            error_type = type(error)
            error_message = _safe_error_message(error)

        del args, kwargs
        if error_type is not None and error_message is not None:
            _raise_public_error(error_type, error_message)
        return cast(_R, result)

    return wrapper


class MemoryService:
    @_sanitize_public_errors
    def __init__(self, database_path: Path, clock: Callable[[], datetime]) -> None:
        self._store = _MemoryStore(database_path)
        self._clock = clock

    @_sanitize_public_errors
    def content_revision(self, user_id: str) -> int:
        _validate_user_id_or_raise(user_id)
        self._cleanup_profiles()
        return self._store.content_revision(user_id)

    @_sanitize_public_errors
    def grant_consent(self, record: ConsentRecord) -> ConsentRecord:
        self._cleanup_profiles()
        try:
            validated = ConsentRecord.model_validate(record.model_dump())
            _validate_user_id(validated.user_id)
            if not validated.granted or validated.granted_at is None:
                raise ValueError("grant required")
            if any(field not in _ALLOWED_FIELDS for field in validated.allowed_fields):
                raise ValueError("unknown field")
            normalized = validated.model_copy(
                update={"allowed_fields": sorted(set(validated.allowed_fields))}
            )
        except Exception:
            raise MemoryValidationError("invalid memory consent") from None
        self._store.save_consent(normalized)
        return normalized

    @_sanitize_public_errors
    def get_consent(self, user_id: str) -> ConsentRecord | None:
        self._cleanup_profiles()
        _validate_user_id_or_raise(user_id)
        return self._store.load_consent(user_id)

    @_sanitize_public_errors
    def create_memory(self, request: CreateMemoryRequest) -> MemoryEntry:
        self._cleanup_profiles()
        try:
            validated = CreateMemoryRequest.model_validate(request.model_dump())
            _validate_user_id(validated.user_id)
        except Exception:
            raise MemoryValidationError("invalid memory request") from None
        return self._store.insert_memory_if_authorized(validated)

    @_sanitize_public_errors
    def list_memories(self, user_id: str) -> tuple[MemoryEntry, ...]:
        self._cleanup_profiles()
        _validate_user_id_or_raise(user_id)
        return self._store.list_memories(user_id)

    @_sanitize_public_errors
    def correct_memory(self, request: CorrectMemoryRequest) -> MemoryEntry:
        self._cleanup_profiles()
        try:
            validated = CorrectMemoryRequest.model_validate(request.model_dump())
            _validate_user_id(validated.user_id)
        except Exception:
            raise MemoryValidationError("invalid memory request") from None
        corrected = self._store.correct_memory_if_authorized(validated)
        if corrected is None:
            raise MemoryAuthorizationError("memory entry unavailable")
        return corrected

    @_sanitize_public_errors
    def retrieve_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> tuple[MemoryEntry, ...]:
        self._cleanup_profiles()
        _validate_user_id_or_raise(user_id)
        try:
            validate_memory_query(query, limit)
        except ValueError:
            raise MemoryValidationError("invalid memory query") from None
        consent = self._store.load_consent(user_id)
        if consent is None or not consent.granted:
            return ()
        entries = self._store.list_memories(user_id)
        return select_matching_memories(entries, query, limit)

    @_sanitize_public_errors
    def delete_memory(self, user_id: str, memory_id: str) -> DeletionReceipt:
        self._cleanup_profiles()
        _validate_user_id_or_raise(user_id)
        _validate_memory_id_or_raise(memory_id)
        receipt = self._empty_receipt(user_id)
        deleted_count = self._store.delete_memory(user_id, memory_id)
        return receipt.model_copy(update={"deleted_count": deleted_count})

    @_sanitize_public_errors
    def revoke_consent(self, user_id: str) -> DeletionReceipt:
        self._cleanup_profiles()
        _validate_user_id_or_raise(user_id)
        receipt = self._empty_receipt(user_id)
        deleted_count = self._store.revoke_consent(user_id)
        return receipt.model_copy(update={"deleted_count": deleted_count})

    @_sanitize_public_errors
    def add_profile_candidate(
        self,
        *,
        user_id: str,
        category: ProfileCategory,
        topic: str,
        text: str,
        source: ProfileSource,
        evidence_date: date,
        evidence_quote: str | None = None,
        confidence: Literal["low", "med", "high"],
    ) -> ProfileItem:
        self._cleanup_profiles()
        _validate_user_id_or_raise(user_id)
        now = self._now()
        try:
            item = ProfileItem(
                profile_id=secrets.token_hex(16),
                user_id=user_id,
                category=category,
                topic=topic,
                text=text,
                source=source,
                evidence_date=evidence_date,
                evidence_quote=evidence_quote,
                confidence=confidence,
                created_at=now,
                updated_at=now,
            )
        except (ValidationError, ValueError):
            raise MemoryValidationError("invalid memory request") from None
        return self._store.insert_profile_if_authorized(item)

    @_sanitize_public_errors
    def list_profile(self, user_id: str) -> tuple[ProfileItem, ...]:
        self._cleanup_profiles()
        _validate_user_id_or_raise(user_id)
        return self._store.list_profiles(user_id)

    @_sanitize_public_errors
    def approve_profile(self, *, user_id: str, profile_id: str) -> ProfileItem:
        self._cleanup_profiles()
        self._validate_profile_target(user_id, profile_id)
        now = self._now()

        def approve(item: ProfileItem) -> ProfileItem:
            if item.status != ProfileStatus.CANDIDATE:
                raise MemoryValidationError("invalid memory request")
            return item.model_copy(
                update={"status": ProfileStatus.OWNER_APPROVED, "updated_at": now}
            )

        return self._store.update_profile(user_id, profile_id, approve)

    @_sanitize_public_errors
    def edit_profile(
        self,
        *,
        user_id: str,
        profile_id: str,
        category: ProfileCategory | None = None,
        topic: str | None = None,
        text: str | None = None,
        source: ProfileSource | None = None,
    ) -> ProfileItem:
        self._cleanup_profiles()
        self._validate_profile_target(user_id, profile_id)
        now = self._now()
        changes = {
            key: value
            for key, value in {
                "category": category,
                "topic": topic,
                "text": text,
                "source": source,
            }.items()
            if value is not None
        }

        def edit(item: ProfileItem) -> ProfileItem:
            if not changes:
                return item
            if item.source == ProfileSource.OWNER_ADDED and source not in (
                None,
                ProfileSource.OWNER_ADDED,
            ):
                raise MemoryValidationError("invalid memory request")
            update = changes | {"version": item.version + 1, "updated_at": now}
            if item.status == ProfileStatus.CONFIRMED:
                update |= {
                    "status": ProfileStatus.OWNER_APPROVED,
                    "confirmed_at": None,
                    "confirmed_by": None,
                    "confirmed_version": None,
                    "use_in_chat": False,
                    "simulated": False,
                }
            try:
                return ProfileItem.model_validate(item.model_dump() | update)
            except ValidationError:
                raise MemoryValidationError("invalid memory request") from None

        return self._store.update_profile(user_id, profile_id, edit)

    @_sanitize_public_errors
    def disable_profile(
        self,
        *,
        user_id: str,
        profile_id: str,
        reason: Literal["owner", "self", "conflict", "expired"],
    ) -> ProfileItem:
        self._cleanup_profiles()
        self._validate_profile_target(user_id, profile_id)
        if reason not in {"owner", "self", "conflict", "expired"}:
            raise MemoryValidationError("invalid memory request")
        now = self._now()
        return self._store.update_profile(
            user_id,
            profile_id,
            lambda item: item.model_copy(
                update={
                    "status": ProfileStatus.DISABLED,
                    "disabled_reason": reason,
                    "updated_at": now,
                }
            ),
        )

    @_sanitize_public_errors
    def delete_profile(self, *, user_id: str, profile_id: str) -> DeletionReceipt:
        self._cleanup_profiles()
        self._validate_profile_target(user_id, profile_id)
        receipt = self._empty_receipt(user_id)
        deleted_count = self._store.delete_profile(user_id, profile_id)
        return receipt.model_copy(update={"deleted_count": deleted_count})

    @_sanitize_public_errors
    def confirm_profile(
        self,
        *,
        user_id: str,
        profile_id: str,
        version: int,
        subject_id: str,
        use_in_chat: bool,
        simulated: bool,
    ) -> ProfileItem:
        self._cleanup_profiles()
        self._validate_profile_target(user_id, profile_id)
        if subject_id != user_id:
            raise MemoryAuthorizationError("memory not authorized")
        if type(version) is not int or type(use_in_chat) is not bool or type(simulated) is not bool:
            raise MemoryValidationError("invalid memory request")
        now = self._now()

        def confirm(item: ProfileItem) -> ProfileItem:
            if item.version != version or item.status not in (
                ProfileStatus.OWNER_APPROVED,
                ProfileStatus.CONFIRMED,
            ):
                raise MemoryValidationError("invalid memory request")
            return item.model_copy(
                update={
                    "status": ProfileStatus.CONFIRMED,
                    "confirmed_at": now,
                    "confirmed_by": subject_id,
                    "confirmed_version": version,
                    "use_in_chat": use_in_chat,
                    "simulated": simulated,
                    "evidence_quote": None,
                    "updated_at": now,
                }
            )

        return self._store.update_profile(user_id, profile_id, confirm, check_conflicts=True)

    @_sanitize_public_errors
    def retrieve_profile(
        self,
        *,
        user_id: str,
        query_texts: Sequence[str],
        limit: int = 3,
    ) -> tuple[ProfileView, ...]:
        self._cleanup_profiles()
        _validate_user_id_or_raise(user_id)
        if (
            isinstance(query_texts, (str, bytes))
            or not isinstance(query_texts, Sequence)
            or any(not isinstance(query, str) for query in query_texts)
            or type(limit) is not int
            or limit < 0
        ):
            raise MemoryValidationError("invalid memory query")
        consent = self._store.load_consent(user_id)
        if (
            consent is None
            or not consent.granted
            or PROFILE_PERMISSION not in consent.allowed_fields
        ):
            return ()
        now = self._now()
        query_bigrams = set().union(*(bigrams(query) for query in query_texts))
        ranked = []
        for item in self._store.list_profiles(user_id):
            if (
                item.status != ProfileStatus.CONFIRMED
                or not item.use_in_chat
                or item.confirmed_version != item.version
                or item.valid_until is None
                or now.date() >= item.valid_until
            ):
                continue
            score = profile_overlap(item, query_bigrams)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: -pair[0])
        return tuple(
            ProfileView(topic=item.topic, text=item.text, category=item.category)
            for _, item in ranked[: min(limit, 3)]
        )

    @staticmethod
    def _validate_profile_target(user_id: str, profile_id: str) -> None:
        _validate_user_id_or_raise(user_id)
        _validate_memory_id_or_raise(profile_id)

    def _cleanup_profiles(self) -> None:
        self._store.cleanup_profiles(self._now())

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise MemoryValidationError("invalid memory clock")
        return now.astimezone(UTC)

    def _empty_receipt(self, user_id: str) -> DeletionReceipt:
        try:
            return DeletionReceipt(user_id=user_id, deleted_count=0, completed_at=self._clock())
        except Exception:
            raise MemoryValidationError("invalid memory clock") from None


def _service_with_store(store: _MemoryStore, clock: Callable[[], datetime]) -> MemoryService:
    # Internal factory hook: keeps the public constructor signature unchanged.
    service = MemoryService.__new__(MemoryService)
    service._store = store
    service._clock = clock
    return service


def _validate_user_id(user_id: object) -> None:
    if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 120:
        raise ValueError("invalid user")


def _validate_user_id_or_raise(user_id: object) -> None:
    try:
        _validate_user_id(user_id)
    except Exception:
        raise MemoryValidationError("invalid memory user") from None


def _validate_memory_id_or_raise(memory_id: object) -> None:
    if (
        not isinstance(memory_id, str)
        or len(memory_id) != 32
        or any(character not in "0123456789abcdef" for character in memory_id)
    ):
        raise MemoryValidationError("invalid memory id")


def _safe_error_message(error: MemoryServiceError) -> str:
    if isinstance(error, MemoryStoreError):
        return "memory store unavailable"
    if isinstance(error, MemoryAuthorizationError):
        allowed = {
            "memory entry unavailable",
            "memory field not authorized",
            "memory write not authorized",
        }
        return error.args[0] if error.args and error.args[0] in allowed else "memory not authorized"
    allowed = {
        "invalid memory clock",
        "invalid memory consent",
        "invalid memory id",
        "invalid memory query",
        "invalid memory request",
        "invalid memory user",
    }
    return error.args[0] if error.args and error.args[0] in allowed else "invalid memory request"


def _raise_public_error(error_type: type[MemoryServiceError], error_message: str) -> None:
    raise error_type(error_message)
