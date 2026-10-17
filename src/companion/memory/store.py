from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from companion.contracts import ConsentRecord, MemoryEntry
from companion.memory.errors import (
    MemoryAuthorizationError,
    MemoryServiceError,
    MemoryStoreError,
    MemoryValidationError,
)
from companion.memory.models import CorrectMemoryRequest, CreateMemoryRequest
from companion.memory.profile import PROFILE_PERMISSION, ProfileItem, ProfileStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consents (
    user_id TEXT PRIMARY KEY,
    granted INTEGER NOT NULL CHECK (granted IN (0, 1)),
    granted_at TEXT,
    allowed_fields TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    field TEXT NOT NULL,
    value TEXT NOT NULL,
    source_timestamp TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES consents(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS memories_by_user ON memories(user_id);
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    disabled_at TEXT,
    simulated_only INTEGER NOT NULL DEFAULT 0 CHECK (simulated_only IN (0, 1)),
    FOREIGN KEY (user_id) REFERENCES consents(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS profiles_by_user ON profiles(user_id, created_at);
CREATE TABLE IF NOT EXISTS content_revisions (
    user_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision >= 0)
);
CREATE TRIGGER IF NOT EXISTS content_revision_consents_insert
AFTER INSERT ON consents
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER IF NOT EXISTS content_revision_consents_update
AFTER UPDATE ON consents
WHEN OLD.granted IS NOT NEW.granted
  OR OLD.granted_at IS NOT NEW.granted_at
  OR OLD.allowed_fields IS NOT NEW.allowed_fields
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER IF NOT EXISTS content_revision_consents_delete
AFTER DELETE ON consents
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (OLD.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER IF NOT EXISTS content_revision_memories_insert
AFTER INSERT ON memories
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER IF NOT EXISTS content_revision_memories_update
AFTER UPDATE ON memories
WHEN OLD.field IS NOT NEW.field
  OR OLD.value IS NOT NEW.value
  OR OLD.source_timestamp IS NOT NEW.source_timestamp
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER IF NOT EXISTS content_revision_memories_delete
AFTER DELETE ON memories
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (OLD.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER IF NOT EXISTS content_revision_profiles_insert
AFTER INSERT ON profiles
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER IF NOT EXISTS content_revision_profiles_update
AFTER UPDATE ON profiles
WHEN OLD.created_at IS NOT NEW.created_at
  OR OLD.payload IS NOT NEW.payload
  OR OLD.disabled_at IS NOT NEW.disabled_at
  OR OLD.simulated_only IS NOT NEW.simulated_only
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER IF NOT EXISTS content_revision_profiles_delete
AFTER DELETE ON profiles
BEGIN
    INSERT INTO content_revisions(user_id, revision) VALUES (OLD.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
"""


class _MemoryStore:
    def __init__(
        self, database_path: Path, *, boundary_check: Callable[[], None] | None = None
    ) -> None:
        self._database_path = database_path
        self._boundary_check = boundary_check
        self._initialize()

    def content_revision(self, user_id: str) -> int:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT revision FROM content_revisions WHERE user_id = ?", (user_id,)
                ).fetchone()
            return row[0] if row is not None else 0
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def save_consent(self, record: ConsentRecord) -> None:
        try:
            with self._connect() as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO consents(user_id, granted, granted_at, allowed_fields)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        granted = excluded.granted,
                        granted_at = excluded.granted_at,
                        allowed_fields = excluded.allowed_fields
                    """,
                    (
                        record.user_id,
                        int(record.granted),
                        _utc_iso(record.granted_at) if record.granted_at else None,
                        json.dumps(record.allowed_fields, ensure_ascii=False, sort_keys=True),
                    ),
                )
                allowed = record.allowed_fields if record.granted else []
                if PROFILE_PERMISSION not in allowed:
                    connection.execute("DELETE FROM profiles WHERE user_id = ?", (record.user_id,))
                placeholders = ",".join("?" for _ in allowed)
                connection.execute(
                    "DELETE FROM memories WHERE user_id = ?"
                    + (f" AND field NOT IN ({placeholders})" if allowed else ""),
                    (record.user_id, *allowed),
                )
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def load_consent(self, user_id: str) -> ConsentRecord | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT user_id, granted, granted_at, allowed_fields
                    FROM consents
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
            if row is None:
                return None
            return ConsentRecord.model_validate(
                {
                    "user_id": row[0],
                    "granted": bool(row[1]),
                    "granted_at": row[2],
                    "allowed_fields": json.loads(row[3]),
                }
            )
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def insert_memory_if_authorized(self, request: CreateMemoryRequest) -> MemoryEntry:
        memory_id = secrets.token_hex(16)
        source_timestamp = request.source_timestamp.astimezone(UTC)
        try:
            with self._connect() as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                consent = connection.execute(
                    "SELECT granted, allowed_fields FROM consents WHERE user_id = ?",
                    (request.user_id,),
                ).fetchone()
                if consent is None or not bool(consent[0]):
                    raise MemoryAuthorizationError("memory write not authorized")
                if request.field.value not in json.loads(consent[1]):
                    raise MemoryAuthorizationError("memory field not authorized")
                connection.execute(
                    """
                    INSERT INTO memories(memory_id, user_id, field, value, source_timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        request.user_id,
                        request.field.value,
                        request.value,
                        _utc_iso(source_timestamp),
                    ),
                )
            return MemoryEntry(
                memory_id=memory_id,
                user_id=request.user_id,
                field=request.field.value,
                value=request.value,
                source_timestamp=source_timestamp,
                deleted_at=None,
            )
        except MemoryAuthorizationError:
            raise
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def list_memories(self, user_id: str) -> tuple[MemoryEntry, ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT memory_id, user_id, field, value, source_timestamp
                    FROM memories
                    WHERE user_id = ?
                    ORDER BY source_timestamp DESC, memory_id ASC
                    """,
                    (user_id,),
                ).fetchall()
            return tuple(
                MemoryEntry(
                    memory_id=row[0],
                    user_id=row[1],
                    field=row[2],
                    value=row[3],
                    source_timestamp=datetime.fromisoformat(row[4]),
                    deleted_at=None,
                )
                for row in rows
            )
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def correct_memory_if_authorized(self, request: CorrectMemoryRequest) -> MemoryEntry | None:
        source_timestamp = request.source_timestamp.astimezone(UTC)
        try:
            with self._connect() as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                consent = connection.execute(
                    "SELECT granted, allowed_fields FROM consents WHERE user_id = ?",
                    (request.user_id,),
                ).fetchone()
                if consent is None or not bool(consent[0]):
                    raise MemoryAuthorizationError("memory write not authorized")
                existing = connection.execute(
                    "SELECT field FROM memories WHERE user_id = ? AND memory_id = ?",
                    (request.user_id, request.memory_id),
                ).fetchone()
                if existing is None:
                    raise MemoryAuthorizationError("memory entry unavailable")
                if existing[0] not in json.loads(consent[1]):
                    raise MemoryAuthorizationError("memory field not authorized")
                updated = connection.execute(
                    """
                    UPDATE memories
                    SET value = ?, source_timestamp = ?
                    WHERE user_id = ? AND memory_id = ?
                    """,
                    (
                        request.value,
                        _utc_iso(source_timestamp),
                        request.user_id,
                        request.memory_id,
                    ),
                ).rowcount
                if updated == 0:
                    return None
                row = connection.execute(
                    """
                    SELECT memory_id, user_id, field, value, source_timestamp
                    FROM memories
                    WHERE user_id = ? AND memory_id = ?
                    """,
                    (request.user_id, request.memory_id),
                ).fetchone()
            return _memory_from_row(row) if row is not None else None
        except MemoryAuthorizationError:
            raise
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def delete_memory(self, user_id: str, memory_id: str) -> int:
        try:
            with self._connect() as connection, connection:
                return connection.execute(
                    "DELETE FROM memories WHERE user_id = ? AND memory_id = ?",
                    (user_id, memory_id),
                ).rowcount
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def revoke_consent(self, user_id: str) -> int:
        try:
            with self._connect() as connection, connection:
                connection.execute(
                    """
                    INSERT INTO consents(user_id, granted, granted_at, allowed_fields)
                    VALUES (?, 0, NULL, '[]')
                    ON CONFLICT(user_id) DO UPDATE SET
                        granted = 0,
                        granted_at = NULL,
                        allowed_fields = '[]'
                    """,
                    (user_id,),
                )
                deleted = connection.execute(
                    "DELETE FROM memories WHERE user_id = ?", (user_id,)
                ).rowcount
                return (
                    deleted
                    + connection.execute(
                        "DELETE FROM profiles WHERE user_id = ?", (user_id,)
                    ).rowcount
                )
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def insert_profile_if_authorized(self, item: ProfileItem) -> ProfileItem:
        try:
            with self._connect() as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                _authorize_profile(connection, item.user_id)
                connection.execute(
                    "INSERT INTO profiles(profile_id, user_id, created_at, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        item.profile_id,
                        item.user_id,
                        _utc_iso(item.created_at),
                        item.model_dump_json(),
                    ),
                )
            return item
        except MemoryServiceError:
            raise
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def list_profiles(self, user_id: str) -> tuple[ProfileItem, ...]:
        try:
            with self._connect() as connection:
                return _profiles_for_user(connection, user_id)
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def update_profile(
        self,
        user_id: str,
        profile_id: str,
        transform: Callable[[ProfileItem], ProfileItem],
        *,
        check_conflicts: bool = False,
    ) -> ProfileItem:
        try:
            with self._connect() as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                _authorize_profile(connection, user_id)
                row = connection.execute(
                    "SELECT payload, simulated_only FROM profiles "
                    "WHERE user_id = ? AND profile_id = ?",
                    (user_id, profile_id),
                ).fetchone()
                if row is None:
                    raise MemoryAuthorizationError("memory entry unavailable")
                item = transform(ProfileItem.model_validate_json(row[0]))
                if item.status == ProfileStatus.CONFIRMED and row[1] and not item.simulated:
                    raise MemoryValidationError("invalid memory request")
                _save_profile(connection, item)
                if check_conflicts:
                    siblings = tuple(
                        sibling
                        for sibling in _profiles_for_user(connection, user_id)
                        if sibling.topic == item.topic and sibling.status == ProfileStatus.CONFIRMED
                    )
                    if len({sibling.text for sibling in siblings}) > 1:
                        for sibling in siblings:
                            disabled = sibling.model_copy(
                                update={
                                    "status": ProfileStatus.DISABLED,
                                    "disabled_reason": "conflict",
                                    "updated_at": item.updated_at,
                                }
                            )
                            _save_profile(connection, disabled)
                            if sibling.profile_id == profile_id:
                                item = disabled
            return item
        except MemoryServiceError:
            raise
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def delete_profile(self, user_id: str, profile_id: str) -> int:
        try:
            with self._connect() as connection, connection:
                deleted = connection.execute(
                    "DELETE FROM profiles WHERE user_id = ? AND profile_id = ?",
                    (user_id, profile_id),
                ).rowcount
                if not deleted:
                    raise MemoryAuthorizationError("memory entry unavailable")
                return deleted
        except MemoryServiceError:
            raise
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def cleanup_profiles(self, now: datetime) -> None:
        try:
            with self._connect() as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute("SELECT payload, disabled_at FROM profiles").fetchall()
                for payload, disabled_at in rows:
                    item = ProfileItem.model_validate_json(payload)
                    deadline = None
                    if item.status == ProfileStatus.CANDIDATE:
                        deadline = item.created_at + timedelta(days=30)
                    elif item.status == ProfileStatus.DISABLED:
                        deadline = datetime.fromisoformat(disabled_at) + timedelta(days=30)
                    elif item.valid_until is not None:
                        expiry = datetime.combine(item.valid_until, datetime.min.time(), UTC)
                        if now >= expiry:
                            deadline = expiry + timedelta(days=30)
                            if now < deadline:
                                _save_profile(
                                    connection,
                                    item.model_copy(
                                        update={
                                            "status": ProfileStatus.DISABLED,
                                            "disabled_reason": "expired",
                                            "updated_at": expiry,
                                        }
                                    ),
                                )
                    if deadline is not None and now >= deadline:
                        connection.execute(
                            "DELETE FROM profiles WHERE user_id = ? AND profile_id = ?",
                            (item.user_id, item.profile_id),
                        )
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    def _initialize(self) -> None:
        try:
            if self._boundary_check is None:
                self._database_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)
                version = connection.execute("PRAGMA user_version").fetchone()
                if version is None or int(version[0]) < 1:
                    connection.commit()
                    connection.execute("VACUUM")
                    connection.execute("PRAGMA user_version = 1")
                    connection.commit()
        except Exception:
            raise MemoryStoreError("memory store unavailable") from None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._boundary_check is None:
            connection = sqlite3.connect(self._database_path)
        else:
            # Recheck after opening to detect persistent boundary changes. SQLite
            # still opens by path, so this does not eliminate the TOCTOU window.
            self._boundary_check()
            connection = sqlite3.connect(f"{self._database_path.as_uri()}?mode=rw", uri=True)
        try:
            if self._boundary_check is not None:
                self._boundary_check()
            connection.execute("PRAGMA foreign_keys = ON")
            secure_delete = connection.execute("PRAGMA secure_delete = ON").fetchone()
            if secure_delete is None or int(secure_delete[0]) != 1:
                raise sqlite3.OperationalError("secure deletion unavailable")
            yield connection
        finally:
            connection.close()


def _memory_from_row(row: sqlite3.Row | tuple[object, ...]) -> MemoryEntry:
    return MemoryEntry(
        memory_id=row[0],
        user_id=row[1],
        field=row[2],
        value=row[3],
        source_timestamp=datetime.fromisoformat(str(row[4])),
        deleted_at=None,
    )


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _authorize_profile(connection: sqlite3.Connection, user_id: str) -> None:
    consent = connection.execute(
        "SELECT granted, allowed_fields FROM consents WHERE user_id = ?", (user_id,)
    ).fetchone()
    if consent is None or not consent[0]:
        raise MemoryAuthorizationError("memory write not authorized")
    if PROFILE_PERMISSION not in json.loads(consent[1]):
        raise MemoryAuthorizationError("memory field not authorized")


def _profiles_for_user(connection: sqlite3.Connection, user_id: str) -> tuple[ProfileItem, ...]:
    rows = connection.execute(
        "SELECT payload FROM profiles WHERE user_id = ? ORDER BY created_at, profile_id", (user_id,)
    ).fetchall()
    return tuple(ProfileItem.model_validate_json(row[0]) for row in rows)


def _save_profile(connection: sqlite3.Connection, item: ProfileItem) -> None:
    # Retention starts at the first disable, even if the owner subsequently edits the record.
    disabled_at = _utc_iso(item.updated_at) if item.status == ProfileStatus.DISABLED else None
    connection.execute(
        "UPDATE profiles SET payload = ?, disabled_at = COALESCE(disabled_at, ?), "
        "simulated_only = MAX(simulated_only, ?) WHERE user_id = ? AND profile_id = ?",
        (item.model_dump_json(), disabled_at, int(item.simulated), item.user_id, item.profile_id),
    )
