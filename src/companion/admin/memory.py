"""Owner-only UI operations over the public Memory service."""

import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from companion.contracts import ConsentRecord, MemoryEntry
from companion.memory import (
    PROFILE_PERMISSION,
    CorrectMemoryRequest,
    DeletionReceipt,
    MemoryField,
    MemoryService,
)

FIELDS = tuple(field.value for field in MemoryField)
LABELS = {"preferred_name": "偏好称呼", "preference": "偏好", "explicit_note": "明确备注"}


class MemoryTargetUnavailable(ValueError):
    pass


class MemoryStateChanged(ValueError):
    pass


class MemoryResultUnconfirmed(ValueError):
    pass


@dataclass(frozen=True)
class MemoryFormState:
    consent: bytes | None
    entries: tuple[bytes, ...]
    target_id: str | None


def valid_memory_id(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value) is not None


class MemoryAdmin:
    def __init__(
        self, factory: Callable[[], MemoryService] | None, clock: Callable[[], datetime]
    ) -> None:
        self._factory = factory
        self._clock = clock
        self._cached: MemoryService | None = None

    def service(self) -> MemoryService:
        """Return the shared service, constructing it only on first use."""
        if self._cached is None:
            if self._factory is None:
                raise ValueError("memory unavailable")
            service = self._factory()
            if type(service) is not MemoryService:
                raise ValueError("memory unavailable")
            self._cached = service
        return self._cached

    @staticmethod
    def validate_consent(value: object, owner: str) -> ConsentRecord | None:
        if value is None:
            return None
        if type(value) is not ConsentRecord:
            raise ValueError("memory unavailable")
        record = ConsentRecord.model_validate(value.model_dump(warnings=False), strict=True)
        if (
            record.user_id != owner
            or any(f not in (*FIELDS, PROFILE_PERMISSION) for f in record.allowed_fields)
            or len(set(record.allowed_fields)) != len(record.allowed_fields)
            or (record.granted and record.granted_at is None)
            or (not record.granted and (record.granted_at is not None or record.allowed_fields))
        ):
            raise ValueError("memory unavailable")
        return record

    @staticmethod
    def _entry(value: object, owner: str) -> MemoryEntry:
        if type(value) is not MemoryEntry:
            raise ValueError("memory unavailable")
        entry = MemoryEntry.model_validate(value.model_dump(warnings=False), strict=True)
        if (
            entry.user_id != owner
            or entry.field not in FIELDS
            or entry.deleted_at is not None
            or not valid_memory_id(entry.memory_id)
            or not entry.value.strip()
            or len(entry.value) > 500
        ):
            raise ValueError("memory unavailable")
        return entry

    def _read(self, owner: str, target_id: str | None = None):
        service = self.service()
        consent = self.validate_consent(service.get_consent(owner), owner)
        values = service.list_memories(owner)
        if type(values) is not tuple:
            raise ValueError("memory unavailable")
        entries = tuple(self._entry(value, owner) for value in values)
        if len({entry.memory_id for entry in entries}) != len(entries):
            raise ValueError("memory unavailable")
        if any(
            consent is None or not consent.granted or entry.field not in consent.allowed_fields
            for entry in entries
        ):
            raise ValueError("memory unavailable")
        if self.validate_consent(service.get_consent(owner), owner) != consent:
            raise MemoryStateChanged("memory changed")
        state = MemoryFormState(
            consent.model_dump_json().encode() if consent is not None else None,
            tuple(entry.model_dump_json().encode() for entry in entries),
            target_id,
        )
        return consent, entries, state

    def view(self, owner: str, action: str, target_id: str | None):
        consent, entries, state = self._read(owner, target_id)
        if action == "list":
            body = "<h1>本人记忆管理</h1>"
            body += (
                "<p>已授权字段: "
                + (", ".join(LABELS[f] for f in consent.allowed_fields if f in FIELDS) or "无")
                + "</p>"
                if consent and consent.granted
                else "<p>未授权跨会话记忆保存</p>"
            )
            body += (
                '<p><a href="/admin/memory/consent">独立记忆授权与字段选择</a></p>'
                '<p><a href="/admin/memory/revoke">撤回授权并删除全部本人记忆</a></p>'
            )
            for entry in entries:
                body += (
                    f"<p>{LABELS[entry.field]}: {html.escape(entry.value)}</p>"
                    f'<a href="/admin/memory/correct?memory_id={entry.memory_id}">纠正</a> '
                    f'<a href="/admin/memory/delete?memory_id={entry.memory_id}">删除</a>'
                )
            return body, state
        if action == "consent":
            body = (
                "<h1>本人记忆独立授权</h1><p>选中字段将跨会话本地保存并用于后续对话检索。"
                "相关记忆片段可能用于已配置的云端回答。实际数据发送和费用授权仍独立。"
                "风格同意不授予记忆同意。本页只授权本人, 不代他人同意。</p>"
                "<p>缩小字段会删除被取消字段的记忆。全部不选请使用撤回授权入口。</p>"
            )
            for field in FIELDS:
                checked = (
                    " checked"
                    if consent and consent.granted and field in consent.allowed_fields
                    else ""
                )
                body += (
                    f'<label><input type="checkbox" name="{field}" value="yes"{checked}>'
                    f"{LABELS[field]}</label>"
                )
        elif action == "revoke":
            body = (
                "<h1>撤回本人记忆授权</h1><p>撤回后删除全部本人记忆, 后续检索为空。"
                "重新同意不会恢复已删除内容。</p>"
            )
        else:
            target = next((entry for entry in entries if entry.memory_id == target_id), None)
            if target is None:
                raise MemoryTargetUnavailable("target unavailable")
            label = "纠正" if action == "correct" else "删除"
            body = f"<h1>{label}本人记忆</h1><p>{html.escape(target.value)}</p>"
            if action == "correct":
                body += (
                    '<label>新内容<textarea name="value" maxlength="500">'
                    f"{html.escape(target.value)}</textarea></label>"
                )
        return body, state

    def submit(self, owner: str, action: str, form: dict[str, str], expected: MemoryFormState):
        consent, entries, current = self._read(owner, expected.target_id)
        if current != expected:
            raise MemoryStateChanged("memory changed")
        service = self.service()
        target = next((e for e in entries if e.memory_id == expected.target_id), None)
        if action in {"correct", "delete"} and target is None:
            raise MemoryTargetUnavailable("target unavailable")
        now = self._clock()
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("memory clock unavailable")
        new_consent = None
        corrected = None
        profile_count = (
            len(service.list_profile(owner))
            if action == "revoke" and consent and PROFILE_PERMISSION in consent.allowed_fields
            else 0
        )
        if action == "consent":
            new_consent = ConsentRecord(
                user_id=owner,
                granted=True,
                granted_at=now,
                allowed_fields=sorted(
                    [f for f in FIELDS if f in form]
                    + (
                        [PROFILE_PERMISSION]
                        if consent
                        and consent.granted
                        and PROFILE_PERMISSION in consent.allowed_fields
                        else []
                    )
                ),
            )
        elif action == "correct":
            corrected = CorrectMemoryRequest(
                user_id=owner, memory_id=target.memory_id, value=form["value"], source_timestamp=now
            )
        # Once the public write is attempted, all failures are an unconfirmed result.
        try:
            if action == "consent":
                result = self.validate_consent(service.grant_consent(new_consent), owner)
                if result != new_consent:
                    raise ValueError("memory readback mismatch")
            elif action == "correct":
                result = self._entry(service.correct_memory(corrected), owner)
                wanted = target.model_copy(update={"value": form["value"], "source_timestamp": now})
                if result != wanted:
                    raise ValueError("memory readback mismatch")
            else:
                receipt = (
                    service.delete_memory(owner, target.memory_id)
                    if action == "delete"
                    else service.revoke_consent(owner)
                )
                if type(receipt) is not DeletionReceipt:
                    raise ValueError("memory readback mismatch")
                receipt = DeletionReceipt.model_validate(
                    receipt.model_dump(warnings=False), strict=True
                )
                if receipt.user_id != owner or receipt.deleted_count != (
                    1 if action == "delete" else len(entries) + profile_count
                ):
                    raise ValueError("memory readback mismatch")
            after_consent, after_entries, _ = self._read(owner)
            if action == "consent":
                wanted_entries = tuple(e for e in entries if e.field in new_consent.allowed_fields)
                if after_consent != new_consent or after_entries != wanted_entries:
                    raise ValueError("memory readback mismatch")
            elif action == "correct":
                expected_entries = {e.memory_id: e for e in entries}
                expected_entries[target.memory_id] = result
                if (
                    after_consent != consent
                    or {e.memory_id: e for e in after_entries} != expected_entries
                ):
                    raise ValueError("memory readback mismatch")
            elif action == "delete":
                if after_consent != consent or after_entries != tuple(
                    e for e in entries if e.memory_id != target.memory_id
                ):
                    raise ValueError("memory readback mismatch")
                self._verify_retrieval(owner, target.value, target.memory_id)
            else:
                if after_consent is None or after_consent.granted or after_entries:
                    raise ValueError("memory readback mismatch")
                if profile_count and service.list_profile(owner):
                    raise ValueError("memory readback mismatch")
                for entry in entries:
                    if self._verify_retrieval(owner, entry.value):
                        raise ValueError("memory readback mismatch")
        except Exception:
            raise MemoryResultUnconfirmed("memory result unconfirmed") from None

    def _verify_retrieval(self, owner: str, query: str, deleted_id: str | None = None):
        values = self.service().retrieve_memories(owner, query)
        if type(values) is not tuple:
            raise ValueError("memory unavailable")
        entries = tuple(self._entry(value, owner) for value in values)
        if deleted_id and any(e.memory_id == deleted_id for e in entries):
            raise ValueError("memory readback mismatch")
        return entries
