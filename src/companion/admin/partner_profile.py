"""Local management of one synthetic partner, including simulated self confirmation."""

import html
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime

from companion.admin.memory import (
    MemoryAdmin,
    MemoryResultUnconfirmed,
    MemoryStateChanged,
    MemoryTargetUnavailable,
)
from companion.contracts import ConsentRecord
from companion.memory import (
    PROFILE_PERMISSION,
    DeletionReceipt,
    MemoryAuthorizationError,
    MemoryService,
    ProfileItem,
    ProfileStatus,
    ProfileView,
)

SYNTHETIC_PARTNER_ID = "synthetic-partner"
ROOT = "/admin/partner"
SELF = ROOT + "/self"
ACTIONS = {
    "consent",
    "add",
    "approve",
    "edit",
    "disable",
    "delete",
    "self/confirm",
    "self/correct",
    "self/delete",
    "self/revoke",
}
ROUTES = {ROOT, SELF, *(ROOT + "/" + action for action in ACTIONS)}


@dataclass(frozen=True)
class PartnerFormState:
    consent: bytes | None
    entries: tuple[bytes, ...]
    target_id: str | None = None


class PartnerProfileAdmin:
    def __init__(self, service: Callable[[], MemoryService], clock: Callable[[], datetime]):
        self._service = service
        self._clock = clock

    @staticmethod
    def _item(value: ProfileItem) -> ProfileItem:
        item = ProfileItem.model_validate(value.model_dump(warnings=False), strict=True)
        if item.user_id != SYNTHETIC_PARTNER_ID:
            raise ValueError("profile unavailable")
        return item

    def _read(self):
        service = self._service()
        consent = MemoryAdmin.validate_consent(
            service.get_consent(SYNTHETIC_PARTNER_ID), SYNTHETIC_PARTNER_ID
        )
        items = tuple(self._item(item) for item in service.list_profile(SYNTHETIC_PARTNER_ID))
        if items and not self._authorized(consent):
            raise ValueError("profile unavailable")
        if service.get_consent(SYNTHETIC_PARTNER_ID) != consent:
            raise MemoryStateChanged("profile changed")
        return (
            consent,
            items,
            PartnerFormState(
                consent.model_dump_json().encode() if consent else None,
                tuple(item.model_dump_json().encode() for item in items),
            ),
        )

    @staticmethod
    def _authorized(consent: ConsentRecord | None) -> bool:
        return bool(consent and consent.granted and PROFILE_PERMISSION in consent.allowed_fields)

    def view(self, simulated_self: bool, make_form: Callable[..., str]) -> str:
        consent, items, state = self._read()
        body = (
            "<h1>模拟她本人: 对方画像（合成演示）</h1>"  # noqa: RUF001
            if simulated_self
            else "<h1>对方画像（合成演示）: 主人审定</h1>"  # noqa: RUF001
        )
        body += "<p>仅管理合成身份, 所有操作均为模拟; 模拟记录不得迁移为真实确认。</p>"
        body += (
            "<p>模拟授权: "
            + ("已授予画像保存" if self._authorized(consent) else "未授予画像授权")
            + "</p>"
        )
        if simulated_self:
            body += "<p>确认后可能用于聊天（会发到已配置的云端模型）</p>"  # noqa: RUF001
            body += make_form("self/revoke", state, "", "模拟撤回画像授权并删除全部画像")
        else:
            body += '<p><a href="/admin/partner/self">模拟她本人</a></p>'
            controls = (
                '<label>模拟画像授权<select name="mode">'
                '<option value="grant">授予</option>'
                '<option value="revoke">撤回并删除全部画像</option>'
                "</select></label>"
            )
            body += make_form("consent", state, controls, "模拟提交授权选择")
            body += make_form(
                "add",
                state,
                self._edit_controls()
                + '<label>证据日期<input type="date" name="evidence_date" required></label>'
                '<label>证据原句<textarea name="evidence_quote" maxlength="30"></textarea></label>'
                '<label>置信度<select name="confidence"><option>low</option>'
                "<option>med</option><option>high</option></select></label>",
                "模拟新增",
            )
        for item in items:
            if simulated_self and item.status not in {
                ProfileStatus.OWNER_APPROVED,
                ProfileStatus.CONFIRMED,
            }:
                continue
            expiry = item.valid_until.isoformat() if item.valid_until else "尚未确认"
            fields = {
                "topic": item.topic,
                "text": item.text,
                "source": item.source,
                "evidence_date": item.evidence_date,
                "有效期止": expiry,
            }
            if not simulated_self:
                fields |= {
                    "category": item.category,
                    "evidence_quote": item.evidence_quote or "",
                    "confidence": item.confidence,
                    "status": item.status,
                    "version": item.version,
                    "disabled_reason": item.disabled_reason or "",
                }
            body += (
                "<section><dl>"
                + "".join(
                    f"<dt>{key}</dt><dd>{html.escape(str(value))}</dd>"
                    for key, value in fields.items()
                )
                + "</dl>"
            )
            if item.disabled_reason == "conflict":
                body += "<p>冲突停用: 同一主题有不同表述, 请审查后纠正。</p>"
            bound = replace(state, target_id=item.profile_id)
            hidden = (
                f'<input type="hidden" name="profile_id" value="{item.profile_id}">'
                f'<input type="hidden" name="version" value="{item.version}">'
            )
            if simulated_self:
                chat = (
                    '<label><input type="checkbox" name="use_in_chat" value="yes">'
                    "模拟同意用于聊天</label>"
                )
                body += make_form("self/confirm", bound, hidden + chat, "模拟确认")
                correction = (
                    '<label>纠正内容<textarea name="text" maxlength="60" required>'
                    f"{html.escape(item.text)}</textarea></label>"
                )
                body += make_form(
                    "self/correct", bound, hidden + correction + chat, "模拟纠正后确认"
                )
                body += make_form("self/delete", bound, hidden, "模拟删除")
            else:
                if item.status == ProfileStatus.CANDIDATE:
                    body += make_form("approve", bound, hidden, "模拟审定")
                body += make_form("edit", bound, hidden + self._edit_controls(item), "模拟编辑")
                body += make_form("disable", bound, hidden, "模拟停用")
                body += make_form("delete", bound, hidden, "模拟删除")
            body += "</section>"
        return body

    @staticmethod
    def _edit_controls(item: ProfileItem | None = None) -> str:
        body = (
            f'<label>主题<input name="topic" maxlength="12" required value="'
            f'{html.escape(item.topic) if item else ""}"></label>'
            '<label>内容<textarea name="text" maxlength="60" required>'
            f"{html.escape(item.text) if item else ''}</textarea></label>"
            '<label>类别<select name="category">'
        )
        for category in ("habit", "preference", "reminder_pref"):
            selected = " selected" if item and item.category == category else ""
            body += f'<option value="{category}"{selected}>{category}</option>'
        return body + "</select></label>"

    @staticmethod
    def _validate_form(action: str, form: dict[str, str], state: PartnerFormState):
        required = {"csrf", "partner_token"}
        optional = set()
        if state.target_id:
            required |= {"profile_id", "version"}
            if form.get("profile_id") != state.target_id:
                raise MemoryTargetUnavailable("target unavailable")
        if action == "consent":
            required.add("mode")
            if form.get("mode") not in {"grant", "revoke"}:
                raise ValueError("invalid form")
        if action in {"add", "edit"}:
            required |= {"topic", "text", "category"}
        if action == "add":
            required |= {"evidence_date", "evidence_quote", "confidence"}
        if action == "self/correct":
            required.add("text")
        if action in {"self/confirm", "self/correct"}:
            optional.add("use_in_chat")
            if "use_in_chat" in form and form["use_in_chat"] != "yes":
                raise ValueError("invalid form")
        if not required <= set(form) <= required | optional:
            raise ValueError("invalid form")

    def submit(self, action: str, form: dict[str, str], expected: PartnerFormState):
        self._validate_form(action, form, expected)
        consent, items, state = self._read()
        if replace(state, target_id=expected.target_id) != expected:
            raise MemoryStateChanged("profile changed")
        target = next((i for i in items if i.profile_id == expected.target_id), None)
        if target is not None:
            if form["version"] != str(target.version):
                raise MemoryStateChanged("profile changed")
            if action == "approve" and target.status != ProfileStatus.CANDIDATE:
                raise ValueError("invalid status")
            if action in {"self/confirm", "self/correct", "self/delete"} and target.status not in {
                ProfileStatus.OWNER_APPROVED,
                ProfileStatus.CONFIRMED,
            }:
                raise MemoryTargetUnavailable("target unavailable")
        if action != "consent" and action != "self/revoke" and not self._authorized(consent):
            raise MemoryAuthorizationError("profile not authorized")
        changes = {}
        if action in {"add", "edit"}:
            changes = ProfileView.model_validate(
                {k: form[k] for k in ("topic", "text", "category")}
            ).model_dump()
        elif action == "self/correct":
            changes = {
                "text": ProfileView.model_validate(
                    {"topic": target.topic, "category": target.category, "text": form["text"]}
                ).text
            }
        if action == "add":
            changes |= {
                "source": "owner_added",
                "evidence_date": date.fromisoformat(form["evidence_date"]),
                "evidence_quote": form["evidence_quote"] or None,
                "confidence": form["confidence"],
            }
            if len(form["evidence_quote"]) > 30 or form["confidence"] not in {"low", "med", "high"}:
                raise ValueError("invalid evidence")
        service = self._service()
        user = SYNTHETIC_PARTNER_ID
        # Once a write starts, failures include a possible partial correction.
        try:
            if action in {"consent", "self/revoke"}:
                grant = action == "consent" and form["mode"] == "grant"
                fields = set(consent.allowed_fields) if consent and consent.granted else set()
                fields = fields | {PROFILE_PERMISSION} if grant else fields - {PROFILE_PERMISSION}
                if grant or fields:
                    wanted_consent = ConsentRecord(
                        user_id=user,
                        granted=True,
                        granted_at=self._clock(),
                        allowed_fields=sorted(fields),
                    )
                    if service.grant_consent(wanted_consent) != wanted_consent:
                        raise ValueError("readback mismatch")
                else:
                    receipt = service.revoke_consent(user)
                    self._receipt(receipt, len(items))
                    wanted_consent = ConsentRecord(
                        user_id=user, granted=False, granted_at=None, allowed_fields=[]
                    )
            elif action in {"delete", "self/delete"}:
                self._receipt(service.delete_profile(user_id=user, profile_id=target.profile_id), 1)
            else:
                specified = changes
                if action == "add":
                    result = self._item(service.add_profile_candidate(user_id=user, **changes))
                elif action == "approve":
                    result = self._item(
                        service.approve_profile(user_id=user, profile_id=target.profile_id)
                    )
                    specified = {"status": ProfileStatus.OWNER_APPROVED}
                elif action == "disable":
                    result = self._item(
                        service.disable_profile(
                            user_id=user, profile_id=target.profile_id, reason="owner"
                        )
                    )
                    specified = {"status": ProfileStatus.DISABLED, "disabled_reason": "owner"}
                elif action == "edit":
                    result = self._item(
                        service.edit_profile(user_id=user, profile_id=target.profile_id, **changes)
                    )
                else:
                    if action == "self/correct":
                        edited = self._item(
                            service.edit_profile(
                                user_id=user, profile_id=target.profile_id, **changes
                            )
                        )
                        if edited.profile_id != target.profile_id or edited.text != changes["text"]:
                            raise ValueError("readback mismatch")
                        target = edited
                    result = self._item(
                        service.confirm_profile(
                            user_id=user,
                            profile_id=target.profile_id,
                            version=(
                                target.version if action == "self/correct" else int(form["version"])
                            ),
                            subject_id=user,
                            use_in_chat="use_in_chat" in form,
                            simulated=True,
                        )
                    )
                    specified |= {"simulated": True, "use_in_chat": "use_in_chat" in form}
                if target is not None and result.profile_id != target.profile_id:
                    raise ValueError("readback mismatch")
                actual_fields = result.model_dump()
                if any(actual_fields[key] != value for key, value in specified.items()):
                    raise ValueError("readback mismatch")
            after_consent, after_items, _ = self._read()
            if action in {"consent", "self/revoke"}:
                if after_consent != wanted_consent or (not grant and after_items):
                    raise ValueError("readback mismatch")
            elif action in {"delete", "self/delete"}:
                if any(i.profile_id == target.profile_id for i in after_items):
                    raise ValueError("readback mismatch")
            else:
                readback = next((i for i in after_items if i.profile_id == result.profile_id), None)
                if readback != result:
                    raise ValueError("readback mismatch")
        except Exception:
            raise MemoryResultUnconfirmed("profile result unconfirmed") from None

    @staticmethod
    def _receipt(value: DeletionReceipt, count: int):
        receipt = DeletionReceipt.model_validate(value.model_dump(warnings=False), strict=True)
        if receipt.user_id != SYNTHETIC_PARTNER_ID or receipt.deleted_count != count:
            raise ValueError("readback mismatch")
