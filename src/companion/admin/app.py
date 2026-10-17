from __future__ import annotations

import hashlib
import hmac
import html
import os
import secrets
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from http.cookies import SimpleCookie
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

from companion.admin.character import render_character_form
from companion.admin.config import AdminConfig, AdminConfigError, _valid_owner_id, load_admin_config
from companion.admin.focus_labeling import (
    FocusLabelStore,
    focus_position,
    parse_focus_form,
    render_focus_round,
)
from companion.admin.import_source import _FixedImportBatchSource, _LoadedBatch
from companion.admin.labeling import LabelStore, render_round, render_round_list, round_is_open
from companion.admin.memory import (
    FIELDS,
    MemoryAdmin,
    MemoryResultUnconfirmed,
    MemoryStateChanged,
    MemoryTargetUnavailable,
    valid_memory_id,
)
from companion.admin.partner_profile import (
    ROOT as PARTNER_ROOT,
)
from companion.admin.partner_profile import (
    ROUTES as PARTNER_ROUTES,
)
from companion.admin.partner_profile import (
    SELF as PARTNER_SELF,
)
from companion.admin.partner_profile import (
    PartnerFormState,
    PartnerProfileAdmin,
)
from companion.admin.session import _BrowserSession
from companion.admin.snippets import (
    TERM_FIELDS,
    SnippetStore,
    parse_range,
    parse_terms,
    render_audit,
    render_form,
    render_result,
)
from companion.admin.style_routes import _StyleRoutesMixin
from companion.contracts import ChatRecord
from companion.dialogue.character_settings import (
    CharacterSettingsConflict,
    CharacterSettingsInvalidName,
    CharacterSettingsUnavailable,
    read_character_name,
    write_character_name,
)
from companion.dialogue.usage import UsageSnapshot
from companion.importer import (
    ImportBatch,
)
from companion.memory import MemoryAuthorizationError, MemoryService
from companion.privacy import SafeCandidatePackage, select_safe_candidates
from companion.privacy.style_candidate_approval import StyleCandidateApprovalStore
from companion.runtime.settings import Settings
from companion.style import (
    SingleTrustedUserStyleEvidenceStore,
    SingleTrustedUserStyleProfileStore,
)

__all__ = ["AdminApp", "AdminConfig", "AdminConfigError", "build_admin_app", "load_admin_config"]

if TYPE_CHECKING:
    from companion.admin.phase1 import AdminSourceApprovalGateway
    from companion.admin.source_seal import StyleSourceSealStore


class AdminApp(_StyleRoutesMixin):
    """In-memory WSGI entry; production has no batch provider in this slice."""

    def __init__(
        self,
        owner_id: str,
        *,
        host: str,
        port: int,
        clock: Callable[[], float] = time.monotonic,
        batch_loader: Callable[[], ImportBatch | _LoadedBatch] | None = None,
        candidate_reader: (
            Callable[[str, tuple[ChatRecord, ...]], SafeCandidatePackage] | None
        ) = None,
        approval_store: StyleCandidateApprovalStore | None = None,
        runtime_selector: Callable[..., SafeCandidatePackage] | None = None,
        source_seal: StyleSourceSealStore | None = None,
        source_gateway: AdminSourceApprovalGateway | None = None,
        phase1_store: SingleTrustedUserStyleProfileStore | None = None,
        phase1_root: Path | None = None,
        phase5_store: SingleTrustedUserStyleEvidenceStore | None = None,
        usage_reader: Callable[[], UsageSnapshot] | None = None,
        memory_factory: Callable[[], MemoryService] | None = None,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        snippet_root: Path | None = None,
        label_root: Path | None = None,
        character_settings: Settings | None = None,
    ) -> None:
        if host != "127.0.0.1" or type(port) is not int or not 0 < port <= 65535:
            raise ValueError("admin requires a loopback address and assigned port")
        if not _valid_owner_id(owner_id):
            raise ValueError("invalid trusted owner")
        self.host = host
        self.port = port
        self._owner_id = owner_id
        self._clock = clock
        self._batch_loader = batch_loader
        self._candidate_reader = candidate_reader
        self._approval_store = approval_store
        self._runtime_selector = runtime_selector
        self._source_seal = source_seal
        self._source_gateway = source_gateway
        self._phase1_store = phase1_store
        self._phase1_root = phase1_root
        self._phase5_store = phase5_store
        self._usage_reader = usage_reader
        self._memory = MemoryAdmin(memory_factory, wall_clock)
        self._partner = PartnerProfileAdmin(self._memory.service, wall_clock)
        self._wall_clock = wall_clock
        self._snippet_root = snippet_root
        self._label_root = label_root
        self._character_settings = character_settings
        self._labels_tokens: dict[str, tuple[str, str]] = {}
        self._start_code_digest: bytes | None = None
        self._start_code_expires = 0.0
        self._preauth: dict[str, _BrowserSession] = {}
        self._sessions: dict[str, _BrowserSession] = {}
        self._closed = False

    def issue_start_code(self) -> str:
        if self._closed:
            raise RuntimeError("admin closed")
        code = secrets.token_urlsafe(32)
        self._start_code_digest = hashlib.sha256(code.encode()).digest()
        self._start_code_expires = self._clock() + 300
        return code

    def close(self) -> None:
        self._closed = True
        self._start_code_digest = None
        self._preauth.clear()
        self._sessions.clear()

    @staticmethod
    def _cookie(environ: dict[str, Any], name: str) -> str:
        jar = SimpleCookie()
        try:
            jar.load(environ.get("HTTP_COOKIE", ""))
            return jar[name].value if name in jar else ""
        except Exception:
            return ""

    @staticmethod
    def _token(session: _BrowserSession, action: str) -> str:
        return hmac.new(session.secret.encode(), action.encode(), hashlib.sha256).hexdigest()

    def _form(self, environ: dict[str, Any], *, max_bytes: int = 4096) -> dict[str, str]:
        try:
            size = int(environ.get("CONTENT_LENGTH", "0"))
            if size < 0 or size > max_bytes:
                return {}
            raw = environ["wsgi.input"].read(size).decode("utf-8")
            parsed = parse_qs(raw, keep_blank_values=True, strict_parsing=True)
            if any(len(values) != 1 for values in parsed.values()):
                return {}
            return {key: values[0] for key, values in parsed.items()}
        except (ValueError, UnicodeError, KeyError):
            return {}


    @staticmethod
    def _response(
        start_response: Callable[..., Any],
        status: str,
        body: str,
        *,
        cookie: str = "",
        location: str = "",
    ) -> Iterable[bytes]:
        data = body.encode("utf-8")
        headers = [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(data))),
            ("Cache-Control", "no-store"),
            (
                "Content-Security-Policy",
                "default-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
            ),
            ("X-Content-Type-Options", "nosniff"),
        ]
        if cookie:
            headers.append(("Set-Cookie", cookie))
        if location:
            headers.append(("Location", location))
        start_response(status, headers)
        return [data]

    def _memory_response(
        self, environ: dict[str, Any], start_response: Callable[..., Any], session: _BrowserSession
    ) -> Iterable[bytes]:
        path = environ["PATH_INFO"]
        action = "list" if path == "/admin/memory" else path.rsplit("/", 1)[1]
        method = environ["REQUEST_METHOD"]
        if method == "GET":
            session.memory_actions.pop(action, None)
            target_id = None
            try:
                query = parse_qs(
                    environ.get("QUERY_STRING", ""), keep_blank_values=True, strict_parsing=True
                )
                if action in {"correct", "delete"}:
                    if set(query) != {"memory_id"} or len(query["memory_id"]) != 1:
                        raise ValueError("invalid query")
                    target_id = query["memory_id"][0]
                    if not valid_memory_id(target_id):
                        raise ValueError("invalid query")
                elif query:
                    raise ValueError("invalid query")
            except ValueError:
                return self._response(start_response, "400 Bad Request", "记忆请求无效")
            try:
                body, state = self._memory.view(session.owner_id, action, target_id)
            except MemoryTargetUnavailable:
                return self._response(start_response, "403 Forbidden", "Forbidden")
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "记忆管理不可用")
            if action != "list":
                token = secrets.token_urlsafe(32)
                session.memory_actions[action] = (token, state)
                csrf = self._token(session, "memory-" + action)
                body = (
                    f'<form method="post" action="{path}">'
                    + body
                    + f'<input type="hidden" name="csrf" value="{csrf}">'
                    + f'<input type="hidden" name="memory_token" value="{token}">'
                    + '<label><input type="checkbox" name="confirm" value="yes">'
                    + "明确确认本次操作</label>"
                    + "<button>确认</button></form>"
                )
            return self._response(
                start_response,
                "200 OK",
                body
                + '<p><a href="/admin/memory">返回记忆管理</a></p>'
                + '<p><a href="/admin/home">返回主页</a></p>',
            )
        if method != "POST" or action == "list":
            return self._response(start_response, "404 Not Found", "Not found")
        saved = session.memory_actions.pop(action, None)
        form = self._form(environ, max_bytes=8192)
        if (
            saved is None
            or not hmac.compare_digest(form.get("memory_token", ""), saved[0])
            or not hmac.compare_digest(
                form.get("csrf", ""), self._token(session, "memory-" + action)
            )
        ):
            return self._response(start_response, "403 Forbidden", "Forbidden")
        base = {"csrf", "memory_token", "confirm"}
        fields = set(form) - base
        valid = set(form) >= base and form["confirm"] == "yes"
        if action == "consent":
            valid = valid and bool(fields) and fields <= set(FIELDS)
            valid = valid and all(form[f] == "yes" for f in fields)
        elif action == "correct":
            valid = valid and fields == {"value"}
            valid = (
                valid and bool(form.get("value", "").strip()) and len(form.get("value", "")) <= 500
            )
        else:
            valid = valid and not fields
        if not valid or environ.get("QUERY_STRING"):
            return self._response(start_response, "400 Bad Request", "记忆确认无效, 全部未选请撤权")
        try:
            self._memory.submit(session.owner_id, action, form, saved[1])
        except MemoryStateChanged:
            return self._response(start_response, "409 Conflict", "记忆状态已变化, 请重新查看")
        except MemoryResultUnconfirmed:
            return self._response(start_response, "503 Service Unavailable", "记忆操作结果未确认")
        except MemoryTargetUnavailable:
            return self._response(start_response, "403 Forbidden", "Forbidden")
        except Exception:
            return self._response(start_response, "503 Service Unavailable", "记忆管理不可用")
        return self._response(
            start_response,
            "200 OK",
            '<p>记忆操作已完成并读回确认</p><a href="/admin/memory">返回记忆管理</a>',
        )

    def _partner_response(
        self, environ: dict[str, Any], start_response: Callable[..., Any], session: _BrowserSession
    ) -> Iterable[bytes]:
        path = environ["PATH_INFO"]
        method = environ["REQUEST_METHOD"]
        page = path in {PARTNER_ROOT, PARTNER_SELF}
        if (page and method != "GET") or (not page and method != "POST"):
            return self._response(start_response, "404 Not Found", "Not found")
        if method == "GET":
            simulated_self = path == PARTNER_SELF
            session.partner_actions = {
                token: saved for token, saved in session.partner_actions.items()
                if saved[0].startswith(PARTNER_SELF + "/") != simulated_self
            }
            if environ.get("QUERY_STRING"):
                return self._response(start_response, "400 Bad Request", "模拟画像请求无效")

            def make_form(action: str, state: PartnerFormState, controls: str, label: str) -> str:
                route = PARTNER_ROOT + "/" + action
                token = secrets.token_urlsafe(32)
                session.partner_actions[token] = (route, state)
                csrf = self._token(session, "partner-" + action)
                return (
                    f'<form method="post" action="{route}">'
                    f'<input type="hidden" name="csrf" value="{csrf}">'
                    f'<input type="hidden" name="partner_token" value="{token}">'
                    + controls + f"<button>{label}</button></form>"
                )

            try:
                body = self._partner.view(simulated_self, make_form)
            except Exception:
                return self._response(
                    start_response, "503 Service Unavailable", "模拟画像管理不可用"
                )
            return self._response(start_response, "200 OK", body)
        form = self._form(environ, max_bytes=8192)
        saved = session.partner_actions.pop(form.get("partner_token", ""), None)
        action = path.removeprefix(PARTNER_ROOT + "/")
        if saved is None or saved[0] != path or not hmac.compare_digest(
            form.get("csrf", "").encode(), self._token(session, "partner-" + action).encode()
        ):
            return self._response(start_response, "403 Forbidden", "Forbidden")
        if environ.get("QUERY_STRING"):
            return self._response(start_response, "400 Bad Request", "模拟画像请求无效")
        try:
            self._partner.submit(action, form, saved[1])
        except MemoryStateChanged:
            return self._response(
                start_response, "409 Conflict", "模拟画像状态或版本已变化, 请刷新页面"
            )
        except MemoryTargetUnavailable:
            return self._response(start_response, "403 Forbidden", "Forbidden")
        except MemoryAuthorizationError:
            return self._response(
                start_response, "403 Forbidden", "未授予画像授权, 请先授予模拟授权"
            )
        except MemoryResultUnconfirmed:
            return self._response(
                start_response, "503 Service Unavailable", "模拟画像操作结果未确认"
            )
        except ValueError:
            return self._response(start_response, "400 Bad Request", "模拟画像提交无效")
        except Exception:
            return self._response(start_response, "503 Service Unavailable", "模拟画像管理不可用")
        destination = PARTNER_SELF if action.startswith("self/") else PARTNER_ROOT
        return self._response(start_response, "200 OK", "<p>模拟画像操作已完成并读回确认</p>"
                              f'<a href="{destination}">返回模拟画像管理</a>')

    def _snippet_response(
        self, environ: dict[str, Any], start_response: Callable[..., Any], session: _BrowserSession
    ) -> Iterable[bytes]:
        method = environ["REQUEST_METHOD"]
        audit = environ["PATH_INFO"] == "/admin/style/snippets/audit"
        if method not in {"GET", "POST"}:
            return self._response(start_response, "404 Not Found", "Not found")
        if method == "GET" and audit:
            session.audit_token = None
        elif method == "GET":
            session.snippets_token = None
        if self._snippet_root is None:
            return self._response(start_response, "503 Service Unavailable", "片段功能未就绪")
        store = SnippetStore(self._snippet_root)
        if audit:
            return self._snippet_audit_response(environ, start_response, session, store)
        if method == "POST":
            form = self._form(environ, max_bytes=32 * 1024)
            if not hmac.compare_digest(form.get("csrf", ""), self._token(session, "snippets")):
                return self._response(start_response, "403 Forbidden", "Forbidden")
            once = session.snippets_token
            session.snippets_token = None
            if once is None or not hmac.compare_digest(form.get("snippets_token", ""), once):
                return self._response(start_response, "403 Forbidden", "Forbidden")
            if set(form) != {"csrf", "snippets_token", "confirm", *TERM_FIELDS} or (
                form["confirm"] != "yes"
            ):
                return self._response(start_response, "400 Bad Request", "片段生成确认无效")
        try:
            records = self._confirmed_context(session)
        except ValueError:
            return self._response(
                start_response,
                "503 Service Unavailable",
                '<a href="/admin/import/preview">请先完成说话人映射</a>',
            )
        if method == "GET":
            try:
                terms, counts = store.terms(), store.counts()
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "片段功能未就绪")
            session.snippets_token = secrets.token_urlsafe(32)
            body = render_form(
                csrf=self._token(session, "snippets"),
                token=session.snippets_token,
                terms=terms,
                counts=counts,
            )
            return self._response(start_response, "200 OK", body)
        try:
            groups = parse_terms(form)
        except ValueError:
            return self._response(start_response, "400 Bad Request", "片段词无效")
        try:
            counts = store.generate(
                records, groups=groups, created_at=self._wall_clock(), seed=secrets.randbits(32)
            )
        except Exception:
            return self._response(start_response, "503 Service Unavailable", "片段生成失败")
        return self._response(start_response, "200 OK", render_result(counts))

    def _snippet_audit_response(
        self,
        environ: dict[str, Any],
        start_response: Callable[..., Any],
        session: _BrowserSession,
        store: SnippetStore,
    ) -> Iterable[bytes]:
        try:
            sample = store.audit_sample()
            deleted_ids, deleted_ranges = store.exclusions()
        except Exception:
            return self._response(start_response, "503 Service Unavailable", "片段功能未就绪")
        if environ["REQUEST_METHOD"] == "GET":
            session.audit_token = secrets.token_urlsafe(32)
            body = render_audit(
                csrf=self._token(session, "snippets-audit"),
                token=session.audit_token,
                sample=sample,
                excluded=(len(deleted_ids), len(deleted_ranges)),
            )
            return self._response(start_response, "200 OK", body)
        form = self._form(environ, max_bytes=32 * 1024)
        if not hmac.compare_digest(form.get("csrf", ""), self._token(session, "snippets-audit")):
            return self._response(start_response, "403 Forbidden", "Forbidden")
        once = session.audit_token
        session.audit_token = None
        if once is None or not hmac.compare_digest(form.get("audit_token", ""), once):
            return self._response(start_response, "403 Forbidden", "Forbidden")
        known = {f"delete_{item.snippet_id}" for item in sample}
        selected = set(form) - {"csrf", "audit_token", "from", "to"}
        try:
            if (
                not {"from", "to"}.issubset(form)
                or not selected.issubset(known)
                or any(form[key] != "yes" for key in selected)
            ):
                raise ValueError("audit form invalid")
            between = parse_range(form["from"], form["to"])
        except ValueError:
            return self._response(start_response, "400 Bad Request", "删除条件无效")
        if not selected and between is None:
            return self._response(start_response, "400 Bad Request", "删除条件无效")
        ids = frozenset(key.removeprefix("delete_") for key in selected)
        try:
            deleted = store.delete(ids, between)
        except Exception:
            return self._response(start_response, "503 Service Unavailable", "片段库未就绪")
        return self._response(
            start_response,
            "200 OK",
            f"<p>已删除 {deleted} 段</p>"
            '<p><a href="/admin/style/snippets/audit">返回抽样审核</a></p>',
        )

    def _label_response(
        self, environ: dict[str, Any], start_response: Callable[..., Any], session: _BrowserSession
    ) -> Iterable[bytes]:
        method = environ["REQUEST_METHOD"]
        path = environ["PATH_INFO"]
        listing = path == "/admin/style/labels"
        if method not in {"GET", "POST"} or (listing and method != "GET"):
            return self._response(start_response, "404 Not Found", "Not found")
        if method == "GET":
            self._labels_tokens.pop(session.secret, None)
        if self._label_root is None:
            return self._response(start_response, "503 Service Unavailable", "标注任务未就绪")
        store = LabelStore(self._label_root)
        focus_store = FocusLabelStore(self._label_root)
        try:
            tasks = store.tasks()
            now = self._wall_clock()
            if listing:
                progress = {
                    round_["id"]: len(
                        focus_store.steps(round_["id"])[1]
                        if round_.get("kind") == "focus"
                        else store.answers(round_["id"])
                    )
                    for round_ in tasks["rounds"]
                }
                body = render_round_list(tasks, progress, now)
                return self._response(start_response, "200 OK", body)
            round_id = path.removeprefix("/admin/style/labels/")
            round_ = next((entry for entry in tasks["rounds"] if entry["id"] == round_id), None)
            if round_ is None:
                return self._response(start_response, "404 Not Found", "Not found")
            opened = round_is_open(round_, now)
            if method == "POST" and not opened:
                return self._response(start_response, "403 Forbidden", "Forbidden")
            focus = round_.get("kind") == "focus"
            if focus:
                first, answers = focus_store.steps(round_id)
                item_id, step = focus_position(round_, first, answers)
            else:
                answers = store.answers(round_id)
                item_id = next((item for item in round_["order"] if item not in answers), None)
        except (OSError, ValueError, UnicodeError):
            return self._response(start_response, "503 Service Unavailable", "标注任务未就绪")
        csrf_action = f"labels:{round_id}"
        if method == "GET":
            token = secrets.token_urlsafe(32) if opened else ""
            if opened:
                self._labels_tokens[session.secret] = (round_id, token)
            if focus:
                body = render_focus_round(
                    tasks=tasks, round_=round_, marked=len(answers), item_id=item_id,
                    step=step, first=first.get(item_id or ""), opened=opened,
                    csrf=self._token(session, csrf_action), token=token,
                )
            else:
                body = render_round(
                    tasks=tasks, round_=round_, marked=len(answers), item_id=item_id,
                    opened=opened, csrf=self._token(session, csrf_action), token=token,
                )
            return self._response(start_response, "200 OK", body)
        form = self._form(environ)
        if not hmac.compare_digest(
            form.get("csrf", "").encode(), self._token(session, csrf_action).encode()
        ):
            return self._response(start_response, "403 Forbidden", "Forbidden")
        once = self._labels_tokens.pop(session.secret, None)
        if once is None or once[0] != round_id or not hmac.compare_digest(
            form.get("labels_token", "").encode(), once[1].encode()
        ):
            return self._response(start_response, "303 See Other", "", location=path)
        if focus:
            return self._focus_label_post(
                start_response, focus_store, form, tasks, round_id, item_id, step, path
            )
        if (
            set(form) != {"csrf", "labels_token", "action", "item_id", "label"}
            or form["action"] not in {"label", "undo"}
            or form["item_id"] != (item_id if item_id is not None else "")
            or (form["action"] == "label" and (item_id is None or form["label"] not in {
                "?", *(label["code"] for label in tasks["labels"])
            }))
        ):
            return self._response(start_response, "400 Bad Request", "标注请求无效")
        try:
            if form["action"] == "undo":
                store.undo(round_id)
            else:
                store.record(round_id, form["item_id"], form["label"])
        except (OSError, ValueError, UnicodeError):
            return self._response(start_response, "503 Service Unavailable", "标注任务未就绪")
        return self._response(start_response, "303 See Other", "", location=path)

    def _focus_label_post(
        self,
        start_response: Callable[..., Any],
        store: FocusLabelStore,
        form: dict[str, str],
        tasks: dict[str, Any],
        round_id: str,
        item_id: str | None,
        step: int | None,
        path: str,
    ) -> Iterable[bytes]:
        item = tasks["items"][item_id] if item_id is not None else None
        try:
            action, value = parse_focus_form(form, item, item_id, step)
        except ValueError:
            return self._response(start_response, "400 Bad Request", "标注请求无效")
        try:
            if action == "undo":
                store.undo(round_id)
            elif action == "step1":
                store.record_step1(round_id, form["item_id"], *value)
            else:
                store.record_step2(round_id, form["item_id"], value)
        except (OSError, ValueError, UnicodeError):
            return self._response(start_response, "503 Service Unavailable", "标注任务未就绪")
        return self._response(start_response, "303 See Other", "", location=path)

    def _usage_body(self) -> tuple[bool, str]:
        try:
            if self._usage_reader is None:
                raise ValueError("usage unavailable")
            result = self._usage_reader()
            if type(result) is not UsageSnapshot:
                raise ValueError("usage unavailable")
            snapshot = UsageSnapshot.model_validate(
                result.model_dump(mode="python", warnings=False)
            )
            if snapshot.monthly_cap_cny != 100:
                raise ValueError("usage unavailable")
            notice = {
                "allowed": "下一单请求可预留, 实际请求仍须重新检查预算。",
                "blocked": (
                    "100 元已占用, 暂停新的付费请求。"
                    if snapshot.accounted_cny >= snapshot.monthly_cap_cny
                    else "剩余不足下一单请求的保守预留, 暂停新的付费请求。"
                ),
                "review_required": "费用需核对并暂停付费。",
            }[snapshot.status]
            body = (
                f"<p>上海月份: {html.escape(snapshot.month)}</p>"
                f"<p>本地已记账: {snapshot.settled_cny:f} 元</p>"
                f"<p>未结算预留: {snapshot.reserved_cny:f} 元</p>"
                f"<p>总占用: {snapshot.accounted_cny:f} 元</p>"
                f"<p>月度上限: {snapshot.monthly_cap_cny:f} 元</p>"
                f"<p>{notice}</p>"
                "<p>这些金额是本地记账和预留, 不是供应商账单, "
                "页面状态不授予付费或数据发送许可。</p>"
            )
        except Exception:
            return False, "<p>费用状态不可用</p>"
        return True, body

    def _character_response(
        self,
        environ: dict[str, Any],
        start_response: Callable[..., Any],
        session: _BrowserSession,
    ) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD")
        if method not in {"GET", "POST"}:
            return self._response(start_response, "404 Not Found", "Not Found")
        if self._character_settings is None:
            return self._response(
                start_response, "503 Service Unavailable", "角色名称配置不可用"
            )
        if method == "GET":
            session.character_form = None
            if environ.get("QUERY_STRING"):
                return self._response(start_response, "400 Bad Request", "角色名称请求无效")
            try:
                name, digest = read_character_name(self._character_settings)
            except CharacterSettingsUnavailable as exc:
                return self._response(start_response, "503 Service Unavailable", str(exc))
            token = secrets.token_urlsafe(32)
            session.character_form = (token, digest)
            return self._response(
                start_response,
                "200 OK",
                render_character_form(
                    name=name, csrf=self._token(session, "character"), token=token
                ),
            )
        form = self._form(environ)
        if environ.get("QUERY_STRING") or set(form) != {
            "csrf",
            "character_token",
            "name",
            "confirm",
        }:
            return self._response(start_response, "400 Bad Request", "角色名称请求无效")
        if not hmac.compare_digest(
            form["csrf"].encode(), self._token(session, "character").encode()
        ):
            return self._response(start_response, "403 Forbidden", "Forbidden")
        binding = session.character_form
        session.character_form = None
        if binding is None or not hmac.compare_digest(
            form["character_token"].encode(), binding[0].encode()
        ):
            return self._response(start_response, "403 Forbidden", "Forbidden")
        if form["confirm"] != "yes":
            return self._response(start_response, "400 Bad Request", "角色名称请求无效")
        try:
            write_character_name(self._character_settings, form["name"], expected_digest=binding[1])
        except CharacterSettingsInvalidName as exc:
            return self._response(start_response, "400 Bad Request", str(exc))
        except CharacterSettingsConflict as exc:
            return self._response(start_response, "409 Conflict", str(exc))
        except CharacterSettingsUnavailable as exc:
            return self._response(start_response, "503 Service Unavailable", str(exc))
        return self._response(
            start_response,
            "200 OK",
            "<p>角色名称已保存，重启对话后生效。</p>"  # noqa: RUF001
            '<p><a href="/admin/character">返回角色名称</a></p>',
        )

    def __call__(
        self, environ: dict[str, Any], start_response: Callable[..., Any]
    ) -> Iterable[bytes]:
        def forbidden() -> Iterable[bytes]:
            return self._response(start_response, "403 Forbidden", "Forbidden")

        if self._closed or environ.get("HTTP_HOST") != f"{self.host}:{self.port}":
            return forbidden()
        method = environ.get("REQUEST_METHOD")
        path = environ.get("PATH_INFO")
        if (
            method == "GET"
            and (path in {"/admin/home", "/admin/usage", "/admin/character"}
                 or str(path).startswith(("/admin/memory", PARTNER_ROOT)))
            and environ.get("HTTP_ORIGIN")
            and environ["HTTP_ORIGIN"] != f"http://{self.host}:{self.port}"
        ):
            return forbidden()
        if method == "POST" and environ.get("HTTP_ORIGIN") != f"http://{self.host}:{self.port}":
            return forbidden()
        now = self._clock()
        if method == "GET" and path == "/admin":
            for expired in tuple(self._preauth):
                if now >= self._preauth[expired].expires:
                    self._preauth.pop(expired)
            if len(self._preauth) >= 64:
                self._preauth.pop(next(iter(self._preauth)))
            token = secrets.token_urlsafe(32)
            session = _BrowserSession(
                self._owner_id, secrets.token_urlsafe(32), now + 300, now + 300
            )
            self._preauth[token] = session
            csrf = self._token(session, "unlock")
            body = (
                '<h1>本地管理</h1><form method="post" action="/admin/unlock">'
                f'<input type="hidden" name="csrf" value="{csrf}">'
                '<input type="password" name="code" autocomplete="off">'
                "<button>解锁</button></form>"
            )
            return self._response(
                start_response,
                "200 OK",
                body,
                cookie=f"admin_preauth={token}; HttpOnly; SameSite=Strict; Path=/admin",
            )
        if method == "POST" and path == "/admin/unlock":
            pretoken = self._cookie(environ, "admin_preauth")
            pre = self._preauth.get(pretoken)
            form = self._form(environ)
            code = form.get("code", "")
            if (
                pre is None
                or now >= pre.expires
                or self._start_code_digest is None
                or now >= self._start_code_expires
                or not hmac.compare_digest(form.get("csrf", ""), self._token(pre, "unlock"))
                or not hmac.compare_digest(
                    hashlib.sha256(code.encode()).digest(), self._start_code_digest
                )
            ):
                return forbidden()
            self._preauth.pop(pretoken, None)
            self._start_code_digest = None
            token = secrets.token_urlsafe(32)
            self._sessions[token] = _BrowserSession(
                self._owner_id, secrets.token_urlsafe(32), now + 900, now + 7200
            )
            return self._response(
                start_response,
                "303 See Other",
                "Unlocked",
                cookie=f"admin_session={token}; HttpOnly; SameSite=Strict; Path=/admin",
                location="/admin/home",
            )
        token = self._cookie(environ, "admin_session")
        session = self._sessions.get(token)
        if session is None:
            return forbidden()
        if (
            now >= session.expires
            or now >= session.absolute_expires
            or type(session.owner_id) is not str
            or session.owner_id != self._owner_id
        ):
            self._sessions.pop(token, None)
            return forbidden()
        session.expires = min(now + 900, session.absolute_expires)
        captured = []

        def capture(status, headers):
            captured.append((status, headers))

        chunks = list(self._authenticated_response(environ, capture, session))
        status, headers = captured[0]
        if not status.startswith("3"):
            next_steps = {
                "/admin/import/confirm": ("/admin/style/terms", "隐私词设置"),
                "/admin/style/terms": ("/admin/style/review", "候选审查"),
                "/admin/style/retain": ("/admin/style/consent", "画像保存同意"),
                "/admin/style/consent": ("/admin/style/profile", "生成风格画像"),
                "/admin/style/profile": ("/admin/style/evidence", "表达选择与发布"),
                "/admin/style/clear": ("/admin/style/review", "重新审查候选"),
            }
            if method == "POST" and status.startswith("2") and path in next_steps:
                target, label = next_steps[path]
                chunks.append(f'<p><a href="{target}">下一步: {label}</a></p>'.encode())
            links = '<a href="/admin/home">返回主页</a>'
            if str(path).startswith("/admin/memory/"):
                links += ' · <a href="/admin/memory">返回记忆管理</a>'
            elif str(path).startswith("/admin/style/") and path != "/admin/style/review":
                links += ' · <a href="/admin/style/review">返回候选审查</a>'
            chunks.append(f'<nav aria-label="管理导航">{links}</nav>'.encode())
            headers = [(key, value) for key, value in headers if key != "Content-Length"]
            headers.append(("Content-Length", str(sum(len(chunk) for chunk in chunks))))
        start_response(status, headers)
        return chunks

    def _authenticated_response(
        self,
        environ: dict[str, Any],
        start_response: Callable[..., Any],
        session: _BrowserSession,
    ) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD")
        path = environ.get("PATH_INFO")


        if method == "GET" and path == "/admin/home":
            _, usage_body = self._usage_body()
            return self._response(
                start_response,
                "200 OK",
                "<h1>本地管理</h1>"
                "<h2>导入与风格设置</h2>"
                "<p>首次设置请按顺序完成。每步提交后都有下一步入口。</p>"
                "<ol>"
                '<li><a href="/admin/import/preview">导入预览与说话人映射</a></li>'
                '<li><a href="/admin/style/terms">隐私词设置</a></li>'
                '<li><a href="/admin/style/review">候选审查与逐条批准</a></li>'
                '<li><a href="/admin/style/retain">候选来源封存确认</a></li>'
                '<li><a href="/admin/style/consent">非正文特征长期本地保存同意</a></li>'
                '<li><a href="/admin/style/profile">生成非正文风格画像</a></li>'
                '<li><a href="/admin/style/evidence">本地发布已批准本人表达</a></li>'
                '<li><a href="/admin/style/snippets">成对片段（新风格链路）</a></li>'  # noqa: RUF001
                '<li><a href="/admin/style/labels">回应策略标注</a></li>'
                "</ol><p>风格画像至少需要20条已批准本人样本。无需批准全部候选。</p>"
                "<h2>记忆与费用</h2>"
                '<p><a href="/admin/memory">本人记忆授权与管理</a></p>'
                '<p><a href="/admin/memory/consent">独立记忆授权与字段选择</a></p>'
                '<p><a href="/admin/partner">对方画像（合成演示）</a></p>'  # noqa: RUF001
                '<p><a href="/admin/usage">费用状态</a></p>'
                '<p><a href="/admin/character">角色名称</a></p>'
                "<h2>撤权与删除</h2><p>以下操作进入确认页。不会直接执行。</p>"
                '<p><a href="/admin/style/clear">清除审批记录并重审</a></p>'
                '<p><a href="/admin/style/revoke">撤回长期保存同意</a></p>'
                '<p><a href="/admin/style/delete">彻底删除 Phase 1、Phase 5 与来源封存</a></p>'
                '<p><a href="/admin/style/delete-source">删除已封存来源</a></p>'
                '<p><a href="/admin/memory/revoke">撤回记忆授权并删除本人记忆</a></p>'
                "<h2>费用概览</h2>" + usage_body,
            )
        if method == "GET" and path == "/admin/usage":
            available, body = self._usage_body()
            return self._response(
                start_response,
                "200 OK" if available else "503 Service Unavailable",
                "<h1>费用状态</h1>" + body + '<p><a href="/admin/home">返回主页</a></p>',
            )
        if path == "/admin/character":
            return self._character_response(environ, start_response, session)
        if path in PARTNER_ROUTES:
            return self._partner_response(environ, start_response, session)
        if path in {
            "/admin/memory",
            "/admin/memory/consent",
            "/admin/memory/correct",
            "/admin/memory/delete",
            "/admin/memory/revoke",
        }:
            return self._memory_response(environ, start_response, session)
        if path in {"/admin/style/snippets", "/admin/style/snippets/audit"}:
            return self._snippet_response(environ, start_response, session)
        if path == "/admin/style/labels" or str(path).startswith("/admin/style/labels/"):
            return self._label_response(environ, start_response, session)
        return self._style_response(environ, start_response, session)


def build_admin_app(
    settings: Settings,
    config: AdminConfig,
    *,
    port: int,
    candidate_reader: Callable[[str, tuple[ChatRecord, ...]], SafeCandidatePackage] | None = None,
    approval_store: StyleCandidateApprovalStore | None = None,
    runtime_selector: Callable[..., SafeCandidatePackage] | None = select_safe_candidates,
    usage_reader: Callable[[], UsageSnapshot] | None = None,
    memory_factory: Callable[[], MemoryService] | None = None,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AdminApp:
    """Assemble the product entry using only the fixed private import path."""
    if (
        os.name != "posix"
        or type(settings) is not Settings
        or type(config) is not AdminConfig
        or not _valid_owner_id(config.owner_id)
        or (
            config.import_format is not None
            and (
                type(config.import_format) is not str
                or config.import_format not in {"ciphertalk_json", "standard_json"}
            )
        )
    ):
        raise AdminConfigError("admin configuration unavailable")
    loader = (
        None
        if config.import_format is None
        else _FixedImportBatchSource(settings.private_data_dir, config.import_format).load
    )
    source_seal = None
    source_gateway = None
    phase1_store = None
    phase5_store = None
    current_approval_store = approval_store or StyleCandidateApprovalStore(settings)
    if config.import_format is not None:
        from companion.admin.phase1 import (
            AdminSourceApprovalGateway,
            phase1_paths,
            phase5_paths,
            validate_phase1_paths,
            validate_phase5_paths,
        )
        from companion.admin.source_seal import StyleSourceSealStore

        source_seal = StyleSourceSealStore(settings, config)
        source_gateway = AdminSourceApprovalGateway(
            config.owner_id, source_seal, current_approval_store
        )
        validate_phase1_paths(settings.private_data_dir)
        paths = phase1_paths(settings.private_data_dir)
        phase1_store = SingleTrustedUserStyleProfileStore(
            trusted_user_id=config.owner_id,
            private_data_dir=settings.private_data_dir,
            consent_source=paths[0],
            profile_source=paths[1],
            evidence_source=paths[2],
            statistics_source=paths[3],
            index_source=paths[4],
            cache_source=paths[5],
            strict_source_binding=True,
            source_gateway=source_gateway,
        )
        validate_phase5_paths(settings.private_data_dir)
        evidence_paths = phase5_paths(settings.private_data_dir)
        phase5_store = SingleTrustedUserStyleEvidenceStore(
            trusted_user_id=config.owner_id,
            private_data_dir=settings.private_data_dir,
            evidence_source=evidence_paths[0],
            index_source=evidence_paths[1],
            cache_source=evidence_paths[2],
            version_source=evidence_paths[3],
            strict_source_binding=True,
            source_gateway=source_gateway,
        )

    def read_usage() -> UsageSnapshot:
        from companion.dialogue import load_budget_config
        from companion.dialogue.usage import MonthlyUsageLedger

        budget = load_budget_config(settings)
        return MonthlyUsageLedger(settings.runtime_data_dir / "dialogue", budget).snapshot(
            wall_clock()
        )

    def create_memory_service() -> MemoryService:
        from companion.memory import build_private_memory_service

        return build_private_memory_service(settings, clock=wall_clock)

    return AdminApp(
        config.owner_id,
        host="127.0.0.1",
        port=port,
        batch_loader=loader,
        candidate_reader=candidate_reader,
        approval_store=current_approval_store,
        runtime_selector=runtime_selector if candidate_reader is None else None,
        source_seal=source_seal,
        source_gateway=source_gateway,
        phase1_store=phase1_store,
        phase1_root=settings.private_data_dir,
        phase5_store=phase5_store,
        usage_reader=usage_reader if usage_reader is not None else read_usage,
        memory_factory=memory_factory if memory_factory is not None else create_memory_service,
        wall_clock=wall_clock,
        snippet_root=settings.private_data_dir / "style" / "snippets",
        label_root=settings.private_data_dir / "style" / "m1",
        character_settings=settings,
    )
