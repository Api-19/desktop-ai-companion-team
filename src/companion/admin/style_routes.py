from __future__ import annotations

import hashlib
import hmac
import html
import secrets
import unicodedata
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from companion.admin.approval import _validated_witness
from companion.admin.import_source import _batch_identity
from companion.admin.session import _BrowserSession
from companion.contracts import ChatRecord, SpeakerRole
from companion.importer import (
    ConfirmedSpeakerMapping,
    ImportBatch,
    MessageDirection,
    apply_confirmed_mapping,
    preview_speaker_mapping,
)
from companion.privacy import SafeCandidatePackage
from companion.style import (
    FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
    SourceApprovalState,
    StyleEvidenceCatalog,
    StyleFeatureKind,
    UserStyleProfileBundle,
    UserStyleProfileConsent,
    UserStyleProfileStatus,
    UserStyleSnapshot,
    UserStyleSnapshotReadResult,
    UserStyleSnapshotReadStatus,
    build_formal_style_scenario_catalog,
    generate_style_evidence_catalog,
    generate_user_style_profile,
)


class _StyleRoutesMixin:
    def _load_batch(self) -> tuple[ImportBatch, str]:
        if self._batch_loader is None:
            raise ValueError("batch unavailable")
        return _batch_identity(self._batch_loader())

    @staticmethod
    def _clear_confirmation(session: _BrowserSession) -> None:
        session.confirmed_digest = None
        session.confirmed_records = None
        session.candidate_package_bytes = None
        session.candidate_package_digest = None
        session.decide_token = None
        session.terms_token = None
        session.runtime_terms = None
        session.clear_token = None
        session.retain_token = None
        session.retain_witness = None
        session.delete_source_token = None
        session.consent_token = None
        session.consent_state = None
        session.snippets_token = None
        session.audit_token = None
        _StyleRoutesMixin._clear_style_actions(session)

    @staticmethod
    def _clear_style_actions(session: _BrowserSession) -> None:
        session.profile_token = None
        session.profile_binding = None
        session.evidence_token = None
        session.evidence_binding = None
        session.evidence_basket = ()

    def _bound_state(self, session: _BrowserSession) -> SourceApprovalState:
        return self._bound_source(session)[0]

    def _bound_source(
        self, session: _BrowserSession
    ) -> tuple[SourceApprovalState, SafeCandidatePackage]:
        if (
            self._source_seal is None
            or self._source_gateway is None
            or self._phase1_store is None
            or self._approval_store is None
        ):
            raise ValueError("style consent unavailable")
        records = self._confirmed_context(session)
        package = self._session_package(session, records, allow_create=False)
        sealed = self._source_seal.read(session.owner_id)
        if (
            sealed is None
            or sealed.owner_id != session.owner_id
            or sealed.batch_digest != session.confirmed_digest
            or sealed.package != package
        ):
            raise ValueError("style consent unavailable")
        state = self._source_gateway.current_state(session.owner_id)
        if type(state) is not SourceApprovalState or state.owner_id != session.owner_id:
            raise ValueError("style consent unavailable")
        state = SourceApprovalState.model_validate(state.model_dump(mode="python"), strict=True)
        witness = _validated_witness(
            self._approval_store.approval_witness(session.owner_id, package),
            len(package.candidates),
        )
        if (
            state.batch_digest != sealed.batch_digest
            or state.package_digest != witness.package_digest
            or state.approved_indices != witness.approved_indices
            or state.revision != witness.revision
        ):
            raise ValueError("style consent unavailable")
        return state, package

    def _profile_context(
        self, session: _BrowserSession
    ) -> tuple[SourceApprovalState, SafeCandidatePackage, UserStyleProfileConsent]:
        if self._phase1_store is None or self._phase5_store is None or self._phase1_root is None:
            raise ValueError("style profile unavailable")
        state, package = self._bound_source(session)
        consent = self._phase1_store.view_consent(session.owner_id)
        if (
            type(consent) is not UserStyleProfileConsent
            or consent.user_id != session.owner_id
            or consent.status is not UserStyleProfileStatus.AUTHORIZED
            or consent.scope != "non_text_style_features_local_persistence"
            or consent.source_binding != state
        ):
            raise ValueError("style profile unavailable")
        return state, package, consent

    def _authorized_snapshot(self, owner_id: str) -> UserStyleSnapshot:
        result = self._phase1_store.read_snapshot_state(owner_id)
        if (
            type(result) is not UserStyleSnapshotReadResult
            or result.status is not UserStyleSnapshotReadStatus.AUTHORIZED
            or type(result.snapshot) is not UserStyleSnapshot
        ):
            raise ValueError("style profile unavailable")
        return result.snapshot

    def _evidence_context(
        self, session: _BrowserSession
    ) -> tuple[SourceApprovalState, SafeCandidatePackage, UserStyleSnapshot, tuple[str, ...]]:
        state, package, consent = self._profile_context(session)
        snapshot = self._authorized_snapshot(session.owner_id)
        if snapshot.consent != consent:
            raise ValueError("style evidence unavailable")
        scenarios = tuple(
            item.scenario_id
            for item in build_formal_style_scenario_catalog(
                user_id=session.owner_id,
                trusted_user_id=self._owner_id,
                snapshot=snapshot,
                expected_catalog_version=FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
            )
        )
        return state, package, snapshot, scenarios

    @staticmethod
    def _evidence_key(
        state: SourceApprovalState, snapshot: UserStyleSnapshot
    ) -> tuple[SourceApprovalState, str, str]:
        return (state, snapshot.consent.authorization_version, snapshot.profile.profile_version)

    @staticmethod
    def _profile_summary(snapshot: UserStyleSnapshot) -> str:
        labels = {
            StyleFeatureKind.MESSAGE_LENGTH_Q25: "较短句长 25% 分位",
            StyleFeatureKind.MESSAGE_LENGTH_MEDIAN: "中位句长",
            StyleFeatureKind.MESSAGE_LENGTH_Q75: "较长句长 75% 分位",
        }
        features = snapshot.profile.features
        items = "".join(
            f"<li>{labels[item.kind]}: {item.value:g}</li>"
            for item in features
            if item.kind in labels
        )
        count = max((item.provenance.evidence_count for item in features), default=0)
        return f"<ul>{items}</ul><p>样本数 {count}</p>"

    @staticmethod
    def _parse_index(value: str) -> int | None:
        if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
            return None
        return int(value)

    def _clear_phase5(self, owner_id: str) -> None:
        if (
            type(owner_id) is not str
            or owner_id != self._owner_id
            or self._phase5_store is None
            or self._phase1_root is None
        ):
            raise ValueError("style management unavailable")
        from companion.admin.phase1 import validate_phase5_paths

        validate_phase5_paths(self._phase1_root)
        self._phase5_store.delete_user_evidence(owner_id)
        validate_phase5_paths(self._phase1_root)
        if any(path.exists() for path in self._phase5_store.artifact_paths):
            raise ValueError("style management unavailable")

    def _invalidate_derived(self, owner_id: str) -> bool:
        """Phase 1 persistent invalidation first, then Phase 5; False when unmanaged."""
        if type(owner_id) is not str or owner_id != self._owner_id:
            raise ValueError("style management unavailable")
        if self._phase1_store is None:
            if self._source_seal is not None or self._phase5_store is not None:
                raise ValueError("style management unavailable")
            return False
        from companion.admin.phase1 import validate_phase1_paths

        if self._phase1_root is None or self._phase5_store is None:
            raise ValueError("style management unavailable")
        validate_phase1_paths(self._phase1_root)
        self._phase1_store.delete_user_style(owner_id)
        validate_phase1_paths(self._phase1_root)
        if any(path.exists() for path in self._phase1_store.artifact_paths):
            raise ValueError("style management unavailable")
        self._clear_phase5(owner_id)
        return True

    def _invalidate_style(self, owner_id: str) -> None:
        if not self._invalidate_derived(owner_id):
            return
        if self._source_seal is not None:
            self._source_seal.delete(owner_id)
            if self._source_seal.exists_for_owner(owner_id):
                raise ValueError("style management unavailable")

    @staticmethod
    def _parse_terms(value: str, *, required: bool) -> tuple[str, ...]:
        if type(value) is not str or len(value.encode("utf-8")) > 512:
            raise ValueError("runtime terms unavailable")
        if value == "" and not required:
            return ()
        parts = value.split("\n")
        if not 1 <= len(parts) <= 8 or len(set(parts)) != len(parts):
            raise ValueError("runtime terms unavailable")
        for item in parts:
            if (
                not 1 <= len(item) <= 64
                or item != item.strip()
                or unicodedata.normalize("NFC", item) != item
                or "\ufffd" in item
                or any(unicodedata.category(char).startswith("C") for char in item)
            ):
                raise ValueError("runtime terms unavailable")
        return tuple(parts)

    def _confirmed_context(self, session: _BrowserSession) -> tuple[ChatRecord, ...]:
        try:
            current, digest = self._load_batch()
            if digest != session.confirmed_digest:
                raise ValueError("confirmed batch changed")
            return self._validated_confirmed_records(current, session.confirmed_records)
        except Exception:
            self._clear_confirmation(session)
            raise ValueError("confirmed batch unavailable") from None

    def _session_package(
        self,
        session: _BrowserSession,
        records: tuple[ChatRecord, ...],
        *,
        allow_create: bool,
    ) -> SafeCandidatePackage:
        raw = session.candidate_package_bytes
        if raw is None:
            if not allow_create or (
                self._candidate_reader is None and self._runtime_selector is None
            ):
                raise ValueError("candidate package unavailable")
            if self._runtime_selector is None:
                package = self._candidate_reader(session.owner_id, records)
            else:
                terms = session.runtime_terms
                session.runtime_terms = None
                if terms is None:
                    raise ValueError("runtime terms unavailable")
                package = self._runtime_selector(
                    list(records),
                    processed_at=datetime.now(UTC),
                    true_name_terms=terms[0],
                    approved_nicknames=terms[1],
                )
            if type(package) is not SafeCandidatePackage:
                raise ValueError("candidate package unavailable")
            verified = SafeCandidatePackage.model_validate(
                package.model_dump(mode="python", warnings="error"), strict=True
            )
            raw = verified.model_dump_json(warnings="error").encode()
            session.candidate_package_bytes = raw
            session.candidate_package_digest = hashlib.sha256(raw).digest()
            return verified
        if (
            type(raw) is not bytes
            or type(session.candidate_package_digest) is not bytes
            or not hmac.compare_digest(
                hashlib.sha256(raw).digest(), session.candidate_package_digest
            )
        ):
            self._clear_confirmation(session)
            raise ValueError("candidate package changed")
        try:
            verified = SafeCandidatePackage.model_validate_json(raw, strict=True)
            if verified.model_dump_json(warnings="error").encode() != raw:
                raise ValueError("candidate package changed")
            return verified
        except Exception:
            self._clear_confirmation(session)
            raise ValueError("candidate package changed") from None

    @staticmethod
    def _validated_decisions(value: object, candidate_count: int) -> tuple[tuple[int, str], ...]:
        if type(value) is not tuple or len(value) != candidate_count:
            raise ValueError("approval state unavailable")
        for index, item in enumerate(value):
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or item[0] != index
                or type(item[1]) is not str
                or item[1] not in {"pending", "approved"}
            ):
                raise ValueError("approval state unavailable")
        return value

    @staticmethod
    def _validated_confirmed_records(
        batch: ImportBatch, records: tuple[ChatRecord, ...] | None
    ) -> tuple[ChatRecord, ...]:
        if type(records) is not tuple or len(records) != len(batch.records):
            raise ValueError("confirmed mapping unavailable")
        copies = []
        for source, record in zip(batch.records, records, strict=True):
            if type(record) is not ChatRecord:
                raise ValueError("confirmed mapping unavailable")
            copy = ChatRecord.model_validate(
                record.model_dump(mode="python", warnings="error"), strict=True
            )
            expected = ChatRecord(
                conversation_id=source.conversation_id,
                message_id=source.message_id,
                timestamp=source.timestamp,
                speaker_role=(
                    SpeakerRole.OWNER
                    if source.direction is MessageDirection.OUTBOUND
                    else SpeakerRole.OTHER
                ),
                content_type=source.content_type,
                text=source.text,
            )
            if copy.model_dump(mode="python") != expected.model_dump(mode="python"):
                raise ValueError("confirmed mapping unavailable")
            copies.append(copy)
        return tuple(copies)

    def _style_response(
        self, environ: dict[str, Any], start_response: Callable[..., Any], session: _BrowserSession
    ) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD")
        path = environ.get("PATH_INFO")

        def forbidden() -> Iterable[bytes]:
            return self._response(start_response, "403 Forbidden", "Forbidden")


        if method == "GET" and path == "/admin/import/preview":
            session.preview_digest = None
            session.confirm_token = None
            self._clear_confirmation(session)
            try:
                batch, digest = self._load_batch()
                preview = preview_speaker_mapping(batch.records)
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "导入预览未就绪")
            session.preview_digest = digest
            session.confirm_token = secrets.token_urlsafe(32)
            lines = [
                "<h1>说话人映射预览</h1>",
                '<form method="post" action="/admin/import/confirm">',
                f'<input type="hidden" name="csrf" value="{self._token(session, "confirm")}">',
                f'<input type="hidden" name="confirm_token" value="{session.confirm_token}">',
            ]
            for index, speaker in enumerate(preview.speakers):
                lines.append(
                    f"<p>{html.escape(speaker.source_speaker_id)}: {speaker.message_count}</p>"
                )
                lines.extend(f"<p>{html.escape(sample)}</p>" for sample in speaker.sample_texts)
                lines.append(
                    f'<label>角色<select name="role_{index}">'
                    '<option value="">请选择</option>'
                    '<option value="owner">主人</option>'
                    '<option value="other">对方</option>'
                    "</select></label>"
                )
            lines.append(
                '<label><input type="checkbox" name="confirm" value="yes">确认映射</label>'
            )
            lines.append("<button>确认</button></form>")
            return self._response(start_response, "200 OK", "".join(lines))
        if method == "GET" and path == "/admin/style/review":
            session.decide_token = None
            if (
                (self._candidate_reader is None and self._runtime_selector is None)
                or session.confirmed_digest is None
                or session.confirmed_records is None
                or (
                    self._runtime_selector is not None
                    and session.candidate_package_bytes is None
                    and session.runtime_terms is None
                )
            ):
                return self._response(start_response, "503 Service Unavailable", "候选预览未就绪")
            try:
                records = self._confirmed_context(session)
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "候选预览未就绪")
            try:
                verified = self._session_package(session, records, allow_create=True)
                decisions = (
                    None
                    if self._approval_store is None
                    else self._validated_decisions(
                        self._approval_store.decisions(session.owner_id, verified),
                        len(verified.candidates),
                    )
                )
                decide_token = None if decisions is None else secrets.token_urlsafe(32)
                decide_csrf = None if decisions is None else self._token(session, "decide")
                items = []
                for index, candidate in enumerate(verified.candidates):
                    item = (
                        f'<li id="candidate-{index}">{index}: '
                        f"{html.escape(candidate.category.value)}: "
                        f"{html.escape(candidate.text)}"
                    )
                    if decisions is not None:
                        item += (
                            f" <span>{decisions[index][1]}</span>"
                            '<form method="post" action="/admin/style/decide">'
                            f'<input type="hidden" name="csrf" value="{decide_csrf}">'
                            f'<input type="hidden" name="decide_token" value="{decide_token}">'
                            f'<input type="hidden" name="index" value="{index}">'
                            '<button name="decision" value="approve">批准</button>'
                            '<button name="decision" value="revoke">撤回</button></form>'
                        )
                    items.append(item + "</li>")
                body = "<h1>候选预览</h1><ul>" + "".join(items) + "</ul>"
                if decisions is not None and not items:
                    body += (
                        '<form method="post" action="/admin/style/decide">'
                        f'<input type="hidden" name="csrf" value="{decide_csrf}">'
                        f'<input type="hidden" name="decide_token" value="{decide_token}">'
                        "</form>"
                    )
                if self._approval_store is not None:
                    body += '<p><a href="/admin/style/clear">清除审批记录并重审</a></p>'
                if self._approval_store is not None and self._source_seal is not None:
                    body += '<p><a href="/admin/style/retain">非正文特征本地保存意图</a></p>'
                    body += '<p><a href="/admin/style/delete-source">删除已封存来源</a></p>'
                    body += '<p><a href="/admin/style/consent">非正文特征长期本地保存同意</a></p>'
            except Exception:
                body = "候选预览未就绪"
                if self._approval_store is not None and session.candidate_package_bytes is not None:
                    body += '<p><a href="/admin/style/clear">清除审批记录并重审</a></p>'
                return self._response(start_response, "503 Service Unavailable", body)
            session.decide_token = decide_token
            return self._response(start_response, "200 OK", body)
        if method == "GET" and path == "/admin/style/consent":
            session.consent_token = None
            session.consent_state = None
            try:
                state = self._bound_state(session)
            except Exception:
                return self._response(
                    start_response, "503 Service Unavailable", "长期保存同意未就绪"
                )
            session.consent_state = state
            session.consent_token = secrets.token_urlsafe(32)
            return self._response(
                start_response,
                "200 OK",
                "<h1>非正文风格特征长期本地保存同意</h1>"
                "<p>仅从当前已批准主人候选提取非正文风格特征并长期本地保存。"
                "支持独立撤权和删除。不会授予 Phase 5 具体表达使用或云端发送许可。"
                "记录前会先清除已发布的 Phase 5 表达。本次只记录同意。画像尚未生成。</p>"
                '<form method="post" action="/admin/style/consent">'
                f'<input type="hidden" name="csrf" value="{self._token(session, "consent")}">'
                f'<input type="hidden" name="consent_token" value="{session.consent_token}">'
                '<label><input type="checkbox" name="confirm" value="yes">明确同意</label>'
                "<button>记录长期保存同意</button></form>"
                '<p><a href="/admin/style/revoke">撤权</a> '
                '<a href="/admin/style/delete">删除</a></p>',
            )
        if method == "GET" and path == "/admin/style/profile":
            session.profile_token = None
            session.profile_binding = None
            try:
                state, _, consent = self._profile_context(session)
                try:
                    summary = self._profile_summary(self._authorized_snapshot(session.owner_id))
                except Exception:
                    summary = "<p>尚无可用画像</p>"
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "画像生成未就绪")
            session.profile_binding = (state, consent.authorization_version)
            session.profile_token = secrets.token_urlsafe(32)
            return self._response(
                start_response,
                "200 OK",
                "<h1>非正文风格画像生成</h1>"
                "<p>仅从当前已批准主人候选计算三个非正文句长特征并在本机长期保存。"
                f"当前已批准样本数 {len(state.approved_indices)}。至少需要 20。"
                "生成或重新生成会先清除已发布的 Phase 5 表达。不授予云端请求或费用。</p>"
                + summary
                + '<form method="post" action="/admin/style/profile">'
                f'<input type="hidden" name="csrf" value="{self._token(session, "profile")}">'
                f'<input type="hidden" name="profile_token" value="{session.profile_token}">'
                '<label><input type="checkbox" name="confirm" value="yes">明确确认生成</label>'
                "<button>生成画像</button></form>",
            )
        if method == "GET" and path == "/admin/style/evidence":
            session.evidence_token = None
            try:
                state, package, snapshot, scenarios = self._evidence_context(session)
            except Exception:
                session.evidence_binding = None
                session.evidence_basket = ()
                return self._response(start_response, "503 Service Unavailable", "表达发布未就绪")
            key = self._evidence_key(state, snapshot)
            if session.evidence_binding != key:
                session.evidence_binding = key
                session.evidence_basket = ()
            session.evidence_token = secrets.token_urlsafe(32)
            hidden = f'<input type="hidden" name="evidence_token" value="{session.evidence_token}">'

            def opened(action: str) -> str:
                return (
                    f'<form method="post" action="/admin/style/{action}">'
                    f'<input type="hidden" name="csrf" value="{self._token(session, action)}">'
                    + hidden
                )

            items = "".join(
                f"<li>{index}: {html.escape(scenario)}: "
                f"{html.escape(package.candidates[index].text)}"
                + opened("evidence-remove")
                + f'<input type="hidden" name="index" value="{index}">'
                "<button>移除</button></form></li>"
                for index, scenario in session.evidence_basket
            )
            options = "".join(
                f'<option value="{html.escape(item)}">{html.escape(item)}</option>'
                for item in scenarios
            )
            body = (
                "<h1>已批准本人表达本地发布</h1>"
                "<p>主人选择的已批准本人表达正文将本地长期保存在 style 资源中供后续检索。"
                "这与 Phase 1 仅保存非正文特征的用途不同。本次不授予云端请求或费用。"
                '候选索引见<a href="/admin/style/review">候选预览</a>。'
                "每次加入一条当前已批准、非空且不超过 160 字的候选并明确选择场景。"
                "最多 64 条。</p>"
                f"<h2>待发布 {len(session.evidence_basket)} 条</h2><ol>{items}</ol>"
                + opened("evidence-add")
                + '<label>候选索引<input name="index" inputmode="numeric"></label>'
                f'<label>场景<select name="scenario"><option value="">请选择</option>{options}'
                "</select></label><button>加入</button></form>"
                + opened("evidence-publish")
                + '<label><input type="checkbox" name="confirm_local" value="yes">'
                "明确确认上述表达正文在本机长期保存供后续检索</label>"
                "<button>本地发布</button></form>"
            )
            return self._response(start_response, "200 OK", body)
        if method == "GET" and path in {"/admin/style/revoke", "/admin/style/delete"}:
            action = path.rsplit("/", 1)[1]
            setattr(session, f"{action}_token", None)
            session.consent_token = None
            session.consent_state = None
            if self._phase1_store is None:
                return self._response(start_response, "503 Service Unavailable", "风格清理未就绪")
            once = secrets.token_urlsafe(32)
            setattr(session, f"{action}_token", once)
            return self._response(
                start_response,
                "200 OK",
                "<h1>撤权并删除 Phase 1、Phase 5 与来源封存</h1>"
                "<p>先使长期保存同意失效并清理 Phase 1 六件。再清理 Phase 5 四件已发布表达。"
                "最后删除封存正文。保留 Privacy 逐条决定。</p>"
                f'<form method="post" action="/admin/style/{action}">'
                f'<input type="hidden" name="csrf" value="{self._token(session, action)}">'
                f'<input type="hidden" name="{action}_token" value="{once}">'
                '<label><input type="checkbox" name="confirm" value="yes">明确撤权并删除</label>'
                "<button>确认清理</button></form>",
            )
        if method == "GET" and path == "/admin/style/retain":
            session.retain_token = None
            session.retain_witness = None
            if (
                self._approval_store is None
                or self._source_seal is None
                or session.confirmed_digest is None
                or session.candidate_package_bytes is None
            ):
                return self._response(start_response, "503 Service Unavailable", "来源封存未就绪")
            try:
                records = self._confirmed_context(session)
                package = self._session_package(session, records, allow_create=False)
                witness = _validated_witness(
                    self._approval_store.approval_witness(session.owner_id, package),
                    len(package.candidates),
                )
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "来源封存未就绪")
            session.retain_witness = witness
            session.retain_token = secrets.token_urlsafe(32)
            body = (
                "<h1>非正文风格特征长期本地保存意图</h1>"
                "<p>确认后将完整安全候选包正文保留在本机私有目录。"
                "仅供后续非正文风格特征建档使用。"
                "可通过独立删除入口删除该封存。"
                "逐条批准与本次保存意图不同。Phase 1 持久授权尚未完成。</p>"
                '<form method="post" action="/admin/style/retain">'
                f'<input type="hidden" name="csrf" value="{self._token(session, "retain")}">'
                f'<input type="hidden" name="retain_token" value="{session.retain_token}">'
                '<label><input type="checkbox" name="confirm" value="yes">明确确认</label>'
                "<button>封存当前来源</button></form>"
                '<p><a href="/admin/style/delete-source">删除已封存来源</a></p>'
            )
            return self._response(start_response, "200 OK", body)
        if method == "GET" and path == "/admin/style/delete-source":
            session.delete_source_token = None
            session.consent_token = None
            session.consent_state = None
            session.retain_token = None
            session.retain_witness = None
            if self._source_seal is None:
                return self._response(start_response, "503 Service Unavailable", "来源删除未就绪")
            try:
                if not self._source_seal.exists_for_owner(session.owner_id):
                    raise ValueError("source seal unavailable")
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "来源删除未就绪")
            session.delete_source_token = secrets.token_urlsafe(32)
            body = (
                "<h1>删除已封存来源</h1>"
                "<p>删除本机私有目录中的安全候选包正文。"
                "先使 Phase 1 长期同意失效并清理其六件。再清理 Phase 5 四件已发布表达。"
                "不会撤回 Privacy 逐条批准。</p>"
                '<form method="post" action="/admin/style/delete-source">'
                f'<input type="hidden" name="csrf" value="{self._token(session, "delete-source")}">'
                '<input type="hidden" name="delete_source_token" '
                f'value="{session.delete_source_token}">'
                '<label><input type="checkbox" name="confirm" value="yes">明确删除</label>'
                "<button>删除封存</button></form>"
            )
            return self._response(start_response, "200 OK", body)
        if method == "GET" and path == "/admin/style/clear":
            session.clear_token = None
            session.consent_token = None
            session.consent_state = None
            if (
                self._approval_store is None
                or session.confirmed_digest is None
                or session.candidate_package_bytes is None
            ):
                return self._response(start_response, "503 Service Unavailable", "审批清除未就绪")
            try:
                records = self._confirmed_context(session)
                self._session_package(session, records, allow_create=False)
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "审批清除未就绪")
            session.decide_token = None
            session.retain_token = None
            session.retain_witness = None
            session.clear_token = secrets.token_urlsafe(32)
            body = (
                "<h1>清除审批记录并重审</h1>"
                "<p>这会清除所有已有逐条批准。不会自动批准当前或新候选。"
                "清除前先使 Phase 1 同意失效并清理 Phase 1、Phase 5 资源与来源封存。</p>"
                '<form method="post" action="/admin/style/clear">'
                f'<input type="hidden" name="csrf" value="{self._token(session, "clear")}">'
                f'<input type="hidden" name="clear_token" value="{session.clear_token}">'
                '<label><input type="checkbox" name="confirm" value="yes">确认清除</label>'
                "<button>清除审批记录</button></form>"
            )
            return self._response(start_response, "200 OK", body)
        if method == "GET" and path == "/admin/style/terms":
            session.terms_token = None
            if self._runtime_selector is None or session.confirmed_digest is None:
                return self._response(start_response, "503 Service Unavailable", "运行时词未就绪")
            try:
                self._confirmed_context(session)
            except Exception:
                return self._response(start_response, "503 Service Unavailable", "运行时词未就绪")
            session.terms_token = secrets.token_urlsafe(32)
            body = (
                '<h1>本次隐私词</h1><form method="post" action="/admin/style/terms">'
                f'<input type="hidden" name="csrf" value="{self._token(session, "terms")}">'
                f'<input type="hidden" name="terms_token" value="{session.terms_token}">'
                '<label>真实姓名精确词<textarea name="true_name_terms"></textarea></label>'
                '<label>已确认爱称<textarea name="approved_nicknames"></textarea></label>'
                '<label><input type="checkbox" name="confirm_nicknames" value="yes">'
                "确认上述爱称</label>"
                "<button>提交本次词</button></form>"
            )
            return self._response(start_response, "200 OK", body)
        if method == "POST" and path in {
            "/admin/import/confirm",
            "/admin/style/decide",
            "/admin/style/consent",
            "/admin/style/terms",
            "/admin/style/clear",
            "/admin/style/retain",
            "/admin/style/delete-source",
            "/admin/style/revoke",
            "/admin/style/delete",
            "/admin/style/profile",
            "/admin/style/evidence-add",
            "/admin/style/evidence-remove",
            "/admin/style/evidence-publish",
        }:
            action = path.rsplit("/", 1)[1]
            form = self._form(environ)
            if not hmac.compare_digest(form.get("csrf", ""), self._token(session, action)):
                return forbidden()
            if action == "profile":
                if self._phase1_store is None or self._phase5_store is None:
                    return self._response(
                        start_response, "503 Service Unavailable", "画像生成未就绪"
                    )
                once, expected = session.profile_token, session.profile_binding
                self._clear_style_actions(session)
                if once is None or not hmac.compare_digest(form.get("profile_token", ""), once):
                    return forbidden()
                if set(form) != {"csrf", "profile_token", "confirm"} or form["confirm"] != "yes":
                    return self._response(start_response, "400 Bad Request", "画像生成确认无效")
                attempted = False
                try:
                    from companion.admin.phase1 import validate_phase1_paths

                    state, package, consent = self._profile_context(session)
                    if (state, consent.authorization_version) != expected:
                        raise ValueError("style profile unavailable")
                    self._clear_phase5(session.owner_id)
                    bundle = generate_user_style_profile(
                        user_id=session.owner_id,
                        package=package,
                        consent=consent,
                        source_state=state,
                        observed_at=datetime.now(UTC),
                    )
                    if type(bundle) is not UserStyleProfileBundle:
                        raise ValueError("style profile unavailable")
                    validate_phase1_paths(self._phase1_root)
                    attempted = True
                    self._phase1_store.create_profile(
                        user_id=session.owner_id,
                        profile=bundle.profile,
                        evidence=bundle.evidence,
                        derived=bundle.derived,
                    )
                    validate_phase1_paths(self._phase1_root)
                    snapshot = self._authorized_snapshot(session.owner_id)
                    after_state, _, after_consent = self._profile_context(session)
                    if (
                        snapshot.consent != consent
                        or snapshot.profile != bundle.profile
                        or snapshot.derived != bundle.derived
                        or snapshot.evidence_count != bundle.evidence.total_evidence_count
                        or after_state != state
                        or after_consent != consent
                        or any(path.exists() for path in self._phase5_store.artifact_paths)
                    ):
                        raise ValueError("style profile readback changed")
                except Exception:
                    if attempted:
                        with suppress(Exception):
                            self._invalidate_derived(session.owner_id)
                    return self._response(
                        start_response, "503 Service Unavailable", "画像生成未就绪"
                    )
                return self._response(
                    start_response,
                    "200 OK",
                    "<p>非正文画像已生成并本地保存</p>" + self._profile_summary(snapshot),
                )
            if action.startswith("evidence-"):
                once = session.evidence_token
                session.evidence_token = None
                if once is None or not hmac.compare_digest(form.get("evidence_token", ""), once):
                    return forbidden()
                fields = {
                    "evidence-add": {"index", "scenario"},
                    "evidence-remove": {"index"},
                    "evidence-publish": {"confirm_local"},
                }[action]
                if set(form) != {"csrf", "evidence_token", *fields}:
                    return self._response(start_response, "400 Bad Request", "表达选择无效")
                try:
                    state, package, snapshot, scenarios = self._evidence_context(session)
                    if self._evidence_key(state, snapshot) != session.evidence_binding:
                        raise ValueError("style evidence binding changed")
                except Exception:
                    session.evidence_binding = None
                    session.evidence_basket = ()
                    return self._response(
                        start_response, "503 Service Unavailable", "表达发布未就绪"
                    )
                basket = session.evidence_basket
                chosen = {index for index, _ in basket}
                back = '<p><a href="/admin/style/evidence">返回表达发布</a></p>'
                if action == "evidence-add":
                    index = self._parse_index(form["index"])
                    text = (
                        package.candidates[index].text
                        if index is not None and index in state.approved_indices
                        else ""
                    )
                    if (
                        not text.strip()
                        or len(text) > 160
                        or index in chosen
                        or len(basket) >= 64
                        or form["scenario"] not in scenarios
                    ):
                        return self._response(start_response, "400 Bad Request", "表达选择无效")
                    session.evidence_basket = (*basket, (index, form["scenario"]))
                    return self._response(start_response, "200 OK", "<p>已加入待发布</p>" + back)
                if action == "evidence-remove":
                    index = self._parse_index(form["index"])
                    if index is None or index not in chosen:
                        return self._response(start_response, "400 Bad Request", "表达选择无效")
                    session.evidence_basket = tuple(item for item in basket if item[0] != index)
                    return self._response(start_response, "200 OK", "<p>已移除</p>" + back)
                if form["confirm_local"] != "yes" or not basket:
                    return self._response(start_response, "400 Bad Request", "表达发布确认无效")
                session.evidence_basket = ()
                session.evidence_binding = None
                attempted = False
                try:
                    from companion.admin.phase1 import validate_phase5_paths

                    validate_phase5_paths(self._phase1_root, create=True)
                    catalog = generate_style_evidence_catalog(
                        user_id=session.owner_id,
                        package=package,
                        snapshot=snapshot,
                        source_state=state,
                        selections=basket,
                        observed_at=datetime.now(UTC),
                    )
                    if type(catalog) is not StyleEvidenceCatalog:
                        raise ValueError("style evidence unavailable")
                    attempted = True
                    self._phase5_store.publish_approved_evidence(
                        user_id=session.owner_id, snapshot=snapshot, catalog=catalog
                    )
                    validate_phase5_paths(self._phase1_root)
                    records = self._phase5_store.view_approved_evidence(session.owner_id)
                    after_state, _, after_snapshot, _ = self._evidence_context(session)
                    if (
                        type(records) is not tuple
                        or records != catalog.records
                        or tuple((item.candidate_index, item.scenario_id) for item in records)
                        != basket
                        or after_state != state
                        or after_snapshot != snapshot
                    ):
                        raise ValueError("style evidence readback changed")
                except Exception:
                    if attempted:
                        with suppress(Exception):
                            self._clear_phase5(session.owner_id)
                    return self._response(
                        start_response, "503 Service Unavailable", "表达发布未就绪"
                    )
                return self._response(
                    start_response, "200 OK", f"<p>已在本地发布 {len(records)} 条表达</p>"
                )
            if action == "consent":
                if (
                    self._phase1_store is None
                    or self._phase5_store is None
                    or self._source_gateway is None
                ):
                    return self._response(
                        start_response, "503 Service Unavailable", "长期保存同意未就绪"
                    )
                once, expected = session.consent_token, session.consent_state
                session.consent_token = None
                session.consent_state = None
                if once is None or not hmac.compare_digest(form.get("consent_token", ""), once):
                    return forbidden()
                if set(form) != {"csrf", "consent_token", "confirm"} or form["confirm"] != "yes":
                    return self._response(start_response, "400 Bad Request", "长期保存同意无效")
                self._clear_style_actions(session)
                attempted = False
                try:
                    before = self._bound_state(session)
                    if before != expected or self._phase1_root is None:
                        raise ValueError("style consent unavailable")
                    from companion.admin.phase1 import validate_phase1_paths

                    self._clear_phase5(session.owner_id)

                    validate_phase1_paths(self._phase1_root, create=True)
                    attempted = True
                    consent = self._phase1_store.grant_bound_consent(
                        user_id=session.owner_id, updated_at=datetime.now(UTC)
                    )
                    validate_phase1_paths(self._phase1_root)
                    readback = self._phase1_store.view_consent(session.owner_id)
                    after = self._bound_state(session)
                    if (
                        type(consent) is not UserStyleProfileConsent
                        or readback != consent
                        or consent.user_id != session.owner_id
                        or consent.status is not UserStyleProfileStatus.AUTHORIZED
                        or consent.scope != "non_text_style_features_local_persistence"
                        or consent.source_binding != before
                        or after != before
                        or any(path.exists() for path in self._phase1_store.derived_artifact_paths)
                    ):
                        raise ValueError("style consent unavailable")
                except Exception:
                    if attempted:
                        with suppress(Exception):
                            self._phase1_store.delete_user_style(session.owner_id)
                    return self._response(
                        start_response, "503 Service Unavailable", "长期保存同意未就绪"
                    )
                return self._response(start_response, "200 OK", "<p>同意已记录。画像尚未生成</p>")
            if action in {"revoke", "delete"}:
                once = getattr(session, f"{action}_token")
                setattr(session, f"{action}_token", None)
                if once is None or not hmac.compare_digest(form.get(f"{action}_token", ""), once):
                    return forbidden()
                if set(form) != {"csrf", f"{action}_token", "confirm"} or form["confirm"] != "yes":
                    return self._response(start_response, "400 Bad Request", "风格清理确认无效")
                try:
                    if self._phase1_store is None:
                        raise ValueError("style management unavailable")
                    self._clear_style_actions(session)
                    self._invalidate_style(session.owner_id)
                except Exception:
                    return self._response(
                        start_response, "503 Service Unavailable", "风格清理未就绪"
                    )
                session.consent_token = None
                session.consent_state = None
                session.retain_token = None
                session.retain_witness = None
                self._clear_style_actions(session)
                return self._response(
                    start_response, "200 OK", "<p>Phase 1、Phase 5 与来源封存已清理。</p>"
                )
            if action == "retain":
                if (
                    self._approval_store is None
                    or self._source_seal is None
                    or session.confirmed_digest is None
                    or session.candidate_package_bytes is None
                ):
                    return self._response(
                        start_response, "503 Service Unavailable", "来源封存未就绪"
                    )
                once = session.retain_token
                expected_witness = session.retain_witness
                session.retain_token = None
                session.retain_witness = None
                if once is None or not hmac.compare_digest(form.get("retain_token", ""), once):
                    return forbidden()
                if set(form) != {"csrf", "retain_token", "confirm"} or form["confirm"] != "yes":
                    return self._response(start_response, "400 Bad Request", "来源封存确认无效")
                try:
                    records = self._confirmed_context(session)
                    package = self._session_package(session, records, allow_create=False)
                    before = _validated_witness(
                        self._approval_store.approval_witness(session.owner_id, package),
                        len(package.candidates),
                    )
                    if before != expected_witness:
                        raise ValueError("approval witness changed")
                    sealed = self._source_seal.seal(
                        session.owner_id, session.confirmed_digest, package
                    )
                    readback = self._source_seal.read(session.owner_id)
                    if (
                        readback is None
                        or readback != sealed
                        or readback.owner_id != session.owner_id
                        or readback.batch_digest != session.confirmed_digest
                        or readback.package != package
                    ):
                        raise ValueError("source seal readback changed")
                    after = _validated_witness(
                        self._approval_store.approval_witness(session.owner_id, package),
                        len(package.candidates),
                    )
                    if after != before:
                        raise ValueError("approval witness changed")
                except Exception:
                    return self._response(
                        start_response, "503 Service Unavailable", "来源封存未就绪"
                    )
                return self._response(
                    start_response, "200 OK", "<p>来源封存已完成。Phase 1 授权尚未完成。</p>"
                )
            if action == "delete-source":
                if self._source_seal is None:
                    return self._response(
                        start_response, "503 Service Unavailable", "来源删除未就绪"
                    )
                once = session.delete_source_token
                session.delete_source_token = None
                if once is None or not hmac.compare_digest(
                    form.get("delete_source_token", ""), once
                ):
                    return forbidden()
                if (
                    set(form) != {"csrf", "delete_source_token", "confirm"}
                    or form["confirm"] != "yes"
                ):
                    return self._response(start_response, "400 Bad Request", "来源删除确认无效")
                try:
                    if not self._source_seal.exists_for_owner(session.owner_id):
                        raise ValueError("source seal unavailable")
                    self._clear_style_actions(session)
                    self._invalidate_style(session.owner_id)
                    if self._source_seal.exists_for_owner(session.owner_id):
                        raise ValueError("source seal delete readback changed")
                except Exception:
                    return self._response(
                        start_response, "503 Service Unavailable", "来源删除未就绪"
                    )
                session.retain_token = None
                session.retain_witness = None
                return self._response(start_response, "200 OK", "<p>来源封存已删除</p>")
            if action == "clear":
                if (
                    self._approval_store is None
                    or session.confirmed_digest is None
                    or session.candidate_package_bytes is None
                ):
                    return self._response(
                        start_response, "503 Service Unavailable", "审批清除未就绪"
                    )
                once = session.clear_token
                session.clear_token = None
                if once is None or not hmac.compare_digest(form.get("clear_token", ""), once):
                    return forbidden()
                if set(form) != {"csrf", "clear_token", "confirm"} or form["confirm"] != "yes":
                    return self._response(start_response, "400 Bad Request", "审批清除无效")
                session.decide_token = None
                session.retain_token = None
                session.retain_witness = None
                try:
                    records = self._confirmed_context(session)
                    package = self._session_package(session, records, allow_create=False)
                    self._clear_style_actions(session)
                    self._invalidate_style(session.owner_id)
                    self._approval_store.delete_stale(session.owner_id)
                    states = self._validated_decisions(
                        self._approval_store.decisions(session.owner_id, package),
                        len(package.candidates),
                    )
                    if any(state != "pending" for _, state in states):
                        raise ValueError("approval clear readback mismatch")
                except Exception:
                    return self._response(
                        start_response, "503 Service Unavailable", "审批清除未就绪"
                    )
                return self._response(start_response, "200 OK", "<p>审批记录已清除</p>")
            if action == "terms":
                once = session.terms_token
                session.terms_token = None
                if once is None or not hmac.compare_digest(form.get("terms_token", ""), once):
                    return forbidden()
                session.runtime_terms = None
                session.candidate_package_bytes = None
                session.candidate_package_digest = None
                session.decide_token = None
                session.clear_token = None
                session.retain_token = None
                session.retain_witness = None
                session.delete_source_token = None
                if self._runtime_selector is None or session.confirmed_digest is None:
                    return self._response(
                        start_response, "503 Service Unavailable", "运行时词未就绪"
                    )
                try:
                    self._confirmed_context(session)
                except Exception:
                    return self._response(
                        start_response, "503 Service Unavailable", "运行时词未就绪"
                    )
                if (
                    not {"csrf", "terms_token", "true_name_terms", "approved_nicknames"}.issubset(
                        form
                    )
                    or not set(form).issubset(
                        {
                            "csrf",
                            "terms_token",
                            "true_name_terms",
                            "approved_nicknames",
                            "confirm_nicknames",
                        }
                    )
                    or len(form["true_name_terms"].encode("utf-8"))
                    + len(form["approved_nicknames"].encode("utf-8"))
                    > 1024
                ):
                    return self._response(start_response, "400 Bad Request", "运行时词无效")
                try:
                    names = self._parse_terms(form["true_name_terms"], required=True)
                    nicknames = self._parse_terms(form["approved_nicknames"], required=False)
                except (ValueError, UnicodeError):
                    return self._response(start_response, "400 Bad Request", "运行时词无效")
                if (nicknames and form.get("confirm_nicknames") != "yes") or (
                    not nicknames and "confirm_nicknames" in form
                ):
                    return self._response(start_response, "400 Bad Request", "运行时词无效")
                session.runtime_terms = (names, nicknames)
                return self._response(start_response, "200 OK", "<p>本次词已接收</p>")
            if action == "confirm":
                once = session.confirm_token
                session.confirm_token = None
                if once is None or not hmac.compare_digest(form.get("confirm_token", ""), once):
                    return forbidden()
                self._clear_confirmation(session)
                if session.preview_digest is None:
                    return forbidden()
                try:
                    current, digest = self._load_batch()
                except Exception:
                    return self._response(
                        start_response, "503 Service Unavailable", "映射确认未就绪"
                    )
                if digest != session.preview_digest:
                    session.preview_digest = None
                    return self._response(start_response, "409 Conflict", "预览批次已变化")
                preview = preview_speaker_mapping(current.records)
                expected = {f"role_{index}" for index in range(len(preview.speakers))}
                extras = set(form) - {"csrf", "confirm_token", "confirm", "owner_id"}
                if form.get("confirm") != "yes" or extras != expected:
                    return self._response(start_response, "400 Bad Request", "映射确认无效")
                roles = {}
                for index, speaker in enumerate(preview.speakers):
                    value = form[f"role_{index}"]
                    if value not in {"owner", "other"}:
                        return self._response(start_response, "400 Bad Request", "映射确认无效")
                    roles[speaker.source_speaker_id] = SpeakerRole(value)
                try:
                    mapped = apply_confirmed_mapping(
                        current.records, ConfirmedSpeakerMapping(roles=roles)
                    )
                except Exception:
                    return self._response(start_response, "400 Bad Request", "映射确认无效")
                session.confirmed_records = tuple(mapped)
                session.confirmed_digest = digest
                session.preview_digest = None
                return self._response(
                    start_response, "200 OK", "<p>映射已确认</p><p>风格逐条审批未就绪</p>"
                )
            if action == "decide":
                if (
                    (self._candidate_reader is None and self._runtime_selector is None)
                    or self._approval_store is None
                    or session.confirmed_digest is None
                    or session.confirmed_records is None
                    or session.candidate_package_bytes is None
                ):
                    return self._response(
                        start_response, "503 Service Unavailable", "逐条决定未就绪"
                    )
                once = session.decide_token
                session.decide_token = None
                if once is None or not hmac.compare_digest(form.get("decide_token", ""), once):
                    return forbidden()
                raw_index = form.get("index", "")
                decision = form.get("decision")
                if (
                    not raw_index.isascii()
                    or not raw_index.isdecimal()
                    or decision
                    not in {
                        "approve",
                        "revoke",
                    }
                ):
                    return self._response(start_response, "400 Bad Request", "逐条决定无效")
                index = int(raw_index)
                if str(index) != raw_index:
                    return self._response(start_response, "400 Bad Request", "逐条决定无效")
                try:
                    records = self._confirmed_context(session)
                    package = self._session_package(session, records, allow_create=False)
                except Exception:
                    return self._response(
                        start_response, "503 Service Unavailable", "逐条决定未就绪"
                    )
                if index >= len(package.candidates):
                    return self._response(start_response, "400 Bad Request", "逐条决定无效")
                session.consent_token = None
                session.consent_state = None
                approved = decision == "approve"
                try:
                    self._clear_style_actions(session)
                    self._invalidate_style(session.owner_id)
                    result = self._approval_store.set_approved(
                        session.owner_id, package, index=index, approved=approved
                    )
                    states = self._validated_decisions(
                        self._approval_store.decisions(session.owner_id, package),
                        len(package.candidates),
                    )
                    expected_indices = tuple(i for i, state in states if state == "approved")
                    if (
                        type(result) is not tuple
                        or any(type(item) is not int for item in result)
                        or result != expected_indices
                    ):
                        raise ValueError("approval readback mismatch")
                    state = "approved" if approved else "pending"
                    if states[index][1] != state:
                        raise ValueError("approval readback mismatch")
                except Exception:
                    return self._response(
                        start_response, "503 Service Unavailable", "逐条决定未就绪"
                    )
                return self._response(
                    start_response,
                    "303 See Other",
                    f"<p>{index}: {state}</p>",
                    location=f"/admin/style/review#candidate-{index}",
                )
            return self._response(start_response, "503 Service Unavailable", "审批或确认未就绪")
        return self._response(start_response, "404 Not Found", "Not found")
