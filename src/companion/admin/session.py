from __future__ import annotations

from dataclasses import dataclass, field

from companion.admin.memory import MemoryFormState
from companion.admin.partner_profile import PartnerFormState
from companion.contracts import ChatRecord
from companion.privacy.style_candidate_approval import ApprovalWitness
from companion.style import SourceApprovalState


@dataclass
class _BrowserSession:
    owner_id: str
    secret: str
    expires: float
    absolute_expires: float
    preview_digest: str | None = None
    confirm_token: str | None = None
    confirmed_records: tuple[ChatRecord, ...] | None = None
    confirmed_digest: str | None = None
    candidate_package_bytes: bytes | None = None
    candidate_package_digest: bytes | None = None
    decide_token: str | None = None
    terms_token: str | None = None
    runtime_terms: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    clear_token: str | None = None
    retain_token: str | None = None
    retain_witness: ApprovalWitness | None = None
    delete_source_token: str | None = None
    consent_token: str | None = None
    consent_state: SourceApprovalState | None = None
    revoke_token: str | None = None
    delete_token: str | None = None
    profile_token: str | None = None
    profile_binding: tuple[SourceApprovalState, str] | None = None
    evidence_token: str | None = None
    evidence_binding: tuple[SourceApprovalState, str, str] | None = None
    evidence_basket: tuple[tuple[int, str], ...] = ()
    memory_actions: dict[str, tuple[str, MemoryFormState]] = field(default_factory=dict)
    partner_actions: dict[str, tuple[str, PartnerFormState]] = field(default_factory=dict)
    snippets_token: str | None = None
    audit_token: str | None = None
    character_form: tuple[str, str] | None = None
