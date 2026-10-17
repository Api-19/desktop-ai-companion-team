"""Explicit production assembly shared by Dialogue entry points."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from companion.dialogue.config import DialogueConfig, DialogueSecrets, validate_framework_sources
from companion.dialogue.memory import MemoryGateway
from companion.dialogue.observer import TurnObserver
from companion.dialogue.privacy_revision import ContentRevisionSource
from companion.dialogue.profile import PartnerProfileProvider
from companion.dialogue.provider import OpenAICompatibleProvider
from companion.dialogue.shared_terms import SharedTerm, load_shared_terms
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime.settings import Settings

if TYPE_CHECKING:
    from companion.dialogue.orchestrator import DialogueOrchestrator, SnippetStyleSource
    from companion.dialogue.style_personalization import ProductionStylePersonalization
    from companion.style.snippet_source import FileSnippetStyleSource


def build_snippet_style_runtime(
    config: DialogueConfig, *, clock: Callable[[], datetime]
) -> tuple[FileSnippetStyleSource, str | None]:
    """Build a live partner snippet source and read the habit card once."""
    from companion.style.snippet_source import FileSnippetStyleSource

    sources = config.snippet_style
    if sources is None:
        raise ValueError("snippet style configuration unavailable")
    habit_card = None
    if sources.habit_card_source is not None:
        habit_card = sources.habit_card_source.read_text(encoding="utf-8")
        if not habit_card.strip():
            raise ValueError("habit card is empty")
    source = FileSnippetStyleSource(
        library_source=sources.library_source,
        partner_user_id=sources.partner_user_id,
        style_version=sources.style_version,
        clock=clock,
    )
    return source, habit_card


def load_snippet_shared_terms(config: DialogueConfig) -> tuple[SharedTerm, ...] | None:
    """Load the configured shared-terms dictionary; None when the segment has none."""
    sources = config.snippet_style
    if sources is None:
        raise ValueError("snippet style configuration unavailable")
    if sources.shared_terms_source is None:
        return None
    return load_shared_terms(sources.shared_terms_source)


def build_real_orchestrator(
    settings: Settings,
    config: DialogueConfig,
    secrets_: DialogueSecrets,
    *,
    trusted_user_id: str,
    enable_style_verifier: bool = False,
    enable_production_personalization: bool = False,
    memory_gateway: MemoryGateway | None = None,
    profile_source: PartnerProfileProvider | None = None,
    snippet_style: SnippetStyleSource | None = None,
    snippet_habit_card: str | None = None,
    snippet_shared_terms: Sequence[SharedTerm] | None = None,
    snippet_session_check: bool = False,
    turn_observer: TurnObserver | None = None,
    privacy_revision_sources: Sequence[ContentRevisionSource] = (),
) -> DialogueOrchestrator:
    from companion.dialogue.orchestrator import DialogueOrchestrator
    from companion.style import (
        DeterministicStyleVerifier,
        SingleUserStyleAdapter,
    )

    if snippet_style is None and snippet_habit_card is not None:
        raise ValueError("habit card requires snippet style")
    if snippet_style is not None and (enable_style_verifier or enable_production_personalization):
        raise ValueError("snippet style excludes style verifier and personalization")
    personalization = None
    if enable_production_personalization:
        try:
            personalization = _build_strict_personalization(settings, config, trusted_user_id)
        except Exception:
            raise ValueError("production personalization configuration unavailable") from None

    if snippet_style is None and config.style is None:
        raise ValueError("legacy style configuration unavailable")

    provider = OpenAICompatibleProvider(
        config=config.provider,
        api_key=secrets_.api_key,
        max_output_tokens=config.budget.max_output_tokens,
    )
    return DialogueOrchestrator(
        config=config,
        style_adapter=(
            SingleUserStyleAdapter(
                trusted_user_id=trusted_user_id,
                **config.style.as_call_kwargs(),
            )
            if snippet_style is None
            else None
        ),
        snippet_style=snippet_style,
        snippet_habit_card=snippet_habit_card,
        snippet_shared_terms=snippet_shared_terms,
        snippet_session_check=snippet_session_check,
        veiled_terms_path=(
            Path(__file__).resolve().parents[3] / "config/privacy/veiled_terms.json"
            if snippet_style is not None
            else None
        ),
        provider=provider,
        ledger=MonthlyUsageLedger(settings.runtime_data_dir / "dialogue", config.budget),
        clock=lambda: datetime.now(UTC),
        test_results_dir=settings.test_results_dir,
        memory_gateway=memory_gateway,
        profile_source=profile_source,
        style_verifier=(
            DeterministicStyleVerifier(trusted_user_id=trusted_user_id)
            if enable_style_verifier or enable_production_personalization
            else None
        ),
        style_personalization=personalization,
        turn_observer=turn_observer,
        privacy_revision_sources=privacy_revision_sources,
    )


def _build_strict_personalization(
    settings: Settings, config: DialogueConfig, trusted_user_id: str
) -> ProductionStylePersonalization:
    """Bind Phase 1/5 to the fixed Admin owner, paths and one source-approval gateway."""
    from companion.admin.app import load_admin_config
    from companion.admin.phase1 import (
        AdminSourceApprovalGateway,
        phase1_paths,
        phase5_paths,
        validate_phase1_paths,
        validate_phase5_paths,
    )
    from companion.admin.source_seal import StyleSourceSealStore
    from companion.dialogue.style_personalization import ProductionStylePersonalization
    from companion.privacy.style_candidate_approval import StyleCandidateApprovalStore
    from companion.style import (
        SingleTrustedUserStyleEvidenceStore,
        SingleTrustedUserStyleProfileStore,
    )

    root = settings.private_data_dir
    sources = config.style_framework
    owner = config.style_framework_owner_id
    if (
        sources is None
        or config.style is None
        or type(owner) is not str
        or type(trusted_user_id) is not str
        or owner != trusted_user_id
        or config.style.private_data_dir != root.resolve()
    ):
        raise ValueError
    admin = load_admin_config(settings)
    if admin.import_format is None or admin.owner_id != owner:
        raise ValueError
    first = (
        sources.consent_source,
        sources.profile_source,
        sources.profile_evidence_source,
        sources.statistics_source,
        sources.profile_index_source,
        sources.profile_cache_source,
    )
    fifth = (
        sources.retrieval_evidence_source,
        sources.retrieval_index_source,
        sources.retrieval_cache_source,
        sources.retrieval_version_source,
    )
    if first != phase1_paths(root) or fifth != phase5_paths(root):
        raise ValueError
    validate_framework_sources(sources, root)
    validate_phase1_paths(root)
    validate_phase5_paths(root)
    gateway = AdminSourceApprovalGateway(
        owner,
        StyleSourceSealStore(settings, admin),
        StyleCandidateApprovalStore(settings),
    )
    return ProductionStylePersonalization(
        trusted_user_id=owner,
        profile_gateway=SingleTrustedUserStyleProfileStore(
            trusted_user_id=owner,
            private_data_dir=root,
            consent_source=first[0],
            profile_source=first[1],
            evidence_source=first[2],
            statistics_source=first[3],
            index_source=first[4],
            cache_source=first[5],
            strict_source_binding=True,
            source_gateway=gateway,
        ),
        evidence_gateway=SingleTrustedUserStyleEvidenceStore(
            trusted_user_id=owner,
            private_data_dir=root,
            evidence_source=fifth[0],
            index_source=fifth[1],
            cache_source=fifth[2],
            version_source=fifth[3],
            strict_source_binding=True,
            source_gateway=gateway,
        ),
    )
