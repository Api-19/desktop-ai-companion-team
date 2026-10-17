"""Synthetic strict Phase 1/5 resources built only through real Admin/Privacy/Style services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dialogue_support import synthetic_dialogue_config
from pydantic import SecretStr

from companion.admin.app import AdminConfig, load_admin_config
from companion.admin.phase1 import (
    AdminSourceApprovalGateway,
    phase1_paths,
    phase5_paths,
    validate_phase1_paths,
    validate_phase5_paths,
)
from companion.admin.source_seal import StyleSourceSealStore
from companion.dialogue.models import DialogueConfig, DialogueSecrets, ProductionStyleSources
from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.privacy.style_candidate_approval import StyleCandidateApprovalStore
from companion.runtime.settings import Settings
from companion.style import (
    SingleTrustedUserStyleEvidenceStore,
    SingleTrustedUserStyleProfileStore,
    UserStyleSnapshotReadStatus,
    generate_style_evidence_catalog,
    generate_user_style_profile,
)

OWNER = "synthetic-owner"
T0 = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)
RUN_AT = T0 + timedelta(hours=1)
CANDIDATE_COUNT = 24
APPROVED_COUNT = 22
FIXTURE = Path("fixtures/synthetic/standard_private_chat.json")
PHASE1_FIELDS = (
    "consent_source",
    "profile_source",
    "profile_evidence_source",
    "statistics_source",
    "profile_index_source",
    "profile_cache_source",
)
PHASE5_FIELDS = (
    "retrieval_evidence_source",
    "retrieval_index_source",
    "retrieval_cache_source",
    "retrieval_version_source",
)


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700)
    path.chmod(0o700)


def _private_file(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    path.chmod(0o600)


def synthetic_package() -> SafeCandidatePackage:
    texts = [f"合成表达<{index}>" for index in range(CANDIDATE_COUNT)]
    return SafeCandidatePackage(
        created_at=T0 - timedelta(days=1),
        candidates=[
            SafeStyleCandidate(
                text=text,
                category=CandidateCategory.GREETING,
                scene_tags=[CandidateCategory.GREETING.value],
            )
            for text in texts
        ],
        audit=CandidateSelectionAudit(
            processed_at=T0 - timedelta(days=1),
            input_count=CANDIDATE_COUNT,
            owner_input_count=CANDIDATE_COUNT,
            dropped_other_count=0,
            allowed_category_counts={"greeting": CANDIDATE_COUNT},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=CANDIDATE_COUNT,
            succeeded=True,
        ),
    )


def framework_sources(root: Path) -> ProductionStyleSources:
    return ProductionStyleSources.model_validate(
        {
            **dict(zip(PHASE1_FIELDS, phase1_paths(root), strict=True)),
            **dict(zip(PHASE5_FIELDS, phase5_paths(root), strict=True)),
        }
    )


@dataclass
class StrictEnv:
    root: Path
    settings: Settings
    admin_config: AdminConfig
    package: SafeCandidatePackage
    seal: StyleSourceSealStore
    approvals: StyleCandidateApprovalStore
    gateway: AdminSourceApprovalGateway
    phase1: SingleTrustedUserStyleProfileStore
    phase5: SingleTrustedUserStyleEvidenceStore
    config: DialogueConfig
    secrets: DialogueSecrets

    def publish_phase5(
        self, selections: tuple[tuple[int, str], ...], observed_at: datetime
    ) -> None:
        state = self.gateway.current_state(OWNER)
        read = self.phase1.read_snapshot_state(OWNER)
        assert read.status is UserStyleSnapshotReadStatus.AUTHORIZED
        catalog = generate_style_evidence_catalog(
            user_id=OWNER,
            package=self.package,
            snapshot=read.snapshot,
            source_state=state,
            selections=selections,
            observed_at=observed_at,
        )
        self.phase5.publish_approved_evidence(
            user_id=OWNER, snapshot=read.snapshot, catalog=catalog
        )

    def regenerate_profile(self, observed_at: datetime) -> None:
        consent = self.phase1.view_consent(OWNER)
        bundle = generate_user_style_profile(
            user_id=OWNER,
            package=self.package,
            consent=consent,
            source_state=self.gateway.current_state(OWNER),
            observed_at=observed_at,
        )
        self.phase1.create_profile(
            user_id=OWNER, profile=bundle.profile, evidence=bundle.evidence, derived=bundle.derived
        )


def build_strict_env(
    tmp_path: Path,
    *,
    framework_owner: str | None = OWNER,
) -> StrictEnv:
    root = tmp_path / "private"
    _private_dir(root)
    _private_dir(root / "admin")
    _private_file(
        root / "admin" / "config.toml",
        f'enabled = true\nowner_id = "{OWNER}"\nimport_format = "standard_json"\n'.encode(),
    )
    _private_dir(root / "admin" / "import")
    _private_file(root / "admin" / "import" / "current.json", FIXTURE.read_bytes())
    settings = Settings(
        environment="test",
        log_level="ERROR",
        private_data_dir=root,
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )
    trusted_config = AdminConfig(owner_id=OWNER, import_format="standard_json")
    package = synthetic_package()
    seal = StyleSourceSealStore(settings, trusted_config)
    seal.seal(OWNER, seal._current_digest(), package)
    approvals = StyleCandidateApprovalStore(settings)
    for index in range(APPROVED_COUNT):
        approvals.set_approved(OWNER, package, index=index, approved=True)
    gateway = AdminSourceApprovalGateway(OWNER, seal, approvals)
    validate_phase1_paths(root, create=True)
    validate_phase5_paths(root, create=True)
    first = phase1_paths(root)
    fifth = phase5_paths(root)
    phase1 = SingleTrustedUserStyleProfileStore(
        trusted_user_id=OWNER,
        private_data_dir=root,
        consent_source=first[0],
        profile_source=first[1],
        evidence_source=first[2],
        statistics_source=first[3],
        index_source=first[4],
        cache_source=first[5],
        strict_source_binding=True,
        source_gateway=gateway,
    )
    phase5 = SingleTrustedUserStyleEvidenceStore(
        trusted_user_id=OWNER,
        private_data_dir=root,
        evidence_source=fifth[0],
        index_source=fifth[1],
        cache_source=fifth[2],
        version_source=fifth[3],
        strict_source_binding=True,
        source_gateway=gateway,
    )
    phase1.grant_bound_consent(user_id=OWNER, updated_at=T0)
    base = synthetic_dialogue_config(root)
    env = StrictEnv(
        root=root,
        settings=settings,
        admin_config=trusted_config,
        package=package,
        seal=seal,
        approvals=approvals,
        gateway=gateway,
        phase1=phase1,
        phase5=phase5,
        config=base.model_copy(
            update={
                "style_framework": framework_sources(root),
                "style_framework_owner_id": framework_owner,
            }
        ),
        secrets=DialogueSecrets(api_key=SecretStr("synthetic-placeholder-key")),
    )
    env.regenerate_profile(T0 + timedelta(minutes=1))
    env.publish_phase5(((0, "gratitude"), (1, "apology")), T0 + timedelta(minutes=2))
    # The test root must satisfy the real Admin reader used by the Dialogue builder.
    assert load_admin_config(settings).owner_id == OWNER
    return env
