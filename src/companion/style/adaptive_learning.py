"""Consent-gated, user-scoped adaptive Style Profile lifecycle."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictFloat, StrictStr, StringConstraints, model_validator

from companion.contracts.base import TimezoneRequiredModel
from companion.style.errors import StyleValidationError
from companion.style.evidence_retrieval import StyleEvidenceRecord
from companion.style.private_io import (
    read_private_model,
    write_private_model,
    write_private_models_atomically,
)
from companion.style.user_profile import (
    StyleFeatureKind,
    UserStyleProfileStatus,
    UserStyleSnapshot,
)
from companion.style.verifier import (
    STYLE_VERIFIER_VERSION,
    StyleVerificationAction,
    StyleVerificationReport,
)

ADAPTIVE_STYLE_LEARNING_VERSION = "adaptive-style-learning-v1"
MIN_OBSERVATION_COUNT = 5
MIN_OBSERVATION_UTC_DAYS = 3
MIN_WEIGHTED_SUPPORT = 4.0
CONFLICT_SHARE = 0.20

UserId = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
OpaqueId = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=180,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9:._-]*$",
    ),
]


class _FrozenTimezoneModel(TimezoneRequiredModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdaptiveLearningConsentStatus(StrEnum):
    AUTHORIZED = "authorized"
    REVOKED = "revoked"


class StyleObservationSource(StrEnum):
    OWNER_MESSAGE = "owner_message"


class StyleObservationMeasurementMethod(StrEnum):
    DETERMINISTIC_FORM_MEASUREMENT = "deterministic_form_measurement"


class AdaptiveApprovedEvidenceSource(StrEnum):
    PHASE5_APPROVED_OWNER_EVIDENCE = "phase5_approved_owner_evidence"
    OWNER_DECLARATION = "owner_declaration"


class AdaptiveCandidateStatus(StrEnum):
    READY = "ready"
    CONFLICT = "conflict"
    DRIFT_BLOCKED = "drift_blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AdaptiveArtifactKind(StrEnum):
    STATISTICS = "statistics"
    INDEX = "index"
    CACHE = "cache"
    VERSION = "version"


class AdaptiveActivationKind(StrEnum):
    ACTIVATE = "activate"
    ROLLBACK = "rollback"


class AdaptiveStyleLearningConsent(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueId
    status: AdaptiveLearningConsentStatus
    updated_at: datetime


class StyleFeatureObservation(_FrozenTimezoneModel):
    """A content-free numeric observation from one verified owner message."""

    user_id: UserId
    observation_id: OpaqueId
    feature_kind: StyleFeatureKind
    value: StrictFloat
    source: StyleObservationSource
    measurement_method: StyleObservationMeasurementMethod = (
        StyleObservationMeasurementMethod.DETERMINISTIC_FORM_MEASUREMENT
    )
    semantic_route_used: Literal[False] = False
    authorization_version: OpaqueId
    source_version: OpaqueId
    observed_at: datetime

    @model_validator(mode="after")
    def require_statistical_owner_observation(self) -> Self:
        if self.feature_kind is StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE:
            raise ValueError("expression preferences cannot be inferred from observations")
        upper = 2_000.0 if _is_length(self.feature_kind) else 1.0
        if not 0.0 <= self.value <= upper:
            raise ValueError("adaptive observation value is outside its allowed range")
        return self


class StyleObservationBuffer(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueId
    observations: tuple[StyleFeatureObservation, ...] = Field(default=(), max_length=10_000)
    updated_at: datetime

    @model_validator(mode="after")
    def require_consistent_observations(self) -> Self:
        refs = [item.observation_id for item in self.observations]
        if len(refs) != len(set(refs)):
            raise ValueError("adaptive observation ids must be unique")
        if any(
            item.user_id != self.user_id
            or item.authorization_version != self.authorization_version
            or item.observed_at > self.updated_at
            for item in self.observations
        ):
            raise ValueError("adaptive observation buffer is inconsistent")
        return self


class AdaptiveApprovedEvidence(_FrozenTimezoneModel):
    user_id: UserId
    evidence_ref: OpaqueId
    feature_kind: StyleFeatureKind
    value: StrictFloat | StrictStr
    source: AdaptiveApprovedEvidenceSource
    source_ref: OpaqueId
    source_version: OpaqueId
    authorization_version: OpaqueId
    approved_at: datetime

    @model_validator(mode="after")
    def require_value_shape(self) -> Self:
        if self.feature_kind is StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE:
            if (
                type(self.value) is not str
                or not self.value.strip()
                or len(self.value.strip()) > 32
            ):
                raise ValueError("approved expression preference must be a short string")
        else:
            if type(self.value) is not float:
                raise ValueError("approved statistical evidence must use a float")
            upper = 2_000.0 if _is_length(self.feature_kind) else 1.0
            if not 0.0 <= self.value <= upper:
                raise ValueError("approved evidence value is outside its allowed range")
        return self


class AdaptiveApprovedEvidenceLedger(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueId
    evidence_version: OpaqueId
    records: tuple[AdaptiveApprovedEvidence, ...] = Field(default=(), max_length=1_000)
    updated_at: datetime

    @model_validator(mode="after")
    def require_consistent_evidence(self) -> Self:
        refs = [item.evidence_ref for item in self.records]
        if len(refs) != len(set(refs)):
            raise ValueError("adaptive approved evidence refs must be unique")
        if any(
            item.user_id != self.user_id
            or item.authorization_version != self.authorization_version
            or item.approved_at > self.updated_at
            for item in self.records
        ):
            raise ValueError("adaptive approved evidence ledger is inconsistent")
        return self


class AdaptiveFeatureProposal(_FrozenTimezoneModel):
    feature_kind: StyleFeatureKind
    proposed_value: StrictFloat | StrictStr
    base_value: StrictFloat | StrictStr | None = None
    observation_refs: tuple[OpaqueId, ...] = Field(default=(), max_length=10_000)
    approved_evidence_refs: tuple[OpaqueId, ...] = Field(default=(), max_length=1_000)
    raw_observation_count: int = Field(strict=True, ge=0, le=10_000)
    utc_day_count: int = Field(strict=True, ge=0, le=90)
    weighted_support: float = Field(strict=True, ge=0.0, le=10_000.0)
    upward_weight_share: float = Field(strict=True, ge=0.0, le=1.0)
    downward_weight_share: float = Field(strict=True, ge=0.0, le=1.0)
    conflict: bool
    drift_blocked: bool

    @model_validator(mode="after")
    def require_consistent_summary(self) -> Self:
        if self.feature_kind is StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE:
            if (
                type(self.proposed_value) is not str
                or not self.proposed_value.strip()
                or len(self.proposed_value.strip()) > 32
            ):
                raise ValueError("adaptive expression proposal is invalid")
        else:
            if type(self.proposed_value) is not float:
                raise ValueError("adaptive statistical proposal must use a float")
            upper = 2_000.0 if _is_length(self.feature_kind) else 1.0
            if not 0.0 <= self.proposed_value <= upper:
                raise ValueError("adaptive proposal value is outside its allowed range")
            if self.base_value is not None and type(self.base_value) is not float:
                raise ValueError("adaptive statistical base must use a float")
        if self.raw_observation_count != len(self.observation_refs):
            raise ValueError("adaptive observation summary count is invalid")
        if len(self.observation_refs) != len(set(self.observation_refs)):
            raise ValueError("adaptive observation references must be unique")
        if len(self.approved_evidence_refs) != len(set(self.approved_evidence_refs)):
            raise ValueError("adaptive evidence references must be unique")
        return self


class AdaptiveProfileCandidate(_FrozenTimezoneModel):
    user_id: UserId
    learning_version: OpaqueId
    proposal_id: OpaqueId
    candidate_version: OpaqueId
    authorization_version: OpaqueId
    base_profile_version: OpaqueId
    base_profile_authorization_version: OpaqueId
    base_snapshot: UserStyleSnapshot
    base_active_version: OpaqueId | None = None
    observation_version: OpaqueId
    evidence_version: OpaqueId
    status: AdaptiveCandidateStatus
    feature_proposals: tuple[AdaptiveFeatureProposal, ...] = Field(default=(), max_length=32)
    created_at: datetime

    @model_validator(mode="after")
    def require_consistent_status(self) -> Self:
        if self.learning_version != ADAPTIVE_STYLE_LEARNING_VERSION:
            raise ValueError("adaptive learning version is not registered")
        snapshot = _validated_snapshot(self.base_snapshot, user_id=self.user_id)
        if (
            self.base_profile_version != snapshot.profile.profile_version
            or self.base_profile_authorization_version != snapshot.consent.authorization_version
        ):
            raise ValueError("adaptive candidate base snapshot is inconsistent")
        kinds = [item.feature_kind for item in self.feature_proposals]
        if len(kinds) != len(set(kinds)):
            raise ValueError("adaptive candidate feature kinds must be unique")
        if self.status is AdaptiveCandidateStatus.INSUFFICIENT_EVIDENCE:
            if self.feature_proposals:
                raise ValueError("insufficient adaptive candidate must not expose proposals")
            self._require_deterministic_ids()
            return self
        if not self.feature_proposals:
            raise ValueError("adaptive candidate requires a feature proposal")
        expected = AdaptiveCandidateStatus.READY
        if any(item.conflict for item in self.feature_proposals):
            expected = AdaptiveCandidateStatus.CONFLICT
        elif any(item.drift_blocked for item in self.feature_proposals):
            expected = AdaptiveCandidateStatus.DRIFT_BLOCKED
        if self.status is not expected:
            raise ValueError("adaptive candidate status is inconsistent")
        self._require_deterministic_ids()
        return self

    def _require_deterministic_ids(self) -> None:
        serialized_snapshot = json.dumps(
            self.base_snapshot.model_dump(mode="json", warnings=False),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        serialized = json.dumps(
            [item.model_dump(mode="json", warnings=False) for item in self.feature_proposals],
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        expected_proposal = _stable_id(
            "proposal",
            self.authorization_version,
            self.base_profile_version,
            serialized_snapshot,
            "none" if self.base_active_version is None else self.base_active_version,
            self.observation_version,
            self.evidence_version,
            self.created_at.isoformat(),
            serialized,
        )
        expected_candidate = _stable_id("candidate", expected_proposal, self.status.value)
        if self.proposal_id != expected_proposal or self.candidate_version != expected_candidate:
            raise ValueError("adaptive candidate ids are not deterministic")


class AdaptiveActiveFeature(_FrozenTimezoneModel):
    feature_kind: StyleFeatureKind
    value: StrictFloat | StrictStr
    candidate_version: OpaqueId


class ActiveAdaptiveStyleProfile(_FrozenTimezoneModel):
    user_id: UserId
    active_version: OpaqueId
    candidate_version: OpaqueId
    authorization_version: OpaqueId
    base_profile_version: OpaqueId
    previous_active_version: OpaqueId | None = None
    evidence_version: OpaqueId
    owner_confirmation_version: OpaqueId
    verifier_version: OpaqueId
    activation_kind: AdaptiveActivationKind
    rollback_target_version: OpaqueId | None = None
    features: tuple[AdaptiveActiveFeature, ...] = Field(min_length=1, max_length=32)
    activated_at: datetime

    @property
    def feature_values(self) -> dict[StyleFeatureKind, float | str]:
        return {item.feature_kind: item.value for item in self.features}

    @model_validator(mode="after")
    def require_unique_features(self) -> Self:
        kinds = [item.feature_kind for item in self.features]
        if len(kinds) != len(set(kinds)):
            raise ValueError("active adaptive feature kinds must be unique")
        if self.activation_kind is AdaptiveActivationKind.ACTIVATE:
            if self.rollback_target_version is not None:
                raise ValueError("normal activation cannot bind a rollback target")
        else:
            if self.previous_active_version is None or self.rollback_target_version is None:
                raise ValueError("rollback activation requires source versions")
        expected = _active_version_id(
            activation_kind=self.activation_kind,
            candidate_version=self.candidate_version,
            owner_confirmation_version=self.owner_confirmation_version,
            previous_active_version=self.previous_active_version,
            rollback_target_version=self.rollback_target_version,
            base_profile_version=self.base_profile_version,
            evidence_version=self.evidence_version,
            verifier_version=self.verifier_version,
            features=self.features,
            activated_at=self.activated_at,
        )
        if self.active_version != expected:
            raise ValueError("adaptive active version is not deterministic")
        return self


class AdaptiveProfileHistory(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueId
    entries: tuple[ActiveAdaptiveStyleProfile, ...] = Field(default=(), max_length=1_000)
    updated_at: datetime

    @model_validator(mode="after")
    def require_consistent_history(self) -> Self:
        versions = [item.active_version for item in self.entries]
        if len(versions) != len(set(versions)):
            raise ValueError("adaptive active history versions must be unique")
        if any(
            item.user_id != self.user_id
            or item.authorization_version != self.authorization_version
            or item.activated_at > self.updated_at
            for item in self.entries
        ):
            raise ValueError("adaptive active history is inconsistent")
        return self


class AdaptiveDerivedMetadata(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueId
    artifact_kind: AdaptiveArtifactKind
    candidate_version: OpaqueId | None = None
    active_version: OpaqueId | None = None
    history_count: int = Field(strict=True, ge=0, le=1_000)
    updated_at: datetime


class AdaptiveFeatureCorrection(_FrozenTimezoneModel):
    user_id: UserId
    feature_kind: StyleFeatureKind
    corrected_value: StrictFloat | StrictStr
    authorization_version: OpaqueId
    correction_version: OpaqueId
    corrected_at: datetime
    explicitly_confirmed: bool

    @model_validator(mode="after")
    def require_explicit_valid_correction(self) -> Self:
        if self.explicitly_confirmed is not True:
            raise ValueError("adaptive correction requires explicit owner confirmation")
        AdaptiveApprovedEvidence(
            user_id=self.user_id,
            evidence_ref=self.correction_version,
            feature_kind=self.feature_kind,
            value=self.corrected_value,
            source=AdaptiveApprovedEvidenceSource.OWNER_DECLARATION,
            source_ref=self.correction_version,
            source_version=self.correction_version,
            authorization_version=self.authorization_version,
            approved_at=self.corrected_at,
        )
        return self


class AdaptiveLearningState(_FrozenTimezoneModel):
    consent: AdaptiveStyleLearningConsent
    observations: StyleObservationBuffer | None = None
    approved_evidence: AdaptiveApprovedEvidenceLedger | None = None
    candidate: AdaptiveProfileCandidate | None = None
    active: ActiveAdaptiveStyleProfile | None = None
    history: AdaptiveProfileHistory | None = None


class SingleTrustedUserAdaptiveStyleStore:
    """Fixed-path Phase 7 store for the single currently trusted owner."""

    def __init__(
        self,
        *,
        trusted_user_id: str,
        private_data_dir: Path,
        consent_source: Path,
        observation_source: Path,
        approved_evidence_source: Path,
        candidate_source: Path,
        active_source: Path,
        history_source: Path,
        statistics_source: Path,
        index_source: Path,
        cache_source: Path,
        version_source: Path,
    ) -> None:
        self._trusted_user_id = _validated_user_id(trusted_user_id)
        self._private_data_dir = private_data_dir
        self._consent_source = consent_source
        self._observation_source = observation_source
        self._approved_evidence_source = approved_evidence_source
        self._candidate_source = candidate_source
        self._active_source = active_source
        self._history_source = history_source
        self._statistics_source = statistics_source
        self._index_source = index_source
        self._cache_source = cache_source
        self._version_source = version_source
        resolved = self._validate_fixed_paths(self.artifact_paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("adaptive style artifact paths must be distinct")
        self._resolved_private_data_dir = self._private_data_dir.resolve(strict=True)

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return (self._consent_source, *self.derived_artifact_paths)

    @property
    def derived_artifact_paths(self) -> tuple[Path, ...]:
        return (
            self._observation_source,
            self._approved_evidence_source,
            self._candidate_source,
            self._active_source,
            self._history_source,
            self._statistics_source,
            self._index_source,
            self._cache_source,
            self._version_source,
        )

    def _validate_fixed_paths(self, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        try:
            root = self._private_data_dir.resolve(strict=True)
            resolved = tuple(path.resolve(strict=False) for path in paths)
        except OSError:
            raise ValueError("adaptive style paths could not be validated") from None
        pinned_root = getattr(self, "_resolved_private_data_dir", root)
        if root != pinned_root:
            raise ValueError("adaptive style private data path changed")
        if any(not path.is_relative_to(root) for path in resolved):
            raise ValueError("adaptive style paths must stay inside private data")
        return resolved

    def _require_owner(self, user_id: str) -> None:
        if type(user_id) is not str or user_id != self._trusted_user_id:
            raise StyleValidationError("untrusted_style_user", "style user is not trusted")
        try:
            self._validate_fixed_paths(self.artifact_paths)
        except ValueError:
            raise StyleValidationError(
                "unsafe_adaptive_style_path", "adaptive style path is invalid"
            ) from None

    def _read_consent(self) -> AdaptiveStyleLearningConsent | None:
        if not self._consent_source.exists():
            return None
        return read_private_model(
            AdaptiveStyleLearningConsent,
            self._consent_source,
            self._private_data_dir,
        )

    def _require_authorized(self, user_id: str) -> AdaptiveStyleLearningConsent:
        self._require_owner(user_id)
        consent = self._read_consent()
        if (
            consent is None
            or consent.user_id != user_id
            or consent.status is not AdaptiveLearningConsentStatus.AUTHORIZED
        ):
            raise StyleValidationError(
                "adaptive_style_not_authorized", "adaptive style learning is not authorized"
            )
        return consent

    def grant_learning_consent(
        self,
        *,
        user_id: str,
        consent: AdaptiveStyleLearningConsent,
    ) -> None:
        self._require_owner(user_id)
        if (
            type(consent) is not AdaptiveStyleLearningConsent
            or consent.user_id != user_id
            or consent.status is not AdaptiveLearningConsentStatus.AUTHORIZED
        ):
            raise StyleValidationError(
                "invalid_adaptive_consent", "adaptive style consent is invalid"
            )
        validated = AdaptiveStyleLearningConsent.model_validate(
            consent.model_dump(mode="python", warnings=False)
        )
        previous = self._read_consent()
        if previous != validated:
            self._delete_paths(self.derived_artifact_paths)
        write_private_model(validated, self._consent_source, self._private_data_dir)

    def view_learning_consent(self, user_id: str) -> AdaptiveStyleLearningConsent | None:
        self._require_owner(user_id)
        consent = self._read_consent()
        return consent if consent is not None and consent.user_id == user_id else None

    def record_observation(
        self,
        *,
        user_id: str,
        observation: StyleFeatureObservation,
    ) -> None:
        consent = self._require_authorized(user_id)
        validated = _deep_model(StyleFeatureObservation, observation)
        if (
            validated is None
            or validated.user_id != user_id
            or validated.authorization_version != consent.authorization_version
            or validated.observed_at < consent.updated_at
        ):
            raise StyleValidationError(
                "invalid_adaptive_observation", "adaptive style observation is invalid"
            )
        current = self._read_optional(StyleObservationBuffer, self._observation_source)
        observations = () if current is None else current.observations
        if current is not None and (
            current.user_id != user_id
            or current.authorization_version != consent.authorization_version
        ):
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")
        if any(item.observation_id == validated.observation_id for item in observations):
            raise StyleValidationError(
                "duplicate_adaptive_observation", "adaptive observation already exists"
            )
        buffer = StyleObservationBuffer(
            user_id=user_id,
            authorization_version=consent.authorization_version,
            observations=(*observations, validated),
            updated_at=max(item.observed_at for item in (*observations, validated)),
        )
        write_private_model(buffer, self._observation_source, self._private_data_dir)

    def propose_candidate(
        self,
        *,
        user_id: str,
        base_snapshot: UserStyleSnapshot,
        proposal_at: datetime,
    ) -> AdaptiveProfileCandidate:
        consent = self._require_authorized(user_id)
        snapshot = _validated_snapshot(base_snapshot, user_id=user_id)
        _require_aware(proposal_at)
        buffer = self._read_optional(StyleObservationBuffer, self._observation_source)
        ledger = self._read_optional(AdaptiveApprovedEvidenceLedger, self._approved_evidence_source)
        prior_candidate = self._read_optional(AdaptiveProfileCandidate, self._candidate_source)
        active = self._read_optional(ActiveAdaptiveStyleProfile, self._active_source)
        history = self._read_optional(AdaptiveProfileHistory, self._history_source)
        self._validate_optional_scopes(consent, buffer, ledger, prior_candidate, active, history)
        self._validate_active_chain(active, history)
        if prior_candidate is not None:
            self._validate_metadata(prior_candidate, active=active, history=history)
        proposals = _statistical_proposals(
            snapshot=snapshot,
            active=active,
            observations=() if buffer is None else buffer.observations,
            proposal_at=proposal_at,
        )
        proposals = _apply_quantile_gate(proposals, snapshot=snapshot, active=active)
        candidate = _candidate(
            user_id=user_id,
            consent=consent,
            snapshot=snapshot,
            active=active,
            observation_version=_observation_version(buffer),
            evidence_version=(
                "no-approved-evidence-v1" if ledger is None else ledger.evidence_version
            ),
            proposals=proposals,
            created_at=proposal_at,
        )
        self._publish_candidate(
            candidate,
            active=active,
            history_count=0 if history is None else len(history.entries),
        )
        return candidate

    def approve_phase5_expression(
        self,
        *,
        user_id: str,
        base_snapshot: UserStyleSnapshot,
        evidence_record: StyleEvidenceRecord,
        evidence_version: str,
        approved_at: datetime,
    ) -> AdaptiveProfileCandidate:
        consent = self._require_authorized(user_id)
        snapshot = _validated_snapshot(base_snapshot, user_id=user_id)
        record = _deep_model(StyleEvidenceRecord, evidence_record)
        _require_aware(approved_at)
        if (
            record is None
            or record.user_id != user_id
            or len(record.content.strip()) > 32
            or record.approved_at > approved_at
        ):
            raise StyleValidationError(
                "invalid_adaptive_evidence", "adaptive approved evidence is invalid"
            )
        approved = AdaptiveApprovedEvidence(
            user_id=user_id,
            evidence_ref=record.evidence_ref,
            feature_kind=StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE,
            value=record.content,
            source=AdaptiveApprovedEvidenceSource.PHASE5_APPROVED_OWNER_EVIDENCE,
            source_ref=record.evidence_ref,
            source_version=record.source_version,
            authorization_version=consent.authorization_version,
            approved_at=approved_at,
        )
        return self._candidate_from_approved(
            user_id=user_id,
            snapshot=snapshot,
            consent=consent,
            approved=approved,
            evidence_version=evidence_version,
            created_at=approved_at,
        )

    def correct_feature(
        self,
        *,
        user_id: str,
        base_snapshot: UserStyleSnapshot,
        correction: AdaptiveFeatureCorrection,
    ) -> AdaptiveProfileCandidate:
        consent = self._require_authorized(user_id)
        snapshot = _validated_snapshot(base_snapshot, user_id=user_id)
        validated = _deep_model(AdaptiveFeatureCorrection, correction)
        if (
            validated is None
            or validated.user_id != user_id
            or validated.authorization_version != consent.authorization_version
            or validated.corrected_at < consent.updated_at
        ):
            raise StyleValidationError(
                "invalid_adaptive_correction", "adaptive style correction is invalid"
            )
        approved = AdaptiveApprovedEvidence(
            user_id=user_id,
            evidence_ref=validated.correction_version,
            feature_kind=validated.feature_kind,
            value=validated.corrected_value,
            source=AdaptiveApprovedEvidenceSource.OWNER_DECLARATION,
            source_ref=validated.correction_version,
            source_version=validated.correction_version,
            authorization_version=consent.authorization_version,
            approved_at=validated.corrected_at,
        )
        return self._candidate_from_approved(
            user_id=user_id,
            snapshot=snapshot,
            consent=consent,
            approved=approved,
            evidence_version=f"evidence-{validated.correction_version}",
            created_at=validated.corrected_at,
        )

    def _candidate_from_approved(
        self,
        *,
        user_id: str,
        snapshot: UserStyleSnapshot,
        consent: AdaptiveStyleLearningConsent,
        approved: AdaptiveApprovedEvidence,
        evidence_version: str,
        created_at: datetime,
    ) -> AdaptiveProfileCandidate:
        _require_opaque(evidence_version, "adaptive evidence version")
        current_ledger = self._read_optional(
            AdaptiveApprovedEvidenceLedger, self._approved_evidence_source
        )
        prior_candidate = self._read_optional(AdaptiveProfileCandidate, self._candidate_source)
        records = () if current_ledger is None else current_ledger.records
        if current_ledger is not None and (
            current_ledger.user_id != user_id
            or current_ledger.authorization_version != consent.authorization_version
        ):
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")
        if any(item.evidence_ref == approved.evidence_ref for item in records):
            raise StyleValidationError(
                "duplicate_adaptive_evidence", "adaptive approved evidence already exists"
            )
        ledger = AdaptiveApprovedEvidenceLedger(
            user_id=user_id,
            authorization_version=consent.authorization_version,
            evidence_version=evidence_version,
            records=(*records, approved),
            updated_at=created_at,
        )
        active = self._read_optional(ActiveAdaptiveStyleProfile, self._active_source)
        history = self._read_optional(AdaptiveProfileHistory, self._history_source)
        self._validate_optional_scopes(consent, current_ledger, prior_candidate, active, history)
        self._validate_active_chain(active, history)
        if prior_candidate is not None:
            self._validate_metadata(prior_candidate, active=active, history=history)
        base_value = _current_value(snapshot, active, approved.feature_kind)
        drift = _drift_exceeded(approved.feature_kind, base_value, approved.value)
        proposal = AdaptiveFeatureProposal(
            feature_kind=approved.feature_kind,
            proposed_value=approved.value,
            base_value=base_value,
            approved_evidence_refs=(approved.evidence_ref,),
            raw_observation_count=0,
            utc_day_count=0,
            weighted_support=0.0,
            upward_weight_share=0.0,
            downward_weight_share=0.0,
            conflict=False,
            drift_blocked=drift,
        )
        proposals = _apply_quantile_gate((proposal,), snapshot=snapshot, active=active)
        candidate = _candidate(
            user_id=user_id,
            consent=consent,
            snapshot=snapshot,
            active=active,
            observation_version=_observation_version(
                self._read_optional(StyleObservationBuffer, self._observation_source)
            ),
            evidence_version=ledger.evidence_version,
            proposals=proposals,
            created_at=created_at,
        )
        metadata = self._metadata(
            candidate,
            active=active,
            history_count=0 if history is None else len(history.entries),
        )
        write_private_models_atomically(
            [
                (ledger, self._approved_evidence_source),
                (candidate, self._candidate_source),
                *metadata,
            ],
            self._private_data_dir,
        )
        return candidate

    def activate_candidate(
        self,
        *,
        user_id: str,
        base_snapshot: UserStyleSnapshot,
        candidate_version: str,
        owner_confirmation_version: str,
        accepts_drift: bool,
        verifier_report: StyleVerificationReport,
        activated_at: datetime,
    ) -> ActiveAdaptiveStyleProfile:
        consent = self._require_authorized(user_id)
        snapshot = _validated_snapshot(base_snapshot, user_id=user_id)
        _require_opaque(candidate_version, "adaptive candidate version")
        _require_opaque(owner_confirmation_version, "adaptive owner confirmation version")
        _require_aware(activated_at)
        if type(accepts_drift) is not bool:
            raise StyleValidationError(
                "invalid_adaptive_activation", "adaptive activation is invalid"
            )
        candidate = self._read_required(AdaptiveProfileCandidate, self._candidate_source)
        buffer = self._read_optional(StyleObservationBuffer, self._observation_source)
        ledger = self._read_optional(AdaptiveApprovedEvidenceLedger, self._approved_evidence_source)
        current = self._read_optional(ActiveAdaptiveStyleProfile, self._active_source)
        history = self._read_optional(AdaptiveProfileHistory, self._history_source)
        self._validate_optional_scopes(consent, buffer, ledger, candidate, current, history)
        self._validate_active_chain(current, history)
        self._validate_metadata(candidate, active=current, history=history)
        if current is not None and current.candidate_version == candidate.candidate_version:
            raise StyleValidationError(
                "duplicate_adaptive_activation", "adaptive candidate is already active"
            )
        if (
            candidate.user_id != user_id
            or candidate.authorization_version != consent.authorization_version
            or candidate.candidate_version != candidate_version
            or candidate.base_profile_version != snapshot.profile.profile_version
            or candidate.base_profile_authorization_version
            != snapshot.consent.authorization_version
            or candidate.base_snapshot != snapshot
            or candidate.base_active_version
            != (None if current is None else current.active_version)
            or activated_at < candidate.created_at
            or (current is not None and activated_at < current.activated_at)
        ):
            raise StyleValidationError(
                "invalid_adaptive_activation", "adaptive activation is invalid"
            )
        self._validate_candidate_sources(
            candidate,
            snapshot=snapshot,
            buffer=buffer,
            ledger=ledger,
            active=current,
        )
        if candidate.status is AdaptiveCandidateStatus.CONFLICT:
            raise StyleValidationError(
                "adaptive_candidate_conflict", "conflicting adaptive candidate cannot activate"
            )
        if candidate.status is AdaptiveCandidateStatus.INSUFFICIENT_EVIDENCE:
            raise StyleValidationError(
                "adaptive_candidate_insufficient", "adaptive candidate lacks evidence"
            )
        if candidate.status is AdaptiveCandidateStatus.DRIFT_BLOCKED and not accepts_drift:
            raise StyleValidationError(
                "adaptive_drift_approval_required", "adaptive drift requires explicit approval"
            )
        report = _validated_verifier_report(verifier_report)
        if (
            report is None
            or report.action is not StyleVerificationAction.ACCEPT
            or any(not check.covered or not check.passed for check in report.checks)
        ):
            raise StyleValidationError(
                "adaptive_verifier_rejected", "adaptive candidate failed verifier gate"
            )
        existing = () if history is None else history.entries
        history_entries = existing if current is None else (*existing, current)
        active_payload = dict(
            user_id=user_id,
            candidate_version=candidate.candidate_version,
            authorization_version=consent.authorization_version,
            base_profile_version=candidate.base_profile_version,
            previous_active_version=None if current is None else current.active_version,
            evidence_version=candidate.evidence_version,
            owner_confirmation_version=owner_confirmation_version,
            verifier_version=STYLE_VERIFIER_VERSION,
            activation_kind=AdaptiveActivationKind.ACTIVATE,
            features=tuple(
                AdaptiveActiveFeature(
                    feature_kind=item.feature_kind,
                    value=item.proposed_value,
                    candidate_version=candidate.candidate_version,
                )
                for item in candidate.feature_proposals
            ),
            activated_at=activated_at,
        )
        active = ActiveAdaptiveStyleProfile(
            active_version=_active_version_id(
                activation_kind=active_payload["activation_kind"],
                candidate_version=active_payload["candidate_version"],
                owner_confirmation_version=active_payload["owner_confirmation_version"],
                previous_active_version=active_payload["previous_active_version"],
                rollback_target_version=None,
                base_profile_version=active_payload["base_profile_version"],
                evidence_version=active_payload["evidence_version"],
                verifier_version=active_payload["verifier_version"],
                features=active_payload["features"],
                activated_at=active_payload["activated_at"],
            ),
            **active_payload,
        )
        new_history = AdaptiveProfileHistory(
            user_id=user_id,
            authorization_version=consent.authorization_version,
            entries=history_entries,
            updated_at=activated_at,
        )
        metadata = self._metadata(candidate, active=active, history_count=len(history_entries))
        write_private_models_atomically(
            [(active, self._active_source), (new_history, self._history_source), *metadata],
            self._private_data_dir,
        )
        return active

    def rollback(
        self,
        *,
        user_id: str,
        target_active_version: str,
        owner_confirmation_version: str,
        rolled_back_at: datetime,
    ) -> ActiveAdaptiveStyleProfile:
        consent = self._require_authorized(user_id)
        _require_opaque(target_active_version, "adaptive rollback target")
        _require_opaque(owner_confirmation_version, "adaptive rollback approval")
        _require_aware(rolled_back_at)
        current = self._read_required(ActiveAdaptiveStyleProfile, self._active_source)
        history = self._read_required(AdaptiveProfileHistory, self._history_source)
        candidate = self._read_required(AdaptiveProfileCandidate, self._candidate_source)
        self._validate_optional_scopes(consent, current, history, candidate)
        self._validate_active_chain(current, history)
        self._validate_metadata(candidate, active=current, history=history)
        if (
            current.user_id != user_id
            or history.user_id != user_id
            or current.authorization_version != consent.authorization_version
            or history.authorization_version != consent.authorization_version
            or rolled_back_at < current.activated_at
        ):
            raise StyleValidationError("invalid_adaptive_rollback", "adaptive rollback is invalid")
        target = next(
            (item for item in history.entries if item.active_version == target_active_version), None
        )
        if target is None or target.active_version == current.active_version:
            raise StyleValidationError("invalid_adaptive_rollback", "adaptive rollback is invalid")
        rollback_payload = dict(
            user_id=user_id,
            candidate_version=target.candidate_version,
            authorization_version=consent.authorization_version,
            base_profile_version=target.base_profile_version,
            previous_active_version=current.active_version,
            evidence_version=target.evidence_version,
            owner_confirmation_version=owner_confirmation_version,
            verifier_version=target.verifier_version,
            activation_kind=AdaptiveActivationKind.ROLLBACK,
            rollback_target_version=target.active_version,
            features=tuple(
                AdaptiveActiveFeature(
                    feature_kind=item.feature_kind,
                    value=item.value,
                    candidate_version=target.candidate_version,
                )
                for item in target.features
            ),
            activated_at=rolled_back_at,
        )
        rolled_back = ActiveAdaptiveStyleProfile(
            active_version=_active_version_id(
                activation_kind=rollback_payload["activation_kind"],
                candidate_version=rollback_payload["candidate_version"],
                owner_confirmation_version=rollback_payload["owner_confirmation_version"],
                previous_active_version=rollback_payload["previous_active_version"],
                rollback_target_version=rollback_payload["rollback_target_version"],
                base_profile_version=rollback_payload["base_profile_version"],
                evidence_version=rollback_payload["evidence_version"],
                verifier_version=rollback_payload["verifier_version"],
                features=rollback_payload["features"],
                activated_at=rollback_payload["activated_at"],
            ),
            **rollback_payload,
        )
        entries = (*history.entries, current)
        new_history = AdaptiveProfileHistory(
            user_id=user_id,
            authorization_version=consent.authorization_version,
            entries=entries,
            updated_at=rolled_back_at,
        )
        metadata = self._metadata(candidate, active=rolled_back, history_count=len(entries))
        write_private_models_atomically(
            [(rolled_back, self._active_source), (new_history, self._history_source), *metadata],
            self._private_data_dir,
        )
        return rolled_back

    def view_learning_state(self, user_id: str) -> AdaptiveLearningState | None:
        consent = self._require_authorized(user_id)
        observations = self._read_optional(StyleObservationBuffer, self._observation_source)
        approved = self._read_optional(
            AdaptiveApprovedEvidenceLedger, self._approved_evidence_source
        )
        candidate = self._read_optional(AdaptiveProfileCandidate, self._candidate_source)
        active = self._read_optional(ActiveAdaptiveStyleProfile, self._active_source)
        history = self._read_optional(AdaptiveProfileHistory, self._history_source)
        self._validate_optional_scopes(consent, observations, approved, candidate, active, history)
        if candidate is not None:
            self._validate_candidate_references(candidate, observations, approved)
            self._validate_metadata(candidate, active=active, history=history)
        elif any(path.exists() for path in self.derived_artifact_paths[4:]):
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")
        self._validate_active_chain(active, history)
        return AdaptiveLearningState(
            consent=consent,
            observations=observations,
            approved_evidence=approved,
            candidate=candidate,
            active=active,
            history=history,
        )

    def revoke_learning_consent(self, *, user_id: str, updated_at: datetime) -> None:
        consent = self._require_authorized(user_id)
        _require_aware(updated_at)
        if updated_at < consent.updated_at:
            raise StyleValidationError(
                "invalid_adaptive_consent", "adaptive style consent is invalid"
            )
        revoked = AdaptiveStyleLearningConsent(
            user_id=user_id,
            authorization_version=consent.authorization_version,
            status=AdaptiveLearningConsentStatus.REVOKED,
            updated_at=updated_at,
        )
        write_private_model(revoked, self._consent_source, self._private_data_dir)
        self._delete_paths(self.derived_artifact_paths)

    def delete_adaptive_learning(self, user_id: str) -> None:
        self._require_owner(user_id)
        self._delete_paths(self.artifact_paths)

    def _read_optional(self, model_type, path: Path):
        if not path.exists():
            return None
        try:
            return read_private_model(model_type, path, self._private_data_dir)
        except StyleValidationError:
            raise StyleValidationError(
                "adaptive_style_corrupt", "adaptive style data is invalid"
            ) from None

    def _read_required(self, model_type, path: Path):
        value = self._read_optional(model_type, path)
        if value is None:
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")
        return value

    @staticmethod
    def _validate_optional_scopes(consent, *items) -> None:
        for item in items:
            if item is not None and (
                item.user_id != consent.user_id
                or item.authorization_version != consent.authorization_version
            ):
                raise StyleValidationError(
                    "adaptive_style_corrupt", "adaptive style data is invalid"
                )

    def _publish_candidate(
        self,
        candidate: AdaptiveProfileCandidate,
        *,
        active: ActiveAdaptiveStyleProfile | None,
        history_count: int,
    ) -> None:
        write_private_models_atomically(
            [
                (candidate, self._candidate_source),
                *self._metadata(
                    candidate,
                    active=active,
                    history_count=history_count,
                ),
            ],
            self._private_data_dir,
        )

    @staticmethod
    def _validate_active_chain(
        active: ActiveAdaptiveStyleProfile | None,
        history: AdaptiveProfileHistory | None,
    ) -> None:
        if active is None:
            if history is not None:
                raise StyleValidationError(
                    "adaptive_style_corrupt", "adaptive style data is invalid"
                )
            return
        if history is None or history.updated_at != active.activated_at:
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")
        previous: str | None = None
        known_versions: set[str] = set()
        previous_time: datetime | None = None
        for entry in history.entries:
            if (
                entry.previous_active_version != previous
                or (previous_time is not None and entry.activated_at < previous_time)
                or (
                    entry.activation_kind is AdaptiveActivationKind.ROLLBACK
                    and entry.rollback_target_version not in known_versions
                )
            ):
                raise StyleValidationError(
                    "adaptive_style_corrupt", "adaptive style data is invalid"
                )
            known_versions.add(entry.active_version)
            previous = entry.active_version
            previous_time = entry.activated_at
        if (
            active.previous_active_version != previous
            or (previous_time is not None and active.activated_at < previous_time)
            or (
                active.activation_kind is AdaptiveActivationKind.ROLLBACK
                and active.rollback_target_version not in known_versions
            )
        ):
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")

    def _validate_metadata(
        self,
        candidate: AdaptiveProfileCandidate,
        *,
        active: ActiveAdaptiveStyleProfile | None,
        history: AdaptiveProfileHistory | None,
    ) -> None:
        expected = self._metadata(
            candidate,
            active=active,
            history_count=0 if history is None else len(history.entries),
        )
        for expected_model, path in expected:
            actual = self._read_required(AdaptiveDerivedMetadata, path)
            if actual != expected_model:
                raise StyleValidationError(
                    "adaptive_style_corrupt", "adaptive style data is invalid"
                )

    @staticmethod
    def _validate_candidate_references(
        candidate: AdaptiveProfileCandidate,
        buffer: StyleObservationBuffer | None,
        ledger: AdaptiveApprovedEvidenceLedger | None,
    ) -> None:
        if candidate.observation_version != _observation_version(buffer):
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")
        expected_evidence = "no-approved-evidence-v1" if ledger is None else ledger.evidence_version
        if candidate.evidence_version != expected_evidence:
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")
        observation_refs = (
            set() if buffer is None else {item.observation_id for item in buffer.observations}
        )
        evidence_refs = set() if ledger is None else {item.evidence_ref for item in ledger.records}
        if any(
            not set(proposal.observation_refs).issubset(observation_refs)
            or not set(proposal.approved_evidence_refs).issubset(evidence_refs)
            for proposal in candidate.feature_proposals
        ):
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")

    def _validate_candidate_sources(
        self,
        candidate: AdaptiveProfileCandidate,
        *,
        snapshot: UserStyleSnapshot,
        buffer: StyleObservationBuffer | None,
        ledger: AdaptiveApprovedEvidenceLedger | None,
        active: ActiveAdaptiveStyleProfile | None,
    ) -> None:
        self._validate_candidate_references(candidate, buffer, ledger)
        observation_refs = {
            ref for proposal in candidate.feature_proposals for ref in proposal.observation_refs
        }
        approved_refs = {
            ref
            for proposal in candidate.feature_proposals
            for ref in proposal.approved_evidence_refs
        }
        if observation_refs and approved_refs:
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")
        if observation_refs:
            observations = tuple(
                item
                for item in (() if buffer is None else buffer.observations)
                if item.observation_id in observation_refs
            )
            expected = _statistical_proposals(
                snapshot=snapshot,
                active=active,
                observations=observations,
                proposal_at=candidate.created_at,
            )
            expected = _apply_quantile_gate(expected, snapshot=snapshot, active=active)
            if expected != candidate.feature_proposals:
                raise StyleValidationError(
                    "adaptive_style_corrupt", "adaptive style data is invalid"
                )
        elif approved_refs:
            if ledger is None or len(candidate.feature_proposals) != 1:
                raise StyleValidationError(
                    "adaptive_style_corrupt", "adaptive style data is invalid"
                )
            proposal = candidate.feature_proposals[0]
            records = [item for item in ledger.records if item.evidence_ref in approved_refs]
            if (
                len(records) != 1
                or records[0].feature_kind is not proposal.feature_kind
                or records[0].value != proposal.proposed_value
            ):
                raise StyleValidationError(
                    "adaptive_style_corrupt", "adaptive style data is invalid"
                )
            expected = AdaptiveFeatureProposal(
                feature_kind=records[0].feature_kind,
                proposed_value=records[0].value,
                base_value=_current_value(snapshot, active, records[0].feature_kind),
                approved_evidence_refs=(records[0].evidence_ref,),
                raw_observation_count=0,
                utc_day_count=0,
                weighted_support=0.0,
                upward_weight_share=0.0,
                downward_weight_share=0.0,
                conflict=False,
                drift_blocked=_drift_exceeded(
                    records[0].feature_kind,
                    _current_value(snapshot, active, records[0].feature_kind),
                    records[0].value,
                ),
            )
            expected_proposals = _apply_quantile_gate((expected,), snapshot=snapshot, active=active)
            if expected_proposals != candidate.feature_proposals:
                raise StyleValidationError(
                    "adaptive_style_corrupt", "adaptive style data is invalid"
                )
        elif candidate.status is not AdaptiveCandidateStatus.INSUFFICIENT_EVIDENCE:
            raise StyleValidationError("adaptive_style_corrupt", "adaptive style data is invalid")

    def _metadata(
        self,
        candidate: AdaptiveProfileCandidate,
        *,
        active: ActiveAdaptiveStyleProfile | None,
        history_count: int = 0,
    ) -> list[tuple[AdaptiveDerivedMetadata, Path]]:
        paths = {
            AdaptiveArtifactKind.STATISTICS: self._statistics_source,
            AdaptiveArtifactKind.INDEX: self._index_source,
            AdaptiveArtifactKind.CACHE: self._cache_source,
            AdaptiveArtifactKind.VERSION: self._version_source,
        }
        return [
            (
                AdaptiveDerivedMetadata(
                    user_id=candidate.user_id,
                    authorization_version=candidate.authorization_version,
                    artifact_kind=kind,
                    candidate_version=candidate.candidate_version,
                    active_version=None if active is None else active.active_version,
                    history_count=history_count,
                    updated_at=(
                        candidate.created_at
                        if active is None
                        else max(candidate.created_at, active.activated_at)
                    ),
                ),
                path,
            )
            for kind, path in paths.items()
        ]

    def _delete_paths(self, paths: tuple[Path, ...]) -> None:
        try:
            self._validate_fixed_paths(paths)
        except ValueError:
            raise StyleValidationError(
                "adaptive_style_delete_failed", "adaptive style data could not be deleted"
            ) from None
        failed = False
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failed = True
        if failed:
            raise StyleValidationError(
                "adaptive_style_delete_failed", "adaptive style data could not be deleted"
            )


def _validated_user_id(user_id: str) -> str:
    if type(user_id) is not str or user_id != user_id.strip() or not 1 <= len(user_id) <= 120:
        raise ValueError("invalid trusted adaptive style user")
    return user_id


def _is_length(kind: StyleFeatureKind) -> bool:
    return kind in {
        StyleFeatureKind.MESSAGE_LENGTH_Q25,
        StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
        StyleFeatureKind.MESSAGE_LENGTH_Q75,
    }


def _deep_model(model_type, value):
    if type(value) is not model_type:
        return None
    with suppress(Exception):
        return model_type.model_validate(value.model_dump(mode="python", warnings=False))
    return None


def _validated_snapshot(snapshot: object, *, user_id: str) -> UserStyleSnapshot:
    validated = _deep_model(UserStyleSnapshot, snapshot)
    expected_evidence_count = (
        -1
        if validated is None
        else sum(item.provenance.evidence_count for item in validated.profile.features)
    )
    if (
        validated is None
        or validated.consent.user_id != user_id
        or validated.profile.user_id != user_id
        or validated.derived.user_id != user_id
        or validated.profile.authorization_version != validated.consent.authorization_version
        or validated.derived.authorization_version != validated.consent.authorization_version
        or validated.derived.profile_version != validated.profile.profile_version
        or validated.derived.updated_at != validated.profile.updated_at
        or validated.evidence_count != expected_evidence_count
        or validated.consent.status is not UserStyleProfileStatus.AUTHORIZED
    ):
        raise StyleValidationError(
            "invalid_adaptive_base_profile", "adaptive base profile is invalid"
        )
    return validated


def _require_aware(value: datetime) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise StyleValidationError("invalid_adaptive_time", "adaptive style time is invalid")


def _require_opaque(value: str, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 180
        or any(not (char.isalnum() or char in ":._-") for char in value)
    ):
        raise StyleValidationError("invalid_adaptive_version", f"{label} is invalid")


def _stable_id(prefix: str, *parts: str) -> str:
    canonical = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _active_version_id(
    *,
    activation_kind: AdaptiveActivationKind,
    candidate_version: str,
    owner_confirmation_version: str,
    previous_active_version: str | None,
    rollback_target_version: str | None,
    base_profile_version: str,
    evidence_version: str,
    verifier_version: str,
    features: tuple[AdaptiveActiveFeature, ...],
    activated_at: datetime,
) -> str:
    serialized_features = json.dumps(
        [item.model_dump(mode="json", warnings=False) for item in features],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return _stable_id(
        "active" if activation_kind is AdaptiveActivationKind.ACTIVATE else "rollback",
        candidate_version,
        owner_confirmation_version,
        "none" if previous_active_version is None else previous_active_version,
        "none" if rollback_target_version is None else rollback_target_version,
        base_profile_version,
        evidence_version,
        verifier_version,
        serialized_features,
        activated_at.isoformat(),
    )


def _observation_version(buffer: StyleObservationBuffer | None) -> str:
    if buffer is None:
        return "no-observations-v1"
    canonical = json.dumps(
        buffer.model_dump(mode="json", warnings=False),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return _stable_id("observations", canonical)


def _observation_weight(age_days: int) -> float:
    if 0 <= age_days <= 30:
        return 1.0
    if 31 <= age_days <= 60:
        return 0.75
    if 61 <= age_days <= 90:
        return 0.5
    return 0.0


def _base_profile_values(snapshot: UserStyleSnapshot) -> dict[StyleFeatureKind, float | str]:
    return {item.kind: item.value for item in snapshot.profile.features}


def _current_value(
    snapshot: UserStyleSnapshot,
    active: ActiveAdaptiveStyleProfile | None,
    kind: StyleFeatureKind,
) -> float | str | None:
    if active is not None and kind in active.feature_values:
        return active.feature_values[kind]
    return _base_profile_values(snapshot).get(kind)


def _drift_exceeded(
    kind: StyleFeatureKind,
    base_value: float | str | None,
    proposed_value: float | str,
) -> bool:
    if base_value is None or type(base_value) is not float or type(proposed_value) is not float:
        return False
    delta = abs(proposed_value - base_value)
    if _is_length(kind):
        return delta > max(8.0, abs(base_value) * 0.20)
    return delta > 0.10


def _statistical_proposals(
    *,
    snapshot: UserStyleSnapshot,
    active: ActiveAdaptiveStyleProfile | None,
    observations: tuple[StyleFeatureObservation, ...],
    proposal_at: datetime,
) -> tuple[AdaptiveFeatureProposal, ...]:
    by_kind: dict[StyleFeatureKind, list[tuple[StyleFeatureObservation, float]]] = {}
    proposal_utc = proposal_at.astimezone(UTC)
    for item in observations:
        age_seconds = (proposal_utc - item.observed_at.astimezone(UTC)).total_seconds()
        if age_seconds < 0:
            continue
        age_days = int(age_seconds // 86_400)
        weight = _observation_weight(age_days)
        if weight:
            by_kind.setdefault(item.feature_kind, []).append((item, weight))
    proposals: list[AdaptiveFeatureProposal] = []
    for kind in sorted(by_kind, key=str):
        eligible = by_kind[kind]
        days = {item.observed_at.astimezone(UTC).date() for item, _ in eligible}
        weighted_support = round(sum(weight for _, weight in eligible), 6)
        if (
            len(eligible) < MIN_OBSERVATION_COUNT
            or len(days) < MIN_OBSERVATION_UTC_DAYS
            or weighted_support < MIN_WEIGHTED_SUPPORT
        ):
            continue
        proposed = round(
            sum(item.value * weight for item, weight in eligible) / weighted_support,
            6,
        )
        base = _current_value(snapshot, active, kind)
        conflict_base = _current_value(snapshot, active, kind)
        upward = 0.0
        downward = 0.0
        if type(conflict_base) is float:
            upward = sum(weight for item, weight in eligible if item.value > conflict_base)
            downward = sum(weight for item, weight in eligible if item.value < conflict_base)
        up_share = round(upward / weighted_support, 6) if weighted_support else 0.0
        down_share = round(downward / weighted_support, 6) if weighted_support else 0.0
        conflict = up_share >= CONFLICT_SHARE and down_share >= CONFLICT_SHARE
        proposals.append(
            AdaptiveFeatureProposal(
                feature_kind=kind,
                proposed_value=proposed,
                base_value=base,
                observation_refs=tuple(item.observation_id for item, _ in eligible),
                raw_observation_count=len(eligible),
                utc_day_count=len(days),
                weighted_support=weighted_support,
                upward_weight_share=up_share,
                downward_weight_share=down_share,
                conflict=conflict,
                drift_blocked=_drift_exceeded(kind, base, proposed),
            )
        )
    return tuple(proposals)


def _apply_quantile_gate(
    proposals: tuple[AdaptiveFeatureProposal, ...],
    *,
    snapshot: UserStyleSnapshot,
    active: ActiveAdaptiveStyleProfile | None,
) -> tuple[AdaptiveFeatureProposal, ...]:
    values = _base_profile_values(snapshot)
    if active is not None:
        values.update(active.feature_values)
    values.update({item.feature_kind: item.proposed_value for item in proposals})
    ordered_kinds = (
        StyleFeatureKind.MESSAGE_LENGTH_Q25,
        StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
        StyleFeatureKind.MESSAGE_LENGTH_Q75,
    )
    present = [values[kind] for kind in ordered_kinds if kind in values]
    if len(present) < 2 or all(left <= right for left, right in pairwise(present)):
        return proposals
    gated: list[AdaptiveFeatureProposal] = []
    for item in proposals:
        payload = item.model_dump(mode="python", warnings=False)
        if item.feature_kind in ordered_kinds:
            payload["conflict"] = True
        gated.append(AdaptiveFeatureProposal.model_validate(payload))
    return tuple(gated)


def _candidate(
    *,
    user_id: str,
    consent: AdaptiveStyleLearningConsent,
    snapshot: UserStyleSnapshot,
    active: ActiveAdaptiveStyleProfile | None,
    observation_version: str,
    evidence_version: str,
    proposals: tuple[AdaptiveFeatureProposal, ...],
    created_at: datetime,
) -> AdaptiveProfileCandidate:
    if not proposals:
        status = AdaptiveCandidateStatus.INSUFFICIENT_EVIDENCE
    elif any(item.conflict for item in proposals):
        status = AdaptiveCandidateStatus.CONFLICT
    elif any(item.drift_blocked for item in proposals):
        status = AdaptiveCandidateStatus.DRIFT_BLOCKED
    else:
        status = AdaptiveCandidateStatus.READY
    serialized = json.dumps(
        [item.model_dump(mode="json", warnings=False) for item in proposals],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    serialized_snapshot = json.dumps(
        snapshot.model_dump(mode="json", warnings=False),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    proposal_id = _stable_id(
        "proposal",
        consent.authorization_version,
        snapshot.profile.profile_version,
        serialized_snapshot,
        "none" if active is None else active.active_version,
        observation_version,
        evidence_version,
        created_at.isoformat(),
        serialized,
    )
    candidate_version = _stable_id("candidate", proposal_id, status.value)
    return AdaptiveProfileCandidate(
        user_id=user_id,
        learning_version=ADAPTIVE_STYLE_LEARNING_VERSION,
        proposal_id=proposal_id,
        candidate_version=candidate_version,
        authorization_version=consent.authorization_version,
        base_profile_version=snapshot.profile.profile_version,
        base_profile_authorization_version=snapshot.consent.authorization_version,
        base_snapshot=snapshot,
        base_active_version=None if active is None else active.active_version,
        observation_version=observation_version,
        evidence_version=evidence_version,
        status=status,
        feature_proposals=proposals,
        created_at=created_at,
    )


def _validated_verifier_report(report: object) -> StyleVerificationReport | None:
    validated = _deep_model(StyleVerificationReport, report)
    if validated is None or validated.verifier_version != STYLE_VERIFIER_VERSION:
        return None
    return validated
