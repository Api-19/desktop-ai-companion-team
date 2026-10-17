"""Strict, user-scoped Global Style Profile models and single-user storage."""

from __future__ import annotations

import secrets
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Protocol, Self, TypeVar, runtime_checkable

from pydantic import (
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from companion.contracts.base import TimezoneRequiredModel
from companion.style.errors import StyleValidationError
from companion.style.private_io import (
    read_private_model,
    write_private_model,
    write_private_models_atomically,
)

UserId = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
OpaqueVersion = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
Confidence = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


class _FrozenTimezoneModel(TimezoneRequiredModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UserStyleProfileStatus(StrEnum):
    AUTHORIZED = "authorized"
    REVOKED = "revoked"


class SourceApprovalState(_FrozenTimezoneModel):
    """Metadata from a trusted source reader, never an authentication credential."""

    owner_id: UserId
    batch_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    package_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    approved_indices: tuple[StrictInt, ...]
    revision: StrictStr = Field(pattern=r"^v1-[0-9a-f]{64}$")

    @field_validator("approved_indices")
    @classmethod
    def require_ordered_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(index) is not int or index < 0 for index in value):
            raise ValueError("invalid approved indices")
        if tuple(sorted(set(value))) != value:
            raise ValueError("invalid approved indices")
        return value


class SourceApprovalGateway(Protocol):
    def current_state(self, user_id: str) -> SourceApprovalState | None: ...


class StyleFeatureKind(StrEnum):
    """Allow-listed observable form features; content traits are intentionally absent."""

    MESSAGE_LENGTH_Q25 = "message_length_q25"
    MESSAGE_LENGTH_MEDIAN = "message_length_median"
    MESSAGE_LENGTH_Q75 = "message_length_q75"
    TERMINAL_PUNCTUATION_RATIO = "terminal_punctuation_ratio"
    EMOJI_USAGE_RATIO = "emoji_usage_ratio"
    FUNCTION_WORD_RATIO = "function_word_ratio"
    FOLLOW_UP_QUESTION_RATIO = "follow_up_question_ratio"
    CONFIRMED_EXPRESSION_PREFERENCE = "confirmed_expression_preference"


class StyleFeatureSourceType(StrEnum):
    PRIVACY_APPROVED_SAFE_CANDIDATES = "privacy_approved_safe_candidates"
    EXPLICIT_OWNER_CONFIRMATION = "explicit_owner_confirmation"


class StyleObservedFrom(StrEnum):
    OWNER_MESSAGES_ONLY = "owner_messages_only"
    OWNER_DECLARATION = "owner_declaration"


class StyleConfidenceMethod(StrEnum):
    DETERMINISTIC_STATISTIC = "deterministic_statistic"
    CALIBRATED_MODEL = "calibrated_model"
    EXPLICIT_RULE = "explicit_rule"
    USER_DECLARATION = "user_declaration"


class StyleFeatureProvenance(_FrozenTimezoneModel):
    source_type: StyleFeatureSourceType
    evidence_count: int = Field(strict=True, ge=1, le=1_000_000)
    observed_from: StyleObservedFrom
    source_version: OpaqueVersion
    updated_at: datetime
    confidence_method: StyleConfidenceMethod

    @model_validator(mode="after")
    def require_compatible_source(self) -> Self:
        approved = self.source_type is StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES
        if approved and self.observed_from is not StyleObservedFrom.OWNER_MESSAGES_ONLY:
            raise ValueError("approved evidence must be observed from owner messages only")
        if not approved and self.observed_from is not StyleObservedFrom.OWNER_DECLARATION:
            raise ValueError("declared evidence must be observed from an owner declaration")
        if approved and self.confidence_method is StyleConfidenceMethod.USER_DECLARATION:
            raise ValueError("approved evidence cannot use owner-declaration confidence")
        if not approved and self.confidence_method is not StyleConfidenceMethod.USER_DECLARATION:
            raise ValueError("declared evidence must use owner-declaration confidence")
        return self


class UserStyleFeature(_FrozenTimezoneModel):
    kind: StyleFeatureKind
    value: StrictFloat | StrictStr
    confidence: Confidence
    provenance: StyleFeatureProvenance

    @model_validator(mode="after")
    def require_value_shape(self) -> Self:
        if self.kind is StyleFeatureKind.CONFIRMED_EXPRESSION_PREFERENCE:
            if (
                type(self.value) is not str
                or not self.value.strip()
                or len(self.value.strip()) > 32
            ):
                raise ValueError("confirmed expression preference must be a short string")
            if (
                self.provenance.source_type
                is not StyleFeatureSourceType.EXPLICIT_OWNER_CONFIRMATION
            ):
                raise ValueError(
                    "confirmed expression preference requires explicit owner confirmation"
                )
        else:
            if type(self.value) is not float:
                raise ValueError("statistical style feature values must be floats")
            if self.kind in {
                StyleFeatureKind.MESSAGE_LENGTH_Q25,
                StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
                StyleFeatureKind.MESSAGE_LENGTH_Q75,
            }:
                upper = 2_000.0
            else:
                upper = 1.0
            if not 0.0 <= self.value <= upper:
                raise ValueError("style feature value is outside its allowed range")
        return self


class UserStyleProfileConsent(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueVersion
    status: UserStyleProfileStatus
    updated_at: datetime
    scope: StrictStr | None = None
    source_binding: SourceApprovalState | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def require_bound_scope(self) -> Self:
        if (self.scope is None) != (self.source_binding is None):
            raise ValueError("style consent binding is incomplete")
        if self.scope is not None and self.scope != "non_text_style_features_local_persistence":
            raise ValueError("style consent scope is invalid")
        if self.source_binding is not None and self.source_binding.owner_id != self.user_id:
            raise ValueError("style consent owner binding is invalid")
        return self


class UserStyleProfile(_FrozenTimezoneModel):
    user_id: UserId
    profile_version: OpaqueVersion
    authorization_version: OpaqueVersion
    features: tuple[UserStyleFeature, ...] = Field(min_length=1, max_length=32)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def require_consistent_profile(self) -> Self:
        kinds = [feature.kind for feature in self.features]
        if len(kinds) != len(set(kinds)):
            raise ValueError("style profile feature kinds must be unique")
        length_values = {
            feature.kind: feature.value
            for feature in self.features
            if feature.kind
            in {
                StyleFeatureKind.MESSAGE_LENGTH_Q25,
                StyleFeatureKind.MESSAGE_LENGTH_MEDIAN,
                StyleFeatureKind.MESSAGE_LENGTH_Q75,
            }
        }
        comparable_pairs = (
            (StyleFeatureKind.MESSAGE_LENGTH_Q25, StyleFeatureKind.MESSAGE_LENGTH_MEDIAN),
            (StyleFeatureKind.MESSAGE_LENGTH_MEDIAN, StyleFeatureKind.MESSAGE_LENGTH_Q75),
            (StyleFeatureKind.MESSAGE_LENGTH_Q25, StyleFeatureKind.MESSAGE_LENGTH_Q75),
        )
        if any(
            left in length_values
            and right in length_values
            and length_values[left] > length_values[right]
            for left, right in comparable_pairs
        ):
            raise ValueError("message length quantiles must be nondecreasing")
        if self.updated_at < self.created_at:
            raise ValueError("style profile updated_at must not precede created_at")
        if any(feature.provenance.updated_at > self.updated_at for feature in self.features):
            raise ValueError("style feature provenance cannot postdate the profile")
        return self


class UserStyleEvidenceLedger(_FrozenTimezoneModel):
    """Non-content evidence counts associated with one profile version."""

    user_id: UserId
    profile_version: OpaqueVersion
    authorization_version: OpaqueVersion
    feature_evidence: dict[StyleFeatureKind, int] = Field(min_length=1, max_length=32)
    updated_at: datetime

    @field_validator("feature_evidence")
    @classmethod
    def require_positive_exact_counts(
        cls, value: dict[StyleFeatureKind, int]
    ) -> dict[StyleFeatureKind, int]:
        if any(
            type(count) is not int or count < 1 or count > 1_000_000 for count in value.values()
        ):
            raise ValueError("style evidence counts must be positive exact integers")
        return value

    @property
    def total_evidence_count(self) -> int:
        return sum(self.feature_evidence.values())


class UserStyleDerivedArtifacts(_FrozenTimezoneModel):
    """Opaque deletion associations; contains no examples, queries, or message text."""

    user_id: UserId
    profile_version: OpaqueVersion
    authorization_version: OpaqueVersion
    statistics_version: OpaqueVersion
    index_version: OpaqueVersion
    cache_version: OpaqueVersion
    updated_at: datetime


class UserStyleSnapshot(_FrozenTimezoneModel):
    consent: UserStyleProfileConsent
    profile: UserStyleProfile
    evidence_count: int = Field(strict=True, ge=1)
    derived: UserStyleDerivedArtifacts


class UserStyleSnapshotReadStatus(StrEnum):
    AUTHORIZED = "authorized"
    UNAVAILABLE = "unavailable"
    CORRUPT = "corrupt"


class UserStyleSnapshotReadResult(_FrozenTimezoneModel):
    status: UserStyleSnapshotReadStatus
    snapshot: UserStyleSnapshot | None

    @model_validator(mode="after")
    def require_status_snapshot_pair(self) -> Self:
        if (self.snapshot is not None) != (self.status is UserStyleSnapshotReadStatus.AUTHORIZED):
            raise ValueError("style snapshot read state is invalid")
        if self.snapshot is not None:
            consent = self.snapshot.consent
            profile = self.snapshot.profile
            derived = self.snapshot.derived
            if (
                consent.status is not UserStyleProfileStatus.AUTHORIZED
                or (profile.user_id, profile.authorization_version)
                != (consent.user_id, consent.authorization_version)
                or (derived.user_id, derived.profile_version, derived.authorization_version)
                != (consent.user_id, profile.profile_version, consent.authorization_version)
                or derived.updated_at != profile.updated_at
                or self.snapshot.evidence_count
                != sum(feature.provenance.evidence_count for feature in profile.features)
            ):
                raise ValueError("style snapshot read state is invalid")
        return self


SnapshotModelT = TypeVar("SnapshotModelT", bound=_FrozenTimezoneModel)


@runtime_checkable
class UserStyleReadGateway(Protocol):
    def read_snapshot_state(self, user_id: str) -> UserStyleSnapshotReadResult: ...

    def load_snapshot(self, user_id: str) -> UserStyleSnapshot | None: ...

    def view_profile(self, user_id: str) -> UserStyleProfile | None: ...

    def view_consent(self, user_id: str) -> UserStyleProfileConsent | None: ...


@runtime_checkable
class UserStyleManagementGateway(Protocol):
    def grant_consent(self, *, user_id: str, consent: UserStyleProfileConsent) -> None: ...

    def create_profile(
        self,
        *,
        user_id: str,
        profile: UserStyleProfile,
        evidence: UserStyleEvidenceLedger,
        derived: UserStyleDerivedArtifacts,
    ) -> None: ...

    def revoke_consent(self, *, user_id: str, updated_at: datetime) -> None: ...

    def delete_user_style(self, user_id: str) -> None: ...


class UserStyleProfileGateway(UserStyleReadGateway, UserStyleManagementGateway, Protocol):
    pass


class SingleTrustedUserStyleProfileStore:
    """Fixed-path Profile store for the one currently supported trusted user."""

    def __init__(
        self,
        *,
        trusted_user_id: str,
        private_data_dir: Path,
        consent_source: Path,
        profile_source: Path,
        evidence_source: Path,
        statistics_source: Path,
        index_source: Path,
        cache_source: Path,
        strict_source_binding: bool = False,
        source_gateway: SourceApprovalGateway | None = None,
    ) -> None:
        if type(strict_source_binding) is not bool or (
            strict_source_binding and source_gateway is None
        ):
            raise ValueError("strict style source gateway unavailable")
        self._trusted_user_id = self._validated_user_id(trusted_user_id)
        self._strict_source_binding = strict_source_binding
        self._source_gateway = source_gateway
        self._private_data_dir = private_data_dir
        self._consent_source = consent_source
        self._profile_source = profile_source
        self._evidence_source = evidence_source
        self._statistics_source = statistics_source
        self._index_source = index_source
        self._cache_source = cache_source
        paths = self.artifact_paths
        resolved_paths = self._validate_fixed_paths(paths)
        if len(set(resolved_paths)) != len(resolved_paths):
            raise ValueError("style profile artifact paths must be distinct")

    @staticmethod
    def _validated_user_id(user_id: str) -> str:
        if type(user_id) is not str or not user_id.strip() or len(user_id) > 120:
            raise ValueError(
                "user_id must be an exact non-empty builtin str of at most 120 characters"
            )
        return user_id

    def _validate_fixed_paths(self, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        try:
            root = self._private_data_dir.resolve(strict=True)
            resolved = tuple(path.resolve(strict=False) for path in paths)
        except OSError:
            raise ValueError("style profile paths could not be validated") from None
        if any(not path.is_relative_to(root) for path in resolved):
            raise ValueError("style profile paths must stay inside private data")
        return resolved

    def _accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str and user_id == self._trusted_user_id

    def _require_owner(self, user_id: str) -> None:
        if not self._accepts_user(user_id):
            raise StyleValidationError("untrusted_style_user", "style user is not trusted")

    @property
    def consent_source(self) -> Path:
        return self._consent_source

    @property
    def derived_artifact_paths(self) -> tuple[Path, ...]:
        return (
            self._profile_source,
            self._evidence_source,
            self._statistics_source,
            self._index_source,
            self._cache_source,
        )

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return (self._consent_source, *self.derived_artifact_paths)

    def grant_consent(self, *, user_id: str, consent: UserStyleProfileConsent) -> None:
        self._require_owner(user_id)
        if self._strict_source_binding:
            raise StyleValidationError("invalid_style_consent", "style consent is invalid")
        if consent.user_id != user_id or consent.status is not UserStyleProfileStatus.AUTHORIZED:
            raise StyleValidationError("invalid_style_consent", "style consent is invalid")
        previous = self._read_consent()
        write_private_model(consent, self._consent_source, self._private_data_dir)
        if previous is None or previous.authorization_version != consent.authorization_version:
            self._delete_paths(self.derived_artifact_paths)

    def _current_source_state(self, user_id: str) -> SourceApprovalState:
        gateway = self._source_gateway
        if gateway is None:
            raise StyleValidationError("style_source_unavailable", "style source is unavailable")
        try:
            candidate = gateway.current_state(user_id)
            if type(candidate) is not SourceApprovalState:
                raise ValueError("invalid source state")
            state = SourceApprovalState.model_validate(candidate.model_dump(mode="python"))
            if state.owner_id != user_id or not state.approved_indices:
                raise ValueError("invalid source state")
            return state
        except Exception:
            raise StyleValidationError(
                "style_source_unavailable", "style source is unavailable"
            ) from None

    def _bound_source_matches(self, user_id: str, consent: UserStyleProfileConsent) -> bool:
        if (
            consent.scope != "non_text_style_features_local_persistence"
            or consent.source_binding is None
        ):
            return False
        try:
            return self._current_source_state(user_id) == consent.source_binding
        except StyleValidationError:
            return False

    def grant_bound_consent(self, *, user_id: str, updated_at: datetime) -> UserStyleProfileConsent:
        self._require_owner(user_id)
        if not self._strict_source_binding:
            raise StyleValidationError("invalid_style_consent", "style consent is invalid")
        state = self._current_source_state(user_id)
        previous = self._read_consent()
        version = "bound-" + secrets.token_hex(24)
        while previous is not None and version == previous.authorization_version:
            version = "bound-" + secrets.token_hex(24)
        consent = UserStyleProfileConsent(
            user_id=user_id,
            authorization_version=version,
            status=UserStyleProfileStatus.AUTHORIZED,
            updated_at=updated_at,
            scope="non_text_style_features_local_persistence",
            source_binding=state,
        )
        write_private_model(consent, self._consent_source, self._private_data_dir)
        self._delete_paths(self.derived_artifact_paths)
        return consent

    def _read_consent(self) -> UserStyleProfileConsent | None:
        if not self._consent_source.exists():
            return None
        try:
            return read_private_model(
                UserStyleProfileConsent, self._consent_source, self._private_data_dir
            )
        except StyleValidationError:
            return None

    def view_consent(self, user_id: str) -> UserStyleProfileConsent | None:
        if not self._accepts_user(user_id):
            return None
        consent = self._read_consent()
        if consent is None or consent.user_id != user_id:
            return None
        return consent

    def create_profile(
        self,
        *,
        user_id: str,
        profile: UserStyleProfile,
        evidence: UserStyleEvidenceLedger,
        derived: UserStyleDerivedArtifacts,
    ) -> None:
        self._require_owner(user_id)
        consent = self._read_consent()
        if (
            consent is None
            or consent.user_id != user_id
            or consent.status is not UserStyleProfileStatus.AUTHORIZED
        ):
            raise StyleValidationError(
                "style_profile_not_authorized", "style profile is not authorized"
            )
        if self._strict_source_binding and not self._bound_source_matches(user_id, consent):
            raise StyleValidationError(
                "style_profile_not_authorized", "style profile is not authorized"
            )
        identities = {
            (profile.user_id, profile.profile_version, profile.authorization_version),
            (evidence.user_id, evidence.profile_version, evidence.authorization_version),
            (derived.user_id, derived.profile_version, derived.authorization_version),
        }
        expected = (user_id, profile.profile_version, consent.authorization_version)
        if identities != {expected}:
            raise StyleValidationError("invalid_style_profile", "style profile metadata is invalid")
        expected_counts = {
            feature.kind: feature.provenance.evidence_count for feature in profile.features
        }
        if (
            evidence.feature_evidence != expected_counts
            or evidence.updated_at != profile.updated_at
        ):
            raise StyleValidationError("invalid_style_profile", "style profile evidence is invalid")
        if derived.updated_at != profile.updated_at:
            raise StyleValidationError(
                "invalid_style_profile", "style profile derivatives are invalid"
            )
        write_private_models_atomically(
            [
                (profile, self._profile_source),
                (evidence, self._evidence_source),
                (derived, self._statistics_source),
                (derived, self._index_source),
                (derived, self._cache_source),
            ],
            self._private_data_dir,
        )

    @staticmethod
    def _snapshot_path_present(path: Path) -> bool:
        try:
            path.stat()
        except FileNotFoundError:
            return False
        except OSError:
            raise StyleValidationError(
                "style_snapshot_read_failed", "style snapshot could not be read"
            ) from None
        return True

    @staticmethod
    def _snapshot_model(model_type: type[SnapshotModelT], path: Path) -> SnapshotModelT | None:
        try:
            payload = path.read_bytes()
        except OSError:
            raise StyleValidationError(
                "style_snapshot_read_failed", "style snapshot could not be read"
            ) from None
        try:
            return model_type.model_validate_json(payload)
        except (ValidationError, UnicodeError, ValueError):
            return None

    def read_snapshot_state(self, user_id: str) -> UserStyleSnapshotReadResult:
        self._require_owner(user_id)
        try:
            self._validate_fixed_paths(self.artifact_paths)
        except ValueError:
            raise StyleValidationError(
                "style_snapshot_read_failed", "style snapshot could not be read"
            ) from None
        unavailable = UserStyleSnapshotReadResult(
            status=UserStyleSnapshotReadStatus.UNAVAILABLE, snapshot=None
        )
        corrupt = UserStyleSnapshotReadResult(
            status=UserStyleSnapshotReadStatus.CORRUPT, snapshot=None
        )
        if not self._snapshot_path_present(self._consent_source):
            return unavailable
        consent = self._snapshot_model(UserStyleProfileConsent, self._consent_source)
        if consent is None or consent.user_id != user_id:
            return corrupt
        if consent.status is UserStyleProfileStatus.REVOKED:
            return unavailable
        if self._strict_source_binding and not self._bound_source_matches(user_id, consent):
            return corrupt
        present = tuple(self._snapshot_path_present(path) for path in self.derived_artifact_paths)
        if not any(present):
            return unavailable
        if not all(present):
            return corrupt
        profile = self._snapshot_model(UserStyleProfile, self._profile_source)
        evidence = self._snapshot_model(UserStyleEvidenceLedger, self._evidence_source)
        derived_items = tuple(
            self._snapshot_model(UserStyleDerivedArtifacts, path)
            for path in (self._statistics_source, self._index_source, self._cache_source)
        )
        if profile is None or evidence is None or any(item is None for item in derived_items):
            return corrupt
        if len(set(derived_items)) != 1:
            return corrupt
        derived = derived_items[0]
        expected_identity = (user_id, profile.profile_version, consent.authorization_version)
        identities = {
            (profile.user_id, profile.profile_version, profile.authorization_version),
            (evidence.user_id, evidence.profile_version, evidence.authorization_version),
            (derived.user_id, derived.profile_version, derived.authorization_version),
        }
        expected_counts = {
            feature.kind: feature.provenance.evidence_count for feature in profile.features
        }
        if (
            identities != {expected_identity}
            or evidence.feature_evidence != expected_counts
            or evidence.updated_at != profile.updated_at
            or derived.updated_at != profile.updated_at
        ):
            return corrupt
        return UserStyleSnapshotReadResult(
            status=UserStyleSnapshotReadStatus.AUTHORIZED,
            snapshot=UserStyleSnapshot(
                consent=consent,
                profile=profile,
                evidence_count=evidence.total_evidence_count,
                derived=derived,
            ),
        )

    def load_snapshot(self, user_id: str) -> UserStyleSnapshot | None:
        if not self._accepts_user(user_id):
            return None
        try:
            return self.read_snapshot_state(user_id).snapshot
        except StyleValidationError:
            return None

    def view_profile(self, user_id: str) -> UserStyleProfile | None:
        snapshot = self.load_snapshot(user_id)
        return None if snapshot is None else snapshot.profile

    def _delete_paths(self, paths: tuple[Path, ...]) -> None:
        try:
            self._validate_fixed_paths(paths)
        except ValueError:
            raise StyleValidationError(
                "style_profile_delete_failed", "style profile artifacts could not be deleted"
            ) from None
        failed = False
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failed = True
        if failed:
            raise StyleValidationError(
                "style_profile_delete_failed", "style profile artifacts could not be deleted"
            )

    def revoke_consent(self, *, user_id: str, updated_at: datetime) -> None:
        self._require_owner(user_id)
        consent = self._read_consent()
        if consent is None or consent.user_id != user_id:
            raise StyleValidationError(
                "style_profile_not_authorized", "style profile is not authorized"
            )
        revoked = UserStyleProfileConsent(
            user_id=user_id,
            authorization_version=consent.authorization_version,
            status=UserStyleProfileStatus.REVOKED,
            updated_at=updated_at,
            scope=consent.scope,
            source_binding=consent.source_binding,
        )
        write_private_model(revoked, self._consent_source, self._private_data_dir)
        self._delete_paths(self.derived_artifact_paths)

    def delete_user_style(self, user_id: str) -> None:
        self._require_owner(user_id)
        if self._strict_source_binding:
            consent = self._read_consent()
            if consent is not None and consent.user_id == user_id:
                self.revoke_consent(user_id=user_id, updated_at=consent.updated_at)
        self._delete_paths(self.artifact_paths)
