"""User-scoped retrieval of explicitly approved historical style expressions."""

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import ConfigDict, Field, StrictInt, StrictStr, StringConstraints, model_validator

from companion.contracts.base import TimezoneRequiredModel
from companion.style.errors import StyleValidationError
from companion.style.open_set_routing import OpenSetRouteKind, OpenSetStyleRoute
from companion.style.private_io import read_private_model, write_private_models_atomically
from companion.style.user_profile import (
    SourceApprovalGateway,
    SourceApprovalState,
    UserStyleProfileStatus,
    UserStyleSnapshot,
)

STYLE_EVIDENCE_RETRIEVAL_VERSION = "style-evidence-retrieval-v1"
MIN_STYLE_EVIDENCE_CONFIDENCE = 0.6
MAX_RETRIEVED_EVIDENCE = 3

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
EvidenceContent = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=160),
]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
CandidateIndex = Annotated[StrictInt, Field(ge=0)]
EvidenceUsageScope = Literal["approved_style_expressions_local_persistence"]


def _validate_source_binding(user_id, binding, scope, entries) -> None:
    if (binding is None) != (scope is None):
        raise ValueError("style evidence binding is incomplete")
    if binding is None:
        return
    indices = tuple(entry.candidate_index for entry in entries)
    if (
        binding.owner_id != user_id
        or not entries
        or any(index is None or index not in binding.approved_indices for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("style evidence source indices are invalid")


class _FrozenTimezoneModel(TimezoneRequiredModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StyleEvidenceApproval(StrEnum):
    APPROVED_FOR_STYLE = "approved_for_style"


class StyleEvidenceSourceType(StrEnum):
    EXPLICIT_OWNER_APPROVAL = "explicit_owner_approval"


class StyleEvidenceObservedFrom(StrEnum):
    OWNER_MESSAGES_ONLY = "owner_messages_only"


class StyleEvidenceConfidenceMethod(StrEnum):
    EXPLICIT_OWNER_APPROVAL = "explicit_owner_approval"


class StyleEvidenceRetrievalStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    AVAILABLE = "available"
    NO_MATCH = "no_match"
    LOW_CONFIDENCE = "low_confidence"
    CORRUPT = "corrupt"


class StyleEvidenceVersions(_FrozenTimezoneModel):
    evidence_version: OpaqueId
    index_version: OpaqueId
    cache_version: OpaqueId


class StyleEvidenceRecord(_FrozenTimezoneModel):
    """One private expression explicitly approved by the owner for style use."""

    user_id: UserId
    evidence_ref: OpaqueId
    scenario_id: OpaqueId
    content: EvidenceContent
    confidence: UnitInterval
    approval: StyleEvidenceApproval
    source_type: StyleEvidenceSourceType
    observed_from: StyleEvidenceObservedFrom
    confidence_method: StyleEvidenceConfidenceMethod
    source_version: OpaqueId
    approved_at: datetime
    candidate_index: CandidateIndex | None = Field(default=None, repr=False)

    @model_validator(mode="before")
    @classmethod
    def preserve_bound_text(cls, value):
        # Legacy records retain their normalization; selected bound text is exact.
        if type(value) is dict and value.get("candidate_index") is None:
            text = value.get("content")
            if type(text) is str:
                value = dict(value, content=text.strip())
        return value

    @model_validator(mode="after")
    def require_explicit_owner_approval(self) -> Self:
        if (
            self.approval is not StyleEvidenceApproval.APPROVED_FOR_STYLE
            or self.source_type is not StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL
            or self.observed_from is not StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY
            or self.confidence_method is not StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL
        ):
            raise ValueError("style evidence requires explicit owner approval")
        return self


class StyleEvidenceCatalog(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueId
    profile_version: OpaqueId
    versions: StyleEvidenceVersions
    records: tuple[StyleEvidenceRecord, ...] = Field(default=(), max_length=64)
    updated_at: datetime
    source_binding: SourceApprovalState | None = Field(default=None, repr=False)
    usage_scope: EvidenceUsageScope | None = None

    @model_validator(mode="after")
    def require_consistent_catalog(self) -> Self:
        _validate_source_binding(self.user_id, self.source_binding, self.usage_scope, self.records)
        refs = [record.evidence_ref for record in self.records]
        if len(refs) != len(set(refs)):
            raise ValueError("style evidence references must be unique")
        if any(record.user_id != self.user_id for record in self.records):
            raise ValueError("style evidence user scope is inconsistent")
        if any(record.approved_at > self.updated_at for record in self.records):
            raise ValueError("style evidence approval cannot postdate the catalog")
        return self


class StyleEvidenceManifestEntry(_FrozenTimezoneModel):
    """Content-free retrieval metadata for one approved private record."""

    user_id: UserId
    evidence_ref: OpaqueId
    scenario_id: OpaqueId
    confidence: UnitInterval
    approval: StyleEvidenceApproval
    source_type: StyleEvidenceSourceType
    observed_from: StyleEvidenceObservedFrom
    confidence_method: StyleEvidenceConfidenceMethod
    source_version: OpaqueId
    approved_at: datetime
    candidate_index: CandidateIndex | None = Field(default=None, repr=False)


class StyleEvidenceManifest(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueId
    profile_version: OpaqueId
    versions: StyleEvidenceVersions
    entries: tuple[StyleEvidenceManifestEntry, ...] = Field(default=(), max_length=64)
    updated_at: datetime
    source_binding: SourceApprovalState | None = Field(default=None, repr=False)
    usage_scope: EvidenceUsageScope | None = None

    @model_validator(mode="after")
    def require_consistent_manifest(self) -> Self:
        _validate_source_binding(self.user_id, self.source_binding, self.usage_scope, self.entries)
        refs = [entry.evidence_ref for entry in self.entries]
        if len(refs) != len(set(refs)):
            raise ValueError("style evidence manifest references must be unique")
        if any(entry.user_id != self.user_id for entry in self.entries):
            raise ValueError("style evidence manifest user scope is inconsistent")
        if any(entry.approved_at > self.updated_at for entry in self.entries):
            raise ValueError("style evidence approval cannot postdate the manifest")
        return self


class StyleEvidenceIndexEntry(_FrozenTimezoneModel):
    scenario_id: OpaqueId
    evidence_refs: tuple[OpaqueId, ...] = Field(min_length=1, max_length=64)


class StyleEvidenceIndex(_FrozenTimezoneModel):
    user_id: UserId
    evidence_version: OpaqueId
    index_version: OpaqueId
    entries: tuple[StyleEvidenceIndexEntry, ...] = Field(default=(), max_length=64)
    updated_at: datetime


class StyleEvidenceCache(_FrozenTimezoneModel):
    """Private metadata-only cache; approved expression bodies are never duplicated here."""

    user_id: UserId
    evidence_version: OpaqueId
    index_version: OpaqueId
    cache_version: OpaqueId
    route_kind: OpenSetRouteKind | None = None
    scenario_ids: tuple[OpaqueId, ...] = Field(default=(), max_length=4)
    status: StyleEvidenceRetrievalStatus = StyleEvidenceRetrievalStatus.NOT_REQUESTED
    evidence_refs: tuple[OpaqueId, ...] = Field(default=(), max_length=MAX_RETRIEVED_EVIDENCE)
    updated_at: datetime

    @model_validator(mode="after")
    def require_consistent_cache(self) -> Self:
        if len(self.scenario_ids) != len(set(self.scenario_ids)) or len(self.evidence_refs) != len(
            set(self.evidence_refs)
        ):
            raise ValueError("style evidence cache references must be unique")
        if self.status is StyleEvidenceRetrievalStatus.AVAILABLE:
            if self.route_kind is None or not self.evidence_refs:
                raise ValueError("available style evidence cache requires references")
        elif self.evidence_refs:
            raise ValueError("non-available style evidence cache cannot contain references")
        if self.route_kind is None and (
            self.scenario_ids or self.status is not StyleEvidenceRetrievalStatus.NOT_REQUESTED
        ):
            raise ValueError("empty style evidence cache state is inconsistent")
        return self


class StyleEvidenceVersionRecord(_FrozenTimezoneModel):
    user_id: UserId
    authorization_version: OpaqueId
    profile_version: OpaqueId
    versions: StyleEvidenceVersions
    updated_at: datetime


class RetrievedStyleEvidence(_FrozenTimezoneModel):
    user_id: UserId
    evidence_ref: OpaqueId
    scenario_id: OpaqueId
    content: EvidenceContent
    confidence: UnitInterval
    approval: StyleEvidenceApproval
    source_type: StyleEvidenceSourceType
    observed_from: StyleEvidenceObservedFrom
    confidence_method: StyleEvidenceConfidenceMethod
    source_version: OpaqueId
    approved_at: datetime


class StyleEvidenceResult(_FrozenTimezoneModel):
    """Self-verifying result with content-free global evidence and bounded matched content."""

    user_id: UserId
    retrieval_version: OpaqueId
    route_kind: OpenSetRouteKind
    source_route: OpenSetStyleRoute
    status: StyleEvidenceRetrievalStatus
    versions: StyleEvidenceVersions | None = None
    source_manifest: StyleEvidenceManifest | None = None
    matched_sources: tuple[StyleEvidenceRecord, ...] = Field(
        default=(), max_length=MAX_RETRIEVED_EVIDENCE
    )
    matches: tuple[RetrievedStyleEvidence, ...] = Field(
        default=(), max_length=MAX_RETRIEVED_EVIDENCE
    )

    @model_validator(mode="after")
    def require_reproducible_result(self) -> Self:
        if self.retrieval_version != STYLE_EVIDENCE_RETRIEVAL_VERSION:
            raise ValueError("style evidence retrieval version is not registered")
        route = _deep_validated_route(self.source_route)
        if (
            route is None
            or route.user_id != self.user_id
            or route.route_kind is not self.route_kind
        ):
            raise ValueError("style evidence source route is invalid")
        if route.route_kind is OpenSetRouteKind.OPEN_SET:
            if (
                self.status is not StyleEvidenceRetrievalStatus.NOT_REQUESTED
                or self.versions is not None
                or self.source_manifest is not None
                or self.matched_sources
                or self.matches
            ):
                raise ValueError("open-set style evidence retrieval must abstain")
            return self
        if self.status is StyleEvidenceRetrievalStatus.CORRUPT:
            if (
                self.versions is not None
                or self.source_manifest is not None
                or self.matched_sources
                or self.matches
            ):
                raise ValueError("corrupt style evidence result cannot expose partial data")
            return self
        manifest = _deep_validated_manifest(self.source_manifest)
        if manifest is None or not _manifest_matches_route(manifest, route):
            raise ValueError("style evidence manifest is invalid")
        if self.versions != manifest.versions:
            raise ValueError("style evidence versions are inconsistent")
        expected_status, selection = _calculate_retrieval_plan(route, manifest)
        if self.status is not expected_status:
            raise ValueError("style evidence result is not reproducible")
        if expected_status is not StyleEvidenceRetrievalStatus.AVAILABLE:
            if self.matched_sources or self.matches:
                raise ValueError("non-available retrieval cannot expose matched content")
            return self
        expected_matches = _matches_from_sources(selection, self.matched_sources)
        if expected_matches is None or self.matches != expected_matches:
            raise ValueError("style evidence result is not reproducible")
        return self


@runtime_checkable
class StyleEvidenceReadGateway(Protocol):
    def retrieve_evidence(
        self,
        *,
        user_id: str,
        route: OpenSetStyleRoute,
    ) -> StyleEvidenceResult: ...

    def view_approved_evidence(self, user_id: str) -> tuple[StyleEvidenceRecord, ...] | None: ...


@runtime_checkable
class StyleEvidenceManagementGateway(Protocol):
    def publish_approved_evidence(
        self,
        *,
        user_id: str,
        snapshot: UserStyleSnapshot,
        catalog: StyleEvidenceCatalog,
    ) -> None: ...

    def withdraw_evidence(
        self,
        *,
        user_id: str,
        evidence_ref: str,
        versions: StyleEvidenceVersions,
        updated_at: datetime,
    ) -> None: ...

    def delete_user_evidence(self, user_id: str) -> None: ...


class StyleEvidenceGateway(
    StyleEvidenceReadGateway,
    StyleEvidenceManagementGateway,
    Protocol,
):
    pass


def _deep_validated_route(route: object) -> OpenSetStyleRoute | None:
    if type(route) is not OpenSetStyleRoute:
        return None
    with suppress(Exception):
        return OpenSetStyleRoute.model_validate(route.model_dump(mode="python", warnings=False))
    return None


def _deep_validated_catalog(catalog: object) -> StyleEvidenceCatalog | None:
    if type(catalog) is not StyleEvidenceCatalog:
        return None
    with suppress(Exception):
        return StyleEvidenceCatalog.model_validate(
            catalog.model_dump(mode="python", warnings=False)
        )
    return None


def _deep_validated_manifest(manifest: object) -> StyleEvidenceManifest | None:
    if type(manifest) is not StyleEvidenceManifest:
        return None
    with suppress(Exception):
        return StyleEvidenceManifest.model_validate(
            manifest.model_dump(mode="python", warnings=False)
        )
    return None


def _catalog_matches_route(catalog: StyleEvidenceCatalog, route: OpenSetStyleRoute) -> bool:
    snapshot = route.snapshot
    return (
        catalog.user_id == route.user_id
        and catalog.authorization_version == snapshot.consent.authorization_version
        and catalog.profile_version == snapshot.profile.profile_version
        and snapshot.consent.status is UserStyleProfileStatus.AUTHORIZED
        and (
            catalog.source_binding is None
            or catalog.source_binding == snapshot.consent.source_binding
        )
    )


def _manifest_matches_route(manifest: StyleEvidenceManifest, route: OpenSetStyleRoute) -> bool:
    snapshot = route.snapshot
    return (
        manifest.user_id == route.user_id
        and manifest.authorization_version == snapshot.consent.authorization_version
        and manifest.profile_version == snapshot.profile.profile_version
        and snapshot.consent.status is UserStyleProfileStatus.AUTHORIZED
        and (
            manifest.source_binding is None
            or manifest.source_binding == snapshot.consent.source_binding
        )
    )


def _manifest_entry(record: StyleEvidenceRecord) -> StyleEvidenceManifestEntry:
    return StyleEvidenceManifestEntry(
        user_id=record.user_id,
        evidence_ref=record.evidence_ref,
        scenario_id=record.scenario_id,
        confidence=record.confidence,
        approval=record.approval,
        source_type=record.source_type,
        observed_from=record.observed_from,
        confidence_method=record.confidence_method,
        source_version=record.source_version,
        approved_at=record.approved_at,
        candidate_index=record.candidate_index,
    )


def _build_manifest(catalog: StyleEvidenceCatalog) -> StyleEvidenceManifest:
    return StyleEvidenceManifest(
        user_id=catalog.user_id,
        authorization_version=catalog.authorization_version,
        profile_version=catalog.profile_version,
        versions=_copied_versions(catalog.versions),
        entries=tuple(_manifest_entry(record) for record in catalog.records),
        updated_at=catalog.updated_at,
        source_binding=catalog.source_binding,
        usage_scope=catalog.usage_scope,
    )


def _copied_versions(versions: StyleEvidenceVersions) -> StyleEvidenceVersions:
    return StyleEvidenceVersions.model_validate(versions.model_dump(mode="python", warnings=False))


def _calculate_retrieval_plan(
    route: OpenSetStyleRoute,
    manifest: StyleEvidenceManifest,
) -> tuple[
    StyleEvidenceRetrievalStatus,
    tuple[tuple[StyleEvidenceManifestEntry, float], ...],
]:
    entries_by_scenario: dict[str, list[StyleEvidenceManifestEntry]] = defaultdict(list)
    for entry in manifest.entries:
        entries_by_scenario[entry.scenario_id].append(entry)
    relevant: list[tuple[int, float, StyleEvidenceManifestEntry]] = []
    for candidate_index, candidate in enumerate(route.candidates):
        confidence_cap = round(0.5 + (0.5 * candidate.coverage), 6)
        for entry in entries_by_scenario.get(candidate.scenario_id, ()):
            confidence = round(min(entry.confidence, confidence_cap), 6)
            relevant.append((candidate_index, confidence, entry))
    if not relevant:
        return StyleEvidenceRetrievalStatus.NO_MATCH, ()
    accepted = [item for item in relevant if item[1] >= MIN_STYLE_EVIDENCE_CONFIDENCE]
    if not accepted:
        return StyleEvidenceRetrievalStatus.LOW_CONFIDENCE, ()
    accepted.sort(key=lambda item: (item[0], -item[1], item[2].evidence_ref))
    selection = tuple(
        (entry, confidence) for _, confidence, entry in accepted[:MAX_RETRIEVED_EVIDENCE]
    )
    return StyleEvidenceRetrievalStatus.AVAILABLE, selection


def _matches_from_sources(
    selection: tuple[tuple[StyleEvidenceManifestEntry, float], ...],
    sources: tuple[StyleEvidenceRecord, ...],
) -> tuple[RetrievedStyleEvidence, ...] | None:
    if len(selection) != len(sources):
        return None
    matches: list[RetrievedStyleEvidence] = []
    for (entry, confidence), record in zip(selection, sources, strict=True):
        if _manifest_entry(record) != entry:
            return None
        matches.append(
            RetrievedStyleEvidence(
                user_id=record.user_id,
                evidence_ref=record.evidence_ref,
                scenario_id=record.scenario_id,
                content=record.content,
                confidence=confidence,
                approval=record.approval,
                source_type=record.source_type,
                observed_from=record.observed_from,
                confidence_method=record.confidence_method,
                source_version=record.source_version,
                approved_at=record.approved_at,
            )
        )
    return tuple(matches)


def retrieve_style_evidence(
    *,
    user_id: str,
    trusted_user_id: str,
    route: OpenSetStyleRoute,
    catalog: StyleEvidenceCatalog | None,
) -> StyleEvidenceResult:
    """Retrieve only within Phase 3 candidates; open-set routes never inspect a catalog."""
    _require_owner(user_id, trusted_user_id)
    validated_route = _deep_validated_route(route)
    if validated_route is None or validated_route.user_id != user_id:
        raise StyleValidationError(
            "invalid_style_evidence_query", "style evidence query is invalid"
        )
    if validated_route.route_kind is OpenSetRouteKind.OPEN_SET:
        return StyleEvidenceResult(
            user_id=user_id,
            retrieval_version=STYLE_EVIDENCE_RETRIEVAL_VERSION,
            route_kind=validated_route.route_kind,
            source_route=validated_route,
            status=StyleEvidenceRetrievalStatus.NOT_REQUESTED,
        )
    validated_catalog = _deep_validated_catalog(catalog)
    if validated_catalog is None or not _catalog_matches_route(validated_catalog, validated_route):
        raise StyleValidationError(
            "invalid_style_evidence_catalog", "style evidence catalog is invalid"
        )
    manifest = _build_manifest(validated_catalog)
    status, selection = _calculate_retrieval_plan(validated_route, manifest)
    records_by_ref = {record.evidence_ref: record for record in validated_catalog.records}
    matched_sources = tuple(records_by_ref[entry.evidence_ref] for entry, _ in selection)
    matches = _matches_from_sources(selection, matched_sources)
    if matches is None:
        raise StyleValidationError(
            "invalid_style_evidence_catalog", "style evidence catalog is invalid"
        )
    return StyleEvidenceResult(
        user_id=user_id,
        retrieval_version=STYLE_EVIDENCE_RETRIEVAL_VERSION,
        route_kind=validated_route.route_kind,
        source_route=validated_route,
        status=status,
        versions=_copied_versions(manifest.versions),
        source_manifest=manifest,
        matched_sources=matched_sources,
        matches=matches,
    )


def _require_owner(user_id: str, trusted_user_id: str) -> None:
    if (
        type(trusted_user_id) is not str
        or not trusted_user_id
        or trusted_user_id != trusted_user_id.strip()
        or len(trusted_user_id) > 120
        or type(user_id) is not str
        or user_id != trusted_user_id
    ):
        raise StyleValidationError("untrusted_style_user", "style user is not trusted")


def _build_index(catalog: StyleEvidenceCatalog) -> StyleEvidenceIndex:
    refs_by_scenario: dict[str, list[str]] = defaultdict(list)
    for record in catalog.records:
        refs_by_scenario[record.scenario_id].append(record.evidence_ref)
    entries = tuple(
        StyleEvidenceIndexEntry(
            scenario_id=scenario_id,
            evidence_refs=tuple(sorted(refs)),
        )
        for scenario_id, refs in sorted(refs_by_scenario.items())
    )
    return StyleEvidenceIndex(
        user_id=catalog.user_id,
        evidence_version=catalog.versions.evidence_version,
        index_version=catalog.versions.index_version,
        entries=entries,
        updated_at=catalog.updated_at,
    )


def _empty_cache(catalog: StyleEvidenceCatalog) -> StyleEvidenceCache:
    return StyleEvidenceCache(
        user_id=catalog.user_id,
        evidence_version=catalog.versions.evidence_version,
        index_version=catalog.versions.index_version,
        cache_version=catalog.versions.cache_version,
        updated_at=catalog.updated_at,
    )


def _version_record(catalog: StyleEvidenceCatalog) -> StyleEvidenceVersionRecord:
    return StyleEvidenceVersionRecord(
        user_id=catalog.user_id,
        authorization_version=catalog.authorization_version,
        profile_version=catalog.profile_version,
        versions=catalog.versions,
        updated_at=catalog.updated_at,
    )


class SingleTrustedUserStyleEvidenceStore:
    """Fixed-path private evidence store for the single currently supported owner."""

    def __init__(
        self,
        *,
        trusted_user_id: str,
        private_data_dir: Path,
        evidence_source: Path,
        index_source: Path,
        cache_source: Path,
        version_source: Path,
        strict_source_binding: bool = False,
        source_gateway: SourceApprovalGateway | None = None,
    ) -> None:
        _require_owner(trusted_user_id, trusted_user_id)
        if type(strict_source_binding) is not bool or (
            strict_source_binding and source_gateway is None
        ):
            raise ValueError("strict style evidence requires a source gateway")
        self._strict_source_binding = strict_source_binding
        self._source_gateway = source_gateway
        self._trusted_user_id = trusted_user_id
        self._private_data_dir = private_data_dir
        self._evidence_source = evidence_source
        self._index_source = index_source
        self._cache_source = cache_source
        self._version_source = version_source
        resolved = self._validate_fixed_paths(self.artifact_paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("style evidence artifact paths must be distinct")

    @property
    def evidence_source(self) -> Path:
        return self._evidence_source

    @property
    def index_source(self) -> Path:
        return self._index_source

    @property
    def cache_source(self) -> Path:
        return self._cache_source

    @property
    def version_source(self) -> Path:
        return self._version_source

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return (
            self._evidence_source,
            self._index_source,
            self._cache_source,
            self._version_source,
        )

    def _validate_fixed_paths(self, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        try:
            root = self._private_data_dir.resolve(strict=True)
            resolved = tuple(path.resolve(strict=False) for path in paths)
        except OSError:
            raise ValueError("style evidence paths could not be validated") from None
        if any(not path.is_relative_to(root) for path in resolved):
            raise ValueError("style evidence paths must stay inside private data")
        return resolved

    def _require_owner(self, user_id: str) -> None:
        _require_owner(user_id, self._trusted_user_id)

    def _current_source(self) -> SourceApprovalState | None:
        try:
            candidate = self._source_gateway.current_state(self._trusted_user_id)
            if type(candidate) is not SourceApprovalState:
                return None
            state = SourceApprovalState.model_validate(
                candidate.model_dump(mode="python", warnings=False)
            )
            if state.owner_id != self._trusted_user_id or not state.approved_indices:
                return None
            return state
        except Exception:
            return None

    def _source_matches(
        self, catalog: StyleEvidenceCatalog, snapshot: UserStyleSnapshot | None = None
    ) -> bool:
        if not self._strict_source_binding:
            return True
        current = self._current_source()
        return (
            current is not None
            and catalog.source_binding == current
            and catalog.usage_scope == "approved_style_expressions_local_persistence"
            and (
                snapshot is None
                or (
                    snapshot.consent.scope == "non_text_style_features_local_persistence"
                    and snapshot.consent.source_binding == current
                )
            )
        )

    def _publish_catalog(self, catalog: StyleEvidenceCatalog) -> None:
        write_private_models_atomically(
            [
                (catalog, self._evidence_source),
                (_build_index(catalog), self._index_source),
                (_empty_cache(catalog), self._cache_source),
                (_version_record(catalog), self._version_source),
            ],
            self._private_data_dir,
        )

    def publish_approved_evidence(
        self,
        *,
        user_id: str,
        snapshot: UserStyleSnapshot,
        catalog: StyleEvidenceCatalog,
    ) -> None:
        self._require_owner(user_id)
        validated_snapshot: UserStyleSnapshot | None = None
        validated_catalog = _deep_validated_catalog(catalog)
        with suppress(Exception):
            if type(snapshot) is UserStyleSnapshot:
                validated_snapshot = UserStyleSnapshot.model_validate(
                    snapshot.model_dump(mode="python", warnings=False)
                )
        if (
            validated_snapshot is None
            or validated_catalog is None
            or validated_snapshot.consent.status is not UserStyleProfileStatus.AUTHORIZED
            or validated_snapshot.consent.user_id != user_id
            or validated_snapshot.profile.user_id != user_id
            or validated_catalog.user_id != user_id
            or validated_catalog.authorization_version
            != validated_snapshot.consent.authorization_version
            or validated_catalog.profile_version != validated_snapshot.profile.profile_version
            or not self._source_matches(validated_catalog, validated_snapshot)
        ):
            raise StyleValidationError(
                "invalid_style_evidence_catalog", "style evidence catalog is invalid"
            )
        self._publish_catalog(validated_catalog)
        if not self._source_matches(validated_catalog, validated_snapshot):
            raise StyleValidationError("style_source_unavailable", "style source is unavailable")

    def _load_state(
        self,
    ) -> (
        tuple[
            StyleEvidenceCatalog,
            StyleEvidenceIndex,
            StyleEvidenceCache,
            StyleEvidenceVersionRecord,
        ]
        | None
    ):
        if self._strict_source_binding and self._current_source() is None:
            return None
        try:
            catalog = read_private_model(
                StyleEvidenceCatalog, self._evidence_source, self._private_data_dir
            )
            index = read_private_model(
                StyleEvidenceIndex, self._index_source, self._private_data_dir
            )
            cache = read_private_model(
                StyleEvidenceCache, self._cache_source, self._private_data_dir
            )
            version = read_private_model(
                StyleEvidenceVersionRecord, self._version_source, self._private_data_dir
            )
        except StyleValidationError:
            return None
        expected_index = _build_index(catalog)
        expected_version = _version_record(catalog)
        if index != expected_index or version != expected_version:
            return None
        if not self._source_matches(catalog):
            return None
        if (
            cache.user_id != catalog.user_id
            or cache.evidence_version != catalog.versions.evidence_version
            or cache.index_version != catalog.versions.index_version
            or cache.cache_version != catalog.versions.cache_version
            or not set(cache.evidence_refs).issubset(
                {record.evidence_ref for record in catalog.records}
            )
        ):
            return None
        return catalog, index, cache, version

    def retrieve_evidence(
        self,
        *,
        user_id: str,
        route: OpenSetStyleRoute,
    ) -> StyleEvidenceResult:
        self._require_owner(user_id)
        validated_route = _deep_validated_route(route)
        if validated_route is None or validated_route.user_id != user_id:
            raise StyleValidationError(
                "invalid_style_evidence_query", "style evidence query is invalid"
            )
        if validated_route.route_kind is OpenSetRouteKind.OPEN_SET:
            return retrieve_style_evidence(
                user_id=user_id,
                trusted_user_id=self._trusted_user_id,
                route=validated_route,
                catalog=None,
            )
        state = self._load_state()
        if state is None:
            return StyleEvidenceResult(
                user_id=user_id,
                retrieval_version=STYLE_EVIDENCE_RETRIEVAL_VERSION,
                route_kind=validated_route.route_kind,
                source_route=validated_route,
                status=StyleEvidenceRetrievalStatus.CORRUPT,
            )
        catalog = state[0]
        if not _catalog_matches_route(catalog, validated_route) or not self._source_matches(
            catalog, validated_route.snapshot
        ):
            return StyleEvidenceResult(
                user_id=user_id,
                retrieval_version=STYLE_EVIDENCE_RETRIEVAL_VERSION,
                route_kind=validated_route.route_kind,
                source_route=validated_route,
                status=StyleEvidenceRetrievalStatus.CORRUPT,
            )
        result = retrieve_style_evidence(
            user_id=user_id,
            trusted_user_id=self._trusted_user_id,
            route=validated_route,
            catalog=catalog,
        )
        cache = StyleEvidenceCache(
            user_id=user_id,
            evidence_version=catalog.versions.evidence_version,
            index_version=catalog.versions.index_version,
            cache_version=catalog.versions.cache_version,
            route_kind=validated_route.route_kind,
            scenario_ids=tuple(candidate.scenario_id for candidate in validated_route.candidates),
            status=result.status,
            evidence_refs=tuple(item.evidence_ref for item in result.matches),
            updated_at=validated_route.context_delta.created_at,
        )
        try:
            write_private_models_atomically(
                [(cache, self._cache_source)],
                self._private_data_dir,
            )
        except StyleValidationError:
            return StyleEvidenceResult(
                user_id=user_id,
                retrieval_version=STYLE_EVIDENCE_RETRIEVAL_VERSION,
                route_kind=validated_route.route_kind,
                source_route=validated_route,
                status=StyleEvidenceRetrievalStatus.CORRUPT,
            )
        if not self._source_matches(catalog, validated_route.snapshot):
            return StyleEvidenceResult(
                user_id=user_id,
                retrieval_version=STYLE_EVIDENCE_RETRIEVAL_VERSION,
                route_kind=validated_route.route_kind,
                source_route=validated_route,
                status=StyleEvidenceRetrievalStatus.CORRUPT,
            )
        return result

    def view_approved_evidence(self, user_id: str) -> tuple[StyleEvidenceRecord, ...] | None:
        self._require_owner(user_id)
        state = self._load_state()
        if state is None or state[0].user_id != user_id or not self._source_matches(state[0]):
            return None
        return state[0].records

    def withdraw_evidence(
        self,
        *,
        user_id: str,
        evidence_ref: str,
        versions: StyleEvidenceVersions,
        updated_at: datetime,
    ) -> None:
        self._require_owner(user_id)
        state = self._load_state()
        if state is None:
            raise StyleValidationError(
                "style_evidence_unavailable", "style evidence is unavailable"
            )
        catalog = state[0]
        if (
            self._strict_source_binding
            and len(catalog.records) == 1
            and type(evidence_ref) is str
            and catalog.records[0].evidence_ref == evidence_ref
        ):
            try:
                if type(versions) is not StyleEvidenceVersions:
                    raise ValueError("invalid versions")
                StyleEvidenceVersions.model_validate(
                    versions.model_dump(mode="python", warnings=False)
                )
                if (
                    type(updated_at) is not datetime
                    or updated_at.tzinfo is None
                    or updated_at.utcoffset() is None
                ):
                    raise ValueError("invalid time")
            except Exception:
                raise StyleValidationError(
                    "invalid_style_evidence_withdrawal", "style evidence withdrawal is invalid"
                ) from None
            self.delete_user_evidence(user_id)
            return
        if type(evidence_ref) is not str or not any(
            record.evidence_ref == evidence_ref for record in catalog.records
        ):
            raise StyleValidationError(
                "style_evidence_unavailable", "style evidence is unavailable"
            )
        replacement: StyleEvidenceCatalog | None = None
        with suppress(Exception):
            replacement = StyleEvidenceCatalog(
                user_id=user_id,
                authorization_version=catalog.authorization_version,
                profile_version=catalog.profile_version,
                versions=versions,
                records=tuple(
                    record for record in catalog.records if record.evidence_ref != evidence_ref
                ),
                updated_at=updated_at,
                source_binding=catalog.source_binding,
                usage_scope=catalog.usage_scope,
            )
        if replacement is None:
            raise StyleValidationError(
                "invalid_style_evidence_withdrawal", "style evidence withdrawal is invalid"
            ) from None
        self._publish_catalog(replacement)
        if not self._source_matches(replacement):
            raise StyleValidationError("style_source_unavailable", "style source is unavailable")

    def delete_user_evidence(self, user_id: str) -> None:
        self._require_owner(user_id)
        try:
            self._validate_fixed_paths(self.artifact_paths)
        except ValueError:
            raise StyleValidationError(
                "style_evidence_delete_failed", "style evidence artifacts could not be deleted"
            ) from None
        failed = False
        for path in self.artifact_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failed = True
        if failed:
            raise StyleValidationError(
                "style_evidence_delete_failed", "style evidence artifacts could not be deleted"
            )
