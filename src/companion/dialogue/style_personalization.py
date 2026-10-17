"""Request-local production wiring for the opted-in Style framework."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from companion.dialogue.context import PreparedDialogueInput
from companion.dialogue.personalization import PersonalizationSpec, compile_personalization_spec
from companion.style import (
    FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
    OpenSetRouteKind,
    StyleEvidenceResult,
    StyleEvidenceRetrievalStatus,
    UserStyleSnapshotReadResult,
    UserStyleSnapshotReadStatus,
    build_formal_style_scenario_catalog,
    build_pragmatic_context_delta,
    route_open_set_style,
)
from companion.style.evidence_retrieval import StyleEvidenceReadGateway


class ProfileSnapshotReader(Protocol):
    def read_snapshot_state(self, user_id: str) -> UserStyleSnapshotReadResult: ...


class ProductionStylePersonalization:
    """Keep all source material within one request and fail closed on invalid input."""

    def __init__(
        self,
        *,
        trusted_user_id: str,
        profile_gateway: ProfileSnapshotReader,
        evidence_gateway: StyleEvidenceReadGateway,
    ) -> None:
        if type(trusted_user_id) is not str or not 1 <= len(trusted_user_id) <= 120:
            raise ValueError("invalid trusted style user")
        self._trusted_user_id = trusted_user_id
        self._profile = profile_gateway
        self._evidence = evidence_gateway

    def compile(
        self,
        *,
        user_id: str,
        prepared: PreparedDialogueInput,
        idle_seconds: int | None,
        observed_at: datetime,
    ) -> PersonalizationSpec | None:
        if type(user_id) is not str or user_id != self._trusted_user_id:
            raise ValueError("untrusted style user")
        state_candidate = self._profile.read_snapshot_state(user_id)
        if type(state_candidate) is not UserStyleSnapshotReadResult:
            raise ValueError("invalid style snapshot state")
        state = UserStyleSnapshotReadResult.model_validate(
            state_candidate.model_dump(mode="python", warnings=False)
        )
        if state.status is UserStyleSnapshotReadStatus.UNAVAILABLE:
            return None
        if state.status is not UserStyleSnapshotReadStatus.AUTHORIZED or state.snapshot is None:
            raise ValueError("style snapshot unavailable")
        delta = build_pragmatic_context_delta(
            user_id=user_id,
            trusted_user_id=self._trusted_user_id,
            text=prepared.text,
            recent_context=prepared.recent_context,
            idle_seconds=idle_seconds,
            observed_at=observed_at,
        )
        catalog = build_formal_style_scenario_catalog(
            user_id=user_id,
            trusted_user_id=self._trusted_user_id,
            snapshot=state.snapshot,
            expected_catalog_version=FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
        )
        route = route_open_set_style(
            user_id=user_id,
            trusted_user_id=self._trusted_user_id,
            snapshot=state.snapshot,
            context_delta=delta,
            scenario_catalog=catalog,
        )
        retrieval = self._evidence.retrieve_evidence(user_id=user_id, route=route)
        return compile_personalization_spec(
            user_id=user_id,
            route=route,
            retrieval_result=retrieval,
        )

    def verify_current(self, *, user_id: str, spec: PersonalizationSpec) -> None:
        """Recheck the compiled source and evidence just before one styled send."""
        if (
            type(user_id) is not str
            or user_id != self._trusted_user_id
            or type(spec) is not PersonalizationSpec
        ):
            raise ValueError("untrusted style user")
        state_candidate = self._profile.read_snapshot_state(user_id)
        if type(state_candidate) is not UserStyleSnapshotReadResult:
            raise ValueError("invalid style snapshot state")
        state = UserStyleSnapshotReadResult.model_validate(
            state_candidate.model_dump(mode="python", warnings=False)
        )
        route = spec.source_route
        if (
            state.status is not UserStyleSnapshotReadStatus.AUTHORIZED
            or state.snapshot != route.snapshot
        ):
            raise ValueError("style snapshot changed")
        if route.route_kind is OpenSetRouteKind.OPEN_SET:
            if spec.source_retrieval is not None and (
                spec.source_retrieval.status is not StyleEvidenceRetrievalStatus.NOT_REQUESTED
            ):
                raise ValueError("style evidence changed")
            return
        retrieval_candidate = self._evidence.retrieve_evidence(user_id=user_id, route=route)
        if type(retrieval_candidate) is not StyleEvidenceResult:
            raise ValueError("invalid style evidence result")
        retrieval = StyleEvidenceResult.model_validate(
            retrieval_candidate.model_dump(mode="python", warnings=False)
        )
        if (
            retrieval.status is StyleEvidenceRetrievalStatus.CORRUPT
            or retrieval != spec.source_retrieval
        ):
            raise ValueError("style evidence changed")
