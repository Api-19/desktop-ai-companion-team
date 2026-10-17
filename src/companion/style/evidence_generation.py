"""Pure Phase 5 generation for trusted Admin's explicit local-purpose submission.

Inputs and output declarations are not authentication credentials. The caller
must recheck source and authorization before/after submission; cloud use needs
separate explicit authorization. This factory does no I/O or consent granting.
"""

import hashlib
import json
import secrets
from datetime import datetime

from companion.privacy.candidate_models import SafeCandidatePackage
from companion.style.errors import StyleValidationError
from companion.style.evidence_retrieval import (
    StyleEvidenceApproval,
    StyleEvidenceCatalog,
    StyleEvidenceConfidenceMethod,
    StyleEvidenceObservedFrom,
    StyleEvidenceRecord,
    StyleEvidenceSourceType,
    StyleEvidenceVersions,
)
from companion.style.open_set_routing import (
    FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
    build_formal_style_scenario_catalog,
)
from companion.style.user_profile import (
    SourceApprovalState,
    UserStyleProfileStatus,
    UserStyleSnapshot,
)


def generate_style_evidence_catalog(
    *,
    user_id: str,
    package: SafeCandidatePackage,
    snapshot: UserStyleSnapshot,
    source_state: SourceApprovalState,
    selections: tuple[tuple[int, str], ...],
    observed_at: datetime,
) -> StyleEvidenceCatalog:
    """Copy selected approved texts verbatim; 1.0 means explicit approval only.

    Scenario choices are explicit owner choices, not inferred semantic matches.
    ``snapshot`` must come from a strict AUTHORIZED read by the trusted caller.
    """
    try:
        if (
            type(user_id) is not str
            or not user_id
            or user_id != user_id.strip()
            or type(snapshot) is not UserStyleSnapshot
            or type(source_state) is not SourceApprovalState
            or type(package) is not SafeCandidatePackage
        ):
            raise ValueError("invalid inputs")
        verified_snapshot = UserStyleSnapshot.model_validate(
            snapshot.model_dump(mode="python", warnings=False)
        )
        state = SourceApprovalState.model_validate(
            source_state.model_dump(mode="python", warnings=False)
        )
        consent = verified_snapshot.consent
        if (
            consent.user_id != user_id
            or state.owner_id != user_id
            or consent.status is not UserStyleProfileStatus.AUTHORIZED
            or consent.scope != "non_text_style_features_local_persistence"
            or consent.source_binding != state
            or not state.approved_indices
            or type(observed_at) is not datetime
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or observed_at < consent.updated_at
            or type(selections) is not tuple
            or not 1 <= len(selections) <= 64
        ):
            raise ValueError("invalid authorization")
        verified_package = SafeCandidatePackage.model_validate(
            package.model_dump(mode="json", warnings=False)
        )
        digest = hashlib.sha256(
            json.dumps(
                verified_package.model_dump(mode="json", warnings=False),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if digest != state.package_digest or any(
            index >= len(verified_package.candidates) for index in state.approved_indices
        ):
            raise ValueError("invalid source")
        scenarios = {
            item.scenario_id
            for item in build_formal_style_scenario_catalog(
                user_id=user_id,
                trusted_user_id=user_id,
                snapshot=verified_snapshot,
                expected_catalog_version=FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
            )
        }
        version = "evidence-" + secrets.token_hex(24)
        records = []
        selected_indices = set()
        for selection in selections:
            if type(selection) is not tuple or len(selection) != 2:
                raise ValueError("invalid selection")
            index, scenario = selection
            if (
                type(index) is not int
                or index not in state.approved_indices
                or index in selected_indices
                or type(scenario) is not str
                or scenario not in scenarios
            ):
                raise ValueError("invalid selection")
            text = verified_package.candidates[index].text
            if not text.strip() or len(text) > 160:
                raise ValueError("invalid text")
            selected_indices.add(index)
            records.append(
                StyleEvidenceRecord(
                    user_id=user_id,
                    evidence_ref="ref-" + secrets.token_hex(24),
                    scenario_id=scenario,
                    content=text,
                    candidate_index=index,
                    confidence=1.0,
                    approval=StyleEvidenceApproval.APPROVED_FOR_STYLE,
                    source_type=StyleEvidenceSourceType.EXPLICIT_OWNER_APPROVAL,
                    observed_from=StyleEvidenceObservedFrom.OWNER_MESSAGES_ONLY,
                    confidence_method=StyleEvidenceConfidenceMethod.EXPLICIT_OWNER_APPROVAL,
                    source_version=version,
                    approved_at=observed_at,
                )
            )
        return StyleEvidenceCatalog(
            user_id=user_id,
            authorization_version=consent.authorization_version,
            profile_version=verified_snapshot.profile.profile_version,
            versions=StyleEvidenceVersions(
                evidence_version=version,
                index_version="index-" + secrets.token_hex(24),
                cache_version="cache-" + secrets.token_hex(24),
            ),
            records=tuple(records),
            updated_at=observed_at,
            source_binding=state,
            usage_scope="approved_style_expressions_local_persistence",
        )
    except Exception:
        raise StyleValidationError(
            "style_evidence_generation_failed", "style evidence could not be generated"
        ) from None
