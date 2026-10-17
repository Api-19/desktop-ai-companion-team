"""Deterministic, non-text profile generation for a trusted Admin caller.

Inputs are not authentication credentials. The caller must revalidate the trusted
source before persistence through a strict profile store. No I/O occurs here.
"""

import hashlib
import json
import secrets
from datetime import datetime
from math import ceil
from statistics import median

from pydantic import BaseModel, ConfigDict

from companion.privacy.candidate_models import SafeCandidatePackage
from companion.style.errors import StyleValidationError
from companion.style.user_profile import (
    SourceApprovalState,
    StyleConfidenceMethod,
    StyleFeatureKind,
    StyleFeatureProvenance,
    StyleFeatureSourceType,
    StyleObservedFrom,
    UserStyleDerivedArtifacts,
    UserStyleEvidenceLedger,
    UserStyleFeature,
    UserStyleProfile,
    UserStyleProfileConsent,
    UserStyleProfileStatus,
)


class UserStyleProfileBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: UserStyleProfile
    evidence: UserStyleEvidenceLedger
    derived: UserStyleDerivedArtifacts


def generate_user_style_profile(
    *,
    user_id: str,
    package: SafeCandidatePackage,
    consent: UserStyleProfileConsent | None,
    source_state: SourceApprovalState,
    observed_at: datetime,
) -> UserStyleProfileBundle:
    """Generate only three length features; coverage confidence is not calibrated.

    Repeated approved candidates count individually. Python Unicode codepoints,
    excluding every character for which ``isspace()`` is true, define length.
    """
    try:
        if (
            type(user_id) is not str
            or type(consent) is not UserStyleProfileConsent
            or type(source_state) is not SourceApprovalState
            or consent.user_id != user_id
            or source_state.owner_id != user_id
        ):
            raise ValueError("invalid authorization")
        verified_consent = UserStyleProfileConsent.model_validate(consent.model_dump(mode="python"))
        state = SourceApprovalState.model_validate(source_state.model_dump(mode="python"))
        if (
            verified_consent.status is not UserStyleProfileStatus.AUTHORIZED
            or verified_consent.scope != "non_text_style_features_local_persistence"
            or verified_consent.source_binding != state
            or len(state.approved_indices) < 20
            or type(observed_at) is not datetime
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or observed_at < verified_consent.updated_at
            or type(package) is not SafeCandidatePackage
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
        lengths = sorted(
            len(
                "".join(
                    char for char in verified_package.candidates[index].text if not char.isspace()
                )
            )
            for index in state.approved_indices
        )
        if any(length < 1 or length > 2000 for length in lengths):
            raise ValueError("invalid lengths")
        count = len(lengths)
        version = "profile-" + secrets.token_hex(24)
        confidence = min(count / 100, 1.0)
        values = (
            (StyleFeatureKind.MESSAGE_LENGTH_Q25, float(lengths[ceil(count * 0.25) - 1])),
            (StyleFeatureKind.MESSAGE_LENGTH_MEDIAN, float(median(lengths))),
            (StyleFeatureKind.MESSAGE_LENGTH_Q75, float(lengths[ceil(count * 0.75) - 1])),
        )
        provenance = StyleFeatureProvenance(
            source_type=StyleFeatureSourceType.PRIVACY_APPROVED_SAFE_CANDIDATES,
            evidence_count=count,
            observed_from=StyleObservedFrom.OWNER_MESSAGES_ONLY,
            source_version=version,
            updated_at=observed_at,
            confidence_method=StyleConfidenceMethod.DETERMINISTIC_STATISTIC,
        )
        identity = {
            "user_id": user_id,
            "profile_version": version,
            "authorization_version": verified_consent.authorization_version,
        }
        return UserStyleProfileBundle(
            profile=UserStyleProfile(
                **identity,
                features=tuple(
                    UserStyleFeature(
                        kind=kind, value=value, confidence=confidence, provenance=provenance
                    )
                    for kind, value in values
                ),
                created_at=observed_at,
                updated_at=observed_at,
            ),
            evidence=UserStyleEvidenceLedger(
                **identity,
                feature_evidence={kind: count for kind, _ in values},
                updated_at=observed_at,
            ),
            derived=UserStyleDerivedArtifacts(
                **identity,
                statistics_version=version,
                index_version=version,
                cache_version=version,
                updated_at=observed_at,
            ),
        )
    except Exception:
        raise StyleValidationError(
            "style_profile_generation_failed", "style profile could not be generated"
        ) from None
