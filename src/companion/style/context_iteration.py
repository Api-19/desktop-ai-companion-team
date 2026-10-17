from companion.privacy import CandidateCategory
from companion.style.errors import StyleValidationError
from companion.style.evaluation import EVALUATED_CATEGORIES
from companion.style.iteration_models import (
    ContextHoldout,
    ContextSummaryResult,
    ContextSummarySkipReason,
    ContextWindowReference,
    LocalContextSummary,
)


def _build_local_context_summary(
    reference: ContextWindowReference,
    category: CandidateCategory,
    safe_summary: str,
    *,
    holdout: ContextHoldout,
    privacy_approved: bool,
    speaker_mapping_confirmed: bool,
    contains_sensitive_context: bool,
    contains_identity_placeholder: bool,
) -> ContextSummaryResult:
    if reference.window_token not in holdout.holdout_window_tokens:
        raise StyleValidationError(
            "invalid_context_summary",
            "context summary reference must belong to the holdout partition",
        )
    reason = None
    if not speaker_mapping_confirmed:
        reason = ContextSummarySkipReason.SPEAKER_MAPPING_UNCONFIRMED
    elif not privacy_approved:
        reason = ContextSummarySkipReason.PRIVACY_NOT_APPROVED
    elif contains_sensitive_context:
        reason = ContextSummarySkipReason.SENSITIVE_CONTEXT
    elif contains_identity_placeholder:
        reason = ContextSummarySkipReason.IDENTITY_PLACEHOLDER
    elif category not in EVALUATED_CATEGORIES:
        reason = ContextSummarySkipReason.UNSUPPORTED_CATEGORY
    if reason is not None:
        return ContextSummaryResult(
            window_token=reference.window_token,
            summary=None,
            skipped_reason=reason,
        )
    return ContextSummaryResult(
        window_token=reference.window_token,
        summary=LocalContextSummary(
            window_token=reference.window_token,
            category=category,
            prompt=safe_summary,
            privacy_approved=True,
            raw_text_retained=False,
            partner_text_retained=False,
        ),
        skipped_reason=None,
    )


def build_local_context_summary(
    reference: ContextWindowReference,
    category: CandidateCategory,
    safe_summary: str,
    *,
    holdout: ContextHoldout,
    privacy_approved: bool,
    speaker_mapping_confirmed: bool,
    contains_sensitive_context: bool,
    contains_identity_placeholder: bool,
) -> ContextSummaryResult:
    try:
        if any(
            type(value) is not bool
            for value in (
                privacy_approved,
                speaker_mapping_confirmed,
                contains_sensitive_context,
                contains_identity_placeholder,
            )
        ):
            raise ValueError("context flags must be bool")
        reference = ContextWindowReference.model_validate(
            reference.model_dump(mode="json", warnings=False)
        )
        holdout = ContextHoldout.model_validate(holdout.model_dump(mode="json", warnings=False))
        return _build_local_context_summary(
            reference,
            category,
            safe_summary,
            holdout=holdout,
            privacy_approved=privacy_approved,
            speaker_mapping_confirmed=speaker_mapping_confirmed,
            contains_sensitive_context=contains_sensitive_context,
            contains_identity_placeholder=contains_identity_placeholder,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = ("invalid_context_summary", "context summary input is invalid")
    del (
        reference,
        category,
        safe_summary,
        holdout,
        privacy_approved,
        speaker_mapping_confirmed,
        contains_sensitive_context,
        contains_identity_placeholder,
    )
    raise StyleValidationError(failure[0], failure[1]) from None
