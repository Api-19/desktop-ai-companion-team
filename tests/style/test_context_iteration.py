import traceback

import pytest

from companion.contracts.style import StyleIntent, StylePreference
from companion.privacy import CandidateCategory
from companion.style.context_iteration import build_local_context_summary
from companion.style.errors import StyleValidationError
from companion.style.evaluation import build_holdout_evaluation_plan
from companion.style.features import extract_style_features
from companion.style.iteration_models import (
    ContextHoldout,
    ContextWindowReference,
    LocalContextSummary,
)
from companion.style.pack import build_style_pack
from companion.style.retrieval import StyleExampleIndex


def _reference(window_token: str = "a" * 64) -> ContextWindowReference:
    return ContextWindowReference(window_token=window_token, group_token="b" * 64)


def _traceback_retains(error: BaseException, needle: str) -> bool:
    for frame, _ in traceback.walk_tb(error.__traceback__):
        filename = frame.f_code.co_filename.replace("/", "\\")
        if "companion\\style" in filename and needle in repr(frame.f_locals):
            return True
    return False


def _holdout(
    *,
    holdout_window_tokens: list[str] | None = None,
    learning_window_tokens: list[str] | None = None,
) -> ContextHoldout:
    return ContextHoldout(
        dataset_fingerprint="f" * 64,
        learning_window_tokens=learning_window_tokens or ["b" * 64],
        holdout_window_tokens=holdout_window_tokens or ["a" * 64],
        target_holdout_ratio=0.2,
    )


@pytest.mark.parametrize(
    ("privacy_approved", "mapping_confirmed", "sensitive", "placeholder", "reason"),
    [
        (False, True, False, False, "privacy_not_approved"),
        (True, False, False, False, "speaker_mapping_unconfirmed"),
        (True, True, True, False, "sensitive_context"),
        (True, True, False, True, "identity_placeholder"),
    ],
)
def test_summary_boundary_skips_unsafe_inputs(
    privacy_approved: bool,
    mapping_confirmed: bool,
    sensitive: bool,
    placeholder: bool,
    reason: str,
) -> None:
    result = build_local_context_summary(
        reference=_reference(),
        category=CandidateCategory.SLEEP,
        safe_summary="synthetic safe sleep transition",
        holdout=_holdout(),
        privacy_approved=privacy_approved,
        speaker_mapping_confirmed=mapping_confirmed,
        contains_sensitive_context=sensitive,
        contains_identity_placeholder=placeholder,
    )

    assert result.summary is None
    assert result.skipped_reason == reason


def test_summary_boundary_uses_explicit_fail_closed_priority() -> None:
    result = build_local_context_summary(
        reference=_reference(),
        category=CandidateCategory.SLEEP,
        safe_summary="synthetic safe sleep transition",
        holdout=_holdout(),
        privacy_approved=False,
        speaker_mapping_confirmed=False,
        contains_sensitive_context=True,
        contains_identity_placeholder=True,
    )

    assert result.summary is None
    assert result.skipped_reason == "speaker_mapping_unconfirmed"


@pytest.mark.parametrize("flag", ["false", 0, 1, None])
def test_summary_boundary_rejects_non_boolean_runtime_flags(flag: object) -> None:
    with pytest.raises(StyleValidationError, match="invalid_context_summary"):
        build_local_context_summary(
            reference=_reference(),
            category=CandidateCategory.SLEEP,
            safe_summary="synthetic",
            holdout=_holdout(),
            privacy_approved=flag,  # type: ignore[arg-type]
            speaker_mapping_confirmed=True,
            contains_sensitive_context=False,
            contains_identity_placeholder=False,
        )


def test_summary_boundary_revalidates_forged_reference_and_holdout() -> None:
    forged_reference = ContextWindowReference.model_construct(
        window_token="short",
        group_token="b" * 64,
    )
    with pytest.raises(StyleValidationError, match="invalid_context_summary"):
        build_local_context_summary(
            reference=forged_reference,
            category=CandidateCategory.SLEEP,
            safe_summary="synthetic",
            holdout=_holdout(),
            privacy_approved=True,
            speaker_mapping_confirmed=True,
            contains_sensitive_context=False,
            contains_identity_placeholder=False,
        )


@pytest.mark.parametrize(
    "category",
    [
        CandidateCategory.CALIBRATED_DAILY,
        CandidateCategory.SHORT_REPLY,
        CandidateCategory.EMOJI,
    ],
)
def test_summary_boundary_skips_unsupported_fixed_scene_categories(
    category: CandidateCategory,
) -> None:
    result = build_local_context_summary(
        reference=_reference(),
        category=category,
        safe_summary="synthetic safe fixed scene",
        holdout=_holdout(),
        privacy_approved=True,
        speaker_mapping_confirmed=True,
        contains_sensitive_context=False,
        contains_identity_placeholder=False,
    )

    assert result.summary is None
    assert result.skipped_reason == "unsupported_category"


def test_summary_boundary_returns_only_a_safe_local_summary() -> None:
    result = build_local_context_summary(
        reference=_reference(),
        category=CandidateCategory.SLEEP,
        safe_summary="synthetic safe sleep transition",
        holdout=_holdout(),
        privacy_approved=True,
        speaker_mapping_confirmed=True,
        contains_sensitive_context=False,
        contains_identity_placeholder=False,
    )

    assert result.skipped_reason is None
    assert result.summary == LocalContextSummary(
        window_token="a" * 64,
        category=CandidateCategory.SLEEP,
        prompt="synthetic safe sleep transition",
        privacy_approved=True,
        raw_text_retained=False,
        partner_text_retained=False,
    )


@pytest.mark.parametrize(
    ("safe_summary", "sentinel"),
    [
        ("", ""),
        ({"summary": "invalid-safe-summary-sentinel"}, "invalid-safe-summary-sentinel"),
        (
            "overlong-safe-summary-sentinel" + "x" * 160,
            "overlong-safe-summary-sentinel",
        ),
    ],
)
def test_summary_boundary_scrubs_invalid_safe_summary_validation_errors(
    safe_summary: object,
    sentinel: str,
) -> None:
    with pytest.raises(StyleValidationError) as exc_info:
        build_local_context_summary(
            reference=_reference(),
            category=CandidateCategory.SLEEP,
            safe_summary=safe_summary,  # type: ignore[arg-type]
            holdout=_holdout(),
            privacy_approved=True,
            speaker_mapping_confirmed=True,
            contains_sensitive_context=False,
            contains_identity_placeholder=False,
        )

    assert exc_info.value.code == "invalid_context_summary"
    assert str(exc_info.value) == "invalid_context_summary: context summary input is invalid"
    if sentinel:
        assert sentinel not in str(exc_info.value)
        assert not _traceback_retains(exc_info.value, sentinel)


def test_summary_boundary_rejects_learning_window_token() -> None:
    with pytest.raises(StyleValidationError) as exc_info:
        build_local_context_summary(
            reference=_reference("b" * 64),
            category=CandidateCategory.SLEEP,
            safe_summary="synthetic safe learning transition",
            holdout=_holdout(),
            privacy_approved=True,
            speaker_mapping_confirmed=True,
            contains_sensitive_context=False,
            contains_identity_placeholder=False,
        )

    assert exc_info.value.code == "invalid_context_summary"
    assert "b" * 64 not in str(exc_info.value)


def test_holdout_plan_uses_only_exact_category_safe_material(
    synthetic_candidate_package,
) -> None:
    base_pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    style_pack = base_pack.model_copy(
        update={
            "preferences": [
                StylePreference(
                    rule_id="sleep-safe-preference",
                    category=CandidateCategory.SLEEP.value,
                    expression="synthetic sleep preference",
                    trigger_tags=[CandidateCategory.SLEEP.value],
                    max_consecutive_uses=1,
                ),
                StylePreference(
                    rule_id="meals-preference-must-not-leak",
                    category=CandidateCategory.MEALS.value,
                    expression="synthetic meals preference",
                    trigger_tags=[CandidateCategory.MEALS.value],
                    max_consecutive_uses=1,
                ),
            ]
        }
    )
    summaries = [
        LocalContextSummary(
            window_token="d" * 64,
            category=CandidateCategory.SLEEP,
            prompt="synthetic safe sleep transition",
            privacy_approved=True,
            raw_text_retained=False,
            partner_text_retained=False,
        ),
        LocalContextSummary(
            window_token="c" * 64,
            category=CandidateCategory.SLEEP,
            prompt="synthetic safe second sleep transition",
            privacy_approved=True,
            raw_text_retained=False,
            partner_text_retained=False,
        ),
    ]

    plan = build_holdout_evaluation_plan(
        style_pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        summaries,
        holdout=_holdout(holdout_window_tokens=["c" * 64, "d" * 64]),
    )

    assert plan.base_style_version == style_pack.version
    assert [item.window_token for item in plan.items] == ["c" * 64, "d" * 64]
    assert all(item.styled_prompt.startswith(item.baseline_prompt) for item in plan.items)
    assert all("独立 AI" in item.baseline_prompt for item in plan.items)
    assert all("不得冒充主人" in item.baseline_prompt for item in plan.items)
    assert all("不得补写私人经历" in item.baseline_prompt for item in plan.items)
    assert all("synthetic sleep preference" in item.styled_prompt for item in plan.items)
    assert all("synthetic meals preference" not in item.styled_prompt for item in plan.items)
    assert all(len(item.styled_examples) <= 2 for item in plan.items)
    assert all("吃饭" not in item.styled_prompt for item in plan.items)
    assert all("raw context sentinel" not in item.styled_prompt for item in plan.items)
    assert all("partner text sentinel" not in item.styled_prompt for item in plan.items)


def test_holdout_plan_excludes_specific_intent_preferences_without_intent(
    synthetic_candidate_package,
) -> None:
    base_pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    style_pack = base_pack.model_copy(
        update={
            "preferences": [
                StylePreference(
                    rule_id="sleep-legacy-category-preference",
                    category=CandidateCategory.SLEEP.value,
                    expression="synthetic legacy sleep pref",
                    trigger_tags=[CandidateCategory.SLEEP.value],
                    max_consecutive_uses=1,
                ),
                StylePreference(
                    rule_id="sleep-explicit-category-preference",
                    category=CandidateCategory.SLEEP.value,
                    expression="synthetic default sleep pref",
                    trigger_tags=[CandidateCategory.SLEEP.value],
                    intent_tags=[StyleIntent.CATEGORY_DEFAULT],
                    max_consecutive_uses=1,
                ),
                StylePreference(
                    rule_id="sleep-farewell-must-not-leak",
                    category=CandidateCategory.SLEEP.value,
                    expression="synthetic farewell intent pref",
                    trigger_tags=[CandidateCategory.SLEEP.value],
                    intent_tags=[StyleIntent.SLEEP_FAREWELL],
                    max_consecutive_uses=1,
                ),
                StylePreference(
                    rule_id="sleep-reminder-must-not-leak",
                    category=CandidateCategory.SLEEP.value,
                    expression="synthetic reminder intent pref",
                    trigger_tags=[CandidateCategory.SLEEP.value],
                    intent_tags=[StyleIntent.SLEEP_REMINDER],
                    max_consecutive_uses=1,
                ),
            ]
        }
    )
    summary = LocalContextSummary(
        window_token="a" * 64,
        category=CandidateCategory.SLEEP,
        prompt="synthetic safe sleep transition without intent",
        privacy_approved=True,
        raw_text_retained=False,
        partner_text_retained=False,
    )

    item = build_holdout_evaluation_plan(
        style_pack,
        StyleExampleIndex.from_package(synthetic_candidate_package),
        [summary],
        holdout=_holdout(),
    ).items[0]

    assert "synthetic legacy sleep pref" in item.styled_prompt
    assert "synthetic default sleep pref" in item.styled_prompt
    assert "synthetic farewell intent pref" not in item.styled_prompt
    assert "synthetic reminder intent pref" not in item.styled_prompt


@pytest.mark.parametrize(
    "summaries",
    [
        [],
        [
            LocalContextSummary(
                window_token="a" * 64,
                category=CandidateCategory.SLEEP,
                prompt="synthetic safe sleep transition",
                privacy_approved=True,
                raw_text_retained=False,
                partner_text_retained=False,
            ),
            LocalContextSummary(
                window_token="a" * 64,
                category=CandidateCategory.SLEEP,
                prompt="synthetic safe second sleep transition",
                privacy_approved=True,
                raw_text_retained=False,
                partner_text_retained=False,
            ),
        ],
    ],
)
def test_holdout_plan_rejects_empty_or_duplicate_summaries(
    synthetic_candidate_package,
    summaries: list[LocalContextSummary],
) -> None:
    style_pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )

    with pytest.raises(StyleValidationError, match="invalid_holdout_summaries"):
        build_holdout_evaluation_plan(
            style_pack,
            StyleExampleIndex.from_package(synthetic_candidate_package),
            summaries,
            holdout=_holdout(),
        )


def test_holdout_plan_rejects_learning_window_summary(
    synthetic_candidate_package,
) -> None:
    style_pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    learning_summary = LocalContextSummary(
        window_token="b" * 64,
        category=CandidateCategory.SLEEP,
        prompt="synthetic safe learning transition",
        privacy_approved=True,
        raw_text_retained=False,
        partner_text_retained=False,
    )

    with pytest.raises(StyleValidationError) as exc_info:
        build_holdout_evaluation_plan(
            style_pack,
            StyleExampleIndex.from_package(synthetic_candidate_package),
            [learning_summary],
            holdout=_holdout(),
        )

    assert exc_info.value.code == "invalid_holdout_summaries"
    assert "b" * 64 not in str(exc_info.value)


def test_holdout_plan_failure_traceback_does_not_retain_summary_prompt(
    synthetic_candidate_package,
) -> None:
    sentinel = "holdout-summary-prompt-sentinel"
    style_pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )
    learning_summary = LocalContextSummary(
        window_token="b" * 64,
        category=CandidateCategory.SLEEP,
        prompt=sentinel,
        privacy_approved=True,
        raw_text_retained=False,
        partner_text_retained=False,
    )

    with pytest.raises(StyleValidationError) as exc_info:
        build_holdout_evaluation_plan(
            style_pack,
            StyleExampleIndex.from_package(synthetic_candidate_package),
            [learning_summary],
            holdout=_holdout(),
        )

    assert exc_info.value.code == "invalid_holdout_summaries"
    assert sentinel not in str(exc_info.value)
    assert not _traceback_retains(exc_info.value, sentinel)


def test_holdout_plan_revalidates_forged_summary_without_prompt_retention(
    synthetic_candidate_package,
) -> None:
    sentinel = "synthetic-forged-holdout-prompt-sentinel"
    summary = LocalContextSummary.model_construct(
        window_token="a" * 64,
        category=CandidateCategory.SLEEP,
        prompt=sentinel,
        privacy_approved=False,
        raw_text_retained=True,
        partner_text_retained=True,
    )
    pack = build_style_pack(
        synthetic_candidate_package,
        extract_style_features(synthetic_candidate_package),
    )

    with pytest.raises(StyleValidationError) as caught:
        build_holdout_evaluation_plan(
            pack,
            StyleExampleIndex.from_package(synthetic_candidate_package),
            [summary],
            holdout=_holdout(),
        )

    assert caught.value.code == "invalid_holdout_summaries"
    assert sentinel not in str(caught.value)
    assert not _traceback_retains(caught.value, sentinel)


def test_holdout_plan_rejects_runtime_invalid_inputs_fail_closed() -> None:
    with pytest.raises(StyleValidationError, match="invalid_holdout_summaries"):
        build_holdout_evaluation_plan(  # type: ignore[arg-type]
            "invalid-pack",
            "invalid-index",
            [],
            holdout="invalid",
        )
