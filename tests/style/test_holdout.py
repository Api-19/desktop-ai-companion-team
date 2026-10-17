import hashlib
import traceback
from collections import defaultdict

import pytest

from companion.style.errors import StyleValidationError
from companion.style.holdout import build_context_holdout
from companion.style.iteration_models import ContextWindowReference


def _token(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def synthetic_references(group_count: int, windows_per_group: int) -> list[ContextWindowReference]:
    return [
        ContextWindowReference(
            window_token=_token(f"window-{group}-{window}"),
            group_token=_token(f"group-{group}"),
        )
        for group in range(group_count)
        for window in range(windows_per_group)
    ]


def grouped_tokens(references: list[ContextWindowReference]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for reference in references:
        grouped[reference.group_token].append(reference.window_token)
    return grouped


def _traceback_retains(error: BaseException, sentinel: str) -> bool:
    return any(
        sentinel in repr(frame.f_locals)
        for frame, _ in traceback.walk_tb(error.__traceback__)
        if "/companion/style/" in frame.f_code.co_filename.replace("\\", "/")
    )


def test_holdout_is_stable_and_keeps_groups_together() -> None:
    references = synthetic_references(group_count=10, windows_per_group=2)

    first = build_context_holdout(references, dataset_fingerprint="a" * 64)
    second = build_context_holdout(list(reversed(references)), dataset_fingerprint="a" * 64)

    assert first == second
    learning = set(first.learning_window_tokens)
    holdout = set(first.holdout_window_tokens)
    for group in grouped_tokens(references).values():
        assert set(group).issubset(learning) or set(group).issubset(holdout)
    assert len(holdout) == 4


def test_holdout_requires_two_groups() -> None:
    with pytest.raises(StyleValidationError, match="insufficient_holdout_groups"):
        build_context_holdout(
            synthetic_references(group_count=1, windows_per_group=2),
            dataset_fingerprint="a" * 64,
        )


def test_holdout_revalidates_forged_reference_tokens_without_retention() -> None:
    sentinel = "z" * 64
    references = [
        ContextWindowReference.model_construct(
            window_token=_token("window-forged"),
            group_token=sentinel,
        ),
        ContextWindowReference(
            window_token=_token("window-valid"),
            group_token=_token("group-valid"),
        ),
    ]

    with pytest.raises(StyleValidationError) as caught:
        build_context_holdout(references, dataset_fingerprint="a" * 64)

    assert caught.value.code == "invalid_context_holdout"
    assert sentinel not in str(caught.value)
    assert not _traceback_retains(caught.value, sentinel)


def test_holdout_ratio_is_based_on_window_count_without_splitting_groups() -> None:
    group_sizes = {"big": 8, "s1": 1, "s2": 1}
    references = [
        ContextWindowReference(
            window_token=_token(f"window-{group}-{window}"),
            group_token=_token(group),
        )
        for group, size in group_sizes.items()
        for window in range(size)
    ]

    holdout = build_context_holdout(
        references,
        dataset_fingerprint="a" * 64,
        target_holdout_ratio=0.2,
    )

    assert len(holdout.holdout_window_tokens) == 2
    assert set(holdout.holdout_window_tokens) == {
        _token("window-s1-0"),
        _token("window-s2-0"),
    }


@pytest.mark.parametrize(
    "target_holdout_ratio",
    [float("nan"), float("inf"), float("-inf"), "synthetic-private-ratio"],
)
def test_invalid_holdout_ratio_fails_closed_without_exposing_input(
    target_holdout_ratio: float | str,
) -> None:
    with pytest.raises(StyleValidationError) as exc_info:
        build_context_holdout(
            synthetic_references(group_count=2, windows_per_group=1),
            dataset_fingerprint="a" * 64,
            target_holdout_ratio=target_holdout_ratio,
        )

    assert exc_info.value.code == "invalid_context_holdout"
    assert "synthetic-private-ratio" not in str(exc_info.value)
