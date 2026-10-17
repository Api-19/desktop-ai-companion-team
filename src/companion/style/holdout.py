import hashlib
from collections import defaultdict

from companion.style.errors import StyleValidationError
from companion.style.iteration_models import ContextHoldout, ContextWindowReference


def _group_score(dataset_fingerprint: str, group_token: str) -> str:
    material = f"{dataset_fingerprint}:{group_token}".encode()
    return hashlib.sha256(material).hexdigest()


def _build_context_holdout(
    references: list[ContextWindowReference],
    *,
    dataset_fingerprint: str,
    target_holdout_ratio: float,
) -> ContextHoldout:
    groups: dict[str, list[str]] = defaultdict(list)
    for reference in references:
        groups[reference.group_token].append(reference.window_token)
    if len(groups) < 2:
        raise StyleValidationError(
            "insufficient_holdout_groups",
            "context holdout requires at least two groups",
        )

    ranked_groups = sorted(
        groups,
        key=lambda token: (_group_score(dataset_fingerprint, token), token),
    )
    total_window_count = len(references)
    reachable = [False] * (total_window_count + 1)
    previous_window_count = [-1] * (total_window_count + 1)
    previous_group_index = [-1] * (total_window_count + 1)
    reachable[0] = True
    for group_index, group_token in enumerate(ranked_groups):
        group_size = len(groups[group_token])
        for window_count in range(total_window_count, group_size - 1, -1):
            prior_count = window_count - group_size
            if not reachable[window_count] and reachable[prior_count]:
                reachable[window_count] = True
                previous_window_count[window_count] = prior_count
                previous_group_index[window_count] = group_index
    target_window_count = total_window_count * target_holdout_ratio
    selected_window_count = min(
        (window_count for window_count in range(1, total_window_count) if reachable[window_count]),
        key=lambda window_count: (
            abs(window_count - target_window_count),
            window_count,
        ),
    )
    holdout_groups: set[str] = set()
    while selected_window_count:
        group_index = previous_group_index[selected_window_count]
        holdout_groups.add(ranked_groups[group_index])
        selected_window_count = previous_window_count[selected_window_count]
    holdout = sorted(token for group in holdout_groups for token in groups[group])
    learning = sorted(
        token for group in groups if group not in holdout_groups for token in groups[group]
    )
    return ContextHoldout(
        dataset_fingerprint=dataset_fingerprint,
        learning_window_tokens=learning,
        holdout_window_tokens=holdout,
        target_holdout_ratio=target_holdout_ratio,
    )


def build_context_holdout(
    references: list[ContextWindowReference],
    *,
    dataset_fingerprint: str,
    target_holdout_ratio: float = 0.2,
) -> ContextHoldout:
    try:
        validated_references = [
            ContextWindowReference.model_validate(reference.model_dump(mode="json", warnings=False))
            for reference in references
        ]
        return _build_context_holdout(
            validated_references,
            dataset_fingerprint=dataset_fingerprint,
            target_holdout_ratio=target_holdout_ratio,
        )
    except StyleValidationError as error:
        failure = (error.code, error.detail)
        del error
    except Exception:
        failure = ("invalid_context_holdout", "context holdout input is invalid")
    finally:
        references = None  # type: ignore[assignment]
        dataset_fingerprint = None  # type: ignore[assignment]
        target_holdout_ratio = None  # type: ignore[assignment]
        validated_references = None
    raise StyleValidationError(*failure) from None
