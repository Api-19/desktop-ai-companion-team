"""Tests for single-user Style adapter that binds resources to one owner."""

from __future__ import annotations

from pathlib import Path

import pytest

from companion.contracts import StyleGuidanceStatus, StyleReplyGuidance
from companion.style import SingleUserStyleAdapter


def _make_adapter(tmp_path: Path, trusted_user_id: str) -> SingleUserStyleAdapter:
    """Create a minimal adapter with synthetic sources."""
    # Import locally to use test helpers
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_runtime_guidance import _resources
    finally:
        sys.path.pop(0)

    sources, _ = _resources(tmp_path, approved_nicknames=["合成爱称"])
    return SingleUserStyleAdapter(trusted_user_id=trusted_user_id, **sources)


def test_adapter_rejects_non_builtin_str_user_id_at_construction(tmp_path: Path) -> None:
    """Adapter construction requires exact builtin str, not subclasses."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_runtime_guidance import _resources
    finally:
        sys.path.pop(0)

    sources, _ = _resources(tmp_path)

    class StrSubclass(str):
        pass

    with pytest.raises(TypeError, match="exact builtin str"):
        SingleUserStyleAdapter(trusted_user_id=StrSubclass("user-a"), **sources)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_user_id", ["", "   ", "x" * 121])
def test_adapter_rejects_invalid_user_id_at_construction(tmp_path: Path, bad_user_id: str) -> None:
    """Adapter construction enforces the shared user ID capacity contract."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_runtime_guidance import _resources
    finally:
        sys.path.pop(0)

    sources, _ = _resources(tmp_path)

    with pytest.raises(ValueError, match="1 to 120"):
        SingleUserStyleAdapter(trusted_user_id=bad_user_id, **sources)


def test_adapter_accepts_only_exact_bound_owner(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path, trusted_user_id="owner-a")

    assert adapter.accepts_user("owner-a") is True
    assert adapter.accepts_user("owner-b") is False

    class StrSubclass(str):
        pass

    assert adapter.accepts_user(StrSubclass("owner-a")) is False


def test_owner_request_returns_matched_guidance(tmp_path: Path) -> None:
    """RED: Owner A with verified A gets matched guidance."""
    adapter = _make_adapter(tmp_path, trusted_user_id="owner-a")

    guidance = adapter.guidance_for(
        user_id="owner-a",
        text="今天降温了",
        recent_context=[],
        idle_seconds=0,
    )

    assert guidance.status is StyleGuidanceStatus.MATCHED
    assert guidance.style_version is not None


def test_non_owner_verified_request_fails_closed_without_reading_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED: User B with verified B cannot read owner A's Style sources."""
    adapter = _make_adapter(tmp_path, trusted_user_id="owner-a")
    calls = 0

    def forbidden_runtime_call(*args: object, **kwargs: object) -> StyleReplyGuidance:
        nonlocal calls
        calls += 1
        raise AssertionError("non-owner reached Style runtime")

    monkeypatch.setattr(
        "companion.style.single_user_adapter.build_style_reply_guidance",
        forbidden_runtime_call,
    )

    # User B's request + upstream assertion B (both match)
    # But B != bound owner A, so must fail closed
    guidance = adapter.guidance_for(
        user_id="owner-b",
        text="今天降温了",
        recent_context=[],
        idle_seconds=0,
    )

    assert guidance.status is StyleGuidanceStatus.UNAVAILABLE
    assert guidance.style_version is None
    assert calls == 0


def test_owner_call_after_rejected_non_owner_still_works(tmp_path: Path) -> None:
    """RED: Owner A remains valid after rejecting B."""
    adapter = _make_adapter(tmp_path, trusted_user_id="owner-a")

    # B fails
    guidance_b = adapter.guidance_for(
        user_id="owner-b",
        text="今天降温了",
        recent_context=[],
        idle_seconds=0,
    )
    assert guidance_b.status is StyleGuidanceStatus.UNAVAILABLE

    # A still works (no session pollution)
    guidance_a = adapter.guidance_for(
        user_id="owner-a",
        text="今天降温了",
        recent_context=[],
        idle_seconds=0,
    )
    assert guidance_a.status is StyleGuidanceStatus.MATCHED


@pytest.mark.parametrize(
    "bad_user_id",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty"),
        pytest.param("x" * 1000, id="long"),
    ],
)
def test_adapter_rejects_invalid_user_id_without_reading_sources(
    tmp_path: Path,
    bad_user_id: str | None,
) -> None:
    """RED: Invalid user_id types/values fail closed."""
    adapter = _make_adapter(tmp_path, trusted_user_id="owner-a")

    guidance = adapter.guidance_for(
        user_id=bad_user_id,  # type: ignore[arg-type]
        text="今天降温了",
        recent_context=[],
        idle_seconds=0,
    )

    assert guidance.status is StyleGuidanceStatus.UNAVAILABLE
    assert guidance.style_version is None


def test_adapter_rejects_str_subclass_user_id(tmp_path: Path) -> None:
    """RED: str subclass user_id fails identity check."""
    adapter = _make_adapter(tmp_path, trusted_user_id="owner-a")

    class StrSubclass(str):
        pass

    guidance = adapter.guidance_for(
        user_id=StrSubclass("owner-a"),  # type: ignore[arg-type]
        text="今天降温了",
        recent_context=[],
        idle_seconds=0,
    )

    assert guidance.status is StyleGuidanceStatus.UNAVAILABLE
