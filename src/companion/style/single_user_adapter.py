"""Single-user Style read adapter that binds runtime sources to one trusted user."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from companion.contracts import StyleReplyGuidance
from companion.style.runtime_guidance import build_style_reply_guidance

if TYPE_CHECKING:
    from collections.abc import Sequence


class SingleUserStyleAdapter:
    """
    Binds a fixed set of Style runtime sources to exactly one trusted_user_id.

    Requests for any other user_id fail closed without reading Style files or
    calling the underlying provider.
    """

    def __init__(
        self,
        *,
        trusted_user_id: str,
        candidate_source: Path,
        expected_count: int,
        active_style_source: Path,
        active_manifest_source: Path,
        expected_active_manifest_digest: str,
        calibration_source: Path,
        expected_calibration_fingerprint: str,
        private_data_dir: Path,
        true_name_terms: Sequence[str],
        approved_nicknames: Sequence[str],
    ) -> None:
        """
        Bind Style sources to trusted_user_id at construction time.

        Args:
            trusted_user_id: Exact builtin str of the single allowed user.
            All other parameters: Style runtime sources (unchanged from existing API).
        """
        if type(trusted_user_id) is not str:
            raise TypeError("trusted_user_id must be exact builtin str")
        if not trusted_user_id.strip() or len(trusted_user_id) > 120:
            raise ValueError("trusted_user_id must contain 1 to 120 non-whitespace characters")

        self._trusted_user_id = trusted_user_id
        self._candidate_source = candidate_source
        self._expected_count = expected_count
        self._active_style_source = active_style_source
        self._active_manifest_source = active_manifest_source
        self._expected_active_manifest_digest = expected_active_manifest_digest
        self._calibration_source = calibration_source
        self._expected_calibration_fingerprint = expected_calibration_fingerprint
        self._private_data_dir = private_data_dir
        self._true_name_terms = tuple(true_name_terms)
        self._approved_nicknames = tuple(approved_nicknames)

    @property
    def trusted_user_id(self) -> str:
        """Return the exact user ID bound to this adapter."""
        return self._trusted_user_id

    def accepts_user(self, user_id: str) -> bool:
        """Return whether this adapter may access Style resources for ``user_id``."""
        return type(user_id) is str and user_id == self._trusted_user_id

    def guidance_for(
        self,
        user_id: str,
        text: str,
        recent_context: Sequence[str],
        idle_seconds: int | None,
    ) -> StyleReplyGuidance:
        """
        Return style guidance for the given user_id if it matches the bound owner.

        Args:
            user_id: The user_id from the verified request.
            text: Current user message text.
            recent_context: Recent conversation messages.
            idle_seconds: Seconds since last completed request, or None.

        Returns:
            StyleReplyGuidance with status MATCHED, NO_MATCH, or UNAVAILABLE.

        Raises:
            No exceptions - fails closed to UNAVAILABLE on identity mismatch.
        """
        # Identity gate: only the bound owner can read Style sources
        if not self.accepts_user(user_id):
            # Fail closed without reading any Style files or calling provider
            return StyleReplyGuidance(status="unavailable", style_version=None)

        # Owner verified - delegate to underlying provider with bound sources
        return build_style_reply_guidance(
            text,
            recent_context,
            idle_seconds,
            user_id=user_id,
            candidate_source=self._candidate_source,
            expected_count=self._expected_count,
            active_style_source=self._active_style_source,
            active_manifest_source=self._active_manifest_source,
            expected_active_manifest_digest=self._expected_active_manifest_digest,
            calibration_source=self._calibration_source,
            expected_calibration_fingerprint=self._expected_calibration_fingerprint,
            private_data_dir=self._private_data_dir,
            true_name_terms=self._true_name_terms,
            approved_nicknames=self._approved_nicknames,
        )
