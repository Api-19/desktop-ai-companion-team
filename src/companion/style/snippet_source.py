"""Refresh partner style evidence from the active local snippet snapshot."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from companion.contracts.style import StyleEvidence
from companion.privacy.paired_snippets import ActiveSnippetReadError, read_active_snippet_snapshot
from companion.style.errors import StyleValidationError
from companion.style.snippet_evidence import SnippetStyleAdapter


class FileSnippetStyleSource:
    """Keep reuse history only while the source content remains unchanged."""

    def __init__(
        self,
        *,
        library_source: Path,
        partner_user_id: str,
        style_version: str,
        clock: Callable[[], datetime],
    ) -> None:
        # Delegate ID validation and memory-only behavior to the existing adapter.
        self._empty_adapter = SnippetStyleAdapter(
            snippets=[], partner_user_id=partner_user_id, style_version=style_version, clock=clock
        )
        self._library_source = library_source
        self._exclusions_source = library_source.parent / "deleted_ids.json"
        self._partner_user_id = partner_user_id
        self._style_version = style_version
        self._clock = clock
        self._adapter = self._empty_adapter
        self._revision = ""
        self._refresh()

    def accepts_user(self, user_id: str) -> bool:
        """Apply the adapter ID rules without accessing the library."""
        return self._empty_adapter.accepts_user(user_id)

    def evidence_for(self, user_id: str, text: str, recent_context: Sequence[str]) -> StyleEvidence:
        """Refresh partner evidence; other users receive neutral empty evidence."""
        if user_id != self._partner_user_id:
            return self._empty_adapter.evidence_for(user_id, text, recent_context)
        self._refresh()
        return self._adapter.evidence_for(user_id, text, recent_context)

    def content_revision(self, user_id: str) -> str:
        """Return the snapshot revision without advancing evidence reuse history."""
        if user_id != self._partner_user_id:
            return ""
        self._refresh()
        return self._revision

    def _refresh(self) -> None:
        try:
            pairs, revision = read_active_snippet_snapshot(
                self._library_source, self._exclusions_source
            )
            if revision == self._revision:
                return
            snippets = [
                snippet for pair in pairs if (snippet := pair.as_style_snippet()) is not None
            ]
            self._adapter = SnippetStyleAdapter(
                snippets=snippets,
                partner_user_id=self._partner_user_id,
                style_version=self._style_version,
                clock=self._clock,
            )
            self._revision = revision
            return
        except (ActiveSnippetReadError, ValueError):
            self._adapter = self._empty_adapter
            self._revision = ""
        raise StyleValidationError("snippet_source_unavailable", "snippet source unavailable")
