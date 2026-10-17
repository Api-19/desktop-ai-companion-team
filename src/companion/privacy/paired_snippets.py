"""Partner-turn -> owner-reply snippet pipeline over confirmed chat records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime
from io import StringIO
from itertools import groupby, pairwise
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
)

from companion.contracts.base import ContractModel, TimezoneRequiredModel
from companion.contracts.chat import ChatRecord, SpeakerRole
from companion.contracts.style import StyleSnippet
from companion.privacy.models import RiskCategory
from companion.privacy.redaction import PrivateRedactionTerms, redact_text
from companion.privacy.review import (
    PrivateTerms,
    dataset_fingerprint,
    scan_for_review,
    segment_records,
)
from companion.privacy.veiled import VeiledLexicon

SNIPPET_EXCLUDED_RISKS = frozenset({
    RiskCategory.SEXUAL,
    RiskCategory.CREDENTIAL,
    RiskCategory.FINANCIAL,
    RiskCategory.LOCATION,
    RiskCategory.PRIVATE_EVENT,
})

_MAX_TURNS = 3
_PARTNER_MAX_CHARS = 60
_OWNER_MAX_CHARS = 120
_NEAR_DUPLICATE_JACCARD = 0.8
_PSEUDONYM_CHARACTERS = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
_MEDIA_DEICTIC_TERMS = (
    "这张", "那张", "哪张", "图里", "图中", "图片里", "照片里", "截图里",
    "视频里", "语音里", "上图", "下图",
)
_SHORT_MEDIA_REACTIONS = frozenset({
    "这个呢", "那个呢", "你看看", "看看", "好看吗", "怎么样", "咋样", "好看",
    "不好看", "可以", "不错", "嗯", "对", "是的",
})
_REACTION_ENDINGS = "，。！？!?～~…"  # noqa: RUF001


class PseudonymEntity(ContractModel):
    id: str
    kind: Literal["person", "place"]
    aliases: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(min_length=1)
    pseudonym: str | None

    @field_validator("aliases")
    @classmethod
    def reject_blank_aliases(cls, aliases: list[str]) -> list[str]:
        if any(not alias.strip() for alias in aliases):
            raise ValueError("entity aliases must not be blank")
        return aliases


class _EntityFile(ContractModel):
    version: Literal[1]
    entities: list[PseudonymEntity]


def _entity_mapping(entities: Sequence[PseudonymEntity]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    for entity in entities:
        aliases = set(entity.aliases)
        if mapping.keys() & aliases:
            raise ValueError("entity aliases must not overlap across entities")
        mapping.update(dict.fromkeys(aliases, entity.pseudonym))
    return mapping


def load_entities(path: Path) -> list[PseudonymEntity]:
    """Load the versioned local entity mapping, rejecting ambiguous aliases."""
    payload = _EntityFile.model_validate_json(path.read_text(encoding="utf-8-sig"))
    _entity_mapping(payload.entities)
    return payload.entities


class PairedSnippet(TimezoneRequiredModel):
    snippet_id: str
    conversation_id: str
    started_at: datetime
    partner_turns: list[str]
    owner_reply: list[str]
    source_fingerprint: str
    context_turns: list[str] = Field(default_factory=list)
    privacy_flags: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_flags(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload = handler(self)
        if not self.privacy_flags:
            payload.pop("privacy_flags", None)
        return payload

    def to_style_snippet(self) -> StyleSnippet:
        return StyleSnippet(partner_turns=self.partner_turns, owner_reply=self.owner_reply)

    def as_style_snippet(self) -> StyleSnippet | None:
        """Return short style evidence only when all StyleSnippet limits hold."""
        try:
            return self.to_style_snippet()
        except ValidationError:
            return None


def _snippet_id(conversation_id: str, message_id: str) -> str:
    digest = hashlib.sha256(f"{conversation_id}\x1f{message_id}".encode()).hexdigest()
    return f"snip-{digest[:16]}"


def _sort_key(snippet: PairedSnippet) -> tuple[datetime, str]:
    return snippet.started_at, snippet.snippet_id


def _redacted(
    record: ChatRecord,
    redaction_terms: PrivateRedactionTerms | None,
    pseudonyms: dict[str, str | None],
    pseudonym_pattern: re.Pattern[str] | None,
) -> str:
    text = record.text
    if pseudonym_pattern is None:
        return redact_text(text, private_terms=redaction_terms).text.strip()
    # Match original aliases once. Retained aliases are explicit exceptions to
    # every redaction rule; the remaining stretches still pass through redact_text.
    parts: list[str] = []
    pending: list[str] = []
    cursor = 0
    for match in pseudonym_pattern.finditer(text):
        pending.append(text[cursor:match.start()])
        replacement = pseudonyms[match.group()]
        if replacement is None:
            parts.append(redact_text("".join(pending), private_terms=redaction_terms).text)
            parts.append(match.group())
            pending = []
        else:
            pending.append(replacement)
        cursor = match.end()
    pending.append(text[cursor:])
    parts.append(redact_text("".join(pending), private_terms=redaction_terms).text)
    return "".join(parts).strip()


def _pseudonym_mapping(terms: PrivateRedactionTerms) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for category in ("name", "address"):
        for index, term in enumerate(terms.get(category, ())):
            cycle, offset = divmod(index, len(_PSEUDONYM_CHARACTERS))
            character = _PSEUDONYM_CHARACTERS[offset]
            alias = f"小{character}" if category == "name" else f"{character}地"
            mapping.setdefault(term, alias + (str(cycle + 1) if cycle else ""))
    return mapping


def _blocks_text(record: ChatRecord) -> bool:
    return record.content_type != "text" or not record.text or not record.text.strip()


def build_paired_snippets(
    records: list[ChatRecord],
    *,
    created_at: datetime,
    risk_terms: PrivateTerms | None = None,
    redaction_terms: PrivateRedactionTerms | None = None,
    excluded_risks: frozenset[RiskCategory] = SNIPPET_EXCLUDED_RISKS,
    context_turns: int = 0,
    exclude_near_nontext: bool = False,
    pseudonymize: bool = False,
    entities: Sequence[PseudonymEntity] | None = None,
    full_text: bool = False,
    veiled: VeiledLexicon | None = None,
    media_counts: dict[str, int] | None = None,
    blocked_message_ids: frozenset[str] = frozenset(),
    covered_message_ids: dict[str, frozenset[str]] | None = None,
) -> list[PairedSnippet]:
    """Build redacted pairs with optional same-segment context and media exclusion.

    Blocked messages always exclude full pairs and truncate context without refilling.
    With media exclusion enabled, blocked messages also mark media boundaries.
    Optional covered_message_ids receives each retained pair's full raw interval IDs.
    Media anywhere in the raw full partner-to-owner interval excludes the pair.
    Context keeps only turns after its last media boundary, without refilling.
    Media deictic terms exclude pairs; short reactions do so only at a boundary.
    Entities take precedence over name/address terms, with None pseudonyms
    preserving their aliases. Uncovered terms retain placeholder redaction.
    Full text keeps all partner turns and untruncated context.
    Veiled terms scan the original context and complete partner/owner runs:
    high hits exclude the pair; suspect hits become sorted privacy flags.
    Optional media_counts receives per-build counts before veiled/deletions:
    exclusions use the first matching media/deictic/short-reaction rule, while
    context truncations count even when a later rule excludes the pair.
    """
    fingerprint = dataset_fingerprint(records)
    confirmed = [record for record in records if record.speaker_role is not SpeakerRole.UNKNOWN]
    package = scan_for_review(confirmed, created_at=created_at, private_terms=risk_terms)
    risky_segments = {
        segment.source_segment_id
        for segment in package.risk_segments
        if excluded_risks.intersection(segment.risk_categories)
    }
    pseudonyms: dict[str, str | None] = {}
    pseudonym_pattern = None
    if entities is not None:
        pseudonyms = _entity_mapping(entities)
        redact_text(None, private_terms=redaction_terms)
        uncovered_terms: dict[str, Sequence[str]] = {}
        for category, terms in (redaction_terms or {}).items():
            if category in {"name", "address"}:
                terms = [term for term in terms if term not in pseudonyms]
                if not terms:
                    continue
            uncovered_terms[category] = terms
        redaction_terms = uncovered_terms
    elif pseudonymize and redaction_terms is not None:
        # Validate the original groups at the privacy boundary before removing any.
        redact_text(None, private_terms=redaction_terms)
        pseudonyms = _pseudonym_mapping(redaction_terms)
        redaction_terms = {
            category: terms for category, terms in redaction_terms.items()
            if category not in {"name", "address"}
        }
    if pseudonyms:
        pseudonym_pattern = re.compile(
            "|".join(
                re.escape(term) for term in sorted(pseudonyms, key=lambda term: (-len(term), term))
            )
        )

    counts = {
        "excluded_by_media": 0,
        "excluded_by_deictic": 0,
        "excluded_by_short_reaction": 0,
        "context_truncated_at_media": 0,
    }
    if exclude_near_nontext or blocked_message_ids or covered_message_ids is not None:
        original = sorted(
            records,
            key=lambda record: (record.conversation_id, record.timestamp, record.message_id),
        )
        original_positions = {record.message_id: index for index, record in enumerate(original)}
    snippets: list[PairedSnippet] = []
    for segment_id, segment in segment_records(confirmed):
        if segment_id in risky_segments:
            continue
        usable = [
            record
            for record in segment
            if record.content_type == "text" and record.text is not None and record.text.strip()
        ]
        usable_positions = {record.message_id: index for index, record in enumerate(usable)}
        runs = [(role, list(run)) for role, run in groupby(usable, key=lambda r: r.speaker_role)]
        for (role, partner_run), (next_role, owner_run) in pairwise(runs):
            if role is not SpeakerRole.OTHER or next_role is not SpeakerRole.OWNER:
                continue
            first = partner_run[0]
            preceding = (
                usable[max(0, usable_positions[first.message_id] - context_turns):
                       usable_positions[first.message_id]]
                if context_turns > 0 else []
            )
            original_preceding = preceding
            if exclude_near_nontext or blocked_message_ids:
                first_position = original_positions[first.message_id]
                last_position = original_positions[owner_run[-1].message_id]
                if any(
                    record.message_id in blocked_message_ids
                    or (exclude_near_nontext and _blocks_text(record))
                    for record in original[first_position:last_position + 1]
                ):
                    if exclude_near_nontext:
                        counts["excluded_by_media"] += 1
                    continue
                window_start = preceding[0] if preceding else first
                context_start = original_positions[window_start.message_id]
                # Include exactly one raw record left of the context window.
                last_blocker = next((
                    index for index in range(first_position - 1, max(0, context_start - 1) - 1, -1)
                    if original[index].conversation_id == first.conversation_id
                    and (original[index].message_id in blocked_message_ids
                         or (exclude_near_nontext and _blocks_text(original[index])))
                ), None)
                if last_blocker is not None:
                    if exclude_near_nontext:
                        counts["context_truncated_at_media"] += 1
                    preceding = [record for record in preceding
                                 if original_positions[record.message_id] > last_blocker]
                if blocked_message_ids:
                    # Deleted context must not resurface through veiled privacy flags.
                    last_deleted = next((
                        index for index in range(first_position - 1, context_start - 1, -1)
                        if original[index].message_id in blocked_message_ids
                    ), None)
                    if last_deleted is not None:
                        original_preceding = [
                            record for record in original_preceding
                            if original_positions[record.message_id] > last_deleted
                        ]
            if exclude_near_nontext:
                if any(
                    term in record.text
                    for record in (*preceding, *partner_run, *owner_run)
                    for term in _MEDIA_DEICTIC_TERMS
                ):
                    counts["excluded_by_deictic"] += 1
                    continue
                first_text = (preceding[0] if preceding else first).text
                reaction = "".join(first_text.split()).rstrip(_REACTION_ENDINGS)
                if last_blocker is not None and reaction in _SHORT_MEDIA_REACTIONS:
                    counts["excluded_by_short_reaction"] += 1
                    continue
            privacy_flags: list[str] = []
            if veiled is not None:
                hits = veiled.scan("\n".join(
                    record.text for record in (*original_preceding, *partner_run, *owner_run)
                ))
                if hits.high:
                    continue
                privacy_flags = list(hits.suspect)
            owner_reply = [
                _redacted(record, redaction_terms, pseudonyms, pseudonym_pattern)
                for record in owner_run[:_MAX_TURNS]
            ]
            if any(len(line) > _OWNER_MAX_CHARS for line in owner_reply):
                continue
            partner_turns = [
                _redacted(record, redaction_terms, pseudonyms, pseudonym_pattern)[
                    :None if full_text else _PARTNER_MAX_CHARS
                ]
                for record in (partner_run if full_text else partner_run[-_MAX_TURNS:])
            ]
            context = [
                ("她：" if record.speaker_role is SpeakerRole.OTHER else "他：")  # noqa: RUF001
                + _redacted(record, redaction_terms, pseudonyms, pseudonym_pattern)[
                    :None if full_text else _PARTNER_MAX_CHARS
                ]
                for record in preceding
            ]
            if covered_message_ids is not None:
                snippet_id = _snippet_id(first.conversation_id, first.message_id)
                covered_message_ids[snippet_id] = frozenset(
                    record.message_id for record in original[
                        original_positions[first.message_id]:
                        original_positions[owner_run[-1].message_id] + 1
                    ]
                )
            snippets.append(
                PairedSnippet(
                    snippet_id=_snippet_id(first.conversation_id, first.message_id),
                    conversation_id=first.conversation_id,
                    started_at=first.timestamp,
                    partner_turns=partner_turns,
                    owner_reply=owner_reply,
                    source_fingerprint=fingerprint,
                    context_turns=context,
                    privacy_flags=privacy_flags,
                )
            )
    if media_counts is not None:
        media_counts.update(counts)
    return sorted(snippets, key=_sort_key)


def _bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _jaccard(left: str, right: str) -> float:
    left_bigrams, right_bigrams = _bigrams(left), _bigrams(right)
    union = left_bigrams | right_bigrams
    if not union:
        return 1.0 if left == right else 0.0
    return len(left_bigrams & right_bigrams) / len(union)


def split_holdout(
    snippets: Sequence[PairedSnippet],
    *,
    fraction: float = 0.1,
) -> tuple[list[PairedSnippet], list[PairedSnippet]]:
    ordered = sorted(snippets, key=_sort_key)
    cut = len(ordered) - math.ceil(len(ordered) * fraction)
    library, candidates = ordered[:cut], ordered[cut:]
    library_partners = ["\n".join(item.partner_turns) for item in library]
    holdout = [
        item
        for item in candidates
        if all(
            _jaccard("\n".join(item.partner_turns), partner) < _NEAR_DUPLICATE_JACCARD
            for partner in library_partners
        )
    ]
    return library, holdout


def write_snippets(path: Path, snippets: Sequence[PairedSnippet]) -> None:
    temporary_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(temporary_file.name)
    try:
        with temporary_file as file:
            for snippet in snippets:
                file.write(json.dumps(snippet.model_dump(mode="json"), ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_snippets(path: Path) -> list[PairedSnippet]:
    with path.open(encoding="utf-8") as file:
        return [PairedSnippet.model_validate_json(line) for line in file if line.strip()]


class ActiveSnippetReadError(RuntimeError):
    """The active library could not be read and filtered safely."""


def read_active_snippet_snapshot(
    library_path: Path, exclusions_path: Path,
) -> tuple[list[PairedSnippet], str]:
    """Read each source once, returning original active records and a byte revision."""
    read_failed = False
    try:
        library_bytes = library_path.read_bytes()
    except Exception:
        read_failed = True
    try:
        exclusion_bytes = exclusions_path.read_bytes()
    except FileNotFoundError:
        exclusion_bytes = None
    except Exception:
        read_failed = True
    if read_failed:
        raise ActiveSnippetReadError("active snippet source unavailable")

    try:
        snippets = [
            PairedSnippet.model_validate_json(line)
            for line in StringIO(library_bytes.decode("utf-8"), newline=None)
            if line.strip()
        ]
        if exclusion_bytes is None:
            return snippets, hashlib.sha256(library_bytes).hexdigest()

        saved = json.loads(exclusion_bytes.decode("utf-8"))
        if not isinstance(saved, dict) or saved.keys() != {"snippet_ids", "ranges"}:
            raise ValueError
        ids = saved["snippet_ids"]
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError
        deleted_ids = set(ids)
        if not isinstance(saved["ranges"], list):
            raise ValueError
        ranges: list[tuple[datetime, datetime]] = []
        for item in saved["ranges"]:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(not isinstance(endpoint, str) for endpoint in item)
            ):
                raise ValueError
            start, end = (datetime.fromisoformat(endpoint) for endpoint in item)
            if start.utcoffset() is None or end.utcoffset() is None or start > end:
                raise ValueError
            ranges.append((start, end))

        active = [
            item for item in snippets
            if item.snippet_id not in deleted_ids
            and not any(start <= item.started_at <= end for start, end in ranges)
        ]
        revision = hashlib.sha256(
            b"active-snippets-v1\0" + len(library_bytes).to_bytes(8, "big")
            + library_bytes + exclusion_bytes
        ).hexdigest()
        return active, revision
    except Exception:
        pass
    raise ActiveSnippetReadError("active snippet source unavailable")


def delete_snippets(
    paths: Sequence[Path],
    *,
    snippet_ids: frozenset[str] = frozenset(),
    started_between: tuple[datetime, datetime] | None = None,
) -> int:
    deleted = 0
    for path in paths:
        snippets = read_snippets(path)
        kept = [
            item
            for item in snippets
            if item.snippet_id not in snippet_ids
            and not (
                started_between is not None
                and started_between[0] <= item.started_at <= started_between[1]
            )
        ]
        deleted += len(snippets) - len(kept)
        write_snippets(path, kept)
    return deleted


def sample_for_audit(
    snippets: Sequence[PairedSnippet],
    *,
    size: int = 50,
    seed: int,
) -> list[PairedSnippet]:
    if len(snippets) < size:
        return list(snippets)
    return random.Random(seed).sample(list(snippets), size)
