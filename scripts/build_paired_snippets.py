"""Build private paired snippets locally; emit only status and counts.

Choose an explicit speaker mapping or infer it from import directions. Optional
entities preserve approved names and group aliases under fixed pseudonyms;
uncovered name/address terms still receive redaction placeholders. Optional
full-text output retains complete partner runs and untruncated context. Apply
the output directory's permanent deleted IDs/ranges before holdout splitting.
Their full raw message intervals block pairs and truncate neighbouring context.
Deletion coverage ignores media, veiled and risk exclusions; deleted_ids_matched
counts distinct saved IDs found in that relaxed candidate set.
Snippet and style-eligible counts describe the retained set before splitting.
Veiled exclusion counts compare eligible pairs before permanent deletions;
suspect counts describe retained, flagged snippets before splitting.
Media exclusions count the first matching rule before veiled filtering and
permanent deletions. Context truncations also count subsequently excluded pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Support the documented direct invocation without requiring PYTHONPATH.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from companion.contracts.chat import SpeakerRole
from companion.importer.models import (
    ConfirmedSpeakerMapping,
    ImportSeverity,
    MessageDirection,
    SourceChatRecord,
)
from companion.importer.speaker_mapping import apply_confirmed_mapping
from companion.importer.standard_json import StandardJsonAdapter
from companion.privacy import (
    RiskCategory,
    build_paired_snippets,
    load_entities,
    load_veiled_terms,
    sample_for_audit,
    split_holdout,
    write_snippets,
)
from companion.privacy.redaction import redact_text
from companion.runtime.settings import load_settings


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("invalid arguments")


def _read_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _term_group(payload: object) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("expected term object")
    for terms in payload.values():
        if (
            not isinstance(terms, list)
            or not terms
            or any(not isinstance(term, str) or not term.strip() for term in terms)
        ):
            raise ValueError("invalid terms")
    return payload


def _direction_mapping(records: list[SourceChatRecord]) -> ConfirmedSpeakerMapping:
    roles: dict[str, SpeakerRole] = {}
    for direction, role in ((MessageDirection.OUTBOUND, SpeakerRole.OWNER),
                            (MessageDirection.INBOUND, SpeakerRole.OTHER)):
        speakers = {record.source_speaker_id for record in records if record.direction is direction}
        if len(speakers) > 1:
            raise ValueError("direction has multiple speakers")
        roles.update(dict.fromkeys(speakers, role))
    return ConfirmedSpeakerMapping(roles=roles)


def _build(args: argparse.Namespace) -> dict[str, str | int]:
    batch = StandardJsonAdapter().parse(args.chat)
    if any(issue.severity is ImportSeverity.ERROR for issue in batch.issues):
        raise ValueError("import error")
    mapping = (_direction_mapping(batch.records) if args.direction_mapping
               else ConfirmedSpeakerMapping(roles=_read_object(args.mapping)))
    records = apply_confirmed_mapping(batch.records, mapping)
    terms = _read_object(args.terms)
    if terms.keys() - {"redaction_terms", "risk_terms"}:
        raise ValueError("unknown term group")
    redaction_terms = _term_group(terms.get("redaction_terms", {}))
    risk_terms = {
        RiskCategory(category): values
        for category, values in _term_group(terms.get("risk_terms", {})).items()
    }
    # Validate labels even when all source segments will be excluded.
    redact_text(None, private_terms=redaction_terms)
    veiled = load_veiled_terms(args.veiled) if args.veiled is not None else None
    build_options = dict(
        created_at=datetime.now(UTC),
        redaction_terms=redaction_terms,
        risk_terms=risk_terms,
        entities=load_entities(args.entities) if args.entities is not None else None,
        context_turns=args.context_turns,
        full_text=args.full_text,
        exclude_near_nontext=args.exclude_near_nontext,
    )
    deleted_ids: set[str] = set()
    ranges: list[tuple[datetime, datetime]] = []
    deleted_path = args.out_dir / "deleted_ids.json"
    if deleted_path.exists():
        saved = _read_object(deleted_path)
        deleted_ids = set(saved["snippet_ids"])
        ranges = [(datetime.fromisoformat(start), datetime.fromisoformat(end))
                  for start, end in saved["ranges"]]
    coverage: dict[str, frozenset[str]] = {}
    coverage_options = {**build_options, "exclude_near_nontext": False,
                        "excluded_risks": frozenset(), "veiled": None}
    coverage_candidates = build_paired_snippets(
        records, covered_message_ids=coverage, **coverage_options,
    )
    deleted_candidates = {
        item.snippet_id for item in coverage_candidates if item.snippet_id in deleted_ids
        or any(start <= item.started_at <= end for start, end in ranges)
    }
    blocked_message_ids = frozenset(
        message_id for snippet_id in deleted_candidates for message_id in coverage[snippet_id]
    )
    deleted_ids_matched = len(deleted_ids.intersection(coverage))
    # Preserve pre-deletion counts using the formal rules.
    media_counts: dict[str, int] = {}
    candidates = build_paired_snippets(records, media_counts=media_counts, **build_options)
    excluded_by_veiled = 0
    if veiled is not None:
        scanned = build_paired_snippets(records, veiled=veiled, **build_options)
        excluded_by_veiled = len(candidates) - len(scanned)
        candidates = scanned
    excluded_by_deleted = sum(item.snippet_id in deleted_candidates for item in candidates)
    snippets = build_paired_snippets(
        records, veiled=veiled, blocked_message_ids=blocked_message_ids, **build_options,
    )
    library, holdout = split_holdout(snippets, fraction=0.1)
    audit = sample_for_audit(library, size=50, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_snippets(args.out_dir / "library.jsonl", library)
    write_snippets(args.out_dir / "holdout.jsonl", holdout)
    write_snippets(args.out_dir / "audit_sample.jsonl", audit)
    return {
        "status": "ok",
        "records": len(records),
        "snippets": len(snippets),
        "library": len(library),
        "holdout": len(holdout),
        "audit": len(audit),
        "excluded_by_deleted": excluded_by_deleted,
        "blocked_messages": len(blocked_message_ids),
        "deleted_ids_matched": deleted_ids_matched,
        "style_eligible": sum(item.as_style_snippet() is not None for item in snippets),
        "excluded_by_veiled": excluded_by_veiled,
        "flagged_suspect": sum(bool(item.privacy_flags) for item in snippets),
        **media_counts,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _Parser()
        for option in ("chat", "terms", "out-dir"):
            parser.add_argument(f"--{option}", type=Path, required=True)
        mapping = parser.add_mutually_exclusive_group(required=True)
        mapping.add_argument("--mapping", type=Path)
        mapping.add_argument("--direction-mapping", action="store_true")
        parser.add_argument("--entities", type=Path)
        parser.add_argument("--veiled", type=Path)
        parser.add_argument("--context-turns", type=int, default=0)
        parser.add_argument("--full-text", action="store_true")
        parser.add_argument("--exclude-near-nontext", action="store_true")
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args(argv)
        private_root = load_settings(Path.cwd()).private_data_dir.resolve()
        args.out_dir = args.out_dir.resolve()
        if args.out_dir == private_root or not args.out_dir.is_relative_to(private_root):
            print('{"status":"out_dir_not_private"}')
            return 2
        result = _build(args)
    except Exception:
        # This CLI boundary must never disclose input values or exception details.
        print('{"status":"failed"}')
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
