"""Synthetic shared-term exclusions, including overlapping occurrences."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from test_shared_terms import _entry, _write_dictionary

from companion.dialogue.shared_terms import SharedTerm, load_shared_terms, match_shared_terms


@pytest.mark.parametrize(
    "text, hit",
    [("扭转乾坤", False), ("坤", True), ("一坤时", True), ("乾坤里那个坤", True)],
)
def test_exclude_within_matches_only_unexcluded_occurrences(text: str, hit: bool) -> None:
    term = SharedTerm.model_validate({**_entry("坤", "合成词义"), "exclude_within": ["乾坤"]})
    assert match_shared_terms([term], [text]) == ((term,) if hit else ())


@pytest.mark.parametrize(
    "term, excluded, text, hit",
    [
        ("坤", ["乾坤", "坤包"], "乾坤和坤包", False),
        ("坤", ["乾坤", "坤包"], "坤包和坤", True),
        ("坤", ["乾坤"], "乾坤乾坤", False),
        ("坤", ["坤"], "坤和坤", False),
        ("坤", ["坤坤"], "坤坤坤", False),
        ("aa", ["baa"], "baaa", True),
        ("坤", ["乾坤"], "合成无命中", False),
    ],
)
def test_exclusions_cover_all_occurrences_and_preserve_overlapping_term_hits(
    term: str, excluded: list[str], text: str, hit: bool
) -> None:
    entry = SharedTerm.model_validate({**_entry(term, "合成词义"), "exclude_within": excluded})
    assert match_shared_terms([entry], [text]) == ((entry,) if hit else ())


def test_exclusions_are_evaluated_separately_for_each_text() -> None:
    term = SharedTerm.model_validate({**_entry("坤", "合成词义"), "exclude_within": ["乾坤"]})
    assert match_shared_terms([term], ["乾坤", "坤"]) == (term,)
    assert match_shared_terms([term], ["乾坤", "扭转乾坤"]) == ()


def test_absent_exclusions_preserve_substring_matching() -> None:
    term = SharedTerm.model_validate(_entry("坤", "合成词义"))
    assert term.exclude_within == ()
    assert match_shared_terms([term], ["扭转乾坤"]) == (term,)


@pytest.mark.parametrize("excluded", ([""], ["合成无关词"], ["乾坤", ""], [1], "乾坤"))
def test_invalid_exclusions_are_rejected_by_entry_and_file(
    tmp_path: Path, excluded: object
) -> None:
    payload = {**_entry("坤", "合成词义"), "exclude_within": excluded}
    with pytest.raises(ValidationError):
        SharedTerm.model_validate(payload)
    path = _write_dictionary(tmp_path / "terms.json", {"version": 1, "entries": [payload]})
    with pytest.raises(ValueError, match="shared terms configuration unavailable"):
        load_shared_terms(path)


def test_dictionary_json_list_loads_as_tuple_and_matches(tmp_path: Path) -> None:
    path = _write_dictionary(tmp_path / "terms.json", {"version": 1, "entries": [
        {**_entry("坤", "合成词义"), "exclude_within": ["乾坤", "坤包"]},
    ]})
    terms = load_shared_terms(path)
    assert terms[0].exclude_within == ("乾坤", "坤包")
    assert match_shared_terms(terms, ["扭转乾坤和坤包"]) == ()
    assert match_shared_terms(terms, ["乾坤里那个坤"]) == terms
