# ruff: noqa: RUF001
import json
from pathlib import Path

import pytest

from companion.privacy.veiled import VeiledHits, VeiledLexicon, load_veiled_terms

LEXICON_PATH = Path(__file__).resolve().parents[2] / "config/privacy/veiled_terms.json"


@pytest.fixture
def lexicon() -> VeiledLexicon:
    return load_veiled_terms(LEXICON_PATH)


@pytest.mark.parametrize("text", [
    "做爱", "做 爱", "做\t.*- _~·\n爱", "做\u200b爱", "做\u200c\u200d\u2060\ufeff爱",
    "ZUOAI", "ＺＵＯＡＩ", "z u o a i", "f s n", "ＦＳＮ", "法*死*你",
])
def test_normalization_and_aliases_report_canonical_term(lexicon, text):
    expected = ("sexual:法死你" if "法" in text or "s" in text.lower() or "Ｓ" in text
                else "sexual:做爱")
    assert lexicon.scan(text).high == (expected,)


def test_hits_are_unique_sorted_and_keep_both_tiers(lexicon):
    assert lexicon.scan("法死你 做爱 做 爱 胸 袜子 胸") == VeiledHits(
        high=("sexual:做爱", "sexual:法死你"),
        suspect=("body:胸", "body:袜子"),
    )


def test_empty_and_unmatched_text(lexicon):
    assert lexicon.scan("") == VeiledHits(high=(), suspect=())
    assert lexicon.scan("今天阳光很好") == VeiledHits(high=(), suspect=())


def test_emoji_combinations_match_literal_substrings(lexicon):
    assert lexicon.scan("看看🍆🍑呀").high == ("emoji_combo:🍆🍑",)
    hits = lexicon.scan("🍆 * 🍑")
    assert hits.high == ()
    assert hits.suspect == ("emoji:🍆", "emoji:🍑")


def test_separators_can_be_disabled_and_regex_characters_are_literal(tmp_path):
    path = tmp_path / "veiled.json"
    payload = {"version": 1, "high": {"synthetic": ["a+b", "做爱"]},
               "suspect": {}, "variants": {"allow_separators": False,
               "pinyin_initials": {"做爱": ["za"]}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    lexicon = load_veiled_terms(path)
    assert lexicon.scan("做 爱 z a ab aaab a + b").high == ()
    assert lexicon.scan("A+B ZA").high == ("synthetic:a+b", "synthetic:做爱")
    payload["variants"]["allow_separators"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_veiled_terms(path).scan("a + b z a").high == (
        "synthetic:a+b", "synthetic:做爱",
    )


def test_single_character_does_not_absorb_neighboring_characters(lexicon):
    assert lexicon.scan("胸 * 袜 子").suspect == ("body:胸", "body:袜子")
    assert lexicon.scan("法，死，你").high == ()


@pytest.mark.parametrize("change", [
    {"version": 2}, {"high": {"sexual": [""]}}, {"suspect": {"body": "胸"}},
    {"variants": {"pinyin_initials": {"unknown": ["xx"]}}},
])
def test_invalid_lexicon_is_rejected_at_file_boundary(tmp_path, change):
    payload = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    payload.update(change)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_veiled_terms(path)


def test_every_configured_term_and_alias_has_its_category_and_tier(lexicon):
    payload = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    aliases = payload["variants"]["pinyin_initials"]
    for tier in ("high", "suspect"):
        for category, terms in payload[tier].items():
            for term in terms:
                for text in (term, *aliases.get(term, [])):
                    hits = lexicon.scan(text)
                    assert f"{category}:{term}" in getattr(hits, tier)


def test_review_round_one_terms_hit_expected_tiers(lexicon):
    assert "sexual:当机长" in lexicon.scan("想当机长").high
    assert "ambiguous_sexual:玩我" in lexicon.scan("来玩我呀").suspect
    assert "reproductive:头胎" in lexicon.scan("头胎是男孩").high
