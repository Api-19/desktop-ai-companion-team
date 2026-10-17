"""Synthetic tests for local paired-snippet evidence."""

import hashlib
import random
from datetime import datetime, timedelta

import pytest

from companion.contracts.style import AddresseeRegister, StyleEvidence, StyleSnippet
from companion.style import SnippetIndex, SnippetStyleAdapter

DAY = datetime(2026, 10, 6)


def snippet(partner: str, reply: str = "回复") -> StyleSnippet:
    return StyleSnippet(partner_turns=[partner], owner_reply=[reply])


def library(size: int = 60) -> list[StyleSnippet]:
    return [snippet(f"合成语境{i}", "回" * (i + 1)) for i in range(size)]


def adapter(snippets, *, day=DAY):
    return SnippetStyleAdapter(
        snippets=snippets, partner_user_id="partner", style_version="v1", clock=lambda: day
    )


def used(evidence):
    return evidence.similar + evidence.general


def test_bm25_ranking_zero_score_k_exclude_and_ties():
    index = SnippetIndex(
        [snippet("甲乙丁"), snippet("甲乙丙"), snippet("戊己庚"), snippet("甲乙丙")]
    )
    assert index.search("甲乙丙", k=6) == [1, 3, 0]
    assert index.search("甲乙丙", k=1) == [1]
    assert index.search("甲乙丙", k=6, exclude={1, 0}) == [3]
    assert index.search("甲乙丙", k=0) == []
    assert index.search("完全无交集", k=6) == []


def test_single_character_and_whitespace_tokenization():
    index = SnippetIndex([snippet("甲"), snippet("乙"), snippet("甲乙")])
    assert index.search(" \t甲\n", k=6) == [0]
    assert index.search("甲 \n乙", k=6) == [2]
    assert index.search(" \t\n", k=6) == []


def test_partner_turns_are_joined_for_indexing():
    index = SnippetIndex([StyleSnippet(partner_turns=["甲", "乙"], owner_reply=["回"])])
    assert index.search("甲乙", k=6) == [0]


def test_intimate_limits_disjointness_and_contract_validation():
    evidence = adapter(library()).evidence_for("partner", "合成语境", [])
    assert evidence.addressee_register is AddresseeRegister.INTIMATE
    assert evidence.style_version == "v1"
    assert len(evidence.similar) == len(evidence.general) == 6
    assert all(item not in evidence.general for item in evidence.similar)
    assert StyleEvidence.model_validate(evidence.model_dump()) == evidence


def test_duplicate_content_is_removed_before_indexing_and_sampling():
    first = snippet("晚安", "晚安")
    other = snippet("吃了吗", "吃了")
    subject = SnippetStyleAdapter(
        snippets=[first, snippet("晚安", "晚安"), other],
        partner_user_id="gf",
        style_version="v1",
        clock=lambda: DAY,
    )
    evidence = subject.evidence_for("gf", "晚安", [])
    assert evidence.similar == [first]
    assert evidence.similar[0] is first
    assert evidence.general == [other]
    selected = used(evidence)
    assert all(item not in selected[:index] for index, item in enumerate(selected))
    assert StyleEvidence.model_validate(evidence.model_dump()) == evidence
    for _ in range(3):
        assert used(subject.evidence_for("gf", "晚安", [])) == []
    assert subject.evidence_for("gf", "晚安", []) == evidence


def test_content_deduplication_preserves_first_occurrence_order_and_turn_boundaries():
    first = snippet("甲乙", "第一")
    second = snippet("甲乙", "第二")
    split = StyleSnippet(partner_turns=["甲", "乙"], owner_reply=["第一"])
    subject = adapter([first, second, snippet("甲乙", "第一"), split])
    evidence = subject.evidence_for("partner", "甲乙", [])
    assert evidence.similar == [first, second, split]
    assert evidence.similar[0] is first
    assert evidence.general == []


def test_query_uses_only_last_two_context_entries_and_text():
    snippets = [snippet("甲乙"), snippet("丙丁"), snippet("戊己"), snippet("庚辛")]
    evidence = adapter(snippets).evidence_for("partner", "庚辛", ["甲乙", "丙丁", "戊己"])
    assert evidence.similar == snippets[1:]


def test_neutral_is_empty_and_does_not_change_partner_history():
    snippets = library()
    actual, control = adapter(snippets), adapter(snippets)
    assert actual.evidence_for("partner", "合成语境", []) == control.evidence_for(
        "partner", "合成语境", []
    )
    for user_id in ("owner", "visitor", "another"):
        evidence = actual.evidence_for(user_id, "合成语境", [])
        assert evidence.addressee_register is AddresseeRegister.NEUTRAL
        assert evidence.similar == evidence.general == []
    assert actual.evidence_for("partner", "合成语境", []) == control.evidence_for(
        "partner", "合成语境", []
    )


def test_daily_seed_reproducible_and_changes_general_next_day():
    snippets = library()
    first = adapter(snippets).evidence_for("partner", "无匹配", [])
    assert first == adapter(snippets).evidence_for("partner", "无匹配", [])
    next_day = adapter(snippets, day=DAY + timedelta(days=1)).evidence_for("partner", "无匹配", [])
    assert first.general != next_day.general


@pytest.mark.parametrize("size", [1, 2, 4, 8, 10, 60])
def test_length_strata_remainder_in_longest_and_exact_seed(size):
    # Reversed input proves the strata are built by owner reply length.
    snippets = list(reversed(library(size)))
    evidence = adapter(snippets).evidence_for("partner", "无匹配", [])
    ordered = sorted(snippets, key=lambda item: len("".join(item.owner_reply)))
    width = size // 3
    strata = [ordered[:width], ordered[width : 2 * width], ordered[2 * width :]]
    seed = int(hashlib.sha256(f"{DAY.date().isoformat()}\x1f无匹配".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    expected = []
    for group in strata:
        expected.extend(rng.sample(group, 2) if len(group) >= 2 else group)
    assert evidence.similar == []
    assert evidence.general == expected


def test_default_evidence_matches_legacy_algorithm_across_requests():
    snippets = list(reversed(library()))
    subject = adapter(snippets)
    index = SnippetIndex(snippets)
    history = []
    for text, context in [("无匹配", []), ("合成语境", ["旧", "合成语境1", "新"])] * 4:
        excluded = set().union(*history[-3:])
        similar = index.search("\n".join([*context[-2:], text]), k=6, exclude=excluded)
        unavailable = excluded | set(similar)
        candidates = sorted(
            (i for i in range(len(snippets)) if i not in unavailable),
            key=lambda i: len("".join(snippets[i].owner_reply)),
        )
        width = len(candidates) // 3
        strata = [candidates[:width], candidates[width : 2 * width], candidates[2 * width :]]
        seed_text = f"{DAY.date().isoformat()}\x1f{text}"
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        general = []
        for group in strata:
            general.extend(rng.sample(group, 2) if len(group) >= 2 else group)
        expected = StyleEvidence(
            style_version="v1",
            addressee_register=AddresseeRegister.INTIMATE,
            similar=[snippets[i] for i in similar],
            general=[snippets[i] for i in general],
        )
        actual = subject.evidence_for("partner", text, context)
        assert actual == expected
        assert actual.model_dump_json() == expected.model_dump_json()
        history.append(set(similar + general))


def line_library(long_count=5):
    # Interleaved line counts and decreasing lengths distinguish index order from length order.
    return [
        StyleSnippet(
            partner_turns=[f"合成语境{i}-{lines}"],
            owner_reply=["回" * (10 - i)] * lines,
        )
        for i in range(5)
        for lines in (3, 1, 2)
        if lines != 3 or i < long_count
    ]


@pytest.mark.parametrize("long_count", [5, 1, 0])
def test_line_strata_counts_order_exact_seed_and_determinism(long_count):
    snippets = line_library(long_count)

    def new_adapter():
        return SnippetStyleAdapter(
            snippets=snippets,
            partner_user_id="partner",
            style_version="v1",
            clock=lambda: DAY,
            general_strata="lines",
        )

    evidence = new_adapter().evidence_for("partner", "无匹配", [])
    assert evidence.similar == []
    assert [len(item.owner_reply) for item in evidence.general] == [1, 1, 2, 2] + [
        3
    ] * min(long_count, 2)
    seed = int(hashlib.sha256(f"{DAY.date().isoformat()}\x1f无匹配".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    expected = []
    for lines in (1, 2, 3):
        group = [item for item in snippets if len(item.owner_reply) == lines]
        expected.extend(rng.sample(group, 2) if len(group) >= 2 else group)
    assert evidence.general == expected
    assert evidence == new_adapter().evidence_for("partner", "无匹配", [])


def test_line_strata_preserves_similar_exclusion_history_and_neutral():
    snippets = line_library()[:12]
    subject = SnippetStyleAdapter(
        snippets=snippets,
        partner_user_id="partner",
        style_version="v1",
        clock=lambda: DAY,
        general_strata="lines",
    )
    first = subject.evidence_for("partner", "合成语境", [])
    assert len(used(first)) == 12
    assert first.similar == [snippets[i] for i in SnippetIndex(snippets).search("合成语境", k=6)]
    assert all(item not in first.general for item in first.similar)
    assert subject.evidence_for("visitor", "合成语境", []) == StyleEvidence(
        style_version="v1", addressee_register=AddresseeRegister.NEUTRAL
    )
    history = [first]
    for _ in range(3):
        evidence = subject.evidence_for("partner", "合成语境", [])
        assert all(item not in used(previous) for previous in history for item in used(evidence))
        history.append(evidence)
    assert subject.evidence_for("partner", "合成语境", []) == first


def test_invalid_general_strata():
    with pytest.raises(ValueError, match="general_strata"):
        SnippetStyleAdapter(
            snippets=[],
            partner_user_id="partner",
            style_version="v1",
            clock=lambda: DAY,
            general_strata="x",
        )


def test_recent_three_calls_exclude_both_kinds_and_expire_after_three_calls():
    # One full batch: calls 2-4 exclude call 1; call 5 may reuse it.
    snippets = library(12)
    subject = adapter(snippets)
    first = subject.evidence_for("partner", "合成语境", [])
    assert len(used(first)) == 12
    for _ in range(3):
        assert used(subject.evidence_for("partner", "合成语境", [])) == []
    assert subject.evidence_for("partner", "合成语境", []) == first


def test_rolling_history_with_multiple_nonempty_batches():
    subject = adapter(library())
    history = []
    for _ in range(7):
        evidence = subject.evidence_for("partner", "合成语境", [])
        assert all(
            item not in used(previous) for previous in history[-3:] for item in used(evidence)
        )
        history.append(evidence)


def test_empty_library():
    subject = adapter([])
    assert SnippetIndex([]).search("甲乙", k=6) == []
    for user_id in ("partner", "other"):
        assert used(subject.evidence_for(user_id, "甲乙", [])) == []


class StringSubclass(str):
    pass


@pytest.mark.parametrize("value", [None, 1, StringSubclass("partner")])
def test_partner_id_requires_exact_string(value):
    with pytest.raises(TypeError):
        SnippetStyleAdapter(
            snippets=[], partner_user_id=value, style_version="v1", clock=lambda: DAY
        )


@pytest.mark.parametrize("value", ["", " \t\n", "x" * 121])
def test_partner_id_rejects_invalid_string(value):
    with pytest.raises(ValueError):
        SnippetStyleAdapter(
            snippets=[], partner_user_id=value, style_version="v1", clock=lambda: DAY
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a", True),
        ("x" * 120, True),
        ("", False),
        (" ", False),
        ("x" * 121, False),
        (None, False),
        (StringSubclass("a"), False),
    ],
)
def test_accepts_all_valid_users(value, expected):
    assert adapter([]).accepts_user(value) is expected
