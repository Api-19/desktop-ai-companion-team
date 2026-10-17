"""Synthetic veiled variants exercise the real paired-snippet boundary."""

# ruff: noqa: RUF001
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from companion.contracts import ChatRecord, SpeakerRole
from companion.privacy import build_paired_snippets
from companion.privacy.veiled import load_veiled_terms

HIGH_VARIANTS = [
    ("做爱", "sexual:做爱"), ("做 爱", "sexual:做爱"),
    ("做.*- _~·爱", "sexual:做爱"), ("做\u200b爱", "sexual:做爱"),
    ("ZUOAI", "sexual:做爱"), ("ＺＵＯＡＩ", "sexual:做爱"),
    ("z u o a i", "sexual:做爱"), ("法死你", "sexual:法死你"),
    ("法*死*你", "sexual:法死你"), ("f s n", "sexual:法死你"),
    ("ＦＳＮ", "sexual:法死你"), ("Ｆ．Ｓ．Ｎ", "sexual:法死你"),
    ("想Ｃ", "sexual:想c"), ("G 点", "sexual:g点"),
    ("验_孕", "reproductive:验孕"), ("姨妈还没来", "reproductive:姨妈还没来"),
    ("🍆🍑", "emoji_combo:🍆🍑"), ("🍑🍆", "emoji_combo:🍑🍆"),
    ("🍆💦", "emoji_combo:🍆💦"), ("👅💦", "emoji_combo:👅💦"),
]
SUSPECT_VARIANTS = [
    ("上床", "ambiguous_sexual:上床"), ("上 床", "ambiguous_sexual:上床"),
    ("湿~了", "ambiguous_sexual:湿了"), ("硬_了", "ambiguous_sexual:硬了"),
    ("袜 子", "body:袜子"), ("内*衣", "body:内衣"), ("胸", "body:胸"),
    ("ＮＭＳＬ", "vulgar_abbr:nmsl"), ("N m S l", "vulgar_abbr:nmsl"),
    ("ＤＯ", "intimacy:do"), ("E-X", "third_party_private:ex"),
    ("月\u200d经", "menstrual:月经"), ("🍆", "emoji:🍆"), ("🥵", "emoji:🥵"),
]
NORMAL_SENTENCES = [
    "我上床睡觉了", "衣服湿了", "没办法你自己看", "面包硬了", "飞机起飞了",
    "我去买袜子", "今晚早点睡", "今天阳光很好", "一起吃面包吧", "我去图书馆看书",
    "新衣服很合身", "我想办法解决", "花瓶里的花开了",
]


@pytest.fixture
def lexicon():
    path = Path(__file__).resolve().parents[2] / "config/privacy/veiled_terms.json"
    return load_veiled_terms(path)


def pair(text, side):
    start = datetime(2026, 10, 7, tzinfo=UTC)
    return [
        ChatRecord(conversation_id="synthetic-veiled", message_id=f"synthetic-{i}",
                   timestamp=start + timedelta(minutes=i), speaker_role=role,
                   content_type="text", text=text if role == side else "合成普通回复")
        for i, role in enumerate((SpeakerRole.OTHER, SpeakerRole.OWNER))
    ]


@pytest.mark.parametrize(("text", "flag"), HIGH_VARIANTS)
@pytest.mark.parametrize("side", [SpeakerRole.OTHER, SpeakerRole.OWNER])
def test_high_variants_are_all_excluded(lexicon, text, flag, side):
    assert flag in lexicon.scan(text).high
    records = pair(text, side)
    # Disable the older risk exclusion to prove this detector excludes every bait.
    options = {"created_at": records[0].timestamp, "excluded_risks": frozenset()}
    assert len(build_paired_snippets(records, **options)) == 1
    assert build_paired_snippets(records, veiled=lexicon, **options) == []


@pytest.mark.parametrize(("text", "flag"), SUSPECT_VARIANTS)
@pytest.mark.parametrize("side", [SpeakerRole.OTHER, SpeakerRole.OWNER])
def test_suspect_variants_are_all_flagged(lexicon, text, flag, side):
    records = pair(text, side)
    snippets = build_paired_snippets(records, created_at=records[0].timestamp, veiled=lexicon)
    assert len(snippets) == 1
    assert flag in snippets[0].privacy_flags
    assert snippets[0].privacy_flags == sorted(set(snippets[0].privacy_flags))


@pytest.mark.parametrize("text", NORMAL_SENTENCES)
def test_normal_sentences_are_never_high_excluded(lexicon, text):
    records = pair(text, SpeakerRole.OTHER)
    assert lexicon.scan(text).high == ()
    assert len(build_paired_snippets(records, created_at=records[0].timestamp, veiled=lexicon)) == 1
