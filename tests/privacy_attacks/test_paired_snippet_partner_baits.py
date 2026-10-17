"""Synthetic partner-side privacy baits must never reach paired snippet output."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from companion.contracts import ChatRecord, SpeakerRole
from companion.privacy import build_paired_snippets

BASE_TIME = datetime(2026, 10, 6, 8, 0, tzinfo=UTC)
CREATED_AT = datetime(2026, 10, 6, 20, 0, tzinfo=UTC)
NAME_BAITS = ("司马合成甲", "欧阳合成乙")
ADDRESS_BAITS = ("合成市虚构区样例路17号", "测试省假想县演示街8弄")
ACCOUNT_BAITS = ("synthetic_acct_9031",)
STRUCTURED_BAITS = (
    "13912345678",
    "11010519900307123X",
    "bait.partner@example.test",
    "6222 0212 3456 7890 123",
    "192.168.31.207",
)
BAITS = NAME_BAITS + ADDRESS_BAITS + ACCOUNT_BAITS + STRUCTURED_BAITS
REDACTION_TERMS = {
    "name": list(NAME_BAITS),
    "address": list(ADDRESS_BAITS),
    "account": list(ACCOUNT_BAITS),
}


def _records() -> list[ChatRecord]:
    records: list[ChatRecord] = []
    for index, bait in enumerate(BAITS):
        start = BASE_TIME + timedelta(hours=index)
        records.extend(
            [
                ChatRecord(
                    conversation_id="synthetic-bait-conv",
                    message_id=f"bait-{index:02d}-partner",
                    timestamp=start,
                    speaker_role=SpeakerRole.OTHER,
                    content_type="text",
                    text=f"顺便说一下{bait} 你看着办",
                ),
                ChatRecord(
                    conversation_id="synthetic-bait-conv",
                    message_id=f"bait-{index:02d}-owner",
                    timestamp=start + timedelta(minutes=1),
                    speaker_role=SpeakerRole.OWNER,
                    content_type="text",
                    text="好的我知道了",
                ),
            ]
        )
    return records


def test_partner_side_baits_never_appear_in_paired_snippets() -> None:
    assert len(BAITS) == 10

    snippets = build_paired_snippets(
        _records(),
        created_at=CREATED_AT,
        redaction_terms=REDACTION_TERMS,
    )

    output = json.dumps([item.model_dump(mode="json") for item in snippets], ensure_ascii=False)
    assert sum(output.count(bait) for bait in BAITS) == 0
    partner_turns = [turn for item in snippets for turn in item.partner_turns]
    assert len(partner_turns) == len(NAME_BAITS + ADDRESS_BAITS + ACCOUNT_BAITS)
    assert sum("[NAME]" in turn for turn in partner_turns) == len(NAME_BAITS)
    assert sum("[ADDRESS]" in turn for turn in partner_turns) == len(ADDRESS_BAITS)
    assert sum("[ACCOUNT]" in turn for turn in partner_turns) == len(ACCOUNT_BAITS)
