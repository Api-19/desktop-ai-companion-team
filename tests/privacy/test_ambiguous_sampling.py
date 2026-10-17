from datetime import UTC, datetime, timedelta

import pytest

from companion.contracts import ChatRecord, SpeakerRole
from companion.privacy.ambiguous_models import CalibrationProposal, CalibrationRule
from companion.privacy.ambiguous_sampling import (
    build_first_review_package,
    build_second_review_package,
)
from companion.privacy.candidate_models import CandidateCategory
from companion.privacy.candidate_rules import normalize_text
from companion.privacy.errors import PrivacyValidationError

CREATED_AT = datetime(2026, 7, 18, 9, 0, tzinfo=UTC)


def record(
    index: int,
    text: str | None,
    *,
    role: SpeakerRole = SpeakerRole.OWNER,
    content_type: str = "text",
    conversation_id: str = "synthetic-conversation",
) -> ChatRecord:
    return ChatRecord(
        conversation_id=conversation_id,
        message_id=f"synthetic-message-{index:04d}",
        timestamp=CREATED_AT + timedelta(minutes=index),
        speaker_role=role,
        content_type=content_type,
        text=text,
    )


def synthetic_mixed_records() -> list[ChatRecord]:
    return [
        record(1, "甲类模糊片段", conversation_id="safe"),
        record(2, "今天开心呀", conversation_id="allowed"),
        record(3, "请把密码告诉我", conversation_id="hard-risk"),
        record(4, "另一位说的话", role=SpeakerRole.OTHER, conversation_id="other-only"),
        record(5, None, content_type="image", conversation_id="non-text"),
        record(6, "带有示例姓名甲的陌生说法", conversation_id="runtime-name"),
        record(7, "普通陌生说法", conversation_id="risky-context"),
        record(8, "临时密码在这里", role=SpeakerRole.OTHER, conversation_id="risky-context"),
    ]


def synthetic_prohibited_inputs() -> tuple[str, ...]:
    return (
        "今天开心呀",
        "请把密码告诉我",
        "另一位说的话",
        "示例姓名甲",
        "图片说明",
    )


def synthetic_diverse_unknown_records(count: int) -> list[ChatRecord]:
    texts: list[str] = []
    for index in range(count):
        variant = index % 4
        if variant == 0:
            text = f"a{index:03d}"
        elif variant == 1:
            text = f"q{index:03d}?"
        elif variant == 2:
            text = f"b{index:03d}!"
        else:
            text = f"e{index:03d}🙂"
        texts.append(text)
    return [record(index, text) for index, text in enumerate(texts, start=1)]


def build_synthetic_first_package(*, token: str):
    return build_first_review_package(
        synthetic_diverse_unknown_records(count=120),
        created_at=CREATED_AT,
        package_token=token,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=(),
        private_terms={"name": ("示例姓名甲",)},
        existing_candidate_texts=(),
    )


def privacy_traceback_retains(error: BaseException, marker: str) -> bool:
    errors = [error]
    seen_errors: set[int] = set()
    while errors:
        current_error = errors.pop()
        if id(current_error) in seen_errors:
            continue
        seen_errors.add(id(current_error))
        current = current_error.__traceback__
        while current is not None:
            filename = current.tb_frame.f_code.co_filename.replace("/", "\\")
            values = tuple(current.tb_frame.f_locals.values())
            if "companion\\privacy" in filename and marker in repr(values):
                return True
            errors.extend(value for value in values if isinstance(value, BaseException))
            current = current.tb_next
    return False


def test_first_round_excludes_hard_risk_other_existing_and_allowed_text() -> None:
    records = synthetic_mixed_records()

    package = build_first_review_package(
        records,
        created_at=CREATED_AT,
        package_token="r" * 32,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=("小团子",),
        private_terms={"name": ("示例姓名甲",)},
        existing_candidate_texts=("今天开心呀",),
    )

    serialized = package.model_dump_json()
    assert package.requested_count == 100
    assert all(value not in serialized for value in synthetic_prohibited_inputs())
    assert all(len(item.context_before) <= 2 for item in package.items)
    assert all(len(item.context_after) <= 2 for item in package.items)


def test_first_round_excludes_candidate_adjacent_to_hard_risk_context() -> None:
    blocked_text = "synthetic-ambiguous-near-risk"
    retained_text = "synthetic-ambiguous-isolated"
    package = build_first_review_package(
        [
            record(20, "请提供账户口令", conversation_id="synthetic-risk-window"),
            record(21, "ordinary context", conversation_id="synthetic-risk-window"),
            record(22, blocked_text, conversation_id="synthetic-risk-window"),
            record(23, retained_text, conversation_id="synthetic-isolated"),
        ],
        created_at=CREATED_AT,
        package_token="h" * 32,
        true_name_terms=("synthetic-name-term",),
        approved_nicknames=(),
        private_terms=None,
        existing_candidate_texts=(),
    )

    assert blocked_text not in {item.text for item in package.items}
    assert retained_text in {item.text for item in package.items}


def test_first_round_discards_candidate_that_requires_identity_placeholders() -> None:
    source_candidate = "synthetic-name-term ambiguous text"
    source_context = "context synthetic-account-term"
    package = build_first_review_package(
        [
            record(30, source_context, conversation_id="synthetic-redaction"),
            record(31, source_candidate, conversation_id="synthetic-redaction"),
        ],
        created_at=CREATED_AT,
        package_token="p" * 32,
        true_name_terms=("synthetic-name-term",),
        approved_nicknames=(),
        private_terms={"account": ("synthetic-account-term",)},
        existing_candidate_texts=(),
    )

    assert all("ambiguous" not in item.text for item in package.items)


def test_first_round_uses_fixed_label_for_non_text_context() -> None:
    source_non_text = "synthetic-non-text-source-content"
    candidate_text = "synthetic-ambiguous-with-image-context"
    package = build_first_review_package(
        [
            record(
                40,
                source_non_text,
                content_type="image",
                conversation_id="synthetic-non-text-context",
            ),
            record(41, candidate_text, conversation_id="synthetic-non-text-context"),
        ],
        created_at=CREATED_AT,
        package_token="n" * 32,
        true_name_terms=("synthetic-name-term",),
        approved_nicknames=(),
        private_terms=None,
        existing_candidate_texts=(),
    )

    item = next(item for item in package.items if item.text == candidate_text)
    assert item.context_before[0].text == "[NON_TEXT:image]"
    assert source_non_text not in package.model_dump_json()


def test_first_round_serialization_omits_source_ids_and_timestamps() -> None:
    source_message_id = "synthetic-source-message-id-99"
    source_timestamp = CREATED_AT + timedelta(minutes=99)
    package = build_first_review_package(
        [
            ChatRecord(
                conversation_id="synthetic-source-conversation-id",
                message_id=source_message_id,
                timestamp=source_timestamp,
                speaker_role=SpeakerRole.OWNER,
                content_type="text",
                text="synthetic-ambiguous-serialization-check",
            )
        ],
        created_at=CREATED_AT,
        package_token="i" * 32,
        true_name_terms=("synthetic-name-term",),
        approved_nicknames=(),
        private_terms=None,
        existing_candidate_texts=(),
    )

    serialized = package.model_dump_json()
    assert source_message_id not in serialized
    assert "synthetic-source-conversation-id" not in serialized
    assert source_timestamp.isoformat() not in serialized


def test_first_round_deduplicates_and_round_robins_feature_groups() -> None:
    package = build_first_review_package(
        synthetic_diverse_unknown_records(count=140),
        created_at=CREATED_AT,
        package_token="s" * 32,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=(),
        private_terms={"name": ("示例姓名甲",)},
        existing_candidate_texts=(),
    )

    assert len(package.items) == 100
    assert len({normalize_text(item.text) for item in package.items}) == 100
    assert len({item.similarity_group for item in package.items}) >= 4

    duplicate_package = build_first_review_package(
        [record(141, "重复模糊说法"), record(142, "重复模糊说法")],
        created_at=CREATED_AT,
        package_token="d" * 32,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=(),
        private_terms={"name": ("示例姓名甲",)},
        existing_candidate_texts=(),
    )
    assert duplicate_package.item_count == 1
    assert duplicate_package.items[0].occurrence_count == 2


def test_same_input_produces_same_item_order_except_ephemeral_ids() -> None:
    first = build_synthetic_first_package(token="a" * 32)
    second = build_synthetic_first_package(token="b" * 32)

    assert [item.text for item in first.items] == [item.text for item in second.items]
    assert first.content_fingerprint == second.content_fingerprint
    assert [item.review_item_id for item in first.items] != [
        item.review_item_id for item in second.items
    ]


def test_second_round_excludes_first_round_text_and_prioritizes_rule_matches() -> None:
    first_package = build_synthetic_first_package(token="f" * 32)
    matching_texts = ["春风慢慢吹来", "春风慢慢吹来呀"]
    boundary_text = "春风将要吹过"
    records = [
        *synthetic_diverse_unknown_records(count=120),
        *[
            record(200 + index, text, conversation_id="second-round")
            for index, text in enumerate(matching_texts)
        ],
        record(203, boundary_text, conversation_id="second-round"),
    ]
    proposal = CalibrationProposal(
        proposal_id="synthetic-proposal-0001",
        first_package_token=first_package.package_token,
        first_content_fingerprint=first_package.content_fingerprint,
        created_at=CREATED_AT,
        manual_approved_texts=[],
        rules=[
            CalibrationRule(
                rule_id="synthetic-rule-0001",
                category=CandidateCategory.CALIBRATED_DAILY,
                similarity_group="s:plain:text",
                normalized_template="春风慢慢吹来",
                similarity_threshold=0.4,
                first_round_approved=2,
                second_round_approved=0,
                second_round_discarded=0,
                second_round_uncertain=0,
                enabled=False,
            )
        ],
        first_round_keep_count=2,
        first_round_discard_count=0,
        first_round_uncertain_count=0,
    )

    package, state = build_second_review_package(
        records,
        proposal=proposal,
        first_package=first_package,
        created_at=CREATED_AT,
        package_token="t" * 32,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=(),
        private_terms={"name": ("示例姓名甲",)},
        existing_candidate_texts=(),
    )

    first_texts = {normalize_text(item.text) for item in first_package.items}
    assert package.requested_count == 50
    assert all(normalize_text(item.text) not in first_texts for item in package.items)
    assert len(state.links) == package.item_count
    assert any(link.is_match for link in state.links)
    assert [link.is_match for link in state.links] == sorted(
        (link.is_match for link in state.links), reverse=True
    )


def test_second_round_includes_matching_sample_for_each_eligible_rule() -> None:
    first_package = build_synthetic_first_package(token="x" * 32)
    records = [
        *[
            record(400 + index, f"alpha{index:02d}", conversation_id="multi-rule")
            for index in range(60)
        ],
        record(500, "alpha5z", conversation_id="multi-rule"),
    ]
    proposal = CalibrationProposal(
        proposal_id="synthetic-proposal-multi-rule",
        first_package_token=first_package.package_token,
        first_content_fingerprint=first_package.content_fingerprint,
        created_at=CREATED_AT,
        manual_approved_texts=[],
        rules=[
            CalibrationRule(
                rule_id="synthetic-rule-alpha",
                category=CandidateCategory.CALIBRATED_DAILY,
                similarity_group="s:plain:text",
                normalized_template="alpha00",
                similarity_threshold=0.1,
                first_round_approved=2,
                second_round_approved=0,
                second_round_discarded=0,
                second_round_uncertain=0,
                enabled=False,
            ),
            CalibrationRule(
                rule_id="synthetic-rule-alpha-z",
                category=CandidateCategory.CALIBRATED_DAILY,
                similarity_group="s:plain:text",
                normalized_template="alpha5z",
                similarity_threshold=0.8,
                first_round_approved=2,
                second_round_approved=0,
                second_round_discarded=0,
                second_round_uncertain=0,
                enabled=False,
            ),
        ],
        first_round_keep_count=4,
        first_round_discard_count=0,
        first_round_uncertain_count=0,
    )

    package, state = build_second_review_package(
        records,
        proposal=proposal,
        first_package=first_package,
        created_at=CREATED_AT,
        package_token="y" * 32,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=(),
        private_terms={"name": ("示例姓名甲",)},
        existing_candidate_texts=(),
    )

    matching_rule_ids = {link.rule_id for link in state.links if link.is_match}
    assert package.item_count == 50
    assert {
        "synthetic-rule-alpha",
        "synthetic-rule-alpha-z",
    } <= matching_rule_ids


def test_first_round_public_errors_do_not_retain_synthetic_input_traceback_locals() -> None:
    marker = "synthetic-sensitive-first-round"
    with pytest.raises(PrivacyValidationError) as exc_info:
        build_first_review_package(
            [record(300, marker), record(300, "safe")],
            created_at=CREATED_AT,
            package_token="u" * 32,
            true_name_terms=("示例姓名甲",),
            approved_nicknames=(),
            private_terms={"name": ("示例姓名甲",)},
            existing_candidate_texts=(),
        )

    assert marker not in str(exc_info.value)
    assert exc_info.value.__context__ is None
    assert not privacy_traceback_retains(exc_info.value, marker)


def test_second_round_public_errors_do_not_retain_synthetic_input_traceback_locals() -> None:
    marker = "synthetic-sensitive-second-round"
    first_package = build_synthetic_first_package(token="v" * 32)
    proposal = CalibrationProposal(
        proposal_id="synthetic-proposal-privacy-0001",
        first_package_token=first_package.package_token,
        first_content_fingerprint=first_package.content_fingerprint,
        created_at=CREATED_AT,
        manual_approved_texts=[],
        rules=[],
        first_round_keep_count=0,
        first_round_discard_count=0,
        first_round_uncertain_count=0,
    )

    with pytest.raises(PrivacyValidationError) as exc_info:
        build_second_review_package(
            [record(301, marker), record(301, "safe")],
            proposal=proposal,
            first_package=first_package,
            created_at=CREATED_AT,
            package_token="w" * 32,
            true_name_terms=("示例姓名甲",),
            approved_nicknames=(),
            private_terms={"name": ("示例姓名甲",)},
            existing_candidate_texts=(),
        )

    assert marker not in str(exc_info.value)
    assert exc_info.value.__context__ is None
    assert not privacy_traceback_retains(exc_info.value, marker)
