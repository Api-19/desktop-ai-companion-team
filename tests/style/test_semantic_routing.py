import pytest

from companion.privacy import CandidateCategory
from companion.style.semantic_routing import derive_calibrated_category


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("早安呀", CandidateCategory.GREETING),
        ("谢谢你", CandidateCategory.COURTESY),
        ("出门记得带伞", CandidateCategory.WEATHER),
        ("我去洗碗了", CandidateCategory.HOUSEHOLD),
        ("快递到了", CandidateCategory.SHOPPING),
        ("我下课了", CandidateCategory.WORK_STUDY),
    ],
)
def test_routes_one_high_confidence_category(text, expected):
    assert derive_calibrated_category(text) is expected


def test_unmatched_text_stays_general():
    assert derive_calibrated_category("合成普通日常") is None


def test_multi_category_text_stays_general():
    assert derive_calibrated_category("下课去拿快递") is None
