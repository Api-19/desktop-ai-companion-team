from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

import pytest

from companion.privacy import (
    CandidateCategory,
    CandidateSelectionAudit,
    PrivacyValidationError,
    SafeCandidatePackage,
    SafeStyleCandidate,
    validate_safe_candidate_runtime_material,
)

NOW = datetime(2026, 7, 27, tzinfo=UTC)
MALICIOUS_SENTINEL = "SYNTHETIC_MALICIOUS_RUNTIME_SENTINEL"


class _IterBomb(Sequence[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError(MALICIOUS_SENTINEL)

    def __getitem__(self, index: int) -> str:
        return "synthetic-name"

    def __len__(self) -> int:
        return 1


class _LenBomb(Sequence[str], Iterator[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        raise StopIteration

    def __getitem__(self, index: int) -> str:
        raise IndexError

    def __len__(self) -> int:
        raise RuntimeError(MALICIOUS_SENTINEL)


class _StripBomb(str):
    def strip(self, chars: str | None = None) -> str:
        raise RuntimeError(MALICIOUS_SENTINEL)


class _UnsafePrivacyFailureSequence(Sequence[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        raise PrivacyValidationError(
            "invalid_candidate_terms",
            MALICIOUS_SENTINEL,
        )

    def __getitem__(self, index: int) -> str:
        return "synthetic-name"

    def __len__(self) -> int:
        return 1


class _MutatedPrivacyFailureSequence(Sequence[str]):
    def __repr__(self) -> str:
        return MALICIOUS_SENTINEL

    def __iter__(self) -> Iterator[str]:
        error = PrivacyValidationError(
            "invalid_candidate_terms",
            "candidate terms are invalid",
        )
        error.code = []  # type: ignore[assignment]
        error.args = (_StripBomb(MALICIOUS_SENTINEL),)
        raise error

    def __getitem__(self, index: int) -> str:
        raise IndexError(index)

    def __len__(self) -> int:
        return 1


class _ModelDumpBomb(SafeCandidatePackage):
    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError(MALICIOUS_SENTINEL)


def _package(
    text: str = "晚安synthetic-nickname",
    category: CandidateCategory = CandidateCategory.GREETING,
) -> SafeCandidatePackage:
    return SafeCandidatePackage(
        created_at=NOW,
        candidates=[
            SafeStyleCandidate(
                text=text,
                category=category,
                scene_tags=[category.value],
            )
        ],
        audit=CandidateSelectionAudit(
            processed_at=NOW,
            input_count=1,
            owner_input_count=1,
            dropped_other_count=0,
            allowed_category_counts={category.value: 1},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=1,
            succeeded=True,
        ),
    )


def _assert_rejected_without_mutation(
    package: SafeCandidatePackage,
    *,
    true_name_terms: object,
    approved_nicknames: object,
) -> PrivacyValidationError:
    original = package.model_dump_json()

    with pytest.raises(PrivacyValidationError) as captured:
        validate_safe_candidate_runtime_material(
            package,
            true_name_terms=true_name_terms,  # type: ignore[arg-type]
            approved_nicknames=approved_nicknames,  # type: ignore[arg-type]
        )

    assert package.model_dump_json() == original
    return captured.value


def test_runtime_material_accepts_normalized_duplicate_terms_and_approved_nickname() -> None:
    package = _package()

    result = validate_safe_candidate_runtime_material(
        package,
        true_name_terms=(" synthetic-private-name ", "synthetic-private-name"),
        approved_nicknames=(" synthetic-nickname ", "synthetic-nickname"),
    )

    assert result is None


def test_runtime_material_rejects_unapproved_nickname_boundary() -> None:
    error = _assert_rejected_without_mutation(
        _package(),
        true_name_terms=("synthetic-private-name",),
        approved_nicknames=(),
    )

    assert error.code == "candidate_verification_failed"


def test_runtime_material_accepts_calibrated_daily_without_generation_profile() -> None:
    package = _package(
        "今天整理了一下桌面",
        CandidateCategory.CALIBRATED_DAILY,
    )

    result = validate_safe_candidate_runtime_material(
        package,
        true_name_terms=("synthetic-private-name",),
        approved_nicknames=(),
    )

    assert result is None


def test_runtime_material_rejects_calibrated_daily_that_is_now_a_normal_category() -> None:
    error = _assert_rejected_without_mutation(
        _package(
            "晚安synthetic-nickname",
            CandidateCategory.CALIBRATED_DAILY,
        ),
        true_name_terms=("synthetic-private-name",),
        approved_nicknames=("synthetic-nickname",),
    )

    assert error.code == "candidate_verification_failed"


def test_runtime_material_rejects_name_after_whitespace_normalization() -> None:
    package = _package("今天天气不错", CandidateCategory.WEATHER)

    error = _assert_rejected_without_mutation(
        package,
        true_name_terms=(" 天气 ",),
        approved_nicknames=(),
    )

    assert error.code == "candidate_verification_failed"


def test_runtime_material_rejects_hard_risk() -> None:
    error = _assert_rejected_without_mutation(
        _package("密码是合成值"),
        true_name_terms=("synthetic-private-name",),
        approved_nicknames=(),
    )

    assert error.code == "candidate_verification_failed"
    assert str(error) == ("candidate_verification_failed: candidate package verification failed")


@pytest.mark.parametrize(
    ("true_name_terms", "approved_nicknames"),
    [
        ((), ()),
        ((" ",), ()),
        ((" shared-term ",), ("shared-term",)),
        (("synthetic-private-name",), ("私人秘密",)),
        (("synthetic-private-name", 7), ()),
    ],
)
def test_runtime_material_rejects_invalid_or_cross_domain_configuration(
    true_name_terms: object,
    approved_nicknames: object,
) -> None:
    error = _assert_rejected_without_mutation(
        _package(),
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
    )

    assert error.code == "invalid_candidate_terms"
    assert str(error) == "invalid_candidate_terms: candidate terms are invalid"


def test_runtime_material_deeply_revalidates_package_schema() -> None:
    package = _package()
    forged = SafeCandidatePackage.model_construct(
        created_at=package.created_at,
        candidates=package.candidates,
        audit=package.audit.model_copy(update={"output_count": 2}),
    )

    error = _assert_rejected_without_mutation(
        forged,
        true_name_terms=("synthetic-private-name",),
        approved_nicknames=("synthetic-nickname",),
    )

    assert error.code == "candidate_verification_failed"


def test_runtime_material_failure_does_not_retain_body_or_identity_terms() -> None:
    sensitive_name = "SYNTHETIC_RUNTIME_NAME_SENTINEL"
    sensitive_text = f"hello-{sensitive_name}"
    package = _package(sensitive_text)

    error = _assert_rejected_without_mutation(
        package,
        true_name_terms=(f" {sensitive_name} ",),
        approved_nicknames=(),
    )

    assert sensitive_name not in str(error)
    assert sensitive_text not in str(error)
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename:
            frame_values = repr(traceback.tb_frame.f_locals)
            assert sensitive_name not in frame_values
            assert sensitive_text not in frame_values
        traceback = traceback.tb_next


@pytest.mark.parametrize(
    ("package_factory", "true_name_terms", "approved_nicknames"),
    [
        (_package, _IterBomb(), ()),
        (_package, _LenBomb(), ()),
        (_package, (_StripBomb(MALICIOUS_SENTINEL),), ()),
        (_package, _UnsafePrivacyFailureSequence(), ()),
        (_package, _MutatedPrivacyFailureSequence(), ()),
        (_package, ("synthetic-name",), _IterBomb()),
        (
            lambda: _ModelDumpBomb.model_validate(_package(MALICIOUS_SENTINEL).model_dump()),
            ("synthetic-name",),
            (),
        ),
    ],
)
def test_runtime_material_sanitizes_hostile_deep_validation_failures(
    package_factory: object,
    true_name_terms: Sequence[str],
    approved_nicknames: Sequence[str],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = package_factory()  # type: ignore[operator]

    with pytest.raises(PrivacyValidationError) as captured:
        validate_safe_candidate_runtime_material(
            package,
            true_name_terms=true_name_terms,
            approved_nicknames=approved_nicknames,
        )

    error = captured.value
    assert error.code == "candidate_verification_failed"
    assert str(error) == ("candidate_verification_failed: candidate package verification failed")
    assert MALICIOUS_SENTINEL not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if "companion\\privacy" in traceback.tb_frame.f_code.co_filename.replace("/", "\\"):
            assert MALICIOUS_SENTINEL not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    output = capsys.readouterr()
    assert MALICIOUS_SENTINEL not in output.out
    assert MALICIOUS_SENTINEL not in output.err
    assert MALICIOUS_SENTINEL not in caplog.text
