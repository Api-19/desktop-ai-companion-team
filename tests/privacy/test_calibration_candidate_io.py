import json
import os
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from companion.privacy import PrivacyValidationError
from companion.privacy.calibration_io import (
    CALIBRATION_CANDIDATE_RELATIVE_PATH,
    load_calibration_candidate_package,
    publish_calibration_candidate_package,
    read_calibration_review_decisions,
    revoke_calibration_candidates,
    write_calibration_review_package,
)
from companion.privacy.calibration_models import (
    CalibrationCandidate,
    CalibrationCandidateAudit,
    CalibrationCandidatePackage,
    CalibrationReviewDecision,
    CalibrationReviewDecisionValue,
    CalibrationReviewPackage,
    CalibrationReviewSubmission,
    calibration_candidate_id,
    calibration_identity_config_fingerprint,
    calibration_review_fingerprint,
    calibration_review_package_fingerprint,
    calibration_submission_id,
)
from companion.privacy.candidate_models import CandidateCategory

NOW = datetime(2026, 7, 22, tzinfo=UTC)
HOSTILE_SENTINEL = "SYNTHETIC_HOSTILE_CALIBRATION_IO_SENTINEL"


class _HostileTerms(Sequence[str]):
    def __init__(self, attack: str) -> None:
        self.attack = attack

    def __iter__(self) -> Iterator[str]:
        if self.attack == "privacy":
            raise PrivacyValidationError("unsafe_calibration_path", HOSTILE_SENTINEL)
        if self.attack == "iter":
            raise RuntimeError(HOSTILE_SENTINEL)
        return super().__iter__()

    def __len__(self) -> int:
        if self.attack == "len":
            raise RuntimeError(HOSTILE_SENTINEL)
        return 1

    def __getitem__(self, index: int) -> str:
        if index:
            raise IndexError(index)
        if self.attack == "strip":
            return _HostileStrip(HOSTILE_SENTINEL)
        return "synthetic-name"

    def __repr__(self) -> str:
        return HOSTILE_SENTINEL


class _HostileStrip(str):
    def strip(self, *args: object, **kwargs: object) -> str:
        raise RuntimeError(HOSTILE_SENTINEL)


class _HostilePackage(CalibrationCandidatePackage):
    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError(HOSTILE_SENTINEL)


class _HostilePrivateRoot:
    def __truediv__(self, other: object) -> object:
        raise RuntimeError(HOSTILE_SENTINEL)

    def __repr__(self) -> str:
        return HOSTILE_SENTINEL


def _assert_io_failure_is_sanitized(
    error: PrivacyValidationError,
    *,
    expected_code: str,
    expected_detail: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert error.code == expected_code
    assert str(error) == f"{expected_code}: {expected_detail}"
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename:
            assert HOSTILE_SENTINEL not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    captured = capsys.readouterr()
    assert HOSTILE_SENTINEL not in captured.out
    assert HOSTILE_SENTINEL not in captured.err
    assert HOSTILE_SENTINEL not in caplog.text


def _submission() -> CalibrationReviewSubmission:
    return CalibrationReviewSubmission(
        submission_id=calibration_submission_id(
            scenario_id="shopping-status",
            category=CandidateCategory.SHOPPING,
            intent="shopping_status",
        ),
        scenario_id="shopping-status",
        category=CandidateCategory.SHOPPING,
        intent="shopping_status",
        text="先看看也行",
        source="explicit_owner_ideal_response",
    )


def _review_package() -> CalibrationReviewPackage:
    submission = _submission()
    return CalibrationReviewPackage(
        submitted_at=NOW,
        content_fingerprint=calibration_review_fingerprint([submission]),
        submission_count=1,
        submissions=[submission],
    )


def _candidate(text: str, intent: str) -> CalibrationCandidate:
    return CalibrationCandidate(
        candidate_id=calibration_candidate_id(
            category=CandidateCategory.SHOPPING,
            intent=intent,
            text=text,
        ),
        category=CandidateCategory.SHOPPING,
        intent=intent,
        text=text,
    )


def _candidate_package(
    *candidates: CalibrationCandidate,
    true_name_terms: tuple[str, ...] = ("示例姓名甲",),
    approved_nicknames: tuple[str, ...] = ("小团子",),
) -> CalibrationCandidatePackage:
    values = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    audit = CalibrationCandidateAudit(
        reviewed_at=NOW,
        submission_count=len(values),
        approved_count=len(values),
        rejected_count=0,
        privacy_rejected_count=0,
        revoked_count=0,
        output_count=len(values),
    )
    return CalibrationCandidatePackage(
        created_at=NOW,
        content_fingerprint=calibration_review_fingerprint(values),
        identity_config_fingerprint=calibration_identity_config_fingerprint(
            true_name_terms,
            approved_nicknames,
        ),
        candidate_count=len(values),
        candidates=values,
        audit=audit,
    )


def _publish(package: CalibrationCandidatePackage, private_root: Path) -> Path:
    return publish_calibration_candidate_package(
        package,
        private_data_dir=private_root,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=("小团子",),
    )


@pytest.mark.parametrize("operation", ["publish", "load", "revoke"])
@pytest.mark.parametrize("attack", ["iter", "len", "strip", "privacy"])
def test_lifecycle_io_sanitizes_hostile_identity_validation(
    tmp_path: Path,
    operation: str,
    attack: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_root = tmp_path / "private_data"
    candidate = _candidate("先看看也行", "shopping_status")
    package = _candidate_package(candidate)
    formal = private_root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    if operation != "publish":
        formal.parent.mkdir(parents=True)
        formal.write_text(package.model_dump_json(), encoding="utf-8")
    original = formal.read_bytes() if formal.exists() else None

    with pytest.raises(PrivacyValidationError) as captured:
        if operation == "publish":
            publish_calibration_candidate_package(
                package,
                private_data_dir=private_root,
                true_name_terms=_HostileTerms(attack),
                approved_nicknames=(),
            )
        elif operation == "load":
            load_calibration_candidate_package(
                private_data_dir=private_root,
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=_HostileTerms(attack),
                approved_nicknames=(),
            )
        else:
            revoke_calibration_candidates(
                [candidate.candidate_id],
                private_data_dir=private_root,
                revoked_at=NOW + timedelta(minutes=1),
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=_HostileTerms(attack),
                approved_nicknames=(),
            )

    expected = {
        "publish": (
            "calibration_verification_failed",
            "calibration candidate package verification failed",
        ),
        "load": ("invalid_calibration_package", "calibration candidate package is invalid"),
        "revoke": (
            "invalid_calibration_revocation",
            "calibration candidate revocation is invalid",
        ),
    }[operation]
    _assert_io_failure_is_sanitized(
        captured.value,
        expected_code=expected[0],
        expected_detail=expected[1],
        caplog=caplog,
        capsys=capsys,
    )
    if original is None:
        assert not formal.exists()
    else:
        assert formal.read_bytes() == original


def test_publish_sanitizes_hostile_package_model_dump(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_root = tmp_path / "private_data"
    package = _HostilePackage.model_validate(
        _candidate_package(_candidate("先看看也行", "shopping_status")).model_dump()
    )

    with pytest.raises(PrivacyValidationError) as captured:
        _publish(package, private_root)

    _assert_io_failure_is_sanitized(
        captured.value,
        expected_code="calibration_verification_failed",
        expected_detail="calibration candidate package verification failed",
        caplog=caplog,
        capsys=capsys,
    )
    assert not (private_root / CALIBRATION_CANDIDATE_RELATIVE_PATH).exists()


def test_publish_failure_after_validation_does_not_retain_validated_package(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = _candidate_package(
        _candidate(HOSTILE_SENTINEL, "shopping_status"),
    )

    with pytest.raises(PrivacyValidationError) as captured:
        publish_calibration_candidate_package(
            package,
            private_data_dir=_HostilePrivateRoot(),  # type: ignore[arg-type]
            true_name_terms=("示例姓名甲",),
            approved_nicknames=("小团子",),
        )

    _assert_io_failure_is_sanitized(
        captured.value,
        expected_code="calibration_verification_failed",
        expected_detail="calibration candidate package verification failed",
        caplog=caplog,
        capsys=capsys,
    )


def test_fixed_formal_path_and_private_review_round_trip(tmp_path: Path) -> None:
    private_root = tmp_path / "private_data"
    review_path = private_root / "calibration" / "review.json"
    written = write_calibration_review_package(
        _review_package(),
        destination=review_path,
        private_data_dir=private_root,
    )
    assert written == review_path.resolve()

    submission = _submission()
    decision = CalibrationReviewDecision(
        submission_id=submission.submission_id,
        submission_fingerprint=submission.content_fingerprint,
        review_package_fingerprint=calibration_review_package_fingerprint(
            [submission], submitted_at=NOW
        ),
        decided_at=NOW,
        decision=CalibrationReviewDecisionValue.APPROVE,
    )
    decision_path = private_root / "calibration" / "decisions.json"
    decision_path.write_text(
        json.dumps({"decisions": [decision.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    assert read_calibration_review_decisions(
        decision_path,
        private_data_dir=private_root,
    ) == [decision]
    assert CALIBRATION_CANDIDATE_RELATIVE_PATH.as_posix() == (
        "style_candidates/calibration/current/calibration-candidates.json"
    )


@pytest.mark.parametrize("operation", ["write", "read"])
def test_private_io_rejects_path_escape(tmp_path: Path, operation: str) -> None:
    private_root = tmp_path / "private_data"
    outside = tmp_path / "outside.json"
    if operation == "read":
        outside.write_text('{"decisions": []}', encoding="utf-8")

    with pytest.raises(PrivacyValidationError) as error:
        if operation == "write":
            write_calibration_review_package(
                _review_package(),
                destination=outside,
                private_data_dir=private_root,
            )
        else:
            read_calibration_review_decisions(
                outside,
                private_data_dir=private_root,
            )

    assert error.value.code == "unsafe_calibration_path"


def test_private_io_rejects_reparse_component_when_supported(tmp_path: Path) -> None:
    private_root = tmp_path / "private_data"
    outside = tmp_path / "outside"
    outside.mkdir()
    private_root.mkdir()
    link = private_root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(PrivacyValidationError) as error:
        write_calibration_review_package(
            _review_package(),
            destination=link / "review.json",
            private_data_dir=private_root,
        )

    assert error.value.code == "unsafe_calibration_path"


@pytest.mark.parametrize("payload", ["{", "[]", '{"decisions":[],"extra":true}'])
def test_decision_reader_rejects_malformed_or_unknown_json(tmp_path: Path, payload: str) -> None:
    private_root = tmp_path / "private_data"
    source = private_root / "decisions.json"
    source.parent.mkdir(parents=True)
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(PrivacyValidationError) as error:
        read_calibration_review_decisions(source, private_data_dir=private_root)

    assert error.value.code == "invalid_calibration_decisions"


def test_publish_loads_only_the_fixed_formal_package(tmp_path: Path) -> None:
    private_root = tmp_path / "private_data"
    package = _candidate_package(_candidate("先看看也行", "shopping_status"))

    destination = _publish(package, private_root)
    loaded = load_calibration_candidate_package(
        private_data_dir=private_root,
        expected_fingerprint=package.content_fingerprint,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=("小团子",),
    )

    assert destination == (private_root / CALIBRATION_CANDIDATE_RELATIVE_PATH).resolve()
    assert loaded == package
    with pytest.raises(PrivacyValidationError) as error:
        load_calibration_candidate_package(
            private_data_dir=private_root,
            expected_fingerprint="f" * 64,
            true_name_terms=("示例姓名甲",),
            approved_nicknames=("小团子",),
        )
    assert error.value.code == "stale_calibration_package"


def test_failed_replacement_preserves_existing_formal_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private_data"
    old = _candidate_package(_candidate("先看看也行", "shopping_status"))
    new = _candidate_package(_candidate("可以再等等", "shopping_wait"))
    formal = _publish(old, private_root)
    old_bytes = formal.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("synthetic replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(PrivacyValidationError) as error:
        _publish(new, private_root)

    assert error.value.code == "calibration_publish_failed"
    assert formal.read_bytes() == old_bytes


def test_publish_rejects_forged_identity_config_fingerprint_without_replacement(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private_data"
    old = _candidate_package(_candidate("先看看也行", "shopping_status"))
    formal = _publish(old, private_root)
    old_bytes = formal.read_bytes()
    forged = _candidate_package(_candidate("可以再等等", "shopping_wait")).model_copy(
        update={"identity_config_fingerprint": "f" * 64}
    )

    with pytest.raises(PrivacyValidationError) as error:
        _publish(forged, private_root)

    assert error.value.code == "stale_calibration_identity_config"
    assert formal.read_bytes() == old_bytes


def test_failed_post_replace_verification_restores_existing_formal_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private_data"
    old = _candidate_package(_candidate("先看看也行", "shopping_status"))
    new = _candidate_package(_candidate("可以再等等", "shopping_wait"))
    formal = _publish(old, private_root)
    old_bytes = formal.read_bytes()
    original = CalibrationCandidatePackage.model_validate_json
    calls = 0

    def fail_formal_readback(payload: object, *args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic post-replace verification failure")
        return original(payload, *args, **kwargs)

    monkeypatch.setattr(
        CalibrationCandidatePackage,
        "model_validate_json",
        fail_formal_readback,
    )
    with pytest.raises(PrivacyValidationError):
        _publish(new, private_root)

    assert formal.read_bytes() == old_bytes


def test_failed_first_publish_verification_leaves_no_formal_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private_data"
    package = _candidate_package(_candidate("先看看也行", "shopping_status"))
    formal = private_root / CALIBRATION_CANDIDATE_RELATIVE_PATH
    original = CalibrationCandidatePackage.model_validate_json
    calls = 0

    def fail_formal_readback(payload: object, *args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic first-publish verification failure")
        return original(payload, *args, **kwargs)

    monkeypatch.setattr(
        CalibrationCandidatePackage,
        "model_validate_json",
        fail_formal_readback,
    )
    with pytest.raises(PrivacyValidationError):
        _publish(package, private_root)

    assert not formal.exists()


def test_publish_revalidates_hard_risk_even_for_constructed_models(tmp_path: Path) -> None:
    private_root = tmp_path / "private_data"
    text = "密码是合成值"
    forged_candidate = CalibrationCandidate.model_construct(
        candidate_id=calibration_candidate_id(
            category=CandidateCategory.SHOPPING,
            intent="shopping_status",
            text=text,
        ),
        category=CandidateCategory.SHOPPING,
        intent="shopping_status",
        text=text,
    )
    forged_package = CalibrationCandidatePackage.model_construct(
        created_at=NOW,
        content_fingerprint=calibration_review_fingerprint([forged_candidate]),
        identity_config_fingerprint=calibration_identity_config_fingerprint(
            ("示例姓名甲",),
            ("小团子",),
        ),
        candidate_count=1,
        candidates=[forged_candidate],
        audit=CalibrationCandidateAudit(
            reviewed_at=NOW,
            submission_count=1,
            approved_count=1,
            rejected_count=0,
            privacy_rejected_count=0,
            revoked_count=0,
            output_count=1,
        ),
    )

    with pytest.raises(PrivacyValidationError) as error:
        _publish(forged_package, private_root)

    assert error.value.code == "calibration_verification_failed"


def test_revocation_regenerates_deterministically_and_missing_id_is_atomic(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private_data"
    removed = _candidate("先看看也行", "shopping_status")
    retained = _candidate("可以再等等", "shopping_wait")
    original = _candidate_package(removed, retained)
    formal = _publish(original, private_root)

    regenerated = revoke_calibration_candidates(
        [removed.candidate_id],
        private_data_dir=private_root,
        revoked_at=NOW + timedelta(minutes=1),
        expected_fingerprint=original.content_fingerprint,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=("小团子",),
    )

    assert [candidate.candidate_id for candidate in regenerated.candidates] == [
        retained.candidate_id
    ]
    assert regenerated.audit.revoked_count == 1
    assert (
        load_calibration_candidate_package(
            private_data_dir=private_root,
            expected_fingerprint=regenerated.content_fingerprint,
            true_name_terms=("示例姓名甲",),
            approved_nicknames=("小团子",),
        )
        == regenerated
    )
    bytes_after_success = formal.read_bytes()

    with pytest.raises(PrivacyValidationError) as error:
        revoke_calibration_candidates(
            [removed.candidate_id],
            private_data_dir=private_root,
            revoked_at=NOW + timedelta(minutes=2),
            expected_fingerprint=regenerated.content_fingerprint,
            true_name_terms=("示例姓名甲",),
            approved_nicknames=("小团子",),
        )

    assert error.value.code == "calibration_candidate_missing"
    assert formal.read_bytes() == bytes_after_success


@pytest.mark.parametrize("operation", ["load", "revoke"])
def test_removed_approved_nickname_invalidates_formal_package_without_replacement(
    tmp_path: Path,
    operation: str,
) -> None:
    private_root = tmp_path / "private_data"
    removed_nickname = "小团子"
    candidate = _candidate(f"{removed_nickname}晚安", "sleep_goodnight")
    package = _candidate_package(candidate)
    formal = _publish(package, private_root)
    original_bytes = formal.read_bytes()

    with pytest.raises(PrivacyValidationError) as error:
        if operation == "load":
            load_calibration_candidate_package(
                private_data_dir=private_root,
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=("示例姓名甲",),
                approved_nicknames=(),
            )
        else:
            revoke_calibration_candidates(
                [candidate.candidate_id],
                private_data_dir=private_root,
                revoked_at=NOW + timedelta(minutes=1),
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=("示例姓名甲",),
                approved_nicknames=(),
            )

    assert error.value.code == "stale_calibration_identity_config"
    assert formal.read_bytes() == original_bytes


@pytest.mark.parametrize("operation", ["publish", "load", "revoke"])
def test_whitespace_normalized_true_name_is_applied_to_formal_content_gate(
    tmp_path: Path,
    operation: str,
) -> None:
    private_root = tmp_path / "private_data"
    sensitive_name = "仅用于合成规范化正式门禁的姓名"
    normalized_terms = (f"  {sensitive_name}  ",)
    candidate = _candidate(f"可以问问{sensitive_name}", "shopping_status")
    package = _candidate_package(
        candidate,
        true_name_terms=normalized_terms,
        approved_nicknames=(),
    )
    formal = private_root / CALIBRATION_CANDIDATE_RELATIVE_PATH

    if operation != "publish":
        formal.parent.mkdir(parents=True)
        formal.write_text(package.model_dump_json(), encoding="utf-8")
    original_bytes = formal.read_bytes() if formal.exists() else None

    with pytest.raises(PrivacyValidationError) as error:
        if operation == "publish":
            publish_calibration_candidate_package(
                package,
                private_data_dir=private_root,
                true_name_terms=normalized_terms,
                approved_nicknames=(),
            )
        elif operation == "load":
            load_calibration_candidate_package(
                private_data_dir=private_root,
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=normalized_terms,
                approved_nicknames=(),
            )
        else:
            revoke_calibration_candidates(
                [candidate.candidate_id],
                private_data_dir=private_root,
                revoked_at=NOW + timedelta(minutes=1),
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=normalized_terms,
                approved_nicknames=(),
            )

    assert error.value.code == "calibration_verification_failed"
    if original_bytes is None:
        assert not formal.exists()
    else:
        assert formal.read_bytes() == original_bytes


@pytest.mark.parametrize("operation", ["load", "revoke"])
def test_updated_identity_terms_fail_closed_without_changing_or_leaking_formal_package(
    tmp_path: Path, operation: str
) -> None:
    private_root = tmp_path / "private_data"
    sensitive_name = "仅用于合成身份更新测试的姓名"
    sensitive_text = f"可以问问{sensitive_name}"
    candidate = _candidate(sensitive_text, "shopping_status")
    package = _candidate_package(
        candidate,
        true_name_terms=("仅用于合成发布时身份词",),
        approved_nicknames=(),
    )
    formal = publish_calibration_candidate_package(
        package,
        private_data_dir=private_root,
        true_name_terms=("仅用于合成发布时身份词",),
        approved_nicknames=(),
    )
    original_bytes = formal.read_bytes()

    try:
        if operation == "load":
            load_calibration_candidate_package(
                private_data_dir=private_root,
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=(sensitive_name,),
                approved_nicknames=(),
            )
        else:
            revoke_calibration_candidates(
                [candidate.candidate_id],
                private_data_dir=private_root,
                revoked_at=NOW + timedelta(minutes=1),
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=(sensitive_name,),
                approved_nicknames=(),
            )
    except PrivacyValidationError as error:
        assert error.code == "stale_calibration_identity_config"
        assert sensitive_name not in str(error)
        assert sensitive_text not in str(error)
        assert str(private_root) not in str(error)
        traceback = error.__traceback__
        while traceback is not None:
            filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
            if "companion\\privacy" in filename:
                values = repr(traceback.tb_frame.f_locals)
                assert sensitive_name not in values
                assert sensitive_text not in values
                assert str(private_root) not in values
            traceback = traceback.tb_next
    else:
        raise AssertionError("expected identity revalidation failure")

    assert formal.read_bytes() == original_bytes


@pytest.mark.parametrize("operation", ["load", "revoke"])
def test_identity_config_failure_does_not_retain_terms_body_digest_or_path(
    tmp_path: Path,
    operation: str,
) -> None:
    private_root = tmp_path / "private_data"
    sensitive_nickname = "仅用于合成指纹泄漏测试的爱称"
    sensitive_text = f"{sensitive_nickname}晚安"
    candidate = _candidate(sensitive_text, "sleep_goodnight")
    package = _candidate_package(
        candidate,
        approved_nicknames=(sensitive_nickname,),
    )
    formal = publish_calibration_candidate_package(
        package,
        private_data_dir=private_root,
        true_name_terms=("示例姓名甲",),
        approved_nicknames=(sensitive_nickname,),
    )
    original_bytes = formal.read_bytes()
    private_digest = package.identity_config_fingerprint

    try:
        if operation == "load":
            load_calibration_candidate_package(
                private_data_dir=private_root,
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=("示例姓名甲",),
                approved_nicknames=(),
            )
        else:
            revoke_calibration_candidates(
                [candidate.candidate_id],
                private_data_dir=private_root,
                revoked_at=NOW + timedelta(minutes=1),
                expected_fingerprint=package.content_fingerprint,
                true_name_terms=("示例姓名甲",),
                approved_nicknames=(),
            )
    except PrivacyValidationError as error:
        for sensitive in (
            sensitive_nickname,
            sensitive_text,
            private_digest,
            str(private_root),
        ):
            assert sensitive not in str(error)
        traceback = error.__traceback__
        while traceback is not None:
            filename = traceback.tb_frame.f_code.co_filename.replace("/", "\\")
            if "companion\\privacy" in filename:
                values = repr(traceback.tb_frame.f_locals)
                for sensitive in (
                    sensitive_nickname,
                    sensitive_text,
                    private_digest,
                    str(private_root),
                ):
                    assert sensitive not in values
            traceback = traceback.tb_next
    else:
        raise AssertionError("expected identity config fingerprint failure")

    assert formal.read_bytes() == original_bytes
