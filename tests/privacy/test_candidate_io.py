import inspect
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import pytest

from companion.privacy import PrivacyValidationError, candidate_io
from companion.privacy.ambiguous_models import CalibrationProfile
from companion.privacy.candidate_io import (
    replace_unsafe_result_with_candidate_package as _public_replace_candidate_package,
)
from companion.privacy.candidate_models import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)

CREATED_AT = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
TRUE_NAME_TERMS = ("synthetic-private-name",)
APPROVED_NICKNAMES = ("synthetic-nickname",)


def replace_unsafe_result_with_candidate_package(
    package: SafeCandidatePackage,
    *,
    destination: Path,
    unsafe_result: Path,
    private_data_dir: Path,
    true_name_terms: tuple[str, ...] = TRUE_NAME_TERMS,
    approved_nicknames: tuple[str, ...] = APPROVED_NICKNAMES,
    calibration_profile: CalibrationProfile | None = None,
) -> Path:
    return _public_replace_candidate_package(
        package,
        destination=destination,
        unsafe_result=unsafe_result,
        private_data_dir=private_data_dir,
        true_name_terms=true_name_terms,
        approved_nicknames=approved_nicknames,
        calibration_profile=calibration_profile,
    )


def synthetic_package(
    *,
    text: str = "晚安synthetic-nickname",
    category: CandidateCategory = CandidateCategory.GREETING,
) -> SafeCandidatePackage:
    return SafeCandidatePackage(
        created_at=CREATED_AT,
        candidates=[
            SafeStyleCandidate(
                text=text,
                category=category,
                scene_tags=[category.value],
            )
        ],
        audit=CandidateSelectionAudit(
            processed_at=CREATED_AT,
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


def synthetic_calibration_profile(*, text: str) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id="synthetic-candidate-profile",
        activated_at=CREATED_AT,
        manual_approved_texts=[text],
        rules=[],
        first_round_keep_count=2,
        first_round_discard_count=0,
        first_round_uncertain_count=0,
        second_round_keep_count=1,
        second_round_discard_count=0,
        second_round_uncertain_count=0,
        user_confirmed=True,
        enabled=True,
    )


def privacy_traceback_retains(error: BaseException, marker: str) -> bool:
    current = error.__traceback__
    while current is not None:
        filename = current.tb_frame.f_code.co_filename.replace("/", "\\")
        if "companion\\privacy" in filename and marker in repr(current.tb_frame.f_locals):
            return True
        current = current.tb_next
    return False


def temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp")


def test_public_replace_requires_runtime_name_and_nickname_terms() -> None:
    signature = inspect.signature(_public_replace_candidate_package)

    assert signature.parameters["true_name_terms"].default is inspect.Parameter.empty
    assert signature.parameters["approved_nicknames"].default is inspect.Parameter.empty


def test_calibrated_candidate_requires_matching_enabled_profile(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "candidate.json"
    unsafe = private_root / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    package = synthetic_package(
        text="今天整理了一下桌面",
        category=CandidateCategory.CALIBRATED_DAILY,
    )

    with pytest.raises(PrivacyValidationError) as missing_profile:
        replace_unsafe_result_with_candidate_package(
            package,
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
            calibration_profile=None,
        )

    assert missing_profile.value.code == "candidate_verification_failed"

    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    written = replace_unsafe_result_with_candidate_package(
        package,
        destination=destination,
        unsafe_result=unsafe,
        private_data_dir=private_root,
        calibration_profile=synthetic_calibration_profile(text="今天整理了一下桌面"),
    )

    assert written == destination.resolve()


def test_calibrated_candidate_does_not_override_existing_allow_category(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "candidate.json"
    unsafe = private_root / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    package = synthetic_package(
        text="晚安synthetic-nickname",
        category=CandidateCategory.CALIBRATED_DAILY,
    )

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            package,
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
            calibration_profile=synthetic_calibration_profile(text="晚安synthetic-nickname"),
        )

    assert exc_info.value.code == "candidate_verification_failed"
    assert unsafe.exists()


def test_formal_calibrated_profile_mismatch_rolls_back_and_preserves_unsafe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "candidate.json"
    unsafe = private_root / "unsafe.json"
    unsafe_placeholder = "synthetic unsafe placeholder"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text(unsafe_placeholder, encoding="utf-8")
    package = synthetic_package(
        text="今天整理了一下桌面",
        category=CandidateCategory.CALIBRATED_DAILY,
    )
    original_runtime_validation = candidate_io._validate_runtime_content
    runtime_validation_count = 0

    def fail_only_formal_calibrated_validation(
        package: SafeCandidatePackage,
        true_name_terms: tuple[str, ...],
        approved_nicknames: tuple[str, ...],
        calibration_profile: CalibrationProfile | None,
    ) -> None:
        nonlocal runtime_validation_count
        runtime_validation_count += 1
        if runtime_validation_count < 3:
            original_runtime_validation(
                package,
                true_name_terms,
                approved_nicknames,
                calibration_profile,
            )
            return
        original_runtime_validation(
            package,
            true_name_terms,
            approved_nicknames,
            synthetic_calibration_profile(text="不匹配的合成文本"),
        )

    monkeypatch.setattr(
        candidate_io,
        "_validate_runtime_content",
        fail_only_formal_calibrated_validation,
    )

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            package,
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
            calibration_profile=synthetic_calibration_profile(text="今天整理了一下桌面"),
        )

    assert exc_info.value.code == "candidate_verification_failed"
    assert unsafe.read_text(encoding="utf-8") == unsafe_placeholder
    assert not destination.exists()


def test_temporary_write_and_readback_do_not_reopen_the_unique_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_write_text = Path.write_text
    original_read_text = Path.read_text

    def reject_temporary_path_write(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> int:
        if path.parent == destination.parent and path.suffix == ".tmp":
            raise OSError("synthetic path-bound temporary write")
        return original_write_text(path, *args, **kwargs)

    def reject_temporary_path_read(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path.parent == destination.parent and path.suffix == ".tmp":
            raise OSError("synthetic path-bound temporary read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", reject_temporary_path_write)
    monkeypatch.setattr(Path, "read_text", reject_temporary_path_read)

    written = replace_unsafe_result_with_candidate_package(
        synthetic_package(),
        destination=destination,
        unsafe_result=unsafe,
        private_data_dir=private_root,
    )

    assert written == destination.resolve()
    assert not unsafe.exists()


def test_temporary_serialization_write_and_readback_use_captured_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_mkstemp = __import__("tempfile").mkstemp
    original_fdopen = os.fdopen
    temporary_descriptor: int | None = None
    opened_descriptors: list[int] = []
    descriptor_events: list[tuple[str, int]] = []
    written_payloads: list[bytes] = []
    read_payloads: list[bytes] = []

    class InstrumentedTemporaryFile:
        def __init__(self, stream: BinaryIO) -> None:
            self.stream = stream

        def __enter__(self) -> "InstrumentedTemporaryFile":
            self.stream.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self.stream.__exit__(*args)

        def write(self, payload: bytes) -> int:
            descriptor = self.stream.fileno()
            assert descriptor == temporary_descriptor
            descriptor_events.append(("write", descriptor))
            written_payloads.append(payload)
            return self.stream.write(payload)

        def read(self) -> bytes:
            descriptor = self.stream.fileno()
            assert descriptor == temporary_descriptor
            descriptor_events.append(("read", descriptor))
            payload = self.stream.read()
            read_payloads.append(payload)
            return payload

        def __getattr__(self, name: str) -> object:
            return getattr(self.stream, name)

    def capture_candidate_descriptor(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal temporary_descriptor
        descriptor, path = original_mkstemp(*args, **kwargs)
        if kwargs.get("suffix") == ".tmp":
            temporary_descriptor = descriptor
        return descriptor, path

    def instrument_captured_descriptor(
        descriptor: int,
        *args: object,
        **kwargs: object,
    ) -> BinaryIO | InstrumentedTemporaryFile:
        stream = original_fdopen(descriptor, *args, **kwargs)
        if descriptor == temporary_descriptor and args == ("w+b",):
            opened_descriptors.append(descriptor)
            return InstrumentedTemporaryFile(stream)
        return stream

    monkeypatch.setattr("tempfile.mkstemp", capture_candidate_descriptor)
    monkeypatch.setattr(os, "fdopen", instrument_captured_descriptor)

    replace_unsafe_result_with_candidate_package(
        synthetic_package(),
        destination=destination,
        unsafe_result=unsafe,
        private_data_dir=private_root,
    )

    assert temporary_descriptor is not None
    assert opened_descriptors == [temporary_descriptor]
    assert descriptor_events == [
        ("write", temporary_descriptor),
        ("read", temporary_descriptor),
    ]
    assert len(written_payloads) == 1
    assert read_payloads == written_payloads
    assert not unsafe.exists()


@pytest.mark.parametrize("tamper_kind", ["identity", "content"])
def test_formal_destination_is_identity_and_content_verified_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    package = synthetic_package()
    original_link = os.link
    original_unlink = Path.unlink
    original_write_text = Path.write_text

    def tamper_after_publication(source: Path, target: Path) -> None:
        original_link(source, target)
        if tamper_kind == "identity":
            original_unlink(target)
            original_write_text(target, package.model_dump_json(), encoding="utf-8")
        else:
            original_write_text(target, "synthetic invalid formal content", encoding="utf-8")

    monkeypatch.setattr(os, "link", tamper_after_publication)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            package,
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code in {
        "candidate_temporary_invalid",
        "candidate_verification_failed",
    }
    assert unsafe.exists()
    assert not destination.exists()


@pytest.mark.parametrize(
    "package",
    [
        synthetic_package(text="合成成人暗示内容"),
        synthetic_package(text="晚安synthetic-private-name"),
        synthetic_package(text="谢谢你"),
    ],
)
def test_runtime_content_validation_fails_before_unsafe_deletion(
    tmp_path: Path,
    package: SafeCandidatePackage,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            package,
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
            true_name_terms=TRUE_NAME_TERMS,
            approved_nicknames=APPROVED_NICKNAMES,
        )

    assert exc_info.value.code == "candidate_verification_failed"
    assert unsafe.exists()
    assert not destination.exists()


def test_formal_destination_runtime_validation_failure_rolls_back_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe_placeholder = "synthetic unsafe placeholder"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text(unsafe_placeholder, encoding="utf-8")
    original_runtime_validation = candidate_io._validate_runtime_content
    runtime_validation_count = 0

    def fail_only_formal_runtime_validation(
        package: SafeCandidatePackage,
        true_name_terms: tuple[str, ...],
        approved_nicknames: tuple[str, ...],
        calibration_profile: CalibrationProfile | None,
    ) -> None:
        nonlocal runtime_validation_count
        runtime_validation_count += 1
        if runtime_validation_count < 3:
            assert not destination.exists()
            original_runtime_validation(
                package,
                true_name_terms,
                approved_nicknames,
                calibration_profile,
            )
            return

        assert runtime_validation_count == 3
        assert destination.exists()
        assert unsafe.exists()
        rejected_candidate = package.candidates[0].model_copy(update={"text": "合成成人暗示内容"})
        rejected_package = package.model_copy(update={"candidates": [rejected_candidate]})
        original_runtime_validation(
            rejected_package,
            true_name_terms,
            approved_nicknames,
            calibration_profile,
        )

    monkeypatch.setattr(
        candidate_io,
        "_validate_runtime_content",
        fail_only_formal_runtime_validation,
    )

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "candidate_verification_failed"
    assert runtime_validation_count == 3
    assert not destination.exists()
    assert unsafe.read_text(encoding="utf-8") == unsafe_placeholder


def test_successful_replace_writes_candidate_then_deletes_unsafe_result(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    package = synthetic_package()

    written = replace_unsafe_result_with_candidate_package(
        package,
        destination=destination,
        unsafe_result=unsafe,
        private_data_dir=private_root,
    )

    assert written == destination.resolve()
    assert (
        SafeCandidatePackage.model_validate_json(destination.read_text(encoding="utf-8")) == package
    )
    assert not unsafe.exists()
    assert not temporary_path(destination).exists()


@pytest.mark.parametrize("path_name", ["destination", "unsafe_result"])
def test_paths_must_stay_inside_private_root(tmp_path: Path, path_name: str) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    values = {"destination": destination, "unsafe_result": unsafe}
    values[path_name] = tmp_path / "outside" / f"synthetic-secret-{path_name}.json"

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=values["destination"],
            unsafe_result=values["unsafe_result"],
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_candidate_path"
    assert not privacy_traceback_retains(exc_info.value, "synthetic-secret")


def test_rejects_equal_candidate_and_unsafe_paths(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    same_path = private_root / "results" / "synthetic-shared.json"

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=same_path,
            unsafe_result=same_path,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_candidate_path"
    assert not same_path.exists()
    assert not temporary_path(same_path).exists()


def test_rejects_existing_destination_without_overwriting_it(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("synthetic prior candidate", encoding="utf-8")
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_candidate_path"
    assert destination.read_text(encoding="utf-8") == "synthetic prior candidate"
    assert unsafe.exists()
    assert not temporary_path(destination).exists()


def test_missing_unsafe_result_fails_without_creating_candidate_or_temp(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "missing-unsafe.json"

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_result_missing"
    assert not destination.exists()
    assert not list(private_root.rglob("*.tmp"))


def test_legacy_fixed_temp_is_not_overwritten(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    legacy_temp = temporary_path(destination)
    legacy_temp.parent.mkdir(parents=True)
    legacy_temp.write_text("synthetic legacy temporary", encoding="utf-8")
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")

    replace_unsafe_result_with_candidate_package(
        synthetic_package(),
        destination=destination,
        unsafe_result=unsafe,
        private_data_dir=private_root,
    )

    assert legacy_temp.read_text(encoding="utf-8") == "synthetic legacy temporary"
    assert destination.exists()
    assert not unsafe.exists()


def test_destination_created_during_write_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_link = os.link
    original_write_text = Path.write_text

    def create_destination_at_publication(source: Path, target: Path) -> None:
        original_write_text(target, "synthetic raced destination", encoding="utf-8")
        original_link(source, target)

    monkeypatch.setattr(os, "link", create_destination_at_publication)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_candidate_path"
    assert destination.read_text(encoding="utf-8") == "synthetic raced destination"
    assert unsafe.exists()
    assert not [
        path for path in destination.parent.glob("*.tmp") if path != temporary_path(destination)
    ]


def test_readback_mismatch_does_not_replace_or_delete_any_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_validate = SafeCandidatePackage.model_validate_json

    def mismatched_readback(cls: type[SafeCandidatePackage], payload: str) -> SafeCandidatePackage:
        verified = original_validate(payload)
        return verified.model_copy(update={"candidates": []})

    monkeypatch.setattr(
        SafeCandidatePackage,
        "model_validate_json",
        classmethod(mismatched_readback),
    )

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "candidate_verification_failed"
    assert unsafe.exists()
    assert not destination.exists()
    assert not temporary_path(destination).exists()


def test_delete_failure_rolls_back_new_formal_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_only_unsafe(path: Path, *args: object, **kwargs: object) -> None:
        if path == unsafe:
            raise OSError("synthetic unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_only_unsafe)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "candidate_replace_failed"
    assert unsafe.exists()
    assert not destination.exists()
    assert not temporary_path(destination).exists()


def test_published_temp_cleanup_failure_rolls_back_and_preserves_unsafe_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_every_candidate_temp_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent == destination.parent and path.suffix == ".tmp":
            raise OSError("synthetic persistent temporary cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_every_candidate_temp_unlink)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "candidate_cleanup_failed"
    assert unsafe.exists()
    assert not destination.exists()


def test_rollback_quarantines_formal_candidate_when_direct_removal_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_unsafe_and_destination(path: Path, *args: object, **kwargs: object) -> None:
        if path in {unsafe, destination}:
            raise OSError("synthetic rollback unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_unsafe_and_destination)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "candidate_replace_failed"
    assert unsafe.exists()
    assert not destination.exists()
    assert not list(destination.parent.glob("*.quarantine-*"))


def test_rollback_keeps_quarantine_out_of_formal_destination_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_rollback_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path in {unsafe, destination} or ".quarantine-" in path.name:
            raise OSError("synthetic rollback cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_rollback_cleanup)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "candidate_rollback_failed"
    assert unsafe.exists()
    assert not destination.exists()
    assert list(destination.parent.glob("*.quarantine-*"))
    assert not privacy_traceback_retains(exc_info.value, "quarantine")


def test_rollback_reports_a_distinct_error_when_quarantine_move_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_unlink = Path.unlink
    original_replace = os.replace

    def fail_unsafe_and_destination(path: Path, *args: object, **kwargs: object) -> None:
        if path in {unsafe, destination}:
            raise OSError("synthetic rollback unlink failure")
        original_unlink(path, *args, **kwargs)

    def fail_quarantine_move(source: Path, target: Path) -> None:
        if source == destination and ".quarantine-" in target.name:
            raise OSError("synthetic quarantine move failure")
        original_replace(source, target)

    monkeypatch.setattr(Path, "unlink", fail_unsafe_and_destination)
    monkeypatch.setattr(os, "replace", fail_quarantine_move)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "candidate_rollback_failed"
    assert unsafe.exists()
    assert destination.exists()
    assert not list(destination.parent.glob("*.quarantine-*"))
    assert not privacy_traceback_retains(exc_info.value, "quarantine")


def test_quarantine_move_failure_reports_placeholder_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_unlink = Path.unlink
    original_replace = os.replace

    def fail_rollback_and_placeholder_cleanup(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if path in {unsafe, destination} or ".quarantine-" in path.name:
            raise OSError("synthetic quarantine placeholder cleanup failure")
        original_unlink(path, *args, **kwargs)

    def fail_quarantine_move(source: Path, target: Path) -> None:
        if source == destination and ".quarantine-" in target.name:
            raise OSError("synthetic quarantine move failure")
        original_replace(source, target)

    monkeypatch.setattr(Path, "unlink", fail_rollback_and_placeholder_cleanup)
    monkeypatch.setattr(os, "replace", fail_quarantine_move)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "candidate_cleanup_failed"
    assert unsafe.exists()
    assert destination.exists()
    assert list(destination.parent.glob("*.quarantine-*"))
    assert not privacy_traceback_retains(exc_info.value, "quarantine")


def test_unique_temp_swap_attempt_fails_closed_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_fdopen = os.fdopen
    original_unlink = Path.unlink
    original_write_text = Path.write_text
    replacement_attempted = False
    replaced = False

    def replace_temp_before_descriptor_write(
        descriptor: int,
        mode: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal replaced, replacement_attempted
        if not replaced and mode == "w+b":
            candidates = list(destination.parent.glob(f".{destination.name}.*.tmp"))
            assert len(candidates) == 1
            candidate = candidates[0]
            replacement_attempted = True
            original_unlink(candidate)
            original_write_text(candidate, "synthetic swapped temporary", encoding="utf-8")
            replaced = True
        return original_fdopen(descriptor, mode, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", replace_temp_before_descriptor_write)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code in {
        "candidate_replace_failed",
        "candidate_temporary_invalid",
    }
    assert replacement_attempted
    assert unsafe.exists()
    assert not destination.exists()


def test_detected_reparse_component_during_recheck_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    destination = private_root / "style_candidates" / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    destination_parent_checks = 0

    def become_reparse_after_initial_check(path: Path) -> bool:
        nonlocal destination_parent_checks
        if path == destination.parent:
            destination_parent_checks += 1
            return destination_parent_checks >= 2
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", become_reparse_after_initial_check)

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_candidate_path"
    assert unsafe.exists()
    assert not destination.exists()
    assert destination_parent_checks >= 2


def test_internal_symlink_component_is_rejected_when_supported(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    actual_parent = private_root / "actual-candidates"
    actual_parent.mkdir()
    linked_parent = private_root / "linked-candidates"
    try:
        linked_parent.symlink_to(actual_parent, target_is_directory=True)
    except OSError:
        pytest.skip("synthetic directory symlinks are unavailable on this platform")

    destination = linked_parent / "safe-candidates.json"
    unsafe = private_root / "privacy_results" / "unsafe.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("synthetic unsafe placeholder", encoding="utf-8")

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            synthetic_package(),
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_candidate_path"
    assert unsafe.exists()
    assert not (actual_parent / "safe-candidates.json").exists()


def test_public_failure_traceback_does_not_retain_package_or_paths(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    destination = tmp_path / "synthetic-secret-destination.json"
    unsafe = private_root / "privacy_results" / "synthetic-secret-unsafe.json"
    package = synthetic_package()

    with pytest.raises(PrivacyValidationError) as exc_info:
        replace_unsafe_result_with_candidate_package(
            package,
            destination=destination,
            unsafe_result=unsafe,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_candidate_path"
    assert not privacy_traceback_retains(exc_info.value, "synthetic-secret")
    assert not privacy_traceback_retains(exc_info.value, "synthetic-private-name")
    assert not privacy_traceback_retains(exc_info.value, "synthetic-nickname")
