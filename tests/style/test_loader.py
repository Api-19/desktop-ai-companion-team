import traceback
from pathlib import Path

import pytest

from companion.style import StyleValidationError, load_safe_candidate_package


def _traceback_retains(error: BaseException, needle: str) -> bool:
    for frame, _ in traceback.walk_tb(error.__traceback__):
        filename = frame.f_code.co_filename.replace("/", "\\")
        if "companion\\style" in filename and needle in repr(frame.f_locals):
            return True
    return False


def _write_package(root: Path, filename: str, package: object) -> Path:
    source = root / filename
    root.mkdir(parents=True)
    source.write_text(package.model_dump_json(), encoding="utf-8")
    return source


def test_loads_package_inside_private_root(tmp_path: Path, synthetic_candidate_package) -> None:
    private_root = tmp_path / "private"
    source = _write_package(private_root, "candidate.json", synthetic_candidate_package)

    loaded = load_safe_candidate_package(
        source,
        private_data_dir=private_root,
        expected_count=3,
    )

    assert loaded == synthetic_candidate_package


def test_wrong_count_exposes_only_safe_error(tmp_path: Path, synthetic_candidate_package) -> None:
    private_root = tmp_path / "private"
    source = _write_package(
        private_root,
        "synthetic-secret-path.json",
        synthetic_candidate_package,
    )

    with pytest.raises(StyleValidationError) as exc_info:
        load_safe_candidate_package(
            source,
            private_data_dir=private_root,
            expected_count=78,
        )

    assert exc_info.value.code == "candidate_count_mismatch"
    assert "synthetic-secret-path" not in str(exc_info.value)
    assert "合成晚安呀" not in str(exc_info.value)
    assert not _traceback_retains(exc_info.value, "synthetic-secret-path")
    assert not _traceback_retains(exc_info.value, "合成晚安呀")


def test_rejects_path_outside_private_root(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()

    with pytest.raises(StyleValidationError) as exc_info:
        load_safe_candidate_package(
            tmp_path / "outside.json",
            private_data_dir=private_root,
            expected_count=3,
        )

    assert exc_info.value.code == "unsafe_style_path"


def test_invalid_schema_does_not_retain_payload_or_path(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    source = private_root / "synthetic-secret-invalid.json"
    source.write_text('{"synthetic-secret-value":true}', encoding="utf-8")

    with pytest.raises(StyleValidationError) as exc_info:
        load_safe_candidate_package(
            source,
            private_data_dir=private_root,
            expected_count=3,
        )

    assert exc_info.value.code == "invalid_private_model"
    assert not _traceback_retains(exc_info.value, "synthetic-secret-value")
    assert not _traceback_retains(exc_info.value, "synthetic-secret-invalid")


def test_expected_count_must_be_positive(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()

    with pytest.raises(StyleValidationError) as exc_info:
        load_safe_candidate_package(
            private_root / "candidate.json",
            private_data_dir=private_root,
            expected_count=0,
        )

    assert exc_info.value.code == "invalid_expected_count"
