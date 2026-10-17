from pathlib import Path

from companion.privacy import SafeCandidatePackage
from companion.style.errors import StyleValidationError
from companion.style.private_io import read_private_model


def _load_safe_candidate_package(
    source: Path,
    private_data_dir: Path,
    expected_count: int,
) -> SafeCandidatePackage:
    if expected_count <= 0:
        raise StyleValidationError(
            "invalid_expected_count",
            "expected count must be positive",
        )
    package = read_private_model(SafeCandidatePackage, source, private_data_dir)
    if len(package.candidates) != expected_count or package.audit.output_count != expected_count:
        raise StyleValidationError(
            "candidate_count_mismatch",
            "candidate count is invalid",
        )
    return package


def load_safe_candidate_package(
    source: Path,
    *,
    private_data_dir: Path,
    expected_count: int,
) -> SafeCandidatePackage:
    try:
        return _load_safe_candidate_package(source, private_data_dir, expected_count)
    except StyleValidationError as error:
        failure = (error.code, error.detail)
    del source, private_data_dir, expected_count
    raise StyleValidationError(failure[0], failure[1]) from None
