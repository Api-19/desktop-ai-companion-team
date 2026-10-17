import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from companion.contracts import SpeakerRole
from companion.privacy import ambiguous_review_io
from companion.privacy.ambiguous_models import (
    AmbiguousDecisionValue,
    AmbiguousReviewItem,
    AmbiguousReviewPackage,
    ReviewContextItem,
    ReviewRound,
)
from companion.privacy.ambiguous_review_html import render_ambiguous_review_html
from companion.privacy.ambiguous_review_io import (
    read_ambiguous_decisions,
    write_ambiguous_review,
)
from companion.privacy.candidate_io import _CandidateIoFailure
from companion.privacy.errors import PrivacyValidationError

CREATED_AT = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)
FINGERPRINT = "a" * 64
TOKEN = "synthetic-package-token-000000000001"


def synthetic_review_package(
    *,
    item_count: int = 1,
    text: str = "ordinary synthetic candidate",
) -> AmbiguousReviewPackage:
    items = [
        AmbiguousReviewItem(
            review_item_id=f"synthetic-item-{index + 1:04d}",
            text=text if index == 0 else f"ordinary synthetic candidate {index}",
            context_before=[
                ReviewContextItem(role=SpeakerRole.OTHER, text="synthetic context before")
            ],
            context_after=[
                ReviewContextItem(role=SpeakerRole.OWNER, text="synthetic context after")
            ],
            similarity_group="m:plain:text",
            occurrence_count=index + 1,
        )
        for index in range(item_count)
    ]
    return AmbiguousReviewPackage(
        package_token=TOKEN,
        review_round=ReviewRound.FIRST,
        created_at=CREATED_AT,
        content_fingerprint=FINGERPRINT,
        requested_count=100,
        item_count=len(items),
        items=items,
    )


def decision_document(
    package: AmbiguousReviewPackage,
    decisions: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "package_token": package.package_token,
        "content_fingerprint": package.content_fingerprint,
        "decisions": decisions
        if decisions is not None
        else [
            {"review_item_id": item.review_item_id, "decision": "keep"} for item in package.items
        ],
    }


def write_decisions(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def privacy_traceback_retains(error: BaseException, marker: str) -> bool:
    current = error.__traceback__
    while current is not None:
        filename = current.tb_frame.f_code.co_filename.replace("/", "\\")
        values = tuple(current.tb_frame.f_locals.values())
        if "companion\\privacy" in filename and marker in repr(values):
            return True
        current = current.tb_next
    return False


def test_html_has_no_network_resources_and_never_interprets_candidate_markup() -> None:
    package = synthetic_review_package(text="<img src=x onerror=alert(1)>")

    html = render_ambiguous_review_html(package)

    assert "connect-src 'none'" in html
    assert "default-src 'none'" in html
    assert "textContent" in html
    assert "innerHTML" not in html
    assert "http://" not in html
    assert "https://" not in html
    assert "<img src=x" not in html
    assert "\\u003cimg src=x onerror=alert(1)\\u003e" in html
    assert "Previous" in html
    assert "Next" in html
    assert "Clear progress" in html
    assert "occurrence_count" in html
    assert "Context is for review only" in html


def test_write_ambiguous_review_writes_package_and_html_inside_private_root(
    tmp_path: Path,
) -> None:
    package = synthetic_review_package(item_count=2)
    private_root = tmp_path / "private"
    package_path = private_root / "review" / "package.json"
    html_path = private_root / "review" / "index.html"

    written_package, written_html = write_ambiguous_review(
        package,
        package_path=package_path,
        html_path=html_path,
        private_data_dir=private_root,
    )

    assert written_package == package_path.resolve()
    assert written_html == html_path.resolve()
    assert (
        AmbiguousReviewPackage.model_validate_json(package_path.read_text(encoding="utf-8"))
        == package
    )
    assert "localStorage" in html_path.read_text(encoding="utf-8")


def test_write_ambiguous_review_rejects_overwrite_and_path_escape(tmp_path: Path) -> None:
    package = synthetic_review_package()
    private_root = tmp_path / "private"
    package_path = private_root / "package.json"
    html_path = private_root / "index.html"
    write_ambiguous_review(
        package,
        package_path=package_path,
        html_path=html_path,
        private_data_dir=private_root,
    )

    with pytest.raises(PrivacyValidationError) as overwrite:
        write_ambiguous_review(
            package,
            package_path=package_path,
            html_path=private_root / "other.html",
            private_data_dir=private_root,
        )
    assert overwrite.value.code == "unsafe_review_path"

    with pytest.raises(PrivacyValidationError) as escape:
        write_ambiguous_review(
            package,
            package_path=tmp_path / "outside.json",
            html_path=private_root / "safe.html",
            private_data_dir=private_root,
        )
    assert escape.value.code == "unsafe_review_path"


def test_write_ambiguous_review_revalidates_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = synthetic_review_package()
    private_root = tmp_path / "private"
    original_review_path = ambiguous_review_io._review_path
    calls = 0

    def reparse_after_temporary(path: Path, private_data_dir: Path, *, include_leaf: bool) -> Path:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ambiguous_review_io._ReviewIoFailure(
                "unsafe_review_path",
                "review path must stay inside the private-data directory",
            )
        return original_review_path(path, private_data_dir, include_leaf=include_leaf)

    monkeypatch.setattr(ambiguous_review_io, "_review_path", reparse_after_temporary)

    with pytest.raises(PrivacyValidationError) as exc_info:
        write_ambiguous_review(
            package,
            package_path=private_root / "package.json",
            html_path=private_root / "index.html",
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_review_path"


def test_write_ambiguous_review_translates_helper_failures_and_redacts_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "synthetic-sensitive-write-marker"
    package = synthetic_review_package(text=marker)

    def fail_identity(*_args: object, **_kwargs: object) -> None:
        raise _CandidateIoFailure(
            "candidate_temporary_invalid",
            "temporary identity failed",
        )

    monkeypatch.setattr(ambiguous_review_io, "_verify_temporary_identity", fail_identity)

    with pytest.raises(PrivacyValidationError) as exc_info:
        write_ambiguous_review(
            package,
            package_path=tmp_path / "private" / "package.json",
            html_path=tmp_path / "private" / "index.html",
            private_data_dir=tmp_path / "private",
        )

    assert exc_info.value.code == "review_verification_failed"
    assert marker not in str(exc_info.value)
    assert exc_info.value.__context__ is None
    assert not privacy_traceback_retains(exc_info.value, marker)


def test_write_ambiguous_review_revalidates_immediately_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = synthetic_review_package()
    private_root = tmp_path / "private"
    original_link = ambiguous_review_io.os.link
    original_review_path = ambiguous_review_io._review_path
    original_verify_temporary_identity = ambiguous_review_io._verify_temporary_identity
    linked = False
    post_link_path_checked = False

    def tracking_link(src: Path, dst: Path) -> None:
        nonlocal linked
        original_link(src, dst)
        linked = True

    def tracking_review_path(path: Path, private_data_dir: Path, *, include_leaf: bool) -> Path:
        nonlocal post_link_path_checked
        resolved = original_review_path(path, private_data_dir, include_leaf=include_leaf)
        if linked and include_leaf:
            post_link_path_checked = True
        return resolved

    def fail_if_identity_precedes_post_link_path_check(path: Path, expected: object) -> None:
        if linked and not post_link_path_checked:
            raise _CandidateIoFailure(
                "candidate_temporary_invalid",
                "post-link identity checked before path",
            )
        original_verify_temporary_identity(path, expected)  # type: ignore[arg-type]

    monkeypatch.setattr(ambiguous_review_io.os, "link", tracking_link)
    monkeypatch.setattr(ambiguous_review_io, "_review_path", tracking_review_path)
    monkeypatch.setattr(
        ambiguous_review_io,
        "_verify_temporary_identity",
        fail_if_identity_precedes_post_link_path_check,
    )

    write_ambiguous_review(
        package,
        package_path=private_root / "package.json",
        html_path=private_root / "index.html",
        private_data_dir=private_root,
    )

    assert post_link_path_checked


@pytest.mark.parametrize(
    "mutation,expected_code",
    [
        ("missing", "incomplete_review_decisions"),
        ("extra", "incomplete_review_decisions"),
        ("duplicate", "invalid_review_package"),
        ("wrong_token", "stale_review_package"),
        ("wrong_fingerprint", "stale_review_package"),
        ("bad_decision", "invalid_review_package"),
    ],
)
def test_decision_import_fails_atomically_for_invalid_documents(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    package = synthetic_review_package(item_count=3)
    document = decision_document(package)
    decisions = list(document["decisions"])  # type: ignore[arg-type]
    if mutation == "missing":
        document["decisions"] = decisions[:-1]
    elif mutation == "extra":
        document["decisions"] = [
            *decisions,
            {"review_item_id": "synthetic-item-9999", "decision": "keep"},
        ]
    elif mutation == "duplicate":
        document["decisions"] = [decisions[0], decisions[0], decisions[2]]
    elif mutation == "wrong_token":
        document["package_token"] = "wrong-package-token-00000000000001"
    elif mutation == "wrong_fingerprint":
        document["content_fingerprint"] = "b" * 64
    elif mutation == "bad_decision":
        decisions[0] = {"review_item_id": "synthetic-item-0001", "decision": "maybe"}
        document["decisions"] = decisions
    decision_path = write_decisions(tmp_path / "decisions.json", document)

    with pytest.raises(PrivacyValidationError) as exc_info:
        read_ambiguous_decisions(
            decision_path,
            package=package,
            private_data_dir=tmp_path,
        )

    assert exc_info.value.code == expected_code


def test_decision_import_accepts_complete_browser_export(tmp_path: Path) -> None:
    package = synthetic_review_package(item_count=2)
    decision_path = write_decisions(tmp_path / "decisions.json", decision_document(package))

    decisions = read_ambiguous_decisions(
        decision_path,
        package=package,
        private_data_dir=tmp_path,
    )

    assert decisions.package_token == package.package_token
    assert decisions.content_fingerprint == package.content_fingerprint
    assert decisions.item_count == 2
    assert [item.decision for item in decisions.decisions] == [
        AmbiguousDecisionValue.KEEP,
        AmbiguousDecisionValue.KEEP,
    ]


def test_decision_import_rejects_top_level_extra_fields(tmp_path: Path) -> None:
    package = synthetic_review_package()
    document = decision_document(package)
    document["candidate_text"] = "synthetic-prohibited-candidate-copy"
    decision_path = write_decisions(tmp_path / "decisions.json", document)

    with pytest.raises(PrivacyValidationError) as exc_info:
        read_ambiguous_decisions(
            decision_path,
            package=package,
            private_data_dir=tmp_path,
        )

    assert exc_info.value.code == "invalid_review_package"
    assert "synthetic-prohibited-candidate-copy" not in str(exc_info.value)


def test_decision_import_rejects_path_escape(tmp_path: Path) -> None:
    package = synthetic_review_package()
    private_root = tmp_path / "private"
    private_root.mkdir()
    decision_path = write_decisions(tmp_path / "outside.json", decision_document(package))

    with pytest.raises(PrivacyValidationError) as exc_info:
        read_ambiguous_decisions(
            decision_path,
            package=package,
            private_data_dir=private_root,
        )

    assert exc_info.value.code == "unsafe_review_path"


def test_decision_errors_do_not_retain_synthetic_input_traceback_locals(
    tmp_path: Path,
) -> None:
    marker = "synthetic-sensitive-decision-marker"
    package = synthetic_review_package(item_count=1)
    decision_path = write_decisions(
        tmp_path / "decisions.json",
        decision_document(
            package,
            [{"review_item_id": "synthetic-item-0001", "decision": marker}],
        ),
    )

    with pytest.raises(PrivacyValidationError) as exc_info:
        read_ambiguous_decisions(
            decision_path,
            package=package,
            private_data_dir=tmp_path,
        )

    assert marker not in str(exc_info.value)
    assert exc_info.value.__context__ is None
