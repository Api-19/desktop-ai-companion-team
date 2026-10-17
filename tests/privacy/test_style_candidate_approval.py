import json
import os
import re
from datetime import UTC, datetime

import pytest

from companion.privacy import PrivacyValidationError
from companion.privacy.candidate_models import (
    CandidateCategory,
    CandidateSelectionAudit,
    SafeCandidatePackage,
    SafeStyleCandidate,
)
from companion.privacy.style_candidate_approval import (
    APPROVAL_RELATIVE_PATH,
    StyleCandidateApprovalStore,
)
from companion.runtime.settings import Settings


def package(*texts: str) -> SafeCandidatePackage:
    now = datetime(2026, 9, 24, tzinfo=UTC)
    return SafeCandidatePackage(
        created_at=now,
        candidates=[
            SafeStyleCandidate(
                text=text,
                category=CandidateCategory.GREETING,
                scene_tags=[CandidateCategory.GREETING.value],
            )
            for text in texts
        ],
        audit=CandidateSelectionAudit(
            processed_at=now,
            input_count=len(texts) + 1,
            owner_input_count=len(texts),
            dropped_other_count=1,
            allowed_category_counts={"greeting": len(texts)} if texts else {},
            rejection_reason_counts={},
            unknown_count=0,
            deduplicated_count=0,
            output_count=len(texts),
            succeeded=True,
        ),
    )


@pytest.fixture
def store(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    settings = Settings(
        environment="test",
        log_level="ERROR",
        private_data_dir=private,
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )
    return StyleCandidateApprovalStore(settings)


def test_absent_sidecar_and_zero_candidates_are_unapproved(store):
    assert store.approved_indices("synthetic-owner", package("合成问候甲")) == ()
    assert store.approved_indices("synthetic-owner", package()) == ()
    assert store.approval_witness("synthetic-owner", package("合成问候甲")) is None
    assert not (store.private_data_dir / APPROVAL_RELATIVE_PATH).exists()
    with pytest.raises(PrivacyValidationError):
        store.set_approved("synthetic-owner", package(), index=0, approved=True)


def test_single_item_approval_revoke_view_and_delete(store):
    current = package("合成问候甲", "合成问候乙")
    assert store.set_approved("synthetic-owner", current, index=1, approved=True) == (1,)
    assert store.approved_indices("synthetic-owner", current) == (1,)
    assert store.decisions("synthetic-owner", current) == ((0, "pending"), (1, "approved"))
    path = store.private_data_dir / APPROVAL_RELATIVE_PATH
    payload = path.read_text(encoding="utf-8")
    assert "合成问候" not in payload
    assert "other" not in payload
    assert json.loads(payload)["owner_id"] == "synthetic-owner"
    assert store.set_approved("synthetic-owner", current, index=1, approved=False) == ()
    assert store.approved_indices("synthetic-owner", current) == ()
    store.delete("synthetic-owner", current)
    assert not path.exists()
    assert store.approved_indices("synthetic-owner", current) == ()


def test_each_decision_has_fresh_witness_even_for_same_approved_set(store):
    current = package("合成问候甲", "合成问候乙")
    assert store.approval_witness("synthetic-owner", current) is None
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    first = store.approval_witness("synthetic-owner", current)
    assert first is not None
    assert re.fullmatch(r"v1-[0-9a-f]{64}", first.revision)
    assert re.fullmatch(r"[0-9a-f]{64}", first.package_digest)
    assert first.approved_indices == (0,)
    assert set(vars(first)) == {"package_digest", "approved_indices", "revision"}
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    repeated = store.approval_witness("synthetic-owner", current)
    assert repeated.approved_indices == (0,)
    assert repeated.revision != first.revision
    store.set_approved("synthetic-owner", current, index=0, approved=False)
    revoked = store.approval_witness("synthetic-owner", current)
    assert revoked.approved_indices == ()
    assert revoked.revision != repeated.revision
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    restored = store.approval_witness("synthetic-owner", current)
    assert restored.approved_indices == (0,)
    assert restored.revision not in {first.revision, repeated.revision, revoked.revision}
    store.delete("synthetic-owner", current)
    assert store.approval_witness("synthetic-owner", current) is None
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    assert store.approval_witness("synthetic-owner", current).revision != restored.revision


def test_legacy_sidecar_requires_explicit_clear_and_new_decision(store):
    current = package("合成问候甲")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    path = store.private_data_dir / APPROVAL_RELATIVE_PATH
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy.pop("revision")
    path.write_text(json.dumps(legacy), encoding="utf-8")
    for read in (store.approval_witness, store.decisions, store.approved_indices):
        with pytest.raises(PrivacyValidationError):
            read("synthetic-owner", current)
    with pytest.raises(PrivacyValidationError):
        store.set_approved("synthetic-owner", current, index=0, approved=True)
    store.delete_stale("synthetic-owner")
    assert store.approval_witness("synthetic-owner", current) is None
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    assert store.approval_witness("synthetic-owner", current) is not None


@pytest.mark.parametrize("revision", [None, "", "v1-" + "G" * 64, "v1-" + "0" * 63])
def test_invalid_revision_rejects_witness_and_decisions(store, revision):
    current = package("合成问候甲")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    path = store.private_data_dir / APPROVAL_RELATIVE_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    data["revision"] = revision
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PrivacyValidationError):
        store.approval_witness("synthetic-owner", current)
    with pytest.raises(PrivacyValidationError):
        store.decisions("synthetic-owner", current)


def test_witness_wrong_owner_or_package_fails_closed(store):
    current = package("合成问候甲")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    for owner, candidate in (
        ("other-owner", current),
        ("synthetic-owner", package("合成问候乙")),
    ):
        with pytest.raises(PrivacyValidationError):
            store.approval_witness(owner, candidate)


def test_owner_and_complete_package_binding(store):
    current = package("合成问候甲", "合成问候乙")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    for owner, candidate in (
        ("forged-owner", current),
        ("synthetic-owner", package("合成问候乙", "合成问候甲")),
        ("synthetic-owner", package("合成问候甲")),
        ("synthetic-owner", package("合成问候甲", "合成问候丙")),
    ):
        with pytest.raises(PrivacyValidationError):
            store.approved_indices(owner, candidate)
        with pytest.raises(PrivacyValidationError):
            store.set_approved(owner, candidate, index=0, approved=True)
    assert store.approved_indices("synthetic-owner", current) == (0,)


def test_old_package_requires_explicit_owner_bound_deletion(store):
    original = package("合成问候甲")
    changed = package("合成问候乙")
    store.set_approved("synthetic-owner", original, index=0, approved=True)
    old_revision = store.approval_witness("synthetic-owner", original).revision
    with pytest.raises(PrivacyValidationError):
        store.approved_indices("synthetic-owner", changed)
    with pytest.raises(PrivacyValidationError):
        store.delete_stale("forged-owner")
    store.delete_stale("synthetic-owner")
    assert store.approval_witness("synthetic-owner", changed) is None
    assert store.approved_indices("synthetic-owner", changed) == ()
    store.set_approved("synthetic-owner", changed, index=0, approved=True)
    assert store.approval_witness("synthetic-owner", changed).revision != old_revision


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "bad_state", "bad_digest", "bad_json"]
)
def test_corrupt_sidecar_fails_closed(store, mutation):
    current = package("合成问候甲", "合成问候乙")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    path = store.private_data_dir / APPROVAL_RELATIVE_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        data["items"].pop()
    elif mutation == "duplicate":
        data["items"].append(data["items"][0])
    elif mutation == "bad_state":
        data["items"][0]["state"] = "unknown"
    elif mutation == "bad_digest":
        data["items"][0]["digest"] = "0" * 64
    payload = "{" if mutation == "bad_json" else json.dumps(data)
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(PrivacyValidationError):
        store.approved_indices("synthetic-owner", current)
    with pytest.raises(PrivacyValidationError):
        store.approval_witness("synthetic-owner", current)


def test_bad_permissions_and_links_rejected(store):
    current = package("合成问候甲")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    path = store.private_data_dir / APPROVAL_RELATIVE_PATH
    os.chmod(path, 0o644)
    with pytest.raises(PrivacyValidationError):
        store.approved_indices("synthetic-owner", current)
    os.chmod(path, 0o600)
    second = path.with_name("second")
    os.link(path, second)
    with pytest.raises(PrivacyValidationError):
        store.approved_indices("synthetic-owner", current)
    second.unlink()
    path.unlink()
    path.symlink_to(store.private_data_dir / "outside")
    with pytest.raises(PrivacyValidationError):
        store.approved_indices("synthetic-owner", current)


def test_failed_readback_does_not_publish_new_approval(store, monkeypatch):
    current = package("合成问候甲", "合成问候乙")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    old_witness = store.approval_witness("synthetic-owner", current)
    from companion.privacy import style_candidate_approval

    original = style_candidate_approval._read_sidecar
    calls = 0

    def fail_post_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic readback failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(style_candidate_approval, "_read_sidecar", fail_post_write)
    with pytest.raises(PrivacyValidationError):
        store.set_approved("synthetic-owner", current, index=1, approved=True)
    monkeypatch.setattr(style_candidate_approval, "_read_sidecar", original)
    assert store.approved_indices("synthetic-owner", current) == (0,)
    assert store.approval_witness("synthetic-owner", current) == old_witness


def test_first_write_readback_failure_leaves_no_sidecar(store, monkeypatch):
    current = package("合成问候甲")
    from companion.privacy import style_candidate_approval

    original = style_candidate_approval._read_sidecar
    calls = 0

    def fail_post_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic readback failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(style_candidate_approval, "_read_sidecar", fail_post_write)
    with pytest.raises(PrivacyValidationError):
        store.set_approved("synthetic-owner", current, index=0, approved=True)
    monkeypatch.setattr(style_candidate_approval, "_read_sidecar", original)
    assert store.approved_indices("synthetic-owner", current) == ()
    assert store.approval_witness("synthetic-owner", current) is None
    assert not (store.private_data_dir / APPROVAL_RELATIVE_PATH).exists()


def test_write_exception_after_replace_restores_previous_decisions(store, monkeypatch):
    current = package("合成问候甲", "合成问候乙")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    from companion.privacy import style_candidate_approval

    original = style_candidate_approval._atomic_write
    calls = 0

    def fail_after_replace(path, raw):
        nonlocal calls
        calls += 1
        original(path, raw)
        if calls == 1:
            raise OSError("synthetic write failure")

    monkeypatch.setattr(style_candidate_approval, "_atomic_write", fail_after_replace)
    with pytest.raises(PrivacyValidationError):
        store.set_approved("synthetic-owner", current, index=1, approved=True)
    monkeypatch.setattr(style_candidate_approval, "_atomic_write", original)
    assert store.approved_indices("synthetic-owner", current) == (0,)


def test_readback_and_rollback_failure_removes_new_approval(store, monkeypatch):
    current = package("合成问候甲", "合成问候乙")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    from companion.privacy import style_candidate_approval

    original_read = style_candidate_approval._read_sidecar
    original_write = style_candidate_approval._atomic_write
    reads = 0
    writes = 0

    def fail_readback(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            raise OSError("synthetic readback failure")
        return original_read(*args, **kwargs)

    def fail_rollback(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic rollback failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(style_candidate_approval, "_read_sidecar", fail_readback)
    monkeypatch.setattr(style_candidate_approval, "_atomic_write", fail_rollback)
    with pytest.raises(PrivacyValidationError):
        store.set_approved("synthetic-owner", current, index=1, approved=True)
    monkeypatch.setattr(style_candidate_approval, "_read_sidecar", original_read)
    monkeypatch.setattr(style_candidate_approval, "_atomic_write", original_write)
    assert store.approved_indices("synthetic-owner", current) == ()
    assert store.approval_witness("synthetic-owner", current) is None
    assert not (store.private_data_dir / APPROVAL_RELATIVE_PATH).exists()


def test_directory_permissions_and_link_fail_closed(store):
    current = package("合成问候甲")
    store.set_approved("synthetic-owner", current, index=0, approved=True)
    directory = (store.private_data_dir / APPROVAL_RELATIVE_PATH).parent
    os.chmod(directory, 0o755)
    with pytest.raises(PrivacyValidationError):
        store.approved_indices("synthetic-owner", current)
    os.chmod(directory, 0o700)
    path = directory / "current.json"
    path.unlink()
    directory.rmdir()
    directory.symlink_to(store.private_data_dir)
    with pytest.raises(PrivacyValidationError):
        store.approved_indices("synthetic-owner", current)


def test_invalid_owner_index_and_package_are_rejected(store):
    current = package("合成问候甲")
    for owner in ("", "x" * 121, " owner", "owner ", "owner\n", "owner\x00"):
        with pytest.raises(PrivacyValidationError):
            store.approved_indices(owner, current)
    for index in (-1, 1, True):
        with pytest.raises(PrivacyValidationError):
            store.set_approved("synthetic-owner", current, index=index, approved=True)
    forged = current.model_copy(deep=True)
    forged.audit.output_count = 2
    with pytest.raises(PrivacyValidationError):
        store.set_approved("synthetic-owner", forged, index=0, approved=True)
    assert not (store.private_data_dir / APPROVAL_RELATIVE_PATH).exists()


def test_private_root_ancestor_symlink_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    private = actual / "private"
    private.mkdir(mode=0o700)
    link = tmp_path / "alias"
    link.symlink_to(actual, target_is_directory=True)
    settings = Settings(
        environment="test",
        log_level="ERROR",
        private_data_dir=link / "private",
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )
    with pytest.raises(PrivacyValidationError):
        StyleCandidateApprovalStore(settings)


def test_public_error_omits_owner_text_and_path_from_traceback(store):
    owner = "synthetic-owner-sensitive-marker"
    text = "synthetic-candidate-sensitive-marker"
    current = package(text)
    store.set_approved(owner, current, index=0, approved=True)
    with pytest.raises(PrivacyValidationError) as captured:
        store.approved_indices("forged-owner-sensitive-marker", current)
    error = captured.value
    assert error.__cause__ is None
    assert error.__context__ is None
    trace = error.__traceback__
    while trace is not None:
        if "companion/privacy" in trace.tb_frame.f_code.co_filename:
            local_values = repr(trace.tb_frame.f_locals)
            assert owner not in local_values
            assert text not in local_values
            assert str(store.private_data_dir) not in local_values
        trace = trace.tb_next
