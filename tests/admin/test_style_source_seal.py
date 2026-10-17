import json
import os
import stat
from datetime import UTC, datetime, timedelta

import pytest
from test_fixed_import_source import private_source
from test_local_entry import settings
from test_mapping_confirmation import confirm, preview_form, unlock
from test_runtime_terms import submit, terms_form
from test_style_decisions import review, two_candidates

from companion.admin.app import build_admin_app, load_admin_config
from companion.admin.source_seal import StyleSourceSealError, StyleSourceSealStore


def setup_store(tmp_path):
    root, source = private_source(tmp_path)
    trusted = settings(root)
    config = load_admin_config(trusted)
    digest = build_admin_app(trusted, config, port=12345)._load_batch()[1]
    return StyleSourceSealStore(trusted, config), trusted, config, source, digest


def test_seal_restarts_and_reads_full_validated_package(tmp_path):
    store, trusted, config, _, digest = setup_store(tmp_path)
    package = two_candidates()
    result = store.seal("synthetic-owner", digest, package)
    assert result.owner_id == "synthetic-owner"
    assert result.batch_digest == digest
    assert result.package == package
    reopened = StyleSourceSealStore(trusted, config)
    assert reopened.read("synthetic-owner") == result
    path = trusted.private_data_dir / "admin/style-source/current.json"
    raw = path.read_bytes()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert not (trusted.private_data_dir / "admin/style-source/.current-pending").exists()
    assert b"synthetic one" in raw
    assert b"true_name_terms" not in raw and b"approved_nicknames" not in raw
    assert b"synthetic-private-raw-value" not in raw
    assert b"approval_revision" not in raw and b"consent" not in raw


def test_no_seal_and_product_routes_never_write(tmp_path):
    store, trusted, config, _, _ = setup_store(tmp_path)
    assert store.read("synthetic-owner") is None
    app = build_admin_app(trusted, config, port=12345)
    cookie = unlock(app)
    assert confirm(app, cookie, preview_form(app, cookie))[0] == 200
    assert submit(app, cookie, terms_form(app, cookie))[0] == 200
    assert review(app, cookie)[0] == 200
    assert not (trusted.private_data_dir / "admin/style-source/current.json").exists()


@pytest.mark.parametrize("change", ["whitespace", "missing"])
def test_source_change_or_loss_blocks_read_and_new_seal(tmp_path, change):
    store, _, _, source, digest = setup_store(tmp_path)
    store.seal("synthetic-owner", digest, two_candidates())
    if change == "whitespace":
        source.write_bytes(source.read_bytes() + b" ")
    else:
        source.unlink()
    with pytest.raises(StyleSourceSealError):
        store.read("synthetic-owner")
    with pytest.raises(StyleSourceSealError):
        store.seal("synthetic-owner", digest, two_candidates())


def test_same_source_can_reseal_different_package_and_delete(tmp_path):
    store, trusted, config, _, digest = setup_store(tmp_path)
    first = two_candidates()
    second = two_candidates(datetime(2026, 9, 24, tzinfo=UTC) + timedelta(seconds=1))
    store.seal("synthetic-owner", digest, first)
    store.seal("synthetic-owner", digest, second)
    assert StyleSourceSealStore(trusted, config).read("synthetic-owner").package == second
    with pytest.raises(StyleSourceSealError):
        store.delete("forged-owner")
    assert store.read("synthetic-owner").package == second
    store.delete("synthetic-owner")
    assert store.read("synthetic-owner") is None


def test_owner_can_delete_valid_seal_after_original_source_is_missing(tmp_path):
    store, trusted, config, source, digest = setup_store(tmp_path)
    store.seal("synthetic-owner", digest, two_candidates())
    source.unlink()
    reopened = StyleSourceSealStore(trusted, config)
    assert reopened.exists_for_owner("synthetic-owner") is True
    with pytest.raises(StyleSourceSealError):
        reopened.exists_for_owner("forged-owner")
    with pytest.raises(StyleSourceSealError):
        reopened.delete("forged-owner")
    assert reopened.exists_for_owner("synthetic-owner") is True
    reopened.delete("synthetic-owner")
    assert reopened.exists_for_owner("synthetic-owner") is False


@pytest.mark.parametrize(
    "fault",
    [
        "bad_json",
        "bad_package",
        "permissions",
        "executable",
        "symlink",
        "hardlink",
        "directory_permissions",
        "directory_symlink",
        "oversize",
    ],
)
def test_unsafe_or_corrupt_seal_fails_closed(tmp_path, fault):
    store, trusted, _, _, digest = setup_store(tmp_path)
    store.seal("synthetic-owner", digest, two_candidates())
    path = trusted.private_data_dir / "admin/style-source/current.json"
    if fault == "bad_json":
        path.write_bytes(b"not-json")
    elif fault == "bad_package":
        value = json.loads(path.read_text())
        value["package"]["candidates"] = "invalid"
        path.write_text(json.dumps(value))
    elif fault == "permissions":
        path.chmod(0o644)
    elif fault == "executable":
        path.chmod(0o700)
    elif fault == "symlink":
        target = tmp_path / "synthetic-target.json"
        path.rename(target)
        path.symlink_to(target)
    elif fault == "hardlink":
        os.link(path, tmp_path / "synthetic-hardlink.json")
    elif fault == "directory_permissions":
        path.parent.chmod(0o755)
    elif fault == "directory_symlink":
        target = tmp_path / "synthetic-seal-directory"
        path.parent.rename(target)
        path.parent.symlink_to(target, target_is_directory=True)
    else:
        with path.open("wb") as output:
            output.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(StyleSourceSealError):
        store.read("synthetic-owner")
    with pytest.raises(StyleSourceSealError):
        store.seal("synthetic-owner", digest, two_candidates())


def test_wrong_owner_and_bad_caller_package_fail_closed(tmp_path):
    store, _, _, _, digest = setup_store(tmp_path)
    with pytest.raises(StyleSourceSealError):
        store.seal("forged-owner", digest, object())
    store.seal("synthetic-owner", digest, two_candidates())
    with pytest.raises(StyleSourceSealError):
        store.read("forged-owner")
    with pytest.raises(StyleSourceSealError):
        store.delete("forged-owner")


def test_source_changes_during_read_fail_closed(tmp_path, monkeypatch):
    store, _, _, source, digest = setup_store(tmp_path)
    store.seal("synthetic-owner", digest, two_candidates())
    original = store._current_digest
    calls = []

    def changing():
        calls.append(True)
        if len(calls) == 2:
            source.write_bytes(source.read_bytes() + b" ")
        return original()

    monkeypatch.setattr(store, "_current_digest", changing)
    with pytest.raises(StyleSourceSealError):
        store.read("synthetic-owner")


def test_readback_failure_rolls_back_previous_seal(tmp_path, monkeypatch):
    store, trusted, config, _, digest = setup_store(tmp_path)
    first = two_candidates()
    store.seal("synthetic-owner", digest, first)
    original = store._read_path
    calls = []

    def fail_once():
        calls.append(True)
        if len(calls) == 2:
            raise OSError("synthetic readback failure")
        return original()

    monkeypatch.setattr(store, "_read_path", fail_once)
    with pytest.raises(StyleSourceSealError):
        store.seal(
            "synthetic-owner",
            digest,
            two_candidates(datetime(2026, 9, 24, tzinfo=UTC) + timedelta(seconds=1)),
        )
    assert StyleSourceSealStore(trusted, config).read("synthetic-owner").package == first


def test_write_failure_after_replace_recovers_previous_seal(tmp_path, monkeypatch):
    store, trusted, config, _, digest = setup_store(tmp_path)
    first = two_candidates()
    store.seal("synthetic-owner", digest, first)
    original = store._atomic_write
    calls = []

    def fail_after_publish(raw):
        calls.append(True)
        original(raw)
        if len(calls) == 1:
            raise OSError("synthetic fsync failure")

    monkeypatch.setattr(store, "_atomic_write", fail_after_publish)
    with pytest.raises(StyleSourceSealError):
        store.seal(
            "synthetic-owner",
            digest,
            two_candidates(datetime(2026, 9, 24, tzinfo=UTC) + timedelta(seconds=1)),
        )
    assert StyleSourceSealStore(trusted, config).read("synthetic-owner").package == first


def test_double_write_recovery_failure_clears_new_publication(tmp_path, monkeypatch):
    store, trusted, config, _, digest = setup_store(tmp_path)
    store.seal("synthetic-owner", digest, two_candidates())
    original = store._atomic_write
    calls = []

    def fail_both(raw):
        calls.append(True)
        if len(calls) == 1:
            original(raw)
        raise OSError("synthetic write failure")

    monkeypatch.setattr(store, "_atomic_write", fail_both)
    with pytest.raises(StyleSourceSealError):
        store.seal(
            "synthetic-owner",
            digest,
            two_candidates(datetime(2026, 9, 24, tzinfo=UTC) + timedelta(seconds=1)),
        )
    assert StyleSourceSealStore(trusted, config).read("synthetic-owner") is None


def test_undeletable_new_publication_keeps_pending_marker_and_blocks_restart(tmp_path, monkeypatch):
    store, trusted, config, _, digest = setup_store(tmp_path)
    store.seal("synthetic-owner", digest, two_candidates())
    original_write = store._atomic_write
    calls = []

    def fail_both(raw):
        calls.append(True)
        if len(calls) == 1:
            original_write(raw)
        raise OSError("synthetic write failure")

    current = trusted.private_data_dir / "admin/style-source/current.json"
    original_unlink = type(current).unlink

    def cannot_unlink(path, *args, **kwargs):
        if path == current:
            raise PermissionError("synthetic unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(store, "_atomic_write", fail_both)
    monkeypatch.setattr(type(current), "unlink", cannot_unlink)
    with pytest.raises(StyleSourceSealError):
        store.seal(
            "synthetic-owner",
            digest,
            two_candidates(datetime(2026, 9, 24, tzinfo=UTC) + timedelta(seconds=1)),
        )
    assert current.exists()
    marker = trusted.private_data_dir / "admin/style-source/.current-pending"
    assert marker.exists() and marker.stat().st_size == 0
    with pytest.raises(StyleSourceSealError):
        StyleSourceSealStore(trusted, config).read("synthetic-owner")


def test_source_changes_after_publish_restores_previous_package(tmp_path, monkeypatch):
    store, trusted, config, source, digest = setup_store(tmp_path)
    old_source = source.read_bytes()
    first = two_candidates()
    store.seal("synthetic-owner", digest, first)
    path = trusted.private_data_dir / "admin/style-source/current.json"
    previous_raw = path.read_bytes()
    original = store._current_digest
    calls = []

    def changing():
        calls.append(True)
        if len(calls) == 2:
            source.write_bytes(old_source + b" ")
        return original()

    monkeypatch.setattr(store, "_current_digest", changing)
    with pytest.raises(StyleSourceSealError):
        store.seal(
            "synthetic-owner",
            digest,
            two_candidates(datetime(2026, 9, 24, tzinfo=UTC) + timedelta(seconds=1)),
        )
    assert path.read_bytes() == previous_raw
    with pytest.raises(StyleSourceSealError):
        StyleSourceSealStore(trusted, config).read("synthetic-owner")
