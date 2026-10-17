import os
from pathlib import Path

import pytest
from test_local_entry import call, config, settings
from test_mapping_confirmation import confirm, preview_form, unlock

from companion.admin.app import AdminConfigError, build_admin_app, load_admin_config
from companion.importer import StandardJsonAdapter

FIXTURES = {
    "ciphertalk_json": Path("fixtures/synthetic/ciphertalk_private_chat.json"),
    "standard_json": Path("fixtures/synthetic/standard_private_chat.json"),
}


def private_source(tmp_path, import_format="standard_json"):
    root = tmp_path / "private"
    config(
        root,
        f'enabled = true\nowner_id = "synthetic-owner"\nimport_format = "{import_format}"\n',
    )
    folder = root / "admin" / "import"
    folder.mkdir(mode=0o700)
    path = folder / "current.json"
    path.write_bytes(FIXTURES[import_format].read_bytes())
    path.chmod(0o600)
    return root, path


def product_app(root):
    trusted_settings = settings(root)
    loaded = load_admin_config(trusted_settings)
    return build_admin_app(trusted_settings, loaded, port=12345)


@pytest.mark.parametrize("import_format", ["ciphertalk_json", "standard_json"])
def test_product_assembly_previews_and_confirms_fixed_synthetic_source(tmp_path, import_format):
    root, _ = private_source(tmp_path, import_format)
    app = product_app(root)
    cookie = unlock(app)
    form = preview_form(app, cookie)
    status, _, page = confirm(app, cookie, form)
    assert status == 200 and "映射已确认" in page
    records = app._sessions[cookie.split("=", 1)[1]].confirmed_records
    assert records is not None and len(records) == 3
    assert "synthetic-private-raw-value" not in page


def test_old_config_has_no_source_and_rejects_preview(tmp_path):
    root = tmp_path / "private"
    config(root)
    app = product_app(root)
    cookie = unlock(app)
    assert call(app, "/admin/import/preview", cookie=cookie)[0] == 503


@pytest.mark.parametrize("value", ["../other", "local_send", "", "STANDARD_JSON"])
def test_invalid_import_format_fails_before_assembly(tmp_path, value):
    root = tmp_path / "private"
    config(root, f'enabled = true\nowner_id = "synthetic-owner"\nimport_format = "{value}"\n')
    with pytest.raises(AdminConfigError):
        load_admin_config(settings(root))


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "file_permissions",
        "directory_permissions",
        "directory_symlink",
        "symlink",
        "hardlink",
        "oversize",
        "bad_json",
    ],
)
def test_fixed_source_failures_do_not_expose_content_or_path(tmp_path, fault):
    root, path = private_source(tmp_path)
    if fault == "missing":
        path.unlink()
    elif fault == "file_permissions":
        path.chmod(0o644)
    elif fault == "directory_permissions":
        path.parent.chmod(0o755)
    elif fault == "directory_symlink":
        relocated = tmp_path / "synthetic-import-dir"
        path.parent.rename(relocated)
        path.parent.symlink_to(relocated, target_is_directory=True)
    elif fault == "symlink":
        outside = tmp_path / "synthetic-outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif fault == "hardlink":
        os.link(path, tmp_path / "synthetic-hardlink.json")
    elif fault == "oversize":
        with path.open("wb") as output:
            output.truncate(64 * 1024 * 1024 + 1)
    else:
        path.write_text('{"SYNTHETIC_SECRET_SENTINEL":')
    app = product_app(root)
    cookie = unlock(app)
    status, _, page = call(app, "/admin/import/preview", cookie=cookie)
    assert status == 503
    assert "SYNTHETIC_SECRET_SENTINEL" not in page
    assert str(path) not in page


def test_parser_mutation_during_preview_and_source_change_before_confirm(tmp_path, monkeypatch):
    root, path = private_source(tmp_path)
    app = product_app(root)
    cookie = unlock(app)
    original = StandardJsonAdapter.parse

    def mutate(self, source):
        result = original(self, source)
        path.write_bytes(path.read_bytes())
        return result

    monkeypatch.setattr(StandardJsonAdapter, "parse", mutate)
    status, _, page = call(app, "/admin/import/preview", cookie=cookie)
    assert status == 503 and str(path) not in page
    monkeypatch.setattr(StandardJsonAdapter, "parse", original)
    form = preview_form(app, cookie)
    path.write_bytes(path.read_bytes() + b" ")
    status, _, page = confirm(app, cookie, form)
    assert status == 409 and str(path) not in page
    assert app._sessions[cookie.split("=", 1)[1]].confirmed_records is None
