import pytest
from test_bound_consent import sealed_ready
from test_local_entry import call
from test_mapping_confirmation import confirm, preview_form, unlock
from test_memory_management import FIELDS, factory_app, grant, request
from test_profile_and_evidence import consented, generate
from test_retain_intent import form, post
from test_runtime_terms import ready as runtime_ready
from test_runtime_terms import submit, terms_form
from test_usage_page import app_with, snapshot


@pytest.mark.parametrize(
    "path",
    [
        "/admin/import/preview",
        "/admin/style/terms",
        "/admin/style/review",
        "/admin/style/clear",
        "/admin/style/retain",
        "/admin/style/consent",
        "/admin/style/profile",
        "/admin/style/evidence",
        "/admin/style/revoke",
        "/admin/style/delete",
        "/admin/style/delete-source",
        "/admin/memory",
        "/admin/memory/consent",
        "/admin/memory/revoke",
        "/admin/usage",
        "/admin/character",
    ],
)
def test_authenticated_pages_and_unavailable_pages_offer_navigation(tmp_path, path):
    app, cookie, _, _, _ = factory_app(tmp_path)
    status, _, page = request(app, path, cookie=cookie)
    assert status in {200, 503}
    assert '<nav aria-label="管理导航">' in page
    assert '<a href="/admin/home">返回主页</a>' in page


def test_locked_page_does_not_offer_authenticated_navigation():
    app = app_with(snapshot)
    status, _, page = call(app, "/admin/style/review")
    assert status == 403
    assert 'aria-label="管理导航"' not in page


def test_memory_result_navigation_and_response_length(tmp_path):
    app, cookie, _, _, _ = factory_app(tmp_path)
    status, headers, page = grant(app, cookie, FIELDS)
    assert status == 200
    assert '<nav aria-label="管理导航">' in page
    assert '<a href="/admin/memory">返回记忆管理</a>' in page
    assert int(headers["Content-Length"]) == len(page.encode())


def test_style_error_navigation_has_correct_length():
    app = app_with(snapshot)
    cookie = unlock(app)
    status, headers, page = call(app, "/admin/style/terms", cookie=cookie)
    assert status == 503
    assert '<a href="/admin/style/review">返回候选审查</a>' in page
    assert int(headers["Content-Length"]) == len(page.encode())


def test_home_offers_import_preview():
    app = app_with(snapshot)
    cookie = unlock(app)
    status, _, page = call(app, "/admin/home", cookie=cookie)
    assert status == 200
    assert '<a href="/admin/import/preview">导入预览与说话人映射</a>' in page


def test_home_exposes_every_primary_workflow_entry_without_submitting():
    app = app_with(snapshot)
    cookie = unlock(app)
    status, _, page = call(app, "/admin/home", cookie=cookie)
    assert status == 200
    for path in (
        "/admin/import/preview",
        "/admin/style/terms",
        "/admin/style/review",
        "/admin/style/retain",
        "/admin/style/consent",
        "/admin/style/profile",
        "/admin/style/evidence",
        "/admin/style/clear",
        "/admin/style/revoke",
        "/admin/style/delete",
        "/admin/style/delete-source",
        "/admin/memory",
        "/admin/memory/consent",
        "/admin/memory/revoke",
        "/admin/usage",
        "/admin/character",
        "/admin/style/snippets",
    ):
        assert f'href="{path}"' in page
    assert "按顺序完成" in page
    assert "<h2>撤权与删除</h2>" in page
    assert "<form" not in page


def test_terms_success_offers_next_step_without_creating_candidates(tmp_path):
    app, cookie, _, calls, _ = runtime_ready(tmp_path)
    status, _, page = confirm(app, cookie, preview_form(app, cookie))
    assert status == 200
    assert '<a href="/admin/style/terms">下一步: 隐私词设置</a>' in page
    status, _, page = submit(app, cookie, terms_form(app, cookie))
    assert status == 200
    assert '<a href="/admin/style/review">下一步: 候选审查</a>' in page
    assert calls == []


def test_retain_success_offers_independent_consent(tmp_path):
    app, cookie, _, _, _, _ = sealed_ready(tmp_path)
    _, _, values = form(app, cookie, "/admin/style/retain", "retain")
    status, _, page = post(app, cookie, "/admin/style/retain", values)
    assert status == 200
    assert '<a href="/admin/style/consent">下一步: 画像保存同意</a>' in page
    assert app._phase1_store.view_consent("synthetic-owner") is None


def test_clear_success_returns_to_review_without_approving(tmp_path):
    app, cookie, package, _, _, _ = sealed_ready(tmp_path)
    _, _, values = form(app, cookie, "/admin/style/clear", "clear")
    status, _, page = post(app, cookie, "/admin/style/clear", values)
    assert status == 200
    assert '<a href="/admin/style/review">下一步: 重新审查候选</a>' in page
    assert all(
        state == "pending" for _, state in app._approval_store.decisions("synthetic-owner", package)
    )


def test_consent_success_offers_profile_without_generating_it(tmp_path):
    app, cookie, _, _, _, _ = sealed_ready(tmp_path)
    _, _, values = form(app, cookie, "/admin/style/consent", "consent")
    status, _, page = post(app, cookie, "/admin/style/consent", values)
    assert status == 200
    assert '<a href="/admin/style/profile">下一步: 生成风格画像</a>' in page
    assert not any(path.exists() for path in app._phase1_store.derived_artifact_paths)


def test_profile_success_offers_evidence_without_publishing_it(tmp_path):
    app, cookie, _, _, _, _ = consented(tmp_path)
    status, _, page = generate(app, cookie)
    assert status == 200
    assert '<a href="/admin/style/evidence">下一步: 表达选择与发布</a>' in page
    assert not any(path.exists() for path in app._phase5_store.artifact_paths)


def test_memory_list_offers_return_home(tmp_path):
    app, cookie, _, _, _ = factory_app(tmp_path)
    status, _, page = request(app, "/admin/memory", cookie=cookie)
    assert status == 200
    assert '<a href="/admin/home">返回主页</a>' in page
