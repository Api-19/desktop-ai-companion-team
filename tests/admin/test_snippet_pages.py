# ruff: noqa: RUF001
import json
import re
from datetime import UTC, datetime, timedelta

import pytest
from test_local_entry import call, config, settings
from test_mapping_confirmation import ORIGIN, confirm, preview_form, unlock

import companion.admin.snippets as snippets_module
from companion.admin.app import AdminApp, build_admin_app, load_admin_config
from companion.contracts import ChatRecord, SpeakerRole
from companion.importer import ImportBatch, MessageDirection, SourceChatRecord

PHRASES = [
    "今天下雨了吗",
    "晚饭吃什么呀",
    "周末去爬山",
    "电影好看吗",
    "猫咪又调皮",
    "明天早点起",
    "新买的杯子",
    "公园花开了",
    "听了一首歌",
    "咖啡太苦啦",
    "地铁好挤啊",
    "书看完了吗",
    "楼下有烧烤",
    "月亮好圆呀",
    "风筝飞走了",
    "作业写完没",
    "蛋糕烤焦了",
    "晚上看星星",
    "鞋子湿透了",
    "<b>茶凉了</b>",
]
BASE = datetime(2026, 9, 1, tzinfo=UTC)
NOW = datetime(2026, 10, 6, tzinfo=UTC)
SYNTHETIC_NAME = "王小明"
FILES = ("terms.json", "library.jsonl", "holdout.jsonl", "audit_sample.jsonl")
SNIPPET_FILES = FILES[1:]


def snippet_batch() -> ImportBatch:
    records = []
    for index, phrase in enumerate(PHRASES):
        started = BASE + timedelta(hours=2 * index)
        records.append(
            SourceChatRecord(
                conversation_id="synthetic-chat",
                message_id=f"p{index}",
                timestamp=started,
                source_speaker_id="synthetic-other",
                source_speaker_name="Other",
                direction=MessageDirection.INBOUND,
                content_type="text",
                text=f"{SYNTHETIC_NAME}说{phrase}",
            )
        )
        records.append(
            SourceChatRecord(
                conversation_id="synthetic-chat",
                message_id=f"o{index}",
                timestamp=started + timedelta(minutes=1),
                source_speaker_id="synthetic-owner",
                source_speaker_name="Owner",
                direction=MessageDirection.OUTBOUND,
                content_type="text",
                text=f"SYNTHETIC_OWNER_REPLY_{index}",
            )
        )
    return ImportBatch(records=records, issues=[])


def make_app(tmp_path, *, snippet_root="default") -> AdminApp:
    root = tmp_path / "private" / "style" / "snippets" if snippet_root == "default" else None
    return AdminApp(
        "synthetic-owner",
        host="127.0.0.1",
        port=12345,
        batch_loader=snippet_batch,
        snippet_root=root,
        wall_clock=lambda: NOW,
    )


def mapped(tmp_path, **kwargs):
    app = make_app(tmp_path, **kwargs)
    cookie = unlock(app)
    status, _, _ = confirm(app, cookie, preview_form(app, cookie))
    assert status == 200
    return app, cookie


def root_of(app: AdminApp):
    return app._snippet_root


def snippet_form(app: AdminApp, cookie: str, **terms: str) -> dict[str, str]:
    status, _, page = call(app, "/admin/style/snippets", cookie=cookie)
    assert status == 200
    values = {
        "csrf": re.search(r'name="csrf" value="([^"]+)"', page).group(1),
        "snippets_token": re.search(r'name="snippets_token" value="([^"]+)"', page).group(1),
        "name": SYNTHETIC_NAME,
        "address": "",
        "account": "",
        "private_event": "",
        "intimate": "宝贝",
        "confirm": "yes",
    }
    values.update(terms)
    return values


def submit(app: AdminApp, cookie: str, form: dict[str, str]):
    return call(
        app, "/admin/style/snippets", method="POST", cookie=cookie, origin=ORIGIN, form=form
    )


def generated(tmp_path):
    app, cookie = mapped(tmp_path)
    status, _, page = submit(app, cookie, snippet_form(app, cookie))
    assert status == 200
    return app, cookie, page


def lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit_form(app: AdminApp, cookie: str) -> tuple[str, dict[str, str]]:
    status, _, page = call(app, "/admin/style/snippets/audit", cookie=cookie)
    assert status == 200
    return page, {
        "csrf": re.search(r'name="csrf" value="([^"]+)"', page).group(1),
        "audit_token": re.search(r'name="audit_token" value="([^"]+)"', page).group(1),
        "from": "",
        "to": "",
    }


def audit_submit(app: AdminApp, cookie: str, form: dict[str, str]):
    return call(
        app,
        "/admin/style/snippets/audit",
        method="POST",
        cookie=cookie,
        origin=ORIGIN,
        form=form,
    )


@pytest.mark.parametrize("path", ["/admin/style/snippets", "/admin/style/snippets/audit"])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_locked_access_is_rejected(tmp_path, path, method):
    app = make_app(tmp_path)
    status, _, page = call(app, path, method=method, origin=ORIGIN)
    assert status == 403
    assert "片段" not in page


def test_without_mapping_points_to_import_preview(tmp_path):
    app = make_app(tmp_path)
    cookie = unlock(app)
    status, _, page = call(app, "/admin/style/snippets", cookie=cookie)
    assert status == 503
    assert '<a href="/admin/import/preview">请先完成说话人映射</a>' in page
    assert "snippets_token" not in page


@pytest.mark.parametrize("path", ["/admin/style/snippets", "/admin/style/snippets/audit"])
def test_missing_snippet_root_is_unavailable(tmp_path, path):
    app, cookie = mapped(tmp_path, snippet_root=None)
    status, _, page = call(app, path, cookie=cookie)
    assert status == 503
    assert "片段功能未就绪" in page


def test_product_assembly_uses_private_snippet_directory(tmp_path):
    root = tmp_path / "private"
    config(root)
    trusted = settings(root)
    app = build_admin_app(trusted, load_admin_config(trusted), port=12345)
    assert app._snippet_root == root / "style" / "snippets"


def test_form_shows_zero_counts_and_all_fields(tmp_path):
    app, cookie = mapped(tmp_path)
    status, _, page = call(app, "/admin/style/snippets", cookie=cookie)
    assert status == 200
    for field in ("name", "address", "account", "private_event", "intimate"):
        assert f'<textarea name="{field}">' in page
    assert 'name="confirm" value="yes"' in page
    assert "会覆盖现有片段库与留出集" in page
    assert "片段库：0" in page and "留出集：0" in page and "审核样本：0" in page
    assert not root_of(app).exists()


def test_success_writes_private_files_and_only_counts(tmp_path):
    app, _, page = generated(tmp_path)
    root = root_of(app)
    assert root.stat().st_mode & 0o777 == 0o700
    for name in FILES:
        assert (root / name).stat().st_mode & 0o777 == 0o600
    library = lines(root / "library.jsonl")
    holdout = lines(root / "holdout.jsonl")
    audit = lines(root / "audit_sample.jsonl")
    assert (len(library), len(holdout), len(audit)) == (18, 2, 18)
    assert "片段：20" in page
    assert "片段库：18" in page
    assert "留出集：2" in page
    assert "审核样本：18" in page
    assert '<a href="/admin/style/snippets/audit">下一步：抽样审核</a>' in page
    assert SYNTHETIC_NAME not in page
    assert "SYNTHETIC_OWNER_REPLY" not in page
    assert not any(phrase in page for phrase in PHRASES)
    assert "[NAME]" not in page
    terms = json.loads((root / "terms.json").read_text(encoding="utf-8"))
    assert terms == {"redaction_terms": {"name": [SYNTHETIC_NAME]}, "intimate_terms": ["宝贝"]}


def test_privacy_terms_are_replaced_in_every_snippet_file(tmp_path):
    app, _, _ = generated(tmp_path)
    root = root_of(app)
    for name in SNIPPET_FILES:
        raw = (root / name).read_text(encoding="utf-8")
        assert SYNTHETIC_NAME not in raw
        assert "小甲" in raw


def test_counts_on_form_match_files(tmp_path):
    app, cookie, _ = generated(tmp_path)
    status, _, page = call(app, "/admin/style/snippets", cookie=cookie)
    assert status == 200
    assert "片段库：18" in page and "留出集：2" in page and "审核样本：18" in page


def test_form_prefills_saved_terms_escaped(tmp_path):
    app, cookie = mapped(tmp_path)
    form = snippet_form(app, cookie, address="<i>甲街</i>\n乙路", intimate="宝贝\n<b>&心</b>")
    status, _, _ = submit(app, cookie, form)
    assert status == 200
    status, _, page = call(app, "/admin/style/snippets", cookie=cookie)
    assert status == 200
    assert '<textarea name="address">&lt;i&gt;甲街&lt;/i&gt;\n乙路</textarea>' in page
    assert '<textarea name="intimate">宝贝\n&lt;b&gt;&amp;心&lt;/b&gt;</textarea>' in page
    assert f'<textarea name="name">{SYNTHETIC_NAME}</textarea>' in page
    assert "<i>甲街" not in page and "<b>&心" not in page


def test_terms_are_trimmed_deduplicated_and_empty_groups_omitted(tmp_path):
    app, cookie = mapped(tmp_path)
    form = snippet_form(
        app, cookie, name=f" {SYNTHETIC_NAME} \r\n\r\n{SYNTHETIC_NAME}\n李四", intimate=""
    )
    status, _, _ = submit(app, cookie, form)
    assert status == 200
    terms = json.loads((root_of(app) / "terms.json").read_text(encoding="utf-8"))
    assert terms == {"redaction_terms": {"name": [SYNTHETIC_NAME, "李四"]}, "intimate_terms": []}


def _assert_nothing_written(app: AdminApp) -> None:
    root = root_of(app)
    assert not any((root / name).exists() for name in FILES)


@pytest.mark.parametrize(
    "change",
    [
        {"extra": "x"},
        {"confirm": None},
        {"intimate": None},
        {"confirm": "no"},
        {"name": "长" * 41},
        {"name": "\n".join(f"w{index}" for index in range(501))},
        {"name": "\n".join(f"w{index}" for index in range(300)),
         "intimate": "\n".join(f"v{index}" for index in range(201))},
    ],
)
def test_invalid_forms_are_rejected_without_writing(tmp_path, change):
    app, cookie = mapped(tmp_path)
    form = snippet_form(app, cookie)
    for key, value in change.items():
        if value is None:
            form.pop(key)
        else:
            form[key] = value
    status, _, page = submit(app, cookie, form)
    assert status == 400
    _assert_nothing_written(app)
    assert SYNTHETIC_NAME not in page


def test_invalid_terms_use_fixed_message(tmp_path):
    app, cookie = mapped(tmp_path)
    status, _, page = submit(app, cookie, snippet_form(app, cookie, name="长" * 41))
    assert status == 400
    assert "片段词无效" in page


def test_bad_csrf_is_forbidden_without_writing(tmp_path):
    app, cookie = mapped(tmp_path)
    form = snippet_form(app, cookie)
    form["csrf"] = "0" * 64
    status, _, _ = submit(app, cookie, form)
    assert status == 403
    _assert_nothing_written(app)


def test_token_is_single_use(tmp_path):
    app, cookie = mapped(tmp_path)
    form = snippet_form(app, cookie)
    status, _, _ = submit(app, cookie, {**form, "confirm": "no"})
    assert status == 400
    status, _, _ = submit(app, cookie, form)
    assert status == 403
    _assert_nothing_written(app)


def test_new_mapping_invalidates_snippet_token(tmp_path):
    app, cookie = mapped(tmp_path)
    form = snippet_form(app, cookie)
    call(app, "/admin/import/preview", cookie=cookie)
    status, _, _ = submit(app, cookie, form)
    assert status == 403
    _assert_nothing_written(app)


def test_symlinked_directory_is_refused(tmp_path):
    app, cookie = mapped(tmp_path)
    form = snippet_form(app, cookie)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    root_of(app).parent.mkdir(parents=True)
    root_of(app).symlink_to(elsewhere)
    status, _, _ = submit(app, cookie, form)
    assert status == 503
    assert list(elsewhere.iterdir()) == []


def test_generation_failure_leaves_no_snippet_files(tmp_path, monkeypatch):
    app, cookie = mapped(tmp_path)
    form = snippet_form(app, cookie)

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(snippets_module, "build_paired_snippets", broken)
    status, _, page = submit(app, cookie, form)
    assert status == 503
    assert "片段生成失败" in page
    root = root_of(app)
    assert not any((root / name).exists() for name in SNIPPET_FILES)
    assert [path.name for path in root.iterdir()] == ["terms.json"]


def test_audit_page_lists_sample_escaped(tmp_path):
    app, cookie, _ = generated(tmp_path)
    page, _ = audit_form(app, cookie)
    audit = lines(root_of(app) / "audit_sample.jsonl")
    for item in audit:
        assert f'name="delete_{item["snippet_id"]}" value="yes"' in page
    assert "<b>" not in page
    assert "SYNTHETIC_OWNER_REPLY_0" in page
    assert 'type="datetime-local" name="from"' in page
    assert 'type="datetime-local" name="to"' in page


def test_audit_page_escapes_partner_text(tmp_path):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    library = (root / "library.jsonl").read_text(encoding="utf-8")
    (root / "audit_sample.jsonl").write_text(
        library.replace("今天下雨了吗", "<script>x</script>"), encoding="utf-8"
    )
    page, _ = audit_form(app, cookie)
    assert "&lt;script&gt;x&lt;/script&gt;" in page
    assert "<script>" not in page


def test_delete_by_id_cascades(tmp_path):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    page, form = audit_form(app, cookie)
    targets = [item["snippet_id"] for item in lines(root / "audit_sample.jsonl")[:2]]
    for target in targets:
        form[f"delete_{target}"] = "yes"
    status, _, page = audit_submit(app, cookie, form)
    assert status == 200
    assert "已删除 2 段" in page
    assert '<a href="/admin/style/snippets/audit">' in page
    for name in SNIPPET_FILES:
        assert not any(item["snippet_id"] in targets for item in lines(root / name))
    assert len(lines(root / "library.jsonl")) == 16
    assert len(lines(root / "holdout.jsonl")) == 2
    assert len(lines(root / "audit_sample.jsonl")) == 16


def test_delete_by_shanghai_time_range_cascades(tmp_path):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    _, form = audit_form(app, cookie)
    form["from"] = "2026-09-02T18:00"
    form["to"] = "2026-09-02T22:00"
    status, _, page = audit_submit(app, cookie, form)
    assert status == 200
    assert "已删除 3 段" in page
    assert len(lines(root / "library.jsonl")) == 17
    assert len(lines(root / "holdout.jsonl")) == 0
    assert len(lines(root / "audit_sample.jsonl")) == 17


def test_delete_count_uses_unique_snippet_ids(tmp_path):
    app, _, _ = generated(tmp_path)
    store = snippets_module.SnippetStore(root_of(app))
    target = snippets_module.read_snippets(store.audit_path)[0]
    library_ids = {
        item.snippet_id for item in snippets_module.read_snippets(store.library_path)
    }
    assert target.snippet_id in library_ids

    deleted = store.delete(
        frozenset({target.snippet_id}), (target.started_at, target.started_at)
    )

    assert deleted == 1


def test_repeated_delete_count_is_zero(tmp_path):
    app, _, _ = generated(tmp_path)
    store = snippets_module.SnippetStore(root_of(app))
    target_id = snippets_module.read_snippets(store.audit_path)[0].snippet_id
    snippet_ids = frozenset({target_id})

    assert store.delete(snippet_ids, None) == 1
    excluded_before = store.exclusions()[0]
    assert target_id in excluded_before
    assert store.delete(snippet_ids, None) == 0

    assert store.exclusions()[0] == excluded_before
    assert all(
        item.snippet_id != target_id
        for path in store.snippet_paths
        for item in snippets_module.read_snippets(path)
    )


def test_audit_requires_ids_or_range(tmp_path):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    before = {name: (root / name).read_bytes() for name in SNIPPET_FILES}
    _, form = audit_form(app, cookie)
    status, _, _ = audit_submit(app, cookie, form)
    assert status == 400
    assert {name: (root / name).read_bytes() for name in SNIPPET_FILES} == before


@pytest.mark.parametrize(
    "change",
    [
        {"delete_snip-0000000000000000": "yes"},
        {"from": "2026-09-02T18:00"},
        {"from": "2026-09-02T22:00", "to": "2026-09-02T18:00"},
        {"from": "2026-09-02T18:00+08:00", "to": "2026-09-02T22:00+08:00"},
        {"extra": "x"},
    ],
)
def test_invalid_audit_forms_are_rejected(tmp_path, change):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    before = {name: (root / name).read_bytes() for name in SNIPPET_FILES}
    _, form = audit_form(app, cookie)
    form.update(change)
    status, _, _ = audit_submit(app, cookie, form)
    assert status == 400
    assert {name: (root / name).read_bytes() for name in SNIPPET_FILES} == before


def test_audit_csrf_and_token_replay_are_forbidden(tmp_path):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    before = {name: (root / name).read_bytes() for name in SNIPPET_FILES}
    _, form = audit_form(app, cookie)
    status, _, _ = audit_submit(app, cookie, {**form, "csrf": "0" * 64})
    assert status == 403
    status, _, _ = audit_submit(app, cookie, form)
    assert status == 400
    form["from"], form["to"] = "2026-09-02T18:00", "2026-09-02T22:00"
    status, _, _ = audit_submit(app, cookie, form)
    assert status == 403
    assert {name: (root / name).read_bytes() for name in SNIPPET_FILES} == before


def test_audit_without_library_is_unavailable(tmp_path):
    app, cookie = mapped(tmp_path)
    _, form = audit_form(app, cookie)
    form["from"], form["to"] = "2026-09-02T18:00", "2026-09-02T22:00"
    status, _, _ = audit_submit(app, cookie, form)
    assert status == 503
    assert not root_of(app).exists()


def test_home_links_snippet_page(tmp_path):
    app = make_app(tmp_path)
    cookie = unlock(app)
    status, _, page = call(app, "/admin/home", cookie=cookie)
    assert status == 200
    assert '<li><a href="/admin/style/snippets">成对片段（新风格链路）</a></li>' in page


def regenerate(app: AdminApp, cookie: str):
    status, _, page = submit(app, cookie, snippet_form(app, cookie))
    assert status == 200
    return page


def all_ids(root) -> set[str]:
    return {item["snippet_id"] for name in SNIPPET_FILES for item in lines(root / name)}


def test_deleted_ids_stay_excluded_after_regeneration(tmp_path):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    _, form = audit_form(app, cookie)
    targets = {item["snippet_id"] for item in lines(root / "audit_sample.jsonl")[:2]}
    for target in targets:
        form[f"delete_{target}"] = "yes"
    status, _, _ = audit_submit(app, cookie, form)
    assert status == 200
    assert (root / "deleted_ids.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((root / "deleted_ids.json").read_text(encoding="utf-8")) == {
        "snippet_ids": sorted(targets),
        "ranges": [],
    }
    page = regenerate(app, cookie)
    assert "片段：18" in page
    assert not targets & all_ids(root)
    page, _ = audit_form(app, cookie)
    assert "已永久排除：2 段 + 0 个时间段" in page


def test_deleted_range_stays_excluded_after_regeneration(tmp_path):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    _, form = audit_form(app, cookie)
    form["from"], form["to"] = "2026-09-02T18:00", "2026-09-02T22:00"
    status, _, _ = audit_submit(app, cookie, form)
    assert status == 200
    saved = json.loads((root / "deleted_ids.json").read_text(encoding="utf-8"))
    assert len(saved["snippet_ids"]) == 3
    assert saved["ranges"] == [["2026-09-02T18:00:00+08:00", "2026-09-02T22:00:00+08:00"]]
    page = regenerate(app, cookie)
    assert "片段：17" in page
    window = (BASE + timedelta(hours=34), BASE + timedelta(hours=38))
    for name in SNIPPET_FILES:
        for item in lines(root / name):
            assert not window[0] <= datetime.fromisoformat(item["started_at"]) <= window[1]
    page, _ = audit_form(app, cookie)
    assert "已永久排除：3 段 + 1 个时间段" in page


def test_exclusions_accumulate_across_deletions(tmp_path):
    app, cookie, _ = generated(tmp_path)
    root = root_of(app)
    _, form = audit_form(app, cookie)
    first = lines(root / "audit_sample.jsonl")[0]["snippet_id"]
    status, _, _ = audit_submit(app, cookie, {**form, f"delete_{first}": "yes"})
    assert status == 200
    _, form = audit_form(app, cookie)
    form["from"], form["to"] = "2026-09-02T18:00", "2026-09-02T22:00"
    status, _, _ = audit_submit(app, cookie, form)
    assert status == 200
    page, _ = audit_form(app, cookie)
    assert "已永久排除：4 段 + 1 个时间段" in page
    assert "片段：16" in regenerate(app, cookie)


def test_generation_enables_context_pseudonyms_and_media_exclusion(tmp_path):
    records = [
        ChatRecord(
            conversation_id="synthetic-context",
            message_id=f"m{index}",
            timestamp=BASE + timedelta(minutes=index),
            speaker_role=SpeakerRole.OTHER if index % 2 == 0 else SpeakerRole.OWNER,
            content_type="text",
            text=f"王小明在合成街 <b>{index}</b>",
        )
        for index in range(14)
    ]
    records.append(ChatRecord(
        conversation_id="synthetic-context", message_id="image",
        timestamp=BASE + timedelta(minutes=12, seconds=30),
        speaker_role=SpeakerRole.OTHER, content_type="image", text=None,
    ))
    records.sort(key=lambda record: record.timestamp)
    store = snippets_module.SnippetStore(tmp_path / "snippets")
    groups = {field: [] for field in snippets_module.TERM_FIELDS}
    groups.update(name=[SYNTHETIC_NAME], address=["合成街"])
    counts = store.generate(records, groups=groups, created_at=NOW, seed=0)
    sample = snippets_module.read_snippets(store.library_path)
    sample += snippets_module.read_snippets(store.holdout_path)
    assert counts[0] == 6
    last = max(sample, key=lambda item: item.started_at)
    assert len(last.context_turns) == 10
    assert last.context_turns[0] == "她：小甲在甲地 <b>0</b>"
    assert last.partner_turns == ["小甲在甲地 <b>10</b>"]
    assert all(item.started_at != BASE + timedelta(minutes=12) for item in sample)
    raw = "".join(path.read_text() for path in store.snippet_paths)
    assert SYNTHETIC_NAME not in raw and "合成街" not in raw
    page = snippets_module.render_audit(csrf="c", token="t", sample=[last], excluded=(0, 0))
    assert "<p>前文：她：小甲在甲地 &lt;b&gt;0&lt;/b&gt;<br>" in page
    assert "他：小甲在甲地 &lt;b&gt;9&lt;/b&gt;</p>" in page
    assert page.index("前文：") < page.index("对方：") < page.index("主人：")
    assert "<b>" not in page
    empty_page = snippets_module.render_audit(
        csrf="c", token="t", sample=[min(sample, key=lambda item: item.started_at)],
        excluded=(0, 0),
    )
    assert "前文：" not in empty_page


def test_form_explains_pseudonyms_and_media_exclusion(tmp_path):
    app, cookie = mapped(tmp_path)
    status, _, page = call(app, "/admin/style/snippets", cookie=cookie)
    assert status == 200
    assert (
        "名字与地点会替换成固定化名（小甲、甲地…），按每行顺序编号；"
        "含图片等非文字消息附近的对话不会进入片段库。"
    ) in page
