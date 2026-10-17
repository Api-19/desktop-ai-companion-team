import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from test_local_entry import call, config, settings
from test_mapping_confirmation import unlock

from companion.admin.app import AdminApp, build_admin_app, load_admin_config
from companion.dialogue import UsageSnapshot
from companion.dialogue.models import BudgetConfig
from companion.dialogue.usage import MonthlyUsageLedger

NOW = datetime(2026, 9, 30, 16, tzinfo=UTC)


def snapshot(status="allowed", settled="2", reserved="3"):
    return UsageSnapshot(
        month="2026-10",
        settled_cny=Decimal(settled),
        reserved_cny=Decimal(reserved),
        accounted_cny=Decimal(settled) + Decimal(reserved),
        monthly_cap_cny=Decimal("100"),
        status=status,
    )


def app_with(reader, clock=lambda: 0):
    return AdminApp(
        "synthetic-owner", host="127.0.0.1", port=12345, clock=clock, usage_reader=reader
    )


@pytest.mark.parametrize("path", ["/admin/home", "/admin/usage"])
@pytest.mark.parametrize(
    "fault", ["locked", "expired", "host", "origin", "owner", "closed", "post"]
)
def test_usage_gates_before_reader(path, fault):
    reads = []
    now = [0]
    app = app_with(lambda: reads.append(True) or snapshot(), lambda: now[0])
    cookie = "" if fault == "locked" else unlock(app)
    kwargs = {}
    if fault == "expired":
        now[0] = 900
    if fault == "host":
        kwargs["host"] = "invalid:12345"
    if fault == "origin":
        kwargs["origin"] = "http://invalid"
    if fault == "owner":
        app._sessions[cookie.split("=", 1)[1]].owner_id = "other"
    if fault == "closed":
        app.close()
    if fault == "post":
        kwargs.update(method="POST", origin="http://invalid")
    assert call(app, path, cookie=cookie, **kwargs)[0] == 403
    assert reads == []


@pytest.mark.parametrize(
    ("status", "settled", "reserved", "notice"),
    [
        ("allowed", "2", "3", "下一单请求可预留"),
        ("blocked", "98", "2", "100 元已占用"),
        ("blocked", "98", "1", "剩余不足下一单请求"),
        ("review_required", "2", "3", "需核对并暂停付费"),
    ],
)
def test_usage_and_home_show_aggregates_and_notice(status, settled, reserved, notice):
    reads = []
    app = app_with(lambda: reads.append(True) or snapshot(status, settled, reserved))
    cookie = unlock(app)
    for path in ["/admin/home", "/admin/usage"]:
        code, headers, page = call(app, path, cookie=cookie)
        assert code == 200 and notice in page
        assert headers["Cache-Control"] == "no-store"
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert "本地已记账" in page and "预留" in page and "供应商账单" in page
        assert "2026-10" in page and "100" in page and "占用" in page
        assert "synthetic-owner" not in page
        if path == "/admin/home":
            assert "/admin/usage" in page
    assert len(reads) == 2


@pytest.mark.parametrize(
    "fault",
    ["missing", "exception", "dict", "forged_month", "forged_money", "forged_status", "wrong_cap"],
)
def test_usage_unavailable_is_fixed_and_no_fake_zero(fault):
    value = snapshot()
    if fault == "dict":
        value = value.model_dump()
    elif fault == "forged_month":
        value = value.model_copy(update={"month": "<script>private</script>"})
    elif fault == "forged_money":
        value = value.model_copy(update={"settled_cny": Decimal("NaN")})
    elif fault == "forged_status":
        value = value.model_copy(update={"status": "private-error"})
    elif fault == "wrong_cap":
        value = value.model_copy(update={"monthly_cap_cny": Decimal("200")})

    def reader():
        if fault == "exception":
            raise OSError("SYNTHETIC_PRIVATE_DETAIL /private/value")
        return value

    app = (
        AdminApp("synthetic-owner", host="127.0.0.1", port=12345)
        if fault == "missing"
        else app_with(reader)
    )
    cookie = unlock(app)
    code, _, page = call(app, "/admin/usage", cookie=cookie)
    assert code == 503 and "费用状态不可用" in page
    assert "SYNTHETIC_PRIVATE_DETAIL" not in page and "<script>" not in page
    assert "本地已记账" not in page and "下一单请求可预留" not in page
    code, _, page = call(app, "/admin/home", cookie=cookie)
    assert code == 200 and "费用状态不可用" in page and "/admin/usage" in page


def product(tmp_path, monkeypatch):
    root = tmp_path / "private"
    config(root)
    trusted = settings(root)
    budget = BudgetConfig(
        monthly_cap_cny="100",
        worst_case_request_cny="1",
        input_cny_per_million="1",
        output_cny_per_million="1",
        max_output_tokens=100,
    )
    folder = root / "dialogue"
    folder.mkdir()
    path = folder / "runtime-config.json"
    path.write_text(
        json.dumps(
            {
                "budget": budget.model_dump(mode="json"),
                "provider": "broken",
                "style": "SYNTHETIC_PRIVATE_DETAIL",
            }
        )
    )
    import companion.dialogue as dialogue

    original = dialogue.load_budget_config
    reads = []

    def budget_reader(value):
        reads.append(True)
        return original(value)

    monkeypatch.setattr(dialogue, "load_budget_config", budget_reader)

    def forbidden(*a, **k):
        raise AssertionError("forbidden secret/provider/mutation access")

    monkeypatch.setattr(dialogue, "load_dialogue_config", forbidden)
    for name in ["reserve", "settle", "cancel_unused"]:
        monkeypatch.setattr(MonthlyUsageLedger, name, forbidden)
    app = build_admin_app(trusted, load_admin_config(trusted), port=12345, wall_clock=lambda: NOW)
    assert reads == []
    return app, trusted, path, budget


def test_product_delays_budget_read_and_missing_month_has_no_writes(tmp_path, monkeypatch):
    app, trusted, path, _ = product(tmp_path, monkeypatch)
    saved = path.read_bytes()
    path.unlink()
    assert call(app, "/admin/usage")[0] == 403
    path.write_bytes(saved)
    before = {p: p.read_bytes() for p in trusted.private_data_dir.rglob("*") if p.is_file()}
    cookie = unlock(app)
    for _ in range(2):
        code, _, page = call(app, "/admin/usage", cookie=cookie)
        assert code == 200 and "2026-10" in page and "下一单请求可预留" in page
    assert not trusted.runtime_data_dir.exists()
    assert before == {p: p.read_bytes() for p in trusted.private_data_dir.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "fault",
    ["config_missing", "config_json", "config_budget", "ledger_json", "ledger_link", "naive_clock"],
)
def test_product_reader_failures_are_unavailable(tmp_path, monkeypatch, fault):
    app, trusted, path, _ = product(tmp_path, monkeypatch)
    if fault == "config_missing":
        path.unlink()
    elif fault == "config_json":
        path.write_text("SYNTHETIC_PRIVATE_DETAIL")
    elif fault == "config_budget":
        path.write_text("{}")
    elif fault == "naive_clock":
        app = build_admin_app(
            trusted,
            load_admin_config(trusted),
            port=12345,
            wall_clock=lambda: datetime(2026, 10, 1),
        )
    else:
        folder = trusted.runtime_data_dir / "dialogue"
        folder.mkdir(parents=True)
        ledger = folder / "dialogue-usage-2026-10.json"
        if fault == "ledger_link":
            ledger.symlink_to(tmp_path / "absent")
        else:
            ledger.write_text("SYNTHETIC_PRIVATE_DETAIL")
    cookie = unlock(app)
    code, _, page = call(app, "/admin/usage", cookie=cookie)
    assert code == 503 and "费用状态不可用" in page
    assert str(path) not in page and "SYNTHETIC_PRIVATE_DETAIL" not in page


def test_main_uses_product_reader_without_real_listener(tmp_path, monkeypatch, capsys):
    app, trusted, _, _ = product(tmp_path, monkeypatch)
    import companion.admin.__main__ as main

    monkeypatch.setattr(main, "load_settings", lambda *_: trusted)
    # Keep the real builder, injecting only deterministic wall time.
    builder = main.build_admin_app
    monkeypatch.setattr(
        main,
        "build_admin_app",
        lambda s, c, *, port: builder(s, c, port=port, wall_clock=lambda: NOW),
    )

    class Server:
        server_port = 12345

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_app(self, value):
            self.app = value

        def serve_forever(self):
            cookie = unlock(self.app)
            assert call(self.app, "/admin/usage", cookie=cookie)[0] == 200
            raise KeyboardInterrupt

    monkeypatch.setattr(main, "make_server", lambda *a, **k: Server())
    assert main.main() == 0
    capsys.readouterr()
    assert not trusted.runtime_data_dir.exists()
    app.close()


@pytest.mark.parametrize(
    ("settled", "reserved", "review", "notice"),
    [
        ("2", "3", False, "下一单请求可预留"),
        ("99", "1", False, "100 元已占用"),
        ("99", "0.5", False, "剩余不足下一单请求"),
        ("2", "3", True, "需核对并暂停付费"),
    ],
)
def test_product_existing_ledger_is_read_only(
    tmp_path, monkeypatch, settled, reserved, review, notice
):
    app, trusted, _, _ = product(tmp_path, monkeypatch)
    folder = trusted.runtime_data_dir / "dialogue"
    folder.mkdir(parents=True)
    path = folder / "dialogue-usage-2026-10.json"
    reservation_id = "a" * 32
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "month": "2026-10",
                "settled_cny": settled,
                "reservations": {reservation_id: reserved},
                "requires_review": review,
            }
        )
    )
    before = path.read_bytes()
    cookie = unlock(app)
    for route in ["/admin/home", "/admin/usage"]:
        code, _, page = call(app, route, cookie=cookie)
        assert code == 200 and notice in page
        assert reservation_id not in page and str(path) not in page
    assert path.read_bytes() == before and list(folder.iterdir()) == [path]
