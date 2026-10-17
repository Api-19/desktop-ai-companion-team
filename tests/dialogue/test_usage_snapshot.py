from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_usage import SHANGHAI_NOW, make_ledger

from companion.dialogue import UsageSnapshot
from companion.dialogue.usage import UsageLedgerError


def test_missing_snapshot_has_no_directory_or_mutating_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "absent"
    ledger = make_ledger(root)
    for name in ("reserve", "settle", "cancel_unused", "_atomic_write"):
        monkeypatch.setattr(ledger, name, lambda *a, **k: pytest.fail("mutation"))
    snapshot = ledger.snapshot(SHANGHAI_NOW)
    assert isinstance(snapshot, UsageSnapshot)
    assert snapshot.month == "2026-07"
    assert snapshot.accounted_cny == snapshot.settled_cny == snapshot.reserved_cny == 0
    assert snapshot.monthly_cap_cny == Decimal("100.00")
    assert snapshot.status == "allowed"
    assert not root.exists()
    assert set(snapshot.model_dump()) == {
        "month",
        "settled_cny",
        "reserved_cny",
        "accounted_cny",
        "monthly_cap_cny",
        "status",
    }
    with pytest.raises(ValidationError):
        snapshot.status = "blocked"


def test_snapshot_reads_shanghai_month_without_writing(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    first, second = ledger.reserve(2, SHANGHAI_NOW)
    ledger.settle(first, None, SHANGHAI_NOW)
    path = tmp_path / "dialogue-usage-2026-07.json"
    before = path.read_bytes()
    a = ledger.snapshot(SHANGHAI_NOW)
    b = ledger.snapshot(SHANGHAI_NOW)
    assert a == b
    assert a.settled_cny == a.reserved_cny == Decimal("1.00")
    assert a.accounted_cny == Decimal("2.00")
    assert second.reservation_id not in a.model_dump_json()
    assert path.read_bytes() == before
    assert len(list(tmp_path.iterdir())) == 1
    next_month = ledger.snapshot(datetime(2026, 7, 31, 16, tzinfo=UTC))
    assert next_month.month == "2026-08"
    assert next_month.accounted_cny == 0
    assert not (tmp_path / "dialogue-usage-2026-08.json").exists()


@pytest.mark.parametrize(
    ("settled", "review", "status"),
    [
        ("99", False, "allowed"),
        ("99.000001", False, "blocked"),
        ("100", False, "blocked"),
        ("0", True, "review_required"),
    ],
)
def test_snapshot_status_matches_next_single_reservation(
    tmp_path: Path, settled: str, review: bool, status: str
) -> None:
    ledger = make_ledger(tmp_path)
    (tmp_path / "dialogue-usage-2026-07.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "month": "2026-07",
                "settled_cny": settled,
                "reservations": {},
                "requires_review": review,
            }
        )
    )
    assert ledger.snapshot(SHANGHAI_NOW).status == status


@pytest.mark.parametrize(
    "fault",
    [
        "json",
        "negative",
        "nan",
        "bad_reservation",
        "month",
        "naive",
        "read",
        "dangling",
        "root_link",
        "ancestor_link",
    ],
)
def test_snapshot_failures_never_become_zero_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    ledger = make_ledger(tmp_path)
    path = tmp_path / "dialogue-usage-2026-07.json"
    payload = {
        "schema_version": 1,
        "month": "2026-07",
        "settled_cny": "0",
        "reservations": {},
        "requires_review": False,
    }
    now = SHANGHAI_NOW
    if fault == "negative":
        payload["settled_cny"] = "-1"
    elif fault == "nan":
        payload["settled_cny"] = "NaN"
    elif fault == "bad_reservation":
        payload["reservations"] = {"f" * 32: "-1"}
    elif fault == "month":
        payload["month"] = "2026-06"
    elif fault == "naive":
        now = now.replace(tzinfo=None)
    if fault == "dangling":
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_text("synthetic-json-sentinel" if fault == "json" else json.dumps(payload))
    if fault in {"root_link", "ancestor_link"}:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        ledger = make_ledger(alias if fault == "root_link" else alias / "nested")
    if fault == "read":
        monkeypatch.setattr(
            ledger,
            "_load_month",
            lambda _: (_ for _ in ()).throw(OSError("synthetic-path-sentinel")),
        )
    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$") as caught:
        ledger.snapshot(now)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    tb = caught.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_filename.endswith("dialogue/usage.py"):
            assert not any(
                str(tmp_path) in repr(v) or "synthetic-path-sentinel" in repr(v)
                for v in tb.tb_frame.f_locals.values()
            )
        tb = tb.tb_next
