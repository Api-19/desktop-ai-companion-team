# ruff: noqa: RUF001

import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal
from unittest.mock import Mock, call

import pytest

from companion.dialogue.usage import UsageLedgerError, UsageSnapshot
from companion.runtime import budget_notice
from companion.runtime.budget_notice import BUDGET_NOTICE_CODE, BudgetNoticeGate


def make_snapshot(
    status: Literal["allowed", "blocked", "review_required"] = "blocked",
    month: str = "2026-10",
    accounted: str = "100",
) -> UsageSnapshot:
    return UsageSnapshot(
        month=month,
        settled_cny=Decimal(accounted),
        reserved_cny=Decimal("0"),
        accounted_cny=Decimal(accounted),
        monthly_cap_cny=Decimal("100"),
        status=status,
    )


def test_blocked_episode_notifies_once() -> None:
    gate = BudgetNoticeGate()
    snapshot = make_snapshot()

    assert BUDGET_NOTICE_CODE == "budget_blocked"
    assert gate.observe(snapshot) == BUDGET_NOTICE_CODE
    assert gate.observe(snapshot) is None
    assert gate.observe(make_snapshot(accounted="101")) is None


def test_allowed_then_blocked_notifies_again() -> None:
    gate = BudgetNoticeGate()

    assert gate.observe(make_snapshot()) == BUDGET_NOTICE_CODE
    assert gate.observe(make_snapshot("allowed", accounted="0")) is None
    assert gate.observe(make_snapshot()) == BUDGET_NOTICE_CODE
    assert gate.observe(make_snapshot()) is None


def test_month_change_resets_gate() -> None:
    gate = BudgetNoticeGate()

    assert gate.observe(make_snapshot()) == BUDGET_NOTICE_CODE
    assert gate.observe(make_snapshot()) is None
    assert gate.observe(make_snapshot(month="2026-11")) == BUDGET_NOTICE_CODE
    assert gate.observe(make_snapshot(month="2026-11")) is None


def test_review_required_does_not_notify_or_rearm() -> None:
    gate = BudgetNoticeGate()

    assert gate.observe(make_snapshot("review_required")) is None
    assert gate.observe(make_snapshot()) == BUDGET_NOTICE_CODE
    assert gate.observe(make_snapshot("review_required")) is None
    assert gate.observe(make_snapshot()) is None
    assert gate.observe(make_snapshot("review_required", month="2026-11")) is None
    assert gate.observe(make_snapshot(month="2026-11")) == BUDGET_NOTICE_CODE


def test_new_gate_can_notify_existing_block() -> None:
    snapshot = make_snapshot()
    first = BudgetNoticeGate()
    second = BudgetNoticeGate()

    assert first.observe(snapshot) == BUDGET_NOTICE_CODE
    assert second.observe(snapshot) == BUDGET_NOTICE_CODE
    assert first.observe(snapshot) is None
    assert second.observe(snapshot) is None


def test_gate_uses_status_instead_of_amount() -> None:
    gate = BudgetNoticeGate()
    snapshot = make_snapshot(accounted="99.000001")

    assert snapshot.accounted_cny < snapshot.monthly_cap_cny
    assert gate.observe(snapshot) == BUDGET_NOTICE_CODE
    assert gate.observe(make_snapshot("allowed")) is None
    assert gate.observe(snapshot) == BUDGET_NOTICE_CODE


def test_snapshot_failure_does_not_notify_or_rearm(monkeypatch) -> None:
    now = datetime(2026, 10, 17, tzinfo=UTC)
    clock = Mock(return_value=now)
    ledger = Mock()
    ledger.snapshot.side_effect = [
        UsageLedgerError("synthetic failure"), make_snapshot(),
        UsageLedgerError("synthetic failure"), make_snapshot(),
        make_snapshot("allowed", accounted="0"), make_snapshot(),
    ]
    sender = Mock()
    monkeypatch.setattr(budget_notice, "send_mac_budget_notice", sender, raising=False)
    notifier = budget_notice.MacBudgetNotifier(ledger=ledger, clock=clock)

    for index, count in enumerate((0, 1, 1, 1, 1, 2), start=1):
        notifier.check()
        assert sender.call_count == count
        assert ledger.snapshot.call_args_list == [call(now)] * index
        assert clock.call_count == index


def test_notification_process_receives_only_fixed_text(monkeypatch, capsys, caplog) -> None:
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("SYNTHETIC_PROVIDER_API_KEY", "synthetic-secret")

    budget_notice.send_mac_budget_notice()

    expected = (
        'display notification "本月云端聊天额度不足，已暂停新的付费请求，请主人查看管理页。" '
        'with title "kiki 费用提醒"'
    )
    assert expected == budget_notice.MAC_BUDGET_NOTICE_SCRIPT
    run.assert_called_once_with(
        ["/usr/bin/osascript", "-e", expected], check=True, timeout=2.0,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        shell=False, env={},
    )
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


@pytest.mark.parametrize("error", [
    OSError("synthetic-sensitive-error"),
    subprocess.CalledProcessError(1, "synthetic-sensitive-command", output="sensitive-output"),
    subprocess.TimeoutExpired("synthetic-sensitive-command", 2.0, output="sensitive-output"),
])
def test_notification_process_failures_are_silent(error, monkeypatch, capsys, caplog) -> None:
    run = Mock(side_effect=error)
    monkeypatch.setattr(subprocess, "run", run)

    assert budget_notice.send_mac_budget_notice() is None

    run.assert_called_once()
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_notification_process_unexpected_error_propagates(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=RuntimeError("unexpected")))
    with pytest.raises(RuntimeError, match="unexpected"):
        budget_notice.send_mac_budget_notice()


@pytest.mark.parametrize("error", [
    OSError("synthetic failure"),
    subprocess.CalledProcessError(1, "synthetic-command"),
    subprocess.TimeoutExpired("synthetic-command", 2.0),
])
def test_failed_delivery_is_not_retried_during_blocked_episode(error, monkeypatch) -> None:
    run = Mock(side_effect=error)
    monkeypatch.setattr(subprocess, "run", run)
    ledger = Mock()
    ledger.snapshot.return_value = make_snapshot()
    notifier = budget_notice.MacBudgetNotifier(
        ledger=ledger, clock=lambda: datetime(2026, 10, 17, tzinfo=UTC),
    )

    notifier.check()
    notifier.check()
    notifier.check()

    run.assert_called_once()
    assert ledger.snapshot.call_count == 3
