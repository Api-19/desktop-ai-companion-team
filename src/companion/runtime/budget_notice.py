# ruff: noqa: RUF001

import subprocess
from collections.abc import Callable
from datetime import datetime

from companion.dialogue.usage import MonthlyUsageLedger, UsageLedgerError, UsageSnapshot

BUDGET_NOTICE_CODE = "budget_blocked"
MAC_BUDGET_NOTICE_SCRIPT = (
    'display notification "本月云端聊天额度不足，已暂停新的付费请求，请主人查看管理页。" '
    'with title "kiki 费用提醒"'
)


class BudgetNoticeGate:
    """Track budget-blocked episodes in memory for one host instance."""

    def __init__(self) -> None:
        self._month: str | None = None
        self._notified: bool = False

    def observe(self, snapshot: UsageSnapshot) -> str | None:
        if snapshot.month != self._month:
            self._month = snapshot.month
            self._notified = False
        if snapshot.status == "allowed":
            self._notified = False
            return None
        if snapshot.status == "blocked" and not self._notified:
            self._notified = True
            return BUDGET_NOTICE_CODE
        return None


class MacBudgetNotifier:
    def __init__(self, *, ledger: MonthlyUsageLedger, clock: Callable[[], datetime]) -> None:
        self._ledger = ledger
        self._clock = clock
        self._gate = BudgetNoticeGate()

    def check(self) -> None:
        try:
            snapshot = self._ledger.snapshot(self._clock())
        except UsageLedgerError:
            return
        if self._gate.observe(snapshot) is not None:
            send_mac_budget_notice()


def send_mac_budget_notice() -> None:
    try:
        subprocess.run(
            ["/usr/bin/osascript", "-e", MAC_BUDGET_NOTICE_SCRIPT],
            check=True,
            timeout=2.0,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            env={},
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return
