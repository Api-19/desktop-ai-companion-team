import json
import os
import stat
import traceback
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from companion.dialogue.models import BudgetConfig, ProviderUsage
from companion.dialogue.usage import (
    BudgetExceeded,
    MonthlyUsageLedger,
    UsageLedgerError,
    UsageReviewRequired,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
SHANGHAI_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=SHANGHAI)


def make_ledger(
    root: Path,
    *,
    monthly_cap: str = "100.00",
    worst_case: str = "1.00",
    input_rate: str = "1.00",
    output_rate: str = "2.00",
) -> MonthlyUsageLedger:
    return MonthlyUsageLedger(
        root,
        BudgetConfig(
            monthly_cap_cny=Decimal(monthly_cap),
            worst_case_request_cny=Decimal(worst_case),
            input_cny_per_million=Decimal(input_rate),
            output_cny_per_million=Decimal(output_rate),
            max_output_tokens=256,
        ),
    )


def test_reservation_stops_before_monthly_cap(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, monthly_cap="100.00", worst_case="60.00")
    first = ledger.reserve(count=1, now=SHANGHAI_NOW)
    ledger.settle(first[0], usage=None, now=SHANGHAI_NOW)

    with pytest.raises(BudgetExceeded, match=r"^monthly budget exceeded$"):
        ledger.reserve(count=1, now=SHANGHAI_NOW)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode contract")
def test_ledger_writes_owner_only_even_with_permissive_umask(tmp_path: Path) -> None:
    previous = os.umask(0)
    try:
        ledger = make_ledger(tmp_path)
        reservation = ledger.reserve(count=1, now=SHANGHAI_NOW)[0]
        ledger.settle(reservation, usage=None, now=SHANGHAI_NOW)
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "dialogue-usage-2026-07.json").stat().st_mode) == 0o600


def test_ledger_syncs_file_before_replace_and_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_replace = Path.replace

    def sync(fd: int) -> None:
        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")

    def replace(path: Path, destination: Path) -> Path:
        events.append("replace")
        return real_replace(path, destination)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(Path, "replace", replace)
    make_ledger(tmp_path).reserve(count=1, now=SHANGHAI_NOW)
    assert events == ["file", "replace"] + (["directory"] if os.name == "posix" else [])


def test_ledger_file_sync_failure_preserves_prior_state_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_ledger(tmp_path)
    ledger.reserve(count=1, now=SHANGHAI_NOW)
    destination = tmp_path / "dialogue-usage-2026-07.json"
    before = destination.read_bytes()

    def fail_sync(fd: int) -> None:
        raise OSError("synthetic sync failure")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$"):
        ledger.reserve(count=1, now=SHANGHAI_NOW)
    assert destination.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory sync contract")
def test_directory_sync_failure_keeps_visible_reservation_conservative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_sync = os.fsync

    def fail_directory_sync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("synthetic directory sync failure")
        real_sync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_sync)
    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$"):
        make_ledger(tmp_path, worst_case="100.00").reserve(count=1, now=SHANGHAI_NOW)
    assert list(tmp_path.glob("*.tmp")) == []
    with pytest.raises(BudgetExceeded):
        make_ledger(tmp_path, worst_case="100.00").ensure_available(count=1, now=SHANGHAI_NOW)


def test_ab_reserves_both_requests_atomically(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, monthly_cap="100.00", worst_case="50.01")

    with pytest.raises(BudgetExceeded, match=r"^monthly budget exceeded$"):
        ledger.reserve(count=2, now=SHANGHAI_NOW)

    assert not (tmp_path / "dialogue-usage-2026-07.json").exists()


def test_reservation_quantization_never_rounds_down_capacity(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, worst_case="1.0000001")
    for _ in range(99):
        ledger.reserve(count=1, now=SHANGHAI_NOW)

    with pytest.raises(BudgetExceeded, match=r"^monthly budget exceeded$"):
        ledger.reserve(count=1, now=SHANGHAI_NOW)


def test_batch_preflight_matches_individual_conservative_reservations(tmp_path: Path) -> None:
    ledger = make_ledger(
        tmp_path,
        worst_case="0.0000001",
        input_rate="0.0000001",
        output_rate="0.0000001",
    )
    destination = tmp_path / "dialogue-usage-2026-07.json"
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "month": "2026-07",
                "settled_cny": "99.999999",
                "reservations": {},
                "requires_review": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BudgetExceeded, match=r"^monthly budget exceeded$"):
        ledger.ensure_available(count=2, now=SHANGHAI_NOW)
    with pytest.raises(BudgetExceeded, match=r"^monthly budget exceeded$"):
        ledger.reserve(count=2, now=SHANGHAI_NOW)

    assert json.loads(destination.read_text(encoding="utf-8"))["reservations"] == {}


def test_capacity_preflight_does_not_mutate_ledger(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, monthly_cap="100.00", worst_case="1.00")

    ledger.ensure_available(count=13, now=SHANGHAI_NOW)

    assert not (tmp_path / "dialogue-usage-2026-07.json").exists()


def test_usage_above_reservation_halts_future_paid_requests(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, worst_case="1.00")
    reservation = ledger.reserve(count=1, now=SHANGHAI_NOW)[0]
    ledger.settle(
        reservation,
        ProviderUsage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
        ),
        SHANGHAI_NOW,
    )

    with pytest.raises(UsageReviewRequired, match=r"^usage ledger requires review$"):
        ledger.reserve(count=1, now=SHANGHAI_NOW)


def test_sub_micro_actual_cost_above_reservation_still_requires_review(tmp_path: Path) -> None:
    ledger = make_ledger(
        tmp_path,
        worst_case="50.00",
        input_rate="50.0000004",
        output_rate="1.00",
    )
    reservation = ledger.reserve(count=1, now=SHANGHAI_NOW)[0]

    ledger.settle(
        reservation,
        ProviderUsage(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            total_tokens=1_000_000,
        ),
        SHANGHAI_NOW,
    )

    state = json.loads((tmp_path / "dialogue-usage-2026-07.json").read_text(encoding="utf-8"))
    assert state["settled_cny"] == "50.000000"
    assert state["requires_review"] is True
    with pytest.raises(UsageReviewRequired, match=r"^usage ledger requires review$"):
        ledger.reserve(count=1, now=SHANGHAI_NOW)


def test_settlement_uses_exact_decimal_actual_cost(tmp_path: Path) -> None:
    ledger = make_ledger(
        tmp_path,
        worst_case="1.00",
        input_rate="1.234567",
        output_rate="2.345678",
    )
    reservation = ledger.reserve(count=1, now=SHANGHAI_NOW)[0]

    ledger.settle(
        reservation,
        ProviderUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        SHANGHAI_NOW,
    )

    state = json.loads((tmp_path / "dialogue-usage-2026-07.json").read_text(encoding="utf-8"))
    assert state["settled_cny"] == "0.000006"
    assert state["reservations"] == {}
    assert state["requires_review"] is False


def test_settlement_quantization_never_rounds_down_actual_cost(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, input_rate="0.1000000", output_rate="1.00")
    reservation = ledger.reserve(count=1, now=SHANGHAI_NOW)[0]

    ledger.settle(
        reservation,
        ProviderUsage(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        SHANGHAI_NOW,
    )

    state = json.loads((tmp_path / "dialogue-usage-2026-07.json").read_text(encoding="utf-8"))
    assert state["settled_cny"] == "0.000001"


@pytest.mark.parametrize("usage", [None, "invalid usage sentinel"])
def test_missing_or_invalid_usage_retains_full_reservation(
    tmp_path: Path, usage: object | None
) -> None:
    ledger = make_ledger(tmp_path, worst_case="1.25")
    reservation = ledger.reserve(count=1, now=SHANGHAI_NOW)[0]

    ledger.settle(reservation, usage=usage, now=SHANGHAI_NOW)  # type: ignore[arg-type]

    state = json.loads((tmp_path / "dialogue-usage-2026-07.json").read_text(encoding="utf-8"))
    assert state["settled_cny"] == "1.250000"
    assert state["reservations"] == {}
    assert state["requires_review"] is False


def test_timeout_settlement_retains_full_reservation(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, worst_case="1.25")
    reservation = ledger.reserve(count=1, now=SHANGHAI_NOW)[0]

    ledger.settle(reservation, usage=None, now=SHANGHAI_NOW)

    state = json.loads((tmp_path / "dialogue-usage-2026-07.json").read_text(encoding="utf-8"))
    assert state["settled_cny"] == "1.250000"


def test_month_rollover_uses_shanghai_calendar_not_utc_date(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, worst_case="1.00")
    utc_before_shanghai_midnight = datetime(2026, 7, 31, 15, 30, tzinfo=UTC)
    utc_after_shanghai_midnight = datetime(2026, 7, 31, 16, 30, tzinfo=UTC)

    ledger.reserve(count=1, now=utc_before_shanghai_midnight)
    ledger.reserve(count=1, now=utc_after_shanghai_midnight)

    assert (tmp_path / "dialogue-usage-2026-07.json").exists()
    assert (tmp_path / "dialogue-usage-2026-08.json").exists()


@pytest.mark.parametrize(
    "payload",
    [
        "{not-json",
        json.dumps(
            {
                "schema_version": 1,
                "month": "2026-07",
                "settled_cny": "0.00",
                "reservations": {},
                "requires_review": False,
                "unexpected": "state-sentinel",
            }
        ),
    ],
)
def test_corrupt_or_unknown_ledger_fails_closed_without_traceback_state(
    tmp_path: Path, payload: str
) -> None:
    ledger = make_ledger(tmp_path)
    destination = tmp_path / "dialogue-usage-2026-07.json"
    destination.write_text(payload, encoding="utf-8")

    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$") as caught:
        ledger.reserve(count=1, now=SHANGHAI_NOW)

    _assert_sanitized_ledger_traceback(
        caught.value,
        forbidden_values=(str(destination), "state-sentinel", payload),
    )


def test_atomic_replacement_failure_preserves_prior_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_ledger(tmp_path)
    destination = tmp_path / "dialogue-usage-2026-07.json"
    original = json.dumps(
        {
            "schema_version": 1,
            "month": "2026-07",
            "settled_cny": "0.000000",
            "reservations": {},
            "requires_review": False,
        }
    )
    destination.write_text(original, encoding="utf-8")

    def deterministic_token(byte_count: int) -> str:
        return "a" * (byte_count * 2)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace-failure-sentinel")

    monkeypatch.setattr("companion.dialogue.usage.secrets.token_hex", deterministic_token)
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$") as caught:
        ledger.reserve(count=1, now=SHANGHAI_NOW)

    assert destination.read_text(encoding="utf-8") == original
    _assert_sanitized_ledger_traceback(
        caught.value,
        forbidden_values=(
            str(destination),
            "replace-failure-sentinel",
            "a" * 32,
            "1.000000",
        ),
    )


def test_settlement_atomic_replacement_failure_clears_amount_traceback_locals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_ledger(tmp_path, worst_case="1.234567", input_rate="2.00")

    def deterministic_token(byte_count: int) -> str:
        return "b" * (byte_count * 2)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("settlement-replace-sentinel")

    monkeypatch.setattr("companion.dialogue.usage.secrets.token_hex", deterministic_token)
    reservation = ledger.reserve(count=1, now=SHANGHAI_NOW)[0]
    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$") as caught:
        ledger.settle(
            reservation,
            ProviderUsage(
                prompt_tokens=1_000_000,
                completion_tokens=0,
                total_tokens=1_000_000,
            ),
            SHANGHAI_NOW,
        )

    _assert_sanitized_ledger_traceback(
        caught.value,
        forbidden_values=(
            "settlement-replace-sentinel",
            "b" * 32,
            "1.234567",
            "2.000000",
        ),
    )


def test_ledger_contains_only_non_content_accounting_fields(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    ledger.reserve(count=1, now=SHANGHAI_NOW)

    payload = json.loads((tmp_path / "dialogue-usage-2026-07.json").read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "month",
        "settled_cny",
        "reservations",
        "requires_review",
    }
    encoded = json.dumps(payload)
    assert all(field not in encoded for field in ("request", "prompt", "answer", "context"))


def _assert_sanitized_ledger_traceback(error: Exception, forbidden_values: tuple[str, ...]) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    assert all(value not in str(error) for value in forbidden_values)
    for frame, _ in traceback.walk_tb(error.__traceback__):
        filename = frame.f_code.co_filename.replace("/", "\\")
        if filename.endswith("src\\companion\\dialogue\\usage.py"):
            assert all(
                forbidden not in repr(value)
                for value in frame.f_locals.values()
                for forbidden in forbidden_values
            )


def test_cancel_unused_releases_only_the_exact_unsent_reservation(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    first, second = ledger.reserve(count=2, now=SHANGHAI_NOW)
    ledger.settle(first, usage=None, now=SHANGHAI_NOW)

    ledger.cancel_unused(second, now=SHANGHAI_NOW)

    payload = json.loads((tmp_path / "dialogue-usage-2026-07.json").read_text(encoding="utf-8"))
    assert payload["reservations"] == {}
    assert Decimal(payload["settled_cny"]) == Decimal("1.00")
    make_ledger(tmp_path).ensure_available(count=99, now=SHANGHAI_NOW)


@pytest.mark.parametrize("fault", ("duplicate", "month", "amount", "missing", "type"))
def test_cancel_unused_fails_closed_without_releasing_capacity(tmp_path: Path, fault: str) -> None:
    ledger = make_ledger(tmp_path)
    (reservation,) = ledger.reserve(count=1, now=SHANGHAI_NOW)
    destination = tmp_path / "dialogue-usage-2026-07.json"
    if fault == "duplicate":
        ledger.cancel_unused(reservation, now=SHANGHAI_NOW)
        ledger.reserve(count=1, now=SHANGHAI_NOW)
        target: object = reservation
    elif fault == "month":
        target = reservation.model_copy(update={"month": "2026-06"})
    elif fault == "amount":
        target = reservation.model_copy(update={"reserved_cny": Decimal("0.50")})
    elif fault == "missing":
        target = reservation.model_copy(update={"reservation_id": "f" * 32})
    else:
        target = {"reservation_id": reservation.reservation_id}
    before = destination.read_bytes()

    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$") as caught:
        ledger.cancel_unused(target, now=SHANGHAI_NOW)

    assert destination.read_bytes() == before
    _assert_sanitized_ledger_traceback(
        caught.value, forbidden_values=(str(destination), reservation.reservation_id)
    )


def test_cancel_unused_rejects_corrupt_ledger_without_traceback_state(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    (reservation,) = ledger.reserve(count=1, now=SHANGHAI_NOW)
    destination = tmp_path / "dialogue-usage-2026-07.json"
    destination.write_text("state-sentinel", encoding="utf-8")

    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$") as caught:
        ledger.cancel_unused(reservation, now=SHANGHAI_NOW)

    assert destination.read_text(encoding="utf-8") == "state-sentinel"
    _assert_sanitized_ledger_traceback(
        caught.value, forbidden_values=(str(destination), "state-sentinel")
    )


def test_cancel_unused_rejects_sub_quantum_amount_change(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    (reservation,) = ledger.reserve(count=1, now=SHANGHAI_NOW)
    destination = tmp_path / "dialogue-usage-2026-07.json"
    before = destination.read_bytes()
    altered = reservation.model_copy(
        update={"reserved_cny": reservation.reserved_cny - Decimal("0.000000001")}
    )
    with pytest.raises(UsageLedgerError, match=r"^usage ledger unavailable$"):
        ledger.cancel_unused(altered, now=SHANGHAI_NOW)
    assert destination.read_bytes() == before
    restarted = make_ledger(tmp_path)
    restarted.cancel_unused(reservation, now=SHANGHAI_NOW)
    assert json.loads(destination.read_text())["reservations"] == {}
