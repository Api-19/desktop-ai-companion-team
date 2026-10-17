from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from companion.dialogue.models import BudgetConfig, ProviderUsage

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_CNY_QUANTUM = Decimal("0.000001")
_RESERVATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class UsageLedgerError(RuntimeError):
    pass


class BudgetExceeded(UsageLedgerError):
    pass


class UsageReviewRequired(UsageLedgerError):
    pass


class UsageReservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reservation_id: str = Field(pattern=r"[0-9a-f]{32}")
    month: str = Field(pattern=r"\d{4}-\d{2}")
    reserved_cny: Decimal = Field(gt=0)


class UsageSnapshot(BaseModel):
    """Content-free read-only accounting for the next single request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    settled_cny: Decimal = Field(ge=0, allow_inf_nan=False)
    reserved_cny: Decimal = Field(ge=0, allow_inf_nan=False)
    accounted_cny: Decimal = Field(ge=0, allow_inf_nan=False)
    monthly_cap_cny: Decimal = Field(gt=0, allow_inf_nan=False)
    status: Literal["allowed", "blocked", "review_required"]


class _LedgerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    month: str = Field(pattern=r"\d{4}-\d{2}")
    settled_cny: Decimal = Field(ge=0)
    reservations: dict[str, Decimal]
    requires_review: bool

    @model_validator(mode="after")
    def require_valid_amounts(self) -> _LedgerState:
        amounts = (self.settled_cny, *self.reservations.values())
        if any(not amount.is_finite() for amount in amounts):
            raise ValueError("invalid ledger amount")
        if any(amount <= 0 for amount in self.reservations.values()):
            raise ValueError("invalid reservation")
        if any(_RESERVATION_ID_PATTERN.fullmatch(key) is None for key in self.reservations):
            raise ValueError("invalid reservation")
        return self


class MonthlyUsageLedger:
    """Single-process, Shanghai-month usage ledger with conservative reservations."""

    def __init__(self, root: Path, config: BudgetConfig) -> None:
        self._root = root
        self._config = config

    def snapshot(self, now: datetime) -> UsageSnapshot:
        """Read current Shanghai-month totals without reserving or writing."""
        state: _LedgerState | None = None
        snapshot: UsageSnapshot | None = None
        current: Path | None = None
        failure = False
        try:
            month = _shanghai_month(now)
            current = self._root.absolute()
            while True:
                if current.is_symlink():
                    raise ValueError("invalid ledger root")
                if current == current.parent:
                    break
                current = current.parent
            if self._path_for(month).is_symlink():
                raise ValueError("invalid ledger file")
            state = self._load_month(month)
            reserved = sum(state.reservations.values(), Decimal("0"))
            decision = self._availability_decision(state, 1)
            snapshot = UsageSnapshot(
                month=month,
                settled_cny=state.settled_cny,
                reserved_cny=reserved,
                accounted_cny=_cny(state.settled_cny + reserved),
                monthly_cap_cny=self._config.monthly_cap_cny,
                status={"allowed": "allowed", "exceeded": "blocked", "review": "review_required"}[
                    decision
                ],
            )
        except Exception:
            failure = True
        del state, current, now, self
        if failure:
            del snapshot
            raise UsageLedgerError("usage ledger unavailable")
        return snapshot

    def ensure_available(self, count: int, now: datetime) -> None:
        state: _LedgerState | None = None
        failure = False
        decision = "allowed"
        try:
            month = _shanghai_month(now)
            _validate_count(count)
            state = self._load_month(month)
            decision = self._availability_decision(state, count)
        except Exception:
            failure = True

        del state, now, self
        if failure:
            raise UsageLedgerError("usage ledger unavailable")
        if decision == "review":
            raise UsageReviewRequired("usage ledger requires review")
        if decision == "exceeded":
            raise BudgetExceeded("monthly budget exceeded")

    def reserve(self, count: int, now: datetime) -> tuple[UsageReservation, ...]:
        state: _LedgerState | None = None
        reservations: tuple[UsageReservation, ...] = ()
        failure = False
        decision = "allowed"
        try:
            month = _shanghai_month(now)
            _validate_count(count)
            state = self._load_month(month)
            decision = self._availability_decision(state, count)
            if decision == "allowed":
                reservations = self._append_reservations(state, count)
                self._atomic_write(state)
        except Exception:
            failure = True

        del state, now, self
        if failure:
            del reservations
            raise UsageLedgerError("usage ledger unavailable")
        if decision == "review":
            del reservations
            raise UsageReviewRequired("usage ledger requires review")
        if decision == "exceeded":
            del reservations
            raise BudgetExceeded("monthly budget exceeded")
        return reservations

    def settle(
        self,
        reservation: UsageReservation,
        usage: ProviderUsage | None,
        now: datetime,
    ) -> None:
        state: _LedgerState | None = None
        reserved: Decimal | None = None
        settled: Decimal | None = None
        requires_review: bool | None = None
        failure = False
        try:
            month = _shanghai_month(now)
            state = self._load_month(month)
            if reservation.month != month:
                raise ValueError("reservation month mismatch")
            reserved = state.reservations.pop(reservation.reservation_id)
            settled, requires_review = self._settlement_amount(reserved, usage)
            state.settled_cny = _cny(state.settled_cny + settled)
            state.requires_review = state.requires_review or requires_review
            self._atomic_write(state)
        except Exception:
            failure = True

        del state, reserved, settled, requires_review, reservation, usage, now, self
        if failure:
            raise UsageLedgerError("usage ledger unavailable")

    def cancel_unused(self, reservation: UsageReservation, now: datetime) -> None:
        """Release one reservation only after the caller proved no provider attempt happened."""
        state: _LedgerState | None = None
        reserved: Decimal | None = None
        failure = False
        try:
            if type(reservation) is not UsageReservation:
                raise ValueError("invalid reservation")
            reservation = UsageReservation.model_validate(
                reservation.model_dump(mode="python", warnings=False)
            )
            month = _shanghai_month(now)
            state = self._load_month(month)
            if reservation.month != month:
                raise ValueError("reservation month mismatch")
            reserved = state.reservations.pop(reservation.reservation_id)
            if reserved != reservation.reserved_cny:
                raise ValueError("reservation amount mismatch")
            self._atomic_write(state)
        except Exception:
            failure = True

        del state, reserved, reservation, now, self
        if failure:
            raise UsageLedgerError("usage ledger unavailable")

    def _availability_decision(self, state: _LedgerState, count: int) -> str:
        if state.requires_review:
            return "review"
        required = self._reservation_amount() * count
        accounted = _cny(state.settled_cny + sum(state.reservations.values(), Decimal("0")))
        if accounted + required > _cny(self._config.monthly_cap_cny):
            return "exceeded"
        return "allowed"

    def _append_reservations(self, state: _LedgerState, count: int) -> tuple[UsageReservation, ...]:
        reservation_amount = self._reservation_amount()
        created: list[UsageReservation] = []
        for _ in range(count):
            reservation_id = secrets.token_hex(16)
            if reservation_id in state.reservations:
                raise RuntimeError("reservation collision")
            state.reservations[reservation_id] = reservation_amount
            created.append(
                UsageReservation(
                    reservation_id=reservation_id,
                    month=state.month,
                    reserved_cny=reservation_amount,
                )
            )
        return tuple(created)

    def _reservation_amount(self) -> Decimal:
        return _cny(self._config.worst_case_request_cny)

    def _settlement_amount(
        self, reserved: Decimal, usage: ProviderUsage | None
    ) -> tuple[Decimal, bool]:
        if not isinstance(usage, ProviderUsage):
            return reserved, False
        try:
            validated_usage = ProviderUsage.model_validate(usage.model_dump())
            actual_cny = (
                Decimal(validated_usage.prompt_tokens) * self._config.input_cny_per_million
                + Decimal(validated_usage.completion_tokens) * self._config.output_cny_per_million
            ) / Decimal(1_000_000)
            if not actual_cny.is_finite() or actual_cny < 0:
                raise ValueError("invalid actual cost")
        except Exception:
            return reserved, False
        if actual_cny > reserved:
            return reserved, True
        return _cny(actual_cny), False

    def _load_month(self, month: str) -> _LedgerState:
        destination = self._path_for(month)
        if not destination.exists():
            return _LedgerState(
                schema_version=1,
                month=month,
                settled_cny=Decimal("0"),
                reservations={},
                requires_review=False,
            )
        state = _read_state(destination)
        if state.month != month:
            raise ValueError("ledger month mismatch")
        return state

    def _atomic_write(self, state: _LedgerState) -> None:
        destination = self._path_for(state.month)
        temporary: Path | None = None
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._root,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                file.write(_serialize_state(state))
                file.flush()
                os.fsync(file.fileno())
            verified = _read_state(temporary)
            if verified != state:
                raise ValueError("ledger readback mismatch")
            temporary.replace(destination)
            if os.name == "posix":
                directory_fd = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _path_for(self, month: str) -> Path:
        return self._root / f"dialogue-usage-{month}.json"


def _validate_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("invalid count")


def _shanghai_month(now: datetime) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("naive clock")
    return now.astimezone(_SHANGHAI).strftime("%Y-%m")


def _cny(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ValueError("invalid money")
    return value.quantize(_CNY_QUANTUM, rounding=ROUND_CEILING)


def _read_state(path: Path) -> _LedgerState:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("ledger payload is invalid")
    return _LedgerState.model_validate(payload)


def _serialize_state(state: _LedgerState) -> str:
    payload = {
        "schema_version": state.schema_version,
        "month": state.month,
        "settled_cny": format(_cny(state.settled_cny), "f"),
        "reservations": {
            reservation_id: format(_cny(amount), "f")
            for reservation_id, amount in state.reservations.items()
        },
        "requires_review": state.requires_review,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
