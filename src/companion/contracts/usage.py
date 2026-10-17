from decimal import Decimal

from companion.contracts.base import ContractModel


class UsageDecision(ContractModel):
    allowed: bool
    month_total_cny: Decimal
    reason: str | None
