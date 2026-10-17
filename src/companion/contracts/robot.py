from typing import Literal

from companion.contracts.base import ContractModel


class RobotEvent(ContractModel):
    state: Literal["idle", "listening", "thinking", "speaking", "muted", "error"]
    detail: str | None = None
