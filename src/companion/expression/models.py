from enum import StrEnum

from pydantic import ConfigDict

from companion.contracts.base import ContractModel


class Face(StrEnum):
    NEUTRAL = "neutral"
    ATTENTIVE = "attentive"
    THINKING = "thinking"
    TALKING = "talking"
    MUTED = "muted"
    TROUBLED = "troubled"


class HeadMotion(StrEnum):
    NONE = "none"
    NOD = "nod"
    TILT = "tilt"


class ExpressionFrame(ContractModel):
    model_config = ConfigDict(frozen=True)

    face: Face
    motion: HeadMotion
