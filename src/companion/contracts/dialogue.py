from companion.contracts.base import ContractModel


class DialogueRequest(ContractModel):
    user_id: str
    text: str
    recent_context: list[str]


class DialogueResponse(ContractModel):
    text: str
    disclosed_ai_identity: bool
    billable: bool
    blocked_reason: str | None
