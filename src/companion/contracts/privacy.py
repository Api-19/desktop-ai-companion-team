from companion.contracts.base import ContractModel
from companion.contracts.chat import ChatRecord


class RedactedRecord(ContractModel):
    record: ChatRecord
    redacted_text: str | None
    redaction_labels: list[str]
