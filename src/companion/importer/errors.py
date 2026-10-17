class ImportValidationError(ValueError):
    def __init__(
        self,
        code: str,
        detail: str,
        *,
        message_index: int | None = None,
        field_path: str | None = None,
    ) -> None:
        self.code = code
        self.message_index = message_index
        self.field_path = field_path

        metadata: list[str] = []
        if message_index is not None:
            metadata.append(f"message_index={message_index}")
        if field_path is not None:
            metadata.append(f"field_path={field_path}")
        suffix = f" ({', '.join(metadata)})" if metadata else ""
        super().__init__(f"{code}: {detail}{suffix}")
