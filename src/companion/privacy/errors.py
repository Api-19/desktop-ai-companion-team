class PrivacyValidationError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


class _PrivacyFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


def _sanitized_public_privacy_failure(
    error: Exception,
    *,
    internal_failure_type: type[Exception],
    allowed_details: dict[str, tuple[str, ...]],
    fallback: tuple[str, str],
) -> tuple[str, str]:
    try:
        code = error.code
        if type(code) is not str:
            return fallback
        details = allowed_details.get(code)
        if details is None:
            return fallback
        if type(error) is internal_failure_type:
            detail = error.detail
            if type(detail) is str and detail in details:
                return code, detail
        elif type(error) is PrivacyValidationError:
            args = error.args
            if type(args) is tuple and len(args) == 1 and type(args[0]) is str:
                for detail in details:
                    if args[0] == f"{code}: {detail}":
                        return code, detail
    except Exception:
        return fallback
    return fallback
