class MemoryServiceError(RuntimeError):
    pass


class MemoryValidationError(MemoryServiceError):
    pass


class MemoryAuthorizationError(MemoryServiceError):
    pass


class MemoryStoreError(MemoryServiceError):
    pass
