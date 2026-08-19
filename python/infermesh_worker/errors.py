class AdapterError(Exception):
    def __init__(self, message: str, *, execution_started: bool) -> None:
        super().__init__(message)
        self.execution_started = execution_started
