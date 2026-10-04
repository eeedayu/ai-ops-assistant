class AppException(Exception):
    """
    应用基础异常。
    """

    def __init__(
        self,
        message: str,
        code: str = "app_error",
        status_code: int = 500,
    ):
        super().__init__(message)

        self.message = message
        self.code = code
        self.status_code = status_code


class LLMServiceError(AppException):
    def __init__(
        self,
        message: str = "LLM service error",
    ):
        super().__init__(
            message=message,
            code="llm_service_error",
            status_code=503,
        )


class SessionNotFoundError(AppException):
    def __init__(
        self,
        message: str = "Session not found",
    ):
        super().__init__(
            message=message,
            code="session_not_found",
            status_code=404,
        )


class InvalidRequestError(AppException):
    def __init__(
        self,
        message: str = "Invalid request",
    ):
        super().__init__(
            message=message,
            code="invalid_request",
            status_code=400,
        )