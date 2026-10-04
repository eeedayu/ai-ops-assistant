import uuid
import time

from starlette.middleware.base import (
    BaseHTTPMiddleware,
)
from starlette.requests import Request


class RequestIDMiddleware(
    BaseHTTPMiddleware
):

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):

        request_id = request.headers.get(
            "X-Request-ID"
        )

        if not request_id:
            request_id = str(
                uuid.uuid4()
            )

        request.state.request_id = (
            request_id
        )

        start = time.perf_counter()

        response = await call_next(
            request
        )

        duration = (
            time.perf_counter()
            - start
        )

        response.headers[
            "X-Process-Time"
        ] = f"{duration:.4f}"

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response