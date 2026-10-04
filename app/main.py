from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.middleware.request_id import (
    RequestIDMiddleware,
)
from app.core.logging import (
    setup_logging,
)
from app.api.agent import (
    router as agent_router,
)

from app.api.rag import (
    router as rag_router,
)


setup_logging()


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.add_middleware(
    RequestIDMiddleware
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://eeedayu.dpdns.org:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    agent_router
)

@app.exception_handler(AppException)
async def app_exception_handler(
    request: Request,
    exc: AppException,
):

    request_id = getattr(
        request.state,
        "request_id",
        "",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
        },
    )


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


app.include_router(
    chat_router
)

app.include_router(
    rag_router
)