import logging
from fastapi import APIRouter, Depends, Request

from app.dependencies import get_chat_service
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from app.services.chat_service import ChatService


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    http_request: Request,
    request: ChatRequest,
    chat_service: ChatService = Depends(
        get_chat_service
    ),
):

    logger.info(
        "chat request session_id=%s request_id=%s",
        request.session_id,
        getattr(
            http_request.state,
            "request_id",
            "",
        ),
    )
    
    answer = await chat_service.chat(
        session_id=request.session_id,
        user_message=request.message,
    )

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
    )