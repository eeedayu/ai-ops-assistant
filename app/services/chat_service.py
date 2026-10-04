from typing import Any

from app.core.config import get_settings
from app.core.exceptions import SessionNotFoundError
from app.repositories.session_repository import (
    SessionRepository,
)
from app.services.llm_service import LLMService


class ChatService:

    def __init__(
        self,
        session_repository: SessionRepository,
        llm_service: LLMService,
    ):
        self.session_repository = (
            session_repository
        )

        self.llm_service = llm_service

        self.settings = get_settings()

    def _build_messages(
        self,
        session_id: str,
        user_message: str,
    ) -> list[dict[str, Any]]:

        messages = (
            self.session_repository
            .get_messages(session_id)
        )

        if not messages:

            messages.append(
                {
                    "role": "system",
                    "content": (
                        "你是一名智能运维助手。"
                        "负责帮助用户分析服务器、"
                        "网络、数据库和应用系统故障。"
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return self._truncate_messages(
            messages
        )

    def _truncate_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        max_messages = (
            self.settings
            .max_context_messages
        )

        if len(messages) <= max_messages:
            return messages

        system_messages = [
            message
            for message in messages
            if message["role"] == "system"
        ]

        other_messages = [
            message
            for message in messages
            if message["role"] != "system"
        ]

        keep_count = max(
            max_messages - len(system_messages),
            1,
        )

        return (
            system_messages
            + other_messages[-keep_count:]
        )

    async def chat(
        self,
        session_id: str,
        user_message: str,
    ) -> str:

        self.session_repository.get_or_create(
            session_id
        )

        messages = self._build_messages(
            session_id=session_id,
            user_message=user_message,
        )

        self.session_repository.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        answer = await self.llm_service.chat(
            messages
        )

        self.session_repository.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        return answer