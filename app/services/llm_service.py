from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import LLMServiceError
from app.schemas.agent import DiagnosisResult


class LLMService:

    def __init__(self):
        settings = get_settings()

        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
        )

        self.model = settings.llm_model
        self.reasoning_effort = (
            settings.llm_reasoning_effort
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, Any]],
    ) -> str:

        try:

            response = (
                await self.client
                .chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            )

            content = (
                response.choices[0]
                .message
                .content
            )

            return content or ""

        except Exception as exc:

            raise LLMServiceError(
                "LLM调用失败"
            ) from exc

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:

        try:

            stream = (
                await self.client
                .chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
            )

            async for chunk in stream:

                content = (
                    chunk.choices[0]
                    .delta
                    .content
                )

                if content:
                    yield content

        except Exception as exc:

            raise LLMServiceError(
                "LLM流式调用失败"
            ) from exc

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:

        try:

            response = (
                await self.client
                .chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    reasoning_effort=(
                        self.reasoning_effort
                    ),
                )
            )

            message = (
                response
                .choices[0]
                .message
            )

            return message.model_dump(
                exclude_none=True
            )

        except Exception as exc:

            raise LLMServiceError(
                "LLM Tool Calling 调用失败"
            ) from exc

    async def generate_diagnosis(
        self,
        messages: list[dict[str, Any]],
    ) -> DiagnosisResult:

        try:

            response = (
                await self.client
                .beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一名高级智能运维专家。"
                                "请根据前面收集到的故障现象、"
                                "设备状态、日志和监控指标，"
                                "完成最终故障诊断。"
                                "只能基于已获得的证据进行判断。"
                            ),
                        },
                        *messages,
                    ],
                    response_format=DiagnosisResult,
                )
            )

            result = (
                response
                .choices[0]
                .message
                .parsed
            )

            if result is None:
                raise LLMServiceError(
                    "模型没有返回有效诊断结果"
                )

            return result

        except Exception as exc:

            raise LLMServiceError(
                "生成诊断结果失败"
            ) from exc