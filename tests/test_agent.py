from app.agents.agent import AgentService
from app.agents.registry import (
    build_default_registry,
)
from app.repositories.session_repository import (
    SessionRepository,
)


class FakeLLMService:

    def __init__(self):

        self.calls = 0

    async def chat_with_tools(
        self,
        messages,
        tools,
    ):

        self.calls += 1

        if self.calls == 1:

            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": (
                                "get_device_status"
                            ),
                            "arguments": "{}",
                        },
                    }
                ],
            }

        return {
            "role": "assistant",
            "content": (
                "根据设备状态，"
                "CPU使用率较高，"
                "建议进一步检查高CPU进程。"
            ),
        }


async def test_agent_loop():

    repository = (
        SessionRepository()
    )

    llm = FakeLLMService()

    registry = (
        build_default_registry()
    )

    agent = AgentService(
        session_repository=repository,
        llm_service=llm,
        tool_registry=registry,
    )

    state = await agent.run(
        session_id="test-agent",
        user_message=(
            "帮我分析服务器状态"
        ),
    )

    assert (
        state.status
        == "completed"
    )

    assert (
        state.final_answer
        is not None
    )

    assert len(
        state.tool_calls
    ) == 1

    assert (
        state.tool_calls[0]
        .tool_name
        == "get_device_status"
    )