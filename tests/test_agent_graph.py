from langchain_core.messages import AIMessage

from app.agent.graph import should_continue
from app.agent.graph_state import AgentState


def make_state(message: AIMessage) -> AgentState:
    return {
        "messages": [message],
        "user_input": "",
        "device_id": None,
        "memories": [],
        "diagnosis": None,
        "trace": [],
    }


def test_should_continue_routes_tool_calls_to_tools():
    state = make_state(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_device_status",
                    "args": {},
                    "id": "call-1",
                }
            ],
        )
    )

    assert should_continue(state) == "tools"


def test_should_continue_routes_final_response_to_diagnosis():
    state = make_state(AIMessage(content="诊断结果"))

    assert should_continue(state) == "diagnose"
