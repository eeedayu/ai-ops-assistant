from typing import Annotated, Any
import operator
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):

    # ReAct 对话上下文
    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    # 用户上下文
    user_input: str

    device_id: str | None

    # Long-term Memory 检索结果
    memories: list[dict[str, Any]]

    # 最终诊断
    diagnosis: dict[str, Any] | None

    # Trace
    trace: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]


class ResumeRequest(BaseModel):

    approved: bool

    reason: str | None = None