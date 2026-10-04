from langchain_openai import ChatOpenAI

from app.agent.graph_state import AgentState
from app.core.config import get_settings
from app.agent.tool import (
    search_knowledge,
    get_device_status,
    get_recent_logs,
    query_monitor_metrics,
    restart_device
)


settings = get_settings()

llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    reasoning_effort=settings.llm_reasoning_effort,
)

tools = [
    search_knowledge,
    get_device_status,
    get_recent_logs,
    query_monitor_metrics,
    restart_device,
]

llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """
你是一名 AI 运维故障诊断 Agent。

任务：

1. 理解用户描述
2. 根据需要主动调用工具
3. 使用知识库获取相关运维知识
4. 使用设备状态、日志、监控获取实时证据
5. 综合多个工具结果进行诊断
6. 高风险操作必须等待人工审批
7. 不要把历史 Memory 当成当前实时状态
8. 最终不要编造没有证据支持的结论
"""


def agent_node(
    state: AgentState,
):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Memory 注入
    memories = state.get(
        "memories",
        [],
    )

    if memories:

        memory_text = "\n\n".join(
            item["content"]
            for item in memories
        )

        messages.append({
            "role": "system",
            "content": (
                "以下是相关历史诊断记录，"
                "只能作为辅助参考：\n"
                + memory_text
            ),
        })

    messages.extend(
        state["messages"]
    )

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response],

        "trace": [
            {
                "event": "agent_llm",
                "data": {
                    "has_tool_calls":
                        bool(
                            response.tool_calls
                        ),
                },
            }
        ],
    }