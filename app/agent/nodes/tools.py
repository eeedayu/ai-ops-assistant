import json

from langchain_core.messages import ToolMessage

from app.agent.graph_state import AgentState
from app.agent.tool import (
    search_knowledge,
    get_device_status,
    get_recent_logs,
    query_monitor_metrics,
    restart_device
)


TOOLS = {
    "search_knowledge":
        search_knowledge,

    "get_device_status":
        get_device_status,

    "get_recent_logs":
        get_recent_logs,

    "query_monitor_metrics":
        query_monitor_metrics,

    "restart_device":
        restart_device,
}


async def tool_node(
    state: AgentState,
):

    last_message = state["messages"][-1]

    results = []

    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]

        tool = TOOLS.get(tool_name)

        if tool is None:

            results.append(
                ToolMessage(
                    content=(
                        f"Tool {tool_name} 不存在"
                    ),
                    tool_call_id=
                        tool_call["id"],
                )
            )

            continue

        try:

            result = await tool.ainvoke(
                tool_call["args"]
            )

            results.append(
                ToolMessage(
                    content=json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                    tool_call_id=
                        tool_call["id"],
                )
            )

        except Exception as e:

            results.append(
                ToolMessage(
                    content=json.dumps(
                        {
                            "status":
                                "error",

                            "message":
                                str(e),
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=
                        tool_call["id"],
                )
            )

    return {
        "messages": results,

        "trace": [
            {
                "event":
                    "tool_execution",
                "data": {
                    "count":
                        len(results),
                },
            }
        ],
    }