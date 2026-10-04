import uuid
from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime
from app.agent.graph_state import AgentState


def load_memory(
    state: AgentState,
    runtime: Runtime,
):
    """
    Agent 开始前读取 Long-term Memory
    """
    store = runtime.store
    query = state["user_input"]

    device_id = state.get("device_id")

    namespace = (
        "ops_memory",
        device_id or "global",
    )

    # 第一版使用 Store 的简单读取方式。
    # 后面可以升级为向量检索。
    items = store.search(
        namespace,
        limit=5,
    )

    memories = []

    for item in items:

        value = item.value

        content = value.get(
            "content",
            "",
        )

        # 简单过滤
        if query:

            keywords = query.split()

            if not any(
                keyword in content
                for keyword in keywords
            ):
                continue

        memories.append({
            "content": content,
            "metadata": value.get(
                "metadata",
                {},
            ),
        })

    trace_event = {
        "event": "memory_loaded",
        "data": {
            "count": len(memories),
        },
    }

    return {
        "memories": memories,
        "trace": [trace_event],
    }


def save_memory(
    state: AgentState,
    runtime: Runtime,
):

    diagnosis = state["diagnosis"]

    device_id = (
        state.get("device_id")
        or "global"
    )

    namespace = (
        "ops_memory",
        device_id,
    )

    store = runtime.store
    store.put(
        namespace,
        str(uuid.uuid4()),
        {
            "content": (
                f"设备：{device_id}\n"
                f"故障类型："
                f"{diagnosis['fault_type']}\n"
                f"诊断："
                f"{diagnosis['summary']}\n"
                f"风险等级："
                f"{diagnosis['risk_level']}\n"
                f"处理建议："
                f"{'；'.join(diagnosis['recommended_actions'])}"
            ),
            "metadata": {
                "type": "diagnosis",
                "device_id":
                    device_id,
                "fault_type":
                    diagnosis[
                        "fault_type"
                    ],
                "risk_level":
                    diagnosis[
                        "risk_level"
                    ],
            },
        },
    )

    return {
        "trace": [
            {
                "event":
                    "memory_saved",

                "data": {
                    "device_id":
                        device_id,
                },
            }
        ],
    }