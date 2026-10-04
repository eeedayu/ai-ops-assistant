from langchain.tools import tool
from langgraph.types import interrupt

from app.dependencies import (get_rag_service
    )
from app.agent.tools.device_status import DeviceStatusTool
from app.agent.tools.monitor_metrics import MonitorMetricsTool
from app.agent.tools.recent_logs import RecentLogsTool
from app.agent.tools.knowledge import KnowledgeSearchTool
from app.agent.tools.restart_service import RestartServiceTool


@tool
async def search_knowledge(
    query: str,
) -> str:
    """
    搜索企业运维知识库，包括设备手册、
    故障案例、维护 SOP 和历史处理方案。
    """
    rag_service = KnowledgeSearchTool(get_rag_service())
    result = await rag_service.execute(
        query=query,
        top_k=5,
    )

    return str(result)


@tool
async def get_device_status(
    device_id: str,
) -> dict:
    """
    获取设备当前实时状态。
    """

    return await DeviceStatusTool().execute()


@tool
async def get_recent_logs(
    limit: int = 5,
) -> list:
    """
    获取设备最近一段时间的运行日志。
    """

    return await RecentLogsTool().execute(
        limit,
    )


@tool
async def query_monitor_metrics(
    metric: str,
) -> dict:
    """
    获取设备实时监控指标。
    """

    return await MonitorMetricsTool().execute(
        metric
    )


@tool
async def restart_device(
    service_name: str,
):
    """
    重启设备。
    高风险操作，需要人工审批。
    """

    # ★ 注意：
    # interrupt 必须放在真实副作用之前
    decision = interrupt({
        "type": "tool_approval",

        "tool_name":
            "restart_device",

        "risk_level":
            "high",

        "arguments": {
            "service_name": service_name,
        },

        "message":
            f"是否批准重启设备 {service_name}？",
    })

    # 用户拒绝
    if decision.get("approved") is not True:

        return (
            "用户拒绝执行 restart_device。"
            f"原因："
            f"{decision.get('reason', '未提供')}"
        )

    # ★ 真正的副作用在 interrupt 后
    return await RestartServiceTool().execute(
        service_name
    )