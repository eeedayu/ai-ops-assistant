from typing import Any
from app.agents.tools.base_tool import BaseTool,ToolRisk


class DeviceStatusTool(BaseTool):

    def __init__(
        self,
    ):
        super().__init__(
            name="get_device_status",
            description=(
                "查询服务器当前实时运行状态，"
                "包括CPU使用率、CPU温度、内存使用率、"
                "磁盘使用率和网络延迟。"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            risk_level=ToolRisk.READ,
        )

    async def execute(self,) -> dict[str, Any]:
        """
        查询服务器实时状态。
        """

        return {
            "device_id": "server-001",
            "cpu_usage": 92,
            "cpu_temperature": 87,
            "memory_usage": 73,
            "disk_usage": 68,
            "network_latency_ms": 21,
        }