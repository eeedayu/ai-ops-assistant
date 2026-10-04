from typing import Any
from app.agents.tools.base_tool import BaseTool,ToolRisk


class MonitorMetricsTool(BaseTool):

    def __init__(
        self,
    ):
        super().__init__(
            name="query_monitor_metrics",
            description=(
                "查询服务器监控指标。"
                "可查询cpu、memory、network。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": [
                            "cpu",
                            "memory",
                            "network",
                        ],
                        "description": (
                            "要查询的监控指标"
                        ),
                    }
                },
                "required": [
                    "metric"
                ],
            },
            risk_level=ToolRisk.READ,
        )

    async def execute(
        self,
        metric: str,
    ) -> dict[str, Any]:
        """
        查询监控指标。
        """

        metrics = {
            "cpu": {
                "avg_5m": 91.2,
                "avg_15m": 87.6,
                "load_1m": 8.42,
                "load_5m": 7.91,
            },
            "memory": {
                "avg_5m": 71.3,
                "avg_15m": 69.8,
            },
            "network": {
                "avg_latency_ms": 23.4,
                "packet_loss": 0.1,
            },
        }

        if metric not in metrics:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        return metrics[metric]