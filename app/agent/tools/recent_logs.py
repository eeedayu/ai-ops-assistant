from app.agents.tools.base_tool import BaseTool,ToolRisk


class RecentLogsTool(BaseTool):

    def __init__(
        self,
    ):
        super().__init__(
            name="get_recent_logs",
            description=(
                "查询服务器最近的系统日志。"
                "适合分析异常发生时间和错误模式。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": (
                            "最多返回多少条日志"
                        ),
                        "minimum": 1,
                        "maximum": 10,
                    }
                },
                "required": [],
            },
            risk_level=ToolRisk.READ,
        )

    async def execute(
        self,
        limit: int = 5,
    ) -> list[str]:
        """
        查询最近系统日志。
        """

        logs = [
            "10:01 CPU usage > 90%",
            "10:03 nginx response time increased",
            "10:05 worker process restarted",
            "10:06 upstream connection timeout",
            "10:08 request latency increased",
            "10:10 worker restart completed",
        ]

        return logs[-limit:]