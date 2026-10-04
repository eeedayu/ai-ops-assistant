from app.agents.tools.base_tool import BaseTool,ToolRisk


class RestartServiceTool(BaseTool):

    def __init__(
        self,
    ):
        super().__init__(
            name="restart_service",
            description=(
                "重启指定系统服务。"
                "这是高风险操作，必须经过用户确认。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": (
                            "要重启的服务名称"
                        ),
                    }
                },
                "required": [
                    "service_name"
                ],
            },
            requires_approval=True,
            risk_level=ToolRisk.HIGH,
        )

    async def execute(
        self,
        service_name: str,
    ) -> dict[str, str]:
        raise NotImplementedError(
            f"Restarting service {service_name!r} is not implemented"
        )