import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.tools import (
    get_device_status,
    get_recent_logs,
    query_monitor_metrics,
    restart_service,
)
from app.rag.service import RAGService
from app.agents.tools.device_status import DeviceStatusTool
from app.agents.tools.monitor_metrics import MonitorMetricsTool
from app.agents.tools.recent_logs import RecentLogsTool
from app.agents.tools.restart_service import RestartServiceTool
from app.agents.tools.knowledge import KnowledgeSearchTool


ToolFunction = Callable[..., Any]


@dataclass
class BaseTool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction

    timeout: float = 10.0
    risk_level: str = "low"


class ToolRegistry:

    def __init__(self):
        self._tools: dict[
            str,
            Any,
        ] = {}

    def register(
        self,
        tool: Any,
    ) -> None:

        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: "
                f"{tool.name}"
            )

        self._tools[tool.name] = tool

    def get(
        self,
        name: str,
    ) -> Any | None:

        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:

        tool = self.get(name)

        if tool is None:
            raise ValueError(
                f"Unknown tool: {name}"
            )

        function: ToolFunction = tool.execute

        if inspect.iscoroutinefunction(function):

            return await asyncio.wait_for(
                function(**arguments),
                timeout=tool.timeout,
            )

        return await asyncio.wait_for(
            asyncio.to_thread(
                function,
                **arguments,
            ),
            timeout=tool.timeout,
        )



def build_default_registry(
    rag_service: RAGService,
) -> ToolRegistry:

    registry = ToolRegistry()

    registry.register(DeviceStatusTool())
    registry.register(MonitorMetricsTool())
    registry.register(RecentLogsTool())
    registry.register(RestartServiceTool())
    registry.register(KnowledgeSearchTool(rag_service))

    return registry

'''
def build_default_registry(
    rag_service=None,
) -> ToolRegistry:

    registry = ToolRegistry()

    registry.register(
            BaseTool(
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
                function=get_device_status,
            )
        )
    
    registry.register(
        BaseTool(
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
            function=get_recent_logs,
        )
    )

    registry.register(
        BaseTool(
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
            function=query_monitor_metrics,
        )
    )

    registry.register(
        BaseTool(            
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
            function=restart_service,
            risk_level="high",
        )
    )

    if rag_service is not None:

        async def search_knowledge(
            query: str,
            top_k: int = 5,
        ):

            results = (
                rag_service.search(
                    query=query,
                    top_k=top_k,
                )
            )

            return {
                "results": [
                    {
                        "source": (
                            result
                            .document
                            .metadata
                            .get(
                                "source",
                                "unknown",
                            )
                        ),
                        "page": (
                            result
                            .document
                            .metadata
                            .get(
                                "page"
                            )
                        ),
                        "section": (
                            result
                            .document
                            .metadata
                            .get(
                                "section"
                            )
                        ),
                        "score": result.score,
                        "content": (
                            result
                            .document
                            .page_content
                        ),
                    }
                    for result in results
                ]
            }

        registry.register(
            BaseTool(
                name="search_knowledge",
                description=(
                    "搜索企业内部知识库。"
                    "可以查询SOP、设备手册、"
                    "故障案例、维护规范等知识。"
                    "当用户询问企业内部规则、"
                    "SOP或设备知识时优先使用该工具。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "要检索的知识问题"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": [
                        "query"
                    ],
                },
                function=search_knowledge,
                risk_level="low",
            )
        )

    return registry
'''