from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ToolRisk:

    READ = "read"

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"


@dataclass
class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    timeout: float = 10.0
    risk_level: str = ToolRisk.LOW
    requires_approval: bool = False

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """声明一个通用的处理函数"""
        pass
