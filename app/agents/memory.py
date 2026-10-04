from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryItem:
    content: str
    created_at: datetime
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class LongTermMemory:

    def __init__(self):
        self.items: list[MemoryItem] = []

    async def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:

        item = MemoryItem(
            content=content,
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        self.items.append(item)

        return item

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:

        results = []

        for item in reversed(self.items):

            if query and query in item.content:
                results.append(item)

            if len(results) >= top_k:
                break

        return results

    async def save_diagnosis(
        self,
        diagnosis,
        device_id: str | None = None,
    ):

        content = (
            f"设备: {device_id or '未知'}\n"
            f"故障类型: {diagnosis.fault_type}\n"
            f"诊断结论: {diagnosis.summary}\n"
            f"风险等级: {diagnosis.risk_level}\n"
            f"处理建议: "
            f"{'；'.join(diagnosis.recommended_actions)}"
        )

        metadata = {
            "type": "diagnosis",
            "device_id": device_id,
            "fault_type": diagnosis.fault_type,
            "risk_level": diagnosis.risk_level,
            "sources": diagnosis.sources,
        }

        return await self.save(
            content=content,
            metadata=metadata,
        )