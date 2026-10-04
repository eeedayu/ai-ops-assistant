from typing import Any

from app.rag.service import RAGService
from app.agents.tools.base_tool import BaseTool,ToolRisk


class KnowledgeSearchTool(BaseTool):

    def __init__(
        self,
        rag_service: RAGService,
    ):
        super().__init__(
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
            risk_level=ToolRisk.READ,
        )
        self.rag_service = rag_service

    async def execute(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        results = await self.rag_service.search(
            query=query,
            prefetch_limit=20,
            top_k=top_k,
        )

        return {
            "results": [
                {
                    "source": result.document.metadata.get(
                        "source",
                        "unknown",
                    ),
                    "page": result.document.metadata.get(
                        "page"
                    ),
                    "section": result.document.metadata.get(
                        "section"
                    ),
                    "score": result.score,
                    "content": result.document.page_content,
                }
                for result in results
            ]
        }