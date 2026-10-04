from typing import Any

from app.rag.embeddings import (
    BGEEmbeddingService,
)
from app.rag.models import (
    ScoredDocument,Document
)
from app.rag.hybrid_qdrant_store import (
    HybridQdrantStore,
)
from app.services.llm_service import (
    LLMService,
)
from app.rag.retrieval.reranker import (
    BGEReranker,
)

AGENT_SYSTEM_PROMPT = """
                            你是一名 AI 运维故障诊断助手。

                            你的任务：
                            1. 理解用户描述的故障现象
                            2. 判断需要哪些信息
                            3. 使用工具获取证据
                            4. 使用知识库查询相关运维知识
                            5. 综合多个证据进行故障诊断
                            6. 给出风险等级和处理建议
                            7. 涉及高风险操作时必须请求人工确认

                            规则：
                            - 不要凭空编造设备状态
                            - 不要把知识库内容当成实时状态
                            - 不要把推测当成事实
                            - 诊断结论必须给出证据
                            - 信息不足时继续调用工具
                            - 最终输出结构化 DiagnosisResult
                            """


class RAGService:

    def __init__(
        self,
        embedding_service: BGEEmbeddingService,
        vector_store: HybridQdrantStore,
        llm_service: LLMService,
        reranker: BGEReranker,
    ):
        self.embedding_service = (
            embedding_service
        )

        self.vector_store = (
            vector_store
        )

        self.llm_service = (
            llm_service
        )

        self.reranker = reranker

    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ):
        """上层服务：纯稠密向量（语义）搜索"""
        # 1. 生成混合向量（内部包含 dense 和 sparse）
        query_embeddings = (
            self.embedding_service
            .embed_query_hybrid(
                query
            )
        )

        # 2. 仅提取 dense 向量并调用底层 vector_store 的 dense_search
        return (
            self.vector_store
            .dense_search(
                dense_query=(
                    query_embeddings[
                        "dense"
                    ]
                ),
                top_k=top_k,
                query_filter=(
                    self.vector_store
                    .build_filter(
                        metadata_filter
                    )
                ),
            )
        )

    def sparse_search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ):
        """上层服务：纯稀疏向量（关键词精准匹配）搜索"""
        # 1. 生成混合向量（内部包含 dense 和 sparse）
        query_embeddings = (
            self.embedding_service
            .embed_query_hybrid(
                query
            )
        )

        # 2. 仅提取 sparse 字典并调用底层 vector_store 的 sparse_search
        return (
            self.vector_store
            .sparse_search(
                sparse_query=(
                    query_embeddings[
                        "sparse"
                    ]
                ),
                top_k=top_k,
                query_filter=(
                    self.vector_store
                    .build_filter(
                        metadata_filter
                    )
                ),
            )
        )

    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        prefetch_limit: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ):

        query_embeddings = (
            self.embedding_service
            .embed_query_hybrid(
                query
            )
        )

        return (
            self.vector_store
            .hybrid_search(
                dense_query=(
                    query_embeddings[
                        "dense"
                    ]
                ),
                sparse_query=(
                    query_embeddings[
                        "sparse"
                    ]
                ),
                top_k=top_k,
                prefetch_limit=prefetch_limit,
                query_filter=(
                    self.vector_store
                    .build_filter(
                        metadata_filter
                    )
                ),
            )
        )

    def points_rerank(
        self,
        query: str,
        top_k: int = 10,
        points: list[ScoredDocument] | None = None,
    ):
        
        if points is None:
            return []

        candidates = []
        for point in points:
            candidates.append({
                "text": point.document.page_content,
                "score": point.score,
                "metadata": point.document.metadata,
            })

        if not candidates:
            return []

        results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        return results

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        prefetch_limit: int = 20,
        metadata_filter=None,
    ):

        result = []

        if mode == "dense":

            result = self.dense_search(
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

        if mode == "sparse":

            result = self.sparse_search(
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

        if mode == "hybrid":

            result = self.hybrid_search(
                query,
                top_k=top_k,
                prefetch_limit=prefetch_limit,
                metadata_filter=metadata_filter,
            )

        if not result:
            raise ValueError(
                f"Unsupported mode: {mode}"
        )

        return self.points_rerank(
            query=query,
            top_k=top_k,
            points=result,
        )

    async def search(
        self,
        query: str,
        prefetch_limit: int = 20,
        top_k: int = 5,
        metadata_filter=None,
        mode: str = "hybrid",
    ):

        documents = self.retrieve(
            query=query,
            prefetch_limit=prefetch_limit,
            top_k=top_k,
            metadata_filter=metadata_filter,
            mode=mode,
        )

        sources = []

        for index, document in enumerate(documents, start=1):
            metadata = document["metadata"]

            sources.append(
                ScoredDocument(
                    document=Document(
                        page_content=document["text"],
                        metadata=metadata
                    ),
                    score=document["rerank_score"]
                )
            )

        return sources

    @staticmethod
    def build_context(
        results: list[ScoredDocument],
    ) -> tuple[str, list[dict[str, Any]]]:

        context_parts = []
        sources = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            document = result.document
            metadata = document.metadata

            source_id = f"S{index}"

            source = {
                "id": source_id,
                "source": metadata.get(
                    "source",
                    "unknown",
                ),
                "page": metadata.get(
                    "page"
                ),
                "section": metadata.get(
                    "section"
                ),
                "score": result.score,
            }

            sources.append(source)

            context_parts.append(
                (
                    f"[{source_id}]\n"
                    f"Source: {source['source']}\n"
                    f"Page: {source['page']}\n"
                    f"Section: {source['section']}\n"
                    f"Content:\n"
                    f"{document.page_content}"
                )
            )

        context = "\n\n".join(
            context_parts
        )

        return context, sources

    async def answer(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any]
        | None = None,
    ) -> dict[str, Any]:

        # Retrieval
        results = await self.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # 没找到相关知识
        if not results:

            return {
                "answer": (
                    "知识库中没有找到足够相关的信息。"
                ),
                "sources": [],
            }

        # Context
        context, sources = (
            self.build_context(results)
        )

        # Prompt
        messages = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n"
                    f"{query}\n\n"
                    f"参考知识：\n"
                    f"{context}\n\n"
                    "请给出清晰、简洁、基于证据的回答。"
                ),
            },
        ]

        # Generation
        answer = await self.llm_service.chat(
            messages
        )

        return {
            "answer": answer,
            "sources": sources,
        }