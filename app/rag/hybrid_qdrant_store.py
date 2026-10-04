from typing import Any

import numpy as np

from qdrant_client import (
    QdrantClient,
    models,
)

from app.rag.models import (
    Document,
    ScoredDocument,
)


class HybridQdrantStore:

    def __init__(
        self,
        url: str,
        collection_name: str,
        dense_size: int = 1024,
    ):
        self.client = QdrantClient(
            url=url
        )

        self.collection_name = (
            collection_name
        )

        self.dense_size = dense_size

    def ensure_collection(self):

        if self.client.collection_exists(
            self.collection_name
        ):
            return

        self.client.create_collection(
            collection_name=self.collection_name,

            vectors_config={
                "dense": models.VectorParams(
                    size=self.dense_size,
                    distance=models.Distance.COSINE,
                ),
            },

            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            },
        )

    @staticmethod
    def _point_id(
        document: Document,
    ) -> str:

        from uuid import uuid5, NAMESPACE_URL

        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
            0,
        )

        chunk_index = document.metadata.get(
            "chunk_index",
            0,
        )

        content = (
            document.page_content
        )

        raw = (
            f"{source}|"
            f"{page}|"
            f"{chunk_index}|"
            f"{content}"
        )

        return str(
            uuid5(
                NAMESPACE_URL,
                raw,
            )
        )

    def upsert(
        self,
        documents: list[Document],
        dense_vectors: np.ndarray,
        sparse_vectors: list[dict[str, Any]],
    ):

        self.ensure_collection()

        if not (
            len(documents)
            == len(dense_vectors)
            == len(sparse_vectors)
        ):
            raise ValueError(
                "documents / dense / sparse 数量不一致"
            )

        points = []

        for document, dense, sparse in zip(
            documents,
            dense_vectors,
            sparse_vectors,
        ):

            payload = {
                "text": document.page_content,
                **document.metadata,
            }

            point = models.PointStruct(
                id=self._point_id(
                    document
                ),

                vector={
                    "dense": dense.tolist(),

                    "sparse": models.SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    ),
                },

                payload=payload,
            )

            points.append(point)

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    @staticmethod
    def build_filter(
        metadata_filter: dict[str, Any] | None,
    ):

        if not metadata_filter:
            return None

        conditions = []

        for key, value in metadata_filter.items():

            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(
                        value=value
                    ),
                )
            )

        return models.Filter(
            must=conditions
        )
    
    def dense_search(
        self,
        dense_query: np.ndarray,
        top_k: int = 10,
        query_filter=None,
    ) -> list[ScoredDocument]:
        """纯稠密向量（语义）搜索"""
        
        # 直接在最外层执行单向量查询
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_query.tolist(),  # 将 numpy 数组转为普通列表
            using="dense",               # 明确指定使用 dense 向量配置
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return self._points_to_documents(result.points)

    def sparse_search(
        self,
        sparse_query: dict[str, Any],
        top_k: int = 10,
        query_filter=None,
    ) -> list[ScoredDocument]:
        """纯稀疏向量（关键词精准匹配）搜索"""
        
        # 将传入的稀疏字典转化为 Qdrant 的 SparseVector 对象
        sparse = models.SparseVector(
            indices=sparse_query["indices"],
            values=sparse_query["values"],
        )

        # 直接在最外层执行单向量查询
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse,                # 传入构建好的稀疏向量
            using="sparse",              # 明确指定使用 sparse 向量配置
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return self._points_to_documents(result.points)

    
    def hybrid_search(
        self,
        dense_query: np.ndarray,
        sparse_query: dict[str, Any],
        top_k: int = 10,
        prefetch_limit: int = 20,
        query_filter=None,
    ) -> list[ScoredDocument]:

        sparse = models.SparseVector(
            indices=sparse_query["indices"],
            values=sparse_query["values"],
        )

        result = self.client.query_points(
            collection_name=(
                self.collection_name
            ),

            prefetch=[
                models.Prefetch(
                    query=dense_query.tolist(),
                    using="dense",
                    limit=prefetch_limit,
                ),

                models.Prefetch(
                    query=sparse,
                    using="sparse",
                    limit=prefetch_limit,
                ),
            ],

            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),

            query_filter=query_filter,

            limit=top_k,

            with_payload=True,
        )

        return self._points_to_documents(
            result.points
        )

    @staticmethod
    def _points_to_documents(
        points,
    ):

        results = []

        for point in points:

            payload = dict(
                point.payload or {}
            )

            text = payload.pop(
                "text",
                "",
            )

            document = Document(
                page_content=text,
                metadata=payload,
            )

            results.append(
                ScoredDocument(
                    document=document,
                    score=float(
                        point.score
                    ),
                )
            )

        return results