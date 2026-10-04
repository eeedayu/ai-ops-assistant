from typing import Any
from uuid import uuid5, NAMESPACE_URL

import numpy as np

from qdrant_client import (
    QdrantClient,
    models,
)

from app.rag.models import (
    Document,
)


class QdrantVectorStore:

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

        raw = (
            f"{source}|"
            f"{page}|"
            f"{chunk_index}|"
            f"{document.page_content}"
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

            points.append(
                models.PointStruct(
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
            )

        if points:
            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=points,
                wait=True,
            )

    def dense_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
    ):

        result = self.client.query_points(
            collection_name=(
                self.collection_name
            ),
            query=query_vector.tolist(),
            using="dense",
            limit=top_k,
            with_payload=True,
        )

        return result.points

    def sparse_search(
        self,
        query_vector: dict[str, Any],
        top_k: int = 20,
    ):

        sparse = models.SparseVector(
            indices=query_vector["indices"],
            values=query_vector["values"],
        )

        result = self.client.query_points(
            collection_name=(
                self.collection_name
            ),
            query=sparse,
            using="sparse",
            limit=top_k,
            with_payload=True,
        )

        return result.points

    def hybrid_candidates(
        self,
        dense_query: np.ndarray,
        sparse_query: dict[str, Any],
        top_k: int = 20,
    ):

        return {
            "dense": self.dense_search(
                dense_query,
                top_k,
            ),
            "sparse": self.sparse_search(
                sparse_query,
                top_k,
            ),
        }