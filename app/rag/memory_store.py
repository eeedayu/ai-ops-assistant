from dataclasses import dataclass

import numpy as np

from app.rag.models import (
    Document,
    ScoredDocument,
)


@dataclass
class VectorRecord:
    document: Document
    vector: np.ndarray


class InMemoryVectorStore:

    def __init__(self):
        self.records: list[
            VectorRecord
        ] = []

    def add(
        self,
        documents: list[Document],
        vectors: np.ndarray,
    ) -> None:

        if len(documents) != len(vectors):
            raise ValueError(
                "documents 和 vectors 数量不一致"
            )

        for document, vector in zip(
            documents,
            vectors,
        ):
            self.records.append(
                VectorRecord(
                    document=document,
                    vector=vector,
                )
            )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
    ) -> list[ScoredDocument]:

        if not self.records:
            return []

        scored = []

        for record in self.records:

            score = float(
                np.dot(
                    query_vector,
                    record.vector,
                )
            )

            scored.append(
                ScoredDocument(
                    document=record.document,
                    score=score,
                )
            )

        scored.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return scored[:top_k]