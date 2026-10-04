from typing import Any

import numpy as np
from FlagEmbedding import BGEM3FlagModel


class BGEEmbeddingService:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 16,
        max_length: int = 2048,
    ):
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
        )

    @staticmethod
    def _normalize_dense(
        vectors: np.ndarray,
    ) -> np.ndarray:

        if vectors.ndim == 1:

            norm = np.linalg.norm(
                vectors
            )

            if norm == 0:
                return vectors

            return vectors / norm

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(
            norms,
            1e-12,
        )

        return vectors / norms

    @staticmethod
    def _normalize_sparse(
        sparse_weights: list[dict[Any, float]],
    ) -> list[dict[str, Any]]:

        results = []

        for weights in sparse_weights:

            indices = []
            values = []

            for token_id, weight in (
                weights.items()
            ):

                weight = float(weight)

                if weight == 0:
                    continue

                indices.append(
                    int(token_id)
                )

                values.append(
                    weight
                )

            results.append(
                {
                    "indices": indices,
                    "values": values,
                }
            )

        return results

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:

        if not texts:

            return np.empty(
                (0, 1024),
                dtype=np.float32,
            )

        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        vectors = np.asarray(
            output["dense_vecs"],
            dtype=np.float32,
        )

        return self._normalize_dense(
            vectors
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:

        output = self.model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        vector = np.asarray(
            output["dense_vecs"][0],
            dtype=np.float32,
        )

        return self._normalize_dense(
            vector
        )

    def embed_documents_hybrid(
        self,
        texts: list[str],
    ) -> dict[str, Any]:

        if not texts:

            return {
                "dense": np.empty(
                    (0, 1024),
                    dtype=np.float32,
                ),
                "sparse": [],
            }

        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vectors = np.asarray(
            output["dense_vecs"],
            dtype=np.float32,
        )

        dense_vectors = (
            self._normalize_dense(
                dense_vectors
            )
        )

        sparse_vectors = (
            self._normalize_sparse(
                output["lexical_weights"]
            )
        )

        return {
            "dense": dense_vectors,
            "sparse": sparse_vectors,
        }

    def embed_query_hybrid(
        self,
        query: str,
    ) -> dict[str, Any]:

        output = self.model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vector = np.asarray(
            output["dense_vecs"][0],
            dtype=np.float32,
        )

        dense_vector = (
            self._normalize_dense(
                dense_vector
            )
        )
 
        sparse_vector = (
            self._normalize_sparse(
                output["lexical_weights"]
            )[0]
        )

        return {
            "dense": dense_vector,
            "sparse": sparse_vector,
        }