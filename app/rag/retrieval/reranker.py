from typing import Any

from FlagEmbedding import FlagReranker


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ):
        self.model = FlagReranker(
            model_name,
            use_fp16=use_fp16,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pairs = []

        for item in candidates:
            text = item["text"]
            pairs.append([query, text])

        scores = self.model.compute_score(
            pairs,
            normalize=True,
        )

        if not isinstance(scores, list):
            scores = [scores]

        results = []

        for item, score in zip(candidates, scores):
            result = dict(item)
            result["rerank_score"] = float(score)
            results.append(result)

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return results[:top_k]