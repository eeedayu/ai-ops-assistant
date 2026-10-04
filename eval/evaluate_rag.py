import json
from pathlib import Path

from app.core.config import get_settings
from app.rag.embeddings import BGEEmbeddingService
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.service import RAGService


def load_dataset(
    path: str = "eval/rag_eval.json",
):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def recall_at_k(
    ranked_sources: list[str],
    relevant_sources: set[str],
    k: int,
) -> float:

    top_k = ranked_sources[:k]

    return float(
        bool(
            set(top_k)
            & relevant_sources
        )
    )


def precision_at_k(
    ranked_sources: list[str],
    relevant_sources: set[str],
    k: int,
) -> float:

    top_k = ranked_sources[:k]

    if not top_k:
        return 0.0

    hits = len(
        set(top_k)
        & relevant_sources
    )

    return hits / len(top_k)


def reciprocal_rank(
    ranked_sources: list[str],
    relevant_sources: set[str],
) -> float:

    for rank, source in enumerate(
        ranked_sources,
        start=1,
    ):

        if source in relevant_sources:

            return 1.0 / rank

    return 0.0


def dcg_at_k(
    ranked_sources: list[str],
    relevant_sources: set[str],
    k: int,
) -> float:

    score = 0.0

    for rank, source in enumerate(
        ranked_sources[:k],
        start=1,
    ):

        relevance = (
            1.0
            if source in relevant_sources
            else 0.0
        )

        if relevance > 0:

            import math

            score += (
                relevance
                / math.log2(rank + 1)
            )

    return score


def ndcg_at_k(
    ranked_sources: list[str],
    relevant_sources: set[str],
    k: int,
) -> float:

    actual = dcg_at_k(
        ranked_sources,
        relevant_sources,
        k,
    )

    ideal_sources = list(
        relevant_sources
    )

    ideal = dcg_at_k(
        ideal_sources,
        relevant_sources,
        min(
            k,
            len(ideal_sources),
        ),
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def main():

    settings = get_settings()

    dataset = load_dataset()

    embedding_service = (
        BGEEmbeddingService(
            model_name=settings.embedding_model,
            use_fp16=settings.embedding_use_fp16,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
        )
    )

    vector_store = (
        QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            vector_size=settings.qdrant_vector_size,
        )
    )

    # 这里只用 search，不需要真正的 LLM
    rag = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=None,
    )

    recall_5_scores = []
    precision_5_scores = []
    mrr_scores = []
    ndcg_5_scores = []

    print(
        "========== RAG Retrieval Evaluation =========="
    )

    for item in dataset:

        query = item["query"]

        relevant_sources = set(
            item["relevant_sources"]
        )

        results = rag.search(
            query=query,
            top_k=5,
        )

        ranked_sources = [
            result.document.metadata.get(
                "source",
                "unknown",
            )
            for result in results
        ]

        recall = recall_at_k(
            ranked_sources,
            relevant_sources,
            5,
        )

        precision = precision_at_k(
            ranked_sources,
            relevant_sources,
            5,
        )

        mrr = reciprocal_rank(
            ranked_sources,
            relevant_sources,
        )

        ndcg = ndcg_at_k(
            ranked_sources,
            relevant_sources,
            5,
        )

        recall_5_scores.append(
            recall
        )

        precision_5_scores.append(
            precision
        )

        mrr_scores.append(
            mrr
        )

        ndcg_5_scores.append(
            ndcg
        )

        print()

        print(
            f"[{item['id']}] "
            f"{query}"
        )

        print(
            f"  Expected: "
            f"{list(relevant_sources)}"
        )

        print(
            f"  Retrieved: "
            f"{ranked_sources}"
        )

        print(
            f"  Recall@5: "
            f"{recall:.2f}"
        )

        print(
            f"  Precision@5: "
            f"{precision:.2f}"
        )

        print(
            f"  MRR: "
            f"{mrr:.2f}"
        )

        print(
            f"  nDCG@5: "
            f"{ndcg:.2f}"
        )

    count = len(dataset)

    if count == 0:
        print(
            "评测集为空"
        )
        return

    avg_recall = (
        sum(recall_5_scores)
        / count
    )

    avg_precision = (
        sum(precision_5_scores)
        / count
    )

    avg_mrr = (
        sum(mrr_scores)
        / count
    )

    avg_ndcg = (
        sum(ndcg_5_scores)
        / count
    )

    print()
    print(
        "========== Final Metrics =========="
    )

    print(
        f"Recall@5: "
        f"{avg_recall:.2%}"
    )

    print(
        f"Precision@5: "
        f"{avg_precision:.2%}"
    )

    print(
        f"MRR: "
        f"{avg_mrr:.4f}"
    )

    print(
        f"nDCG@5: "
        f"{avg_ndcg:.4f}"
    )


if __name__ == "__main__":
    main()