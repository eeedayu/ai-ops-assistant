from app.core.config import get_settings
from app.rag.embeddings import (
    BGEEmbeddingService,
)
from app.rag.hybrid_qdrant_store import (
    HybridQdrantStore,
)


def show(
    name: str,
    results,
):

    print()
    print(
        f"========== {name} =========="
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        payload = (
            result.document.metadata
        )

        print(
            f"{rank:02d} | "
            f"score={result.score:.4f} | "
            f"{payload.get('source')}"
        )


def main():

    settings = get_settings()

    embedding = (
        BGEEmbeddingService(
            model_name=settings.embedding_model,
            use_fp16=settings.embedding_use_fp16,
            batch_size=1,
            max_length=settings.embedding_max_length,
        )
    )

    store = (
        HybridQdrantStore(
            url=settings.qdrant_url,
            collection_name=(
                settings.qdrant_hybrid_collection
            ),
            dense_size=1024,
        )
    )

    query = input(
        "Query: "
    ).strip()

    vectors = (
        embedding.embed_query_hybrid(
            query
        )
    )

    dense = store.dense_search(
        vectors["dense"],
        top_k=10,
    )

    sparse = store.sparse_search(
        vectors["sparse"],
        top_k=10,
    )

    hybrid = store.hybrid_search(
        dense_query=vectors["dense"],
        sparse_query=vectors["sparse"],
        top_k=10,
        prefetch_limit=20,
    )

    show(
        "DENSE",
        dense,
    )

    show(
        "SPARSE",
        sparse,
    )

    show(
        "HYBRID + RRF",
        hybrid,
    )


if __name__ == "__main__":
    main()