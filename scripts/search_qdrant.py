from app.core.config import get_settings
from app.rag.embeddings import (
    BGEEmbeddingService,
)
from app.rag.qdrant_store import (
    QdrantVectorStore,
)


def main():

    settings = get_settings()

    embedding_service = (
        BGEEmbeddingService(
            model_name=(
                settings.embedding_model
            ),
            use_fp16=(
                settings.embedding_use_fp16
            ),
            batch_size=(
                settings.embedding_batch_size
            ),
            max_length=(
                settings.embedding_max_length
            ),
        )
    )

    vector_store = (
        QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=(
                settings.qdrant_collection
            ),
            vector_size=(
                settings.qdrant_vector_size
            ),
        )
    )

    query = input(
        "请输入查询问题: "
    ).strip()

    if not query:
        print("查询不能为空")
        return

    query_vector = (
        embedding_service
        .embed_query(query)
    )

    results = (
        vector_store.search(
            query_vector=query_vector,
            top_k=5,
        )
    )

    print()
    print(
        "========== Search Results =========="
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"----- Result {index} -----"
        )

        print(
            f"Score: "
            f"{result.score:.4f}"
        )

        print(
            "Metadata:"
        )

        for key, value in (
            result
            .document
            .metadata
            .items()
        ):

            print(
                f"  {key}: {value}"
            )

        print()
        print(
            result.document.page_content
        )


if __name__ == "__main__":
    main()