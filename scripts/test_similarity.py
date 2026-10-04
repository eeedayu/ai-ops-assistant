import numpy as np

from app.rag.embeddings import (
    BGEEmbeddingService,
)


def main():

    embedding_service = (
        BGEEmbeddingService(
            use_fp16=True,
            batch_size=4,
        )
    )

    texts = [
        "CPU温度过高应该如何处理？",
        "处理器高温需要检查CPU负载和散热系统。",
        "Nginx出现请求超时应该怎么排查？",
        "磁盘空间不足应该如何处理？",
    ]

    vectors = (
        embedding_service
        .embed_documents(texts)
    )

    query = (
        "服务器CPU温度超过90度怎么办？"
    )

    query_vector = (
        embedding_service
        .embed_query(query)
    )

    scores = vectors @ query_vector

    ranking = np.argsort(
        scores
    )[::-1]

    print(
        f"Query: {query}"
    )

    print()

    for index in ranking:

        print(
            f"{scores[index]:.4f}"
            f"  "
            f"{texts[index]}"
        )


if __name__ == "__main__":
    main()