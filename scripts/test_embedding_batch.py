import time

from app.rag.embeddings import (
    BGEEmbeddingService,
)


def main():

    texts = [
        f"这是第{i}条测试技术文档"
        for i in range(100)
    ]

    embedding_service = (
        BGEEmbeddingService(
            batch_size=16,
            max_length=512,
            use_fp16=True,
        )
    )

    start = time.perf_counter()

    vectors = (
        embedding_service
        .embed_documents(texts)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"文本数量: {len(texts)}"
    )

    print(
        f"Vector shape: {vectors.shape}"
    )

    print(
        f"耗时: {elapsed:.3f}s"
    )

    print(
        f"平均: "
        f"{elapsed / len(texts):.4f}s / text"
    )


if __name__ == "__main__":
    main()