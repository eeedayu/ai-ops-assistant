from qdrant_client import QdrantClient, models


COLLECTION_NAME = "test_collection"


def main():

    client = QdrantClient(
        url="http://localhost:6333"
    )

    # 如果之前存在，先删除
    if client.collection_exists(
        COLLECTION_NAME
    ):
        client.delete_collection(
            COLLECTION_NAME
        )

    # 创建 Collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=4,
            distance=models.Distance.COSINE,
        ),
    )

    # 插入三个向量
    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[
            models.PointStruct(
                id=1,
                vector=[
                    0.9,
                    0.1,
                    0.1,
                    0.1,
                ],
                payload={
                    "text": "CPU高温",
                    "type": "SOP",
                },
            ),
            models.PointStruct(
                id=2,
                vector=[
                    0.1,
                    0.9,
                    0.1,
                    0.1,
                ],
                payload={
                    "text": "Nginx超时",
                    "type": "SOP",
                },
            ),
            models.PointStruct(
                id=3,
                vector=[
                    0.1,
                    0.1,
                    0.9,
                    0.1,
                ],
                payload={
                    "text": "磁盘空间不足",
                    "type": "SOP",
                },
            ),
        ],
    )

    # 查询
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=[
            0.85,
            0.10,
            0.10,
            0.10,
        ],
        limit=2,
        with_payload=True,
    )

    print(
        "Search Results:"
    )

    for point in result.points:

        print(
            f"id={point.id}, "
            f"score={point.score:.4f}, "
            f"payload={point.payload}"
        )


if __name__ == "__main__":
    main()