from FlagEmbedding import BGEM3FlagModel


def main():

    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
    )

    texts = [
        "CPU温度超过90℃需要检查CPU使用率。",
        "Nginx出现HTTP 504超时。",
    ]

    output = model.encode(
        texts,
        batch_size=2,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense = output[
        "dense_vecs"
    ]

    sparse = output[
        "lexical_weights"
    ]

    print(
        "Dense shape:",
        dense.shape,
    )

    print(
        "Sparse:"
    )

    for index, weights in enumerate(
        sparse
    ):

        print(
            f"Document {index}:"
        )

        print(
            weights
        )


if __name__ == "__main__":
    main()