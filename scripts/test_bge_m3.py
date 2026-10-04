import numpy as np

from FlagEmbedding import BGEM3FlagModel


def main():

    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
    )

    sentences = [
        "CPU温度过高应该如何处理？",
        "服务器处理器高温需要检查CPU负载和散热系统。",
        "Nginx出现请求超时应该如何排查？",
    ]

    output = model.encode(
        sentences,
        batch_size=3,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    dense_vectors = output[
        "dense_vecs"
    ]

    print(
        "Vector shape:",
        dense_vectors.shape,
    )

    print(
        "Vector dtype:",
        dense_vectors.dtype,
    )

    print(
        "First vector first 10 values:"
    )

    print(
        dense_vectors[0][:10]
    )

    print(
        "Vector norm:"
    )

    print(
        np.linalg.norm(
            dense_vectors[0]
        )
    )


if __name__ == "__main__":
    main()