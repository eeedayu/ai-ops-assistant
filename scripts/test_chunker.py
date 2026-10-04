from app.rag.chunker import (
    RecursiveTextChunker,
)
from app.rag.models import Document


def main():

    text = """
CPU高温处理规范

当服务器CPU温度持续高于85℃时，
应首先检查CPU当前使用率。

如果CPU使用率持续超过90%，
应检查高CPU占用进程，
并确认是否存在异常计算任务。

如果CPU使用率正常，
但温度持续过高，
则应检查散热系统、风扇状态
以及机箱内部灰尘情况。

当CPU温度超过95℃时，
应提高故障等级。
""".strip()

    document = Document(
        page_content=text,
        metadata={
            "source": "cpu_sop.md",
            "type": "SOP",
        },
    )

    chunker = RecursiveTextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split_text(
        document.page_content
    )

    print(
        f"Chunk数量: {len(chunks)}"
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print()
        print(
            f"===== Chunk {index} ====="
        )

        print(chunk)

        print(
            f"长度: {len(chunk)}"
        )


if __name__ == "__main__":
    main()