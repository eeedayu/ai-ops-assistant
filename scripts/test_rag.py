from pathlib import Path

from app.rag.chunker import TextChunker
from app.rag.embeddings import (
    EmbeddingService,
)
from app.rag.memory_store import (
    InMemoryVectorStore,
)
from app.rag.models import Document
from app.rag.service import RAGService


def load_documents():
    documents = []

    knowledge_dir = Path(
        "knowledge_base"
    )

    for path in knowledge_dir.glob("*.md"):

        text = path.read_text(
            encoding="utf-8"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "type": "SOP",
                },
            )
        )

    return documents


def main():

    embedding_service = (
        EmbeddingService()
    )

    vector_store = (
        InMemoryVectorStore()
    )

    chunker = TextChunker(
        chunk_size=300,
        chunk_overlap=50,
    )

    rag = RAGService(
        embedding_service=(
            embedding_service
        ),
        vector_store=vector_store,
        chunker=chunker,
    )

    documents = load_documents()

    print(
        f"加载文档数量: {len(documents)}"
    )

    for document in documents:

        count = rag.add_document(
            document
        )

        print(
            f"{document.metadata['source']} "
            f"→ {count} chunks"
        )

    query = (
        "CPU温度超过90度应该怎么办？"
    )

    results = rag.search(
        query=query,
        top_k=3,
    )

    print()
    print(
        f"查询: {query}"
    )
    print()

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"===== Result {index} ====="
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Source: "
            f"{result.document.metadata.get('source')}"
        )

        print(
            result.document.page_content
        )

        print()


if __name__ == "__main__":
    main()