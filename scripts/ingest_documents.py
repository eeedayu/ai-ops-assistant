from pathlib import Path

from app.core.config import get_settings
from app.rag.chunker import (
    DocumentChunker,
    RecursiveTextChunker,
)
from app.rag.embeddings import (
    BGEEmbeddingService,
)
from app.rag.loaders import (
    DirectoryDocumentLoader,
    DocumentLoaderFactory,
)
from app.rag.qdrant_store import (
    QdrantVectorStore,
)


def main():

    settings = get_settings()

    knowledge_dir = Path(
        "knowledge_base"
    )

    print(
        "========== Knowledge Ingestion =========="
    )

    print(
        f"Knowledge directory: "
        f"{knowledge_dir}"
    )

    # -----------------------------
    # 1. Document Loader
    # -----------------------------

    loader = (
        DirectoryDocumentLoader(
            DocumentLoaderFactory()
        )
    )

    documents = (
        loader.load_directory(
            knowledge_dir
        )
    )

    print(
        f"Documents: {len(documents)}"
    )

    # -----------------------------
    # 2. Chunker
    # -----------------------------

    text_chunker = (
        RecursiveTextChunker(
            chunk_size=500,
            chunk_overlap=80,
        )
    )

    document_chunker = (
        DocumentChunker(
            text_chunker
        )
    )

    # -----------------------------
    # 3. Embedding
    # -----------------------------

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

    # -----------------------------
    # 4. Qdrant
    # -----------------------------

    z

    vector_store.ensure_collection()

    # -----------------------------
    # 5. Ingestion
    # -----------------------------

    total_chunks = 0

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "unknown",
        )

        chunks = (
            document_chunker
            .split_document(
                document
            )
        )

        if not chunks:
            continue

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        vectors = (
            embedding_service
            .embed_documents(
                texts
            )
        )

        vector_store.add(
            documents=chunks,
            vectors=vectors,
        )

        total_chunks += len(chunks)

        print(
            f"[{index}/{len(documents)}] "
            f"{source} "
            f"→ {len(chunks)} chunks"
        )

    print()
    print(
        "========== Completed =========="
    )

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Chunks: {total_chunks}"
    )

    print(
        f"Collection: "
        f"{settings.qdrant_collection}"
    )


if __name__ == "__main__":
    main()