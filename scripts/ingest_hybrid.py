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
from app.rag.hybrid_qdrant_store import (
    HybridQdrantStore,
)


def main():

    settings = get_settings()

    knowledge_dir = Path(
        "knowledge_base"
    )

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

    chunker = DocumentChunker(
        RecursiveTextChunker(
            chunk_size=500,
            chunk_overlap=80,
        )
    )

    embedding_service = (
        BGEEmbeddingService(
            model_name=settings.embedding_model,
            use_fp16=settings.embedding_use_fp16,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
        )
    )

    vector_store = (
        HybridQdrantStore(
            url=settings.qdrant_url,
            collection_name=(
                settings.qdrant_hybrid_collection
            ),
            dense_size=1024,
        )
    )

    vector_store.ensure_collection()

    total_chunks = 0

    for document in documents:

        chunks = chunker.split_document(
            document
        )

        if not chunks:
            continue

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = (
            embedding_service
            .embed_documents_hybrid(
                texts
            )
        )

        vector_store.upsert(
            documents=chunks,
            dense_vectors=(
                embeddings["dense"]
            ),
            sparse_vectors=(
                embeddings["sparse"]
            ),
        )

        total_chunks += len(chunks)

        print(
            f"{document.metadata.get('source')} "
            f"→ {len(chunks)} chunks"
        )

    print()
    print(
        "========== Hybrid Ingestion Complete =========="
    )

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Chunks: {total_chunks}"
    )

    print(
        f"Collection: "
        f"{settings.qdrant_hybrid_collection}"
    )


if __name__ == "__main__":
    main()