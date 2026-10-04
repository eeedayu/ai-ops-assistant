from functools import lru_cache

from app.agents.agent import AgentService
from app.agents.memory import LongTermMemory
from app.agents.state_store import AgentStateStore
from app.core.config import get_settings
from app.agents.registry import (
    ToolRegistry,
    build_default_registry,
)
from app.repositories.session_repository import (
    SessionRepository,
)
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService
from app.rag.embeddings import (
    BGEEmbeddingService,
)
from app.rag.qdrant_store import (
    QdrantVectorStore,
)
from app.rag.service import RAGService
from app.rag.retrieval.reranker import (
    BGEReranker,
)
from app.rag.hybrid_qdrant_store import HybridQdrantStore


@lru_cache
def get_session_repository() -> SessionRepository:

    return SessionRepository()


@lru_cache
def get_llm_service() -> LLMService:

    return LLMService()


@lru_cache
def get_embedding_service():
    settings = get_settings()

    return BGEEmbeddingService(
        model_name=settings.embedding_model,
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


@lru_cache
def get_qdrant_store() -> HybridQdrantStore:

    settings = get_settings()

    store = HybridQdrantStore(
        url=settings.qdrant_url,
        collection_name=(
            settings.qdrant_collection
        ),
        dense_size=(
            settings.qdrant_vector_size
        ),
    )

    store.ensure_collection()

    return store


@lru_cache
def get_hybrid_qdrant_store() -> HybridQdrantStore:

    settings = get_settings()

    store = HybridQdrantStore(
        url=settings.qdrant_url,
        collection_name=(
            settings.qdrant_hybrid_collection
        ),
        dense_size=(
            settings.qdrant_vector_size
        ),
    )

    store.ensure_collection()

    return store


@lru_cache
def get_state_store() -> AgentStateStore:

    return AgentStateStore()


@lru_cache
def get_memory() -> LongTermMemory:

    return LongTermMemory()


@lru_cache
def get_reranker() -> BGEReranker:

    return BGEReranker()


@lru_cache
def get_rag_service() -> RAGService:

    return RAGService(
        embedding_service=(
            get_embedding_service()
        ),
        vector_store=(
            get_hybrid_qdrant_store()
        ),
        llm_service=(
            get_llm_service()
        ),
        reranker=(
            get_reranker()
        ),
    )


@lru_cache
def get_tool_registry() -> ToolRegistry:

    return build_default_registry(
        rag_service=get_rag_service()
    )


@lru_cache
def get_chat_service() -> ChatService:

    return ChatService(
        session_repository=(
            get_session_repository()
        ),
        llm_service=get_llm_service(),
    )


@lru_cache
def get_agent_service() -> AgentService:

    return AgentService(
        session_repository=(
            get_session_repository()
        ),
        llm_service=get_llm_service(),
        tool_registry=get_tool_registry(),
        memory=get_memory(),
        state_store=get_state_store(),
    )
