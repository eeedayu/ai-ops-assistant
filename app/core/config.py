from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Ops Assistant"
    app_version: str = "0.1.0"
    environment: str = "development"

    agent_max_steps: int = 6
    tool_timeout: float = 10.0

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_reasoning_effort: Literal[
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ] = "none"

    llm_timeout: float = 30.0

    max_context_messages: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16
    embedding_max_length: int = 8192
    embedding_use_fp16: bool = True

    qdrant_url: str = (
        "http://eeedayu.dpdns.org:6333"
    )
    qdrant_collection: str = (
        "ops_knowledge"
    )
    qdrant_hybrid_collection: str = (
        "ops_knowledge_hybrid"
    )
    qdrant_vector_size: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()