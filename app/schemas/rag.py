from typing import Any

from pydantic import BaseModel, Field
from typing import Literal


class RAGRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    filters: dict[str, Any] | None = None


class RAGSource(BaseModel):

    id: str

    source: str

    page: int | None = None

    section: str | None = None

    score: float


class RAGResponse(BaseModel):

    answer: str

    sources: list[RAGSource]


class RAGSearchRequest(BaseModel):

    question: str

    mode: Literal[
        "dense",
        "sparse",
        "hybrid",
    ] = "hybrid"

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    prefetch_limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )

class RAGSearchResponse(BaseModel):

    results: list[RAGSource]