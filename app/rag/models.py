from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    page_content: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ScoredDocument:
    document: Document
    score: float
    

@dataclass
class Chunk:
    chunk_id: str

    page_content: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )