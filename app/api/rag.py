from fastapi import APIRouter, Depends

from app.dependencies import get_rag_service
from app.rag.service import RAGService
from app.schemas.rag import (
    RAGRequest,
    RAGResponse,
    RAGSource,
    RAGSearchRequest,
    RAGSearchResponse,
)


router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)


@router.post(
    "/chat",
    response_model=RAGResponse,
)
async def rag_chat(
    request: RAGRequest,
    rag_service: RAGService = Depends(
        get_rag_service
    ),
):

    result = await rag_service.answer(
        query=request.question,
        top_k=request.top_k,
        metadata_filter=request.filters,
    )

    return RAGResponse(
        answer=result["answer"],
        sources=[
            RAGSource(**source)
            for source in result["sources"]
        ],
    )


@router.post(
    "/search",
    response_model=RAGSearchResponse,
)
async def rag_search(
    request: RAGSearchRequest,
    rag_service: RAGService = Depends(
        get_rag_service
    ),
):

    results = rag_service.search(
        query=request.question,
        top_k=request.top_k,
    )

    sources = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = (
            result.document.metadata
        )

        sources.append(
            RAGSource(
                id=f"S{index}",
                source=metadata.get(
                    "source",
                    "unknown",
                ),
                page=metadata.get(
                    "page"
                ),
                section=metadata.get(
                    "section"
                ),
                score=result.score,
            )
        )

    return RAGSearchResponse(
        results=sources
    )