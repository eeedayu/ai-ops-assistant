from app.rag.service import RAGService
from app.rag.models import (
    Document,
    ScoredDocument,
)


def test_build_context():

    results = [
        ScoredDocument(
            document=Document(
                page_content=(
                    "CPU温度超过85℃时"
                    "需要检查CPU使用率。"
                ),
                metadata={
                    "source": "cpu_sop.md",
                    "page": None,
                    "section": "CPU高温",
                },
            ),
            score=0.89,
        ),
        ScoredDocument(
            document=Document(
                page_content=(
                    "CPU使用率超过90%时，"
                    "需要进一步检查高CPU进程。"
                ),
                metadata={
                    "source": "server_manual.pdf",
                    "page": 12,
                },
            ),
            score=0.81,
        ),
    ]

    context, sources = (
        RAGService.build_context(
            results
        )
    )

    assert "[S1]" in context
    assert "[S2]" in context

    assert (
        sources[0]["source"]
        == "cpu_sop.md"
    )

    assert (
        sources[1]["page"]
        == 12
    )