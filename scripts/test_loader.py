from pathlib import Path

from app.rag.loaders import (
    DirectoryDocumentLoader,
    DocumentLoaderFactory,
)


def main():

    factory = (
        DocumentLoaderFactory()
    )

    loader = (
        DirectoryDocumentLoader(
            factory
        )
    )

    documents = (
        loader.load_directory(
            Path("knowledge_base")
        )
    )

    print(
        f"Document数量: "
        f"{len(documents)}"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print()
        print(
            f"===== Document {index} ====="
        )

        print(
            "Metadata:",
            document.metadata,
        )

        print(
            "Content:"
        )

        print(
            document.page_content[
                :300
            ]
        )


if __name__ == "__main__":
    main()