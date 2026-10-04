from abc import ABC, abstractmethod
from pathlib import Path
import re
from pypdf import PdfReader

from app.rag.models import Document


class BaseDocumentLoader(ABC):

    @abstractmethod
    def load(
        self,
        path: Path,
    ) -> list[Document]:
        """
        将文件解析为 Document 列表。
        """
        raise NotImplementedError



class TextDocumentLoader(
    BaseDocumentLoader
):

    def load(
        self,
        path: Path,
    ) -> list[Document]:

        text = path.read_text(
            encoding="utf-8"
        )

        return [
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                    "file_type": "txt",
                },
            )
        ]


class MarkdownDocumentLoader(
    BaseDocumentLoader
):

    def extract_markdown_sections(
            self,
            text: str,
        ) -> list[tuple[str, str]]:
    
            lines = text.splitlines()
    
            sections = []
    
            current_title = "root"
    
            current_lines = []
    
            for line in lines:
    
                match = re.match(
                    r"^(#{1,6})\s+(.+)$",
                    line.strip(),
                )
    
                if match:
    
                    if current_lines:
    
                        content = "\n".join(
                            current_lines
                        ).strip()
    
                        if content:
    
                            sections.append(
                                (
                                    current_title,
                                    content,
                                )
                            )
    
                    current_title = (
                        match.group(2).strip()
                    )
    
                    current_lines = []
    
                else:
    
                    current_lines.append(line)
    
            if current_lines:
    
                content = "\n".join(
                    current_lines
                ).strip()
    
                if content:
    
                    sections.append(
                        (
                            current_title,
                            content,
                        )
                    )
    
            return sections
    
    def load(
        self,
        path: Path,
    ) -> list[Document]:

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            return []

        sections = (
            self.extract_markdown_sections(
                text
            )
        )

        documents = []

        for section_title, content in sections:

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "source_path": str(path),
                        "file_type": "markdown",
                        "section": section_title,
                    },
                )
            )

        return documents


class PDFDocumentLoader(
    BaseDocumentLoader
):

    def load(
        self,
        path: Path,
    ) -> list[Document]:

        reader = PdfReader(
            str(path)
        )

        documents = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text() or ""

            text = text.strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "source_path": str(path),
                        "file_type": "pdf",
                        "page": page_number,
                    },
                )
            )

        return documents


class DocumentLoaderFactory:

    def __init__(self):

        self._loaders = {
            ".pdf": PDFDocumentLoader(),
            ".md": MarkdownDocumentLoader(),
            ".markdown": MarkdownDocumentLoader(),
            ".txt": TextDocumentLoader(),
        }

    def get_loader(
        self,
        path: Path,
    ) -> BaseDocumentLoader:

        suffix = path.suffix.lower()

        loader = self._loaders.get(
            suffix
        )

        if loader is None:
            raise ValueError(
                f"不支持的文件类型: {suffix}"
            )

        return loader


class DirectoryDocumentLoader:

    def __init__(
        self,
        factory: DocumentLoaderFactory,
    ):
        self.factory = factory

    def load_directory(
        self,
        directory: Path,
    ) -> list[Document]:

        if not directory.exists():
            raise FileNotFoundError(
                f"目录不存在: {directory}"
            )

        documents = []

        for path in sorted(
            directory.rglob("*")
        ):

            if not path.is_file():
                continue

            try:

                loader = (
                    self.factory
                    .get_loader(path)
                )

            except ValueError:
                continue

            loaded = loader.load(path)

            documents.extend(loaded)

        return documents

    