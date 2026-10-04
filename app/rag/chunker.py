from uuid import uuid4

from app.rag.models import Document


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap 必须小于 chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        document: Document,
    ) -> list[Document]:

        text = document.page_content

        if not text:
            return []

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk_text = text[start:end]

            metadata = {
                **document.metadata,
                "chunk_start": start,
                "chunk_end": min(
                    end,
                    len(text),
                ),
            }

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

            if end >= len(text):
                break

            start = (
                end
                - self.chunk_overlap
            )

        return chunks


class RecursiveTextChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ):

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap 必须小于 chunk_size"
            )

        self.chunk_size = chunk_size

        self.chunk_overlap = (
            chunk_overlap
        )

        self.separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            " ",
            "",
        ]

    def split_text(
        self,
        text: str,
    ) -> list[str]:

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        return self._split_recursive(
            text,
            self.separators,
        )

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._hard_split(text)

        separator = separators[0]

        if separator == "":
            return self._hard_split(text)

        parts = text.split(separator)

        chunks = []

        current = ""

        for part in parts:

            piece = part.strip()

            if not piece:
                continue

            candidate = (
                piece
                if not current
                else current
                + separator
                + piece
            )

            if (
                len(candidate)
                <= self.chunk_size
            ):

                current = candidate

            else:

                if current:
                    chunks.append(
                        current
                    )

                if (
                    len(piece)
                    <= self.chunk_size
                ):

                    current = piece

                else:

                    sub_chunks = (
                        self._split_recursive(
                            piece,
                            separators[1:],
                        )
                    )

                    chunks.extend(
                        sub_chunks
                    )

                    current = ""

        if current:
            chunks.append(current)

        return self._add_overlap(
            chunks
        )

    def _hard_split(
        self,
        text: str,
    ) -> list[str]:

        chunks = []

        start = 0

        while start < len(text):

            end = (
                start
                + self.chunk_size
            )

            chunks.append(
                text[start:end]
            )

            start = (
                end
                - self.chunk_overlap
            )

        return chunks

    def _add_overlap(
        self,
        chunks: list[str],
    ) -> list[str]:

        if len(chunks) <= 1:
            return chunks

        results = []

        for index, chunk in enumerate(
            chunks
        ):

            if index == 0:

                results.append(chunk)

                continue

            previous = chunks[
                index - 1
            ]

            overlap = previous[
                -self.chunk_overlap:
            ]

            merged = (
                overlap
                + "\n"
                + chunk
            )

            if len(merged) > (
                self.chunk_size
            ):

                merged = chunk

            results.append(
                merged
            )

        return results


class DocumentChunker:

    def __init__(
        self,
        text_chunker: RecursiveTextChunker,
    ):
        self.text_chunker = text_chunker

    def split_document(
        self,
        document: Document,
    ) -> list[Document]:

        texts = (
            self.text_chunker
            .split_text(
                document.page_content
            )
        )

        chunks = []

        for index, text in enumerate(
            texts
        ):

            metadata = {
                **document.metadata,
                "chunk_index": index,
            }

            chunks.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return chunks