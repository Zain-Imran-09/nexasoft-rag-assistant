import pytest
from langchain_core.documents import Document

from app.rag.chunker import chunk_documents


def test_chunk_size_limit():
    """Verify chunks never exceed the configured size."""

    document = Document(
        page_content="NexaSoft provides AI development services. " * 50,
        metadata={"filename": "services.md"}
    )

    chunks = chunk_documents(
        [document],
        chunk_size=200,
        chunk_overlap=40
    )

    assert len(chunks) > 1

    assert all(
        0 < len(chunk.page_content) <= 200
        for chunk in chunks
    )


def test_metadata_preservation():
    """Verify source information survives chunking."""

    document = Document(
        page_content="NexaSoft company policy. " * 50,
        metadata={
            "source": "data/raw/policy.md",
            "filename": "policy.md",
            "file_type": ".md"
        }
    )

    chunks = chunk_documents([document], 200, 40)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata == document.metadata


def test_multiple_documents():
    """Verify chunking handles multiple source documents."""

    documents = [
        Document(
            page_content="NexaSoft company overview.",
            metadata={"filename": "overview.md"}
        ),
        Document(
            page_content="NexaSoft pricing information.",
            metadata={"filename": "pricing.md"}
        )
    ]

    chunks = chunk_documents(documents, 100, 20)

    filenames = {
        chunk.metadata["filename"]
        for chunk in chunks
    }

    assert filenames == {"overview.md", "pricing.md"}


def test_empty_documents():
    """An empty input list should produce no chunks."""

    assert chunk_documents([]) == []


@pytest.mark.parametrize(
    "size, overlap",
    [
        (0, 0),
        (-100, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ]
)
def test_invalid_settings(size, overlap):
    """Invalid configurations should raise ValueError."""

    with pytest.raises(ValueError):
        chunk_documents(
            [Document(page_content="Example text")],
            chunk_size=size,
            chunk_overlap=overlap
        )

