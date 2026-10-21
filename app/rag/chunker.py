
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split LangChain documents into smaller chunks.

    Args:
        documents: List of documents from the loader.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Target overlap between chunks.

    Returns:
        List of smaller Document objects with metadata.
    """

    # Validate chunking settings
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and < chunk_size"
        )

    # Configure the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # Split while preserving document metadata
    chunks = splitter.split_documents(documents)

    return chunks
