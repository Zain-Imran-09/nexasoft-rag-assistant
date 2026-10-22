from math import isfinite
from pathlib import Path

from app.rag.loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embeddings import get_embedding_service


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_embeddings() -> None:
    # Load all company documents; stop if any file fails.
    documents = load_documents(DATA_DIR, skip_errors=False)

    # Use the baseline selected on Day 5.
    chunks = chunk_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks available for embedding.")

    service = get_embedding_service()
    token_limit = service.model.max_seq_length

    print(f"Model: {service.config.model_name}")
    print(f"Model token limit: {token_limit}")

    # Check every chunk before generating embeddings.
    oversized_chunks = []

    for index, chunk in enumerate(chunks, start=1):
        token_ids = service.model.tokenizer(
            chunk.page_content,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        token_count = len(token_ids)

        print(
            f"Chunk {index:02d} | "
            f"File: {chunk.metadata.get('filename', 'Unknown')} | "
            f"Characters: {len(chunk.page_content)} | "
            f"Tokens: {token_count}"
        )

        if token_count > token_limit:
            oversized_chunks.append(index)

    # Stop rather than silently losing text through truncation.
    if oversized_chunks:
        raise ValueError(
            f"Chunks {oversized_chunks} exceed the "
            f"{token_limit}-token limit. "
            "Adjust chunking before embedding."
        )

    # The embedding service expects strings, not Document objects.
    texts = [chunk.page_content for chunk in chunks]
    vectors = service.embed_documents(texts)

    expected_dimensions = service.model.get_embedding_dimension()

    # Verify one valid vector was generated for each chunk.
    if len(vectors) != len(chunks):
        raise ValueError("Vector count does not match chunk count.")

    for index, vector in enumerate(vectors, start=1):
        if len(vector) != expected_dimensions:
            raise ValueError(
                f"Vector {index} has an unexpected dimension."
            )

        if not all(isfinite(value) for value in vector):
            raise ValueError(
                f"Vector {index} contains invalid numerical values."
            )

        # Normalized vectors should have a length close to 1.
        norm = sum(value * value for value in vector) ** 0.5

        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"Vector {index} is not normalized.")

    # Keep each chunk's text and metadata paired with its vector.
    records = [
        {
            "text": chunk.page_content,
            "metadata": dict(chunk.metadata),
            "vector": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    print(f"\nGenerated vectors: {len(vectors)}")
    print(f"Dimensions per vector: {expected_dimensions}")
    print(f"First record source: {records[0]['metadata'].get('source')}")
    print(f"First vector preview: {records[0]['vector'][:5]}")
    print(
        "\nSUCCESS: Every chunk has a finite, normalized vector "
        "with the expected dimensions."
    )


if __name__ == "__main__":
    inspect_embeddings()