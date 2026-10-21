
from pathlib import Path

from app.rag.loader import load_documents
from app.rag.chunker import chunk_documents


# Resolve project root from the scripts directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing NexaSoft knowledge-base files
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_chunks():
    # Step 1: Load original documents
    documents = load_documents(DATA_DIR, skip_errors=False)

    print(f"Loaded documents: {len(documents)}")

    # Step 2: Compare different chunk configurations
    configurations = [
        (500, 100),
        (1000, 200),
        (1500, 300),
    ]

    for chunk_size, chunk_overlap in configurations:
        print("\n" + "=" * 60)
        print(
            f"Chunk size: {chunk_size} | "
            f"Overlap: {chunk_overlap}"
        )

        # Step 3: Split loaded documents
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        print(f"Total chunks: {len(chunks)}")

        if not chunks:
            print("No chunks generated.")
            continue

        # Step 4: Calculate chunk statistics
        lengths = [
            len(chunk.page_content)
            for chunk in chunks
        ]
        
        # Find unusually short chunks
        short_chunks = [
            chunk for chunk in chunks
            if len(chunk.page_content) < 100
        ]

        print(f"\nShort chunks (<100 chars): {len(short_chunks)}")

        for index, chunk in enumerate(short_chunks, start=1):
            print(f"\nShort Chunk {index}")
            print(f"Source: {chunk.metadata.get('filename')}")
            print(f"Length: {len(chunk.page_content)}")
            print(f"Content: {chunk.page_content!r}")


        print(f"Smallest chunk: {min(lengths)} characters")
        print(f"Largest chunk: {max(lengths)} characters")
        print(
            f"Average chunk: "
            f"{sum(lengths) / len(lengths):.1f} characters"
        )

        # Step 5: Inspect the first three chunks
        for index, chunk in enumerate(chunks[:3], start=1):
            print(f"\n--- Chunk {index} ---")
            print(f"Source: {chunk.metadata.get('source')}")
            print(f"Filename: {chunk.metadata.get('filename')}")
            print(f"Page: {chunk.metadata.get('page', 'N/A')}")
            print(f"Length: {len(chunk.page_content)}")
            print("Content:")
            print(chunk.page_content)


if __name__ == "__main__":
    inspect_chunks()
