
import argparse

from app.rag.generator import generate_answer
from app.rag.retriever import retrieve_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the NexaSoft knowledge base.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show retrieved chunk text and metadata before the answer.",
    )
    args = parser.parse_args()

    question = input("Ask NexaSoft: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # Step 1: Retrieve relevant knowledge from Qdrant.
    chunks = retrieve_chunks(question, top_k=3)

    # Keep only chunks containing usable text.
    usable_chunks = [
        chunk
        for chunk in chunks
        if str(chunk.get("text", "")).strip()
    ]

    print(f"\nRetrieved chunks: {len(usable_chunks)}")

    if usable_chunks:
        print("\nSources retrieved:")
        for index, chunk in enumerate(usable_chunks, start=1):
            filename = chunk.get("filename", "Unknown")
            page = chunk.get("page")
            score = chunk.get("score")
            page_label = f", page {page}" if page is not None else ""
            print(
                f"[Source {index}] {filename}{page_label} "
                f"(similarity score: {score})"
            )
            if args.debug:
                print(f"Source path: {chunk.get('source', 'Unknown')}")
                print(f"Chunk index: {chunk.get('chunk_index', 'Unknown')}")
                print(f"Metadata: {chunk.get('metadata', {})}")
                print(f"Text: {chunk.get('text', '')}")

    # Step 2: Generate an answer using the retrieved context.
    answer = generate_answer(question, usable_chunks)

    print("\nNexaSoft Assistant:")
    print(answer)


if __name__ == "__main__":
    main()
