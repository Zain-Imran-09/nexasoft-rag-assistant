
from app.rag.generator import generate_answer
from app.rag.retriever import retrieve_chunks


def main() -> None:
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
            score = chunk.get("score")
            print(
                f"[Source {index}] {filename} "
                f"(similarity score: {score})"
            )

    # Step 2: Generate an answer using the retrieved context.
    answer = generate_answer(question, usable_chunks)

    print("\nNexaSoft Assistant:")
    print(answer)


if __name__ == "__main__":
    main()
