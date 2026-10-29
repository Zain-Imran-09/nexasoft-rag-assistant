
from app.rag.generator import generate_answer


def main() -> None:
    question = "What is NexaSoft's website development price?"

    # Controlled sample context for our first generator test.
    # We will connect the real retriever in a later step.
    retrieved_chunks = [
        {
            "text": (
                "Website development starts at PKR 50,000. "
                "The final price depends on project requirements."
            ),
            "filename": "pricing.md",
            "source": "pricing.md",
            "chunk_index": 0,
            "score": 0.84,
        }
    ]

    answer = generate_answer(question, retrieved_chunks)

    print("\nQuestion:")
    print(question)

    print("\nGenerated answer:")
    print(answer)


if __name__ == "__main__":
    main()
