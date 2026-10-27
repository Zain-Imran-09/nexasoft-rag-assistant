
from numpy import number
from sqlalchemy import text

from numpy import number

from app.rag.retriever import retrieve_chunks


TEST_QUESTIONS = [
    "How much does website development cost?",
    "What is the starting price for a custom RAG assistant?",
    "How are project payments divided?",
    "When is the company open?",
    "Do students get a discount?",
]

def test_invalid_questions() -> None:
    test_cases = [
        ("", 3),
        ("   ", 3),
        ("A valid question", 0),
        ("A valid question", -1),
    ]

    for question, top_k in test_cases:
        try:
            retrieve_chunks(question, top_k=top_k)
        except ValueError as exc:
            print(f"PASS: question={question!r}, top_k={top_k}: {exc}")
        else:
            raise AssertionError(
                f"Expected ValueError for question={question!r}, "
                f"top_k={top_k}"
            )

def main() -> None:
    for number, question in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "=" * 70)
        print(f"TEST {number}: {question}")
        print("=" * 70)

        results = retrieve_chunks(question, top_k=3)
        print(f"Results found: {len(results)}")

        if not results:
            print("No results returned.")
            continue

        for rank, result in enumerate(results, start=1):
            text = " ".join(result["text"].split())
            if number in (3, 5):
                excerpt = text
            else:
                excerpt = text[:250]
                if len(text) > 250:
                    excerpt += "..."
            print(
                f"\nRank: {rank} | "
                f"Score: {result['score']:.4f}"
            )
            print(f"Source: {result['filename']}")
            print(f"Text: {excerpt}")
            print("\n--- Input validation tests ---")
            test_invalid_questions() 


if __name__ == "__main__":
    main()

