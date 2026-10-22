from app.rag.embeddings import get_embedding_service


def main() -> None:
    service = get_embedding_service()

    query = "How much does website development cost?"

    texts = [
        "What is the price of building a website?",
        "NexaSoft develops websites for businesses.",
        "Our office opens at nine in the morning.",
    ]

    query_vector = service.embed_query(query)
    document_vectors = service.embed_documents(texts)

    print(f"Query: {query}\n")

    results = []

    for text, vector in zip(texts, document_vectors):
        # Our vectors are normalized, so their dot product
        # equals cosine similarity.
        score = sum(
            query_value * document_value
            for query_value, document_value in zip(query_vector, vector)
        )

        results.append((score, text))

    # Display the most similar text first.
    for score, text in sorted(results, key=lambda item: item[0], reverse=True):
        print(f"{score:.4f} | {text}")


if __name__ == "__main__":
    main()