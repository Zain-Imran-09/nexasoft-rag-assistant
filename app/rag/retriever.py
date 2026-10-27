import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from app.rag.embeddings import get_embedding_service

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def get_qdrant_client() -> QdrantClient:
    """Create a client for connecting to Qdrant Cloud."""

    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()

    if not url or not api_key:
        raise ValueError("Missing Qdrant settings in .env.")

    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=30,
    )

def retrieve_chunks(question: str, top_k: int = 3) -> list[dict]:
    """Find the most relevant knowledge-base chunks for a question."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    collection = os.getenv("QDRANT_COLLECTION", "").strip()
    if not collection:
        raise ValueError("Missing QDRANT_COLLECTION in .env.")

    # Convert the user's question into a vector.
    embedding_service = get_embedding_service()
    query_vector = embedding_service.embed_query(question.strip())

    # Search Qdrant for the closest vectors.
    client = get_qdrant_client()
    try:
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}

            results.append(
                {
                    "text": payload.get("text", ""),
                    "score": point.score,
                    "source": metadata.get("source", "Unknown"),
                    "filename": metadata.get("filename", "Unknown"),
                    "chunk_index": payload.get("chunk_index"),
                    "metadata": metadata,
                }
            )

        return results
    finally:
        client.close()
