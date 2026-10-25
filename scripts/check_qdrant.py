import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load settings from the project's .env file.
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    collection_name = os.getenv("QDRANT_COLLECTION", "").strip()

    # Catch missing settings before making a network request.
    if not all([url, api_key, collection_name]):
        raise ValueError(
            "Set QDRANT_URL, QDRANT_API_KEY, and QDRANT_COLLECTION in .env."
        )

    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=30,
    )

    try:
        if not client.collection_exists(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' was not found. "
                "Check its name and whether you created it in this cluster."
            )

        info = client.get_collection(collection_name)
        vector_config = info.config.params.vectors

        # Our embedding pipeline uses one unnamed vector per chunk.
        if isinstance(vector_config, dict):
         # Display names so we can use the same name during indexing.
            print(f"Available vector names: {list(vector_config)}")

            if len(vector_config) != 1:
                raise ValueError(
                    "Expected exactly one dense vector configuration."
                )

            vector_name, vector_config = next(iter(vector_config.items()))
            print(f"Vector name: {vector_name!r}")

        elif isinstance(vector_config, models.VectorParams):
            print("Vector name: unnamed")

        else:
            raise ValueError("No supported dense vector configuration found.")
        
        if vector_config.size != 384:
            raise ValueError(
                f"Expected 384 dimensions, found {vector_config.size}."
            )

        if vector_config.distance != models.Distance.COSINE:
            raise ValueError("Expected the Cosine distance metric.")

        count = client.count(
            collection_name=collection_name,
            exact=True,
        ).count

        print("Connected to Qdrant Cloud successfully.")
        print(f"Collection: {collection_name}")
        print(f"Dimensions: {vector_config.size}")
        print(f"Distance: {vector_config.distance.value}")
        print(f"Stored points: {count}")

    finally:
        client.close()


if __name__ == "__main__":
    main()