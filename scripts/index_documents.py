import json
import os
from collections import defaultdict
from math import isfinite
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from app.rag.loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embeddings import get_embedding_service


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    collection = os.getenv("QDRANT_COLLECTION", "").strip()

    if not all([url, api_key, collection]):
        raise ValueError("Missing Qdrant settings in .env.")

    client = QdrantClient(url=url, api_key=api_key, timeout=60)

    try:
        # Validate the existing collection before generating embeddings.
        info = client.get_collection(collection)
        vector_config = info.config.params.vectors
        vector_name = None

        if isinstance(vector_config, dict):
            if len(vector_config) != 1:
                raise ValueError("Expected one dense vector configuration.")

            vector_name, vector_config = next(iter(vector_config.items()))

        if not isinstance(vector_config, models.VectorParams):
            raise ValueError("Unsupported vector configuration.")

        if vector_config.distance != models.Distance.COSINE:
            raise ValueError("Expected cosine similarity.")

        documents = load_documents(DATA_DIR, skip_errors=False)
        chunks = chunk_documents(
            documents,
            chunk_size=1000,
            chunk_overlap=200,
        )

        print(f"Loaded documents: {len(documents)}")
        print(f"Generated chunks: {len(chunks)}")

        if not chunks:
            raise ValueError("No chunks available for indexing.")

        service = get_embedding_service()
        dimensions = service.model.get_embedding_dimension()

        if vector_config.size != dimensions:
            raise ValueError(
                f"Collection dimensions: {vector_config.size}; "
                f"model dimensions: {dimensions}."
            )

        # Protect against silently truncating oversized chunks.
        for index, chunk in enumerate(chunks, start=1):
            token_ids = service.model.tokenizer(
                chunk.page_content,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]

            if len(token_ids) > service.model.max_seq_length:
                raise ValueError(
                    f"Chunk {index} exceeds the model token limit."
                )

        vectors = service.embed_documents(
            [chunk.page_content for chunk in chunks]
        )

        if len(vectors) != len(chunks):
            raise ValueError("Vector count does not match chunk count.")

        points = []
        source_counts = defaultdict(int)

        for chunk, vector in zip(chunks, vectors):
            # Convert NumPy values, if present, to plain Python floats.
            vector = [float(value) for value in vector]

            if len(vector) != dimensions:
                raise ValueError("Unexpected vector dimensions.")

            if not all(isfinite(value) for value in vector):
                raise ValueError("Vector contains invalid numerical values.")

            norm = sum(value * value for value in vector) ** 0.5
            if abs(norm - 1.0) > 1e-4:
                raise ValueError("Vector is not normalized.")

            metadata = dict(chunk.metadata)
            source = metadata.get("source")

            if not source:
                raise ValueError("Chunk is missing source metadata.")

            # Store a portable source path rather than a laptop-specific path.
            source_path = Path(source).resolve()
            source_key = source_path.relative_to(DATA_DIR.resolve()).as_posix()

            chunk_index = source_counts[source_key]
            source_counts[source_key] += 1

            metadata["source"] = source_key
            metadata["filename"] = source_path.name

            # Identical source, position, and text produce the same UUID.
            identity = json.dumps(
                [collection, source_key, chunk_index, chunk.page_content],
                ensure_ascii=False,
            )
            point_id = str(uuid5(NAMESPACE_URL, identity))

            # An empty-string name represents the default vector.
            vector_data = (
                {vector_name: vector}
                if vector_name
                else vector
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector_data,
                    payload={
                        "text": chunk.page_content,
                        "metadata": metadata,
                        "chunk_index": chunk_index,
                        "embedding_model": service.config.model_name,
                    },
                )
            )

        # Insert new points or replace points with matching IDs.
        client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )
        print(f"Uploaded points: {len(points)}")

        # Read back exactly the records we uploaded.
        stored = client.retrieve(
            collection_name=collection,
            ids=[point.id for point in points],
            with_payload=True,
            with_vectors=False,
        )

        expected = {str(point.id): point.payload for point in points}

        if {str(point.id) for point in stored} != set(expected):
            raise ValueError("Some uploaded points could not be retrieved.")

        for point in stored:
            if point.payload != expected[str(point.id)]:
                raise ValueError(f"Payload mismatch for point {point.id}.")

        total = client.count(
            collection_name=collection,
            exact=True,
        ).count

        print(f"Verified uploaded records: {len(stored)}")
        print(f"Total collection points: {total}")
        print("SUCCESS: Uploaded records and payloads verified.")

    finally:
        client.close()


if __name__ == "__main__":
    main()