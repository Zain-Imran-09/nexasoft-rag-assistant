import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Read variables from the project's .env file.
# Existing system environment variables take priority.
load_dotenv()


@dataclass(frozen=True)
class EmbeddingConfig:
    """Settings used when loading and running the embedding model."""

    model_name: str
    device: str
    batch_size: int


def load_embedding_config() -> EmbeddingConfig:
    """Read and validate embedding settings."""

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ).strip()

    device = os.getenv("EMBEDDING_DEVICE", "cpu").strip()

    # Environment variables are strings, so convert batch size to an integer.
    try:
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    except ValueError as exc:
        raise ValueError(
            "EMBEDDING_BATCH_SIZE must be a whole number."
        ) from exc

    if not model_name:
        raise ValueError("EMBEDDING_MODEL cannot be empty.")

    if device != "cpu":
        raise ValueError(
            "This project's current embedding configuration supports cpu only."
        )

    if batch_size < 1:
        raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1.")

    return EmbeddingConfig(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )