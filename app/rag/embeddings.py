from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import EmbeddingConfig, load_embedding_config


class EmbeddingService:
    """Load an embedding model and convert text into vectors."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

        # Download the model on first use; later loads use the local cache.
        self.model = SentenceTransformer(
            config.model_name,
            device=config.device,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each input text."""

        if not texts:
            return []

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Text at index {index} must be a non-empty string."
                )

            # Check token length so long inputs are not silently truncated.
            token_ids = self.model.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]

            if len(token_ids) > self.model.max_seq_length:
                raise ValueError(
                    f"Text at index {index} has {len(token_ids)} tokens; "
                    f"the model limit is {self.model.max_seq_length}. "
                    "Split it into smaller chunks before embedding."
                )

        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Convert the NumPy array into ordinary Python lists.
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Return one embedding vector for a user's question."""

        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Create the service once and reuse it within this Python process."""

    config = load_embedding_config()
    return EmbeddingService(config)