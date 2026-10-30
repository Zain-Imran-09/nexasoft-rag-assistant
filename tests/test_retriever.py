import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.rag.retriever import retrieve_chunks


class TestRetriever(unittest.TestCase):

    @patch("app.rag.retriever.get_qdrant_client")
    @patch("app.rag.retriever.get_embedding_service")
    def test_returns_citation_and_debug_metadata(
        self,
        mock_embedding_service,
        mock_get_client,
    ):
        mock_embedding_service.return_value.embed_query.return_value = [0.1]
        point = SimpleNamespace(
            payload={
                "text": "Pricing details.",
                "metadata": {
                    "source": "pricing.pdf",
                    "filename": "pricing.pdf",
                    "page": 3,
                },
                "chunk_index": 1,
            },
            score=0.91,
        )
        client = mock_get_client.return_value
        client.query_points.return_value.points = [point]

        with patch.dict("os.environ", {"QDRANT_COLLECTION": "knowledge"}):
            results = retrieve_chunks("What are the prices?")

        self.assertEqual(results[0]["filename"], "pricing.pdf")
        self.assertEqual(results[0]["source"], "pricing.pdf")
        self.assertEqual(results[0]["page"], 3)
        self.assertEqual(results[0]["metadata"]["page"], 3)
        self.assertEqual(results[0]["text"], "Pricing details.")
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
