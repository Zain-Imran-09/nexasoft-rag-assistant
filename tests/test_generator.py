
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.rag.generator import (
    DEFAULT_ANSWER,
    format_context,
    generate_answer,
)


class TestGenerator(unittest.TestCase):

    def setUp(self):
        self.question = "What is the website development price?"
        self.chunks = [
            {
                "text": "Website development starts at PKR 50,000.",
                "filename": "pricing.md",
                "source": "pricing.md",
                "page": 2,
                "chunk_index": 0,
                "score": 0.84,
            }
        ]

    def test_format_context_includes_source_and_text(self):
        context = format_context(self.chunks)

        self.assertIn("pricing.md", context)
        self.assertIn("PKR 50,000", context)
        self.assertIn("[Source 1]", context)
        self.assertIn("Filename: pricing.md", context)
        self.assertIn("Source: pricing.md", context)
        self.assertIn("Page: 2", context)

    def test_empty_context_returns_unknown_without_api_call(self):
        with patch("app.rag.generator.Groq") as mock_groq:
            answer = generate_answer(self.question, [])

        self.assertEqual(answer, DEFAULT_ANSWER)
        mock_groq.assert_not_called()

    def test_blank_question_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_answer("   ", self.chunks)

    def test_non_list_chunks_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_answer(self.question, "not a list")

    @patch("app.rag.generator.Groq")
    def test_generates_answer_from_retrieved_context(self, mock_groq):
        client = mock_groq.return_value

        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Website development starts at PKR 50,000. "
                            "[Source 1]"
                        )
                    )
                )
            ]
        )

        client.chat.completions.create.return_value = fake_response

        answer = generate_answer(self.question, self.chunks)

        self.assertIn("PKR 50,000", answer)
        self.assertIn("[Source 1]", answer)
        client.chat.completions.create.assert_called_once()

        # Verify that the question and retrieved context were sent.
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        prompt = messages[1]["content"]

        self.assertIn(self.question, prompt)
        self.assertIn("PKR 50,000", prompt)
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
