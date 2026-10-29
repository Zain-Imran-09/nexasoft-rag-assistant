
import os

from dotenv import load_dotenv
from groq import Groq


def main() -> None:
    # Load environment variables from the local .env file.
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is missing. Check your local .env file."
        )

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        "Reply with: Groq connection successful."
                    ),
                },
            ],
            max_completion_tokens=150,
        )

        answer = response.choices[0].message.content
        print("SUCCESS: Groq API responded.")
        print(f"Response: {answer}")

    except Exception as exc:
        # Do not print request details or secrets.
        print(f"Groq request failed: {type(exc).__name__}")
        print("Check your API key, model availability, and API limits.")


if __name__ == "__main__":
    main()
