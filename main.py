import os

from dotenv import load_dotenv


load_dotenv()


def main():
    app_name = os.getenv("APP_NAME", "NexaSoft RAG Assistant")
    app_env = os.getenv("APP_ENV", "development")

    print(f"Starting: {app_name}")
    print(f"Environment: {app_env}")


if __name__ == "__main__":
    main()