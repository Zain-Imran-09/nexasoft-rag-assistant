from pathlib import Path
import sys

# Add the project root so this script can import the sibling `app` package
# when it is run directly by file path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our reusable loader function.
from app.rag.loader import load_documents


# Locate the project's data directory.
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"


def main():

    # Load all supported files.
    documents = load_documents(str(DATA_FOLDER))

    # Display the number of Documents.
    print("Total documents loaded:", len(documents))

    # Show basic details for every Document.
    for index, document in enumerate(documents, start=1):

        print(f"\nDocument {index}")
        print("Filename:", document.metadata["filename"])
        print("File type:", document.metadata["file_type"])
        print("Characters:", len(document.page_content))
        print("Source:", document.metadata["source"])

    # Confirm the expected dataset size.
    assert len(documents) == 10, (
        f"Expected 10 documents, got {len(documents)}"
    )

    # Confirm that all Documents contain text.
    assert all(
        document.page_content.strip()
        for document in documents
    ), "One or more documents are empty"

    # Confirm required metadata exists.
    assert all(
        all(
            field in document.metadata
            for field in ("source", "filename", "file_type")
        )
        for document in documents
    ), "Required metadata is missing"

    print("\nSUCCESS: All 10 documents passed basic checks.")


if __name__ == "__main__":
    main()
