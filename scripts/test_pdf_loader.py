
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.loader import load_document


def main():
    pdf_path = (
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "sample_company.pdf"
    )

    documents = load_document(pdf_path)

    print("Documents loaded:", len(documents))

    for index, document in enumerate(documents, start=1):
        print(f"\nDocument {index}")
        print("Content:", document.page_content.strip())
        print("Metadata:", document.metadata)

    assert len(documents) == 2
    assert all(
        doc.metadata["filename"] == "sample_company.pdf"
        for doc in documents
    )

    assert all(
        doc.metadata["file_type"] == ".pdf"
        for doc in documents
    )

    print("\nPDF TEST PASSED")


if __name__ == "__main__":
    main()
