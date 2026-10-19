
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.loader import (
    load_document,
    load_documents,
)


def main():
    # Test 1: Missing file
    try:
        load_document(Path("missing_file.txt"))
    except FileNotFoundError:
        print("PASS: Missing file detected")

    # Test 2: Unsupported file format
    with TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)

        unsupported = directory / "sample.xlsx"
        unsupported.write_text("test", encoding="utf-8")

        try:
            load_document(unsupported)
        except ValueError:
            print("PASS: Unsupported format rejected")

        # Test 3: Corrupt PDF
        bad_pdf = directory / "corrupt.pdf"
        bad_pdf.write_bytes(b"This is not a valid PDF")

        # Valid file for skip-errors testing
        good_file = directory / "valid.txt"
        good_file.write_text(
            "NexaSoft provides software services.",
            encoding="utf-8"
        )

        try:
            load_document(bad_pdf)
        except Exception:
            print("PASS: Corrupt PDF detected")

        # Loading continues despite the corrupt PDF.
        documents = load_documents(
            directory,
            skip_errors=True
        )

        assert len(documents) == 1
        assert documents[0].metadata["filename"] == "valid.txt"

        print("PASS: Valid file loaded despite corrupt PDF")

    print("\nERROR HANDLING TESTS PASSED")


if __name__ == "__main__":
    main()
