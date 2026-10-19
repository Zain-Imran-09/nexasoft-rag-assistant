import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from pypdf.errors import PdfReadError

# Use the application's logging settings to report progress and skipped files.
logger = logging.getLogger(__name__)

# Supported company document formats.
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_document(file_path: Path) -> list[Document]:
    """Read one .md, .txt, or .pdf file and return its non-empty Documents.

    A Document contains text in `page_content` and source details in `metadata`.
    Text files produce one Document; PDFs produce one Document per page before
    empty pages are removed. Errors are passed back to the caller.
    """

    # 1. Convert the input to a Path so strings work too.
    file_path = Path(file_path)

    # 2. Check the file before trying to read it.
    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # suffix gives the final extension, such as '.pdf'. lower() accepts '.PDF'.
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    # 3. Choose how to read the file. Both loaders provide a load() method.
    if extension == ".pdf":
        loader = PyPDFLoader(
            file_path=str(file_path),
            mode="page",
        )
    else:
        # Markdown (.md) is also plain text, so it uses the same text loader.
        loader = TextLoader(
            file_path=str(file_path),
            encoding="utf-8",
        )

    loaded_documents = loader.load()

    # 4. Collect Documents that contain more than spaces or line breaks.
    documents: list[Document] = []
    for document in loaded_documents:
        # strip() checks a copy of the text; the original content stays intact.
        if document.page_content.strip():
            documents.append(document)

    if not documents:
        raise ValueError(f"No readable text found: {file_path}")

    # 5. Add the same source fields to every Document. Other metadata, such as
    # a PDF's page number, stays in the dictionary.
    for document in documents:
        document.metadata["source"] = str(file_path)
        document.metadata["filename"] = file_path.name
        document.metadata["file_type"] = extension

    return documents


def load_documents(
    directory: Path,
    skip_errors: bool = False,
) -> list[Document]:
    """Load supported files directly inside a folder, in sorted path order.

    Subfolders are ignored. By default, a loading error stops the function.
    With skip_errors=True, the errors listed below are logged and loading
    continues with the next file. An empty folder returns an empty list.
    """

    # 1. Check that the folder exists.
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Directory does not exist: {directory}"
        )

    all_documents: list[Document] = []

    # 2. Visit the folder's immediate entries in a predictable order.
    for file_path in sorted(directory.iterdir()):
        # continue skips the rest of this loop and moves to the next entry.
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        # A folder may contain other file types; log them and move on.
        if extension not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "Unsupported file skipped: %s",
                file_path.name,
            )
            continue

        try:
            # 3. Reuse the single-file function for reading and metadata.
            documents = load_document(file_path)

            # extend adds each Document to the result. append would add the
            # whole list as one item, creating a nested list instead.
            all_documents.extend(documents)

            logger.info(
                "Loaded %s (%d Documents)",
                file_path.name,
                len(documents),
            )

        except (OSError, ValueError, RuntimeError, PdfReadError) as error:
            # 4. Decide whether a failed file should stop the folder load.
            if not skip_errors:
                # A bare raise passes the original exception to the caller.
                raise

            # Log the failure; the loop then moves to the next file.
            logger.warning(
                "Failed to load %s: %s",
                file_path.name,
                error,
            )

    return all_documents
