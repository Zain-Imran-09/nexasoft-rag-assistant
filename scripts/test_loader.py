
# Import Path to work with file paths.
from pathlib import Path

# Import LangChain's text-file loader.
from langchain_community.document_loaders import TextLoader


# Find the root directory of our project.
# __file__ is the location of this Python script.
# parent.parent goes from scripts/ to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Construct the path to our first company document.
file_path = PROJECT_ROOT / "data" / "raw" / "company_overview.md"

# Make sure the document exists before loading it.
if not file_path.is_file():
    raise FileNotFoundError(f"Document not found: {file_path}")

# Create the loader and tell it which file to read.
# UTF-8 supports standard English and Unicode text.
loader = TextLoader(
    file_path=str(file_path),
    encoding="utf-8"
)

# Read the document and convert it into LangChain Documents.
documents = loader.load()

# Display how many Documents were returned.
print("Total documents loaded:", len(documents))

# Access the first Document.
document = documents[0]

# Display the actual text content (first 300 characters).
print("\nDocument Content:")
print(document.page_content[:300])

# Display information about the source file.
print("\nDocument Metadata:")
print(document.metadata)
