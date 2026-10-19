# NexaSoft RAG Assistant

A production-style RAG knowledge assistant for a software company.

## Goal

The system allows company admins to manage approved company documents
and allows website users to ask questions and receive grounded answers
with source citations.

## Planned Stack

- Python
- LangChain
- Qdrant
- FastAPI
- SQLite / PostgreSQL
- Docker
- AWS


## Document Loading

The NexaSoft RAG Assistant supports loading
company documents into LangChain Document objects.

### Supported Formats

- Markdown (.md)
- Plain text (.txt)
- PDF (.pdf)

### Implementation

Main module:
app/rag/loader.py

Functions:
- load_document(): Loads one supported file.
- load_documents(): Loads supported files in a folder.

### Document Metadata

Each Document includes:

- source: Original file path
- filename: Original filename
- file_type: Source file extension

PDF documents may also include page metadata.

### Error Handling

Unsupported formats are skipped during folder loading.

Unreadable supported files:
- Raise an exception by default.
- Can be skipped using skip_errors=True.
- Produce warning logs when skipped.

The default fail-fast behavior helps prevent
incomplete knowledge bases from going unnoticed.

### Running the Tests

From the project root:

```powershell
python -m scripts.test_all_loaders
python -m scripts.test_pdf_loader
python -m scripts.test_loader_errors
```

### Current Results

- 10 fictional company source documents verified.
- TXT and Markdown loading successful.
- Two-page test PDF loaded successfully.
- Missing-file detection successful.
- Unsupported-format detection successful.
- Corrupt-PDF handling successful.

### Limitations

- No scanned-PDF OCR.
- No image or table understanding.
- No document chunking.
- No embeddings or vector indexing.
- No LLM-powered question answering yet.
