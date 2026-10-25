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


## Document Chunking

The NexaSoft RAG Assistant uses LangChain's
RecursiveCharacterTextSplitter to split loaded
documents into smaller units for retrieval.

### Initial Configuration

- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Length function: len()
- Metadata: Preserved from original documents

### Experiments

| Size | Overlap | Total Chunks | Short Chunks (<100) |
|------|---------|--------------|---------------------|
| 500  | 100     | 22           | 1                   |
| 1000 | 200     | 12           | 0                   |
| 1500 | 300     | 10           | 0                   |

### Decision

1000/200 was selected as the initial baseline
because it produces manageable chunks while
preserving meaningful document context.

A 43-character standalone chunk was detected
in pricing.md under the 500/100 configuration.

This decision is preliminary and will be
evaluated against retrieval quality later.

### Validation

Run automated tests:

python -m pytest tests/test_chunker.py -v

Run chunk inspection:

python -m scripts.inspect_chunks

## Day 6 — Embeddings Pipeline

Implemented a reusable embedding service that converts text chunks into numerical vectors for future semantic retrieval.

### Model and configuration

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Device:** CPU
- **Vector dimensions:** 384
- **Maximum input length:** 256 tokens, including special tokens
- **Batch size:** 8
- **Normalization:** Enabled

Settings are loaded from environment variables through `app/core/config.py`:

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=8
```

Copy these settings into `.env` for local use and document them in `.env.example`. Keep `.env` out of Git.

### Files added

| File | Purpose |
|---|---|
| `app/core/config.py` | Loads and validates embedding configuration |
| `app/rag/embeddings.py` | Loads the model and generates document and query vectors |
| `scripts/test_embedding_similarity.py` | Compares a question with related and unrelated sample texts |
| `scripts/inspect_embeddings.py` | Verifies embeddings for the company document chunks |

### Service behavior

- `get_embedding_service()` caches and reuses the service within one Python process.
- `embed_documents()` accepts a list of strings and returns one vector per string.
- `embed_query()` accepts one question and returns one vector.
- Empty document lists return an empty list.
- Blank or non-string texts raise a validation error.
- Inputs exceeding the model token limit are rejected before embedding.
- The first model load requires internet access to download the model. Later loads can reuse the local model cache.

### Verification results

Using the Day 5 baseline of **1,000 characters per chunk** with **200 characters of target overlap**:

| Check | Result |
|---|---|
| Documents loaded | 10 |
| Chunks generated | 12 |
| Vectors generated | 12 |
| Dimensions per vector | 384 |
| Chunk token counts | 52–254 |
| All chunks within the 256-token limit | Passed |
| All vector values finite | Passed |
| All vectors normalized | Passed |
| Dependency check | No broken requirements found |

The largest chunk by token count came from `faq.txt` at **254 tokens**. Future document changes must be checked again because character limits do not guarantee token limits.

### Semantic similarity example

Query: **“How much does website development cost?”**

| Sample text | Cosine similarity |
|---|---:|
| What is the price of building a website? | 0.8414 |
| NexaSoft develops websites for businesses. | 0.3794 |
| Our office opens at nine in the morning. | 0.0298 |

The pricing sentence ranked highest. These scores measure similarity; they are not confidence percentages. This small example does not establish retrieval quality across the full knowledge base.

### Run verification

Run from the project root:

```powershell
conda activate nexasoft-rag
python -m scripts.test_embedding_similarity
python -m scripts.inspect_embeddings
python -m pip check
```

### Dependencies added

```text
sentence-transformers==6.2.0
langchain-text-splitters==1.1.3
```

`langchain-text-splitters` is recorded explicitly because the chunker uses it directly.

### Current scope

The inspection script connects loading, chunking, and embedding for verification. It keeps text, metadata, and vectors paired in memory.

Vector database storage, production ingestion integration, retrieval, and answer generation will be implemented in later stages.

## Day 7 — Qdrant Vector Database

Connected the NexaSoft RAG Assistant's embedding pipeline to Qdrant Cloud and indexed the company knowledge base.

### Collection Configuration

| Setting | Value |
|---|---|
| Database | Qdrant Cloud |
| Collection | `nexasoft_knowledge_base` |
| Vector dimensions | 384 |
| Distance metric | Cosine |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Chunk size | 1,000 characters |
| Chunk overlap | 200 characters |

### Indexing Results

| Check | Result |
|---|---|
| Source documents loaded | 10 |
| Chunks generated | 12 |
| Points uploaded | 12 |
| Uploaded records verified | 12 |
| Total collection points | 12 |
| Qdrant connection | Passed |
| Collection configuration check | Passed |
| Payload verification in indexing script | Passed |

### Stored Point Payload

Each indexed point stores the text chunk and its associated information so the application can identify where retrieved content came from.

Expected payload fields:

- `text` — the text content of the chunk.
- `metadata` — document metadata, including `source` and `filename`.
- `chunk_index` — the chunk's index.
- `embedding_model` — the model used to generate the embedding.

### Verification Commands

Run these commands from the project root:

```powershell
conda activate nexasoft-rag
python -m scripts.check_qdrant
python -m scripts.index_documents
```

The indexing script successfully uploaded and verified 12 records. The Qdrant connection and collection configuration checks also passed.

### Current Scope

The knowledge base is now indexed in Qdrant Cloud. Document loading, chunking, embedding generation, and vector storage have been implemented and tested individually through the project's verification scripts.

Semantic retrieval and answer generation are not yet part of this completed stage.
