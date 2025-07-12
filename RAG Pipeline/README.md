## Overview

1. **chunking.py**: Extracts sections and tables from a PDF into structured JSON chunks
2. **faiss_index_creator.py**: Converts chunks to embeddings and builds a FAISS search index
3. **rag_app_faiss.py**: Provides a Dash UI for retrieving chunks, configuring LLMs, and generating Bowtie diagrams with Mermaid visualization

---

## Files & What They Do

### `chunking.py`
Extracts content from a technical FMEA PDF and saves three types of chunked outputs:

- `sections.json`: textual sections extracted using PyMuPDF
- `tables.json`: structured tables extracted using pdfplumber + captions
- `combined_chunks.json`: unified, sorted merge of both tables and text

**Output folder:**
- Three JSON files (sections, tables, combined) for downstream FAISS creation

---

### `faiss_index_creator.py`
Takes the output of `chunking.py` and encodes all chunks into sentence embeddings (via `sentence-transformers`). Then, it builds FAISS indices for each source.

**Output folder:**
- `faiss_chunks/`
  - `faiss_sections.idx`
  - `faiss_tables.idx`
  - `faiss_combined.idx`
  - Corresponding `*_metadata.json` files

These can be queried at runtime to retrieve the top-k semantically similar chunks.

---

### `rag_app_faiss.py`
Dash UI for:

- Selecting a FAISS source (tables, text, or combined)
- Entering a part/component name
- Retrieving and previewing top-k context chunks
- Selecting LLM model, temperature, top-p, top-k
- Choosing prompt type (zero, few, CoT)
- Running the LLM over the retrieved context
- Visualizing the result as a Mermaid diagram

**How to run:**
```bash
pip install -r requirements.txt
python rag_app_faiss.py
```

**Inputs:**
- FAISS indices (from previous step)
- Prompt templates
- Local GGUF models

**Outputs:**
- Bowtie JSONs stored to disk per model
- Mermaid visualization of each output

---

## Folder: `faiss_chunks`
Stores all `.idx` files and corresponding metadata JSONs for chunk search. This folder is auto-populated when `faiss_index_creator.py` is run.
