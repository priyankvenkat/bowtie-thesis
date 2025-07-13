# RAG Pipeline

This pipeline enables semantic retrieval from chunked FMEA documents using FAISS and passes retrieved context into local LLMs to generate Bowtie diagrams via a Dash interface.

---

### Demo
https://github.com/user-attachments/assets/0c5ba7d3-e01e-4056-94b5-2a125f147599

---

## 📁 File Overview

| File                   | Description |
|------------------------|-------------|
| `chunking.py`          | Extracts structured content from FMEA PDFs into three JSONs: sections (text), tables, and combined. |
| `faiss_index_creator.py` | Converts chunked JSONs into sentence embeddings and builds searchable FAISS indices. |
| `rag_app_faiss.py`     | Dash app for retrieving chunks, configuring LLMs (model + prompt), running inference, and rendering Mermaid diagrams. |

---

## 🧠 Pipeline Overview

### 1. PDF → Chunks
- `chunking.py` separates the PDF into:
  - `sections.json`: text blocks with headings
  - `tables.json`: tables with column structure
  - `combined_chunks.json`: merged and sorted by page + position

### 2. Chunks → FAISS
- `faiss_index_creator.py` encodes chunk content using `sentence-transformers` and builds `.idx` files
- Outputs:
  - `faiss_chunks/faiss_sections.idx`
  - `faiss_chunks/faiss_tables.idx`
  - `faiss_chunks/faiss_combined.idx`
  - Plus metadata JSONs for traceability

### 3. UI: RAG App
- `rag_app_faiss.py` lets users:
  - Select index source (tables, text, combined)
  - Enter part name (e.g., “sensor”)
  - Choose prompt type (zero/few/CoT), temperature, top-p, etc.
  - Run a local LLM (via `llama-cpp`)
  - Preview the JSON and render a Mermaid diagram

---
