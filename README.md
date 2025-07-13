# 🎓 Bowtie Thesis: LLM-Based Causal Diagram Generation

This project explores the use of large language models (LLMs) to automate the extraction of Bowtie diagrams and Causal Loop Diagrams (CLDs) from FMEA-style technical documents. It combines OCR, RAG, Vision LLMs, and Knowledge Graphs in a modular Dash interface.

---

## 🗂️ Folder Overview

Each folder includes its own `README.md` explaining scripts, inputs/outputs, and UI demos.

### 🚀 Pipelines

- [`Automatic_JSON_Metrics/`](./Automatic_JSON_Metrics)  
  Compute node, edge, and Graph Edit Distance (GED) metrics between predicted and GT Bowtie JSONs. Includes batch evaluation and Excel export.

- [`Dual LLM Pipeline/`](./Dual%20LLM%20Pipeline)  
  Vision-to-markdown extraction using Pixtral → Bowtie JSON generation using a second LLM (e.g., LLaMA).

- [`Neo4js Pipeline/`](./Neo4js%20Pipeline)  
  Extract causal chains via Vision LLM → insert into Neo4j → query graph to reconstruct Bowtie JSONs.

- [`OCR Pipeline/`](./OCR%20Pipeline)  
  Uses img2table/Tesseract or PaddleOCR to extract tables/text from scanned PDFs and images for prompt input.

- [`RAG Pipeline/`](./RAG%20Pipeline)  
  Chunk PDFs → build FAISS index → search context → prompt LLM → generate Bowtie JSON. Includes Dash app with Mermaid rendering.

- [`Sobol Code/`](./Sobol%20Code)  
  Sensitivity analysis using Sobol (Run 1 & 2). Includes stochasticity tests, prompt variability, context types, and model selection logic.

---

### 🧰 Utilities

- `model_selection.py`  
  CLD extraction and hallucination testing from narrative text using DeepSeek/Mistral.

- `JSON_to_Bowtie.py`  
  Converts any Bowtie JSON into Mermaid diagram syntax.

- `Dockerfile` / `docker-compose.yml`  
  Build and run the full app with all dependencies and optional Neo4j support.

---

## 📈 Pipeline Flow

```mermaid
%%{init: {"theme": "default", "themeVariables": { "fontSize": "14px", "fontFamily": "Inter, sans-serif", "primaryTextColor": "#000" }}}%%

graph TD
  %% Method Choice
  A[Start: Input FMEA] --> B{Choose Method}

  %% RAG Pipeline - Light Yellow
  subgraph RAG Pipeline
    direction TB
    C1[RAG Pipeline]:::rag
    D1[Chunk PDF → JSON]:::rag
    E1[FAISS Index + Search]:::rag
    F1[Context + Prompt → LLM]:::rag
  end
  B --> C1
  C1 --> D1 --> E1 --> F1 --> G

  %% OCR Pipeline - Light Blue
  subgraph OCR Pipeline
    direction TB
    C2[OCR Pipeline]:::ocr
    D2[Extract Text/Tables: OCR]:::ocr
    F2[Send to LLM]:::ocr
  end
  B --> C2
  C2 --> D2 --> F2 --> G

  %% Dual LLM Pipeline - Light Green
  subgraph Dual LLM Pipeline
    direction TB
    C3[Dual LLM Pipeline]:::dual
    D3[Pixtral Extracts Table JSON]:::dual
    F3[Reasoning LLM → Bowtie JSON]:::dual
  end
  B --> C3
  C3 --> D3 --> F3 --> G

  %% Neo4j Pipeline - Light Purple
  subgraph Neo4j Knowledge Graph Pipeline
    direction TB
    C4[Neo4j Knowledge Graph Pipeline]:::neo
    D4[Vision LLM Extracts SPO Triples]:::neo
    E4[Store in Neo4j]:::neo
    F4[Query Graph → Bowtie JSON]:::neo
  end
  B --> C4
  C4 --> D4 --> E4 --> F4 --> G

  %% Shared Post-processing
  G[Generate Bowtie JSON]
  G --> H[Evaluate vs Ground Truth]
  G --> I[Render Mermaid Diagram]

  %% Styling
  classDef rag fill:#FFFACD,color:#000;
  classDef ocr fill:#D8EEFF,color:#000;
  classDef dual fill:#DFFFD7,color:#000;
  classDef neo fill:#EAD7FF,color:#000;


