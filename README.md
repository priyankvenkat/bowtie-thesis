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

## 📈 Pipeline Flow (Mermaid)

```mermaid
%%{init: {'theme': 'default'}}%%
graph TD

  %% Color styling
  classDef rag fill=#E0F7FA,stroke=#00ACC1,stroke-width=2
  classDef ocr fill=#FFF3E0,stroke=#FB8C00,stroke-width=2
  classDef dual fill=#F3E5F5,stroke=#8E24AA,stroke-width=2
  classDef kg fill=#E8F5E9,stroke=#43A047,stroke-width=2
  classDef output fill=#ECEFF1,stroke=#607D8B,stroke-width=2

  A[Start: Input FMEA]

  A --> B{Choose Method}
  B --> C1[RAG Pipeline]
  B --> C2[OCR Pipeline]
  B --> C3[Dual LLM Pipeline]
  B --> C4[Neo4j Knowledge Graph Pipeline]

  %% RAG
  C1 --> D1[Chunk PDF → JSON]
  D1 --> E1[FAISS Index + Search]
  E1 --> F1[Context + Prompt → LLM]
  F1 --> G[Generate Bowtie JSON]

  %% OCR
  C2 --> D2[Extract Text/Tables via OCR]
  D2 --> F2[Send to LLM]
  F2 --> G

  %% Dual LLM
  C3 --> D3[Pixtral → Table Markdown + Summary]
  D3 --> F3[Reasoning LLM → Bowtie JSON]
  F3 --> G

  %% Knowledge Graph
  C4 --> D4[Vision LLM → SPO Triples]
  D4 --> E4[Store in Neo4j]
  E4 --> F4[Query Graph → Bowtie JSON]
  F4 --> G

  %% Output
  G --> H[Evaluate vs Ground Truth]
  G --> I[Render Mermaid Diagram]

  %% Apply styles
  class C1,D1,E1,F1 rag
  class C2,D2,F2 ocr
  class C3,D3,F3 dual
  class C4,D4,E4,F4 kg
  class G,H,I output


