# Bowtie Thesis

## Project Folder Information

The repository is organised into the following major folders and files, each corresponding to a component of the overall workflow:

- **Automatic_JSON_Metrics**: Scripts and data for automatically computing node, edge, and GED metrics from predicted and ground-truth JSON Bowtie diagrams.  
- **Dual LLM Pipeline**: Vision+text dual-LLM workflows (e.g., Pixtral image-to-markdown plus LLM Bowtie JSON generation).  
- **Neo4js Pipeline**: Prototype code for extracting SPO triples from text via vision LLM, storing in Neo4j, and generating Bowtie JSON by querying the graph.  
- **OCR Pipeline**: Integration of OCR (e.g., PaddleOCR or pdfplumber) to extract text and tables from scanned PDFs/images for downstream LLM prompts.  
- **RAG Pipeline**: Components for chunking, FAISS indexing, and the Dash app that retrieves context and runs LLMs to generate Bowtie JSON.  
- **Sobol Code**: Scripts to perform Sobol sensitivity analysis (Run 1 & 2) and stochasticity experiments, narrative experiments, varying prompts, context, and model parameters.  
- **Dockerfile**: This file defines the project's Docker image, installs dependencies, and sets up the runtime environment.  
- **docker-compose.yml**: Orchestrates multi-container setups (e.g., web app, Neo4j database) for development and deployment.  
- **JSON_to_Bowtie.py**: Utility script to convert Bowtie JSON outputs into Mermaid diagrams.  
- **model_selection.py**: Script for CLD generation and hallucination testing, extracting causal loops from narrative text.  

Within each folder, there is an additional README file specific to the folder's contents and what each file is responsible for. Where possible, a video of the output is added as a reference. 


# 🎓 Bowtie Thesis: LLM-Based Causal Diagram Generation

This project explores the use of Large Language Models (LLMs) for automating the extraction of Bowtie diagrams and Causal Loop Diagrams (CLDs) from FMEA-style technical documents. It combines OCR, RAG, Vision LLMs, and Knowledge Graphs in a modular Dash interface.

---

## 🗂️ Folder Overview

Each folder includes its own `README.md` explaining scripts, inputs/outputs, and optional UI demos.

### 🚀 Pipelines

- [`Automatic_JSON_Metrics/`](./Automatic_JSON_Metrics)  
  Compute node, edge, and Graph Edit Distance (GED) metrics between predicted and GT Bowtie JSONs. Includes batch evaluation and Excel export.

- [`Dual LLM Pipeline/`](./Dual%20LLM%20Pipeline)  
  Vision-to-markdown extraction using Pixtral → Bowtie JSON generation using a second LLM (e.g., LLaMA).

- [`Neo4js Pipeline/`](./Neo4js%20Pipeline)  
  Extract SPO triples via Vision LLM → insert into Neo4j → query graph to reconstruct Bowtie JSONs.

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
graph TD
  A[Upload Document] --> B{OCR and Chunking}
  B --> C[Extract Context: Tables or Text]
  C --> D[Select Model + Prompt]
  D --> E[LLM Output → Bowtie JSON]
  E --> F[Evaluate vs Ground Truth]
  E --> G[Render Mermaid Diagram]
  G --> H[Visual Review & Override]
