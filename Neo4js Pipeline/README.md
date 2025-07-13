# Bowtie Diagram Extraction via Vision-Language Models and Knowledge Graphs

This project implements a prototype pipeline that extracts causal information from technical diagrams or failure reports (e.g., FMEA tables) using Vision LLMs and stores the output in a Neo4j knowledge graph. The extracted data is then converted into structured Bowtie diagram JSONs and visualized interactively using Mermaid.js in a Dash web app.

---

### Demo
https://github.com/user-attachments/assets/7bcaa688-711c-44c4-8690-17cb9774b695

---

## 📁 Project Structure

| File              | Description |
|-------------------|-------------|
| `app.py`          | Main Dash app. Defines layout and tabbed interface for uploading images, generating Bowtie JSON, and rendering Mermaid diagrams. |
| `callbacks.py`    | Dash callbacks for handling UI events (image upload, JSON generation, Mermaid rendering). Also connects LLM and parsing logic. |
| `config.py`       | Stores API keys, model names, and Neo4j connection credentials. |
| `kg_pipeline.py`  | Core pipeline for extracting causal chains from images and storing them in a Neo4j knowledge graph. Also handles reconstruction of Bowtie JSON from the graph. |
| `parsing.py` | Utilities to normalise LLM output, expand nested structures, and convert Bowtie JSON to Mermaid syntax for visualisation. |
| `prompts.py`      | Manages structured prompt templates (zero-shot, few-shot, CoT) used for Bowtie JSON generation. |
| `test_neo.py`     | Standalone script for testing Neo4j connectivity and running sample queries to validate the graph structure. |

---

## 🧠 Pipeline Overview

### 1. Input
- Upload a technical diagram or FMEA-style image via the Dash UI.

### 2. Extraction
- A Vision LLM (e.g., Pixtral) processes the image and returns structured causal chains in JSON format.
- Each chain includes:
  - `causes` (list)
  - `mechanism`
  - `critical_event`
  - `consequences` (list)
  - `barriers` (list)

### 3. Knowledge Graph
- Extracted chains are stored in Neo4j using relationships like `CAUSES`, `TRIGGERS`, `LEADS_TO`, and `MITIGATED_BY`.

### 4. Bowtie Reconstruction
- A graph query fetches nodes and relationships around a selected critical event and reconstructs a Bowtie JSON.
- This JSON is visualized in the Dash app as a Mermaid diagram.

---

