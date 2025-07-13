# OCR to Bowtie Dash App

A Dash-based application to extract tables and text from scanned FMEA documents (images or PDFs), parse them into structured context, and generate Bowtie diagram JSONs using local LLMs (via llama-cpp). The app also supports rendering Mermaid diagrams from JSON.

---

### Demo
https://github.com/user-attachments/assets/5d00109a-3673-4019-aa30-1fbb76e015d0

---


## 📁 File Overview

| File | Description |
|------|-------------|
| `ocr_app.py` | Full Dash application with two tabs: OCR-to-Bowtie and Mermaid Visualizer. Integrates OCR, LLM inference, JSON parsing, and diagram rendering. |

---

## 🧠 Pipeline Overview

### 1. Input
- Upload scanned **images or PDFs**
- Optionally define column headers (e.g., "Failure Mode", "Failure Cause", "Failure Effect")
- Provide part/component name

### 2. OCR Extraction
- Uses Tesseract or `img2table` to extract table and free text from images
- Cleans and merges broken rows
- Produces markdown-formatted tables and OCR'd text

### 3. LLM Inference (via llama-cpp)
- Combines markdown + text + part name into a structured prompt
- Sends it to a local LLM (e.g., LLaMA-3-8B)
- Receives a Bowtie JSON
- Handles seeding for consistent results

### 4. Mermaid Rendering
- The second tab allows users to paste/upload Bowtie JSON
- Renders a graph using Mermaid's `graph LR` syntax

---
