
# Dual LLM Pipeline

This folder contains a Dash-based application that performs vision-based extraction of causal information from technical images (e.g., FMEA tables), processes it using local LLMs, and visualises the result as Bowtie diagrams using Mermaid.

---

### Demo
https://github.com/user-attachments/assets/0fda09fe-ec25-41f9-b3de-37bf43267412

---

## 📁 Project Structure

| File             | Description |
|------------------|-------------|
| `app.py`         | Main Dash app. Defines layout and tabbed interface for uploading images, generating Bowtie JSON, and rendering Mermaid diagrams. |
| `callbacks.py`   | Dash callbacks handle UI events (image upload, JSON generation, Mermaid rendering). Also connects LLM and parsing logic. |
| `config.py`      | Stores API keys, model names, and configuration options for vision and reasoning stages. |
| `llm_runner.py`  | Handles invocation of local LLaMA models using `llama-cpp`, with adjustable parameters like temperature, top-p, and max tokens. |
| `parsing.py`     | Utilities to parse markdown, normalise LLM output, and generate Mermaid graphs from Bowtie JSON. |
| `prompts.py`     | Selects and injects structured prompts based on current user settings (prompt type, strictness, etc.). |

---


## 🧠 Pipeline Overview

### 1. Input
- User uploads an image (JPG/PNG) containing a technical table or scanned content.
- Optionally provides part/component name and selects a prompt type (zero-shot, few-shot, CoT).

### 2. Vision LLM Extraction
- The image is processed by a remote vision model (e.g., Mistral API or Pixtral).
- It returns:
  - Markdown-formatted table
  - Summary text
  - Extracted causal components

### 3. Reasoning LLM
- The markdown + summary context is passed into a local reasoning model (LLaMA/Qwen/Mistral via `llama-cpp`).
- Based on the prompt configuration, it outputs structured Bowtie JSON.

### 4. Mermaid Visualization
- The app supports a second tab for pasting or uploading the Bowtie JSON.
- A Mermaid graph is rendered inline to visualise threats, mechanisms, barriers, and consequences.

---

