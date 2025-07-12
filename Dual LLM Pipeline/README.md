
Vision+text dual-LLM workflows combining image-to-markdown extraction and Bowtie JSON generation.

**Demo**

### Demo
![Demo of App](Video/Dual LLM Pipeline.gif)

**Contents:**

- **app.py**  
  Dash application entry point. Handles the main layout and routing logic with two tabs:
  - **Tab 1:** Upload image → extract table → convert to markdown + summary using vision LLM → select part name and prompt type → generate Bowtie JSON using text LLM.
  - **Tab 2:** Render Mermaid diagram from uploaded Bowtie JSON.

- **callbacks.py**  
  Defines interactive logic and response functions for:
  - Model selection (LLaMA/Qwen/Mistral)
  - Prompt dispatching to the vision API (e.g., Mistral)
  - Seed/stochasticity control
  - Rendering Mermaid diagrams and returning structured Bowtie output.

- **config.py**  
  Centralised configuration of:
  - Mistral API keys and endpoints
  - Available model paths
  - Supported prompt types and strictness levels.

- **llm_runner.py**  
  Calls local `llama-cpp` models with adjustable parameters, including:
  - Temperature
  - Top-p
  - Context size
  - Max tokens

- **parsing.py**  
  Functions to:
  - Parse and clean LLM output (e.g., markdown from vision stage)
  - Generate Mermaid diagrams from Bowtie JSON
  - Extract structured fields from markdown tables or raw text

- **prompts.py**  
  Handles logic to select and inject prompts (based on user settings from config.py) for both the vision and reasoning stages.
