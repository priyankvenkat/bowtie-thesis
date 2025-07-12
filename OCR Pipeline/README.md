
The app runs at: [http://localhost:8052](http://localhost:8052)

This single Dash script handles:

- Layout and tabbed UI for OCR and Mermaid rendering
- Image/PDF ingestion and table parsing
- Prompt construction and LLM inference
- Mermaid code generation from Bowtie JSON
- Support for model/seed configuration

## Inputs

- Images (JPG, PNG) or PDF files with tabular data
- Manually entered part name
- Custom column header overrides per file

## Outputs

- Markdown and extracted raw text from uploaded files
- Bowtie JSON(s) per model and seed
- Mermaid diagrams rendered from the Bowtie JSON
