# Bowtie Thesis

## Project Folder Information

The repository is organised into the following major folders and files, each corresponding to a component of the overall workflow:

- **Automatic_JSON_Metrics**: Scripts and data for automatically computing node, edge, and GED metrics from predicted and ground-truth JSON Bowtie diagrams.  
- **Dual LLM Pipeline**: Vision+text dual-LLM workflows (e.g., Pixtral image-to-markdown plus LLM Bowtie JSON generation).  
- **Manual_Bowtie_Metrics**: Code and templates for manually matching and reviewing Bowtie nodes and edges for expert evaluation.  
- **Neo4js Pipeline**: Prototype code for extracting SPO triples from text via vision LLM, storing in Neo4j, and generating Bowtie JSON by querying the graph.  
- **OCR Pipeline**: Integration of OCR (e.g., PaddleOCR or pdfplumber) to extract text and tables from scanned PDFs/images for downstream LLM prompts.  
- **RAG Pipeline**: Components for chunking, FAISS indexing, and the Dash app that retrieves context and runs LLMs to generate Bowtie JSON.  
- **Sobol Code**: Scripts to perform Sobol sensitivity analysis (Run 1 & 2) and stochasticity experiments, narrative experiments, varying prompts, context, and model parameters.  
- **Dockerfile**: This file defines the project's Docker image, installs dependencies, and sets up the runtime environment.  
- **docker-compose.yml**: Orchestrates multi-container setups (e.g., web app, Neo4j database) for development and deployment.  
- **JSON_to_Bowtie.py**: Utility script to convert Bowtie JSON outputs into Mermaid diagrams.  
- **model_selection.py**: Script for CLD generation and hallucination testing, extracting causal loops from narrative text.  

Within each folder, there is an additional README file specific to the folder's contents and what each file is responsible for. Where possible, a video of the output is added as a reference. 
