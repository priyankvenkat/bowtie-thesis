
## Files & Roles

### `sobol Run1.ipynb`
Notebook that runs analysis and visualizes first-order and total-order Sobol sensitivity indices for Run 1 (baseline).

### `Sobol Run2 1024.ipynb`
Notebook that processes and visualizes Sobol indices from Run 2 (with 1024 samples). Used to analyze how different parameters (prompt type, strictness, context) affect output variability.

### `detached sobol run.py`
Main script for running the full Sobol experiment (used for both Run 1 and 2). Supports:
- Configurable prompt type
- Context injection
- Dynamic seed selection
- Model cycling

Produces Bowtie JSONs for each model/sample combo.

### `stochastic experiment.py`
Standalone script to evaluate how model outputs vary with different seeds and fixed prompts.
- Change `PART_NAME`, number of seeds, or `RAG_CONTEXT` to switch experiments.
- Runs top-2 prompt settings per model repeatedly.

### `Stochastic Experiment.ipynb`
Analysis notebook for evaluating JSON output variability across seeds, with metrics and plots (e.g., number of unique outputs, failure rates, node mismatch).

---

## Prompt Templates

Two folders exist outside this directory that are required:

### `prompt_templates/`
Used for structured table-based input. Folders are structured as:
```
prompt_templates/
├── cot/
│   ├── low.txt
│   ├── medium.txt
│   └── high.txt
├── few/
│   └── ...
└── zero/
    └── ...
```

### `prompt_templates_narrative/`
Same structure, but designed for free-text or paragraph-style input.

