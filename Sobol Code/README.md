# Sobol Sensitivity, Stochasticity Experiments and Narrative Input Experiments

This directory contains scripts and notebooks to evaluate how prompt parameters, context type, and random seeds affect Bowtie diagram generation. Two primary analyses are supported:

- **Sobol sensitivity analysis** (Run 1 & Run 2)
- **Stochasticity experiments** (seed variability & narrative input) 

---

## 📁 Files & Roles

| File                          | Description |
|-------------------------------|-------------|
| `sobol Run1.ipynb`            | Notebook that computes and plots first-order and total-order Sobol indices for Run 1. |
| `Sobol Run2 1024.ipynb`       | Extended notebook for analysing Sobol indices using 1024 samples. Focuses on prompt type, strictness, and context effects. |
| `detached sobol run.py`       | Full pipeline for running the Sobol experiment. Dynamically cycles models, seeds, and prompt configurations. Produces Bowtie JSON outputs. |
| `stochastic experiment.py`    | Script for evaluating output variability across seeds. Runs each model on top-2 prompt settings repeatedly. Customizable part/context/seed count. |
| `Stochastic Experiment.ipynb` | Notebook that analyses variability in Bowtie JSONs generated from multiple seeds. Includes plots, stats, and error tracking. |

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

---
