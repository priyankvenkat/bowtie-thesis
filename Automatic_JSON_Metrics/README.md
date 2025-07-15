# Bowtie Evaluation Dashboard

A Dash-based interface for visualising, validating, and scoring automatically generated Bowtie diagrams against ground-truth references. Supports role-aware matching, node/edge metrics, and manual override.

---

### Demo
https://github.com/user-attachments/assets/e4526a28-2836-4455-95d1-b4aa39bace46

---

## 📁 File Overview

| File / Folder              | Description |
|----------------------------|-------------|
| `app.py`                   | Entry point Dash app. Hosts the UI for reviewing metric results, inspecting mismatches, and rendering graphs. |
| `layout.py`                | Defines the structure of the app including tabs, tables, dropdowns, and other layout elements. |
| `callbacks.py`             | Main callback logic driving metric computation and interface interactivity. |
| `callback_review_tab.py`   | Generates the review tab UI with tables and side-by-side comparison. |
| `callback_manual_match.py` | Enables manual node matching when automatic alignment fails or is ambiguous. |
| `callback_node_override.py`| Lets users correct role or label mismatches for individual nodes. |

---

## 🧰 Utility Modules: `utils/`

| File              | Description |
|-------------------|-------------|
| `file_parser.py`     | Normalises input prediction/GT Bowtie JSON into a standard schema. |
| `event_matcher.py`   | Aligns `critical_event` nodes between prediction and ground-truth, using strict or semantic strategies. |
| `node_matcher.py`    | Matches individual nodes across files, considering label similarity and role type. |
| `metrics.py`         | Computes precision, recall, F1, Jaccard at both node and edge levels, broken down by role. |
| `graph_builder.py`   | Converts Bowtie JSONs to directed graphs for Graph Edit Distance (GED) comparison. |
| `excel_exporter.py`  | Outputs all metrics (node, edge, GED, match logs) into structured Excel workbooks. |
| `batch_evaluator.py` | Automates comparison of folders of prediction vs GT files. Generates bulk metrics. |

---

## 🧠 Core Features

- **Automatic Evaluation**
  - Computes node-level and edge-level accuracy (F1, precision, recall, Jaccard)
  - Computes graph-based similarity (GED) between predicted and GT diagrams

- **Manual Correction**
  - Supports UI-driven node matching override
  - Visual indicators for unmatched, missing, or mismatched nodes/edges

- **Excel Export**
  - Each run exports detailed logs into multi-sheet `.xlsx` files
  - Includes CE-level summaries and breakdown by role (cause, consequence, etc.)

---
