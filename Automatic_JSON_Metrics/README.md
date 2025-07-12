
**Contents:**

- **app.py**: Entry point Dash app to visualise automatic metric results and review mismatches.

**Callbacks & UI Modules:**

- **callbacks.py**: Main Dash callbacks orchestrating metric calculations and UI updates.
- **callback_manual_match.py**: Handles manual override of node matches in the review tab.
- **callback_node_override.py**: Allows users to correct individual node alignments via the interface.
- **callback_review_tab.py**: Displays detailed comparison tables and highlights mismatches.
- **layout.py**: Defines the Dash app layout (tabs, tables, graphs).

**Utility Package (utils):**
Contains files that help the above key modules:
- **batch_evaluator.py**: Runs batch comparisons between ground-truth and prediction JSON folders, computing node and edge metrics programmatically.
- **file_parser.py**: Parses and normalises input JSON files into a consistent internal format for evaluation.
- **node_matcher.py**: Implements semantic and strict matching algorithms to align predicted nodes with ground-truth nodes.
- **metrics.py**: Computes node-level (precision, recall, F1, Jaccard) and edge-level metrics based on matched entities.
- **graph_builder.py**: Converts JSON Bowtie representations into graph structures (e.g., NetworkX) for GED computation.
- **excel_exporter.py**: Exports computed metrics and detailed logs into Excel workbooks with multiple sheets.
- **event_matcher.py**: Specialised matcher for aligning critical events between prediction and ground-truth sets.

