# callback_node_override.py
from dash import Input, Output, State, html, dcc, MATCH, ALL, ctx, callback
import dash_bootstrap_components as dbc
from utils.file_parser import parse_contents
from utils.graph_builder import build_graph
from utils.node_matcher import role_aware_node_match

"""
Provides additional manual override tools for unmatched nodes.

- Displays dropdown menus for any unmatched nodes.
- Uses role-aware matching to guide user decisions.
- Enables the "Submit Node Match" button when needed.

Relies on:
- graph_builder.py
- node_matcher.py
- file_parser.py

"""


def register_node_override_callbacks(app):
    """
    Registers callbacks for manual dropdown-based node override logic.
    """

    @app.callback(
        Output("node-match-dropdown-metadata", "data"),
        Input("submit-manual-match", "n_clicks"),
        State("upload-gt", "contents"),
        State("upload-pred", "contents"),
        State("manual-match-store", "data"),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def update_dropdown_metadata(n_clicks, gt_content, pred_content, manual_matches):
        """
        Generates dropdown options for each unmatched GT node per CE.

        Args:
            n_clicks (int): Number of times submit clicked.
            gt_content (str): Base64 GT file content.
            pred_content (str): Base64 prediction file content.
            manual_matches (dict): Optional CE-level manual mappings.

        Returns:
            list: Dropdown metadata per unmatched GT node, including role and matching Pred options.
        """
    
        if not gt_content or not pred_content:
            return []

        gt_data = parse_contents(gt_content)
        pred_data = parse_contents(pred_content)

        gt_map = {entry["critical_event"]: entry for entry in gt_data}
        pred_map = {entry["critical_event"]: entry for entry in pred_data}

        dropdown_metadata = []
        matched_ces = {}
        for gt_ce in gt_map:
            for pred_ce in pred_map:
                if gt_ce.lower() == pred_ce.lower():
                    matched_ces[gt_ce] = pred_ce
        for pred_ce, gt_ce in (manual_matches or {}).items():
            matched_ces[gt_ce] = pred_ce

        for gt_ce, pred_ce in matched_ces.items():
            gt_graph = build_graph(gt_map[gt_ce], is_gt=True)
            pred_graph = build_graph(pred_map[pred_ce], is_gt=False)
            gt_nodes = gt_graph["nodes"]
            pred_nodes = pred_graph["nodes"]
            node_matches, unmatched_gt, unmatched_pred = role_aware_node_match(gt_nodes, pred_nodes)

            for i, (gt_node, gt_role) in enumerate(unmatched_gt):
                pred_options = [p for p in pred_nodes if p[1] == gt_role]
                option_labels = [label for label, _ in pred_options]
                dropdown_metadata.append({
                    "ce": gt_ce,
                    "index": i,
                    "label": gt_node,
                    "role": gt_role,
                    "options": option_labels
                })

        return dropdown_metadata

    @app.callback(
        Output("show-submit-node-button", "data"),
        Input("node-match-dropdown-metadata", "data"),
        allow_duplicate=True
    )
    def update_button_visibility(dropdown_metadata):
        """
        Shows or hides the Submit button based on whether unmatched GT nodes exist.

        Args:
            dropdown_metadata (list): List of GT node match UI data.

        Returns:
            bool: Whether to show the submit button.
        """
        return True if dropdown_metadata else False

