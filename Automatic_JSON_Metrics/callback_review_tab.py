# callback_review_tab.py
from dash import Input, Output, State, html, dcc, callback
import dash_bootstrap_components as dbc
from collections import defaultdict

"""
Controls UI rendering for the review and override tab.

- Displays matched and unmatched node analysis per critical event.
- Shows prediction and GT content side-by-side.
- Integrates with callback_node_ui and node analysis stores.

Enhances review workflow after initial evaluation.
"""


def register_review_tab_callbacks(app):

    @app.callback(
        Output("review-tab-content", "children"),
        Input("node-analysis-store", "data"),
        prevent_initial_call=True
    )
    def render_review_tab(node_analysis):
        """
        Displays matched, unmatched GT, and unmatched predicted nodes per critical event.

        Args:
            node_analysis (dict): CE-wise node match analysis (GT nodes, Pred nodes, matches).

        Returns:
            html.Div: UI blocks summarizing match status for each CE.
        """
        if not node_analysis:
            return html.Div("\u26A0\uFE0F No node match analysis available yet.")

        review_blocks = []

        for ce, data in node_analysis.items():
            gt_nodes = data["gt_nodes"]
            pred_nodes = data["pred_nodes"]
            matched = data.get("node_matches", {})
            unmatched_gt = data.get("unmatched_gt", [])
            unmatched_pred = data.get("unmatched_pred", [])

            grouped = defaultdict(list)
            for label, role in unmatched_gt:
                grouped[f"GT MISSING ({role})"].append(label)
            for label, role in unmatched_pred:
                grouped[f"PRED EXTRA ({role})"].append(label)
            for gt_label, pred_label in matched.items():

                grouped[f"MATCHED ({next((r for l, r in gt_nodes if l == gt_label), 'unknown')})"].append(f"{gt_label} → {pred_label}")

            ce_block = [html.H5(f"Review: {ce}")]
            for category, labels in grouped.items():
                ce_block.append(html.Div([
                    html.Strong(category),
                    html.Ul([html.Li(label) for label in labels], className="mb-2")
                ]))

            review_blocks.append(html.Div(ce_block, className="mb-4 p-3 border rounded"))

        return html.Div(review_blocks)
