from dash import Input, Output, State, MATCH, ALL, html, dcc
import dash_bootstrap_components as dbc
import json
import base64
from utils.file_parser import normalize_json_file

def parse_uploaded_json(contents):
    """Decode base64 content to JSON."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return json.loads(decoded.decode('utf-8'))

def register_manual_match_callbacks(app):
    @ app.callback(
        Output("manual-match-section", "children"),
        Input("upload-gt", "contents"),
        Input("upload-preds", "contents")
    )
    def display_manual_match_ui(gt_content, pred_contents):
        if not gt_content or not pred_contents:
            return []

        # Parse uploaded files
        gt_data = parse_uploaded_json(gt_content)
        pred_data = parse_uploaded_json(pred_contents[0]) if isinstance(pred_contents, list) else parse_uploaded_json(pred_contents)

        # Normalize both files
        with open("temp_manual_gt.json", "w", encoding="utf-8") as f:
            json.dump(gt_data, f)
        with open("temp_manual_pred.json", "w", encoding="utf-8") as f:
            json.dump(pred_data, f)

        gt_entries = normalize_json_file("temp_manual_gt.json")
        pred_entries = normalize_json_file("temp_manual_pred.json")

        gt_ces = [e["critical_event"].strip() for e in gt_entries]
        pred_ces = [e["critical_event"].strip() for e in pred_entries]

        # Auto match: case-insensitive
        matched = set()
        auto_map = {}

        for pred in pred_ces:
            pred_norm = pred.strip().lower()
            for gt in gt_ces:
                gt_norm = gt.strip().lower()
                if pred_norm == gt_norm:
                    matched.add(pred_norm)
                    auto_map[pred] = gt
                    break  # only first match used

        # Fix: Lowercased check for unmatched
        unmatched_preds = [ce for ce in pred_ces if ce.strip().lower() not in matched]

        print("🔍 Auto-matched CE pairs:", auto_map)
        print("❌ Unmatched predicted CEs for manual dropdown:", unmatched_preds)

        if not unmatched_preds:
            return html.Div("✅ All predicted critical events matched automatically.")

        # Build dropdowns for unmatched
        dropdowns = []
        for i, pred_ce in enumerate(unmatched_preds):
            dropdowns.append(
                dbc.Row([
                    dbc.Col(html.Div(f"Prediction: {pred_ce}"), width=6),
                    dbc.Col(
                        dcc.Dropdown(
                            id={"type": "manual-ce-map", "index": i},
                            options=[{"label": gt_ce, "value": gt_ce} for gt_ce in gt_ces],
                            placeholder="Select matching GT CE"
                        ),
                        width=6
                    )
                ], className="mb-2")
            )

        return html.Div([
            html.H5("Manual Critical Event Matching"),
            html.Div(dropdowns),
            dbc.Button("Submit Manual Matches", id="submit-manual-match", color="secondary", className="mt-3"),
            html.Hr(),
            html.Div("✅ Auto-matched CEs:", className="mt-2 fw-bold"),
            html.Ul([html.Li(f"{pred} → {gt}") for pred, gt in auto_map.items()])
        ])

    @app.callback(
        Output("manual-match-store", "data"),
        Input("submit-manual-match", "n_clicks"),
        State("upload-gt", "contents"),
        State("upload-preds", "contents"),
        State({"type": "manual-ce-map", "index": ALL}, "value"),
        State({"type": "manual-ce-map", "index": ALL}, "id"),
        prevent_initial_call=True
    )
    def store_manual_match(n_clicks, gt_content, pred_contents, selected_gt_ces, ids):
        gt_data = parse_uploaded_json(gt_content)
        pred_data = parse_uploaded_json(pred_contents[0]) if isinstance(pred_contents, list) else parse_uploaded_json(pred_contents)

        with open("temp_manual_gt.json", "w", encoding="utf-8") as f:
            json.dump(gt_data, f)
        with open("temp_manual_pred.json", "w", encoding="utf-8") as f:
            json.dump(pred_data, f)

        gt_entries = normalize_json_file("temp_manual_gt.json")
        pred_entries = normalize_json_file("temp_manual_pred.json")

        gt_ces = [e["critical_event"].strip() for e in gt_entries]
        pred_ces = [e["critical_event"].strip() for e in pred_entries]

        # Auto-match
        matched = set()
        for pred in pred_ces:
            for gt in gt_ces:
                if pred.strip().lower() == gt.strip().lower():
                    matched.add(pred.strip().lower())


        unmatched_preds = [ce for ce in pred_ces if ce.strip().lower() not in matched]

        # Manual match collection (lowercase key)
        manual_map = {}
        for selected, id_info in zip(selected_gt_ces, ids):
            index = id_info.get("index")
            if index is not None and index < len(unmatched_preds):
                pred_ce = unmatched_preds[index].strip().lower()
                manual_map[pred_ce] = selected.strip()
        print("✅ Final manual CE map submitted:", manual_map)
        return manual_map
