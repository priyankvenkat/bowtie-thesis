import os
import base64
import json
import pandas as pd
import time
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from collections import Counter, defaultdict

from utils.file_parser import normalize_json_file
from utils.event_matcher import match_critical_events
from utils.node_matcher import role_aware_node_match
from utils.graph_builder import build_graph
from utils.metrics import compute_node_metrics, compute_edge_metrics, compute_ged
from utils.excel_exporter import export_metrics_to_excel

def parse_uploaded_json(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return json.loads(decoded.decode('utf-8'))


def merge_entries_by_critical_event(entries):
    grouped = defaultdict(lambda: {
        "critical_event": "",
        "causes": set(),
        "mechanisms": set(),
        "preventive_barriers": set(),
        "consequences": set()
    })

    for entry in entries:
        ce = entry["critical_event"].strip().lower()
        grouped[ce]["critical_event"] = entry["critical_event"].strip()

        grouped[ce]["causes"].update(entry.get("causes", []))
        grouped[ce]["mechanisms"].update(entry.get("mechanisms", []))
        grouped[ce]["preventive_barriers"].update(entry.get("preventive_barriers", []))
        grouped[ce]["consequences"].update(entry.get("consequences", []))

    merged = []
    for ce_key, e in grouped.items():
        merged.append({
            "critical_event": e["critical_event"],
            "causes": sorted(e["causes"]),
            "mechanisms": sorted(e["mechanisms"]),
            "preventive_barriers": sorted(e["preventive_barriers"]),
            "consequences": sorted(e["consequences"])
        })
    return merged


def deduplicate_predictions_if_needed(entries):
    ce_counts = Counter([e["critical_event"].strip().lower() for e in entries])
    if any(count > 1 for count in ce_counts.values()):
        print("🔄 Detected duplicate critical events — merging prediction entries...")
        return merge_entries_by_critical_event(entries)
    else:
        print("✅ No CE duplicates — using raw prediction entries.")
        return entries

def register_callbacks(app):

    @app.callback(
        Output("node-metrics-table", "data"),
        Output("edge-metrics-table", "data"),
        Output("ged-metrics-table", "data"),
        Output("download-link", "children"),
        
        Input("upload-gt", "contents"),
        Input("upload-preds", "contents"),
        Input("manual-match-store", "data"),
        
        State("upload-gt", "filename"),
        State("upload-preds", "filename"),
        State("model", "value"),
        State("method", "value"),
        State("prompt", "value"),
        State("domain", "value")
    )

    def handle_uploaded_files(
        gt_content,
        pred_contents,
        manual_match_data,
        gt_name,
        pred_names,
        selected_model,
        selected_method,
        selected_prompt,
        selected_domain
    ):
        if not gt_content or not pred_contents:
            raise PreventUpdate

        if isinstance(pred_contents, str):
            pred_contents = [pred_contents]
        if isinstance(pred_names, str):
            pred_names = [pred_names]

        if not pred_names or len(pred_names) != len(pred_contents):
            pred_names = [f"prediction_{i+1}.json" for i in range(len(pred_contents))]
            print("⚠️ pred_names missing or mismatched — using fallback names:", pred_names)

        selected_model = str(selected_model or "Unknown")
        selected_method = str(selected_method or "Unknown")
        selected_prompt = str(selected_prompt or "Unknown")
        selected_domain = str(selected_domain or "Unknown")

        gt_data = parse_uploaded_json(gt_content)
        with open("temp_gt.json", "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)
        gt_entries = normalize_json_file("temp_gt.json")

        node_rows, edge_rows, ged_rows = [], [], []

        for i, contents in enumerate(pred_contents):
            pred_data = parse_uploaded_json(contents)
            temp_file = f"temp_pred_{i}.json"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(pred_data, f, indent=2)

            pred_entries = normalize_json_file(temp_file)
            pred_entries = deduplicate_predictions_if_needed(pred_entries)
            file_label = os.path.basename(pred_names[i]).replace(".json", "")

            # 🔁 Match CEs
            ce_match_info = match_critical_events(gt_entries, pred_entries)
            ce_mapping = ce_match_info["matched"]
            unmatched_gt = ce_match_info["unmatched_gt"]
            unmatched_preds = ce_match_info["unmatched_preds"]

            if manual_match_data:
                ce_mapping.update(manual_match_data)

            # ✅ Process matched pairs
            for pred_ce_raw in pred_entries:
                # pred_ce = pred_ce_raw["critical_event"].strip().lower()

                # if pred_ce not in ce_mapping:
                #     continue

                # gt_ce = ce_mapping[pred_ce]

                # Normalize the predicted CE from this entry
                raw_pred_ce = pred_ce_raw["critical_event"].strip()
                normalized_pred_ce = raw_pred_ce.lower()

                # Use exact match if possible
                gt_ce = ce_mapping.get(normalized_pred_ce)

                # Fallback: try fuzzy match if no exact key
                if not gt_ce:
                    # Try manually scanning
                    for k, v in ce_mapping.items():
                        if raw_pred_ce.lower() == k:
                            gt_ce = v
                            break
                        elif raw_pred_ce.strip().lower() in k or k in raw_pred_ce.strip().lower():
                            gt_ce = v
                            break

                # If still not found, skip
                if not gt_ce:
                    print(f"⚠️ Skipping unmatched predicted CE: {raw_pred_ce}")
                    continue

                gt_struct = next(
                    e for e in gt_entries
                    if e["critical_event"].strip().lower() == gt_ce.strip().lower()
                )
                pred_struct = pred_ce_raw

                g_gt = build_graph(gt_struct, is_prediction=False)
                g_pred = build_graph(pred_struct, is_prediction=True)

                node_matches, _, _ = role_aware_node_match(
                    gt_nodes=g_gt["nodes"],
                    pred_nodes=g_pred["nodes"],
                    current_gt_ce=gt_ce,
                    ce_manual_map=manual_match_data or {},
                    ce_mapping=ce_mapping,
                    manual_node_map=None,
                    threshold=0.7
                )

                node_metrics = compute_node_metrics(g_gt, g_pred, file_label, gt_ce, node_matches)
                edge_metrics = compute_edge_metrics(g_gt, g_pred, file_label, gt_ce, node_matches)
                ged_score = compute_ged(g_gt, g_pred, node_matches)

                for row in node_metrics + edge_metrics:
                    row["Model"] = selected_model
                    row["Pipeline"] = selected_method
                    row["Prompt"] = selected_prompt
                    row["Part"] = selected_domain
                    row["Critical Event"] = gt_ce
                    row["Predicted CE"] = pred_ce_raw["critical_event"]

                node_rows.extend(node_metrics)
                edge_rows.extend(edge_metrics)

                ged_rows.append({
                    "File": file_label,
                    "Critical Event": gt_ce,
                    "Predicted CE": pred_ce_raw["critical_event"],
                    "GED": ged_score,
                    "Model": selected_model,
                    "Pipeline": selected_method,
                    "Prompt": selected_prompt,
                    "Part": selected_domain
                })

            # ❌ Handle unmatched GT CEs (False Negatives)
            for gt_ce in unmatched_gt:
                gt_struct = next(
                    e for e in gt_entries
                    if e["critical_event"].strip().lower() == gt_ce.strip().lower()
                )
                g_gt = build_graph(gt_struct, is_prediction=False)
                g_dummy = {"nodes": [], "edges": []}

                node_metrics = compute_node_metrics(g_gt, g_dummy, file_label, gt_ce, {})
                edge_metrics = compute_edge_metrics(g_gt, g_dummy, file_label, gt_ce, {})

                for row in node_metrics + edge_metrics:
                    row["Model"] = selected_model
                    row["Pipeline"] = selected_method
                    row["Prompt"] = selected_prompt
                    row["Part"] = selected_domain
                    row["Critical Event"] = gt_ce
                    row["Predicted CE"] = "None"

                node_rows.extend(node_metrics)
                edge_rows.extend(edge_metrics)

                ged_rows.append({
                    "File": file_label,
                    "Critical Event": gt_ce,
                    "Predicted CE": "None",
                    "GED": len(g_gt["nodes"]) + len(g_gt["edges"]),
                    "Model": selected_model,
                    "Pipeline": selected_method,
                    "Prompt": selected_prompt,
                    "Part": selected_domain
                })

            # ⚠️ Handle unmatched predicted CEs (False Positives)
            for pred_ce_str in unmatched_preds:
                pred_struct = next(
                    e for e in pred_entries
                    if e["critical_event"].strip() == pred_ce_str.strip()
                )
                g_pred = build_graph(pred_struct, is_prediction=True)
                g_dummy = {"nodes": [], "edges": []}

                node_metrics = compute_node_metrics(g_dummy, g_pred, file_label, "None", {})
                edge_metrics = compute_edge_metrics(g_dummy, g_pred, file_label, "None", {})

                for row in node_metrics + edge_metrics:
                    row["Model"] = selected_model
                    row["Pipeline"] = selected_method
                    row["Prompt"] = selected_prompt
                    row["Part"] = selected_domain
                    row["Critical Event"] = "None"
                    row["Predicted CE"] = pred_ce_str

                node_rows.extend(node_metrics)
                edge_rows.extend(edge_metrics)

                ged_rows.append({
                    "File": file_label,
                    "Critical Event": "None",
                    "Predicted CE": pred_ce_str,
                    "GED": len(g_pred["nodes"]) + len(g_pred["edges"]),
                    "Model": selected_model,
                    "Pipeline": selected_method,
                    "Prompt": selected_prompt,
                    "Part": selected_domain
                })

        # Save Excel
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.xlsx"
        output_path = os.path.join("downloads", filename)
        export_metrics_to_excel(
            pd.DataFrame(node_rows),
            pd.DataFrame(edge_rows),
            pd.DataFrame(ged_rows),
            output_path
        )

        download_link = html.A(
            f"📥 Download Excel Results ({filename})",
            href=f"/download/{filename}",
            target="_blank"
        )

        return (
            pd.DataFrame(node_rows).to_dict("records"),
            pd.DataFrame(edge_rows).to_dict("records"),
            pd.DataFrame(ged_rows).to_dict("records"),
            download_link
        )

############################################################
# import os
# import base64
# import json
# import pandas as pd
# import time
# from dash import Input, Output, State, html
# from dash.exceptions import PreventUpdate
# from collections import Counter, defaultdict

# from utils.file_parser import normalize_json_file
# from utils.event_matcher import match_critical_events
# from utils.node_matcher import role_aware_node_match
# from utils.graph_builder import build_graph
# from utils.metrics import compute_node_metrics, compute_edge_metrics, compute_ged
# from utils.excel_exporter import export_metrics_to_excel

# def parse_uploaded_json(contents):
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)
#     return json.loads(decoded.decode('utf-8'))


# def merge_entries_by_critical_event(entries):
#     grouped = defaultdict(lambda: {
#         "critical_event": "",
#         "causes": set(),
#         "mechanisms": set(),
#         "preventive_barriers": set(),
#         "consequences": set()
#     })

#     for entry in entries:
#         ce = entry["critical_event"].strip().lower()
#         grouped[ce]["critical_event"] = entry["critical_event"].strip()

#         grouped[ce]["causes"].update(entry.get("causes", []))
#         grouped[ce]["mechanisms"].update(entry.get("mechanisms", []))
#         grouped[ce]["preventive_barriers"].update(entry.get("preventive_barriers", []))
#         grouped[ce]["consequences"].update(entry.get("consequences", []))

#     merged = []
#     for ce_key, e in grouped.items():
#         merged.append({
#             "critical_event": e["critical_event"],
#             "causes": sorted(e["causes"]),
#             "mechanisms": sorted(e["mechanisms"]),
#             "preventive_barriers": sorted(e["preventive_barriers"]),
#             "consequences": sorted(e["consequences"])
#         })
#     return merged


# def deduplicate_predictions_if_needed(entries):
#     ce_counts = Counter([e["critical_event"].strip().lower() for e in entries])
#     if any(count > 1 for count in ce_counts.values()):
#         print("🔄 Detected duplicate critical events — merging prediction entries...")
#         return merge_entries_by_critical_event(entries)
#     else:
#         print("✅ No CE duplicates — using raw prediction entries.")
#         return entries

# def register_callbacks(app):

#     @app.callback(
#         Output("node-metrics-table", "data"),
#         Output("edge-metrics-table", "data"),
#         Output("ged-metrics-table", "data"),
#         Output("download-link", "children"),
        
#         Input("upload-gt", "contents"),
#         Input("upload-preds", "contents"),
#         Input("manual-match-store", "data"),
        
#         State("upload-gt", "filename"),
#         State("upload-preds", "filename"),
#         State("model", "value"),
#         State("method", "value"),
#         State("prompt", "value"),
#         State("domain", "value")
#     )

#     def handle_uploaded_files(
#     gt_content,
#     pred_contents,
#     manual_match_data,
#     gt_name,
#     pred_names,
#     selected_model,
#     selected_method,
#     selected_prompt,
#     selected_domain
#     ):

#         if not gt_content or not pred_contents:
#             raise PreventUpdate

#         # Normalize to list
#         if isinstance(pred_contents, str):
#             pred_contents = [pred_contents]
#         if isinstance(pred_names, str):
#             pred_names = [pred_names]

#         # If pred_names is None, generate fallback names
#         if not pred_names or len(pred_names) != len(pred_contents):
#             pred_names = [f"prediction_{i+1}.json" for i in range(len(pred_contents))]
#             print("⚠️ pred_names missing or mismatched — using fallback names:", pred_names)

#         # Ensure dropdown values are strings
#         selected_model = str(selected_model or "Unknown")
#         selected_method = str(selected_method or "Unknown")
#         selected_prompt = str(selected_prompt or "Unknown")
#         selected_domain = str(selected_domain or "Unknown")

#         # Parse and normalize GT
#         gt_data = parse_uploaded_json(gt_content)
#         with open("temp_gt.json", "w", encoding="utf-8") as f:
#             json.dump(gt_data, f, indent=2)
#         gt_entries = normalize_json_file("temp_gt.json")
        
#         node_rows, edge_rows, ged_rows = [], [], []

#         for i, contents in enumerate(pred_contents):
#             pred_data = parse_uploaded_json(contents)
#             temp_file = f"temp_pred_{i}.json"
#             with open(temp_file, "w", encoding="utf-8") as f:
#                 json.dump(pred_data, f, indent=2)

#             pred_entries = normalize_json_file(temp_file)
#             # Only apply for prediction files (not GT)
#             pred_entries = deduplicate_predictions_if_needed(pred_entries)

#             file_label = os.path.basename(pred_names[i]).replace(".json", "")

#             ce_mapping, _ = match_critical_events(gt_entries, pred_entries)

#             if manual_match_data:
#                 ce_mapping.update(manual_match_data)

#             for pred_ce_raw in pred_entries:
#                 pred_ce = pred_ce_raw["critical_event"].strip().lower()
#                 if pred_ce not in ce_mapping:
#                     continue
#                 gt_ce = ce_mapping[pred_ce]


#                 # gt_struct = next(e for e in gt_entries if e["critical_event"].lower() == gt_ce)
#                 gt_struct = next(
#                     e for e in gt_entries
#                     if e["critical_event"].strip().lower() == gt_ce.strip().lower()
#                 )

#                 pred_struct = pred_ce_raw

#                 g_gt = build_graph(gt_struct, is_prediction=False)
#                 g_pred = build_graph(pred_struct, is_prediction=True)

#                 node_matches, _, _ = role_aware_node_match(
#                     gt_nodes=g_gt["nodes"],
#                     pred_nodes=g_pred["nodes"],
#                     current_gt_ce=gt_ce,
#                     ce_manual_map=manual_match_data or {},
#                     ce_mapping=ce_mapping,
#                     manual_node_map=None,
#                     threshold=0.7
#                 )

#                 node_metrics = compute_node_metrics(g_gt, g_pred, file_label, gt_ce, node_matches)
#                 edge_metrics = compute_edge_metrics(g_gt, g_pred, file_label, gt_ce, node_matches)
#                 ged_score = compute_ged(g_gt, g_pred, node_matches)

#                 for row in node_metrics + edge_metrics:
#                     row["Model"] = selected_model
#                     row["Pipeline"] = selected_method
#                     row["Prompt"] = selected_prompt
#                     row["Part"] = selected_domain
#                     row["Critical Event"] = gt_ce
#                     row["Predicted CE"] = pred_ce_raw["critical_event"]

#                 node_rows.extend(node_metrics)
#                 edge_rows.extend(edge_metrics)

#                 ged_rows.append({
#                     "File": file_label,
#                     "Critical Event": gt_ce,
#                     "Predicted CE": pred_ce_raw["critical_event"],
#                     "GED": ged_score,
#                     "Model": selected_model,
#                     "Pipeline": selected_method,
#                     "Prompt": selected_prompt,
#                     "Part": selected_domain
#                 })

#         # Save Excel
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         filename = f"metrics_{timestamp}.xlsx"
#         output_path = os.path.join("downloads", filename)
#         export_metrics_to_excel(
#             pd.DataFrame(node_rows),
#             pd.DataFrame(edge_rows),
#             pd.DataFrame(ged_rows),
#             output_path
#         )

#         download_link = html.A(
#             f"📥 Download Excel Results ({filename})",
#             href=f"/download/{filename}",
#             target="_blank"
#         )

#         return (
#             pd.DataFrame(node_rows).to_dict("records"),
#             pd.DataFrame(edge_rows).to_dict("records"),
#             pd.DataFrame(ged_rows).to_dict("records"),
#             download_link
#         )
