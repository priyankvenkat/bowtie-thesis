import os
import pandas as pd
from tqdm import tqdm

from utils.file_parser import normalize_json_file
from utils.event_matcher import match_critical_events
from utils.graph_builder import build_graph
from utils.node_matcher import match_nodes
from utils.metrics import (
    compute_node_metrics,
    compute_edge_metrics,
    compute_ged
)
from utils.excel_exporter import export_metrics_to_excel

def evaluate_all_predictions(gt_file: str, prediction_files: list, output_excel: str):
    """
    Evaluate a batch of prediction JSONs against the ground truth and export metrics.

    Args:
        gt_file (str): Path to ground truth JSON file.
        prediction_files (List[str]): List of prediction JSON paths.
        output_excel (str): Output Excel path to save all metrics.
    """
    print("🔍 Loading and normalizing ground truth...")
    gt_entries = normalize_json_file(gt_file)

    node_rows = []
    edge_rows = []
    ged_rows = []
    unmatched_log = []

    for pred_file in tqdm(prediction_files, desc="Evaluating Predictions"):
        file_label = os.path.basename(pred_file).replace(".json", "")
        pred_entries = normalize_json_file(pred_file)

        # Match critical events
        ce_mapping, unmatched_pred_ces = match_critical_events(gt_entries, pred_entries)

        for ce in unmatched_pred_ces:
            unmatched_log.append({
                "File": file_label,
                "Unmatched Predicted CE": ce
            })

        for pred_ce_raw in pred_entries:
            pred_ce = pred_ce_raw["critical_event"].strip().lower()
            if pred_ce not in ce_mapping:
                continue

            gt_ce = ce_mapping[pred_ce]
            gt_struct = next(e for e in gt_entries if e["critical_event"].lower() == gt_ce)
            pred_struct = pred_ce_raw

            # Build graphs
            g_gt = build_graph(gt_struct)
            g_pred = build_graph(pred_struct)

            # Match nodes
            gt_node_labels = [n[0] for n in g_gt["nodes"]]
            pred_node_labels = [n[0] for n in g_pred["nodes"]]
            node_matches, _ = match_nodes(gt_node_labels, pred_node_labels)

            # Compute metrics
            node_metrics = compute_node_metrics(g_gt, g_pred, file_label, gt_ce, node_matches)
            edge_metrics = compute_edge_metrics(g_gt, g_pred, file_label, gt_ce, node_matches)
            ged_score = compute_ged(g_gt, g_pred, node_matches)

            # Store results
            node_rows.extend(node_metrics)
            edge_rows.extend(edge_metrics)
            ged_rows.append({
                "File": file_label,
                "Critical Event": gt_ce,
                "GED": ged_score
            })

    # Convert to DataFrames
    df_node = pd.DataFrame(node_rows)
    df_edge = pd.DataFrame(edge_rows)
    df_ged = pd.DataFrame(ged_rows)
    df_unmatched = pd.DataFrame(unmatched_log)

    # Export to Excel
    print("📊 Saving results to Excel...")
    export_metrics_to_excel(df_node, df_edge, df_ged, output_excel, unmatched_df=df_unmatched)
    print(f"✅ Metrics saved to {output_excel}")

# import glob
# from batch_evaluator import evaluate_all_predictions

# gt_path = "ground_truth/gt_sensor.json"
# pred_files = glob.glob("predictions/*.json")
# output_excel = "results/metrics_sensor_batch.xlsx"

# evaluate_all_predictions(gt_path, pred_files, output_excel)
