import pandas as pd

def export_metrics_to_excel(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    df_ged: pd.DataFrame,
    output_path: str,
    unmatched_df: pd.DataFrame = None
):
    """
    Saves all evaluation metrics to an Excel file with multiple sheets.

    Args:
        df_node (pd.DataFrame): Node-level metrics.
        df_edge (pd.DataFrame): Edge-level metrics.
        df_ged (pd.DataFrame): GED scores.
        output_path (str): File path to save Excel.
        unmatched_df (pd.DataFrame, optional): Unmatched critical events (if any).
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_node.to_excel(writer, sheet_name="NodeMetrics", index=False)
        df_edge.to_excel(writer, sheet_name="EdgeMetrics", index=False)
        df_ged.to_excel(writer, sheet_name="GEDMetrics", index=False)

        if unmatched_df is not None and not unmatched_df.empty:
            unmatched_df.to_excel(writer, sheet_name="UnmatchedCEs", index=False)
