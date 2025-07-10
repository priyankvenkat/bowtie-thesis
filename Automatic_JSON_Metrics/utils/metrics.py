from collections import defaultdict


def compute_node_metrics(gt_graph, pred_graph, file_label, ce_label, node_matches):
    """
    Computes role-aware node-level metrics and returns row-wise results for Excel.

    Args:
        gt_graph (dict): Ground truth graph with 'nodes' (label, role).
        pred_graph (dict): Prediction graph with 'nodes' (label, role).
        file_label (str): Filename of the prediction source.
        ce_label (str): Critical Event associated.
        node_matches (dict): Mapping from (GT label, role) → (Pred label, role).

    Returns:
        List[dict]: One row per role (including 'Overall') with metrics.
    """
    role_tp = defaultdict(int)
    role_fp = defaultdict(int)
    role_fn = defaultdict(int)

    gt_nodes = gt_graph['nodes']
    pred_nodes = pred_graph['nodes']
    matched_pred_nodes = set(node_matches.values())  # set of (label, role)

    if ce_label.lower() == "seal leakage":
        print("🧠 GT causes:")
        for label, role in gt_nodes:
            if role == "cause":
                print("   GT:", label)

        print("🧠 Predicted causes:")
        for label, role in pred_nodes:
            if role == "cause":
                print("   Pred:", label)

        print("🧠 Matched:")
        for (gt_label, gt_role), (pred_label, pred_role) in node_matches.items():
            if gt_role == "cause":
                print(f"   MATCHED: {gt_label} → {pred_label}")
                
    # ✅ Role-aware TP/FN for GT nodes
    for node in gt_nodes:
        if node in node_matches:
            role_tp[node[1]] += 1
        else:
            role_fn[node[1]] += 1

    # ✅ Role-aware FP for unmatched predicted nodes
    for node in pred_nodes:
        if node not in matched_pred_nodes:
            role_fp[node[1]] += 1

    # 📊 Compute per-role metrics
    rows = []
    all_roles = set(role_tp) | set(role_fp) | set(role_fn)
    for role in sorted(all_roles):
        tp, fp, fn = role_tp[role], role_fp[role], role_fn[role]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0

        rows.append({
            "File": file_label,
            "Critical Event": ce_label,
            "Role": role.capitalize(),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3),
            "Jaccard": round(jaccard, 3),
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

    # ➕ Add Overall metrics
    tp = sum(role_tp.values())
    fp = sum(role_fp.values())
    fn = sum(role_fn.values())
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0

    rows.append({
        "File": file_label,
        "Critical Event": ce_label,
        "Role": "Overall",
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "Jaccard": round(jaccard, 3),
        "TP": tp,
        "FP": fp,
        "FN": fn
    })

    return rows

def compute_edge_metrics(gt_graph, pred_graph, file_label, ce_label, node_matches):
    """
    Computes per-edge-type metrics (e.g., Cause→Mechanism) and returns as flat rows.

    Args:
        gt_graph (dict): GT graph with 'nodes' and 'edges'.
        pred_graph (dict): Prediction graph.
        file_label (str): Prediction filename.
        ce_label (str): Critical event name.
        node_matches (dict): Mapping from (GT label, role) → (Pred label, role)

    Returns:
        List[dict]: Per edge-type + overall rows.
    """
    edge_tp = defaultdict(int)
    edge_fp = defaultdict(int)
    edge_fn = defaultdict(int)

    # ✅ Role-aware reverse mapping: pred_node → gt_node
    reverse_map = {pred: gt for gt, pred in node_matches.items()}

    # 🔄 Remap predicted edges into GT label space
    remapped_pred_edges = set()
    for u, v in pred_graph["edges"]:
        u_mapped = reverse_map.get(u, u)
        v_mapped = reverse_map.get(v, v)
        remapped_pred_edges.add((u_mapped, v_mapped))

    gt_edges = set(gt_graph["edges"])
    all_edges = gt_edges | remapped_pred_edges

    # 🔍 Extract edge types (role-to-role direction)
    edge_types = {}
    for u, v in all_edges:
        u_role = u[1] if isinstance(u, tuple) else ""
        v_role = v[1] if isinstance(v, tuple) else ""
        edge_type = f"{u_role.capitalize()}→{v_role.capitalize()}"
        edge_types[(u, v)] = edge_type

    # ✅ Count TPs and FNs
    for edge in gt_edges:
        edge_type = edge_types.get(edge, "Unknown")
        if edge in remapped_pred_edges:
            edge_tp[edge_type] += 1
        else:
            edge_fn[edge_type] += 1

    # ✅ Count FPs
    for edge in remapped_pred_edges:
        edge_type = edge_types.get(edge, "Unknown")
        if edge not in gt_edges:
            edge_fp[edge_type] += 1

    # 📊 Compile per-edge-type metrics
    rows = []
    all_edge_types = set(edge_tp.keys()) | set(edge_fp.keys()) | set(edge_fn.keys())

    for et in sorted(all_edge_types):
        tp, fp, fn = edge_tp[et], edge_fp[et], edge_fn[et]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0

        rows.append({
            "File": file_label,
            "Critical Event": ce_label,
            "Edge Type": et,
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3),
            "Jaccard": round(jaccard, 3),
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

    # ✅ Add overall summary row
    tp, fp, fn = sum(edge_tp.values()), sum(edge_fp.values()), sum(edge_fn.values())
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0

    rows.append({
        "File": file_label,
        "Critical Event": ce_label,
        "Edge Type": "Overall",
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "Jaccard": round(jaccard, 3),
        "TP": tp,
        "FP": fp,
        "FN": fn
    })

    return rows

def compute_ged(gt_graph, pred_graph, node_matches):
    """
    Approximates Graph Edit Distance as the sum of unmatched nodes and edges:
    GED = node_FP + node_FN + edge_FP + edge_FN

    Args:
        gt_graph (dict): Ground truth graph with role-tagged nodes.
        pred_graph (dict): Predicted graph with role-tagged nodes.
        node_matches (dict): Mapping from (GT label, role) → (Pred label, role)

    Returns:
        int: Approximate GED score.
    """
    # ✅ Use role-aware node tuples
    gt_nodes = set(gt_graph["nodes"])
    pred_nodes = set(pred_graph["nodes"])
    matched_pred_nodes = set(node_matches.values())

    node_fn = len([n for n in gt_nodes if n not in node_matches])
    node_fp = len([n for n in pred_nodes if n not in matched_pred_nodes])

    # ✅ Use role-aware reverse mapping
    reverse_map = {v: k for k, v in node_matches.items()}
    remapped_pred_edges = set()
    for u, v in pred_graph["edges"]:
        u_mapped = reverse_map.get(u, u)
        v_mapped = reverse_map.get(v, v)
        remapped_pred_edges.add((u_mapped, v_mapped))

    gt_edges = set(gt_graph["edges"])

    edge_fn = len([e for e in gt_edges if e not in remapped_pred_edges])
    edge_fp = len([e for e in remapped_pred_edges if e not in gt_edges])

    ged_score = node_fp + node_fn + edge_fp + edge_fn
    return ged_score



# def compute_node_metrics(gt_graph, pred_graph, file_label, ce_label, node_matches):
#     """
#     Computes role-aware node-level metrics and returns row-wise results for Excel.

#     Args:
#         gt_graph (dict): Ground truth graph with 'nodes' (label, role).
#         pred_graph (dict): Prediction graph with 'nodes' (label, role).
#         file_label (str): Filename of the prediction source.
#         ce_label (str): Critical Event associated.
#         node_matches (dict): Mapping from GT labels to predicted labels.

#     Returns:
#         List[dict]: One row per role (including 'Overall') with metrics.
#     """
#     role_tp = defaultdict(int)
#     role_fp = defaultdict(int)
#     role_fn = defaultdict(int)

#     gt_nodes = gt_graph['nodes']
#     pred_nodes = pred_graph['nodes']
#     matched_pred_nodes = set(node_matches.values())

#     for label, role in gt_nodes:
#         if label in node_matches:
#             role_tp[role] += 1
#         else:
#             role_fn[role] += 1

#     for label, role in pred_nodes:
#         if label not in matched_pred_nodes:
#             role_fp[role] += 1

#     rows = []
#     all_roles = set(list(role_tp.keys()) + list(role_fp.keys()) + list(role_fn.keys()))
#     for role in all_roles:
#         tp, fp, fn = role_tp[role], role_fp[role], role_fn[role]
#         prec = tp / (tp + fp) if (tp + fp) else 0
#         rec = tp / (tp + fn) if (tp + fn) else 0
#         f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
#         jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0

#         rows.append({
#             "File": file_label,
#             "Critical Event": ce_label,
#             "Role": role.capitalize(),
#             "Precision": round(prec, 3),
#             "Recall": round(rec, 3),
#             "F1": round(f1, 3),
#             "Jaccard": round(jaccard, 3),
#             "TP": tp,
#             "FP": fp,
#             "FN": fn
#         })

#     # Add Overall
#     tp, fp, fn = sum(role_tp.values()), sum(role_fp.values()), sum(role_fn.values())
#     prec = tp / (tp + fp) if (tp + fp) else 0
#     rec = tp / (tp + fn) if (tp + fn) else 0
#     f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
#     jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0

#     rows.append({
#         "File": file_label,
#         "Critical Event": ce_label,
#         "Role": "Overall",
#         "Precision": round(prec, 3),
#         "Recall": round(rec, 3),
#         "F1": round(f1, 3),
#         "Jaccard": round(jaccard, 3),
#         "TP": tp,
#         "FP": fp,
#         "FN": fn
#     })

#     return rows

# def compute_edge_metrics(gt_graph, pred_graph, file_label, ce_label, node_matches):
#     """
#     Computes per-edge-type metrics (e.g., Cause->Mechanism) and returns as flat rows.

#     Args:
#         gt_graph (dict): GT graph with 'nodes' and 'edges'.
#         pred_graph (dict): Prediction graph.
#         file_label (str): Prediction filename.
#         ce_label (str): Critical event name.
#         node_matches (dict): Mapping from GT to predicted node labels.

#     Returns:
#         List[dict]: Per edge-type + overall rows.
#     """
#     edge_tp = defaultdict(int)
#     edge_fp = defaultdict(int)
#     edge_fn = defaultdict(int)

#     remapped_pred_edges = set()
#     reverse_map = {v: k for k, v in node_matches.items()}

#     for u, v in pred_graph["edges"]:
#         u_mapped = reverse_map.get(u, u)
#         v_mapped = reverse_map.get(v, v)
#         remapped_pred_edges.add((u_mapped, v_mapped))

#     gt_edges = set(gt_graph["edges"])
#     all_edges = gt_edges | remapped_pred_edges

#     node_roles = {label: role for label, role in gt_graph["nodes"] + pred_graph["nodes"]}
#     edge_types = {}

#     for u, v in all_edges:
#         src_role = node_roles.get(u, "")
#         tgt_role = node_roles.get(v, "")
#         edge_type = f"{src_role.capitalize()}->{tgt_role.capitalize()}"
#         edge_types[(u, v)] = edge_type


#     for edge in gt_edges:
#         edge_type = edge_types.get(edge, "Unknown")
#         if edge in remapped_pred_edges:
#             edge_tp[edge_type] += 1
#         else:
#             edge_fn[edge_type] += 1

#     for edge in remapped_pred_edges:
#         edge_type = edge_types.get(edge, "Unknown")
#         if edge not in gt_edges:
#             edge_fp[edge_type] += 1

#     rows = []
#     all_edge_types = set(edge_tp.keys()) | set(edge_fp.keys()) | set(edge_fn.keys())

#     for et in all_edge_types:
#         tp, fp, fn = edge_tp[et], edge_fp[et], edge_fn[et]
#         prec = tp / (tp + fp) if (tp + fp) else 0
#         rec = tp / (tp + fn) if (tp + fn) else 0
#         f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
#         jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0

#         rows.append({
#             "File": file_label,
#             "Critical Event": ce_label,
#             "Edge Type": et,
#             "Precision": round(prec, 3),
#             "Recall": round(rec, 3),
#             "F1": round(f1, 3),
#             "Jaccard": round(jaccard, 3),
#             "TP": tp,
#             "FP": fp,
#             "FN": fn
#         })

#     # Overall edge row
#     tp, fp, fn = sum(edge_tp.values()), sum(edge_fp.values()), sum(edge_fn.values())
#     prec = tp / (tp + fp) if (tp + fp) else 0
#     rec = tp / (tp + fn) if (tp + fn) else 0
#     f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
#     jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0

#     rows.append({
#         "File": file_label,
#         "Critical Event": ce_label,
#         "Edge Type": "Overall",
#         "Precision": round(prec, 3),
#         "Recall": round(rec, 3),
#         "F1": round(f1, 3),
#         "Jaccard": round(jaccard, 3),
#         "TP": tp,
#         "FP": fp,
#         "FN": fn
#     })

#     return rows


# def compute_ged(gt_graph, pred_graph, node_matches):
#     """
#     Approximates Graph Edit Distance as the sum of unmatched nodes and edges:
#     GED = node_FP + node_FN + edge_FP + edge_FN

#     Args:
#         gt_graph (dict): Ground truth graph.
#         pred_graph (dict): Predicted graph.
#         node_matches (dict): Mapping from GT → Pred node labels.

#     Returns:
#         int: Approximate GED.
#     """
#     gt_nodes = {label for label, _ in gt_graph["nodes"]}
#     pred_nodes = {label for label, _ in pred_graph["nodes"]}
#     matched_pred_nodes = set(node_matches.values())

#     node_fn = len([n for n in gt_nodes if n not in node_matches])
#     node_fp = len([n for n in pred_nodes if n not in matched_pred_nodes])

#     reverse_map = {v: k for k, v in node_matches.items()}
#     remapped_pred_edges = set()
#     for u, v in pred_graph["edges"]:
#         u_mapped = reverse_map.get(u, u)
#         v_mapped = reverse_map.get(v, v)
#         remapped_pred_edges.add((u_mapped, v_mapped))

#     gt_edges = set(gt_graph["edges"])

#     edge_fn = len([e for e in gt_edges if e not in remapped_pred_edges])
#     edge_fp = len([e for e in remapped_pred_edges if e not in gt_edges])

#     ged_score = node_fp + node_fn + edge_fp + edge_fn
#     return ged_score