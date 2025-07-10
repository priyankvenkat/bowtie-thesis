
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

model = SentenceTransformer("all-MiniLM-L6-v2")

def token_overlap(a, b):
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


def role_aware_node_match(gt_nodes, pred_nodes, current_gt_ce=None, ce_mapping=None, ce_manual_map=None, manual_node_map=None, threshold=0.6):
    """
    Matches GT nodes to predicted nodes using:
    - manual overrides (CE and node)
    - CE-level remapping (from substring/SBERT match)
    - exact match > token overlap > SBERT similarity

    Returns:
        Tuple[
            Dict[(str, str), (str, str)],  # GT (label, role) → Pred (label, role)
            List[Tuple[str, str]],         # unmatched GT nodes
            List[Tuple[str, str]]          # unmatched Pred nodes
        ]
    """
    matches = {}
    unmatched_gt = []
    unmatched_pred = set(pred_nodes)

    pred_by_role = defaultdict(list)
    for label, role in pred_nodes:
        if label.strip().lower() not in ("", "none", "null", "n/a"):
            pred_by_role[role].append((label, role))

    for gt_label, gt_role in gt_nodes:
        # Manual node override
        if manual_node_map and gt_label in manual_node_map:
            override_pred = manual_node_map[gt_label]
            if (override_pred, gt_role) in pred_nodes:
                matches[(gt_label, gt_role)] = (override_pred, gt_role)
                unmatched_pred.discard((override_pred, gt_role))
                continue

        # Critical event remapping
        if gt_role == "critical_event" and current_gt_ce:
            for pred_label, pred_role in pred_nodes:
                if pred_role == "critical_event":
                    if ce_manual_map and ce_manual_map.get(pred_label.strip().lower()) == current_gt_ce.strip():
                        matches[(gt_label, gt_role)] = (pred_label, pred_role)
                        unmatched_pred.discard((pred_label, pred_role))
                        break
                    elif current_gt_ce.strip().lower() == ce_mapping.get(pred_label.strip().lower(), "").strip().lower():
                        matches[(gt_label, gt_role)] = (pred_label, pred_role)
                        unmatched_pred.discard((pred_label, pred_role))
                        break
            if (gt_label, gt_role) in matches:
                continue

        # Match by role (exact > token overlap > SBERT)
        best_match = None
        best_score = 0
        match_type = None

        for pred_label, pred_role in pred_by_role.get(gt_role, []):
            if (pred_label, pred_role) in matches.values():
                continue 

            if pred_label.strip().lower() == gt_label.strip().lower():
                best_match = (pred_label, pred_role)
                match_type = "exact"
                break

            tok_score = token_overlap(gt_label, pred_label)
            if tok_score >= 0.5:
                best_match = (pred_label, pred_role)
                match_type = f"token_overlap ({tok_score:.2f})"
                break

            sbert_score = util.cos_sim(
                model.encode(gt_label, convert_to_tensor=True),
                model.encode(pred_label, convert_to_tensor=True)
            ).item()

            if sbert_score > best_score and sbert_score >= threshold:
                best_match = (pred_label, pred_role)
                best_score = sbert_score
                match_type = f"sbert ({sbert_score:.2f})"

        if best_match:
            matches[(gt_label, gt_role)] = best_match
            unmatched_pred.discard(best_match)
            print(f"✅ Matched '{gt_label}' ↔ '{best_match[0]}' via {match_type}")
        else:
            unmatched_gt.append((gt_label, gt_role))

    unmatched_pred = list(unmatched_pred)
    return matches, unmatched_gt, unmatched_pred

# from sentence_transformers import SentenceTransformer, util
# from collections import defaultdict

# # Load SBERT once
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def role_aware_node_match(gt_nodes, pred_nodes, current_gt_ce=None, ce_mapping=None, ce_manual_map=None, manual_node_map=None, threshold=0.7):
#     """
#     Matches GT nodes to predicted nodes using:
#     - manual overrides (CE and node)
#     - CE-level remapping (from substring/SBERT match)
#     - SBERT semantic similarity by role

#     Args:
#         gt_nodes (List[Tuple[str, str]]): List of (label, role) from GT
#         pred_nodes (List[Tuple[str, str]]): List of (label, role) from prediction
#         current_gt_ce (str): Ground truth CE (matched)
#         ce_manual_map (dict): optional manual CE mapping (pred_ce.lower() → gt_ce)
#         manual_node_map (dict): optional manual node overrides
#         threshold (float): SBERT cosine similarity threshold

#     Returns:
#         Tuple[
#             Dict[(str, str), (str, str)],  # GT (label, role) → Pred (label, role)
#             List[Tuple[str, str]],         # unmatched GT nodes
#             List[Tuple[str, str]]          # unmatched Pred nodes
#         ]
#     """
#     matches = {}
#     unmatched_gt = []
#     unmatched_pred = set(pred_nodes)

#     # Group predicted labels by role, skipping 'none'
#     pred_by_role = defaultdict(list)
#     for label, role in pred_nodes:
#         if label.strip().lower() not in ("", "none", "null", "n/a"):
#             pred_by_role[role].append((label, role))

#     for gt_label, gt_role in gt_nodes:
#         # Priority 1: manual node override
#         if manual_node_map and gt_label in manual_node_map:
#             override_pred = manual_node_map[gt_label]
#             if (override_pred, gt_role) in pred_nodes:
#                 matches[(gt_label, gt_role)] = (override_pred, gt_role)
#                 unmatched_pred.discard((override_pred, gt_role))
#                 continue

#         # Priority 2: CE node remapping (manual or automatic)
#         if gt_role == "critical_event" and current_gt_ce:
#             for pred_label, pred_role in pred_nodes:
#                 if pred_role == "critical_event":
#                     if ce_manual_map and ce_manual_map.get(pred_label.strip().lower()) == current_gt_ce.strip():
#                         matches[(gt_label, gt_role)] = (pred_label, pred_role)
#                         unmatched_pred.discard((pred_label, pred_role))
#                         break
#                     elif current_gt_ce.strip().lower() == ce_mapping.get(pred_label.strip().lower(), "").strip().lower():
#                         matches[(gt_label, gt_role)] = (pred_label, pred_role)
#                         unmatched_pred.discard((pred_label, pred_role))
#                         break
#             if (gt_label, gt_role) in matches:
#                 continue

#         # Priority 3: SBERT similarity within the same role
#         best_match = None
#         best_score = 0

#         for pred_label, pred_role in pred_by_role.get(gt_role, []):
#             score = util.cos_sim(
#                 model.encode(gt_label, convert_to_tensor=True),
#                 model.encode(pred_label, convert_to_tensor=True)
#             ).item()

#             if score > best_score and score >= threshold:
#                 best_match = (pred_label, pred_role)
#                 best_score = score

#         if best_match:
#             matches[(gt_label, gt_role)] = best_match
#             unmatched_pred.discard(best_match)
#         else:
#             unmatched_gt.append((gt_label, gt_role))

#     unmatched_pred = list(unmatched_pred)

#     return matches, unmatched_gt, unmatched_pred



#####################

# from sentence_transformers import SentenceTransformer, util
# from collections import defaultdict

# # Load SBERT once
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def role_aware_node_match(gt_nodes, pred_nodes, current_gt_ce=None, ce_mapping=None, ce_manual_map=None, manual_node_map=None, threshold=0.7):
#     """
#     Matches GT nodes to predicted nodes using:
#     - manual overrides (CE and node)
#     - CE-level remapping (from substring/SBERT match)
#     - SBERT semantic similarity by role

#     Args:
#         gt_nodes (List[Tuple[str, str]]): List of (label, role) from GT
#         pred_nodes (List[Tuple[str, str]]): List of (label, role) from prediction
#         current_gt_ce (str): Ground truth CE (matched)
#         ce_manual_map (dict): optional manual CE mapping (pred_ce.lower() → gt_ce)
#         manual_node_map (dict): optional manual node overrides
#         threshold (float): SBERT cosine similarity threshold

#     Returns:
#         Tuple[Dict[str, str], List[Tuple[str, str]], List[Tuple[str, str]]]:
#             - GT label → predicted label
#             - unmatched GT nodes
#             - unmatched predicted nodes
#     """
#     matches = {}
#     unmatched_gt = []
#     unmatched_pred = list(pred_nodes)

#     # Group predicted labels by role, skipping 'none'
#     pred_by_role = defaultdict(list)
#     for label, role in pred_nodes:
#         if label.strip().lower() not in ("", "none", "null", "n/a"):
#             pred_by_role[role].append(label)

#     for gt_label, gt_role in gt_nodes:
#         # Priority 1: manual node override
#         if manual_node_map and gt_label in manual_node_map:
#             override_pred = manual_node_map[gt_label]
#             if override_pred in pred_by_role.get(gt_role, []):
#                 matches[gt_label] = override_pred
#                 unmatched_pred = [p for p in unmatched_pred if p[0] != override_pred]
#                 continue

#         # Priority 2: CE node remapping (manual or automatic)

#         if gt_role == "critical_event" and current_gt_ce:
#             for pred_label, pred_role in pred_nodes:
#                 if pred_role == "critical_event":
#                     if ce_manual_map and ce_manual_map.get(pred_label.strip().lower()) == current_gt_ce.strip():
#                         matches[gt_label] = pred_label
#                         unmatched_pred = [p for p in unmatched_pred if p[0] != pred_label]
#                         break
#                     elif current_gt_ce.strip().lower() == ce_mapping.get(pred_label.strip().lower(), "").strip().lower():
#                         matches[gt_label] = pred_label
#                         unmatched_pred = [p for p in unmatched_pred if p[0] != pred_label]
#                         break
#             if gt_label in matches:
#                 continue

#         # Priority 3: SBERT similarity within the same role
#         best_match = None
#         best_score = 0

#         for pred_label in pred_by_role.get(gt_role, []):
#             score = util.cos_sim(
#                 model.encode(gt_label, convert_to_tensor=True),
#                 model.encode(pred_label, convert_to_tensor=True)
#             ).item()

#             if score > best_score and score >= threshold:
#                 best_match = pred_label
#                 best_score = score

#         if best_match:
#             matches[gt_label] = best_match

#             unmatched_pred = [p for p in unmatched_pred if p[0] != best_match]
#         else:
#             unmatched_gt.append((gt_label, gt_role))

#     # 🔍 Debug logging
#     # print("🔍 GT nodes:", gt_nodes)
#     # print("🔍 Predicted nodes:", pred_nodes)
#     # print("✅ Matched nodes:", matches)
#     # print("❌ Unmatched GT nodes:", unmatched_gt)
#     # print("⚠️  Unmatched Predicted nodes:", unmatched_pred)

#     return matches, unmatched_gt, unmatched_pred
