from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np

# Load model once
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def token_overlap(a, b):
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


def match_critical_events(gt_entries, pred_entries, threshold=0.6):
    """
    Globally optimal 1-to-1 matching of predicted CEs to GT CEs via SBERT + Hungarian algorithm.
    Optionally filters out weak matches by similarity threshold.

    Args:
        gt_entries (List[dict])
        pred_entries (List[dict])
        threshold (float): Minimum SBERT similarity to accept match.

    Returns:
        dict:
            - matched: Dict[pred_ce.lower()] → gt_ce (original casing)
            - unmatched_preds: List[str]
            - unmatched_gt: List[str]
    """
    gt_ces_raw = [e["critical_event"].strip() for e in gt_entries]
    pred_ces_raw = [e["critical_event"].strip() for e in pred_entries]

    if not gt_ces_raw or not pred_ces_raw:
        return {"matched": {}, "unmatched_preds": pred_ces_raw, "unmatched_gt": gt_ces_raw}

    gt_embs = sbert_model.encode(gt_ces_raw, convert_to_tensor=True)
    pred_embs = sbert_model.encode(pred_ces_raw, convert_to_tensor=True)

    sim_matrix = util.cos_sim(pred_embs, gt_embs).cpu().numpy()  # [pred_i][gt_j]

    # Convert similarity to cost (for minimization)
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = {}
    unmatched_preds = []
    used_gt = set()

    for pred_idx, gt_idx in zip(row_ind, col_ind):
        sim_score = sim_matrix[pred_idx][gt_idx]
        pred = pred_ces_raw[pred_idx].strip()
        gt = gt_ces_raw[gt_idx].strip()

        if sim_score >= threshold:
            matched[pred.lower()] = gt
            used_gt.add(gt)
            print(f"✅ Matched: '{pred}' → '{gt}' (score: {sim_score:.2f})")
        else:
            unmatched_preds.append(pred)
            print(f"❌ Rejected (score < {threshold}): '{pred}' → '{gt}' (score: {sim_score:.2f})")


    unmatched_gt = [gt for gt in gt_ces_raw if gt not in used_gt]


    # Step 2: Substring fallback for remaining unmatched predictions
    for pred in [p for p in pred_ces_raw if p.lower() not in matched]:
        for gt in unmatched_gt:
            if pred.lower() in gt.lower() or gt.lower() in pred.lower():
                matched[pred.lower()] = gt
                unmatched_gt.remove(gt)
                print(f"🔁 Substring fallback: '{pred}' → '{gt}'")
                break

    return {
        "matched": matched,
        "unmatched_preds": unmatched_preds,
        "unmatched_gt": unmatched_gt
    }

# from sentence_transformers import SentenceTransformer, util
# import torch

# # Load SBERT model once
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# def match_critical_events(gt_entries, pred_entries, threshold=0.6):
#     """
#     Greedy 1:1 matching of predicted CEs to GT CEs via:
#     1. Substring match (case-insensitive)
#     2. SBERT cosine similarity (fallback)

#     Args:
#         gt_entries (List[dict])
#         pred_entries (List[dict])
#         threshold (float): SBERT similarity threshold

#     Returns:
#         dict with keys:
#             - matched: Dict[pred_ce.lower()] → gt_ce (original casing)
#             - unmatched_preds: List[str] (original casing)
#             - unmatched_gt: List[str] (original casing)
#     """
#     gt_ces_raw = [e["critical_event"].strip() for e in gt_entries]
#     gt_ces_lc = [ce.lower() for ce in gt_ces_raw]
#     gt_map = dict(zip(gt_ces_lc, gt_ces_raw))  # lc → original
#     unused_gt = set(gt_ces_lc)

#     pred_ces_raw = [e["critical_event"].strip() for e in pred_entries]
#     pred_ces_lc = [ce.lower() for ce in pred_ces_raw]

#     matched = {}
#     unmatched = []

#     def token_overlap(a, b):
#         a_tokens = set(a.lower().split())
#         b_tokens = set(b.lower().split())
#         if not a_tokens or not b_tokens:
#             return 0
#         overlap = a_tokens & b_tokens
#         return len(overlap) / max(len(a_tokens), len(b_tokens))


#     # Step 1: Substring match
#     for pred_raw, pred in zip(pred_ces_raw, pred_ces_lc):
#         best_sub = None
#         for gt in unused_gt:
#             score = token_overlap(pred, gt)
#             if score >= 0.5:
#                 print(f"✅ Token overlap match: '{pred}' ↔ '{gt}' (score: {score:.2f})")
#                 best_sub = gt
#                 break
#             # if pred in gt or gt in pred:
#             #     best_sub = gt
#             #     break
#         if best_sub:
#             matched[pred] = gt_map[best_sub]
#             unused_gt.remove(best_sub)

#     # Step 2: SBERT fallback for remaining unmatched preds
#     remaining_preds = [p for p in pred_ces_raw if p.strip().lower() not in matched]
#     remaining_gt = [gt_map[gt] for gt in unused_gt]

#     if remaining_preds and remaining_gt:
#         pred_embs = sbert_model.encode(remaining_preds, convert_to_tensor=True)
#         gt_embs = sbert_model.encode(remaining_gt, convert_to_tensor=True)

#         for i, pred_raw in enumerate(remaining_preds):
#             pred = pred_raw.strip().lower()
#             sims = util.cos_sim(pred_embs[i], gt_embs)[0]
#             best_idx = torch.argmax(sims).item()
#             score = sims[best_idx].item()
#             best_gt = remaining_gt[best_idx]
#             best_gt_lc = best_gt.strip().lower()

#             if score >= threshold:
#                 matched[pred] = best_gt
#                 if best_gt_lc in unused_gt:
#                     unused_gt.remove(best_gt_lc)
#                 else:
#                     print(f"⚠️ Could not remove '{best_gt_lc}' — not in unused_gt")
#                     print("🔍 Unused GT CEs:", list(unused_gt))
#                 print(f"✅ SBERT match: '{pred_raw}' → '{best_gt}' (score: {score:.2f})")
#             else:
#                 unmatched.append(pred_raw)
#                 print(f"❌ Unmatched CE: '{pred_raw}' (best score: {score:.2f})")
#     else:
#         unmatched.extend(remaining_preds)

#     # Logging for substring matches
#     for pred_lc, gt in matched.items():
#         print(f"✅ CE mapped: '{pred_lc}' → '{gt}'")

#     return {
#         "matched": matched,                         # pred_ce.lower() → gt_ce
#         "unmatched_preds": unmatched,              # unmatched predicted CEs
#         "unmatched_gt": [gt_map[gt] for gt in unused_gt]  # unmatched GT CEs (original casing)
#     }

# from sentence_transformers import SentenceTransformer, util
# import torch

# # Load SBERT model once
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# def match_critical_events(gt_entries, pred_entries, threshold=0.7):
#     """
#     Greedy 1:1 matching of predicted CEs to GT CEs via:
#     1. Substring match (case-insensitive)
#     2. SBERT cosine similarity (fallback)

#     Args:
#         gt_entries (List[dict])
#         pred_entries (List[dict])
#         threshold (float): SBERT similarity threshold

#     Returns:
#         Tuple[
#             Dict[str, str],  # pred_ce.lower() → gt_ce (original casing)
#             List[str]        # unmatched predicted CEs (original casing)
#         ]
#     """
#     gt_ces_raw = [e["critical_event"].strip() for e in gt_entries]
#     gt_ces_lc = [ce.lower() for ce in gt_ces_raw]
#     gt_map = dict(zip(gt_ces_lc, gt_ces_raw))  # lc → original
#     unused_gt = set(gt_ces_lc)

#     pred_ces_raw = [e["critical_event"].strip() for e in pred_entries]
#     pred_ces_lc = [ce.lower() for ce in pred_ces_raw]

#     matched = {}
#     unmatched = []

#     # Step 1: Substring match
#     for pred_raw, pred in zip(pred_ces_raw, pred_ces_lc):
#         best_sub = None
#         for gt in unused_gt:
#             if pred in gt or gt in pred:
#                 best_sub = gt
#                 break
#         if best_sub:
#             matched[pred] = gt_map[best_sub]
#             unused_gt.remove(best_sub)

#     # Step 2: SBERT fallback for unmatched preds
#     remaining_preds = [p for p in pred_ces_raw if p.strip().lower() not in matched]
#     remaining_gt = [gt_map[gt] for gt in unused_gt]

#     if remaining_preds and remaining_gt:
#         pred_embs = sbert_model.encode(remaining_preds, convert_to_tensor=True)
#         gt_embs = sbert_model.encode(remaining_gt, convert_to_tensor=True)

#         for i, pred_raw in enumerate(remaining_preds):
#             pred = pred_raw.strip().lower()
#             sims = util.cos_sim(pred_embs[i], gt_embs)[0]
#             best_idx = torch.argmax(sims).item()
#             score = sims[best_idx].item()
#             if score >= threshold:
#                 best_gt = remaining_gt[best_idx]
#                 matched[pred] = best_gt
#                 gt_lc_match = [k for k, v in gt_map.items() if v == best_gt]
#                 if gt_lc_match:
#                     unused_gt.remove(gt_lc_match[0])
#                 else:
#                     print(f"⚠️ Could not find lowercase key for GT: {best_gt}")
#                 # unused_gt.remove(best_gt.lower())
#                 print(f"✅ SBERT match: '{pred_raw}' → '{best_gt}' (score: {score:.2f})")
#             else:
#                 unmatched.append(pred_raw)
#                 print(f"❌ Unmatched CE: '{pred_raw}' (best score: {score:.2f})")
#     else:
#         unmatched.extend(remaining_preds)

#     # Substring match logs
#     for pred_lc, gt in matched.items():
#         print(f"✅ CE mapped: '{pred_lc}' → '{gt}'")

#     # return matched, unmatched
#     return {
#         "matched": matched,                         # pred_ce.lower() → gt_ce (original casing)
#         "unmatched_preds": unmatched,              # list of unmatched predicted CEs (original casing)
#         "unmatched_gt": [gt_map[gt] for gt in unused_gt]  # unmatched ground-truth CEs (original casing)
#     }


##############################

# def match_critical_events(gt_entries, pred_entries, threshold=0.7):
#     """
#     Matches predicted critical events to ground truth events using:
#     1. Exact or substring match
#     2. SBERT cosine similarity
#     3. Logs unmatched predictions for manual override

#     Args:
#         gt_entries (List[dict]): Normalized GT entries
#         pred_entries (List[dict]): Normalized prediction entries
#         threshold (float): Cosine similarity threshold

#     Returns:
#         Tuple[
#             Dict[str, str],  # pred_CE -> matched_gt_CE
#             List[str]        # unmatched predicted CEs
#         ]
#     """
#     gt_ces = [e["critical_event"].strip().lower() for e in gt_entries]
#     pred_ces = [e["critical_event"].strip().lower() for e in pred_entries]

#     matched = {}
#     unmatched = []

#     for pred in pred_ces:
#         # 1. Substring match
#         substr_matches = [gt for gt in gt_ces if pred in gt or gt in pred]
#         if substr_matches:
#             matched[pred] = substr_matches[0]
#             continue

#         # 2. SBERT similarity (device-safe)
#         pred_emb = sbert_model.encode(pred, convert_to_tensor=True)
#         gt_embs = sbert_model.encode(gt_ces, convert_to_tensor=True)

#         # Compute cosine similarity
#         sims = util.cos_sim(pred_emb, gt_embs)[0]

#         # Safe argmax on any device
#         best_idx = torch.argmax(sims).item()
#         best_score = sims[best_idx].item()

#         if best_score >= threshold:
#             matched[pred] = gt_ces[best_idx]
#         else:
#             unmatched.append(pred)

#     return matched, unmatched
