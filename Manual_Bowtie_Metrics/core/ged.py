

# def compute_ged(edges1, edges2, gt_roles=None, pred_roles=None, threshold=0.85):
#     gt_nodes = list({normalize(n) for e in edges1 for n in e})
#     pred_nodes = list({normalize(n) for e in edges2 for n in e})

#     emb_gt = model.encode(gt_nodes, convert_to_tensor=True)
#     emb_pred = model.encode(pred_nodes, convert_to_tensor=True)
#     sim_matrix = util.pytorch_cos_sim(emb_pred, emb_gt).cpu().numpy()

#     pred_to_gt = {}
#     used_gt = set()
#     mapping_scores = []

#     # for i, pred in enumerate(pred_nodes):
#     #     best_j = -1
#     #     best_score = 0
#     #     for j, gt in enumerate(gt_nodes):
#     #         # Optional: enforce role matching if provided
#     #         if gt_roles and pred_roles:
#     #             if pred_roles.get(pred_nodes[i]) != gt_roles.get(gt_nodes[j]):
#     #                 continue
#     #         sim = sim_matrix[i][j]
#     #         if sim > threshold and sim > best_score and gt not in used_gt:
#     #             best_j = j
#     #             best_score = sim

#     #     if best_j != -1:
#     #         pred_to_gt[pred_nodes[i]] = gt_nodes[best_j]
#     #         used_gt.add(gt_nodes[best_j])
#     #     else:
#     #         # ❌ unmatched prediction — don’t remap
#     #         pred_to_gt[pred_nodes[i]] = None

#     #     mapping_scores.append({
#     #         "Predicted": pred_nodes[i],
#     #         "Matched_GT": gt_nodes[best_j] if best_j != -1 else "❌ No match",
#     #         "Cosine_Similarity": round(best_score, 3)
#     #     })

#     for i, pred in enumerate(pred_nodes):
#         best_j = -1
#         best_score = 0
#         pred_label = pred_nodes[i]
#         pred_tokens = [t.strip() for t in re.split(r'[,;/]', pred_label)]

#         for j, gt in enumerate(gt_nodes):
#             gt_label = gt_nodes[j]
#             gt_tokens = [t.strip() for t in re.split(r'[,;/]', gt_label)]

#             # Skip role mismatches
#             if gt_roles and pred_roles:
#                 if pred_roles.get(pred_label) != gt_roles.get(gt_label):
#                     continue

#             # Compare every token of pred to every token of gt
#             for p_token in pred_tokens:
#                 for g_token in gt_tokens:
#                     if not p_token or not g_token:
#                         continue
#                     sim = util.pytorch_cos_sim(
#                         model.encode(p_token, convert_to_tensor=True),
#                         model.encode(g_token, convert_to_tensor=True)
#                     ).item()
#                     if sim > threshold and sim > best_score and gt_label not in used_gt:
#                         best_j = j
#                         best_score = sim

#         if best_j != -1:
#             pred_to_gt[pred_label] = gt_nodes[best_j]
#             used_gt.add(gt_nodes[best_j])
#         else:
#             pred_to_gt[pred_label] = None

#         mapping_scores.append({
#             "Predicted": pred_label,
#             "Matched_GT": gt_nodes[best_j] if best_j != -1 else "❌ No match",
#             "Cosine_Similarity": round(best_score, 3)
#         })  

#     # === Remap edges ===
#     remapped_edges = []
#     for a, b in edges2:
#         remapped_a = pred_to_gt.get(normalize(a))
#         remapped_b = pred_to_gt.get(normalize(b))
#         if remapped_a and remapped_b:
#             remapped_edges.append((normalize(remapped_a), normalize(remapped_b)))

#     norm_gt_edges = [(normalize(a), normalize(b)) for a, b in edges1]

#     G1 = nx.DiGraph(); G1.add_edges_from(norm_gt_edges)
#     G2 = nx.DiGraph(); G2.add_edges_from(remapped_edges)

#     global last_cosine_mappings
#     last_cosine_mappings = mapping_scores

#     print("📌 Remapped Predicted Edges:")
#     for e in remapped_edges:
#         print("   ", e)

#     print("📌 Ground Truth Edges:")
#     for e in norm_gt_edges:
#         print("   ", e)

#     return nx.graph_edit_distance(G1, G2)
# GED computation logic

import networkx as nx
from sentence_transformers import SentenceTransformer, util
from .graph_utils import normalize
from networkx.algorithms.similarity import optimize_graph_edit_distance
import re 

model = SentenceTransformer('all-MiniLM-L6-v2')
last_cosine_mappings = []



def normalize_roles(role_dict):
    return {normalize(k): v for k, v in role_dict.items()}

def compute_ged(edges1, edges2, gt_roles=None, pred_roles=None, threshold=0.8):
    gt_roles = normalize_roles(gt_roles or {})
    pred_roles = normalize_roles(pred_roles or {})

    gt_nodes = list({normalize(n) for e in edges1 for n in e})
    pred_nodes = list({normalize(n) for e in edges2 for n in e})

    emb_gt = model.encode(gt_nodes, convert_to_tensor=True)
    emb_pred = model.encode(pred_nodes, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(emb_pred, emb_gt).cpu().numpy()

    pred_to_gt = {}
    used_gt = set()
    mapping_scores = []

    for i, pred_label in enumerate(pred_nodes):
        best_j = -1
        best_score = 0.0

        for j, gt_label in enumerate(gt_nodes):
            # Skip if role mismatch
            if gt_roles and pred_roles:
                if pred_roles.get(pred_label) != gt_roles.get(gt_label):
                    continue

            sim = sim_matrix[i][j]
            if sim > threshold and sim > best_score and gt_label not in used_gt:
                best_j = j
                best_score = sim

        if best_j != -1:
            pred_to_gt[pred_label] = gt_nodes[best_j]
            used_gt.add(gt_nodes[best_j])
        else:
            pred_to_gt[pred_label] = None

        mapping_scores.append({
            "Predicted": pred_label,
            "Matched_GT": gt_nodes[best_j] if best_j != -1 else "❌ No match",
            "Cosine_Similarity": round(best_score, 3),
            "RoleMatch": pred_roles.get(pred_label) == gt_roles.get(gt_nodes[best_j]) if best_j != -1 else False
        })

    remapped_edges = []
    for a, b in edges2:
        remapped_a = pred_to_gt.get(normalize(a))
        remapped_b = pred_to_gt.get(normalize(b))
        if remapped_a and remapped_b:
            remapped_edges.append((normalize(remapped_a), normalize(remapped_b)))

    norm_gt_edges = [(normalize(a), normalize(b)) for a, b in edges1]

    G1 = nx.DiGraph(); G1.add_edges_from(norm_gt_edges)
    G2 = nx.DiGraph(); G2.add_edges_from(remapped_edges)

    global last_cosine_mappings
    last_cosine_mappings = mapping_scores

    return nx.graph_edit_distance(G1, G2)

def get_last_cosine_mappings():
    return last_cosine_mappings

