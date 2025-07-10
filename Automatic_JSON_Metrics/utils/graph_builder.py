def build_graph(entry, is_prediction=False):
    """
    Builds a Bowtie graph from a normalized entry.

    Args:
        entry (dict): Normalized entry with keys:
            - critical_event, causes, mechanisms, preventive_barriers, consequences
        is_prediction (bool): True for LLM outputs, False for strict GT

    Returns:
        dict: {
            "nodes": [(label, role)],
            "edges": [((src_label, src_role), (tgt_label, tgt_role))]
        }
    """
    nodes = []
    edges = []

    ce = entry["critical_event"].strip()
    causes = [c.strip() for c in entry.get("causes", []) if c.strip()]
    mechanisms = [m.strip() for m in entry.get("mechanisms", []) if m.strip()]
    barriers = [b.strip() for b in entry.get("preventive_barriers", []) if b.strip()]
    consequences = [c.strip() for c in entry.get("consequences", []) if c.strip()]

    # --- Add nodes (distinct by label + role) ---
    nodes.extend([(c, "cause") for c in causes])
    nodes.extend([(m, "mechanism") for m in mechanisms])
    nodes.append((ce, "critical_event"))
    nodes.extend([(b, "barrier") for b in barriers])
    nodes.extend([(c, "consequence") for c in consequences])

    # --- Cause → Mechanism → CE ---
    if is_prediction:
        if causes and mechanisms:
            for c in causes:
                for m in mechanisms:
                    edges.append(((c, "cause"), (m, "mechanism")))
            for m in mechanisms:
                edges.append(((m, "mechanism"), (ce, "critical_event")))
        elif causes:
            for c in causes:
                edges.append(((c, "cause"), (ce, "critical_event")))
        elif mechanisms:
            for m in mechanisms:
                edges.append(((m, "mechanism"), (ce, "critical_event")))
    else:
        if len(causes) != len(mechanisms):
            raise ValueError(f"GT mismatch: {len(causes)} causes vs {len(mechanisms)} mechanisms for CE: {ce}")
        for c, m in zip(causes, mechanisms):
            edges.append(((c, "cause"), (m, "mechanism")))
            edges.append(((m, "mechanism"), (ce, "critical_event")))

    # --- CE → Barrier ---
    for b in barriers:
        edges.append(((ce, "critical_event"), (b, "barrier")))

    # --- Barrier → Consequence ---
    if is_prediction:
        if barriers and consequences:
            for b in barriers:
                for c in consequences:
                    edges.append(((b, "barrier"), (c, "consequence")))
        elif consequences:
            for c in consequences:
                edges.append(((ce, "critical_event"), (c, "consequence")))
    else:
        if len(barriers) != len(consequences):
            raise ValueError(f"GT mismatch: {len(barriers)} barriers vs {len(consequences)} consequences for CE: {ce}")
        for b, c in zip(barriers, consequences):
            edges.append(((b, "barrier"), (c, "consequence")))

    return {
        "nodes": sorted(set(nodes)),
        "edges": sorted(set(edges))
    }

# from copy import deepcopy
# """
# Constructs Bowtie diagrams as NetworkX graphs from JSON data.

# - Builds nodes and directional edges according to role-based schema.
# - Handles both GT and predicted JSON formats.
# - Normalizes roles and edges (e.g., cause → mechanism → CE → barrier → consequence).

# Used by:
# - callback_node_analysis.py
# - callback_node_override.py
# - callbacks.py
# """
# def build_graph(entry, is_prediction=False):
#     """
#     Builds a Bowtie graph from a normalized entry.

#     Args:
#         entry (dict): Normalized entry with keys:
#             - critical_event, causes, mechanisms, preventive_barriers, consequences
#         is_prediction (bool): True for LLM outputs, False for strict GT

#     Returns:
#         dict: { "nodes": [(label, role)], "edges": [(source, target)] }
#     """
#     nodes = []
#     edges = []

#     ce = entry["critical_event"].strip()
#     causes = [c.strip() for c in entry.get("causes", []) if c.strip()]
#     mechanisms = [m.strip() for m in entry.get("mechanisms", []) if m.strip()]
#     barriers = [b.strip() for b in entry.get("preventive_barriers", []) if b.strip()]
#     consequences = [c.strip() for c in entry.get("consequences", []) if c.strip()]

#     # --- Add nodes ---
#     nodes.extend([(c, "cause") for c in causes])
#     nodes.extend([(m, "mechanism") for m in mechanisms])
#     nodes.append((ce, "critical_event"))
#     nodes.extend([(b, "barrier") for b in barriers])
#     nodes.extend([(c, "consequence") for c in consequences])

#     # --- Cause → Mechanism → CE ---
#     if is_prediction:
#         if causes and mechanisms:
#             if len(causes) == len(mechanisms):
#                 for c, m in zip(causes, mechanisms):
#                     edges.append((c, m))
#                     edges.append((m, ce))
#             elif len(mechanisms) == 1:
#                 for c in causes:
#                     edges.append((c, mechanisms[0]))
#                     edges.append((mechanisms[0], ce))
#             elif len(causes) == 1:
#                 for m in mechanisms:
#                     edges.append((causes[0], m))
#                     edges.append((m, ce))
#             else:
#                 # fallback: cause → CE
#                 for c in causes:
#                     edges.append((c, ce))
#         elif causes:
#             for c in causes:
#                 edges.append((c, ce))
#         elif mechanisms:
#             for m in mechanisms:
#                 edges.append((m, ce))
#     else:
#         if len(causes) != len(mechanisms):
#             raise ValueError(f"GT mismatch: {len(causes)} causes vs {len(mechanisms)} mechanisms for CE: {ce}")
#         for c, m in zip(causes, mechanisms):
#             edges.append((c, m))
#             edges.append((m, ce))

#     # --- CE → Barrier ---
#     for b in barriers:
#         edges.append((ce, b))

#     # --- Barrier → Consequence ---
#     if is_prediction:
#         if barriers and consequences:
#             if len(barriers) == len(consequences):
#                 for b, c in zip(barriers, consequences):
#                     edges.append((b, c))
#             elif len(barriers) == 1:
#                 for c in consequences:
#                     edges.append((barriers[0], c))
#             elif len(consequences) == 1:
#                 for b in barriers:
#                     edges.append((b, consequences[0]))
#             else:
#                 for c in consequences:
#                     edges.append((ce, c))  # fallback: CE → consequence
#         elif consequences:
#             for c in consequences:
#                 edges.append((ce, c))
#     else:
#         if len(barriers) != len(consequences):
#             raise ValueError(f"GT mismatch: {len(barriers)} barriers vs {len(consequences)} consequences for CE: {ce}")
#         for b, c in zip(barriers, consequences):
#             edges.append((b, c))

#     return {
#         "nodes": sorted(set(nodes)),
#         "edges": sorted(set(edges))
#     }

