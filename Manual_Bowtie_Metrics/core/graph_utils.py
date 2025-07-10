# normalize(), role helpers

import re

# def normalize(text):
#     if not isinstance(text, str):
#         return ""

#     text = text.lower().strip()

#     # Replace common delimiters with a comma
#     text = text.replace("/", ",")
#     text = text.replace(";", ",")

#     # Remove non-alphanumeric except space and comma
#     text = re.sub(r'[^a-z0-9, ]+', '', text)

#     # Collapse multiple spaces
#     text = re.sub(r'\s+', ' ', text)

#     return text.strip()

def normalize(text):
    import re
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()

    # Replace delimiters with commas
    text = text.replace("/", ",").replace(";", ",")

    # Normalize comma spacing (remove space after/before, then re-insert one space)
    text = re.sub(r'\s*,\s*', ',', text)

    # Remove all characters except a-z, 0-9, spaces, and commas
    text = re.sub(r'[^a-z0-9, ]+', '', text)

    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Final trim
    return text.strip()



# def suggest_roles(edges):
#     in_deg = {}
#     out_deg = {}
#     for a, b in edges:
#         out_deg[a] = out_deg.get(a, 0) + 1
#         in_deg[b] = in_deg.get(b, 0) + 1
#     all_nodes = set(in_deg.keys()) | set(out_deg.keys())
#     role_map = {}
#     for n in all_nodes:
#         indeg = in_deg.get(n, 0)
#         outdeg = out_deg.get(n, 0)
#         if indeg == 0 and outdeg > 0:
#             role = "Cause"
#         elif indeg > 0 and outdeg == 0:
#             role = "Consequence"
#         elif indeg > 0 and outdeg > 0:
#             role = "Mechanism"
#         else:
#             role = ""
#         role_map[n] = role
#     return role_map

def suggest_roles(edges):
    from collections import defaultdict

    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    incoming = defaultdict(set)
    outgoing = defaultdict(set)

    for a, b in edges:
        out_deg[a] += 1
        in_deg[b] += 1
        outgoing[a].add(b)
        incoming[b].add(a)

    all_nodes = set(in_deg) | set(out_deg)
    role_map = {}

    for node in all_nodes:
        indeg = in_deg[node]
        outdeg = out_deg[node]
        ins = incoming[node]
        outs = outgoing[node]

        # 💥 Likely Central Event: connects both sides and appears in middle
        if indeg > 0 and outdeg > 0:
            # Heuristic: central event is node where left (cause) and right (consequence) converge
            if any(pred not in all_nodes for pred in ins) or any(succ not in all_nodes for succ in outs):
                role = "Mechanism"
            else:
                # Try to guess central if it's the only shared node in both incoming & outgoing
                role = "Critical Event"

        # ✅ Cause: only outgoing edges
        elif indeg == 0 and outdeg > 0:
            role = "Cause"

        # ✅ Consequence: only incoming edges
        elif indeg > 0 and outdeg == 0:
            role = "Consequence"

        else:
            role = "Barrier"  # could also be undefined, depending on context

        role_map[node] = role

    return role_map
