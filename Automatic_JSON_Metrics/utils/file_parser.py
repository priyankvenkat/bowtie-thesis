import json
import base64
import io

"""
Handles upload parsing and normalization of JSON structure.

- Decodes uploaded base64 file contents.
- Parses JSON into structured format.
- Normalizes prediction format to ensure compatibility.

Used by:
- callbacks.py
- callback_node_analysis.py
- callback_manual_match.py
"""


def parse_contents(contents):
    """
    Parses base64-encoded content from Dash Upload into JSON.

    Args:
        contents (str): Base64-encoded string from upload.

    Returns:
        dict/list: Parsed JSON object or list of entries.
    """
    import base64
    import io

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        data = json.load(io.StringIO(decoded.decode('utf-8')))
        return data
    except Exception as e:
        print("Error parsing JSON:", e)
        return None

def normalize_json_file(filepath):
    """
    Loads and normalizes a JSON file containing Bowtie prediction entries.
    Returns a list of dictionaries, one per critical event.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to parse {filepath}: {e}")
        return []

    if isinstance(data, dict):
        data = [data]

    normalized = []

    for entry in data:
        if not isinstance(entry, dict):
            continue

        norm = {
            "critical_event": entry.get("critical_event", "").strip(),
            "causes": [],
            "mechanisms": [],
            "preventive_barriers": [],
            "consequences": []
        }

        # --- Handle threat-style nested structure ---
        if "threats" in entry:
            for threat in entry["threats"]:
                # Causes
                causes = threat.get("cause", [])
                if isinstance(causes, str):
                    # causes = [c.strip() for c in causes.replace(",", ";").split(";")]
                    causes = [c.strip() for c in causes.split(";")]

                elif isinstance(causes, list):
                    causes = [c.strip() for c in causes if isinstance(c, str)]
                norm["causes"].extend(causes)

                # Mechanism
                mech = threat.get("mechanism", "")
                if isinstance(mech, str) and mech.strip().lower() not in ("", "none", "null", "n/a"):
                    norm["mechanisms"].append(mech.strip())

                # Barrier
                barrier = threat.get("preventive_barriers", [])
                if isinstance(barrier, str):
                    barrier = [b.strip() for b in barrier.replace(",", ";").split(";")]
                elif isinstance(barrier, list):
                    barrier = [b.strip() for b in barrier if isinstance(b, str)]
                norm["preventive_barriers"].extend(barrier)

        else:
            # Causes
            causes = entry.get("cause", entry.get("causes", []))
            if isinstance(causes, str):
                # causes = [c.strip() for c in causes.replace(",", ";").split(";")]
                causes = [c.strip() for c in causes.split(";")]

            elif isinstance(causes, list):
                causes = [c.strip() for c in causes if isinstance(c, str)]
            norm["causes"].extend(causes)

            # Mechanisms
            mechanisms = entry.get("mechanism", entry.get("mechanisms", []))
            if isinstance(mechanisms, str):
                mechanisms = [m.strip() for m in mechanisms.replace(",", ";").split(";")]
            elif isinstance(mechanisms, list):
                mechanisms = [m.strip() for m in mechanisms if isinstance(m, str)]
            else:
                mechanisms = []
            norm["mechanisms"].extend([
                m for m in mechanisms if m.lower() not in ("", "none", "null", "n/a")
            ])

            # Barriers
            barriers = entry.get("preventive_barriers", [])
            if isinstance(barriers, str):
                barriers = [b.strip() for b in barriers.replace(",", ";").split(";")]
            elif isinstance(barriers, list):
                barriers = [b.strip() for b in barriers if isinstance(b, str)]
            else:
                barriers = []
            norm["preventive_barriers"].extend(barriers)

        # Consequences
        cons = entry.get("consequence", entry.get("consequences", []))
        if isinstance(cons, str):
            # cons = [c.strip() for c in cons.replace(",", ";").split(";")]
            cons = [c.strip() for c in cons.split(";")]

        elif isinstance(cons, list):
            cons = [c.strip() for c in cons if isinstance(c, str)]
        else:
            cons = []
        norm["consequences"].extend(cons)

        normalized.append(norm)
        print("✅ Normalized entry:")
        print(json.dumps(norm, indent=2))

    return normalized

# def normalize_json_file(filepath):
#     """
#     Loads and normalizes a JSON file containing Bowtie prediction entries.
#     Returns a list of dictionaries, one per critical event.
#     """
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"❌ Failed to parse {filepath}: {e}")
#         return []

#     if isinstance(data, dict):
#         data = [data]

#     normalized = []

#     for entry in data:
#         if not isinstance(entry, dict):
#             continue

#         norm = {
#             "critical_event": entry.get("critical_event", "").strip(),
#             "causes": [],
#             "mechanisms": [],
#             "preventive_barriers": [],
#             "consequences": []
#         }

#         # --- Handle threat-style nested structure ---
#         if "threats" in entry:
#             for threat in entry["threats"]:
#                 # Causes
#                 causes = threat.get("cause", [])
#                 if isinstance(causes, str):
#                     causes = [c.strip() for c in causes.replace(",", ";").split(";")]
#                 elif isinstance(causes, list):
#                     causes = [c.strip() for c in causes if isinstance(c, str)]
#                 norm["causes"].extend(causes)

#                 # Mechanism
#                 mech = threat.get("mechanism", "")
#                 if isinstance(mech, str) and mech.strip().lower() not in ("", "none", "null", "n/a"):
#                     norm["mechanisms"].append(mech.strip())

#                 # Barrier
#                 barrier = threat.get("preventive_barriers", [])
#                 if isinstance(barrier, str):
#                     norm["preventive_barriers"].append(barrier.strip())
#                 elif isinstance(barrier, list):
#                     norm["preventive_barriers"].extend([b.strip() for b in barrier if isinstance(b, str)])

#         else:
#             # Causes
#             causes = entry.get("cause", entry.get("causes", []))
#             if isinstance(causes, str):
#                 causes = [c.strip() for c in causes.replace(",", ";").split(";")]
#             elif isinstance(causes, list):
#                 causes = [c.strip() for c in causes if isinstance(c, str)]
#             norm["causes"].extend(causes)

#             # Mechanisms
#             mechanisms = entry.get("mechanism", entry.get("mechanisms", []))
#             if isinstance(mechanisms, str):
#                 mechanisms = [mechanisms.strip()]
#             elif not isinstance(mechanisms, list):
#                 mechanisms = []
#             else:
#                 mechanisms = [m.strip() for m in mechanisms if isinstance(m, str)]

#             # Filter only truly empty values
#             norm["mechanisms"].extend([
#                 m for m in mechanisms if m.lower() not in ("", "none", "null", "n/a")
#             ])

#             # Barriers
#             barrier = entry.get("preventive_barriers", [])
#             if isinstance(barrier, str):
#                 norm["preventive_barriers"].append(barrier.strip())
#             elif isinstance(barrier, list):
#                 norm["preventive_barriers"].extend([b.strip() for b in barrier if isinstance(b, str)])

#         # Consequences
#         cons = entry.get("consequence", entry.get("consequences", []))
#         if isinstance(cons, str):
#             norm["consequences"].append(cons.strip())
#         elif isinstance(cons, list):
#             norm["consequences"].extend([c.strip() for c in cons if isinstance(c, str)])

#         normalized.append(norm)
#         print("✅ Normalized entry:")
#         print(json.dumps(norm, indent=2))

#     return normalized

