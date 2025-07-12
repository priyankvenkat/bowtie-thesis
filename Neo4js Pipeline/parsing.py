import json

def parse_llm_output(data):
    """
    Parse the LLM output into a consistent bowtie JSON structure.
    """
    def extract_threats(threats_raw):
        threats = []
        if all(isinstance(t, dict) for t in threats_raw):
            for t in threats_raw:
                threats.append({
                    "mechanism": t.get("mechanism", "Unknown Mechanism"),
                    "cause": t.get("cause") or t.get("threat") or "Unknown Cause",
                    "preventive_barriers": t.get("preventive_barriers", [])
                })
        elif all(isinstance(t, str) for t in threats_raw):
            for t in threats_raw:
                # If only a string is provided, use it for both mechanism and cause.
                threats.append({
                    "mechanism": t,
                    "cause": t,
                    "preventive_barriers": []
                })
        return threats

    return {
        # Normalize to use "top_event" for downstream consistency.
        "top_event": data.get("top_event") or data.get("critical_event", "Unknown Event"),
        "threats": extract_threats(data.get("threats", [])),
        "consequences": data.get("consequences", []),
        "preventive_barriers": data.get("preventive_barriers", []),
        "mitigative_barriers": data.get("mitigative_barriers", [])
    }    

def generate_mermaid_from_bowtie(data):
    """
    Generate Mermaid syntax for Bowtie diagram(s) from parsed JSON data.
    If data is a list, process each diagram separately.
    """
    # Debug print to show the type of the input data.
    print("Type of data:", type(data))
    if isinstance(data, list):
        print("Data is a list of length:", len(data))
    elif isinstance(data, dict):
        print("Data is a dict.")
    else:
        print("Unexpected data type:", type(data))
    
    def escape(text):
        return str(text).replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')').replace('"', '').replace("'", '')
    
    def build_mermaid(diagram):
        # Ensure diagram is a dict.
        if not isinstance(diagram, dict):
            raise ValueError(f"Expected diagram to be a dict, but got {type(diagram)}: {diagram}")
        
        lines = ["graph LR"]
        # Use "top_event" as the key, or fall back to "critical_event"
        te_label = escape(diagram.get("top_event", diagram.get("critical_event", "Critical Event")))
        lines.append(f"  TE[{te_label}]")
        
        # Dictionaries to track created nodes for mechanisms
        mech_to_id = {}
        mech_count = 0
        threat_count = 0
        barrier_count = 0

        threats = diagram.get("threats", [])
        if not isinstance(threats, list):
            raise ValueError(f"'threats' should be a list, but got {type(threats)}: {threats}")
        
        for threat in threats:
            # Determine if threat is a dict, a list, or a simple string.
            if isinstance(threat, dict):
                mechanism = threat.get("mechanism", "Unknown Mechanism")
                cause = threat.get("cause", "Unknown Cause")
                barriers = threat.get("preventive_barriers", [])
            elif isinstance(threat, list):
                # If threat is a list, join its elements as both mechanism and cause.
                cause = " ".join(map(str, threat))
                mechanism = cause
                barriers = []
            else:
                # Assume threat is a string.
                cause = threat
                mechanism = threat
                barriers = []
            
            # Check if mechanism and cause are identical, handling None gracefully.
            if (mechanism or "").strip() == (cause or "").strip():
                threat_count += 1
                t_id = f"T{threat_count}"
                # Create a single threat node that directly connects to the top event.
                lines.append(f"  {t_id}[Threat: {escape(mechanism or 'Unknown Threat')}] --> TE")
                # Process preventive barriers attached to this threat.
                if not isinstance(barriers, list):
                    raise ValueError(f"'preventive_barriers' should be a list, but got {type(barriers)}: {barriers}")
                for barrier in barriers:
                    barrier_count += 1
                    b_id = f"PB{barrier_count}"
                    lines.append(f"  {b_id}[Barrier: {escape(barrier)}] --> {t_id}")
            else:
                # If mechanism and cause differ, process them separately.
                # Create a node for the mechanism if not already created.
                if (mechanism or "").strip() not in mech_to_id:
                    mech_count += 1
                    m_id = f"M{mech_count}"
                    mech_to_id[(mechanism or "").strip()] = m_id
                    lines.append(f"  {m_id}[Mechanism: {escape(mechanism or 'Unknown Mechanism')}] --> TE")
                else:
                    m_id = mech_to_id[(mechanism or "").strip()]
                
                threat_count += 1
                c_id = f"C{threat_count}"
                lines.append(f"  {c_id}[Cause: {escape(cause or 'Unknown Cause')}] --> {m_id}")
                
                # Process preventive barriers for the threat.
                if not isinstance(barriers, list):
                    raise ValueError(f"'preventive_barriers' should be a list, but got {type(barriers)}: {barriers}")
                for barrier in barriers:
                    barrier_count += 1
                    b_id = f"PB{barrier_count}"
                    lines.append(f"  {b_id}[Barrier: {escape(barrier)}] --> {c_id}")
        
        # Process consequences: connect each consequence to the top event.
        consequences = diagram.get("consequences", [])
        if not isinstance(consequences, list):
            raise ValueError(f"'consequences' should be a list, but got {type(consequences)}: {consequences}")
        for i, consequence in enumerate(consequences):
            cons_id = f"Cons{i+1}"
            lines.append(f"  TE --> {cons_id}[Consequence: {escape(consequence)}]")
        
        # Process mitigative barriers: connect them to the top event.
        mitigative_barriers = diagram.get("mitigative_barriers", [])
        if not isinstance(mitigative_barriers, list):
            raise ValueError(f"'mitigative_barriers' should be a list, but got {type(mitigative_barriers)}: {mitigative_barriers}")
        for i, barrier in enumerate(mitigative_barriers):
            mb_id = f"MB{i+1}"
            lines.append(f"  TE --> {mb_id}[Mitigative Barrier: {escape(barrier)}]")
        
        return "\n".join(lines)
    
    # Determine if the input data is a list or a single dictionary.
    if isinstance(data, list):
        mermaid_codes = []
        for idx, diagram in enumerate(data):
            try:
                code = build_mermaid(diagram)
                mermaid_codes.append(code)
            except Exception as e:
                mermaid_codes.append(f"Error processing diagram {idx}: {str(e)}")
        return "\n\n".join(mermaid_codes)
    elif isinstance(data, dict):
        return build_mermaid(data)
    else:
        raise ValueError("Input data is neither a dict nor a list.")


def expand_mechanism_structure(data):
    """
    Expand a nested mechanism → causes structure into a flat list for processing.
    """
    expanded_threats = []
    for item in data.get("threats", []):
        mech = item.get("mechanism", "Unknown Mechanism")
        for cause in item.get("causes", []):
            expanded_threats.append({
                "mechanism": mech,
                "threat": cause,
                "preventive_barriers": []
            })
    return {
        "critical_event": data.get("critical_event", "Unknown Event"),
        "threats": expanded_threats,
        "consequences": data.get("consequences", []),
        "preventive_barriers": data.get("preventive_barriers", []),
        "mitigative_barriers": data.get("mitigative_barriers", [])
    }