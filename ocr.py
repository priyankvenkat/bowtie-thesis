import dash
from dash import dcc, html, Input, Output, State
from dash_extensions import Mermaid
import base64
import json
import re
import io
import zipfile

app = dash.Dash(__name__)
app.title = "Bowtie JSON → Mermaid ZIP Export"


def sanitize_id(text):
    """
    String to be used as a Mermaid node ID

    Args:
        text (str): Input text.

    Returns:
        str: Safe and concise node ID.
    """

    text = re.sub(r'\W+', '_', text)
    text = text.strip('_')
    if not text or not text[0].isalpha():
        text = "id_" + text
    text = text.lower()
    # Avoid ending in reserved words like 'end'
    reserved_words = ["end", "start", "link"]
    parts = text.split('_')
    if parts and parts[-1] in reserved_words:
        parts[-1] += "_node"
    return "_".join(parts)[:40]


def safe_strip(text, fallback=""):
    """
    Strips text or returns a fallback if empty.

    Args:
        text (str): Input text.
        fallback (str): Default return value if text is empty.

    Returns:
        str: Stripped string or fallback.
    """
    return (text or "").strip() or fallback


def normalize_text_field(value, fallback="Unknown"):
    """
    Normalizes a text field to a non-empty string, converting lists and checking for null-like values.

    Args:
        value (str|list|None): Raw input value.
        fallback (str): Default fallback if value is missing or invalid.

    Returns:
        str: Cleaned string.
    """
    if value is None:
        return fallback
    if isinstance(value, list):
        text = "; ".join([str(v).strip() for v in value if isinstance(v, str)])
        return text or fallback
    elif isinstance(value, str):
        value = value.strip()
        if value.lower() in ["", "none", "null"]:
            return fallback
        return value
    return fallback

def ensure_list(val):
    """
    Converts input into a list if not already. Handles str, None, or malformed inputs.
    Also splits semicolon- or comma-separated strings.

    Args:
        val: Input value (str, list, None, or other).

    Returns:
        list: A list of cleaned strings.
    """
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    elif isinstance(val, str):
        # Split on comma or semicolon
        return [x.strip() for x in re.split(r'[;]', val) if x.strip()]
    elif val is None:
        return []
    else:
        return [str(val).strip()]

def to_mermaid(bowtie, index=0):
    """
    Converts a single Bowtie diagram (in JSON format) to a Mermaid diagram code string.

    Args:
        bowtie (dict): Bowtie JSON object.
        index (int): Index for unique node IDs.

    Returns:
        str: Mermaid syntax string.
    """
    event_label = normalize_text_field(bowtie.get("critical_event"), "Unknown Event")
    lines = [f"graph LR", f'    CE["{event_label}"]:::critical']

    # === Threats ===
    threats = bowtie.get("threats", [])
    if not threats:
        threats = [{
            "mechanism": bowtie.get("mechanism", None),
            "cause": bowtie.get("causes", []) or bowtie.get("cause", [])
        }]

    for i, threat in enumerate(threats):
        mechanism_label = normalize_text_field(threat.get("mechanism"), "None")
        causes = ensure_list(threat.get("cause") or threat.get("causes"))

        if not causes:
            cause_id = sanitize_id(f"T{i}_0_Unknown_Cause")
            mechanism_id = sanitize_id(f"M{i}_0_{mechanism_label[:10]}")
            lines.append(f'    {cause_id}["Unknown Cause"]')
            lines.append(f'    {mechanism_id}["{mechanism_label}"]')
            lines.append(f'    {cause_id} --> {mechanism_id}')
            lines.append(f'    {mechanism_id} --> CE')
        else:
            for j, cause in enumerate(causes):
                cause_id = sanitize_id(f"T{i}_{j}_{cause[:10]}")
                mechanism_id = sanitize_id(f"M{i}_{j}_{mechanism_label[:10]}")
                lines.append(f'    {cause_id}["{cause}"]')
                lines.append(f'    {mechanism_id}["{mechanism_label}"]')
                lines.append(f'    {cause_id} --> {mechanism_id}')
                lines.append(f'    {mechanism_id} --> CE')

    # === Consequences ===
    # consequences = ensure_list(bowtie.get("consequences"))

    # effects = []
    # for consequence in consequences:
    #     if isinstance(consequence, dict):
    #         desc = consequence.get("description") or consequence.get("effect") or ""
    #         effects.extend([e.strip() for e in desc.split(";") if e.strip()])
    #     elif isinstance(consequence, str):
    #         effects.append(consequence.strip())
    #     elif isinstance(consequence, list):
    #         for e in consequence:
    #             if isinstance(e, str):
    #                 effects.append(e.strip())
    #     else:
    #         effects.append(str(consequence).strip())

    consequence_raw = bowtie.get("consequences")
    effects = []

    if isinstance(consequence_raw, list):
        for item in consequence_raw:
            if isinstance(item, dict):
                desc = item.get("description") or item.get("effect") or ""
                effects.extend([x.strip() for x in re.split(r'[;]', desc) if x.strip()])
            elif isinstance(item, str):
                effects.extend([x.strip() for x in re.split(r'[;]', item) if x.strip()])
            else:
                effects.append(str(item).strip())
    elif isinstance(consequence_raw, str):
        effects.extend([x.strip() for x in re.split(r'[;]', consequence_raw) if x.strip()])
    elif isinstance(consequence_raw, dict):
        desc = consequence_raw.get("description") or consequence_raw.get("effect") or ""
        effects.extend([x.strip() for x in re.split(r'[;]', desc) if x.strip()])


    # === Barriers ===
    threat_barriers = []
    for threat in threats:
        b = threat.get("preventive_barriers") or threat.get("mitigative_barriers")
        threat_barriers.extend(ensure_list(b))

    global_barriers = ensure_list(bowtie.get("mitigative_barriers")) + ensure_list(bowtie.get("preventive_barriers"))
    barriers = threat_barriers if threat_barriers else global_barriers
    if not barriers:
        barriers = ["None"]

    # === Diagram edges for consequences ===
    for k, effect in enumerate(effects):
        effect_id = sanitize_id(f"C{k}_{effect[:10]}")
        barrier_label = barriers[k % len(barriers)]
        barrier_id = sanitize_id(f"MB_{k}_{barrier_label[:10]}")
        lines.append(f'    CE --> {barrier_id}["{barrier_label}"]')
        lines.append(f'    {barrier_id} --> {effect_id}["{effect}"]')

    return "\n".join(lines)

# Layout
app.layout = html.Div([
    html.H2("📁 Drop or Paste Bowtie JSON"),
    dcc.Upload(
        id='upload-json',
        children=html.Div(['Drag and Drop or ', html.A('Select a JSON File')]),
        style={
            'width': '100%', 'padding': '20px', 'borderWidth': '2px',
            'borderStyle': 'dashed', 'borderRadius': '10px', 'textAlign': 'center'
        },
        multiple=False
    ),
    html.H4("📝 Or paste JSON content below:"),
    dcc.Textarea(
        id='paste-json',
        placeholder='Paste raw JSON content here...',
        style={'width': '100%', 'height': 200, 'marginTop': '10px'}
    ),
    html.Button("Render Diagrams", id="process-button", n_clicks=0, style={"marginTop": "10px"}),
    html.Button("⬇️ Download All Diagrams as ZIP", id="download-zip-button", n_clicks=0, style={"marginTop": "10px", "marginLeft": "10px"}),
    dcc.Download(id="download-zip"),
    dcc.Store(id="stored-diagrams", data=[]),
    html.Div(id='output-container', style={"marginTop": "30px"})
])

def unwrap_bowtie_json(data):
    """
    Extracts a list of Bowtie diagram entries from potentially nested JSON structures.

    Args:
        data (dict | list): Parsed JSON content.

    Returns:
        list | None: List of Bowtie entries or None if format unrecognized.
    """
    if isinstance(data, dict):
        if "nodes" in data and isinstance(data["nodes"], list):
            return data["nodes"]
        elif "bowtie_diagram" in data and isinstance(data["bowtie_diagram"], dict):
            if "critical_events" in data["bowtie_diagram"]:
                return data["bowtie_diagram"]["critical_events"]
        elif "bowties" in data and isinstance(data["bowties"], list):
            return data["bowties"]
        elif "diagrams" in data and isinstance(data["diagrams"], list):
            return data["diagrams"]
        elif "Bowtie" in data and isinstance(data["Bowtie"], list):  # ✅ handle "Bowtie" capitalized key
            return data["Bowtie"]
        elif any(k in data for k in ["critical_event", "cause", "mechanism"]):
            return [data]  # single diagram dict
    elif isinstance(data, list):
        return data
    return None

# Callback for processing upload or pasted content
@app.callback(
    Output('output-container', 'children'),
    Output('stored-diagrams', 'data'),
    Input("process-button", "n_clicks"),
    State('upload-json', 'contents'),
    State('paste-json', 'value'),
    prevent_initial_call=True
)
def process_input(n_clicks, upload_contents, pasted_value):
    """
    Callback to process uploaded or pasted JSON content and convert to Mermaid diagrams.

    Inputs:
        n_clicks (int): Button click count.
        upload_contents (str): Base64-encoded content from file upload.
        pasted_value (str): Raw JSON string pasted manually.

    Outputs:
        children (list): Mermaid diagram components to display.
        data (list): Mermaid code and filenames for ZIP download.
    """
    try:
        if upload_contents:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            data = json.loads(decoded)
        elif pasted_value:
            data = json.loads(pasted_value)
        else:
            return html.Div("❌ No input found."), []
    except Exception as e:
        return html.Div(f"❌ Error parsing JSON: {e}"), []

    data = unwrap_bowtie_json(data)
    if not isinstance(data, list):
        return html.Div("❌ Invalid JSON format. Expected a list or recognized structure."), []

    diagrams = []
    mermaid_codes = []

    for i, entry in enumerate(data):
        try:
            diagram_code = to_mermaid(entry, i)
            # fname = sanitize_id(entry.get('critical_event') or entry.get("name") or entry.get("id", f"diagram_{i+1}"))[:25]
            # fname = f"{i+1:02d}_" + sanitize_id(entry.get('critical_event') or entry.get("name") or entry.get("id", f"diagram_{i+1}"))[:25]
            fname_raw = entry.get('critical_event') or entry.get("name") or entry.get("id", f"diagram_{i+1}")
            fname = f"{i+1:02d}_" + sanitize_id(normalize_text_field(fname_raw))[:25]

            mermaid_codes.append({
                "filename": f"{i+1:02d}_{fname}.mmd",
                "code": diagram_code
            })
            diagrams.append(html.Div([
                html.H4(f"Diagram {i+1}: {entry.get('critical_event') or entry.get('name') or entry.get('id', 'Unknown Event')}"),
                Mermaid(chart=diagram_code)
            ], style={"marginBottom": "40px", "border": "1px solid #ddd", "padding": "10px", "borderRadius": "10px"}))
        except Exception as e:
            diagrams.append(html.Div(f"❌ Failed to render diagram {i+1}: {e}"))

    return diagrams, mermaid_codes

# ZIP download
@app.callback(
    Output("download-zip", "data"),
    Input("download-zip-button", "n_clicks"),
    State("stored-diagrams", "data"),
    prevent_initial_call=True
)
def download_zip(n_clicks, diagrams):
    """
    Callback to package all Mermaid diagrams as a ZIP archive and trigger download.

    Inputs:
        n_clicks (int): Download button click count.
        diagrams (list): List of dicts containing filenames and Mermaid code.

    Output:
        data: ZIP file as downloadable bytes.
    """
    if not diagrams:
        return dash.no_update

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w") as zf:
        for diagram in diagrams:
            zf.writestr(diagram["filename"], diagram["code"])
    memory_file.seek(0)

    return dcc.send_bytes(memory_file.read(), "bowtie_mermaid_diagrams.zip")

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8055)
