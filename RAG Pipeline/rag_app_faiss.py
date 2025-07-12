
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import base64
from pathlib import Path
from llama_cpp import Llama
import pynvml
import time 
import json5
from dash_extensions import Mermaid

# === Model Setup ===


available_models = {
    "LLaMA-3-8B": "DeepSeek-R1-GGUF/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
    # "Mistral-7B": "DeepSeek-R1-GGUF/Mistral-7B-Instruct-v0.3-Q5_K_S.gguf",
    # "Qwen-7B": "DeepSeek-R1-GGUF/Qwen2.5-7B-Instruct-1M-Q6_K.gguf",
    # "r1-Distill-Llama-8B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    # "r1-Distill-Qwen-7B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q6_K_L.gguf",
    # "r1-Distill-Qwen-32B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf",
    # "LLama-4-Scout": "DeepSeek-R1-GGUF/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf"
    # "Mistral-Small-24B": "DeepSeek-R1-GGUF/Mistral-Small-3.1-24B-Instruct-2503-Q8_0.gguf",
}
PROMPT_TYPE = "zero"
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Dash Setup ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
faiss_indices = {}
faiss_metadata = {}

BASE_DIR = Path(__file__).resolve().parent

# === Load FAISS Indexes ===
def load_faiss_index(name):
    """
    Load FAISS index and metadata into global dictionaries.

    Args:
        name (str): Source name ('tables', 'sections', or 'combined').

    Returns:
        Populates faiss_indices and faiss_metadata dictionaries.
    """
    index_path = BASE_DIR / "faiss_chunks" / f"faiss_{name}.idx"
    metadata_path = BASE_DIR / "faiss_chunks" / f"faiss_{name}_metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        print(f"❌ Missing files for {name}")
        return

    index = faiss.read_index(str(index_path))
    with open(metadata_path) as f:
        metadata = json.load(f)

    faiss_indices[name] = index
    faiss_metadata[name] = metadata

for name in ["tables", "sections", "combined"]:
    load_faiss_index(name)

def is_part_mentioned(chunk, part):
    """
    Check if the given part name appears in the chunk content.

    Args:
        chunk (dict): A chunk dictionary with keys like 'type', 'content', 'table_title', 'rows'.
        part (str): Part name to search for.

    Returns:
        bool: True if part is mentioned in the chunk, else False.
    """
    part_lower = part.lower()
    if chunk["type"] == "text":
        return part_lower in chunk.get("content", "").lower()
    elif chunk["type"] == "table":
        title = chunk.get("table_title", "").lower()
        rows_text = json.dumps(chunk.get("rows", [])).lower()
        return part_lower in title or part_lower in rows_text
    return False

# === Layout ===

app.layout = dbc.Container([
    html.H2("Bowtie Generator with RAG Pipeline"),

    dcc.Tabs(id="tabs", value='tab-generate', children=[
        dcc.Tab(label='Generate Bowtie JSON', value='tab-generate', children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='search-mode',
                    options=[
                        {'label': 'Combined', 'value': 'combined'},
                        {'label': 'Tables only', 'value': 'tables'},
                        {'label': 'Sections only', 'value': 'sections'},
                    ],
                    value='combined',
                    clearable=False
                ), width=4),

                dbc.Col(dcc.Input(id='part-input', type='text', placeholder='Enter part name (opt: add page)', style={'width': '100%'}), width=6),
                dbc.Col(html.Button('Search', id='search-button', n_clicks=0), width=2),
            ]),

            html.Button('Generate Bowtie JSON', id='generate-button', n_clicks=0, className='btn btn-success my-2'),
            dcc.Store(id='stored-matches', data=[]),

            html.Hr(),
            html.H4("Matched Chunks Preview"),
            html.Div(id='matched-chunks', style={'whiteSpace': 'pre-wrap'}),

            html.Hr(),
            html.H4("Bowtie Output"),
            html.Pre(id='bowtie-output', style={'whiteSpace': 'pre-wrap'})
        ]),


        dcc.Tab(label='Mermaid Visualizer', value='tab-mermaid', children=[
            html.H4("Upload Bowtie JSON to Render Interactive Mermaid Diagrams"),
            dcc.Upload(
                id='upload-mermaid-json',
                children=html.Div(['📁 Drag or Upload Bowtie JSON']),
                style={
                    'width': '100%', 'padding': '10px', 'borderWidth': '1px',
                    'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'marginTop': '10px'
                },
                multiple=False
            ),
            html.Div(id='mermaid-render-output')
        ])

        ])
    ])


@app.callback(
    Output('matched-chunks', 'children'),
    Output('stored-matches', 'data'),
    Input('search-button', 'n_clicks'),
    State('part-input', 'value'),
    State('search-mode', 'value')
)

def search_chunks(n_clicks, part_input, mode):
    """
    Search FAISS index for chunks relevant to the input part name.

    Args:
        n_clicks (int): Number of times the search button was clicked.
        part_input (str): User input part name and optionally page number.
        mode (str): Which FAISS index to search ('combined', 'tables', 'sections').

    Returns:
        Tuple[List[str], List[dict]]: 
          - List of preview strings to display matched chunks.
          - List of matched chunk dictionaries (stored for later use).
    """
    if not part_input or mode not in faiss_indices:
        return "❌ No search input or FAISS index not loaded.", []

    tokens = part_input.strip().split()
    page_hint = None
    query_tokens = []
    for t in tokens:
        if t.lower() in ['page', 'pg']: continue
        if t.isdigit(): page_hint = int(t)
        else: query_tokens.append(t)

    query = " ".join(query_tokens)
    query_vec = model.encode([query])
    D, I = faiss_indices[mode].search(np.array(query_vec), k=10)

    raw_matches = []
    for idx in I[0]:
        if idx >= len(faiss_metadata[mode]): continue
        meta = faiss_metadata[mode][idx]
        if page_hint is None or meta.get("page") == page_hint:
            raw_matches.append(meta["chunk"])

    matched = [c for c in raw_matches if is_part_mentioned(c, query)] or raw_matches[:3]

    preview = [
        f"[{c['type'].upper()}] (Page {c['page']}):\n{c.get('content', '') or json.dumps(c.get('rows', ''))[:500]}...\n"
        for c in matched
    ]
    preview.insert(0, f"✅ Showing {len(matched)} chunks matching '{query}'")

    return preview[:5], matched

def ask_model(context, part_name, model_path):
    """
    Send prompt with context to the Llama model and return raw text output.

    Args:
        context (str): Concatenated text context from matched chunks.
        part_name (str): The part/component name to include in prompt.
        model_path (str): Path to the GGUF LLaMA model file.

    Returns:
        str: Raw textual response from the LLaMA model.
    """
    prompt = f""" 
You are provided with technical text and/or tables from an FMEA document related to {part_name}. The input may include:
- Structured tables with columns like: 'Failure Mode', 'Failure Cause', 'Failure Effect'

Instructions:

1. Map each 'Failure Mode' to a "critical_event".  
   - If multiple distinct failure modes are present, generate a separate Bowtie object for each.

2. For each critical event entry, extract the following:
   - "mechanism": if uncertain, set as "Unknown Mechanism"
   - "cause": extract from the 'Failure Cause' or 'Cause' column
   - "preventive_barriers": if uncertain, set as "Unknown Barrier"

3. Extract "consequences" from 'Failure Effect', 'Effect', or 'Local Effect' columns.
   - If no effect is given, use ["Unknown Consequence"]
   - If multiple effects are listed (e.g., with commas, slashes, or conjunctions), split them into separate items

4. Use your judgment to simplify and shorten the names used in the JSON.

5. Treat new lines carefully:
   - A new line does **not** automatically indicate a new consequence, cause, or event
   - If a line starts with a lowercase word or continues the sentence, treat it as a continuation
   - Only split into multiple entries if distinct ideas are clearly listed (e.g., via "and", "or", commas, slashes)

Return only the JSON output. Do not include any explanation or commentary.

--- FMEA Context Start ---
{context}
--- FMEA Context End ---

"""
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_threads=16,
        n_ctx=8192,      
        temp=0.5,
        seed=3407,
        # tensor_split=[0.5, 0.5],
    )
    # response = llm(prompt, max_tokens=12000)
    # return response["choices"][0]["text"]


    response = llm(prompt, max_tokens = 4000)
    text_output = response["choices"][0]["text"]
    # print(text_output)
    # after_last = text_output.rsplit("</think>", 1)[-1]
    # print(after_last)
    return text_output

@app.callback(
    Output('bowtie-output', 'children'),
    Input('generate-button', 'n_clicks'),
    State('stored-matches', 'data'),
    State('part-input', 'value'),
    prevent_initial_call=True
)

def generate_bowtie(n, matches, part):
    """
    Generate Bowtie JSON from matched FAISS chunks by sending context to LLM(s).

    Args:
        n (int): Number of clicks on 'Generate' button.
        matches (List[dict]): Matched chunks returned from semantic search.
        part (str): Part/component name string.

    Returns:
        str: Multiline string summary of LLM generation results for each model.
    """
    if not matches:
        return "No matching context available."

    def chunk_to_llm_context(chunk):
        """
        Convert a chunk into formatted text for LLM context.

        Args:
            chunk (dict): Chunk dictionary.

        Returns:
            str: Formatted string representation of chunk.
        """
        if chunk["type"] == "text":
            return chunk.get("content", "") + "\n"
        elif chunk["type"] == "table":
            title = chunk.get("table_title", "Untitled Table")
            columns = [col or "" for col in chunk.get("columns", [])]
            rows = chunk.get("rows", [])
            markdown = chunk.get("markdown")
            row_str = "\n".join("- " + ", ".join(cell or "" for cell in row) for row in rows[:5])
            col_str = "Columns: " + ", ".join(columns)
            return f"### Table: {title}\n{col_str}\nSample Rows:\n{row_str}\nMarkdown Table:\n{markdown or 'N/A'}"
        return ""

    context_text = "\n".join(chunk_to_llm_context(c) for c in matches)
    base_name = part.replace(" ", "_").lower()
    results = []

    for model_name, model_path in available_models.items():
        try:
            print(f"🔁 Running: {model_name}")
            start_time = time.time()
            result = ask_model(context_text, part, model_path)
            duration = time.time() - start_time

            # Try to extract JSON
            match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', result)
            if not match:
                match = re.search(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', result, re.DOTALL)

            if not match:
                txt_path = f"bowtie___{model_name}___{base_name}___{PROMPT_TYPE}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(result)
                results.append(f"❌ {model_name} returned no JSON ({duration:.2f}s) — saved raw output to {txt_path}")
                continue

            json_str = match.group(1 if match.re.groups else 0).strip()
            if json_str.startswith("```") or json_str.endswith("```"):
                json_str = json_str.strip("`").strip()

            try:
                bowtie_json = json5.loads(json_str)
                out_path = f"bowtie_{model_name}_{base_name}_{PROMPT_TYPE}.json"
                with open(out_path, "w") as f:
                    json.dump(bowtie_json, f, indent=2)
                results.append(f"✅ {model_name} success ({duration:.2f}s) — saved to {out_path}")
            except json.JSONDecodeError as je:
                txt_path = f"bowtie_{model_name}_{base_name}_{PROMPT_TYPE}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(result)
                with open(txt_path.replace('.txt', '.log'), 'w', encoding='utf-8') as f:
                    f.write("RAW LLM RESPONSE:\n" + result + "\n\n---\n\nPARSE ERROR:\n" + str(je))
                results.append(f"❌ {model_name} failed to parse JSON ({duration:.2f}s) — saved to {txt_path}\nError: {je}")

        except Exception as e:
            txt_path = f"bowtie_{model_name}_{base_name}_{PROMPT_TYPE}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result)

            results.append(f"❌ {model_name} exception after {duration:.2f}s: {e}")

    return "\n\n".join(results)


@app.callback(
    Output('mermaid-render-output', 'children'),
    Input('upload-mermaid-json', 'contents')
)
def render_uploaded_mermaid(contents):
    """
    Render Mermaid diagrams from uploaded Bowtie JSON file content.

    Args:
        contents (str): Base64 encoded content of uploaded JSON file.

    Returns:
        List[dash.html.Div]: List of Divs each containing a Mermaid diagram and title.
    """
    if contents is None:
        return dash.no_update

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        data = json.loads(decoded)
    except Exception as e:
        return html.Div(f"❌ Error decoding JSON: {e}")

    def sanitize_id(text):
        """
        Sanitize string to valid Mermaid node id by replacing non-alphanumeric characters with underscores.

        Args:
            text (str): Raw node label text.

        Returns:
            str: Sanitized string.
        """
        import re
        text = re.sub(r'\W+', '_', text)
        if not text or not text[0].isalpha():
            text = "id_" + text
        return text

    def to_mermaid(bowtie):
        """
        Convert Bowtie JSON object to Mermaid diagram code.

        Args:
            bowtie (dict): Bowtie JSON with keys: 'critical_event', 'cause', 'mechanism', 'preventive_barriers', 'consequences'.

        Returns:
            str: Mermaid graph definition string.
        """
        lines = [f"graph LR"]

        ce_label = bowtie.get("critical_event", "Unknown Event")
        lines.append(f'    CE["{ce_label}"]:::critical')

        # --- Causes ---
        causes_raw = bowtie.get("cause", "")
        if isinstance(causes_raw, list):
            causes_list = causes_raw
        else:
            # Split ONLY on semicolon ';'
            causes_list = [c.strip() for c in str(causes_raw).split(";") if c.strip()]

        # --- Mechanisms ---
        mechanism_raw = bowtie.get("mechanism", "")
        if isinstance(mechanism_raw, list):
            mechanisms = mechanism_raw
        else:
            mechanisms = [m.strip() for m in str(mechanism_raw).split(";") if m.strip()]

        # --- Barriers ---
        barrier_raw = bowtie.get("preventive_barriers", "")
        if isinstance(barrier_raw, list):
            barriers = barrier_raw
        else:
            barriers = [b.strip() for b in str(barrier_raw).split(";") if b.strip()]

        # --- Consequences ---
        consequences = bowtie.get("consequences", [])
        if isinstance(consequences, str):
            consequences = [consequences]

        # Split consequences on ';' only as well
        processed_consequences = []
        for cons in consequences:
            parts = [c.strip() for c in cons.split(";") if c.strip()]
            processed_consequences.extend(parts)

        # --- Build graph ---

        # Causes -> Mechanisms or causes -> CE if no mechanisms
        for i, cause in enumerate(causes_list):
            cause_id = sanitize_id(f"cause_{i}_{cause[:10]}")
            lines.append(f'    {cause_id}["{cause}"]')
            if mechanisms:
                for j, mech in enumerate(mechanisms):
                    mech_id = sanitize_id(f"mechanism_{j}_{mech[:10]}")
                    lines.append(f'    {mech_id}["{mech}"]')
                    lines.append(f'    {cause_id} --> {mech_id}')
            else:
                lines.append(f'    {cause_id} --> CE')

        # Mechanisms -> CE
        for j, mech in enumerate(mechanisms):
            mech_id = sanitize_id(f"mechanism_{j}_{mech[:10]}")
            lines.append(f'    {mech_id} --> CE')

        # CE -> Barriers -> Consequences or CE -> Consequences if no barriers
        if barriers:
            for k, barrier in enumerate(barriers):
                barrier_id = sanitize_id(f"barrier_{k}_{barrier[:10]}")
                lines.append(f'    {barrier_id}["{barrier}"]')
                lines.append(f'    CE --> {barrier_id}')
                for m, effect in enumerate(processed_consequences):
                    effect_id = sanitize_id(f"effect_{m}_{effect[:10]}")
                    lines.append(f'    {effect_id}["{effect}"]')
                    lines.append(f'    {barrier_id} --> {effect_id}')
        else:
            for m, effect in enumerate(processed_consequences):
                effect_id = sanitize_id(f"effect_{m}_{effect[:10]}")
                lines.append(f'    {effect_id}["{effect}"]')
                lines.append(f'    CE --> {effect_id}')

        return "\n".join(lines)



    # Support single dict or list of Bowtie JSON objects
    if isinstance(data, dict):
        data = [data]

    diagrams = []
    for i, bowtie in enumerate(data):
        mermaid_code = to_mermaid(bowtie)
        diagrams.append(html.Div([
            html.H5(f"Diagram {i+1}: {bowtie.get('critical_event', 'Unknown Event')}"),
            Mermaid(chart=mermaid_code)
        ], style={"marginBottom": "40px"}))

    return diagrams

if __name__ == '__main__':
    app.run(host='0.0.0.0', use_reloader=False, debug=True, port=8051)
