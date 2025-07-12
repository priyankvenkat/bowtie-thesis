
import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import json
import os
import re
import base64
import io
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import pandas as pd
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR
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
    # "Mistral-Small-24B": "DeepSeek-R1-GGUF/Mistral-Small-3.1-24B-Instruct-2503-Q8_0.gguf",
    # "r1-Distill-Qwen-32B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf",
    # "LLama-4-Scout": "DeepSeek-R1-GGUF/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf"
}

# Shared seed list for consistent stochastic runs
# SHARED_SEEDS = [3407, 55, 3036, 37774, 73050]
SHARED_SEEDS = [3407]


def list_gpus():
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print(f"Number of GPUs available: {count}")
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {i}: {name.decode('utf-8') if isinstance(name, bytes) else name}")
            print(f"  Total memory: {meminfo.total / (1024**3):.2f} GB")
            print(f"  Used memory:  {meminfo.used / (1024**3):.2f} GB")
            print(f"  Free memory:  {meminfo.free / (1024**3):.2f} GB")
        pynvml.nvmlShutdown()
    except Exception as e:
        print("Error initializing NVML:", e)

list_gpus()


# def load_model(path):
#     return Llama(
#         model_path=path,
#         n_gpu_layers=-1,
#         n_threads=16,
#         n_ctx=12000,
#         temp=0.6,
#         seed=3407,
#         tensor_split=[0.5, 0.5],
#     )

def load_model(path, seed=3407):
    return Llama(
        model_path=path,
        n_gpu_layers=-1,
        n_threads=16,
        n_ctx=12000,
        temp=0.6,
        seed=seed,
        tensor_split=[0.5, 0.5],
    )

def ask_model(llm, context, part_name):
    prompt = f"""
You are provided with technical text and/or tables from an FMEA document related to {part_name}. The input may include:
- Structured tables with columns like: 'Failure Mode', 'Failure Cause', 'Failure Effect'

Instructions:

1. Map each 'Failure Mode' to a "critical_event".  
   - If multiple distinct failure modes are present, generate a separate Bowtie object for each.

2. For each critical event entry, extract the following:
   - "mechanism": set as "Mechanism"
   - "cause": extract from the 'Failure Cause' or 'Cause' column, there could be multiple causes for one critical event, seprate with ';'
   - "preventive_barriers": set as "Barrier"

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
    # response = llm(prompt, max_tokens=12000)
    # return response["choices"][0]["text"]

    response = llm(prompt, max_tokens = 10000)
    text_output = response["choices"][0]["text"]
    # print(text_output)
    after_last = text_output.rsplit("</think>", 1)[-1]
    # print(after_last)
    return after_last


# === Dash App Layout ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("Bowtie Generator (OCR + Mermaid Conversion)"),

    dcc.Tabs(id="tabs", value='tab-ocr', children=[
        dcc.Tab(label='OCR to Bowtie JSON', value='tab-ocr', children=[
            dcc.Upload(
                id='upload-ocr',
                children=html.Div(['📄 Drag and drop images or PDF files here']),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
                multiple=True
            ),
            html.Div(id='column-heading-inputs'),

            dcc.Input(id='part-name', type='text', placeholder='Enter part name...', style={'width': '50%'}),
            html.Button('Generate Bowtie JSON', id='generate-bowtie', n_clicks=0, className='btn btn-success mt-2'),

            html.Hr(),
            html.H4("Extracted Table (Markdown) + Full Text"),
            html.Pre(id='ocr-preview', style={'whiteSpace': 'pre-wrap', 'maxHeight': '300px', 'overflowY': 'scroll'}),

            html.Hr(),
            html.H4("Bowtie JSON Output"),
            html.Pre(id='bowtie-output', style={'whiteSpace': 'pre-wrap'})
        ]),

        dcc.Tab(label='Mermaid Visualizer', value='tab-mermaid', children=[
            html.H4("Paste or Upload Bowtie JSON to Render as Mermaid Graph"),
            dcc.Textarea(id='json-input', style={'width': '100%', 'height': '300px'}),
            dcc.Upload(
                id='upload-mermaid-json',
                children=html.Div(['📄 Drag or Upload Bowtie JSON']),
                style={
                    'width': '100%', 'padding': '10px', 'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'marginTop': '10px'
                },
                multiple=False
            ),
            html.Button('Render Mermaid', id='render-mermaid', n_clicks=0),
            html.Hr(),
            html.H4("Mermaid Output (graph LR syntax)"),
            html.Pre(id='mermaid-output', style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'})

        # dcc.Tab(label='Mermaid Visualizer', value='tab-mermaid', children=[
        #     html.H4("Upload Bowtie JSON to Render Interactive Mermaid Diagrams"),
        #     dcc.Upload(
        #         id='upload-mermaid-json',
        #         children=html.Div(['📁 Drag or Upload Bowtie JSON']),
        #         style={
        #             'width': '100%', 'padding': '10px', 'borderWidth': '1px',
        #             'borderStyle': 'dashed', 'borderRadius': '5px',
        #             'textAlign': 'center', 'marginTop': '10px'
        #         },
        #         multiple=False
        #     ),
        #     html.Div(id='mermaid-render-output')

        ])
    ])
])

@app.callback(
    Output('column-heading-inputs', 'children'),
    Input('upload-ocr', 'contents'),
    State('upload-ocr', 'filename')
)
def update_column_header_ui(contents, filenames):
    if not contents:
        return dash.no_update

    inputs = []
    for i, fname in enumerate(filenames):
        inputs.append(html.Div([
            html.H5(f"{fname} - Column Headings"),
            dbc.Row([
                dbc.Col(dcc.Input(id={'type': 'col-head', 'index': i*3+j}, type='text',
                                  placeholder=f'Column {j+1}',
                                  value=default,
                                  debounce=True),
                        width=4) for j, default in enumerate(["Failure Mode", "Failure Cause", "Failure Effect"])
            ])
        ], style={'marginBottom': '20px'}))
    return inputs

def merge_broken_rows(df):
    merged = []
    current = ["", "", ""]
    for _, row in df.iterrows():
        if pd.notna(row[0]) and row[0].strip():
            if any(current):
                merged.append(current)
            current = [str(row[0] or ""), str(row[1] or ""), str(row[2] or "")]
        else:
            current = [
                current[0] + " " + (str(row[0]) if row[0] else ""),
                current[1] + " " + (str(row[1]) if row[1] else ""),
                current[2] + " " + (str(row[2]) if row[2] else ""),
            ]
    if any(current):
        merged.append(current)
    return pd.DataFrame(merged, columns=df.columns)

# === Callbacks for Mermaid Visualization ===

@app.callback(
    Output('json-input', 'value'),
    Input('upload-mermaid-json', 'contents')
)
def handle_mermaid_upload(contents):
    if contents is None:
        return dash.no_update
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return decoded.decode('utf-8')


def generate_mermaid_from_bowtie(data):
    def escape(text):
        text = str(text)
        for char in ['[', ']', '{', '}', '"', "'"]:
            text = text.replace(char, '')
        return text

    def build_mermaid(diagram):
        lines = ["graph LR"]
        te_label = escape(diagram.get("top_event") or diagram.get("critical_event") or "Critical Event")
        lines.append(f"  TE[{te_label}]")

        mech_to_id = {}
        mech_count = 0
        threat_count = 0
        barrier_count = 0

        for threat in diagram.get("threats", []):
            mechanism = threat.get("mechanism") or "Unknown Mechanism"
            cause = threat.get("cause") or "Unknown Cause"
            barriers = threat.get("preventive_barriers") or []

            if mechanism.strip() == cause.strip():
                threat_count += 1
                t_id = f"T{threat_count}"
                lines.append(f"  {t_id}[Threat: {escape(mechanism)}] --> TE")
                for barrier in barriers:
                    barrier_count += 1
                    b_id = f"PB{barrier_count}"
                    lines.append(f"  {b_id}[Barrier: {escape(barrier)}] --> {t_id}")
            else:
                if mechanism not in mech_to_id:
                    mech_count += 1
                    m_id = f"M{mech_count}"
                    mech_to_id[mechanism] = m_id
                    lines.append(f"  {m_id}[Mechanism: {escape(mechanism)}] --> TE")
                else:
                    m_id = mech_to_id[mechanism]

                threat_count += 1
                c_id = f"C{threat_count}"
                lines.append(f"  {c_id}[Cause: {escape(cause)}] --> {m_id}")
                for barrier in barriers:
                    barrier_count += 1
                    b_id = f"PB{barrier_count}"
                    lines.append(f"  {b_id}[Barrier: {escape(barrier)}] --> {c_id}")

        consequences = diagram.get("consequences", [])
        if isinstance(consequences, str):
            consequences = [consequences]

        for i, consequence in enumerate(consequences):
            cons_id = f"Cons{i+1}"
            lines.append(f"  TE --> {cons_id}[Consequence: {escape(consequence)}]")


        for i, barrier in enumerate(diagram.get("mitigative_barriers", [])):
            mb_id = f"MB{i+1}"
            lines.append(f"  TE --> {mb_id}[Mitigative Barrier: {escape(barrier)}]")

        return "\n".join(lines)

    if isinstance(data, list):
        return "\n\n".join(build_mermaid(d) for d in data)
    elif isinstance(data, dict):
        return build_mermaid(data)
    else:
        raise ValueError("Invalid input for Mermaid rendering")


@app.callback(
    Output('mermaid-output', 'children'),
    Input('render-mermaid', 'n_clicks'),
    State('json-input', 'value')
)
def render_mermaid(n_clicks, val):
    if not val:
        return "Paste or upload Bowtie JSON above."
    try:
        parsed = json.loads(val)
        return generate_mermaid_from_bowtie(parsed)
    except Exception as e:
        return f"Error: {e}"

# === OCR + Model Inference Callback ===
@app.callback(
    Output('ocr-preview', 'children'),
    Output('bowtie-output', 'children'),
    Input('generate-bowtie', 'n_clicks'),
    State('upload-ocr', 'contents'),
    State('upload-ocr', 'filename'),
    State('part-name', 'value'),
    State({'type': 'col-head', 'index': ALL}, 'value'), 
    prevent_initial_call=True
)
def process_all(n_clicks, contents, filenames, part, all_headers):
    if not contents or not filenames or not part:
        return "Missing input.", ""

    extracted_texts = []
    for content, fname in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        buffer = io.BytesIO(decoded)
        text_block = ""

        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = f"temp_{fname}"
            with open(img_path, "wb") as f:
                f.write(decoded)

            ocr = TesseractOCR(lang="eng")
            image_doc = Img2TableImage(src=img_path)
            tables = image_doc.extract_tables(ocr=ocr, implicit_columns=True, implicit_rows=True, borderless_tables=True)
            os.remove(img_path)

            if tables:
                df = tables[0].df
                user_headers = all_headers[:3]  # if you're only expecting 1 file; use i*3:i*3+3 for multi-file

                if df.shape[1] > len(user_headers):
                    df = df.iloc[:, :len(user_headers)]
                elif df.shape[1] < len(user_headers):
                    for _ in range(len(user_headers) - df.shape[1]):
                        df[df.shape[1]] = ""
                df.columns = user_headers


                df = df[1:].reset_index(drop=True)
                df = df.applymap(lambda x: str(x).strip() if pd.notnull(x) else "")
                df = merge_broken_rows(df)
                text_block += f"**{fname} – Table Extracted:**\n\n{df.to_markdown(index=False)}\n\n"
            else:
                text_block += f"**{fname} – No tables found.**\n\n"

            image = Image.open(io.BytesIO(decoded))
            raw_text = pytesseract.image_to_string(image)
            text_block += f"**Full Text:**\n\n{raw_text}\n\n"

        elif fname.lower().endswith(".pdf"):
            doc = fitz.open(stream=buffer.read(), filetype="pdf")
            pdf_text = "\n\n".join(page.get_text("text") for page in doc)
            text_block += f"**{fname} – PDF Text Extracted:**\n\n{pdf_text}\n\n"

        extracted_texts.append(text_block)

    context = "\n\n".join(extracted_texts)

    all_outputs = {}
    for model_name, model_path in available_models.items():
        print(f"🔄 Running model: {model_name} on {len(SHARED_SEEDS)} seeds")

        for run_idx, seed in enumerate(SHARED_SEEDS):
            print(f"  🌱 Seed {seed} - Run {run_idx+1}")
            try:
                model = load_model(model_path, seed=seed)
                start_time = time.time()
                response = ask_model(model, context, part)
                del model
                duration = time.time() - start_time
            except Exception as e:
                all_outputs[f"{model_name} | seed {seed}"] = f"❌ Failed in {duration:.2f}s: {e}"
                continue

            match = re.search(r'(\[.*\]|\{.*\})', response, re.DOTALL)
            if not match:
                all_outputs[f"{model_name} | seed {seed}"] = f"❌ No valid JSON found (⏱ {duration:.2f}s):\n\n{response}"
                continue

            try:
                json_str = match.group(0).strip().strip('`')
                parsed_json = json5.loads(json_str)
                label = f"{model_name}_seed_{seed}"
                out_path = f"bowtie_{label.replace(' ', '_')}_{part.replace(' ', '_').lower()}_ocr.json"
                with open(out_path, "w") as f:
                    json.dump(parsed_json, f, indent=2)

                all_outputs[f"{model_name} | seed {seed}"] = {
                    "duration": duration,
                    "json": parsed_json
                }

            except Exception as e:
                all_outputs[f"{model_name} | seed {seed}"] = f"❌ JSON parse failed (⏱ {duration:.2f}s): {e}\n\nRaw output:\n{response}"
                txt_path = f"bowtie_{model_name.replace(' ', '_')}_seed_{seed}_{part.replace(' ', '_').lower()}_ocr.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(response)

    def format_output(name, data):
        if isinstance(data, dict) and "json" in data:
            return f"✅ {name} completed in {data['duration']:.2f} seconds"
        else:
            return f"❌ {name} failed or returned invalid output"

    combined_output = "\n".join(format_output(name, data) for name, data in all_outputs.items())
    return context, combined_output

if __name__ == '__main__':
    app.run(host='0.0.0.0', use_reloader=False, debug=True, port=8052)
