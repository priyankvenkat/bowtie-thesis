
from dash import Dash, html, dcc, Input, Output, State
import base64, json
import dash_bootstrap_components as dbc
from callbacks import (
    extract_table_callback,
    update_prompt_callback,
    generate_json_callback,
    generate_mermaid_callback,
    generate_mermaid_from_bowtie,
    extract_triples_callback,  # NEW
    graph_to_json_callback     # NEW
)
from kg_pipeline import register_additional_callbacks


# === Initialize Dash App ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

register_additional_callbacks(app)

# === Define Tabs ===
tabs = dbc.Tabs([
    dbc.Tab(label="🧠 Table to Bowtie JSON", tab_id="json"),
    dbc.Tab(label="📄 Mermaid Visualiser", tab_id="mermaid"),
    dbc.Tab(label="🧠 Vision LLM + KG", tab_id="kg")  # NEW
])

# === Layout for JSON Tab ===
layout_json = dbc.Container([
    html.H2("📊 Table to Bowtie JSON Generator"),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
        style={
            'width': '100%', 'height': '100px', 'lineHeight': '100px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '20px'
        }
    ),
    dcc.Input(
        id='part-name', type='text', placeholder='Enter component name (e.g. static seal)',
        style={'width': '100%', 'marginBottom': '10px'}
    ),
    dbc.Checkbox(
        id='structured-output-toggle', label='Generate structured markdown and JSON output',
        value=True, style={"marginBottom": "15px"}
    ),
    dbc.Button("Step 1: Extract Table Description + Markdown", id="extract-btn", color="primary", className="w-100"),
    html.Br(), html.Br(),
    html.Div(id='markdown-output', style={'whiteSpace': 'pre-wrap', 'color': '#212529'}),
    html.Hr(),
    html.H5("Step 2: Generate Bowtie JSON"),
    dcc.Dropdown(
        id='prompt-type',
        options=[
            {"label": "Zero Shot", "value": "zero"},
            {"label": "Few Shot", "value": "few"},
            {"label": "Chain of Thought", "value": "cot"},
        ],
        placeholder="Select prompt type",
        style={"marginBottom": "10px"}
    ),
    dcc.Textarea(
        id='custom-prompt',
        placeholder="Prompt will appear here after selecting a type. You can edit it before generating JSON.",
        style={'width': '100%', 'height': 150, 'marginBottom': '20px'}
    ),
    dbc.Button("Generate Bowtie JSON", id="generate-json-btn", color="success", className="w-100"),
    html.Br(), html.Br(),
    html.Div(
        id='json-output',
        style={'whiteSpace': 'pre-wrap', 'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px'}
    )
])

#  === Layout for Mermaid Tab ===
layout_mermaid = dbc.Container([
    html.H2("📄 Visualise Bowtie JSON as Mermaid Diagram"),
    html.P("Upload or paste your Bowtie JSON below to generate Mermaid code."),
    dcc.Upload(
        id='upload-json',
        children=html.Div(['Drop a JSON file or ', html.A('Select JSON')]),
        style={
            'width': '100%', 'height': '80px', 'lineHeight': '80px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px'
        },
        multiple=True
    ),
    dcc.Textarea(
        id='manual-json-input',
        placeholder="Or paste JSON here...",
        style={'width': '100%', 'height': 250, 'marginBottom': '15px'}
    ),
    dbc.Button("Generate Mermaid Diagram", id="generate-mermaid-btn", color="info", className="w-100"),
    html.Br(),
    html.Div(id='mermaid-output', style={'whiteSpace': 'pre-wrap', 'marginTop': '15px'})
])


# === Layout for Vision LLM + KG Tab ===
layout_kg = dbc.Container([
    html.H2("📸 Extract SPO Triples and Build Knowledge Graph"),
    dcc.Upload(
        id='upload-vision-img',
        children=html.Div(['Drop image or ', html.A('Select image')]),
        style={
            'width': '100%', 'height': '100px', 'lineHeight': '100px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '20px'
        }
    ),
    dbc.Button("Step 1: Extract Triples (Vision LLM)", id="extract-triples-btn", color="primary", className="w-100"),
    html.Div(id='triple-output', style={'whiteSpace': 'pre-wrap', 'marginTop': '15px'}),
    html.Br(),
    dbc.Input(id="central-event", placeholder="Enter Critical Event", type="text", debounce=True),
    dbc.Button("Step 2: Generate Bowtie JSON from Graph", id="graph-to-json-btn", color="success", className="w-100"),
    html.Div(id="graph-json-output", style={'whiteSpace': 'pre-wrap', 'marginTop': '15px'}),
    html.Hr(),
    html.H5("Export All Bowtie JSONs"),
    dbc.Button("Generate All Bowtie JSONs", id="generate-all-bowties-btn", color="secondary", className="w-100"),
    html.Div(id="all-bowtie-json-output", style={'whiteSpace': 'pre-wrap', 'marginTop': '15px'}),
    html.Hr(),
    html.H5("Reset Graph"),
    dbc.Button("Reset Graph", id="reset-graph-btn", color="danger", className="w-100"),
    html.Div(id="reset-output", style={'whiteSpace': 'pre-wrap', 'marginTop': '15px'}),

])

# === App Layout ===
app.layout = dbc.Container([
    tabs,
    html.Div(id='tab-content')
])

# === Callbacks ===
@app.callback(Output('tab-content', 'children'), Input(tabs, 'active_tab'))
def display_tab(tab):
    if tab == 'json':
        return layout_json
    elif tab == 'mermaid':
        return layout_mermaid
    elif tab == 'kg':
        return layout_kg
    return html.Div("❌ Unknown tab")

@app.callback(
    Output('markdown-output', 'children'),
    Input('extract-btn', 'n_clicks'),
    State('upload-image', 'contents'),
    State('structured-output-toggle', 'value'),
    State('part-name', 'value'),
    prevent_initial_call=True
)
def extract_table(n_clicks, contents, structured_output, part_name):
    return extract_table_callback(n_clicks, contents, structured_output, part_name)

@app.callback(
    Output('custom-prompt', 'value'),
    Input('prompt-type', 'value'),
    State('part-name', 'value'),
    prevent_initial_call=True
)
def update_prompt(prompt_type, part_name):
    return update_prompt_callback(prompt_type, part_name)

@app.callback(
    Output('json-output', 'children'),
    Input('generate-json-btn', 'n_clicks'),
    State('custom-prompt', 'value'),
    State('part-name', 'value'),
    prevent_initial_call=True
)
def generate_json(n_clicks, custom_prompt, part_name):
    return generate_json_callback(n_clicks, custom_prompt, part_name)

@app.callback(
    Output("mermaid-output", "children"),
    Input("generate-mermaid-btn", "n_clicks"),
    State("manual-json-input", "value"),
    State("upload-json", "contents"),
    prevent_initial_call=True
)
def show_mermaid(n_clicks, manual_input, file_contents):
    try:
        json_data_list = []
        # Check if files were uploaded.
        if file_contents:
            # If file_contents is not a list, make it a list.
            if not isinstance(file_contents, list):
                file_contents = [file_contents]
            
            # Process each file.
            for content in file_contents:
                # Split the content to remove metadata.
                header, content_string = content.split(',')
                decoded = base64.b64decode(content_string).decode("utf-8")
                json_data_list.append(json.loads(decoded))
        elif manual_input:
            json_data_list.append(json.loads(manual_input.replace("'", '"')))
        else:
            return "❌ Please upload or paste valid JSON."
        
        # Generate Mermaid code for each JSON file.
        mermaid_codes = []
        for idx, data in enumerate(json_data_list):
            code = generate_mermaid_from_bowtie(data)
            mermaid_codes.append(f"Diagram {idx+1}:\n{code}")
        
        return html.Pre("\n\n".join(mermaid_codes))
    
    except Exception as e:
        return html.Div(f"❌ Error parsing input: {e}", style={"color": "red"})

@app.callback(
    Output("triple-output", "children"),
    Input("extract-triples-btn", "n_clicks"),
    State("upload-vision-img", "contents"),
    prevent_initial_call=True
)
def run_extract_triples(n_clicks, contents):
    return extract_triples_callback(contents)

@app.callback(
    Output("graph-json-output", "children"),
    Input("graph-to-json-btn", "n_clicks"),
    State("central-event", "value"),
    prevent_initial_call=True
)
def run_graph_to_json(n_clicks, ce):
    return graph_to_json_callback(ce)

if __name__ == '__main__':
    app.run(debug=True, port=8056)

