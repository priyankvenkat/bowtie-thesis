import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table

"""
Defines the full Dash app layout.

- Sets up tabs for evaluation and review.
- Accepts GT and 1 or more prediction files.
- Displays node, edge, and GED metrics in tables.
"""

METHODS = ["RAG", "OCR", "Vision"]
PROMPTS = ["zero", "few", "cot", "hybrid"]
DOMAINS = ["shaft", "valve", "sensor", "dynamic-seals"]
MODELS = [
    "Qwen-7B", "Mistral-Instruct", "LLaMA-Instruct",
    "R1-Distill-LLama-8B", "R1-Distill-Qwen-7B"
]

layout = dbc.Container([
    html.H2("📊 Bowtie GED Evaluation Dashboard", className="text-center my-4"),

    dcc.Tabs(id="main-tabs", children=[

        dcc.Tab(label="Evaluation", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ground Truth Upload"),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-gt',
                                children=html.Div(['📁 Drag & Drop or ', html.A('Select GT JSON')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed',
                                    'borderRadius': '5px', 'textAlign': 'center'
                                },
                                multiple=False
                            ),
                            html.Div(id='gt-status', className="text-success mt-2")
                        ])
                    ])
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Prediction Upload (1 or many)"),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-preds',
                                children=html.Div(['📁 Drag & Drop or ', html.A('Select Prediction JSON(s)')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed',
                                    'borderRadius': '5px', 'textAlign': 'center'
                                },
                                multiple=True
                            ),
                            html.Div(id='pred-status', className="text-success mt-2")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("Metadata Inputs"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='method', options=[{"label": m, "value": m} for m in METHODS], value="Dual LLM"), width=3),
                        dbc.Col(dcc.Dropdown(id='prompt', options=[{"label": p, "value": p} for p in PROMPTS], value="cot"), width=3),
                        dbc.Col(dcc.Dropdown(id='domain', options=[{"label": d, "value": d} for d in DOMAINS], value="sensor"), width=3),
                        dbc.Col(dcc.Dropdown(id='model', options=[{"label": m, "value": m} for m in MODELS], value="LLaMA-Instruct"), width=3),
                    ])
                ])
            ], className="mb-3"),

            dbc.Button("🚀 Run Evaluation", id='run-eval', color='primary', className="mb-4"),

            dbc.Card([
                dbc.CardHeader("Node Metrics"),
                dbc.CardBody([
                    dcc.Loading(
                        dash_table.DataTable(id="node-metrics-table", page_size=10, style_table={'overflowX': 'auto'}),
                        type="default"
                    )
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Edge Metrics"),
                dbc.CardBody([
                    dcc.Loading(
                        dash_table.DataTable(id="edge-metrics-table", page_size=10, style_table={'overflowX': 'auto'}),
                        type="default"
                    )
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Graph Edit Distance (GED)"),
                dbc.CardBody([
                    dcc.Loading(
                        dash_table.DataTable(id="ged-metrics-table", page_size=10, style_table={'overflowX': 'auto'}),
                        type="default"
                    )
                ])
            ], className="mb-4"),

            dbc.Alert(id='results-area', is_open=False, duration=4000),
            html.Div(id='download-link', className="mb-4 text-center")
        ]),

        dcc.Tab(label="Review", children=[
            html.Div(id="review-tab-content", className="p-4")
        ])
    ]),

    # Optional stores for future override/match support
    dcc.Store(id="manual-match-store"),
    dcc.Store(id="node-match-store"),
    dcc.Store(id="node-analysis-store"),
    dcc.Store(id="node-match-dropdown-metadata"),
    dcc.Store(id="show-submit-node-button")

], fluid=True)
