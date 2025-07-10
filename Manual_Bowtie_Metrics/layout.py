from dash import html, dcc
from core.constants import METHODS, PROMPTS, DOMAINS, MODELS
from dash import dash_table

layout = html.Div([
    html.H2("GED: Ground Truth vs Predicted Diagram Images"),

    html.Div([
        html.Div([
            html.Label("Upload Ground Truth Image (diagram)"),
            dcc.Upload(
                id='upload-image-gt',
                children=html.Div(['📥 Drag and Drop or Click to Upload GT Image']),
                style={'border': '2px dashed #ccc', 'padding': '10px', 'textAlign': 'center'},
                multiple=False,
                accept='image/*'
            ),
            html.Div(id='upload-status-gt', style={'marginTop': '5px', 'color': 'green'})
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Upload Predicted Image (diagram)"),
            dcc.Upload(
                id='upload-image-pred',
                children=html.Div(['📥 Drag and Drop or Click to Upload Predicted Image']),
                style={'border': '2px dashed #ccc', 'padding': '10px', 'textAlign': 'center'},
                multiple=False,
                accept='image/*'
            ),
            html.Div(id='upload-status-pred', style={'marginTop': '5px', 'color': 'green'})
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Label("🔑 Mistral API Key"),
        dcc.Input(id='api-key', type='password', value="FpL49bp6asB0frkxMbmazONcm3eoubOT",placeholder='sk-...', debounce=True, style={"width": "40%"})
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Div([
            html.Label("Method"),
            dcc.Dropdown(id='method', options=[{'label': m, 'value': m} for m in METHODS], value=METHODS[0]),
        ], style={'width': '19%', 'display': 'inline-block', 'paddingRight': '10px'}),

        html.Div([
            html.Label("Prompt"),
            dcc.Dropdown(id='prompt', options=[{'label': p, 'value': p} for p in PROMPTS], value=PROMPTS[0]),
        ], style={'width': '19%', 'display': 'inline-block', 'paddingRight': '10px'}),

        html.Div([
            html.Label("Domain"),
            dcc.Dropdown(id='domain', options=[{'label': d, 'value': d} for d in DOMAINS], value=DOMAINS[0]),
        ], style={'width': '19%', 'display': 'inline-block', 'paddingRight': '10px'}),

        html.Div([
            html.Label("Model"),
            dcc.Dropdown(id='model', options=[{'label': m, 'value': m} for m in MODELS], value=MODELS[0]),
        ], style={'width': '19%', 'display': 'inline-block', 'paddingRight': '10px'}),

        html.Div([
            html.Label("Central Event"),
            dcc.Input(id='central-event', type='text', placeholder='e.g. Shaft Deflection'),
        ], style={'width': '19%', 'display': 'inline-block'})
    ], style={'marginBottom': '20px'}),


    html.Button("📤 Load Graphs from Images", id='load-graphs-button', n_clicks=0),
    html.Button("🧮 Compute GED", id='compute-button', style={'marginLeft': '10px'}),

    html.Div(id='ged-output', style={'marginTop': 20}),

    html.Hr(),

    html.Div([
        html.H4("🖼️ Uploaded Diagram Previews"),
        html.Div([
            html.Div([
                html.H5("Ground Truth Image"),
                html.Img(id='preview-gt', style={'maxWidth': '100%', 'border': '1px solid #ccc'})
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.H5("Predicted Image"),
                html.Img(id='preview-pred', style={'maxWidth': '100%', 'border': '1px solid #ccc'})
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
        ])
    ], style={'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.Label("✏️ Edit Ground Truth Graph JSON"),
            dcc.Textarea(id='gt-json', style={'width': '100%', 'height': 300}),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("✏️ Edit Predicted Graph JSON"),
            dcc.Textarea(id='pred-json', style={'width': '100%', 'height': 300}),
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
    ]),

    html.Br(),
    html.Div(id='role-assignment-ui'),

    html.H4("📋 GED Results Table"),
    dcc.Loading(
        dash_table.DataTable(
            id='results-table',
            columns=[{"name": i, "id": i, 'editable': True} for i in ["Method", "Prompt", "Domain", "Model", "GED"]],
            editable=True,
            row_deletable=True,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        ),
        type="default"
    )
    
])
