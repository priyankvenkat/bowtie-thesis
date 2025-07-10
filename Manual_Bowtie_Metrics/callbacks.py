# All Dash callbacks go here
import json
import dash
from dash import Input, Output, State, MATCH, ALL, callback_context, dash_table
from dash.exceptions import PreventUpdate
from dash import dcc, html
from core.extract import extract_graph_from_image
from core.ged import compute_ged, get_last_cosine_mappings
from core.metrics_og import compute_node_metrics, export_cosine_mappings_to_ged_sheet, export_node_comparisons_to_excel, export_node_metrics_to_excel,compute_edge_metrics, export_edge_metrics_to_excel, update_csv, export_detailed_edge_comparisons
from core.constants import NODE_ROLES
from core.graph_utils import normalize

def register_callbacks(app):

    @app.callback(
        Output('preview-gt', 'src'),
        Output('preview-pred', 'src'),
        Input('upload-image-gt', 'contents'),
        Input('upload-image-pred', 'contents')
    )
    def update_image_previews(gt_content, pred_content):
        return gt_content, pred_content

    @app.callback(
        Output('upload-status-gt', 'children'),
        Output('upload-status-pred', 'children'),
        Input('upload-image-gt', 'contents'),
        Input('upload-image-pred', 'contents')
    )
    def update_upload_status(gt, pred):
        gt_msg = "✅ Ground Truth image uploaded." if gt else "⚠️ Ground Truth image not uploaded."
        pred_msg = "✅ Predicted image uploaded." if pred else "⚠️ Predicted image not uploaded."
        return gt_msg, pred_msg

    @app.callback(
        Output('gt-json', 'value'),
        Output('pred-json', 'value'),
        Input('load-graphs-button', 'n_clicks'),
        State('upload-image-gt', 'contents'),
        State('upload-image-pred', 'contents'),
        State('api-key', 'value')
    )
    def update_textareas(n, gt_img, pred_img, api_key):
        if not api_key or not gt_img or not pred_img:
            return "", ""
        try:
            gt = extract_graph_from_image(gt_img.split(',')[1], api_key)
            pred = extract_graph_from_image(pred_img.split(',')[1], api_key)
            return json.dumps(gt, indent=2), json.dumps(pred, indent=2)
        except Exception as e:
            return f"Error: {str(e)}", ""

    @app.callback(
        Output('role-assignment-ui', 'children'),
        Input('gt-json', 'value'),
        Input('pred-json', 'value')
    )
    def generate_role_dropdowns(gt_text, pred_text):
        try:
            gt_data = json.loads(gt_text)
            pred_data = json.loads(pred_text)
        except:
            return "⚠️ Invalid JSON"

        def suggest_roles(edges):
            in_deg = {}
            out_deg = {}
            for a, b in edges:
                out_deg[a] = out_deg.get(a, 0) + 1
                in_deg[b] = in_deg.get(b, 0) + 1
            all_nodes = set(in_deg.keys()) | set(out_deg.keys())
            role_map = {}
            for n in all_nodes:
                indeg = in_deg.get(n, 0)
                outdeg = out_deg.get(n, 0)
                if indeg == 0 and outdeg > 0:
                    role = "Cause"
                elif indeg > 0 and outdeg == 0:
                    role = "Consequence"
                elif indeg > 0 and outdeg > 0:
                    role = "Mechanism"
                else:
                    role = ""
                role_map[n] = role
            return role_map

        gt_roles = suggest_roles(gt_data["edges"])
        pred_roles = suggest_roles(pred_data["edges"])

        def create_dropdowns(nodes, prefix, role_map):
            return html.Div([
                html.H5(f"{prefix} Node Roles"),
                html.Div([
                    html.Div([
                        html.Label(n),
                        dcc.Dropdown(
                            id={'type': f'{prefix}-role-dropdown', 'index': n},
                            options=[{'label': r, 'value': r} for r in NODE_ROLES],
                            value=role_map.get(n, ""),
                            placeholder="Select role"
                        )
                    ]) for n in sorted(nodes)
                ])
            ], style={'width': '48%', 'display': 'inline-block'})

        gt_nodes = {n for e in gt_data["edges"] for n in e}
        pred_nodes = {n for e in pred_data["edges"] for n in e}

        return html.Div([
            create_dropdowns(gt_nodes, "gt", gt_roles),
            create_dropdowns(pred_nodes, "pred", pred_roles)
        ])

    @app.callback(
        Output('compute-button', 'disabled'),
        Input('central-event', 'value')
    )
    def disable_compute_if_no_event(ce):
        return not ce or ce.strip() == ""


    @app.callback(
        Output('ged-output', 'children'),
        Output('results-table', 'data'),
        Input('compute-button', 'n_clicks'),
        State('gt-json', 'value'),
        State('pred-json', 'value'),
        State({'type': 'gt-role-dropdown', 'index': ALL}, 'value'),
        State({'type': 'gt-role-dropdown', 'index': ALL}, 'id'),
        State({'type': 'pred-role-dropdown', 'index': ALL}, 'value'),
        State({'type': 'pred-role-dropdown', 'index': ALL}, 'id'),
        State('method', 'value'),
        State('prompt', 'value'),
        State('domain', 'value'),
        State('model', 'value'),
        State('central-event', 'value'),
    )
    def compute_ged_callback(n, gt_text, pred_text,
                              gt_vals, gt_ids, pred_vals, pred_ids,
                              method, prompt, domain, model, central_event):

        if "" in gt_vals or "" in pred_vals:
            return "❌ Please assign a role to every node before computing GED.", dash.no_update

        if not central_event or central_event.strip() == "":
            return "❌ Please enter a Central Event before computing.", dash.no_update

        try:
            gt_graph = json.loads(gt_text)
            pred_graph = json.loads(pred_text)

            gt_roles = {}
            for i, v in zip(gt_ids, gt_vals):
                raw = i['index']
                norm = normalize(raw)
                gt_roles[raw] = v
                gt_roles[norm] = v

            pred_roles = {}
            for i, v in zip(pred_ids, pred_vals):
                raw = i['index']
                norm = normalize(raw)
                pred_roles[raw] = v
                pred_roles[norm] = v

            ged = compute_ged(gt_graph['edges'], pred_graph['edges'], gt_roles, pred_roles)

            metrics = compute_node_metrics(gt_graph['edges'], pred_graph['edges'], gt_roles, pred_roles)
            export_node_metrics_to_excel(metrics, method, prompt, domain, model, central_event)

            edge_metrics = compute_edge_metrics(gt_graph['edges'], pred_graph['edges'], gt_roles, pred_roles)
            export_edge_metrics_to_excel(edge_metrics, method, prompt, domain, model, central_event)

            export_detailed_edge_comparisons(
                gt_edges=gt_graph['edges'],
                pred_edges=pred_graph['edges'],
                gt_roles=gt_roles,
                pred_roles=pred_roles,
                method=method,
                prompt=prompt,
                domain=domain,
                model=model,
                central_event=central_event
            )

            gt_nodes = {normalize(n) for e in gt_graph["edges"] for n in e}
            pred_nodes = {normalize(n) for e in pred_graph["edges"] for n in e}

            export_node_comparisons_to_excel(
                gt_nodes=gt_nodes,
                pred_nodes=pred_nodes,
                gt_roles=gt_roles,
                pred_roles=pred_roles,
                method=method,
                prompt=prompt,
                model_name=model,
                domain=domain,
               central_event=central_event)

            cosine_mappings = get_last_cosine_mappings()

            export_cosine_mappings_to_ged_sheet(
                mapping_scores=cosine_mappings,
                method=method,
                prompt=prompt,
                domain=domain,
                model=model,
                central_event=central_event
            )

            row = {
                "Method": method,
                "Prompt": prompt,
                "Domain": domain,
                "Central Event": central_event,
                "Model": model,
                "GED": ged
            }

            df = update_csv(row)
            return f"✅ GED = {ged:.2f}", df.to_dict("records")

        except Exception as e:
            return f"❌ Error: {str(e)}", dash.no_update
