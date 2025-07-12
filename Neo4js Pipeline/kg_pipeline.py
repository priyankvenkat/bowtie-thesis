import json
from dash import html
from mistralai import Mistral
from neo4j import GraphDatabase
from config import MISTRAL_API_KEY, PIXTRAL_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import re


def reset_neo4j_graph():
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return html.Div("🗑️ Graph database has been reset.", style={"color": "green"})
    except Exception as e:
        return html.Div(f"❌ Error resetting graph: {str(e)}", style={"color": "red"})

# === Neo4j Connection ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def clean_name(name, ce):
    return name.replace(f" ({ce})", "").strip()

def parse_triples_fallback(text):
    pattern = r"(.+?)\s+(causes|leads to|results in|prevents|mitigates)\s+(.+?)(?:[.;]|$)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return [{"source": m[0].strip(), "relation": m[1].lower(), "target": m[2].strip()} for m in matches]

def normalize_relation(relation):
    synonyms = {
        "can cause": "causes",
        "can lead to": "causes",
        "can result in": "causes",
        "leads to": "causes",
        "results in": "causes",
        "need to be reviewed for": "related_to"
    }
    return synonyms.get(relation.lower().strip(), relation.lower().strip())

def extract_triples_from_image(contents):
    try:
        base64_img = contents.split(',')[1]

        prompt = (
            "You are analyzing a failure-related engineering diagram or report image.\n\n"
            "Extract all full causal chains relevant to constructing a Bowtie diagram.\n\n"
            "Each chain should represent a unique critical event and include:\n"
            "- One or more Causes (use a list)\n"
            "- One Mechanism per Cause (set as 'Unknown Mechanism' if unclear)\n"
            "- One Critical Event (the central failure or top event)\n"
            "- One or more Consequences (use a list)\n"
            "- One Preventive Barrier per Consequence (set as 'Unknown Barrier' if unclear)\n\n"
            "Return your answer as a JSON list of objects in the following format:\n"
            "[ { \"causes\": [...], \"mechanism\": ..., \"critical_event\": ..., ... } ]\n\n"
            "If any field is unknown, use:\n"
            "- 'Unknown Mechanism' for missing mechanisms\n"
            "- 'Unknown Critical Event' if unclear\n"
            "- 'Unknown Barrier' inside the list for unclear preventive barriers\n"
            "- 'Unknown Cause' inside the list for missing causes\n"
            "- 'Unknown Consequence' inside the list for missing consequences\n"
            "Do not include any explanations or text outside the JSON block."
        )

        mistral = Mistral(api_key=MISTRAL_API_KEY)
        response = mistral.chat.complete(
            model=PIXTRAL_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img}"}
            ]}]
        )

        response_text = response.choices[0].message.content.strip()
        print("🔍 Raw Pixtral Output:\n", repr(response_text))

        if "```json" in response_text:
            response_text = response_text.split("```json")[-1].strip()
        if "```" in response_text:
            response_text = response_text.split("```", 1)[0].strip()

        try:
            triples = json.loads(response_text)
            if not isinstance(triples, list):
                raise ValueError("Expected a list of triples")
        except Exception:
            print("⚠️ Falling back to regex parser")
            triples = parse_triples_fallback(response_text)

        with driver.session() as session:
            for t in triples:
                ce = t.get("critical_event", "Unknown Critical Event")
                causes = t.get("causes", ["Unknown Cause"])
                mechanism = t.get("mechanism", "Unknown Mechanism") + f" ({ce})"
                consequences = t.get("consequences", [])
                barriers = t.get("preventive_barriers", [])

                # Create critical event node scoped
                session.run("MERGE (e:Entity {name: $ce})", ce=ce)

                # Create mechanism node scoped to CE
                session.run("MERGE (m:Entity {name: $mech})", mech=mechanism)

                # Link causes
                for cause in causes:
                    session.run("""
                        MERGE (c:Entity {name: $cause})
                        MERGE (m:Entity {name: $mech})
                        MERGE (e:Entity {name: $ce})
                        MERGE (c)-[:CAUSES]->(m)
                        MERGE (m)-[:TRIGGERS]->(e)
                    """, cause=cause, mech=mechanism, ce=ce)

                for cons in consequences:
                    session.run("""
                        MERGE (co:Entity {name: $cons})
                        MERGE (e:Entity {name: $ce})
                        MERGE (e)-[:LEADS_TO]->(co)
                    """, cons=cons, ce=ce)

                for bar in barriers:
                    session.run("""
                        MERGE (b:Entity {name: $bar + ' (' + $ce + ')'})
                        MERGE (e:Entity {name: $ce})
                        MERGE (e)-[:MITIGATED_BY]->(b)
                    """, bar=bar, ce=ce)

        return html.Div([
            f"✅ Extracted and stored {len(triples)} Bowtie chains.",
            html.Pre(json.dumps(triples, indent=2, ensure_ascii=False))
        ])
    except Exception as e:
        return html.Div(f"❌ Error: {str(e)}", style={"color": "red"})

def generate_all_bowties_from_graph():
    try:
        all_jsons = []
        with driver.session() as session:
            ces = session.run("""
                MATCH (e:Entity)<-[:TRIGGERS]-(m:Entity)
                WHERE m.name CONTAINS e.name
                RETURN DISTINCT e.name AS ce
            """)
            ce_names = [r["ce"] for r in ces]

            for ce in ce_names:
                result = session.run("""
                    MATCH (cause:Entity)-[:CAUSES]->(mech:Entity)-[:TRIGGERS]->(e:Entity {name: $ce})
                    WHERE mech.name CONTAINS $ce
                    OPTIONAL MATCH (e)-[:LEADS_TO]->(cons:Entity)
                    OPTIONAL MATCH (e)-[:MITIGATED_BY]->(bar:Entity)
                    RETURN 
                        collect(DISTINCT cause.name) AS causes,
                        collect(DISTINCT mech.name) AS mechanism,
                        collect(DISTINCT cons.name) AS consequences,
                        collect(DISTINCT bar.name) AS preventive_barriers
                """, ce=ce)

                row = result.single()

                bowtie_json = {
                    "critical_event": ce,
                    "causes": [c for c in row["causes"] if c],
                    "mechanism": [clean_name(m, ce) for m in row["mechanism"] if m],
                    "consequences": [c for c in row["consequences"] if c],
                    "preventive_barriers": [clean_name(b, ce) for b in row["preventive_barriers"] if b]
                }

                # bowtie_json = {
                #     "critical_event": ce,
                #     "cause": [c for c in row["cause"] if c],
                #     "mechanism": [m for m in row["mechanism"] if m],
                #     "consequences": [c for c in row["consequences"] if c],
                #     "preventive_barriers": [b for b in row["preventive_barriers"] if b]
                # }
                all_jsons.append(bowtie_json)

        out_path = "all_bowties_flat.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_jsons, f, indent=2, ensure_ascii=False)

        return html.Div([
            f"✅ All flat Bowtie JSONs saved to `{out_path}`",
            html.Pre(json.dumps(all_jsons, indent=2, ensure_ascii=False))
        ])
    except Exception as e:
        return html.Div(f"❌ Error generating flat Bowtie JSONs: {str(e)}", style={"color": "red"})

def generate_bowtie_from_graph(central_event):
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (cause:Entity)-[:CAUSES]->(mech:Entity)-[:TRIGGERS]->(ce:Entity {name: $ce})
                WHERE mech.name CONTAINS $ce
                OPTIONAL MATCH (ce)-[:LEADS_TO]->(cons:Entity)
                OPTIONAL MATCH (ce)-[:MITIGATED_BY]->(bar:Entity)
                RETURN 
                    collect(DISTINCT cause.name) AS causes,
                    collect(DISTINCT mech.name) AS mechanism,
                    collect(DISTINCT cons.name) AS consequences,
                    collect(DISTINCT bar.name) AS preventive_barriers
            """, ce=central_event)

            row = result.single()

        bowtie_json = {
            "critical_event": central_event,
            "causes": [c for c in row["causes"] if c],
            "mechanism": [clean_name(m, central_event) for m in row["mechanism"] if m],
            "consequences": [c for c in row["consequences"] if c],
            "preventive_barriers": [clean_name(b, central_event) for b in row["preventive_barriers"] if b]
        }


        # bowtie_json = {
        #     "critical_event": central_event,
        #     "cause": [c for c in row["cause"] if c],
        #     "mechanism": [m for m in row["mechanism"] if m],
        #     "consequences": [c for c in row["consequences"] if c],
        #     "preventive_barriers": [b for b in row["preventive_barriers"] if b]
        # }

        out_path = f"{central_event.replace(' ', '_').lower()}_flat.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bowtie_json, f, indent=2, ensure_ascii=False)

        return html.Div([
            f"✅ Flat Bowtie JSON saved as `{out_path}`",
            html.Pre(json.dumps(bowtie_json, indent=2, ensure_ascii=False))
        ])
    except Exception as e:
        return html.Div(f"❌ Error building flat JSON from graph: {str(e)}", style={"color": "red"})

from dash import Output, Input

def register_additional_callbacks(app):
    @app.callback(
        Output("all-bowtie-json-output", "children"),
        Input("generate-all-bowties-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def run_generate_all_bowties(n_clicks):
        return generate_all_bowties_from_graph()
    @app.callback(
        Output("reset-output", "children"),
        Input("reset-graph-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def run_reset_graph(n_clicks):
        return reset_neo4j_graph()
# import json
# from dash import html
# from mistralai import Mistral
# from neo4j import GraphDatabase
# from config import MISTRAL_API_KEY, PIXTRAL_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
# import re

# # === Neo4j Connection ===
# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# def parse_triples_fallback(text):
#     pattern = r"(.+?)\s+(causes|leads to|results in|prevents|mitigates)\s+(.+?)(?:[.;]|$)"
#     matches = re.findall(pattern, text, flags=re.IGNORECASE)
#     return [{"source": m[0].strip(), "relation": m[1].lower(), "target": m[2].strip()} for m in matches]

# def normalize_relation(relation):
#     synonyms = {
#         "can cause": "causes",
#         "can lead to": "causes",
#         "can result in": "causes",
#         "leads to": "causes",
#         "results in": "causes",
#         "need to be reviewed for": "related_to"
#     }
#     return synonyms.get(relation.lower().strip(), relation.lower().strip())

# def extract_triples_from_image(contents):
#     try:
#         base64_img = contents.split(',')[1]

#         prompt = (
#             "You are analyzing a failure-related engineering diagram or report image.\n\n"
#             "Extract all full causal chains relevant to constructing a Bowtie diagram.\n\n"
#             "Each chain should include:\n"
#             "- Cause\n"
#             "- Mechanism (if identifiable, else use 'Unknown Mechanism')\n"
#             "- Critical Event (the top failure event)\n"
#             "- Consequences (one or more outcomes)\n"
#             "- Barriers (preventive or mitigative, if mentioned)\n\n"
#             "Return your answer as a JSON list of objects in the following format:\n\n"
#             "[\n"
#             "  {\n"
#             "    \"cause\": \"Thermal overload\",\n"
#             "    \"mechanism\": \"Material fatigue\",\n"
#             "    \"critical_event\": \"Seal rupture\",\n"
#             "    \"consequences\": [\"Fluid loss\", \"System failure\"],\n"
#             "    \"barriers\": [\"Relief valve\"]\n"
#             "  },\n"
#             "  {\n"
#             "    \"cause\": \"Vibration\",\n"
#             "    \"mechanism\": \"Unknown Mechanism\",\n"
#             "    \"critical_event\": \"Sensor detachment\",\n"
#             "    \"consequences\": [\"Data loss\"],\n"
#             "    \"barriers\": []\n"
#             "  }\n"
#             "]\n\n"
#             "If any field is unknown, use:\n"
#             "- 'Unknown Mechanism' for missing mechanisms\n"
#             "- 'Unknown Critical Event' if unclear\n"
#             "- An empty list [] for consequences or barriers\n\n"
#             "Do not include any explanations or text outside the JSON block."
#         )

#         mistral = Mistral(api_key=MISTRAL_API_KEY)
#         response = mistral.chat.complete(
#             model=PIXTRAL_MODEL,
#             messages=[{"role": "user", "content": [
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img}"}
#             ]}]
#         )

#         response_text = response.choices[0].message.content.strip()
#         print("🔍 Raw Pixtral Output:\n", repr(response_text))

#         if "```json" in response_text:
#             response_text = response_text.split("```json")[-1].strip()
#         if "```" in response_text:
#             response_text = response_text.split("```", 1)[0].strip()

#         try:
#             triples = json.loads(response_text)
#             if not isinstance(triples, list):
#                 raise ValueError("Expected a list of triples")
#         except Exception:
#             print("⚠️ Falling back to regex parser")
#             triples = parse_triples_fallback(response_text)

#         with driver.session() as session:
#             for t in triples:
#                 cause = t.get("cause", "Unknown Cause")
#                 mechanism = t.get("mechanism", "Unknown Mechanism")
#                 ce = t.get("critical_event", "Unknown Critical Event")
#                 consequences = t.get("consequences", [])
#                 barriers = t.get("barriers", [])

#                 session.run("""
#                     MERGE (c:Entity {name: $cause})
#                     MERGE (m:Entity {name: $mech})
#                     MERGE (e:Entity {name: $ce})
#                     MERGE (c)-[:CAUSES]->(m)
#                     MERGE (m)-[:TRIGGERS]->(e)
#                 """, cause=cause, mech=mechanism, ce=ce)

#                 for cons in consequences:
#                     session.run("""
#                         MERGE (e:Entity {name: $ce})
#                         MERGE (co:Entity {name: $cons})
#                         MERGE (e)-[:LEADS_TO]->(co)
#                     """, ce=ce, cons=cons)

#                 for bar in barriers:
#                     session.run("""
#                         MERGE (e:Entity {name: $ce})
#                         MERGE (b:Entity {name: $bar})
#                         MERGE (e)-[:MITIGATED_BY]->(b)
#                     """, ce=ce, bar=bar)

#         return html.Div([
#             f"✅ Extracted and stored {len(triples)} Bowtie chains.",
#             html.Pre(json.dumps(triples, indent=2, ensure_ascii=False))
#         ])
#     except Exception as e:
#         return html.Div(f"❌ Error: {str(e)}", style={"color": "red"})

# def generate_bowtie_from_graph(central_event):
#     try:
#         with driver.session() as session:
#             result = session.run("""
#                 MATCH (c:Entity {name: $ce})
#                 OPTIONAL MATCH (cause:Entity)-[:CAUSES]->(mech:Entity)-[:TRIGGERS]->(c)
#                 OPTIONAL MATCH (c)-[:MITIGATED_BY]->(bar:Entity)
#                 OPTIONAL MATCH (c)-[:LEADS_TO]->(cons:Entity)
#                 RETURN 
#                     collect(DISTINCT cause.name) AS causes,
#                     collect(DISTINCT mech.name) AS mechanisms,
#                     collect(DISTINCT bar.name) AS barriers,
#                     collect(DISTINCT cons.name) AS consequences
#             """, ce=central_event)

#             row = result.single()

#         bowtie_json = {
#             "critical_event": central_event,
#             "threats": [
#                 {"cause": c, "mechanism": m, "preventive_barriers": ["Barrier"]}
#                 for c, m in zip(row["causes"], row["mechanisms"]) if c and m
#             ],
#             "consequences": [c for c in row["consequences"] if c],
#             "mitigative_barriers": [b for b in row["barriers"] if b]
#         }

#         out_path = f"{central_event.replace(' ', '_').lower()}_from_graph.json"
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(bowtie_json, f, indent=2, ensure_ascii=False)

#         return html.Div([
#             f"✅ Bowtie JSON saved as `{out_path}`",
#             html.Pre(json.dumps(bowtie_json, indent=2, ensure_ascii=False))
#         ])
#     except Exception as e:
#         return html.Div(f"❌ Error building JSON from graph: {str(e)}", style={"color": "red"})

# from dash import Output, Input
