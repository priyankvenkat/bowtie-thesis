# Graph extraction from image
import json
import requests

def extract_graph_from_image(encoded_image, api_key):
    user_prompt = """
    The image shows a Bowtie diagram. Your task is to extract it as a directed graph using two fields: "nodes" and "edges".
Please extract **every node and every edge exactly as shown in the diagram**, even if:
- A node label (e.g., "Mechanism" or "Barrier") appears multiple times
- Edges lead to or from repeated nodes

Treat each visual instance of a node as distinct.

Output the graph in the following JSON format:
{{
  "nodes": ["Node A", "Node B", "Node C"],
  "edges": [["Node A", "Node B"], ["Node B", "Node C"]]
}}
    """
    payload = {
        "model": "pixtral-large-latest",
        "messages": [
            {"role": "system", "content": "You are a system safety expert helping analyze bowtie diagrams."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]
            }
        ],
        "temperature": 0.3
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    r = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
    try:
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        json_str = content.split("```json")[1].split("```")[0]
        return json.loads(json_str)
    except Exception as e:
        raise RuntimeError(f"Pixtral API failed: {r.text if r.status_code != 200 else str(e)}")
