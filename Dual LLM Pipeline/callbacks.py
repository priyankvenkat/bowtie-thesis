import json
import base64
import os
import re
from dash import html
from mistralai import Mistral
from config import MISTRAL_API_KEY, PIXTRAL_MODEL, MISTRAL_MODEL
from prompts import get_prompt
from parsing import parse_llm_output, generate_mermaid_from_bowtie, expand_mechanism_structure
from llm_runner import run_llm_on_all_models
import random

available_models = {
    "LLaMA-3-8B": "DeepSeek-R1-GGUF/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
    # "Mistral-7B": "DeepSeek-R1-GGUF/Mistral-7B-Instruct-v0.3-Q5_K_S.gguf",
    # "Qwen-7B": "DeepSeek-R1-GGUF/Qwen2.5-7B-Instruct-1M-Q6_K.gguf",
    # "Mistral-Small-24B": "DeepSeek-R1-GGUF/Mistral-Small-3.1-24B-Instruct-2503-Q8_0.gguf",
    # "r1-Distill-Llama-8B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    # "r1-Distill-Qwen-7B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q6_K_L.gguf",
    # "r1-Distill-Qwen-32B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf",
    # "LLama-4-Scout": "DeepSeek-R1-GGUF/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf"
}
SHARED_SEEDS = [3407]
n_runs = 1

# Global variable to store table description
table_description_markdown = ""


def extract_table_callback(n_clicks, contents, structured_output, part_name):
    """
    Callback for extracting table description and markdown from an uploaded image.
    """
    global table_description_markdown
    if not contents:
        return "❌ Please upload an image first."

    try:
        base64_img = contents.split(',')[1]
        part = part_name if part_name else "component"
        part_safe = part.replace(" ", "_").lower()

        if structured_output:
            prompt = (
                f"""

You are analyzing an engineering failure report shown in this image for part {part}. Your task is to extract all the textual content in the image and turn it into text

"""


            )
        else:
            document_context = (
                f"This table appears in the context of a FMEA document for mechanical equipment. "
                f"It outlines various failure modes, their effects, and the associated risk levels related to {part}."
            )
            prompt = f"""
Given the image-based table and its original context below, do the following:

Original Document Context:
{document_context}

You are provided with an FMEA-style table that contains the columns: 'Failure Mode', 'Failure Cause', and 'Failure Effect'.

Your task is to convert this into two parts:
1. A markdown table
2. A structured description

Follow these instructions:

1. Reconstruct the table in **markdown format**, preserving:
   - Proper column headers
   - Clear alignment between Failure Mode, Failure Mechanism, and Failure Cause
   - Multiple rows per failure mode if there are multiple causes per mechanism
   - Insert blank cells (` `) if a row shares the same failure mode or cause from above
   - Use monospace layout where needed to ensure formatting remains aligned

2. Provide a structured description of the table:
   - For each **Failure Mode**, list all associated **Causes** that lead to it
   - Then list the **Effects** that result from that Failure Mode, if none exist, use "No Effects"
   - Split compound phrases when they contain multiple independent failures, mechanisms, causes or effects joined by conjunctions or separators.
   - Use your judgment to split compound causes and consequences if they describe distinct technical events, such as opening/closing or 
   - Use bullet points for clarity and avoid repeating the same Failure Mode unnecessarily in the description.

Ensure the markdown table visually matches the original structure and is readable in plain text.


"""

        mistral_pix = Mistral(api_key=MISTRAL_API_KEY)
        response = mistral_pix.chat.complete(
            model=PIXTRAL_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img}"}
            ]}]
        )

        table_description_markdown = response.choices[0].message.content.strip()
        with open(f"{part_safe}_extracted_table.md", "w") as f:
            f.write(table_description_markdown)
        return table_description_markdown
    except Exception as e:
        return f"❌ Error: {str(e)}"

def update_prompt_callback(prompt_type, part_name):
    """
    Callback to update the prompt text based on the selected prompt type.
    """
    part = part_name or "[COMPONENT NAME HERE]"
    return get_prompt(prompt_type, part)


# Smart JSON extractor

def extract_json_from_text(text):
    text = re.sub(r"```(json)?", "", text).strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)

    return match.group(1) if match else None


# def generate_json_callback(n_clicks, custom_prompt, part_name):
#     global table_description_markdown
#     if not table_description_markdown:
#         return html.Div("❌ Please complete Step 1 first.", style={"color": "red"})

#     try:
#         part = part_name if part_name else "component"
#         part_safe = part.replace(" ", "_").lower()

#         outputs = run_llm_on_all_models(custom_prompt, table_description_markdown, available_models)
#         results_ui = []

#         for model_name, raw_text in outputs.items():
#             cleaned = extract_json_from_text(raw_text)

#             if not cleaned:
#                 fallback_path = f"{part_safe}_{model_name}_bowtie_raw.txt"
#                 with open(fallback_path, "w") as f:
#                     f.write(raw_text)
#                 results_ui.append(html.Div([
#                     html.H5(f"⚠️ {model_name} - Couldn't extract JSON, saved raw output: {fallback_path}"),
#                     html.Pre(raw_text)
#                 ]))
#                 continue

#             try:
#                 json_obj = json.loads(cleaned)

#                 if isinstance(json_obj, dict):
#                     json_obj = [json_obj]  # Ensure list format

#                 normalized = []
#                 for diagram in json_obj:
#                     # Preserve LLM-generated threats
#                     threats = []
#                     for threat in diagram.get("threats", []):
#                         threats.append({
#                             "mechanism": threat.get("mechanism", ""),
#                             "cause": threat.get("cause", ""),
#                             "preventive_barriers": threat.get("preventive_barriers", "")
#                         })

#                     # Normalize consequences into a list of strings
#                     consequences_raw = diagram.get("consequences", [])
#                     consequences = []

#                     if isinstance(consequences_raw, list):
#                         for c in consequences_raw:
#                             if isinstance(c, str):
#                                 consequences.extend([line.strip() for line in c.split("\n") if line.strip()])
#                             elif isinstance(c, dict) and "effect" in c:
#                                 consequences.extend([line.strip() for line in c["effect"].split("\n") if line.strip()])
#                     elif isinstance(consequences_raw, str):
#                         consequences = [line.strip() for line in consequences_raw.split("\n") if line.strip()]

#                     normalized.append({
#                         "critical_event": diagram.get("critical_event", ""),
#                         "threats": threats,
#                         "consequences": consequences
#                     })

#                 output_path = f"{part_safe}_{model_name}_bowtie.json"
#                 with open(output_path, "w") as f:
#                     json.dump(normalized, f, indent=2)

#                 results_ui.append(html.Div([
#                     html.H5(f"✅ {model_name} - Saved: {output_path}"),
#                     html.Pre(json.dumps(normalized, indent=2))
#                 ]))

#             except Exception:
#                 fallback_path = f"{part_safe}_{model_name}_bowtie_raw.txt"
#                 with open(fallback_path, "w") as f:
#                     f.write(raw_text)
#                 results_ui.append(html.Div([
#                     html.H5(f"⚠️ {model_name} - JSON parsing failed, saved raw output: {fallback_path}"),
#                     html.Pre(raw_text)
#                 ]))

#         return html.Div(results_ui)

#     except Exception as e:
#         return html.Div(f"❌ Error generating Bowtie JSON: {str(e)}", style={"color": "red"})

def generate_json_callback(n_clicks, custom_prompt, part_name, n_runs=n_runs):
    global table_description_markdown
    if not table_description_markdown:
        return html.Div("❌ Please complete Step 1 first.", style={"color": "red"})

    try:
        part = part_name if part_name else "component"
        part_safe = part.replace(" ", "_").lower()

        # ✅ Use predefined shared seeds
        seeds = SHARED_SEEDS[:n_runs]
        print(f"🌱 Using predefined seeds: {seeds}")

        # ✅ Run LLM inference for each model using the fixed seeds
        outputs = run_llm_on_all_models(custom_prompt, table_description_markdown, available_models, seeds=seeds)

        results_ui = []

        for model_name, model_runs in outputs.items():
            for run_idx, run_info in enumerate(model_runs):
                seed = run_info["seed"]
                raw_text = run_info["output"]
                cleaned = extract_json_from_text(raw_text)

                if not cleaned:
                    fallback_path = f"{part_safe}_{model_name}_run_{run_idx+1}_seed_{seed}.txt"
                    with open(fallback_path, "w") as f:
                        f.write(raw_text)
                    results_ui.append(html.Div([
                        html.H5(f"⚠️ {model_name} - Run {run_idx+1} (Seed {seed}) - Couldn't extract JSON, saved raw output: {fallback_path}"),
                        html.Pre(raw_text)
                    ]))
                    continue

                try:
                    json_obj = json.loads(cleaned)
                    if isinstance(json_obj, dict):
                        json_obj = [json_obj]  # Ensure list format

                    normalized = []
                    for diagram in json_obj:
                        threats = []
                        for threat in diagram.get("threats", []):
                            threats.append({
                                "mechanism": threat.get("mechanism", ""),
                                "cause": threat.get("cause", ""),
                                "preventive_barriers": threat.get("preventive_barriers", "")
                            })

                        consequences_raw = diagram.get("consequences", [])
                        consequences = []

                        if isinstance(consequences_raw, list):
                            for c in consequences_raw:
                                if isinstance(c, str):
                                    consequences.extend([line.strip() for line in c.split("\n") if line.strip()])
                                elif isinstance(c, dict) and "effect" in c:
                                    consequences.extend([line.strip() for line in c["effect"].split("\n") if line.strip()])
                        elif isinstance(consequences_raw, str):
                            consequences = [line.strip() for line in consequences_raw.split("\n") if line.strip()]

                        normalized.append({
                            "critical_event": diagram.get("critical_event", ""),
                            "threats": threats,
                            "consequences": consequences
                        })

                    output_path = f"{part_safe}_{model_name}_run_{run_idx+1}_seed_{seed}_bowtie.json"
                    with open(output_path, "w") as f:
                        json.dump(normalized, f, indent=2)

                    results_ui.append(html.Div([
                        html.H5(f"✅ {model_name} - Run {run_idx+1} (Seed {seed}) - Saved: {output_path}"),
                        html.Pre(json.dumps(normalized, indent=2))
                    ]))

                except Exception:
                    fallback_path = f"{part_safe}_{model_name}_run_{run_idx+1}_seed_{seed}_bowtie_raw.txt"
                    with open(fallback_path, "w") as f:
                        f.write(raw_text)
                    results_ui.append(html.Div([
                        html.H5(f"⚠️ {model_name} - Run {run_idx+1} (Seed {seed}) - JSON parsing failed, saved raw output: {fallback_path}"),
                        html.Pre(raw_text)
                    ]))

        return html.Div(results_ui)

    except Exception as e:
        return html.Div(f"❌ Error generating Bowtie JSON: {str(e)}", style={"color": "red"})


def generate_mermaid_callback(n_clicks, manual_input, file_contents):
    """
    Callback to generate a Mermaid diagram from the provided JSON.
    """
    try:
        json_data = None
        if file_contents:
            content_type, content_string = file_contents.split(',')
            decoded = base64.b64decode(content_string).decode("utf-8")
            json_data = json.loads(decoded)
        elif manual_input:
            json_data = json.loads(manual_input.replace("'", '"'))

        if json_data is None:
            return "❌ Please upload or paste valid JSON."

        # Automatically expand nested structure (mechanism → causes) if needed
        if isinstance(json_data.get("threats", [{}])[0], dict) and "causes" in json_data["threats"][0]:
            parsed = expand_mechanism_structure(json_data)
        else:
            parsed = parse_llm_output(json_data)

        mermaid = generate_mermaid_from_bowtie(parsed)
        return html.Pre(mermaid)
    except Exception as e:
        return html.Div(f"❌ Error parsing input: {e}", style={"color": "red"})
