import json
import base64
import os
import re
from dash import html
from mistralai import Mistral
from config import MISTRAL_API_KEY, PIXTRAL_MODEL, MISTRAL_MODEL
from prompts import get_prompt
from parsing import parse_llm_output, generate_mermaid_from_bowtie, expand_mechanism_structure
from kg_pipeline import extract_triples_from_image, generate_bowtie_from_graph  # NEW

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
                """You are analyzing an engineering failure report shown in this image for part {part}. Your task is to extract all causal information relevant to building a Bowtie diagram. For each causal relationship, output a sentence in the following format:

[Cause] leads to [Mechanism] leads to [Critical Event], which results in [Consequence]. Barriers such as [Barrier] may mitigate or prevent this.

If any elements are missing in the image (e.g., no barriers), just state unknown barrier, consequece, cause, mechanism or central event."""
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

1. Reconstruct the table in markdown format, preserving:
   - Proper column headers
   - Clear alignment between Failure Mode, Failure Mechanism, and Failure Cause
   - Multiple rows per failure mode if there are multiple causes per mechanism
   - Insert blank cells (` `) if a row shares the same failure mode or cause from above
   - Use monospace layout where needed to ensure formatting remains aligned

2. Provide a structured description of the table:
   - For each Failure Mode, list all associated Causes that lead to it
   - Then list the Effects that result from that Failure Mode, if none exist, use "No Effects"
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

def generate_json_callback(n_clicks, custom_prompt, part_name):
    """
    Callback to generate the Bowtie JSON using the provided prompt and table description.
    If the LLM output is a list of diagrams, each diagram is saved as its own JSON file.
    """
    global table_description_markdown
    if not table_description_markdown:
        return html.Div("❌ Please complete Step 1 first.", style={"color": "red"})

    try:
        part = part_name if part_name else "component"
        part_safe = part.replace(" ", "_").lower()

        # Step 1: Call LLM
        mistral_large = Mistral(api_key=MISTRAL_API_KEY)
        response = mistral_large.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{
                "role": "user",
                "content": f"{custom_prompt}\n\n{table_description_markdown}"
            }]
        )

        raw_text = response.choices[0].message.content.strip()
        print("🔍 Raw LLM Response:\n", repr(raw_text))

        # Step 2: Try extracting JSON from response
        cleaned = re.sub(r"```json|```", "", raw_text).strip()
        cleaned = cleaned.replace("'", '"')  # Fix common issues

        try:
            json_obj = json.loads(cleaned)
            
            # Check if the output is a list of diagrams.
            if isinstance(json_obj, list):
                saved_files = []
                for idx, diagram in enumerate(json_obj):
                    output_path = f"{part_safe}_bowtie_{idx+1}.json"
                    with open(output_path, "w") as f:
                        json.dump(diagram, f, indent=2)
                    saved_files.append(output_path)
                
                # Prepare a response showing all saved files.
                file_list = ", ".join(saved_files)
                return html.Div([
                    f"✅ Bowtie JSON saved as separate files: {file_list}",
                    html.Pre(json.dumps(json_obj, indent=2))
                ])
            else:
                # Single JSON object case.
                output_path = f"{part_safe}_bowtie.json"
                with open(output_path, "w") as f:
                    json.dump(json_obj, f, indent=2)
                return html.Div([
                    f"✅ Bowtie JSON saved as {output_path}",
                    html.Pre(json.dumps(json_obj, indent=2))
                ])

        except Exception as json_err:
            fallback_path = f"{part_safe}_bowtie_raw.txt"
            with open(fallback_path, "w") as f:
                f.write(raw_text)
            return html.Div([
                f"⚠️ Couldn't parse JSON. Raw output saved to `{fallback_path}`",
                html.Pre(raw_text)
            ])

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

# === NEW CALLBACKS FOR VISION + GRAPH PIPELINE ===
def extract_triples_callback(contents):
    return extract_triples_from_image(contents)

def graph_to_json_callback(ce_name):
    return generate_bowtie_from_graph(ce_name)


# import json
# import base64
# import os
# import re
# from dash import html
# from mistralai import Mistral
# from config import MISTRAL_API_KEY, PIXTRAL_MODEL, MISTRAL_MODEL
# from prompts import get_prompt
# from parsing import parse_llm_output, generate_mermaid_from_bowtie, expand_mechanism_structure
# from kg_pipeline import extract_triples_from_image, generate_bowtie_from_graph

# # Global variable to store table description
# table_description_markdown = ""


# def extract_triples_callback(contents):
#     return extract_triples_from_image(contents)

# def graph_to_json_callback(ce_name):
#     return generate_bowtie_from_graph(ce_name)


# def extract_table_callback(n_clicks, contents, structured_output, part_name):
#     """
#     Callback for extracting table description and markdown from an uploaded image.
#     """
#     global table_description_markdown
#     if not contents:
#         return "❌ Please upload an image first."

#     try:
#         base64_img = contents.split(',')[1]
#         part = part_name if part_name else "component"
#         part_safe = part.replace(" ", "_").lower()

#         if structured_output:
#             prompt = (
#                 """You are analyzing an engineering failure report shown in this image for part {part}. Your task is to extract all causal information relevant to building a Bowtie diagram. For each causal relationship, output a sentence in the following format:

# [Cause] leads to [Mechanism] leads to [Critical Event], which results in [Consequence]. Barriers such as [Barrier] may mitigate or prevent this.

# If any elements are missing in the image (e.g., no barriers), just state unknown barrier, consequece, cause, mechanism or central event."""
#             )
#         else:
#             document_context = (
#                 f"This table appears in the context of a FMEA document for mechanical equipment. "
#                 f"It outlines various failure modes, their effects, and the associated risk levels related to {part}."
#             )
#             prompt = f"""

# Given the image-based table and its original context below, do the following:

# Original Document Context:
# {document_context}

# You are provided with an FMEA-style table that contains the columns: 'Failure Mode', 'Failure Cause', and 'Failure Effect'.

# Your task is to convert this into two parts:
# 1. A markdown table
# 2. A structured description

# Follow these instructions:

# 1. Reconstruct the table in markdown format, preserving:
#    - Proper column headers
#    - Clear alignment between Failure Mode, Failure Mechanism, and Failure Cause
#    - Multiple rows per failure mode if there are multiple causes per mechanism
#    - Insert blank cells (` `) if a row shares the same failure mode or cause from above
#    - Use monospace layout where needed to ensure formatting remains aligned

# 2. Provide a structured description of the table:
#    - For each Failure Mode, list all associated Causes that lead to it
#    - Then list the Effects that result from that Failure Mode, if none exist, use "No Effects"
#    - Split compound phrases when they contain multiple independent failures, mechanisms, causes or effects joined by conjunctions or separators.
#    - Use your judgment to split compound causes and consequences if they describe distinct technical events, such as opening/closing or 
#    - Use bullet points for clarity and avoid repeating the same Failure Mode unnecessarily in the description.

# Ensure the markdown table visually matches the original structure and is readable in plain text.
# """

#         mistral_pix = Mistral(api_key=MISTRAL_API_KEY)
#         response = mistral_pix.chat.complete(
#             model=PIXTRAL_MODEL,
#             messages=[{"role": "user", "content": [
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img}"}
#             ]}]
#         )

#         table_description_markdown = response.choices[0].message.content.strip()
#         with open(f"{part_safe}_extracted_table.md", "w") as f:
#             f.write(table_description_markdown)
#         return table_description_markdown
#     except Exception as e:
#         return f"❌ Error: {str(e)}"

# def update_prompt_callback(prompt_type, part_name):
#     """
#     Callback to update the prompt text based on the selected prompt type.
#     """
#     part = part_name or "[COMPONENT NAME HERE]"
#     return get_prompt(prompt_type, part)


# def generate_json_callback(n_clicks, custom_prompt, part_name):
#     """
#     Callback to generate the Bowtie JSON using the provided prompt and table description.
#     If the LLM output is a list of diagrams, each diagram is saved as its own JSON file.
#     """
#     global table_description_markdown
#     if not table_description_markdown:
#         return html.Div("❌ Please complete Step 1 first.", style={"color": "red"})

#     try:
#         part = part_name if part_name else "component"
#         part_safe = part.replace(" ", "_").lower()

#         # Step 1: Call LLM
#         mistral_large = Mistral(api_key=MISTRAL_API_KEY)
#         response = mistral_large.chat.complete(
#             model=MISTRAL_MODEL,
#             messages=[{
#                 "role": "user",
#                 "content": f"{custom_prompt}\n\n{table_description_markdown}"
#             }]
#         )

#         raw_text = response.choices[0].message.content.strip()
#         print("🔍 Raw LLM Response:\n", repr(raw_text))

#         # Step 2: Try extracting JSON from response
#         cleaned = re.sub(r"```json|```", "", raw_text).strip()
#         cleaned = cleaned.replace("'", '"')  # Fix common issues

#         try:
#             json_obj = json.loads(cleaned)
            
#             # Check if the output is a list of diagrams.
#             if isinstance(json_obj, list):
#                 saved_files = []
#                 for idx, diagram in enumerate(json_obj):
#                     output_path = f"{part_safe}_bowtie_{idx+1}.json"
#                     with open(output_path, "w") as f:
#                         json.dump(diagram, f, indent=2)
#                     saved_files.append(output_path)
                
#                 # Prepare a response showing all saved files.
#                 file_list = ", ".join(saved_files)
#                 return html.Div([
#                     f"✅ Bowtie JSON saved as separate files: {file_list}",
#                     html.Pre(json.dumps(json_obj, indent=2))
#                 ])
#             else:
#                 # Single JSON object case.
#                 output_path = f"{part_safe}_bowtie.json"
#                 with open(output_path, "w") as f:
#                     json.dump(json_obj, f, indent=2)
#                 return html.Div([
#                     f"✅ Bowtie JSON saved as {output_path}",
#                     html.Pre(json.dumps(json_obj, indent=2))
#                 ])

#         except Exception as json_err:
#             fallback_path = f"{part_safe}_bowtie_raw.txt"
#             with open(fallback_path, "w") as f:
#                 f.write(raw_text)
#             return html.Div([
#                 f"⚠️ Couldn't parse JSON. Raw output saved to `{fallback_path}`",
#                 html.Pre(raw_text)
#             ])

#     except Exception as e:
#         return html.Div(f"❌ Error generating Bowtie JSON: {str(e)}", style={"color": "red"})


# def generate_mermaid_callback(n_clicks, manual_input, file_contents):
#     """
#     Callback to generate a Mermaid diagram from the provided JSON.
#     """
#     try:
#         json_data = None
#         if file_contents:
#             content_type, content_string = file_contents.split(',')
#             decoded = base64.b64decode(content_string).decode("utf-8")
#             json_data = json.loads(decoded)
#         elif manual_input:
#             json_data = json.loads(manual_input.replace("'", '"'))

#         if json_data is None:
#             return "❌ Please upload or paste valid JSON."

#         # Automatically expand nested structure (mechanism → causes) if needed
#         if isinstance(json_data.get("threats", [{}])[0], dict) and "causes" in json_data["threats"][0]:
#             parsed = expand_mechanism_structure(json_data)
#         else:
#             parsed = parse_llm_output(json_data)

#         mermaid = generate_mermaid_from_bowtie(parsed)
#         return html.Pre(mermaid)
#     except Exception as e:
#         return html.Div(f"❌ Error parsing input: {e}", style={"color": "red"})
