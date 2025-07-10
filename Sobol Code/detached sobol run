from pathlib import Path
import os
import json
import random
import pandas as pd
import json5
import re
import math
from llama_cpp import Llama
from SALib.sample import sobol as sobol_sample

# === Setup ===
PART_NAME = "individual sensor"
N = 1024
SEED = 2242
NUM_SEEDS = 1
RESULTS_DIR = "results_categorical_v2"
OUTPUT_DIR = "sobol_outputs_categorical_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
SHARED_SEEDS = [random.randint(1, 999999) for _ in range(50000)]

available_models = {
    "LLaMA-3-8B": "DeepSeek-R1-GGUF/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
    "Mistral-7B": "DeepSeek-R1-GGUF/Mistral-7B-Instruct-v0.3-Q5_K_S.gguf",
    "Qwen-7B": "DeepSeek-R1-GGUF/Qwen2.5-7B-Instruct-1M-Q6_K.gguf",
}

sobol_problem = {
    'num_vars': 3,
    'names': ['prompt_type', 'prompt_strictness_level', 'context_type'],
    'bounds': [[0, 3], [0, 3], [0, 3]]
}

prompt_type_map = {0: "zero", 1: "few", 2: "cot"}
prompt_strictness_map = {0: "low", 1: "medium", 2: "high"}
context_type_map = {0: "ocr", 1: "rag", 2: "vision"}

# === Static Contexts ===

OCR_CONTEXT = """
Table 19-2. Typical Failure Modes of Individual Sensors

FAILURE MODES FAILURE CAUSES FAILURE EFFECT

Incorrect signal from sensor | - Reduced signal level Potential processing error
- Impedance mismatch
- A/D conversion error

Loss of signal from sensor _| - Chip failure Loss of signal to processor element - Corroded sensor

Complete loss of signal in _| - Broken wire Loss of signal to processor
transmission line - Fiber optic, RF interruption

Signal error in transmission | - Power line interference Potential processing error
line - Contaminants in fluid system element

Incorrect signal in - Error in algorithm Processing error
computation device

Power supply loss of - Power supply malfunction Loss of signal to processor
voltage

Improper response to - Incorrect interpretation Potential system malfunction
information recipient - System malfunction
- Error in algorithm

Calibration error - Software error Potential system malfunction
- Error in algorithm

Battery energy [Battery energy depletion —_| - | Battery malfunction | malfunction Loss of Loss of signal to processor _| to processor
"""

RAG_CONTEXT = """
### Table: Table 19-2.  Typical Failure Modes of Individual Sensors FAILURE MODES 

Markdown Table:
| FAILURE MODES | FAILURE CAUSES | FAILURE EFFECT |
| --- | --- | --- |
| Incorrect signal from sensor; element | Reduced signal level; Impedance mismatch; A/D conversion error | Potential processing error |
| Loss of signal from sensor; element | Chip failure; Corroded sensor | Loss of signal to processor |
| Complete loss of signal in; transmission line | Broken wire; Fiber optic, RF interruption | Loss of signal to processor |
| Signal error in transmission; line | Power line interference; Contaminants in fluid system | Potential processing error |
| Incorrect signal in; computation device | Error in algorithm | Processing error |
| Power supply loss of; voltage | Power supply malfunction | Loss of signal to processor |
| Improper response to; information recipient | Incorrect interpretation; System malfunction; Error in algorithm | Potential system malfunction |
| Calibration error | Software error; Error in algorithm | Potential system malfunction |
| Battery energy depletion | Battery malfunction | Loss of signal to processor |"""

VISION_CONTEXT = """
### 2. Structured Description

**Critical event: Incorrect signal from sensor element**
- Causes:
- Reduced signal level
- Impedance mismatch
- A/D conversion error
- Effects:
- Potential processing error

**Critical event: Loss of signal from sensor element**
- Causes:
- Chip failure
- Corroded sensor
- Effects:
- Loss of signal to processor

**Critical event: Complete loss of signal in transmission line**
- Causes:
- Broken wire
- Fiber optic, RF interruption
- Effects:
- Loss of signal to processor

**Critical event: Signal error in transmission line**
- Causes:
- Power line interference
- Contaminants in fluid system
- Effects:
- Potential processing error

**Critical event: Incorrect signal in computation device**
- Causes:
- Error in algorithm
- Effects:
- Processing error

**Critical event: Power supply loss of voltage**
- Causes:
- Power supply malfunction
- Effects:
- Loss of signal to processor

**Critical event: Improper response to information recipient**
- Causes:
- Incorrect interpretation
- System malfunction
- Error in algorithm
- Effects:
- Potential system malfunction

**Critical event: Calibration error**
- Causes:
- Software error
- Error in algorithm
- Effects:
- Potential system malfunction

**Critical event: Battery energy depletion**
- Causes:
- Battery malfunction
- Effects:
- Loss of signal to processor
"""


def get_context(index):
    """
    Retrieve the static context string corresponding to a Sobol sample index.

    Args:
        index (float): Sobol-generated value for context_type.
                       After flooring and clamping, selects one of OCR, RAG, or Vision.

    Returns:
        str: The context text to include in the LLM prompt.
    """
    if index == 0:
        return OCR_CONTEXT
    elif index == 1:
        return RAG_CONTEXT
    elif index == 2:
        return VISION_CONTEXT
    return "Unknown Context"

SCRIPT_DIR = Path(__file__).parent.resolve()
PROMPT_BASE = SCRIPT_DIR / "prompt_templates"


# def load_prompt(prompt_type_val, strictness_val, part_name, context_text):
#     prompt_type = prompt_type_map[min(math.floor(prompt_type_val), 2)]
#     strictness = prompt_strictness_map[min(math.floor(strictness_val), 2)]
#     prompt_path = Path(f"prompt_templates/{prompt_type}/{strictness}.txt")

#     with open(prompt_path, "r") as f:
#         base_prompt = f.read().replace("{part_name}", part_name)

#     full_prompt = f"{base_prompt.strip()}\n\n--- FMEA Context Start ---\n{context_text.strip()}\n--- FMEA Context End ---\n\nReturn only valid JSON, no python script, no function — just the JSON output."
#     return full_prompt, prompt_type, strictness

def load_prompt(prompt_type_val, strictness_val, part_name, context_text):
    """
    Load and construct the LLM prompt based on Sobol parameters.

    Args:
        prompt_type_val (float): Sobol sample for prompt_type variable.
        strictness_val (float): Sobol sample for prompt_strictness_level variable.
        part_name (str): The name of the component/part to inject into prompt.
        context_text (str): The static context text for this run.

    Returns:
        tuple:
            - full_prompt (str): The assembled prompt including context.
            - prompt_type (str): One of 'zero', 'few', 'cot'.
            - strictness (str): One of 'low', 'medium', 'high'.

    Raises:
        FileNotFoundError: If the template file is missing.
    """
    prompt_type = prompt_type_map[min(math.floor(prompt_type_val), 2)]
    strictness = prompt_strictness_map[min(math.floor(strictness_val), 2)]
    prompt_path = PROMPT_BASE / prompt_type / f"{strictness}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

    with open(prompt_path, "r") as f:
        base_prompt = f.read().replace("{part_name}", part_name)

    full_prompt = f"{base_prompt.strip()}\n\n--- FMEA Context Start ---\n{context_text.strip()}\n--- FMEA Context End ---\n\nReturn only valid JSON, no python script, no function — just the JSON output."
    return full_prompt, prompt_type, strictness

def ask_model_paramset(prompt, model_path, seed):
    """
    Send a prompt to the LLM with specified decoding parameters and return raw output.

    Args:
        prompt (str): The text prompt to send.
        model_path (str): Path to the GGUF model file.
        seed (int): Random seed for deterministic sampling.

    Returns:
        str: The raw textual response from the LLM.
    """
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_threads=16,
        n_ctx=6000,
        temp=0.6,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.0,
        seed=seed,
    )
    response = llm(prompt, max_tokens=4000)
    return response["choices"][0]["text"]

def try_parse_json_flexibly(raw_output, model_name, run_id, seed):
    """
    Attempt to extract and save valid JSON from raw LLM output, with fallback logging.

    Strategies:
      1. Remove any            '</think>' markers.
      2. Regex for JSON code blocks.
      3. Regex for first {...} or [...] JSON structure.
      4. Try minor fixes (trailing comma, wrapping in array).

    Args:
        raw_output (str): The text returned by the LLM.
        model_name (str): Identifier for the model run.
        run_id (int): Sobol run index.
        seed (int): Seed used for this run.

    Returns:
        (success, path):
            success (bool): True if JSON parsed and saved; False otherwise.
            path (str): File path of saved JSON or raw text.
    """
    if "</think>" in raw_output:
        raw_output = raw_output.split("</think>")[-1].strip()

    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', raw_output)
    if not match:
        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw_output, re.DOTALL)

    if not match:
        txt_path = os.path.join(RESULTS_DIR, f"bowtie_{model_name}_run_{run_id}_seed_{seed}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_output)
        return False, txt_path

    json_str = match.group(1).strip()
    if json_str.startswith("```") or json_str.endswith("```"):
        json_str = json_str.strip("`").strip()

    for fix_attempt in [json_str, json_str.strip().rstrip(',') + '}', '[' + json_str + ']']:
        try:
            parsed = json5.loads(fix_attempt)
            suffix = 'fixed' if fix_attempt != json_str else 'raw'
            out_path = os.path.join(RESULTS_DIR, f"bowtie_{model_name}_run_{run_id}_seed_{seed}_{suffix}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            return True, out_path
        except Exception:
            continue

    txt_path = os.path.join(RESULTS_DIR, f"bowtie_{model_name}_run_{run_id}_seed_{seed}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_output)
    return False, txt_path

def run_sobol():
    """
    Main Sobol sampling loop to evaluate sensitivity of LLM-based Bowtie generation.

    Workflow:
      1. Generate Sobol sample matrix with first, total, and second order.
      2. For each sample (run_id):
         a. Derive context, prompt_type, and strictness.
         b. For each seed and each model:
            - Invoke the model.
            - Attempt to parse and save JSON.
            - Record status and file paths in metadata.
      3. Save metadata as a CSV for later analysis.

    Returns:
        Workable JSON files for each model run, and a metadata CSV summarizing all runs.
    """
    param_values = sobol_sample.sample(sobol_problem, N, calc_second_order=True)
    metadata_rows = []

    for run_id, (ptype_val, strict_val, ctype_val) in enumerate(param_values):
        context_index = min(math.floor(ctype_val), 2)
        context_text = get_context(context_index)
        prompt_text, prompt_type, strictness = load_prompt(ptype_val, strict_val, PART_NAME, context_text)

        for seed_ix in range(NUM_SEEDS):
            seed = SHARED_SEEDS[run_id * NUM_SEEDS + seed_ix]
            for model_name, model_path in available_models.items():
                print(f"[{model_name}] Run {run_id}, Prompt: {prompt_type}-{strictness}, Context: {context_type_map[context_index]}, Seed: {seed}")

                try:
                    raw_output = ask_model_paramset(prompt_text, model_path, seed)
                    success, path = try_parse_json_flexibly(raw_output, model_name, run_id, seed)
                    metadata_rows.append({
                        "run_id": run_id,
                        "model": model_name,
                        "prompt_type": prompt_type,
                        "prompt_strictness": strictness,
                        "context_type": context_type_map[context_index],
                        "seed": seed,
                        "json_path": path,
                        "status": "success" if success else "fail"
                    })
                except Exception as e:
                    metadata_rows.append({
                        "run_id": run_id,
                        "model": model_name,
                        "prompt_type": prompt_type,
                        "prompt_strictness": strictness,
                        "context_type": context_type_map[context_index],
                        "seed": seed,
                        "json_path": "",
                        "status": f"error: {e}"
                    })

    df = pd.DataFrame(metadata_rows)
    out_path = os.path.join(OUTPUT_DIR, f"sobol_metadata_{PART_NAME.replace(' ', '_')}.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved Sobol metadata to {out_path}")

if __name__ == '__main__':
    run_sobol()



# import os
# import json
# import random
# import pandas as pd
# from pathlib import Path
# from llama_cpp import Llama
# from SALib.sample import sobol as sobol_sample
# import json5
# import re

# # === Setup ===
# PART_NAME = "individual sensor"
# PROMPT_TYPE = "zero"
# N = 64
# SEED = 2242
# NUM_SEEDS = 1
# RESULTS_DIR = "results_v2"
# OUTPUT_DIR = "sobol_outputs_v2"
# os.makedirs(RESULTS_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# random.seed(SEED)
# SHARED_SEEDS = [random.randint(1, 999999) for _ in range(50000)]

# available_models = {
#     "LLaMA-3-8B": "DeepSeek-R1-GGUF/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
#     "Mistral-7B": "DeepSeek-R1-GGUF/Mistral-7B-Instruct-v0.3-Q5_K_S.gguf",
#     "Qwen-7B": "DeepSeek-R1-GGUF/Qwen2.5-7B-Instruct-1M-Q6_K.gguf",
#     # "r1-Distill-Llama-8B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
#     # "r1-Distill-Qwen-7B": "DeepSeek-R1-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q6_K_L.gguf",
# }

# sobol_problem = {
#     'num_vars': 3,
#     'names': ['temp', 'top_p', 'prompt_strictness_level'],
#     'bounds': [[0.1, 1.0], [0.1, 1.0], [0, 3]],
# }
# PROMPT_VARIANTS = [

#     # Level 1 – Moderate Scaffold
#     """Please convert the following content into a Bowtie diagram JSON. 
# At a minimum, each JSON object should contain: 'critical_event', 'causes', 'mechanism', 'preventive_barrier', 'consequences'.
# Return all outputs in a JSON file. Do not include any extra explanation. """,
# #     """Given FMEA data, summarize each failure mode as a structured object with the following keys:
# # - "critical_event"
# # - "cause"
# # - "mechanism"
# # - "preventive_barriers"
# # - "consequences"

# # Return all outputs in a JSON array. Do not include any extra explanation.""",

#     # Level 2 – Detailed Scaffold
#     """You are provided with FMEA information, including tables or technical text. Tables may include 'Failure Mode', 'Failure Cause', and 'Failure Effect'.

# For each failure mode assign it as critical_event and extract the following:
#     - Extract associated "cause", from Cause column 
#     - "mechanism" (if present or 'Unknown Mechanism')
#     - "preventive_barriers" (if present or 'Unknown Barrier')
#     - Extract  "consequences" from Effect column

# Avoid long phrases. Return only the JSON output, without explanation or python scripts or functions. """
# #     """You are provided with FMEA information, including tables or technical text. Tables may include 'Failure Mode', 'Failure Cause', and 'Failure Effect'.

# # For each failure mode:
# # - Identify a simplified "critical_event"
# # - Extract associated "cause", "mechanism" (or 'Unknown Mechanism'), and "preventive_barriers" (or 'Unknown Barrier')
# # - Extract one or more "consequences" from the effect column

# # Output as a JSON array, with one object per failure mode.
# # Avoid long phrases. Return only the JSON, without explanation or markdown.""",

#     # Level 3 – Fully Scaffolded / Procedural
#     """You are provided with technical text and/or tables from an FMEA document related to {part_name}. The input may include:
# - Structured tables with columns like: 'Failure Mode', 'Failure Cause', 'Failure Effect'

# Instructions:

# 1. Map each 'Failure Mode' to a "critical_event".  
#    - If multiple distinct failure modes are present, generate a separate Bowtie object for each.

# 2. For each critical event entry, extract the following:
#    - "mechanism": set as "Mechanism"
#    - "cause": extract from the 'Failure Cause' or 'Cause' column, there could be multiple causes for one critical event, seprate with ';'
#    - "preventive_barriers": set as "Barrier"

# 3. Extract "consequences" from 'Failure Effect', 'Effect', or 'Local Effect' columns.
#    - If no effect is given, use ["Unknown Consequence"]
#    - If multiple effects are listed (e.g., with commas, slashes, or conjunctions), split them into separate items

# 4. Use your judgment to simplify and shorten the names used in the JSON.

# 5. Treat new lines carefully:
#    - A new line does **not** automatically indicate a new consequence, cause, or event
#    - If a line starts with a lowercase word or continues the sentence, treat it as a continuation
#    - Only split into multiple entries if distinct ideas are clearly listed (e.g., via "and", "or", commas, slashes)

# Return only the JSON output. Do not include any explanation or commentary. """
# #     """You are provided with technical text and/or tables from an FMEA document related to {part_name}. The input may include:
# # - Structured tables with columns like: 'Failure Mode', 'Failure Cause', 'Failure Effect'

# # Instructions:

# # 1. Map each 'Failure Mode' to a "critical_event".  
# #    - If multiple distinct failure modes are present, generate a separate Bowtie object for each.

# # 2. For each critical event entry, extract the following:
# #    - "mechanism": set as "Mechanism"
# #    - "cause": extract from the 'Failure Cause' or 'Cause' column, there could be multiple causes for one critical event, seprate with ';'
# #    - "preventive_barriers": set as "Barrier"

# # 3. Extract "consequences" from 'Failure Effect', 'Effect', or 'Local Effect' columns.
# #    - If no effect is given, use ["Unknown Consequence"]
# #    - If multiple effects are listed (e.g., with commas, slashes, or conjunctions), split them into separate items

# # 4. Use your judgment to simplify and shorten the names used in the JSON.

# # 5. Treat new lines carefully:
# #    - A new line does **not** automatically indicate a new consequence, cause, or event
# #    - If a line starts with a lowercase word or continues the sentence, treat it as a continuation
# #    - Only split into multiple entries if distinct ideas are clearly listed (e.g., via "and", "or", commas, slashes)

# # Return only the JSON output. Do not include any explanation or commentary."""
# ]


# # === Full context manually provided ===
# CONTEXT = """### Table: Table 19-2.  Typical Failure Modes of Individual Sensors 

# Markdown Table:
# | FAILURE MODES | FAILURE CAUSES | FAILURE EFFECT |
# | --- | --- | --- |
# | Incorrect signal from sensor; element | Reduced signal level; Impedance mismatch; A/D conversion error | Potential processing error |
# | Loss of signal from sensor; element | Chip failure; Corroded sensor | Loss of signal to processor |
# | Complete loss of signal in; transmission line | Broken wire; Fiber optic, RF interruption | Loss of signal to processor |
# | Signal error in transmission; line | Power line interference; Contaminants in fluid system | Potential processing error |
# | Incorrect signal in; computation device | Error in algorithm | Processing error |
# | Power supply loss of; voltage | Power supply malfunction | Loss of signal to processor |
# | Improper response to; information recipient | Incorrect interpretation; System malfunction; Error in algorithm | Potential system malfunction |
# | Calibration error | Software error; Error in algorithm | Potential system malfunction |
# | Battery energy depletion | Battery malfunction | Loss of signal to processor |
# """

# def generate_prompt(prompt_strictness_level, part_name):
#     index = max(0, min(int(prompt_strictness_level), len(PROMPT_VARIANTS) - 1))
#     base = PROMPT_VARIANTS[index].replace("{part_name}", part_name)
#     return f"{base}\n\n--- FMEA Context Start ---\n{CONTEXT}\n--- FMEA Context End ---\n\nReturn only valid JSON."

# def ask_model_paramset(prompt, model_path, temp, top_p, seed):
#     llm = Llama(
#         model_path=model_path,
#         n_gpu_layers=-1,
#         n_threads=16,
#         n_ctx=4096,
#         temp=temp,
#         top_p=top_p,
#         top_k=40,
#         repeat_penalty=1.0,
#         seed=seed,
#         tensor_split=[0.5, 0.5],
#     )
#     response = llm(prompt, max_tokens=2048)
#     return response["choices"][0]["text"]



# def try_parse_json_flexibly(raw_output, model_name, run_id, seed):
#     import json5
#     import re

#     # Step 1: Trim to content after </think> if it exists
#     if "</think>" in raw_output:
#         raw_output = raw_output.split("</think>")[-1].strip()

#     # Step 2: Try to extract JSON code block or raw object/array
#     match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', raw_output)
#     if not match:
#         match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw_output, re.DOTALL)

#     if not match:
#         # Save raw .txt if nothing matched
#         txt_path = os.path.join(RESULTS_DIR, f"bowtie_{model_name}_run_{run_id}_seed_{seed}.txt")
#         with open(txt_path, "w", encoding="utf-8") as f:
#             f.write(raw_output)
#         return False, f"❌ {model_name} [run {run_id}] no JSON match — saved to {txt_path}"

#     # Step 3: Clean matched content
#     json_str = match.group(1).strip()
#     if json_str.startswith("```") or json_str.endswith("```"):
#         json_str = json_str.strip("`").strip()

#     # Step 4: Try multiple cleanup attempts
#     for fix_attempt in [json_str, json_str.strip().rstrip(',') + '}', '[' + json_str + ']']:
#         try:
#             parsed = json5.loads(fix_attempt)
#             suffix = 'fixed' if fix_attempt != json_str else 'raw'
#             out_path = os.path.join(RESULTS_DIR, f"bowtie_{model_name}_run_{run_id}_seed_{seed}_{suffix}.json")
#             with open(out_path, "w", encoding="utf-8") as f:
#                 json.dump(parsed, f, indent=2)
#             return True, out_path
#         except Exception:
#             continue

#     # Step 5: Fallback .txt if JSON still fails
#     txt_path = os.path.join(RESULTS_DIR, f"bowtie_{model_name}_run_{run_id}_seed_{seed}.txt")
#     with open(txt_path, "w", encoding="utf-8") as f:
#         f.write(raw_output)
#     return False, f"❌ {model_name} [run {run_id}] parse failed — saved to {txt_path}"


# def run_sobol():
#     param_values = sobol_sample.sample(sobol_problem, N, calc_second_order=True)
#     metadata_rows = []
#     total_runs = len(param_values) * len(available_models) * NUM_SEEDS

#     for run_id, (temp, top_p, prompt_strictness) in enumerate(param_values):
#         prompt_index = max(0, min(int(round(prompt_strictness)), len(PROMPT_VARIANTS) - 1))
#         prompt = generate_prompt(prompt_index, PART_NAME)

#         for seed_ix in range(NUM_SEEDS):
#             seed = SHARED_SEEDS[run_id * NUM_SEEDS + seed_ix]
#             for model_name, model_path in available_models.items():
#                 print(f"[{model_name}] Run {run_id}, Prompt {prompt_index}, Temp {temp:.2f}, Top-p {top_p:.2f}, Seed {seed}")
#                 try:
#                     raw_output = ask_model_paramset(prompt, model_path, temp, top_p, seed)
#                     success, path = try_parse_json_flexibly(raw_output, model_name, run_id, seed)
#                     metadata_rows.append({
#                         "run_id": run_id,
#                         "model": model_name,
#                         "temperature": temp,
#                         "top_p": top_p,
#                         "prompt_strictness": prompt_index,
#                         "seed": seed,
#                         "json_path": path,
#                         "status": "success" if success else "fail"
#                     })
#                 except Exception as e:
#                     metadata_rows.append({
#                         "run_id": run_id,
#                         "model": model_name,
#                         "temperature": temp,
#                         "top_p": top_p,
#                         "prompt_strictness": prompt_index,
#                         "seed": seed,
#                         "json_path": "",
#                         "status": f"error: {e}"
#                     })

#     df = pd.DataFrame(metadata_rows)
#     out_path = os.path.join(OUTPUT_DIR, f"sobol_metadata_{PART_NAME.replace(' ', '_')}_{PROMPT_TYPE}.csv")
#     df.to_csv(out_path, index=False)
#     print(f"✅ Saved Sobol metadata to {out_path}")

# if __name__ == '__main__':
#     run_sobol()

