from pathlib import Path
import os
import json
import random
import pandas as pd
import re
import math
from llama_cpp import Llama
import json5

# === Config ===
PART_NAME = "Dynamic Seals"
N_SEEDS = 1
SAVE_DIR = Path("results_stochasticity_varied_context")
SAVE_DIR.mkdir(exist_ok=True)
SEED = 2242
random.seed(SEED)
SHARED_SEEDS = [random.randint(1, 999999) for _ in range(50000)]

# === Models and Top 2 Prompt Configurations ===
top_configs = {
    "LLaMA-3-8B": [("zero", "high"),("cot", "high")],
    "Qwen-7B": [("zero", "medium"),("cot", "medium")],
    "Mistral-7B": [("zero", "high"),("cot", "medium")] 
}


model_paths = {
    "LLaMA-3-8B": "DeepSeek-R1-GGUF/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
    "Qwen-7B":  "DeepSeek-R1-GGUF/Qwen2.5-7B-Instruct-1M-Q6_K.gguf",
    "Mistral-7B": "DeepSeek-R1-GGUF/Mistral-7B-Instruct-v0.3-Q5_K_S.gguf"
}

# === Static RAG Context ===
# RAG_CONTEXT = """
# ======================================================================== #

# The dynamic seal may be used to seal many different liquids at various speeds, pressures, and temperatures. Dynamic seals are made of natural and synthetic rubbers, polymers and elastomers, metallic compounds, and specialty materials. Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive particles present in the fluid during operation will have a strong influence on the wear resistance of seals. Seals typically operate with sliding contact. Elastomer wear is analogous to metal degradation. However, elastomers are more sensitive to thermal deterioration than to mechanical wear. Hard particles can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage.
# The most common modes of seal failure are by fatigue-like surface embrittlement, abrasive removal of material, and corrosion. Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive particles present in the fluid during operation will have a strong influence on the wear resistance of seals, the wear rate of the seal increasing with the quantity of 
# environmental contamination. A good understanding of the wear mechanism involved will help determine potential seal deterioration. For example, contaminants from the environment such as sand can enter the fluid system and become embedded in the elastomeric seals causing abrasive cutting and damage to shafts.
# Compression set refers to the permanent deflection remaining in the seal after complete release of a squeezing load while exposed to a particular temperature level. Compression set reflects the partial loss of elastic memory due to the time effect. Operating over extreme temperatures can result in compression-type seals such as O- rings to leak fluid at low pressures because they have deformed permanently or taken a set after used for a period of time.
# Another potential failure mode to be considered is fatigue failure caused by shaft run-out. A bent shaft can cause vibration throughout the equipment and eventual loss of seal resiliency. Typical failure mechanism and causes for dynamic seals are included in Table 3-2.
# An important factor in the design of dynamic seals is the pressure velocity (PV) coefficient. The PV coefficient is defined as the product of the seal face or system pressure and the fluid velocity. This factor is useful in estimating seal reliability when compared with manufacturer's limits. If the PV limit is exceeded, a seal may wear at a rate greater than desired.

# ======================================================================== #
# Main Failure Mechanisms of Dynamic Seals

# The dynamic seal may be used to seal many different liquids at various speeds, pressures, and temperatures. The sealing surfaces are perpendicular to the shaft with contact between the primary and mating rings to achieve a dynamic seal. Dynamic seals are made of natural and synthetic rubbers, polymers and elastomers, metallic compounds, and specialty materials.
# The most common modes of seal failure are by fatigue-like surface embrittlement, abrasive removal of material, and corrosion. Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive particles present in the fluid during operation will have a strong influence on the wear resistance of seals, the wear rate of the seal increasing with the quantity of environmental contamination. A good understanding of the wear mechanism involved will help determine potential seal deterioration. For example, contaminants from the environment such as sand can enter the fluid system and become embedded in the elastomeric seals causing abrasive cutting and damage to shafts.
# Dynamic seals typically operate with sliding contact. Elastomer wear is analogous to metal degradation. However, elastomers are more sensitive to thermal deterioration than to mechanical wear. Hard particles can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage. Abrasive particles can contribute to seal wear by direct abrasion and by plugging screens and orifices creating a loss of lubricant to the seal.
# Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive contaminant particles present in the fluid during operation will have a strong influence on the wear resistance of seals. Hard particles, for example, can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage.


# Wear often occurs between the primary ring and mating ring. This surface contact is maintained by a spring. There is a film of liquid maintained between the sealing surfaces to eliminate as much friction as possible. For most dynamic seals, the three common points of sealing contact occur between the following points:
# (1) Mating surfaces between primary and mating rings
# (2) Between the rotating component and shaft or sleeve
# (3) Between the stationary component and the gland plate
# The various failure mechanisms and causes for mechanical seals are listed in Table 3-2. Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive particles present in the fluid during operation will have a strong influence on the wear resistance of seals. Seals typically operate with sliding contact. Elastomer wear is analogous to metal degradation. However, elastomers are more sensitive to thermal deterioration than to mechanical wear. Hard particles can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage.
# Compression set refers to the permanent deflection remaining in the seal after complete release of a squeezing load while exposed to a particular temperature level. Compression set reflects the partial loss of elastic memory due to the time effect. Operating over extreme temperatures can result in compression-type seals such as O- rings to leak fluid at low pressures because they have deformed permanently or taken a set after used for a period of time.
# An additional important seal design consideration is seal balance. Seal balance refers to the difference between the pressure of the fluid being sealed and the contact pressure between the seal faces. It is the ratio of hydraulic closing area to seal face area (parameter k in Equation (3-13). A balanced seal is designed so that the effective contact pressure is always less than the fluid pressure, reducing friction at the seal faces. The result is less rubbing wear, less heat generated and higher fluid pressure capability. In an unbalanced seal, fluid pressure is not relieved by the face geometry, the seal faces withstand full system fluid pressure in addition to spring pressure and the face contact pressure is greater than or equal to fluid pressure.

# Seal balance then is a performance characteristic that measures how effective the seal mating surfaces match. If not effectively matched, the seal load at the dynamic facing may be too high causing the liquid film to be squeezed out and vaporized, thus causing a high wear rate. The fluid pressure from one side of the primary ring causes a certain amount of force to impinge on the dynamic seal face. The dynamic facing pressure can be controlled by manipulating the hydraulic closing area with a shoulder on a sleeve or by seal hardware. By increasing the area, the sealing force is increased.
# An important factor in the design of dynamic seals is the pressure velocity (PV) coefficient. The PV coefficient is defined as the product of the seal face or system pressure and the fluid velocity. This factor is useful in estimating seal reliability when compared with manufacturer's limits. If the PV limit is exceeded, a seal may wear at a rate greater than desired.

# ======================================================================== #

# The dynamic seal may be used to seal many different liquids at various speeds, pressures, and temperatures. The sealing surfaces are perpendicular to the shaft with contact between the primary and mating rings to achieve a dynamic seal. Dynamic seals are made of natural and synthetic rubbers, polymers and elastomers, metallic compounds, and specialty materials.
# The most common modes of seal failure are by fatigue-like surface embrittlement, abrasive removal of material, and corrosion. Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive particles present in the fluid during operation will have a strong influence on the wear resistance of seals, the wear rate of the seal increasing with the quantity of environmental contamination. 
# A good understanding of the wear mechanism involved will help determine potential seal deterioration. For example, contaminants from the environment such as sand can enter the fluid system and become embedded in the elastomeric seals causing abrasive cutting and damage to shafts.
# Dynamic seals typically operate with sliding contact. Elastomer wear is analogous to metal degradation. However, elastomers are more sensitive to thermal deterioration than to mechanical wear. Hard particles can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage. Abrasive particles can contribute to seal wear by direct abrasion and by plugging screens and orifices creating a loss of lubricant to the seal.
# Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive contaminant particles present in the fluid during operation will have a strong influence on the wear resistance of seals. Hard particles, for example, can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage.

# The various failure mechanisms and causes for mechanical seals are listed in Table 3-2. Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive particles present in the fluid during operation will have a strong influence on the wear resistance of seals. Seals typically operate with sliding contact. Elastomer wear is analogous to metal degradation. However, elastomers are more sensitive to thermal deterioration than to mechanical wear. Hard particles can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage.
# Compression set refers to the permanent deflection remaining in the seal after complete release of a squeezing load while exposed to a particular temperature level. Compression set reflects the partial loss of elastic memory due to the time effect. Operating over extreme temperatures can result in compression-type seals such as O- rings to leak fluid at low pressures because they have deformed permanently or taken a set after used for a period of time.
# An additional important seal design consideration is seal balance. Seal balance refers to the difference between the pressure of the fluid being sealed and the contact pressure between the seal faces. It is the ratio of hydraulic closing area to seal face area (parameter k in Equation (3-13). A balanced seal is designed so that the effective contact pressure is always less than the fluid pressure, reducing friction at the seal faces. The result is less rubbing wear, less heat generated and higher fluid pressure capability. In an unbalanced seal, fluid pressure is not relieved by the face geometry, the seal faces withstand full system fluid pressure in addition to spring pressure and the face contact pressure is greater than or equal to fluid pressure.

# Seal balance then is a performance characteristic that measures how effective the seal mating surfaces match. If not effectively matched, the seal load at the dynamic facing may be too high causing the liquid film to be squeezed out and vaporized, thus causing a high wear rate. The fluid pressure from one side of the primary ring causes a certain amount of force to impinge on the dynamic seal face. The dynamic facing pressure can be controlled by manipulating the hydraulic closing area with a shoulder on a sleeve or by seal hardware. By increasing the area, the sealing force is increased.
# An important factor in the design of dynamic seals is the pressure velocity (PV) coefficient. The PV coefficient is defined as the product of the seal face or system pressure and the fluid velocity. This factor is useful in estimating seal reliability when compared with manufacturer's limits. If the PV limit is exceeded, a seal may wear at a rate greater than desired.



# === Context loading ===


# RAG_CONTEXT = """

# """

RAG_CONTEXT = """
The dynamic seal may be used to seal many different liquids at various speeds, pressures, and temperatures. The sealing surfaces are perpendicular to the shaft with contact between the primary and mating rings to achieve a dynamic seal. Dynamic seals are made of natural and synthetic rubbers, polymers and elastomers, metallic compounds, and specialty materials.
The most common modes of seal failure are by fatigue-like surface embrittlement, abrasive removal of material, and corrosion. Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive particles present in the fluid during operation will have a strong influence on the wear resistance of seals, the wear rate of the seal increasing with the quantity of environmental contamination. 
A good understanding of the wear mechanism involved will help determine potential seal deterioration. For example, contaminants from the environment such as sand can enter the fluid system and become embedded in the elastomeric seals causing abrasive cutting and damage to shafts.
Dynamic seals typically operate with sliding contact. Elastomer wear is analogous to metal degradation. However, elastomers are more sensitive to thermal deterioration than to mechanical wear. Hard particles can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage. Abrasive particles can contribute to seal wear by direct abrasion and by plugging screens and orifices creating a loss of lubricant to the seal.
Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive contaminant particles present in the fluid during operation will have a strong influence on the wear resistance of seals. Hard particles, for example, can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage.

The various failure mechanisms and causes for mechanical seals are listed in Table 3-2. Wear and sealing efficiency of fluid system seals are related to the characteristics of the surrounding operating fluid. Abrasive particles present in the fluid during operation will have a strong influence on the wear resistance of seals. Seals typically operate with sliding contact. Elastomer wear is analogous to metal degradation. However, elastomers are more sensitive to thermal deterioration than to mechanical wear. Hard particles can become embedded in soft elastomeric and metal surfaces leading to abrasion of the harder mating surfaces forming the seal, resulting in leakage.
Compression set refers to the permanent deflection remaining in the seal after complete release of a squeezing load while exposed to a particular temperature level. Compression set reflects the partial loss of elastic memory due to the time effect. Operating over extreme temperatures can result in compression-type seals such as O- rings to leak fluid at low pressures because they have deformed permanently or taken a set after used for a period of time.
An additional important seal design consideration is seal balance. Seal balance refers to the difference between the pressure of the fluid being sealed and the contact pressure between the seal faces. It is the ratio of hydraulic closing area to seal face area (parameter k in Equation (3-13). A balanced seal is designed so that the effective contact pressure is always less than the fluid pressure, reducing friction at the seal faces. The result is less rubbing wear, less heat generated and higher fluid pressure capability. In an unbalanced seal, fluid pressure is not relieved by the face geometry, the seal faces withstand full system fluid pressure in addition to spring pressure and the face contact pressure is greater than or equal to fluid pressure.

Seal balance then is a performance characteristic that measures how effective the seal mating surfaces match. If not effectively matched, the seal load at the dynamic facing may be too high causing the liquid film to be squeezed out and vaporized, thus causing a high wear rate. The fluid pressure from one side of the primary ring causes a certain amount of force to impinge on the dynamic seal face. The dynamic facing pressure can be controlled by manipulating the hydraulic closing area with a shoulder on a sleeve or by seal hardware. By increasing the area, the sealing force is increased.
An important factor in the design of dynamic seals is the pressure velocity (PV) coefficient. The PV coefficient is defined as the product of the seal face or system pressure and the fluid velocity. This factor is useful in estimating seal reliability when compared with manufacturer's limits. If the PV limit is exceeded, a seal may wear at a rate greater than desired.

"""

# VISION_CONTEXT = """

# """

# === Prompt loading ===
SCRIPT_DIR = Path(__file__).parent.resolve()
PROMPT_BASE = SCRIPT_DIR / "prompt_templates_narrative"

def load_prompt(prompt_type, strictness, context_text):
    """
    Load a base prompt template and inject the part name and context.

    Args:
        prompt_type (str): One of 'zero', 'few', or 'cot' indicating prompt style.
        strictness  (str): One of 'low', 'medium', or 'high' controlling prompt detail.
        context_text(str): The FMEA context (table or narrative) to append.

    Returns:
        str: The full LLM prompt including context markers and instructions.
    """
    prompt_path = PROMPT_BASE / prompt_type / f"{strictness}.txt"
    with open(prompt_path, "r") as f:
        base_prompt = f.read().replace("{part_name}", PART_NAME)

    return (
        f"{base_prompt.strip()}\n\n--- FMEA Context Start ---\n{context_text.strip()}\n--- FMEA Context End ---\n\n"
        "Return only valid JSON, no python script, no function — just the JSON output."
    )

def ask_model(prompt, model_path, seed):
    """
    Invoke the Llama model with a single prompt and return its raw text response.

    Args:
        prompt    (str): Full prompt string for the LLM.
        model_path(str): Filesystem path to the GGUF model.
        seed       (int): Random seed for deterministic sampling.

    Returns:
        str: The raw textual output from the LLM.
    """
    llm = Llama(
        model_path=model_path,
        seed=seed,
        n_ctx=4096,
        temp=0.5,
        top_p=0.95,
    )
    
    return llm(prompt, max_tokens=2000)["choices"][0]["text"]

def try_parse_json_flexibly(raw_output, model_name, prompt_type, strictness, seed_idx, seed):
    """
    Attempt to extract valid JSON from LLM output, saving either the JSON or raw text.

    Steps:
      1. Remove any '</think>' markers and take the trailing text.
      2. Regex to find fenced JSON code blocks or the first {...}/[...] pattern.
      3. If none found, save raw output to .txt.
      4. Otherwise, clean up backticks and try parsing via json5.
      5. Attempt minor fixes: remove trailing commas, wrap in array.
      6. On success, save JSON to a file with suffix 'raw' or 'fixed'.
      7. On failure, save raw text.

    Args:
        raw_output (str): Text returned by the LLM.
        model_name (str): Identifier for the model (e.g., "LLaMA-3-8B").
        prompt_type(str): Prompt style used ('zero', 'cot', etc.).
        strictness (str): Prompt strictness level ('low', 'high', etc.).
        seed_idx   (int): Index of this seed run.
        seed       (int): Actual random seed value.

    Returns:
        (bool, str):
          - success (bool): True if JSON parsed and saved; False otherwise.
          - path    (str): Filepath of the saved JSON or raw text.
    """
    if "</think>" in raw_output:
        raw_output = raw_output.split("</think>")[-1].strip()

    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', raw_output)
    if not match:
        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw_output, re.DOTALL)

    if not match:
        txt_path = SAVE_DIR / f"{model_name}_{prompt_type}_{strictness}_seed{seed_idx:02d}.txt"
        txt_path.write_text(raw_output, encoding="utf-8")
        return False, str(txt_path)

    json_str = match.group(1).strip()
    if json_str.startswith("```") or json_str.endswith("```"):
        json_str = json_str.strip("`").strip()

    fix_attempts = [
        json_str,
        json_str.strip().rstrip(',') + '}',
        '[' + json_str + ']',
    ]

    for fix in fix_attempts:
        try:
            parsed = json5.loads(fix)
            suffix = 'fixed' if fix != json_str else 'raw'
            json5_path = SAVE_DIR / f"{model_name}_{prompt_type}_{strictness}_seed{seed_idx:02d}_{suffix}.json"
            json5_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
            return True, str(json5_path)
        except Exception:
            continue

    txt_path = SAVE_DIR / f"{model_name}_{prompt_type}_{strictness}_seed{seed_idx:02d}.txt"
    txt_path.write_text(raw_output, encoding="utf-8")
    return False, str(txt_path)

def run_stochasticity_experiment():
    """
    Execute the stochasticity experiment over top-2 prompt configs for each model.

    For each model and each of its top 2 (prompt_type, strictness) combos:
      - Build the prompt using RAG_CONTEXT.
      - For each seed in SHARED_SEEDS:
          - Invoke the model.
          - Attempt to parse and save JSON output.
          - Record run metadata (status, output path).

    Finally, save all run metadata to a CSV in SAVE_DIR.
    """
    metadata = []
    for model_name, combos in top_configs.items():
        model_path = model_paths[model_name]
        for combo_id, (ptype, strictness) in enumerate(combos, start=1):
            prompt = load_prompt(ptype, strictness, RAG_CONTEXT)

            for seed_idx in range(N_SEEDS):
                seed = SHARED_SEEDS[seed_idx]
                print(f"[{model_name}] {ptype}-{strictness} | Seed #{seed_idx} ({seed})")

                try:
                    output = ask_model(prompt, model_path, seed)
                    success, path = try_parse_json_flexibly(output, model_name, ptype, strictness, seed_idx, seed)
                    metadata.append({
                        "model": model_name,
                        "combo_id": combo_id,
                        "prompt_type": ptype,
                        "strictness": strictness,
                        "seed_index": seed_idx,
                        "seed": seed,
                        "status": "success" if success else "fail",
                        "output_path": path
                    })
                except Exception as e:
                    metadata.append({
                        "model": model_name,
                        "combo_id": combo_id,
                        "prompt_type": ptype,
                        "strictness": strictness,
                        "seed_index": seed_idx,
                        "seed": seed,
                        "status": f"error: {e}",
                        "output_path": ""
                    })

    df = pd.DataFrame(metadata)
    df.to_csv(SAVE_DIR / f"stochasticity_metadata_{PART_NAME.replace(' ', '_')}.csv", index=False)
    print("✅ All runs complete. Metadata saved.")

if __name__ == "__main__":
    run_stochasticity_experiment()
