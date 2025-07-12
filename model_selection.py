
import os
import json
import re
import pynvml  
from llama_cpp import Llama  

def list_gpus():
    """
    Print information about all available NVIDIA GPUs on the system.

    Uses NVIDIA Management Library (pynvml) to query:
      - GPU count
      - GPU name
      - Total, used, and free memory for each GPU

    """
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print(f"Number of GPUs available: {count}")
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {i}: {name.decode('utf-8') if isinstance(name, bytes) else name}")
            print(f"  Total memory: {meminfo.total / (1024**3):.2f} GB")
            print(f"  Used memory:  {meminfo.used / (1024**3):.2f} GB")
            print(f"  Free memory:  {meminfo.free / (1024**3):.2f} GB")
        pynvml.nvmlShutdown()
    except Exception as e:
        print("Error initializing NVML:", e)

def extract_json(text):
    """
    Extract the first JSON object from a raw LLM response string.

    Tries multiple strategies in order:
      1. JSON fenced in triple backticks with ```json ... ```.
      2. Splitting by '```json' markers.
      3. Fallback: first `{...}` match anywhere in text.

    Args:
        text (str): Raw text returned by the LLM.

    Returns:
        str: The extracted JSON string or None if no JSON found.
    """
    # Strategy 1: Look for JSON enclosed in triple backticks with "json"
    pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Strategy 2: Split by '```json' and then by '```'
    parts = text.split("```json")
    if len(parts) > 1:
        json_part = parts[1].split("```")[0].strip()
        if json_part:
            return json_part

    # Strategy 3: Fallback - search for the first JSON object in the text
    match = re.search(r'({.*})', text, re.DOTALL)
    if match:
        return match.group(1)
    
    return None

def build_prompt(text_input):
    """
    Build a system-dynamics expert prompt for CLD extraction.

    The prompt instructs the LLM to:
      1. Identify variables.
      2. Extract causal relationships (positive/negative, reinforcing/balancing).
      3. Generate a DOT/Graphviz causal loop diagram.
      4. Return a JSON with keys: "variables", "relationships", "dot_code".

    Args:
        text_input (str): The narrative or descriptive text to analyze.

    Returns:
        str: The full prompt string to send to the LLM.
    """
    prompt = f'''You are a domain expert in system dynamics and causal analysis. Your task is to analyze the following piece of text, extract key variables and causal relationships, and then generate a corresponding causal loop diagram (CLD) in DOT/Graphviz format.

Text:
-------------------------------------------------
"{text_input}"
-------------------------------------------------

Please perform the following tasks:
1. **Variable Identification:**  
   - List all the main variables mentioned in the text.

2. **Relationship Extraction:**  
   - Identify and describe the causal relationships among these variables.
   - For each relationship, indicate:
     - Whether it is positive (an increase in the cause leads to an increase in the effect) or negative (an increase in the cause leads to a decrease in the effect).
     - If applicable, specify if the relationship is part of a reinforcing loop (amplifies changes) or a balancing loop (stabilizes the system).
   - Clearly state the cause and effect for each relationship.

3. **Causal Loop Diagram Generation:**  
   - Create a causal loop diagram (CLD) using DOT/Graphviz syntax to visually represent the variables and their relationships.

4. **Structured Output:**  
   - Return your final answer as valid JSON with exactly the following keys:
     - "variables": an array of the identified variables.
     - "relationships": an array of descriptions of the causal relationships.
     - "dot_code": a string containing the complete DOT code for the CLD.

Please ensure that your response contains ONLY the JSON output without any additional commentary or markdown formatting.'''
    return prompt

def main():
    """
    Entry point for model selection & hallucination testing script.

    Workflow:
      1. Print current working directory.
      2. List available GPUs.
      3. Show model directory contents.
      4. Initialize a Llama model.
      5. Loop through predefined text samples:
         - Build prompt
         - Invoke LLM
         - Extract JSON from raw output
         - Validate JSON
      6. Save all parsed results to `all_results_seed23.json`.

    Returns:
        None
    """
    print(f"Current working directory: {os.getcwd()}")
    print("Checking available GPUs:")
    list_gpus()

    model_dir = "DeepSeek-R1-GGUF"
    if os.path.isdir(model_dir):
        print(f"Files in the '{model_dir}' directory:")
        for file in os.listdir(model_dir):
            print(f"- {file}")
    else:
        print(f"Directory '{model_dir}' not found.")

    print("Loading model... (this may take a while)")
    llm = Llama(
        model_path="DeepSeek-R1-GGUF/Mistral-7B-Instruct-v0.3-Q5_K_S.gguf",
        n_gpu_layers=-1,  # Adjust based on your GPU and model requirements.
        n_threads=16,
        n_ctx=5000,       # Context size
        temp=0.9,         # Temperature
        seed=23,
    )
    print("Model loaded.")

    # List of text inputs/prompts
    texts = [
        "The order rate decision, if it is to bring actual inventory towards desired inventory, must increase the order rate as inventory falls below desired inventory. Conversely, as inventory rises toward the desired inventory, order rate should be reduced.",
        "A hot cup of coffee will gradually cool down to room temperature. Its rate of cooling depends on the difference between the temperature of the coffee and the temperature of the room. The greater the difference, the faster the coffee will cool.",
        "The more my uncle smokes, the more addicted he becomes to the nicotine in his cigarettes. After smoking a few cigarettes a long time ago, my uncle began to develop a need for cigarettes. The need caused him to smoke even more, which produced an even stronger need to smoke. The reinforcing behavior in the addiction process is characteristic of positive feedback.",
        "A larger population leads to a higher number of births, and higher births lead to a higher population. The larger population will tend to have a greater number of deaths.",
    ]

    all_results = []
    for idx, text_input in enumerate(texts, start=1):
        print(f"\nProcessing prompt {idx}:")
        prompt = build_prompt(text_input)
        output = llm(prompt, max_tokens=2048)
        
        # If output is a dict, extract the text from the choices list.
        if isinstance(output, dict) and "choices" in output:
            raw_text = output["choices"][0]["text"]
        else:
            raw_text = output

        raw_text = raw_text.strip("` \n")
        print(f"Raw output for prompt {idx}:\n{raw_text}\n")

        json_text = extract_json(raw_text)
        if json_text is None:
            print("Failed to extract JSON from the output.")
            result = {"error": "No JSON block found", "raw_output": raw_text}
        else:
            try:
                result = json.loads(json_text)
            except json.JSONDecodeError as e:
                print("Extracted JSON is invalid:", e)
                result = {"error": "Invalid JSON", "raw_output": json_text}
        
        # Store the result along with the original text and prompt index.
        all_results.append({
            "prompt_index": idx,
            "original_text": text_input,
            "result": result
        })

    # Save all results to a JSON file.
    output_file = "./all_results_seed23.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {output_file}")

if __name__ == "__main__":
    main()
