# NODE_ROLES, MODELS, PROMPTS, etc.

NODE_ROLES = ["Cause", "Mechanism", "Central Event", "Barrier", "Consequence"]

METHODS = ["LLM + RAG", "LLM + OCR", "Dual LLM"]
PROMPTS = ["zero", "zero + medium", "zero + high", "cot + medium", "cot + high","few", "cot", "hybrid"]
DOMAINS = ["shaft failure", "valve assembly", "individual sensor", "dynamic seals"]
MODELS = [
    "Qwen-7B", "Mistral-Instruct", "LLaMA-Instruct",
    "R1-Distill-LLama-8B", "R1-Distill-Qwen-7B", "R1-Distill-Qwen-32B", "Mistral-Small"
]

CSV_LOG = "ged_results.xlsx"
