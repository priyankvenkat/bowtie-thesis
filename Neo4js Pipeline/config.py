import os

# === Configuration ===

# API Keys and model names (make sure to set your environment variable MISTRAL_API_KEY)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "FpL49bp6asB0frkxMbmazONcm3eoubOT")
PIXTRAL_MODEL = "mistral-small-2503"
MISTRAL_MODEL = "mistral-small-2503"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "sensor_bowtie")


BOWTIE_ZERO_SHOT_PROMPT = (
    "You are provided with a technical table description and its structured JSON/markdown representations extracted from an FMEA document related to {part}. "
    "The table may have the following columns:\n"
    " - 'Failure Mode', 'Failure Mechanism', 'Failure Cause'\n"
    " - 'Failure Mode', 'Cause', 'Effect'\n\n"
    "Use these rules to map the table into a Bowtie diagram JSON:\n"
    "1. Map 'Failure Mode' to the top event, labeled as 'critical_event'. If multiple failure modes are present, generate an array of Bowtie JSON objects.\n"
    "2. If 'Failure Mechanism' and 'Failure Cause' columns are present, create a threat object for each row with two separate keys: 'mechanism' (from 'Failure Mechanism') and 'cause' (from 'Failure Cause'). If only a 'Cause' column is present, use that value as the 'cause' and leave 'mechanism' as null or a default string.\n"
    "3. If 'Effect' is present, those entries become 'consequences'. Otherwise, leave 'consequences' empty.\n"
    "4. If no preventive or mitigative barriers are provided, set the 'preventive_barriers' and 'mitigative_barriers' fields as empty lists (or fill them with an educated guess if possible).\n\n"
    "Return the output as valid JSON. Each diagram must have exactly these keys: "
    "'critical_event', 'threats', 'consequences', 'preventive_barriers', and 'mitigative_barriers'.\n"
    "Within 'threats', each object must include the keys: 'mechanism', 'cause', and 'preventive_barriers'. "
    "Do not include any extra text."
)

BOWTIE_FEW_SHOT_PROMPT = (
    "Below is an example of a Bowtie diagram JSON output generated from a failure analysis table:\n\n"
    "Example:\n"
    "{{\n"
    '  "critical_event": "Excessive leakage",\n'
    '  "threats": [\n'
    '    {{"mechanism": "Wear", "cause": "Misalignment", "preventive_barriers": []}},\n'
    '    {{"mechanism": "Wear", "cause": "Shaft out-of-roundness", "preventive_barriers": []}}\n'
    "  ],\n"
    '  "consequences": ["System shutdown", "Environmental contamination"],\n'
    '  "preventive_barriers": [],\n'
    '  "mitigative_barriers": []\n'
    "}}\n\n"
    "Your table may have columns for:\n"
    " - 'Failure Mode', 'Failure Mechanism', 'Failure Cause', or\n"
    " - 'Failure Mode', 'Cause', 'Effect'.\n\n"
    "Map 'Failure Mode' to 'critical_event'. If you have 'Failure Mechanism' and 'Failure Cause', create a threat object for each row with separate keys 'mechanism' and 'cause'. If only 'Cause' is provided, treat that value as the 'cause'. If 'Effect' is present, map it to 'consequences'. "
    "If the table has multiple failure modes, output an array of Bowtie diagram objects. "
    "Return valid JSON with the keys: 'critical_event', 'threats', 'consequences', 'preventive_barriers', and 'mitigative_barriers'. "
    "If no barriers are provided, return empty lists for both 'preventive_barriers' and 'mitigative_barriers'. "
    "Do not include any extra commentary."
)

BOWTIE_COT_PROMPT = (
    "Analyze the following failure analysis table description step-by-step for {part}:\n"
    "1. Examine the columns provided:\n"
    "   - If the columns are 'Failure Mode', 'Failure Mechanism', 'Failure Cause', then map 'Failure Mode' to the top event ('critical_event') and, for each row, create a threat object with two separate keys: 'mechanism' (from 'Failure Mechanism') and 'cause' (from 'Failure Cause').\n"
    "   - If the columns are 'Failure Mode', 'Cause', 'Effect', map 'Failure Mode' to 'critical_event', treat each 'Cause' as the 'cause' in a threat (with 'mechanism' left as null or a default value), and map 'Effect' to 'consequences'.\n"
    "2. If multiple failure modes are present, generate an array of Bowtie diagram objects.\n"
    "3. If no explicit details on preventive or mitigative barriers are provided, set these fields as empty lists.\n\n"
    "Return your answer as valid JSON in the following format:\n"
    "{{\n"
    '  "critical_event": "...",\n'
    '  "threats": [\n'
    '    {{"mechanism": "...", "cause": "...", "preventive_barriers": [...] }},\n'
    '    ...\n'
    "  ],\n"
    '  "consequences": [...],\n'
    '  "preventive_barriers": [...],\n'
    '  "mitigative_barriers": [...]\n'
    "}}\n\n"
    "If there are multiple failure modes, output an array of such objects. Do not add any extra text."
)
