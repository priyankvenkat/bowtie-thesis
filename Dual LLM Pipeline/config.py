import os

# === Configuration ===

# API Keys and model names 
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "FpL49bp6asB0frkxMbmazONcm3eoubOT")
PIXTRAL_MODEL = "mistral-small-latest"
MISTRAL_MODEL = "mistral-small-latest"


BOWTIE_ZERO_SHOT_PROMPT = (
"""
You are provided with technical text and/or tables from an FMEA document related to {part}. The input may include:
- Structured tables with columns like: 'Failure Mode', 'Failure Cause', 'Failure Effect'

Instructions:

1. Map each 'Failure Mode' to a "critical_event".  
   - If multiple distinct failure modes are present, generate a separate Bowtie object for each.

2. For each critical event entry, extract the following in the threats key:
   - "mechanism": set as "Mechanism"
   - "cause": extract from the 'Failure Cause' or 'Cause' column, there could be multiple causes for one critical event, seprate with ';'
   - "preventive_barriers": set as "Barrier"

3. Extract "consequences" from 'Failure Effect', 'Effect', or 'Local Effect' columns.
   - If no effect is given, use ["Unknown Consequence"]
   - If multiple effects are listed (e.g., with commas, slashes, or conjunctions), split them into separate items

4. Use your judgment to simplify and shorten the names used in the JSON.

5. Treat new lines carefully:
   - A new line does **not** automatically indicate a new consequence, cause, or event
   - If a line starts with a lowercase word or continues the sentence, treat it as a continuation
   - Only split into multiple entries if distinct ideas are clearly listed (e.g., via "and", "or", commas, slashes)

Return only the JSON output. Do not include any explanation or commentary.
"""
)

BOWTIE_FEW_SHOT_PROMPT = (
"""
You will be shown FMEA table rows and asked to output Bowtie JSON for the part {part}. Follow these examples carefully.
All outputs must include the keys: 'critical_event', 'cause', 'mechanism', 'preventive_barriers', 'consequences'.

Use below examples as reference only: 
{
  "critical_event": "Critical Event 1" from failure mode column, 
  "cause": "Power cut", "Cause 1", "Cause 2" from Cause column,
  "mechanism": "Mechanism",
  "preventive_barriers": "Barrier",
  "consequences": "Consequence 1", "Consequence 2" from Effect column
}
"""
)

BOWTIE_COT_PROMPT = (
"""
We are converting FMEA data to Bowtie JSON using step-by-step reasoning and strict formatting for part {part}.

Steps:
1. Identify 'Failure Mode' column as -> critical_event
2. Extract from 'Failure Cause' column -> causes
3. Set 'mechanism' as 'Mechanism' 
4. Set 'preventive_barriers' as 'Barrier'
4b. Place the Cause, mechanism and preventive_barriers in the 'threats' key
5. Set 'consequences' from the 'Effects' column  (split by ',', 'and', 'or')
6. Return one JSON per failure mode, each with:
'critical_event', 'cause', 'mechanism', 'preventive_barriers', 'consequences'
7. Bear in mind there can be more than one cause and consequence for a failure mode. 
8. Output a JSON object per failure mode with ALL required keys:
'critical_event', 'cause', 'mechanism', 'preventive_barriers', 'consequences'
"""
)


# BOWTIE_ZERO_SHOT_PROMPT = (
# """


# You are provided with technical text and/or table descriptions extracted from an FMEA document related to {part}. The input may include:
# - Structured tables with columns such as: 'Failure Mode', 'Failure Mechanism', 'Failure Cause'

# Your task is to convert this input into a valid Bowtie diagram JSON.

# Instructions:

# 1. Map each 'Failure Mode' to a 'critical_event'. If multiple distinct failure modes are present, generate a separate Bowtie object for each.
# 2. For each row or described failure, construct 'threats' with:
#    - 'mechanism' (from the "Failure Mechanism" column)
#    - 'cause' (from 'Failure Cause' or 'Cause' column)
#    - 'preventive_barriers' (Force this as "Barrier")
# 3. Force 'Effect' to be "Consequence".
# 4. Treat new lines carefully:
#    - If you think there are distinct 'causes' for the same critical event, put them on new lines in the JSON
#    - A new line does **not** always indicate a new consequence or cause.
#    - If a line begins with a lowercase word or seems grammatically incomplete **assume it continues the previous line**.
#    - Only split into separate items if the entry clearly lists multiple distinct concepts using commas, "and", "or", "and/or", or similar conjunctions.

# Return only the JSON output. Do not include any explanation or commentary.
# """
# )

# BOWTIE_FEW_SHOT_PROMPT = (
# """
# You are provided with technical table data extracted from an FMEA document related to {part}.  
# Your task is to convert the data into valid Bowtie diagram JSON objects.

# Follow these rules:

# - Treat each 'Failure Mode' as a "critical_event".
# - Each row should become a "threat" with:
#    - "mechanism": take from 'Failure Mechanism' if available, otherwise set to 'Mechanism'
#    - "cause": based on the 'Failure Cause' or 'Cause' column
#    - "preventive_barriers": always set to "Barrier"

# - If an 'Effect', 'Failure Effect' or 'Local Effect' column exists, use it as "consequences", else set as "Consequence".
#   - Consequences must never be the same as the critical event.  
#   - If multiple distinct effects are listed (e.g. using commas, "and", "or", "and/or"), split them into separate consequence items.

# - Handle line breaks with care:
#   - A new line does not always mean a new item.
#   - If a line starts with a lowercase word or continues the grammar, assume its a continuation.
#   - Only split if multiple distinct ideas are clearly listed.

# - If multiple failure modes are present, output an array of Bowtie diagram objects.
# - Always include empty lists for "preventive_barriers" and "mitigative_barriers" at the top level.

# Do not include any extra commentary or markdown.

# ---

# Example output for a single failure mode:

# {{
#   "critical_event": "Critical Event 1",
#   "threats": [
#     {{
#       "mechanism": "Mechanism",
#       "cause": "Cause 1",
#       "preventive_barriers": "Barrier"
#     }}
#   ],
#   "consequences": ["Consequence"],
#   "preventive_barriers": [],
#   "mitigative_barriers": []
# }}

# ---

# Example output for multiple failure modes:

# [
#   {{
#     "critical_event": "Critical Event 1",
#     "threats": [
#       {{
#         "mechanism": "Mechanism",
#         "cause": "Cause 1",
#         "preventive_barriers": "Barrier"
#       }}
#     ],
#     "consequences": ["Consequence"],
#     "preventive_barriers": [],
#     "mitigative_barriers": []
#   }},
#   {{
#     "critical_event": "Critical Event 2",
#     "threats": [
#       {{
#         "mechanism": "Mechanism",
#         "cause": "Cause 2",
#         "preventive_barriers": "Barrier"
#       }}
#     ],
#     "consequences": ["Consequence"],
#     "preventive_barriers": [],
#     "mitigative_barriers": []
#   }}
# ]

# ---

# Now convert the following FMEA input into the same JSON structure:
# """
# )

# BOWTIE_COT_PROMPT = (
# """

# You are provided with FMEA table descriptions related to {part}. 

# Follow these step-by-step process to convert them into Bowtie diagram JSON.

# Step 1: Identify each "Failure Mode" and treat it as the "critical_event". If multiple failure modes are present, output an array of Bowtie diagram objects.

# Step 2: For each row, extract:
# - The "cause" (from 'Failure Cause' or 'Cause')
# - The "mechanism" (set to "Mechanism")
# - The "preventive_barriers" (set to "Barrier")

# Step 3: If an "Failure Effect" column is present, extract it as "consequences". Split multiple consequences to new lines if they are distinct.

# Step 4: When parsing bullets or line breaks:
# - The "Cause" is not the same as the "Mechanism" is not the Same as the "Effect" or" Critical Event".
# - Do **not** assume every new line is a new cause or consequence.
# - If a line starts lowercase or continues the phrase, treat it as part of the same item.
# - Only split if clear conjunctions like "and", "or", or commas indicate multiple elements.

# Step 5: Always include:
# - "preventive_barriers": []
# - "mitigative_barriers": []

# Return valid JSON only.
# """
# )
