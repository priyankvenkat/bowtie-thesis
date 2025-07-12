from config import BOWTIE_COT_PROMPT, BOWTIE_FEW_SHOT_PROMPT, BOWTIE_ZERO_SHOT_PROMPT
# BOWTIE_INFER_PROMPT, BOWTIE_PROMPT


def get_prompt(prompt_type, part):
    """
    Return the prompt text based on the prompt type and part name.
    """
    if prompt_type == "zero":
        return BOWTIE_ZERO_SHOT_PROMPT.format(part=part)
    elif prompt_type == "few":
        return BOWTIE_FEW_SHOT_PROMPT.format(part=part)
    elif prompt_type == "cot":
        return BOWTIE_COT_PROMPT.format(part=part)
    return ""