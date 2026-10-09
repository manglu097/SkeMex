"""
Centralized prompt templates for all datasets.

Each dataset has its own prompt-building function that takes the raw sample
fields and returns the final query string to be passed to the agent.
Image paths are always handled separately by the loader (passed via the
`images` key) and are never embedded in the query text.
"""

# ---------------------------------------------------------------------------
# HealthBench
# ---------------------------------------------------------------------------
# user_input : directly use the first content block of the prompt field.
# No extra instruction appended — HealthBench questions are already
# open-ended instructions.

def build_healthbench_query(prompt_content: str) -> str:
    """Build query for HealthBench.

    Args:
        prompt_content: sample['prompt'][0]['content']
    """
    return prompt_content


# ---------------------------------------------------------------------------
# MMMU
# ---------------------------------------------------------------------------
MMMU_SUFFIX = "\nPlease output only the option letter, e.g. A or B."


def build_mmmu_query(question: str, options: list) -> str:
    """Build query for MMMU.

    Args:
        question: sample['question']
        options:  list parsed from sample['options']
    """
    query = question + "\nOptions:\n"
    for i, opt in enumerate(options):
        query += f"{chr(65 + i)}. {opt}\n"
    query += MMMU_SUFFIX
    return query


# ---------------------------------------------------------------------------
# MMMU_Pro
# ---------------------------------------------------------------------------
MMMU_PRO_SUFFIX = "\nPlease output only the option letter, e.g. A or B."


def build_mmmu_pro_query(question: str, options: list) -> str:
    """Build query for MMMU_Pro.

    Args:
        question: sample['question']
        options:  list parsed from sample['options']
    """
    query = question + "\nOptions:\n"
    for i, opt in enumerate(options):
        query += f"{chr(65 + i)}. {opt}\n"
    query += MMMU_PRO_SUFFIX
    return query


# ---------------------------------------------------------------------------
# MediQ
# ---------------------------------------------------------------------------
MEDIQ_SUFFIX = "\nPlease output only the option letter, e.g. A or B."


def build_mediq_query(context: list, question: str, options: dict) -> str:
    """Build query for MediQ.

    Args:
        context:  sample['context']  (list of strings)
        question: sample['question']
        options:  sample['options']  (dict, e.g. {'A': '...', 'B': '...'})
    """
    context_text = "\n".join(context)
    query = context_text + "\n\n" + question + "\nOptions:\n"
    for key in sorted(options.keys()):
        query += f"{key}. {options[key]}\n"
    query += MEDIQ_SUFFIX
    return query


# ---------------------------------------------------------------------------
# AgentClinic_Text
# ---------------------------------------------------------------------------
AGENTCLINIC_TEXT_PREFIX = (
    "You can actively ask the patient questions about their symptoms, "
    "medical history, and lifestyle, and request any necessary physical "
    "examinations or lab tests to gather sufficient information before "
    "making a diagnosis."
    "Below is all of the information you have.\n\n"
)
AGENTCLINIC_TEXT_SUFFIX = (
    "\n\nProvide the single most likely diagnosis. "
    "Output the diagnosis only, no explanation."
)
    
def build_agentclinic_text_query(objective: str) -> str:
    """Build query for AgentClinic_Text.

    Args:
        objective: sample['OSCE_Examination']['Objective_for_Doctor']
    """
    return AGENTCLINIC_TEXT_PREFIX + objective + AGENTCLINIC_TEXT_SUFFIX


# ---------------------------------------------------------------------------
# AgentClinic_MM
# ---------------------------------------------------------------------------
AGENTCLINIC_MM_PREFIX = (
    "You can actively ask the patient questions about their symptoms, "
    "medical history, and lifestyle, and request any necessary physical "
    "examinations or imaging studies to gather sufficient information "
    "before making a diagnosis.\n\n"
)
AGENTCLINIC_MM_SUFFIX = (
    "\n\nProvide the single most likely diagnosis "
    "Output the option letter only, no explanation."
)

def build_agentclinic_mm_query(question: str, answers: list) -> str:
    """Build query for AgentClinic_MM.

    Options are constructed from the answers list; the correct flag is NOT
    revealed — all options are listed in order.

    Args:
        question: sample['question']
        answers:  sample['answers']  (list of {'text': str, 'correct': bool})
    """
    query = AGENTCLINIC_MM_PREFIX + question + "\nOptions:\n"
    for i, ans in enumerate(answers):
        query += f"{chr(65 + i)}. {ans['text']}\n"
    query += AGENTCLINIC_MM_SUFFIX
    return query


# ---------------------------------------------------------------------------
# LiveClin_Text
# ---------------------------------------------------------------------------
LIVECLIN_TEXT_SUFFIX = "\nPlease output only the option letter, e.g. A or B."


def build_liveclin_text_query(scenario: str, question: str, options: dict) -> str:
    """Build query for LiveClin_Text.

    Args:
        scenario: sample['scenario']
        question: sample['question']
        options:  sample['options']  (dict)
    """
    query = scenario + "\n\n" + question + "\nOptions:\n"
    for key in sorted(options.keys()):
        query += f"{key}. {options[key]}\n"
    query += LIVECLIN_TEXT_SUFFIX
    return query


# ---------------------------------------------------------------------------
# LiveClin_MM
# ---------------------------------------------------------------------------
# Images are handled separately by the loader; only the text portion is built
# here.
LIVECLIN_MM_SUFFIX = "\nPlease output only the option letter, e.g. A or B."


def build_liveclin_mm_query(
    scenario: str,
    scenario_table_details: list,
    question: str,
    options: dict,
) -> str:
    """Build query for LiveClin_MM (text portion only; images handled separately).

    Args:
        scenario:               sample['scenario']
        scenario_table_details: sample.get('scenario_table_details', [])
        question:               sample['question']
        options:                sample['options']  (dict)
    """
    query = scenario
    for table in scenario_table_details:
        query += f"\n\n{table['caption']}:\n{table['content']}"
    query += "\n\n" + question + "\nOptions:\n"
    for key in sorted(options.keys()):
        query += f"{key}. {options[key]}\n"
    query += LIVECLIN_MM_SUFFIX
    return query


# ---------------------------------------------------------------------------
# MedXpertQA_Text
# ---------------------------------------------------------------------------
MEDXPERTQA_TEXT_SUFFIX = "\nPlease output only the option letter, e.g. A or B."


def build_medxpertqa_text_query(question: str, options: dict) -> str:
    """Build query for MedXpertQA_Text.

    Args:
        question: sample['question']
        options:  sample['options']  (dict)
    """

    # Remove the legacy answer-choice block and everything after it.
    question = question.split("\nAnswer Choices:")[0].strip()

    query = question + "\nOptions:\n"
    for key in sorted(options.keys()):
        query += f"{key}. {options[key]}\n"

    query += MEDXPERTQA_TEXT_SUFFIX
    return query


# ---------------------------------------------------------------------------
# MedXpertQA_MM
# ---------------------------------------------------------------------------
# Images are handled separately by the loader.
MEDXPERTQA_MM_SUFFIX = "\nPlease output only the option letter, e.g. A or B."


def build_medxpertqa_mm_query(question: str, options: dict) -> str:
    """Build query for MedXpertQA_MM (text portion only; images handled separately).

    Args:
        question: sample['question']
        options:  sample['options']  (dict)
    """

    # Remove the legacy answer-choice block and everything after it.
    question = question.split("\nAnswer Choices:")[0].strip()

    query = question + "\nOptions:\n"
    for key in sorted(options.keys()):
        query += f"{key}. {options[key]}\n"
    query += MEDXPERTQA_MM_SUFFIX
    return query


# ---------------------------------------------------------------------------
# MedJourney
# ---------------------------------------------------------------------------
# user_input: directly use sample['prompt'] — no modification needed.

def build_medjourney_query(prompt: str) -> str:
    """Build query for MedJourney.

    Args:
        prompt: sample['prompt']
    """
    suffix = (
        "\n\nPlease provide a concise answer focusing only on the key points. "
        "Avoid unnecessary explanations."
    )
    return prompt.strip() + suffix

# ---------------------------------------------------------------------------
# LiveMedBench
# ---------------------------------------------------------------------------
# user_input: narrative + core_request — no extra suffix needed.

def build_livemedbbench_query(narrative: str, core_request: str) -> str:
    """Build query for LiveMedBench.

    Args:
        narrative:    sample['narrative']
        core_request: sample['core_request']
    """
    return narrative + "\n\n" + core_request
