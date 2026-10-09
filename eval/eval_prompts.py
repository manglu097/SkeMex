"""
Centralized prompt templates for all evaluation scripts.

Three evaluator types are covered:

1. Standard (accuracy)  — eval_standard.py
   MC_JUDGE_*    : multiple-choice datasets, compare option letters
   OPEN_JUDGE_*  : open-ended datasets, semantic equivalence check

2. HealthBench (rubric, normalised score)  — eval_healthbench.py
   HB_GRADER_*   : per-rubric criterion judge

3. LiveMedBench (rubric, weighted sum)  — eval_livemedbbench.py
   LMB_GRADER_*  : per-rubric criterion judge
"""

# ===========================================================================
# 1. Standard evaluator prompts
# ===========================================================================

# -- Multiple-choice judge --------------------------------------------------
# Used for: MMMU, MMMU_Pro, MediQ, AgentClinic_MM,
#           LiveClin_Text, LiveClin_MM, MedXpertQA_Text, MedXpertQA_MM

MC_JUDGE_SYSTEM = (
    "You are an answer checker. "
    "Given a ground-truth option letter and a model prediction, "
    "decide whether the prediction selects the correct option. "
    'Return JSON only: {"correct": true} or {"correct": false}. '
    "No other text."
)

MC_JUDGE_USER = """\
Ground truth: {ground_truth}
Model prediction: {prediction}

Does the prediction select the correct option? Return JSON only."""

# -- Open-ended judge -------------------------------------------------------
# Used for: AgentClinic_Text, MedJourney

OPEN_JUDGE_SYSTEM = (
    "You are an answer checker. "
    "Given a reference answer and a model prediction, "
    "decide whether the prediction is semantically correct or equivalent. "
    'Return JSON only: {"correct": true} or {"correct": false}. '
    "No other text."
)

OPEN_JUDGE_USER = """\
Reference answer: {ground_truth}
Model prediction: {prediction}

Is the prediction correct or equivalent to the reference? Return JSON only."""

# ===========================================================================
# 2. HealthBench grader prompts
# ===========================================================================
# Scoring: sample_score = Σ(criteria_met_i × points_i) / Σ max(points_i, 0)
# One LLM call for ALL rubric items at once.
# Negative-point rubrics describe undesirable behaviours:
#   criteria_met = true  ↔  the model committed the error.

HB_GRADER_SYSTEM = (
    "You are an objective grader. "
    "Given a conversation and a list of rubric criteria, decide whether each criterion "
    "is met by the assistant response. "
    "For criteria describing undesirable behaviour (e.g. 'fails to …', 'is overly verbose'), "
    "return true only if the assistant actually commits that error. "
    "Return a JSON array of booleans only, one per criterion in the same order. "
    "No explanations, no other text. "
    "Example for 3 criteria: [true, false, true]"
)

HB_GRADER_USER = """\
# Conversation
User: {query}
Assistant: {prediction}

# Rubric criteria (in order)
{criteria_list}

Return a JSON boolean array, one entry per criterion, in the same order."""

# ===========================================================================
# 3. LiveMedBench grader prompts
# ===========================================================================
# Scoring: case_total = clamp(Σ(met_i × points_i) / Σ max(points_i, 0), 0, 1)
# One LLM call for ALL rubric items at once.
# Negative-point rubrics describe errors: met=true if the model commits them.

LMB_GRADER_SYSTEM = (
    "You are an objective grader. "
    "Given a user query, a model response, and a list of rubric criteria, "
    "decide whether each criterion is met. "
    "For positive criteria: true if the response contains the required information. "
    "For negative criteria (errors or undesirable behaviours): true if the model commits the described error. "
    "Return a JSON array of booleans only, one per criterion in the same order. "
    "No explanations, no other text. "
    "Example for 3 criteria: [true, false, true]"
)

LMB_GRADER_USER = """\
User query: {query}

Model response: {prediction}

# Rubric criteria (in order)
{criteria_list}

Return a JSON boolean array, one entry per criterion, in the same order."""
