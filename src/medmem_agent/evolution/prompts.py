from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


RETRIEVED_SKILLS_HEADER = """---
## Retrieved Experience Skills
The following skills are distilled from past trajectories and grouped by branch.
Treat them as prioritised reference — consult them **before** deciding how to act.

**Usage rules**:
- Check each skill's **When to apply** condition against the current step; apply every skill whose condition is met — there is no limit on how many you may use simultaneously.
- If one or more skills shape your reasoning or action, cite each one inside your `<reasoning>` block: *(guided by {skill_id})*
- If no skill is relevant to the current step, proceed without citing.
"""




TASK_CLASSIFIER_PROMPT = """You are a medical task classifier for a clinical AI agent system.

Your job is to assign the current medical task query to one high-level, action-oriented category from the existing registry.
These category labels are used downstream to retrieve reusable, methodology-level clinical skills for the agent. 

Therefore, categories MUST represent the "Core Clinical Operation" (what the agent is clinically asked to do, e.g., diagnosis, treatment planning), NOT the "Medical Specialty/Disease" (what it is about), and absolutely NOT the "Question Format" (how it is presented).

## Existing Category Registry
{categories_list}

## Current Task Query
{query}

## Instructions
1. **Classify by Clinical Action**: Identify the primary methodological clinical task (e.g., `differential_diagnosis`, `treatment_planning`, `medical_imaging_interpretation`) and select the **one** best-matching label from the registry.

2. **STRICT ANTI-PATTERNS (What to Avoid)**:
   - ❌ **NO Format-based labels**: Do NOT classify based on how the query is structured. 
     *(Bad: `multiple_choice_question_answering`, `case_report_analysis`, `true_or_false`)*
   - ❌ **NO Scenario-based labels**: Do NOT classify based on medical specialties, body parts, or specific diseases. 
     *(Bad: `ophthalmic-and-neurological-oncology-pathology`, `surgical-management-hepatobiliary`, `pediatric-gynecologic-pathology`)*
   - ✅ **GOOD (Broad, action-oriented)**: `differential_diagnosis`, `treatment_planning`, `information_gathering`, `patient_triage`.

3. **New Label Constraint**: Only if the core clinical action of the query is fundamentally different from ALL existing registry labels, propose a new label and set `is_new` to true.
   - The new label MUST be a broad, action-oriented noun phrase (e.g., `literature_retrieval`).
   - It MUST NOT contain specific medical specialties, disease names, or question formats (like "mcq" or "qa").

4. **Output**: Return a JSON object only.

## Output Format
{{
  "selected": ["label"],
  "new_label": null,
  "is_new": false
}}

Note: If `is_new` is true, "selected" must be []. Otherwise, "selected" must contain exactly one label."""


# ---------------------------------------------------------------------------
# Rubric datasets: HealthBench, LiveMedBench (continuous score, no binary label)
# ---------------------------------------------------------------------------
RUBRIC_DATASETS = frozenset({"healthbench", "livemedbench"})
DATASET_ALIASES = {
    "livemedbbench": "livemedbench",
}


def normalize_dataset_name(dataset: str) -> str:
    dataset_key = (dataset or "").strip().lower()
    return DATASET_ALIASES.get(dataset_key, dataset_key)


# ---------------------------------------------------------------------------
# Stage 1 — Analysis Pass
# Objectives: identify critical steps, attribute skill adoption credit,
# and emit a lightweight output_intent signal (no skill drafting yet).
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT_BINARY = """You are a medical AI knowledge engineer performing trajectory attribution.

## Task Info
- Dataset: {dataset}
- Task Category: {task_category}
- Final Outcome: {final_outcome}
- Ground Truth: {ground_truth_block}
## Available Tools for this Task
{available_tools_list}

## Injected Skills (if any)
{injected_skills_content}

## Agent Trajectory (steps)
{formatted_steps}

## Instructions
1. **Skill Attribution**: Evaluate each injected skill's influence on the agent's behavior.
   - ADOPTED_POSITIVE: Skill influenced a key decision and contributed to success.
   - ADOPTED_NEGATIVE: Skill influenced a key decision but led to an error.
   - IGNORED: Skill had zero observable influence — behavior would be identical without it. Reserve this label; the bar for ADOPTED is low.
   *(If no skills were injected, output `[]`)*

2. **Pattern Extraction**: Identify the single most decisive turning point. Formulate a dense, actionable takeaway: name the specific tool(s) involved, the exact error or success mechanism, and the reusable lesson.

3. **Output Intent & Branch Signal**: Decide whether to extract new knowledge.
   - Output NONE if the pattern is too case-specific, had no direct effect, or fails the **Granularity gate**: reject fragments tied to one case, one rubric item, or one narrow step-order variant.
   - Output PATCH if the pattern refines an existing injected skill; set `patch_target_id`.
   - Output CREATE if a new reusable pattern was found.
   - **Branch** (required for CREATE/PATCH):
     - `action_level`: Root cause is wrong parameters, input format, or misread output for one specific tool. `tools_involved` must contain exactly one tool name — this becomes `tool_name` in the Mutation Pass.
     - `task_level`: Root cause is wrong tool choice, a missing tool call, or incorrect tool sequence. `tools_involved` should name the relevant tools.
     - `general`: Root cause is a flawed **domain-agnostic** reasoning or search pattern — the lesson must apply equally to non-medical tasks (legal, financial, historical search). Must be expressible as a concrete "When X, do Y" correction. `tools_involved` must be `[]`. If the corrective action references symptoms, drugs, diagnoses, or clinical context, it is NOT `general` — use `task_level` instead. Do NOT use for vague lessons; output NONE instead.
     - When the fix involves any tool, prefer `action_level` or `task_level` over `general`.

Output a JSON object only:
{{
  "skill_adoption": [{{"skill_id": str, "status": str}}],
  "extracted_pattern": str,
  "output_intent": "CREATE" | "PATCH" | "NONE",
  "patch_target_id": str | null,
  "branch_signal": {{
    "suggested_branch": "general" | "task_level" | "action_level",
    "tools_involved": [str],
    "is_single_tool_focus": bool
  }} | null
}}"""


_ANALYSIS_PROMPT_RUBRIC = """You are a medical AI knowledge engineer performing trajectory attribution.

## Task Info
- Dataset: {dataset}
- Task Category: {task_category}
- Rubric Score: {eval_result:.2f} / 1.00
- Unmet Criteria: {unmet_rubrics_block}
## Available Tools for this Task
{available_tools_list}

## Injected Skills (if any)
{injected_skills_content}

## Agent Trajectory (steps)
{formatted_steps}

## Instructions
1. **Skill Attribution**: Evaluate each injected skill's influence on the agent's behavior.
   - ADOPTED_POSITIVE: Skill influenced a key decision and contributed to a higher rubric score.
   - ADOPTED_NEGATIVE: Skill influenced a key decision but caused a rubric criterion to fail.
   - IGNORED: Skill had zero observable influence — behavior would be identical without it. Reserve this label; the bar for ADOPTED is low.
   *(If no skills were injected, output `[]`)*

2. **Pattern Extraction**: Identify the single pattern that most directly caused rubric criteria to be met or missed. Formulate a dense, actionable takeaway: name the specific tool(s) involved, the exact error or success mechanism, and the reusable lesson.

3. **Output Intent & Branch Signal**: Decide whether to extract new knowledge.
   - Output NONE if the pattern had no direct effect on the score, is too case-specific, or fails the **Granularity gate**: reject fragments tied to one case, one rubric item, or one narrow step-order variant.
   - Output PATCH if the pattern refines an existing injected skill; set `patch_target_id`.
   - Output CREATE if a new reusable pattern was found.
   - **Branch** (required for CREATE/PATCH):
     - `action_level`: Root cause is wrong parameters, input format, or misread output for one specific tool. `tools_involved` must contain exactly one tool name — this becomes `tool_name` in the Mutation Pass.
     - `task_level`: Root cause is wrong tool choice, a missing tool call, or incorrect tool sequence. `tools_involved` should name the relevant tools.
     - `general`: Root cause is a flawed **domain-agnostic** reasoning or search pattern — the lesson must apply equally to non-medical tasks (legal, financial, historical search). Must be expressible as a concrete "When X, do Y" correction. `tools_involved` must be `[]`. If the corrective action references symptoms, drugs, diagnoses, or clinical context, it is NOT `general` — use `task_level` instead. Do NOT use for vague lessons; output NONE instead.
     - When the fix involves any tool, prefer `action_level` or `task_level` over `general`.

Output a JSON object only:
{{
  "skill_adoption": [{{"skill_id": str, "status": str}}],
  "extracted_pattern": str,
  "output_intent": "CREATE" | "PATCH" | "NONE",
  "patch_target_id": str | null,
  "branch_signal": {{
    "suggested_branch": "general" | "task_level" | "action_level",
    "tools_involved": [str],
    "is_single_tool_focus": bool
  }} | null
}}"""


# ---------------------------------------------------------------------------
# Stage 2 — Mutation Pass
# Triggered only when output_intent is CREATE or PATCH.
# Objectives: produce a concrete skill draft or patch content.
# Branch guidance and tool_name requirement are enforced here.
# ---------------------------------------------------------------------------

_MUTATION_PROMPT = """You are a medical AI knowledge engineer. Based on the trajectory analysis below,
produce a concrete skill mutation for the medical agent skill library.

## Task Info
- Dataset: {dataset}
- Task Category: {task_category}
- Output Intent: {output_intent}{patch_target_line}

## Extracted Methodology Pattern
{extracted_pattern_text}

## Branch Signal from Analysis
{branch_signal_block}

{injected_skills_block}

## Instructions

### Step 1 — Confirm Branch
Use the suggested branch as a starting point, but verify it matches these definitions:

- **general**: A **domain-agnostic** behavioral habit correcting a recurring reasoning or search pattern. The lesson must apply equally to non-medical tasks. MUST have a concrete trigger AND a specific corrective action. `required_tools` must be `[]`. If the action references symptoms, drugs, diagnoses, or clinical context, use `task_level` instead. Reject pure principles without a trigger+action pair.
  - *Example*: description: "When the first search returns no direct answer, rephrase with synonyms or broader terms before retrying." | core_action: "If the initial search yields no useful result, do NOT repeat the same query. Instead: (1) substitute domain-specific terms with common synonyms; (2) drop overly specific qualifiers; (3) run the reformulated query. Stop once a source directly supports the answer."

- **task_level**: Corrects wrong tool choice, a missing tool call, or incorrect tool sequence for this task category. The skill must name the recommended tool(s), explain why that order matters, and list them in `required_tools`.
  - *Example*: description: "Verify renal and hepatic function before recommending any systemic medication." | core_action: "Before finalizing a drug recommendation: (1) call `drug_info_lookup` for contraindications and dosage adjustments; (2) call `drug_interaction_check` against current medications. Only proceed if both return no major flags."

- **action_level**: Corrects exact parameters, input format, or output handling of ONE specific tool. `tool_name` must be set to the exact function name. Must include a concrete parameter example.
  - *Example*: description: "When tavily_search returns empty results, reformulate the query with synonyms or broader terms." | core_action: "If `tavily_search` returns no useful results, call it again with a reformulated `query`: drop specific qualifiers or substitute synonyms (e.g., 'myocardial infarction' → 'heart attack'). Example: `{{\"query\": \"heart attack first-line treatment\", \"search_depth\": \"advanced\"}}`"

### Step 2 — Draft or Patch

**If CREATE**: produce a complete skill draft.
- `description`: Single sentence stating the specific situational trigger — not a capability claim.
- `applicable_conditions`: Abstract enough to cover the full case class — do NOT anchor to this trajectory's disease, dataset, or scenario.
- `core_action`: Follow the "Trigger → Action" pattern. Start with the condition ("If X...", "When Y..."), then the exact corrective action. Use the few-shot examples above as format reference. Do NOT write generic principles.

**If PATCH**: modify ONLY `core_action` of the target skill. Preserve its structure, `description`, and `applicable_conditions` unchanged.

**Anti-fragmentation rule**: Do NOT create a skill for a single case, disease-specific restatement, or narrow step-order variant that does not generalize across future tasks.

Output a JSON object only:
{{
  "output_action": "CREATE" | "PATCH" | "NONE",
  "skill_draft": {{
    "name": str,
    "description": str,
    "applicable_conditions": str,
    "branch": "general" | "task_level" | "action_level",
    "tool_name": str | null,
    "task_category": [str],
    "required_tools": [str],
    "risk_level": "low" | "medium" | "high",
    "core_action": str
  }} | null,
  "patch_target_id": str | null,
  "patch_section": "Core Action" | null,
  "patch_content": str | null
}}"""


DRAFT_REVIEW_PROMPT = """You are a quality gatekeeper for a medical AI skill library.

## Capacity Status
{capacity_note}

## Existing Skills in This Group
{existing_descriptions}

## Draft Skill
{draft_skill}

## Evaluation Criteria

### 1. Novelty
Pass if the draft's description AND core action together represent a meaningfully distinct
operational pattern that cannot be expressed by patching any existing skill.
- Fail if the draft's core action is a subset, superset, or minor reordering of an existing skill's core action.
- Fail if the only difference from an existing skill is a disease name, drug class, body system, question
  format, or dataset-specific phrasing — these are PATCH targets, not new skills.
- Fail if the draft could be absorbed into an existing skill by adding one or two sentences to its core action.
- If the group has room and no existing skill overlaps, pass novelty.

### 2. Quality
Pass if ALL sub-criteria are met:
- **Description**: A single concise sentence stating the situational trigger ("when to apply" signal).
  Fail if vague (e.g., "when the task is complex") or reads as a general capability claim.
- **Core action**: Specific, executable steps, tool call templates, or parameter examples an agent can follow
  directly. For `task_level`, must name the recommended tool(s) or sequence. For `action_level`, must include
  exact parameter names or values. Fail if it reads as a general principle or lacks concrete operational detail.
- **Branch alignment**: `core_action` must match the declared branch — `general`: domain-agnostic reasoning/search strategy only (no tool names, no medical terminology); `task_level`: tool selection/sequence; `action_level`: fine-grained single-tool usage. Reject a `general` skill if its core action contains symptoms, drugs, diagnoses, or any clinical context.

### 3. Scope Quality / Anti-fragmentation
Pass if the skill generalises across multiple future cases in the same task family.
- Fail if anchored to a single patient case, exam stem, or trajectory.
- Fail if it encodes a single rubric item, dataset artifact, or one-off parameter combination.
- When in doubt about scope, prefer approving and letting governance deprecate it later over rejecting a
  potentially useful pattern.

### 4. PATCH vs. CREATE
Before approving CREATE, ask: could this insight be fully captured by patching an existing skill?
If yes, fail novelty and name the skill that should be patched instead.

## Output
Return a JSON object only. The `reason` field MUST name the specific overlapping skill ID or
quote the exact phrase that caused failure.

{{
  "pass_novelty": true | false,
  "pass_quality": true | false,
  "reason": "Precise one or two sentences. If failing novelty, name the overlapping skill. If failing quality, quote the problematic phrase. If failing scope, describe the fragmentation pattern."
}}"""


MERGE_PROMPT = """You are merging two similar medical agent skills into one concise skill.

## Dominant Skill (preserve its structure)
{dominant_skill}

## Secondary Skill (extract non-redundant additions only)
{secondary_skill}

## Instructions
1. **Keep**: The dominant skill's Core Action structure and all existing content.
2. **Add**: Only content from the secondary skill that is non-redundant and actionable —
   new decision steps, tool recommendations, parameter details, or failure modes not already covered.
3. **Discard**: Anything that overlaps, contradicts, or is implied by existing content.
4. **Condense**: After merging, tighten the language. Remove filler phrases, merge similar
   bullet points, and ensure each line earns its place.
5. **Output**: Return the merged skill body only (the `### Core Action` section and any optional
   boundary note — no front matter, no commentary)."""


PRE_SCREEN_PROMPT = """You are a medical AI retrieval assistant for a clinical AI agent system.

Your job is to pre-screen the most relevant skills from the library for the current medical task.
The selected skill IDs will enter a scoring pipeline — only the top-ranked ones are injected into the agent,
so selecting the most semantically relevant candidates directly improves the agent's performance.

Skills are organised into three branches with distinct roles:
- **general**: High-level reasoning strategy for approaching this class of problem.
- **task_level**: Tool selection and workflow guidance for this specific task category.
- **action_level**: Fine-grained usage guidance for a single specific tool.

## Current Task Query
{query}

## Available Skills (Grouped by Branch)
{skills_by_branch}

## Instructions
1. **Select**: For each branch, choose up to {top_k} skill IDs whose description best matches
   the query's clinical intent. Prioritise specificity over breadth.
2. **Non-empty branch rule**: If a branch contains any candidate skills, you MUST return
   `min({top_k}, branch_size)` IDs from that branch, even when relevance is weak.
   Only return `[]` when the branch itself is empty.
3. **Validity**: Never include IDs not listed above.
4. **Output**: Return a JSON object only.

## Output Format
{{
  "general": ["skill_id_1", ...],
  "task_level": ["skill_id_2", ...],
  "action_level": ["skill_id_3", ...]
}}

Note: All three keys must be present. Each value is a list of skill IDs (strings). Use `[]` only when the corresponding branch has zero available skills."""


KEY_FINDING_PIN_PROMPT = """You are extracting confirmed high-value findings from a medical agent trajectory.

These findings will be pinned and re-injected into every subsequent step to prevent critical information
from being lost when earlier observations are compressed.

## Trajectory Excerpt
{steps_excerpt}

## Instructions
1. **Extract**: Identify all confirmed facts from the observations — these may include clinical findings
   (symptoms, signs, vitals, imaging, labs, allergy history), patient-reported information,
   database or guideline lookups, literature evidence, or tool query results.
2. **Confirmed only**: Include only facts that are explicitly established in the observations.
   Do not infer, speculate, or include pending results.
3. **Concise**: Each bullet must be one short phrase or sentence. No elaboration.
4. **Output**: Return plain text starting with `[Confirmed Findings]`, one bullet per line.

[Confirmed Findings]"""


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def render_categories_list(categories: Iterable[dict]) -> str:
    lines: List[str] = []
    for item in categories:
        lines.append(f"- {item.get('label')}: {item.get('description', '')}")
    return "\n".join(lines) or "- None"


def build_task_classifier_prompt(categories: Iterable[dict], query: str) -> str:
    return TASK_CLASSIFIER_PROMPT.format(
        categories_list=render_categories_list(categories),
        query=query,
    )


def _is_rubric_dataset(dataset: str) -> bool:
    return normalize_dataset_name(dataset) in RUBRIC_DATASETS


def _build_ground_truth_block(ground_truth: Any) -> str:
    """For binary datasets: inject GT only when the answer is wrong (INCORRECT)."""
    if ground_truth is None:
        return ""
    gt_str = str(ground_truth).strip()
    if not gt_str:
        return ""
    return f"- Correct Answer: {gt_str}\n  (Use this only to identify where reasoning diverged, not to memorize the answer.)\n"


def _build_unmet_rubrics_block(rubric_results: Optional[List[Dict[str, Any]]]) -> str:
    """For rubric datasets: list unmet criteria regardless of overall score."""
    if not rubric_results:
        return ""
    unmet = [
        item for item in rubric_results
        if not item.get("criteria_met", item.get("met", True))
    ]
    if not unmet:
        return ""
    lines = ["- Unmet Criteria (use only to identify reasoning gaps, not to memorize answers):"]
    for item in unmet:
        criterion = item.get("criterion", "").strip()
        points = item.get("points", 0)
        if criterion:
            lines.append(f"  - [{points:+g}] {criterion}")
    return "\n".join(lines) + "\n"


def build_analysis_prompt(
    *,
    dataset: str,
    task_category: List[str],
    eval_result: float,
    injected_skills_content: str,
    formatted_steps: str,
    ground_truth: Any = None,
    rubric_results: Optional[List[Dict[str, Any]]] = None,
    available_tools_list: str = "",
) -> str:
    """Build the Stage-1 Analysis Pass prompt.

    Routes to the rubric or binary variant based on dataset name.
    """
    task_cat_str = ", ".join(task_category) if task_category else "None"
    skills_str = injected_skills_content or "None"
    tools_str = available_tools_list or "None"

    if _is_rubric_dataset(dataset):
        unmet_block = _build_unmet_rubrics_block(rubric_results)
        return _ANALYSIS_PROMPT_RUBRIC.format(
            dataset=dataset,
            task_category=task_cat_str,
            eval_result=eval_result,
            unmet_rubrics_block=unmet_block,
            available_tools_list=tools_str,
            injected_skills_content=skills_str,
            formatted_steps=formatted_steps,
        )

    # Binary / accuracy datasets
    is_incorrect = eval_result <= 0.5
    gt_block = _build_ground_truth_block(ground_truth) if is_incorrect else ""
    return _ANALYSIS_PROMPT_BINARY.format(
        dataset=dataset,
        task_category=task_cat_str,
        final_outcome="CORRECT" if not is_incorrect else "INCORRECT",
        ground_truth_block=gt_block,
        available_tools_list=tools_str,
        injected_skills_content=skills_str,
        formatted_steps=formatted_steps,
    )


def build_mutation_prompt(
    *,
    dataset: str,
    task_category: List[str],
    output_intent: str,
    extracted_pattern: str,
    injected_skills_content: str,
    patch_target_id: Optional[str] = None,
    branch_signal: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the Stage-2 Mutation Pass prompt.

    Only called when output_intent is CREATE or PATCH.

    For PATCH: ``patch_target_id`` is shown explicitly in Task Info, and
    ``injected_skills_content`` should contain only the target skill body.
    For CREATE: ``injected_skills_content`` is ignored (set to empty) since
    there is no existing skill to reference.

    ``branch_signal`` is the structured branch hint produced by the Analysis
    Pass.  When present it is rendered into the prompt so the Mutation LLM
    can confirm or override the suggested branch with full context.
    """
    import json as _json
    task_cat_str = ", ".join(task_category) if task_category else "None"
    pattern_text = extracted_pattern.strip() if extracted_pattern else "No pattern extracted."

    # Render branch_signal block
    if branch_signal and isinstance(branch_signal, dict):
        branch_signal_block = (
            f"- Suggested branch: **{branch_signal.get('suggested_branch', 'unknown')}**\n"
            f"- Tools involved: {branch_signal.get('tools_involved', [])}\n"
            f"- Single-tool focus: {branch_signal.get('is_single_tool_focus', False)}"
        )
    else:
        branch_signal_block = "(No branch signal available — determine branch from the pattern above.)"

    if output_intent == "PATCH" and patch_target_id:
        patch_target_line = f"\n- Patch Target Skill ID: {patch_target_id}"
        skills_block = (
            f"## Target Skill to Patch\n{injected_skills_content.strip() or 'None'}"
        )
    else:
        # CREATE: no patch target, no injected skill context needed
        patch_target_line = ""
        skills_block = ""

    return _MUTATION_PROMPT.format(
        dataset=dataset,
        task_category=task_cat_str,
        output_intent=output_intent,
        patch_target_line=patch_target_line,
        extracted_pattern_text=pattern_text,
        branch_signal_block=branch_signal_block,
        injected_skills_block=skills_block,
    )


# Keep old name as alias so any external callers are not broken during transition.
def build_encode_prompt(
    *,
    dataset: str,
    task_category: List[str],
    eval_result: float,
    injected_skills_content: str,
    formatted_steps: str,
    ground_truth: Any = None,
    rubric_results: Optional[List[Dict[str, Any]]] = None,
    available_tools_list: str = "",
) -> str:
    """Backward-compatible alias — delegates to build_analysis_prompt."""
    return build_analysis_prompt(
        dataset=dataset,
        task_category=task_category,
        eval_result=eval_result,
        injected_skills_content=injected_skills_content,
        formatted_steps=formatted_steps,
        ground_truth=ground_truth,
        rubric_results=rubric_results,
        available_tools_list=available_tools_list,
    )


def build_draft_review_prompt(
    existing_descriptions: str,
    draft_skill: str,
    *,
    current_count: int = 0,
    capacity: int = 10,
    group_label: str = "general",
) -> str:
    slots_remaining = max(0, capacity - current_count)
    if slots_remaining == 0:
        capacity_note = (
            f"The group `{group_label}` is FULL ({current_count}/{capacity} slots used). "
            "Only approve if this draft is clearly superior to an existing skill AND that "
            "skill will be deprecated to free a slot. Otherwise fail novelty."
        )
    elif slots_remaining <= max(1, capacity // 4):
        capacity_note = (
            f"The group `{group_label}` is nearly full ({current_count}/{capacity} slots used, "
            f"{slots_remaining} remaining). Apply stricter novelty scrutiny."
        )
    else:
        capacity_note = (
            f"The group `{group_label}` has room ({current_count}/{capacity} slots used, "
            f"{slots_remaining} remaining). Approve if the draft clears quality and novelty."
        )
    return DRAFT_REVIEW_PROMPT.format(
        existing_descriptions=existing_descriptions or "None",
        draft_skill=draft_skill,
        capacity_note=capacity_note,
    )


def build_merge_prompt(dominant_skill: str, secondary_skill: str) -> str:
    return MERGE_PROMPT.format(dominant_skill=dominant_skill, secondary_skill=secondary_skill)


def build_key_finding_pin_prompt(steps_excerpt: str) -> str:
    return KEY_FINDING_PIN_PROMPT.format(steps_excerpt=steps_excerpt)


def build_pre_screen_prompt(
    *,
    query: str,
    skills_by_branch: Dict[str, Any],
    top_k: int,
) -> str:
    """Render the PRE_SCREEN_PROMPT with branch-grouped skill metadata."""
    import json as _json
    return PRE_SCREEN_PROMPT.format(
        query=query,
        skills_by_branch=_json.dumps(skills_by_branch, ensure_ascii=False, indent=2),
        top_k=top_k,
    )


def render_retrieved_skills_block(skills: List[Dict[str, Any]]) -> str:
    """Render a list of retrieved skills into the agent-facing block.

    Each skill is rendered with an explicit "When to apply" line sourced from
    ``applicable_conditions``, so the agent can decide at inference time whether
    the skill is relevant to the current step — rather than blindly following
    every injected skill regardless of context.
    """
    if not skills:
        return ""
    lines = [RETRIEVED_SKILLS_HEADER]
    for skill in skills:
        skill_id = skill.get("skill_id", "")
        name = skill.get("name", "")
        applicable_conditions = skill.get("applicable_conditions", "").strip()
        body = skill.get("body", "")
        when_to_apply_line = (
            f"**When to apply**: {applicable_conditions}\n"
            if applicable_conditions
            else ""
        )
        lines.append(f"### {skill_id} | {name}\n{when_to_apply_line}{body}")
    return "\n\n".join(lines)
