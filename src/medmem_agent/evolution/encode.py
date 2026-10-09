from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

from medmem_agent.config import LLMConfig
from medmem_agent.llm import build_llm, ContentFilterError

from .config import EvoConfig
from .prompts import (
    build_analysis_prompt,
    build_draft_review_prompt,
    build_mutation_prompt,
)
from .storage import SkillStore
from .types import EncodeResult, SkillAdoptionRecord, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEncoder:
    store: SkillStore
    evo_config: EvoConfig
    provider: str = "openai_compatible"
    api_base: str = "https://api.openai.com/v1"
    api_key: str | None = None

    # Mapping from adoption status symbol to the discrete credit weight used
    # by downstream utility updates.  Declared as ClassVar so dataclass does
    # not treat it as an instance field.
    _STATUS_TO_WEIGHT: ClassVar[Dict[str, float]] = {
        "ADOPTED_POSITIVE": 1.0,
        "ADOPTED_NEGATIVE": -1.0,
        "IGNORED": 0.2,
    }

    def encode(self, trajectory: Trajectory, available_tools_list: str = "") -> EncodeResult:
        """Two-pass encode: Analysis Pass then conditional Mutation Pass.

        Pass 1 — Analysis Pass
        ~~~~~~~~~~~~~~~~~~~~~~
        Always runs. Produces:
        - ``extracted_pattern``: a dense, methodology-level description of the
          decisive pattern discovered in the trajectory (success strategy or
          failure root-cause), used as the sole context for Pass 2.
        - ``skill_adoption``: credit attribution for each injected skill.
        - ``output_intent``: lightweight signal (CREATE / PATCH / NONE).

        Pass 2 — Mutation Pass
        ~~~~~~~~~~~~~~~~~~~~~~
        Runs only when ``output_intent`` is CREATE or PATCH. Produces:
        - ``output_action``: final action (may be downgraded to NONE).
        - ``skill_draft`` or ``patch_*`` fields.

        This separation lets the LLM focus on pattern extraction first, then on
        knowledge drafting with the full benefit of the analysis context.

        Args:
            trajectory: The trajectory to encode.
            available_tools_list: Pre-rendered string listing all tools available
                for this dataset (name + description). Injected into the Analysis
                prompt to enable "exploration gap" detection.
        """
        injected_text = self._render_injected_skills(trajectory.injected_skill_ids)
        obs_limit = self.evo_config.context_guard.context_trim_observation_chars
        steps_text = self._format_steps(trajectory.steps, obs_limit=obs_limit)
        eval_result = float(trajectory.eval_result or 0.0)

        ground_truth = trajectory.ground_truth if eval_result <= 0.5 else None
        rubric_results: Optional[List[Dict[str, Any]]] = (
            trajectory.metadata.get("rubric_results") if trajectory.metadata else None
        )

        # ------------------------------------------------------------------
        # Pass 1: Analysis
        # ------------------------------------------------------------------
        analysis_prompt = build_analysis_prompt(
            dataset=trajectory.dataset,
            task_category=trajectory.task_category,
            eval_result=eval_result,
            injected_skills_content=injected_text,
            formatted_steps=steps_text,
            ground_truth=ground_truth,
            rubric_results=rubric_results,
            available_tools_list=available_tools_list,
        )
        llm = self._build_llm(self.evo_config.encode.model, max_tokens=1000)
        try:
            raw_analysis = llm.generate(
                system_prompt="You output strict JSON only.",
                user_prompt=analysis_prompt,
            ).content.strip()
        except (ContentFilterError, RuntimeError) as e:
            logger.warning(
                "Analysis Pass LLM call failed for trajectory %s; skipping encode. Error: %s",
                trajectory.id, e,
            )
            return EncodeResult(
                trajectory_id=trajectory.id,
                extracted_pattern="",
                skill_adoption=[],
                output_action="NONE",
                skill_draft=None,
                patch_target_id=None,
                patch_section=None,
                patch_content=None,
                raw_analysis="",
                raw_mutation="",
            )

        analysis_data = self._safe_parse_json(raw_analysis, trajectory.id, pass_name="analysis")
        extracted_pattern = str(analysis_data.get("extracted_pattern") or "").strip()
        skill_adoption = self._parse_skill_adoption(analysis_data)
        output_intent = str(analysis_data.get("output_intent", "NONE")).upper()
        patch_target_from_analysis = analysis_data.get("patch_target_id")
        # Extract structured branch hint from Analysis Pass (may be None for NONE intent)
        branch_signal = (
            analysis_data.get("branch_signal")
            if isinstance(analysis_data.get("branch_signal"), dict)
            else None
        )

        # ------------------------------------------------------------------
        # Pass 2: Mutation (only when there is an actionable intent)
        # ------------------------------------------------------------------
        output_action = "NONE"
        skill_draft: Optional[Dict[str, Any]] = None
        patch_target_id: Optional[str] = None
        patch_section: Optional[str] = None
        patch_content: Optional[str] = None
        raw_mutation = ""

        if output_intent in ("CREATE", "PATCH"):
            # For PATCH: inject only the target skill body to avoid noise.
            # For CREATE: no injected skill context needed.
            if output_intent == "PATCH" and patch_target_from_analysis:
                mutation_skills_content = self._render_single_skill(patch_target_from_analysis)
            else:
                mutation_skills_content = ""

            mutation_prompt = build_mutation_prompt(
                dataset=trajectory.dataset,
                task_category=trajectory.task_category,
                output_intent=output_intent,
                extracted_pattern=extracted_pattern,
                injected_skills_content=mutation_skills_content,
                patch_target_id=patch_target_from_analysis,
                branch_signal=branch_signal,
            )
            llm_mutation = self._build_llm(self.evo_config.encode.model, max_tokens=1200)
            try:
                raw_mutation = llm_mutation.generate(
                    system_prompt="You output strict JSON only.",
                    user_prompt=mutation_prompt,
                ).content.strip()
            except (ContentFilterError, RuntimeError) as e:
                logger.warning(
                    "Mutation Pass LLM call failed for trajectory %s; skipping mutation. Error: %s",
                    trajectory.id, e,
                )
                raw_mutation = "{}"  # treat as NONE

            mutation_data = self._safe_parse_json(raw_mutation, trajectory.id, pass_name="mutation")
            output_action = str(mutation_data.get("output_action", "NONE")).upper()
            skill_draft = (
                mutation_data.get("skill_draft")
                if isinstance(mutation_data.get("skill_draft"), dict)
                else None
            )
            # patch_target_id: prefer what the Mutation LLM confirmed, fall back
            # to what the Analysis pass identified.
            patch_target_id = mutation_data.get("patch_target_id") or patch_target_from_analysis
            patch_section = mutation_data.get("patch_section")
            patch_content = mutation_data.get("patch_content")

        return EncodeResult(
            trajectory_id=trajectory.id,
            extracted_pattern=extracted_pattern,
            skill_adoption=skill_adoption,
            output_action=output_action,
            skill_draft=skill_draft,
            patch_target_id=patch_target_id,
            patch_section=patch_section,
            patch_content=patch_content,
            branch_signal=branch_signal,
            raw_response=raw_analysis + ("\n---\n" + raw_mutation if raw_mutation else ""),
        )

    def review_and_materialize_draft(self, encode_result: EncodeResult) -> Optional[str]:
        draft = encode_result.skill_draft
        if not draft:
            return None
        draft_markdown = self._render_skill_markdown(draft)
        llm = self._build_llm(self.evo_config.encode.model, max_tokens=600)

        # Build a scoped peer list: only compare against skills in the same
        # branch + group (task_category for task_level, required_tools[0] for
        # action_level).  This mirrors the capacity-control grouping so that
        # the reviewer's novelty check is tight and the capacity ceiling is
        # naturally visible from the peer count.
        branch = draft.get("branch", "general")
        active_statuses = ["draft", "active", "mature"]

        if branch == "task_level":
            cats = draft.get("task_category", [])
            group_key = (cats[0] if cats else "uncategorized").lower()
            peers = [
                s for s in self.store.list_skills(branch="task_level", statuses=active_statuses)
                if (s.task_category[0].lower() if s.task_category else "uncategorized") == group_key
            ]
            capacity = self.evo_config.governance.capacity_task_level_per_category
            group_label = f"task_level / {group_key}"
        elif branch == "action_level":
            tools = draft.get("required_tools", [])
            group_key = (tools[0] if tools else "generic").lower()
            peers = [
                s for s in self.store.list_skills(branch="action_level", statuses=active_statuses)
                if (s.required_tools[0].lower() if s.required_tools else "generic") == group_key
            ]
            capacity = self.evo_config.governance.capacity_action_level_per_tool
            group_label = f"action_level / {group_key}"
        elif branch == "meta_memory":
            peers = self.store.list_skills(branch="meta_memory", statuses=active_statuses)
            capacity = self.evo_config.governance.capacity_meta_memory
            group_label = "meta_memory"
        else:  # general
            peers = self.store.list_skills(branch="general", statuses=active_statuses)
            capacity = self.evo_config.governance.capacity_general
            group_label = "general"

        descriptions = "\n".join(
            f"- {s.id}: {s.description}" for s in peers
        ) or "None"
        review_prompt = build_draft_review_prompt(
            descriptions, draft_markdown,
            current_count=len(peers), capacity=capacity, group_label=group_label,
        )
        try:
            raw = llm.generate(
                system_prompt="You output strict JSON only.",
                user_prompt=review_prompt,
            ).content.strip()
        except (ContentFilterError, RuntimeError) as e:
            logger.warning(
                "Draft review LLM call failed; rejecting draft by default. Error: %s", e
            )
            return None
        try:
            review = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Draft review returned non-JSON; rejecting draft by default.")
            return None
        if not review.get("pass_novelty") or not review.get("pass_quality"):
            logger.info("Skill draft rejected by review: %s", review.get("reason", ""))
            return None
        return self._set_skill_markdown_status(draft_markdown, "active")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_llm(self, model: str, max_tokens: int):
        return build_llm(
            LLMConfig(
                provider=self.provider,
                model=model,
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout_seconds=60,
            )
        )

    def _render_injected_skills(self, skill_ids: List[str]) -> str:
        if not skill_ids:
            return "None"
        blocks: List[str] = []
        for skill_id in skill_ids:
            try:
                skill = self.store.read_skill(skill_id)
                blocks.append(f"## {skill.id} | {skill.name}\n{skill.body}")
            except FileNotFoundError:
                continue
        return "\n\n".join(blocks) or "None"

    def _render_single_skill(self, skill_id: str) -> str:
        """Render the body of a single skill by ID for PATCH context.

        The output intentionally omits a '##' heading so that the caller's
        '## Target Skill to Patch' block remains the sole section header at
        that level, avoiding Markdown heading conflicts.

        Returns an empty string if the skill is not found, so the caller
        can still proceed (the Mutation prompt will show 'None' for the
        target skill body).
        """
        try:
            skill = self.store.read_skill(skill_id)
            return f"**ID**: {skill.id} | **Name**: {skill.name}\n\n{skill.body}"
        except FileNotFoundError:
            logger.warning("Patch target skill '%s' not found in store.", skill_id)
            return ""

    def _format_steps(self, steps: List[Dict[str, Any]], obs_limit: int = 200) -> str:
        """Serialize trajectory steps for the encode prompt.

        Kept fields (in order): step number, reasoning, action, action_input,
        observation (truncated to *obs_limit* chars), answer (final step only),
        error (when present).

        Removed fields: ``llm_output`` (redundant with reasoning/action),
        ``planning`` (rarely populated; reasoning already captures intent),
        ``duration_s`` (irrelevant for pattern extraction).

        All steps are included; per-observation character truncation is handled
        via the ``obs_limit`` parameter (sourced from
        ``context_guard.context_trim_observation_chars``).
        """
        selected = steps if steps else []
        lines: List[str] = []
        for i, step in enumerate(selected):
            step_num = step.get("step", i + 1)
            lines.append(f"[Step {step_num}]")
            if step.get("reasoning"):
                lines.append(f"Reasoning: {step['reasoning']}")
            if step.get("action"):
                lines.append(f"Action: {step['action']}")
            if step.get("action_input") is not None:
                lines.append(f"Action Input: {json.dumps(step.get('action_input'), ensure_ascii=False)}")
            if step.get("observation"):
                obs = step["observation"]
                if obs_limit > 0 and len(obs) > obs_limit:
                    obs = obs[:obs_limit] + "...[truncated]"
                lines.append(f"Observation: {obs}")
            if step.get("answer"):
                lines.append(f"Final Answer: {step['answer']}")
            if step.get("error"):
                lines.append(f"Error: {step['error']}")
            lines.append("")
        return "\n".join(lines).strip() or "No steps available."

    def _safe_parse_json(self, raw: str, trajectory_id: str, pass_name: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, returning an empty dict on failure."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Encode %s pass returned non-JSON for trajectory %s",
                pass_name,
                trajectory_id,
            )
            return {}

    def _parse_skill_adoption(self, data: Dict[str, Any]) -> List[SkillAdoptionRecord]:
        """Parse skill_adoption list safely, guarding against null/non-list values.

        credit_weight is derived from status rather than read from the LLM
        output, so the prompt no longer needs to specify numeric values.
        """
        raw_list = data.get("skill_adoption")
        # Guard: LLM may return null, a dict, or other non-list types.
        if not isinstance(raw_list, list):
            return []
        adoption = []
        for item in raw_list:
            if not isinstance(item, dict) or not item.get("skill_id"):
                continue
            status = str(item.get("status", "IGNORED")).upper()
            credit_weight = self._STATUS_TO_WEIGHT.get(status, 0.2)
            adoption.append(
                SkillAdoptionRecord(
                    skill_id=str(item.get("skill_id")),
                    status=status,
                    credit_weight=credit_weight,
                )
            )
        return adoption

    @staticmethod
    def _set_skill_markdown_status(markdown: str, status: str) -> str:
        import re
        if 'status: "' in markdown:
            return re.sub(r'status:\s*"[^"]+"', f'status: "{status}"', markdown, count=1)
        if "status:" in markdown:
            return re.sub(r"status:\s*[^\n]+", f'status: "{status}"', markdown, count=1)
        return markdown

    def _render_skill_markdown(self, draft: Dict[str, Any]) -> str:
        """Render a skill draft dict into a Markdown file with YAML front matter.

        Front matter fields:
          - ``required_tools``: list of tool function names the skill depends on
            (empty list for pure reasoning skills).
          - ``tool_name``: preserved for ``action_level`` skills as a convenience
            field pointing to the single tool this skill covers.

        Body: a single ``### Core Action`` section.  An optional boundary note
        (e.g., "Not applicable when ...") may be appended inline at the end of
        the Core Action text rather than as a separate section.
        """
        meta = {
            "id": draft.get("id", draft.get("name", "draft_skill")).lower().replace(" ", "_"),
            "name": draft.get("name", "Draft Skill"),
            "description": draft.get("description", ""),
            "version": "1.0",
            "status": draft.get("status", "draft"),
            "branch": draft.get("branch", "general"),
            "task_category": draft.get("task_category", []),
            "source_trajectory_ids": draft.get("source_trajectory_ids", [draft.get("trajectory_id", "")]),
            "applicable_conditions": draft.get("applicable_conditions", ""),
            "required_tools": draft.get("required_tools", []),
            "risk_level": draft.get("risk_level", "low"),
        }
        # Preserve tool_name in front matter for action_level skills (informational).
        if draft.get("tool_name"):
            meta["tool_name"] = draft["tool_name"]

        front_lines = ["---"]
        for key, value in meta.items():
            if isinstance(value, list):
                front_lines.append(f"{key}:")
                for item in value:
                    if item:
                        front_lines.append(f"  - \"{item}\"")
            else:
                front_lines.append(f"{key}: \"{value}\"")
        front_lines.append("---")
        core_action = draft.get("core_action") or draft.get("Core Action") or "- TBD"
        body = f"### Core Action\n{core_action}\n"
        return "\n".join(front_lines) + "\n\n" + body
