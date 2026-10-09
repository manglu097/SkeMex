"""Skill retrieval module for MedAgent-Evo.

Architecture (§4 of the design spec):

  1. **LLM Pre-screening Gating** (§4.1) — optional, triggered when any branch
     has more than ``pre_screen.skip_threshold`` available skills.  A lightweight
     LLM (default ``gpt-4.1-mini``) reads only the core metadata of every
     candidate skill and selects at most ``top_k_per_branch`` IDs per branch.
     This collapses the candidate pool from potentially hundreds of skills down
     to ≤ 15 before the expensive embedding step.

  2. **Three-channel scoring** (§4.2) — the surviving candidates are ranked by
     a weighted combination of:

     * **Sim** — cosine similarity clipped to [0, 1] (Fix 1).
     * **U**   — utility score in [0, 1].
     * **S**   — Ebbinghaus memory strength based on window-index decay (Fix 3).

     Mature skills receive a multiplicative bonus of
     ``(1 + prefer_mature_bonus_ratio)`` on their final score (Fix 2), clipped
     to [0, 1].

  3. **Filter & inject** (§4.3) — low-similarity candidates are dropped, the
     top-K are selected, and their bodies are rendered into the system prompt.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from medmem_agent.config import LLMConfig
from medmem_agent.llm import build_llm, ContentFilterError

from .config import EvoConfig
from .embeddings import EmbeddingManager
from .prompts import build_pre_screen_prompt, render_retrieved_skills_block
from .storage import SkillStore
from .types import RetrievalCandidate, RetrievalResult, SkillRecord
from .utils import memory_strength

logger = logging.getLogger(__name__)

# Branches that participate in pre-screening (meta_memory is handled separately
# by ContextGuard and is never part of the retrieval candidate pool).
_RETRIEVAL_BRANCHES = ("general", "task_level", "action_level")


@dataclass
class SkillRetriever:
    store: SkillStore
    embeddings: EmbeddingManager
    evo_config: EvoConfig
    # LLM credentials forwarded from the evolution API config.
    provider: str = "openai_compatible"
    api_base: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        task_category: List[str],
        current_window: int = 0,
        available_tools: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """Run the full two-stage retrieval pipeline and return a RetrievalResult.

        Parameters
        ----------
        query:
            The task query string used for semantic similarity matching.
        task_category:
            List of task-category labels for conditional utility lookup.
        current_window:
            The current global window index (Fix 3).  Used to compute
            window-based Ebbinghaus memory strength for each candidate.

        Stage 1 — LLM Pre-screening Gating (§4.1):
            Triggered when any retrieval branch has > ``skip_threshold`` skills.
            Produces a branch-wise whitelist of at most ``top_k_per_branch`` IDs.

        Stage 2 — Three-channel scoring (§4.2 + §4.3):
            Scores surviving candidates by Sim + U + S, applies the similarity
            threshold, and selects the final top-K for injection using per-branch
            quota to guarantee at least one skill per non-empty branch.

        Parameters
        ----------
        available_tools:
            List of tool function names available to the agent in this session.
            Skills whose ``required_tools`` list contains any tool not in this
            set are excluded before scoring.  Pass ``None`` to skip the check.
        """
        utility_index = self.store.load_utility_index()
        skills = self.store.list_skills(statuses=["active", "mature"])

        # Partition skills by branch for the pre-screening step.
        branch_skills: Dict[str, List[SkillRecord]] = defaultdict(list)
        for skill in skills:
            if skill.branch in _RETRIEVAL_BRANCHES:
                branch_skills[skill.branch].append(skill)

        # ----------------------------------------------------------------
        # task_level category hard-filter with empty-set fallback.
        #
        # task_level skills carry a task_category tag that records which
        # clinical category they belong to.  When the current task has a
        # known category, restrict the task_level candidate pool to skills
        # whose tag overlaps with it.  This prevents cross-category noise
        # from consuming pre-screen slots and improves retrieval precision.
        #
        # Fallback: if the filtered pool is empty (cold-start, mis-
        # classification, or all task_level skills are uncategorised),
        # revert to the full task_level list so the system never silently
        # returns zero task-level candidates.
        # ----------------------------------------------------------------
        if task_category:
            category_set = set(task_category)
            filtered_task_level = [
                s for s in branch_skills["task_level"]
                if category_set.intersection(s.task_category)
            ]
            if filtered_task_level:
                branch_skills["task_level"] = filtered_task_level
                logger.debug(
                    "task_level hard-filter: %d → %d skills (categories: %s)",
                    len(skills),
                    len(filtered_task_level),
                    sorted(category_set),
                )
            else:
                logger.debug(
                    "task_level hard-filter: no category match for %s; "
                    "falling back to full task_level pool (%d skills).",
                    sorted(category_set),
                    len(branch_skills["task_level"]),
                )

        # ----------------------------------------------------------------
        # Stage 1: LLM Pre-screening Gating (§4.1)
        # ----------------------------------------------------------------
        pre_screen_cfg = self.evo_config.retrieval.pre_screen
        allowed_ids: Optional[Set[str]] = self._run_pre_screen(
            query=query,
            branch_skills=branch_skills,
            skip_threshold=pre_screen_cfg.skip_threshold,
            top_k_per_branch=pre_screen_cfg.top_k_per_branch,
        )

        # ----------------------------------------------------------------
        # Stage 2: Three-channel scoring (§4.2)
        # ----------------------------------------------------------------
        query_text = query.strip()
        candidates: List[RetrievalCandidate] = []
        retrieval_cfg = self.evo_config.retrieval

        available_tools_set: Optional[Set[str]] = (
            set(available_tools) if available_tools is not None else None
        )

        for skill in skills:
            # Only score skills in the three retrieval branches; meta_memory
            # skills are never injected into the task prompt.
            if skill.branch not in _RETRIEVAL_BRANCHES:
                continue

            # Skip skills that were filtered out by the pre-screen gate.
            if allowed_ids is not None and skill.id not in allowed_ids:
                continue

            # required_tools check: skip skills that need tools not available
            # in the current session.  Skills with an empty required_tools list
            # are always eligible (no tool dependency).
            if available_tools_set is not None and skill.required_tools:
                if not all(t in available_tools_set for t in skill.required_tools):
                    logger.debug(
                        "Skill '%s' skipped: required_tools %s not all in available tools.",
                        skill.id,
                        skill.required_tools,
                    )
                    continue

            candidate_text = self._skill_embedding_text(skill)
            # Fix 1: cosine_similarity is already clipped to [0, 1] in utils.py.
            similarity = self.embeddings.similarity(query_text, candidate_text)
            if similarity < retrieval_cfg.min_similarity_threshold:
                continue

            utility = self._utility_for_skill(skill, task_category, utility_index)

            # Fix 3: memory strength is now window-based.
            util_row = utility_index.get(skill.id, {})
            last_adopted_window = int(util_row.get("last_adopted_window", -1))
            tau_windows = float(
                util_row.get("memory_tau_windows", self.evo_config.utility.memory_tau_base_windows)
            )
            mem_strength = memory_strength(last_adopted_window, current_window, tau_windows)

            raw_score = (
                retrieval_cfg.lambda_similarity * similarity
                + retrieval_cfg.lambda_utility * utility
                + retrieval_cfg.lambda_memory * mem_strength
            )

            # Fix 2: mature bonus is multiplicative (percentage-based), clipped to [0, 1].
            if skill.status == "mature":
                raw_score = min(1.0, raw_score * (1.0 + retrieval_cfg.prefer_mature_bonus_ratio))

            candidates.append(
                RetrievalCandidate(
                    skill_id=skill.id,
                    branch=skill.branch,
                    status=skill.status,
                    similarity=similarity,
                    utility=utility,
                    memory_strength=mem_strength,
                    score=raw_score,
                    name=skill.name,
                    description=skill.description,
                    applicable_conditions=skill.applicable_conditions,
                    task_category=skill.task_category,
                    body=skill.body,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        top = self._select_per_branch_quota(candidates, retrieval_cfg.top_k)
        block = render_retrieved_skills_block(
            [
                {
                    "skill_id": item.skill_id,
                    "name": item.name,
                    "applicable_conditions": item.applicable_conditions,
                    "task_category": item.task_category,
                    "utility": item.utility,
                    "body": item.body,
                }
                for item in top
            ]
        )
        return RetrievalResult(
            task_category=task_category,
            injected_skill_ids=[item.skill_id for item in top],
            candidates=top,
            retrieved_skills_block=block,
        )

    # ------------------------------------------------------------------
    # Pre-screening gate (§4.1)
    # ------------------------------------------------------------------

    def _run_pre_screen(
        self,
        *,
        query: str,
        branch_skills: Dict[str, List[SkillRecord]],
        skip_threshold: int,
        top_k_per_branch: int,
    ) -> Optional[Set[str]]:
        """Run the LLM pre-screening gate and return an allowed-ID whitelist.

        Returns ``None`` when the gate is skipped (all branches are small
        enough), meaning the caller should not filter by ID at all.  Returns a
        ``set`` of skill IDs when the gate fires; only those IDs survive into
        Stage 2.

        Skip condition (§4.1):
            If **all** branches have ≤ ``skip_threshold`` skills, skip the gate.
        Trigger condition (§4.1):
            If **any** branch has > ``skip_threshold`` skills, run the gate.
        """
        pre_screen_cfg = self.evo_config.retrieval.pre_screen
        if not pre_screen_cfg.enabled:
            logger.debug("Pre-screen gating is disabled by config; skipping.")
            return None

        branch_counts = {b: len(s) for b, s in branch_skills.items()}
        total_skills = sum(branch_counts.values())
        if total_skills == 0:
            logger.debug("No retrieval-eligible skills found; skipping pre-screen.")
            return None

        any_branch_exceeds = any(cnt > skip_threshold for cnt in branch_counts.values())
        if not any_branch_exceeds:
            logger.debug(
                "All branches have ≤ %d skills (counts: %s); skipping pre-screen gate.",
                skip_threshold,
                branch_counts,
            )
            return None

        logger.debug(
            "Pre-screen gate triggered (branch counts: %s, skip_threshold=%d).",
            branch_counts,
            skip_threshold,
        )

        # Build the metadata payload for each branch.
        # Both `description` (semantic intent) and `applicable_conditions`
        # (concise when-to-apply sentence) are sent to the pre-screen LLM
        # to give it the clearest possible signal for relevance matching.
        skills_by_branch_meta: Dict[str, List[dict]] = {}
        for branch in _RETRIEVAL_BRANCHES:
            skills_by_branch_meta[branch] = [
                {
                    "id": s.id,
                    "description": s.description,
                    "applicable_conditions": s.applicable_conditions,
                }
                for s in branch_skills.get(branch, [])
            ]

        prompt = build_pre_screen_prompt(
            query=query,
            skills_by_branch=skills_by_branch_meta,
            top_k=top_k_per_branch,
        )

        # Call the lightweight LLM.
        llm = build_llm(
            LLMConfig(
                provider=self.provider,
                model=pre_screen_cfg.model,
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=pre_screen_cfg.temperature,
                max_tokens=pre_screen_cfg.max_tokens,
                timeout_seconds=pre_screen_cfg.timeout_seconds,
            )
        )
        try:
            raw = llm.generate(
                system_prompt="You output strict JSON only.",
                user_prompt=prompt,
            ).content.strip()
        except ContentFilterError as e:
            logger.warning(
                "Pre-screen LLM call blocked by content filter; "
                "skipping pre-screen gate and returning all candidates. Error: %s", e
            )
            return None
        except RuntimeError as e:
            logger.warning(
                "Pre-screen LLM call failed after all retries; "
                "skipping pre-screen gate and returning all candidates. Error: %s", e
            )
            return None

        allowed_ids = self._parse_pre_screen_response(
            raw=raw,
            branch_skills=branch_skills,
            top_k_per_branch=top_k_per_branch,
        )
        logger.debug(
            "Pre-screen gate selected %d skill IDs: %s",
            len(allowed_ids),
            sorted(allowed_ids),
        )
        return allowed_ids

    @staticmethod
    def _parse_pre_screen_response(
        raw: str,
        branch_skills: Dict[str, List[SkillRecord]],
        top_k_per_branch: int,
    ) -> Set[str]:
        """Parse the LLM pre-screen JSON response into a validated ID set.

        Handles malformed JSON gracefully: on any parse error the gate falls
        back to returning **all** available IDs (i.e., no filtering), so the
        system degrades gracefully rather than crashing.

        Validation rules applied to the LLM output:
        - Only IDs that actually exist in the branch's skill list are kept.
        - Each branch is capped at ``top_k_per_branch`` IDs.
        """
        valid_ids_per_branch: Dict[str, Set[str]] = {
            branch: {s.id for s in skills}
            for branch, skills in branch_skills.items()
        }
        all_valid_ids: Set[str] = set().union(*valid_ids_per_branch.values())

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Pre-screen LLM returned non-JSON output; falling back to full candidate set. "
                "Raw response (first 200 chars): %s",
                raw[:200],
            )
            return all_valid_ids

        if not isinstance(parsed, dict):
            logger.warning(
                "Pre-screen LLM returned unexpected type %s; falling back to full candidate set.",
                type(parsed).__name__,
            )
            return all_valid_ids

        allowed: Set[str] = set()
        for branch in _RETRIEVAL_BRANCHES:
            branch_selection = parsed.get(branch, [])
            branch_candidates = branch_skills.get(branch, [])
            valid_branch_ids = valid_ids_per_branch.get(branch, set())
            fallback_ids = [s.id for s in branch_candidates[:top_k_per_branch]]
            if not isinstance(branch_selection, list):
                logger.warning(
                    "Pre-screen response for branch '%s' is not a list (%s); falling back to branch-local candidates.",
                    branch,
                    type(branch_selection).__name__,
                )
                allowed.update(fallback_ids)
                continue

            selected_for_branch: List[str] = []
            for sid in branch_selection:
                if not isinstance(sid, str):
                    continue
                if sid not in valid_branch_ids:
                    logger.debug(
                        "Pre-screen selected unknown skill ID '%s' for branch '%s'; ignoring.",
                        sid,
                        branch,
                    )
                    continue
                selected_for_branch.append(sid)
                if len(selected_for_branch) >= top_k_per_branch:
                    break

            if branch_candidates and not selected_for_branch:
                logger.warning(
                    "Pre-screen returned an empty valid selection for non-empty branch '%s'; falling back to branch-local candidates.",
                    branch,
                )
                allowed.update(fallback_ids)
                continue

            allowed.update(selected_for_branch)

        if not allowed:
            logger.warning(
                "Pre-screen gate returned an empty selection; falling back to full candidate set."
            )
            return all_valid_ids

        return allowed

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _select_per_branch_quota(
        candidates: List[RetrievalCandidate],
        top_k: int,
    ) -> List[RetrievalCandidate]:
        """Select top-K candidates with per-branch quota guarantee.

        ``top_k`` must be a multiple of the number of retrieval branches (3).
        Each branch is guaranteed at least ``top_k // 3`` slots.  Remaining
        slots (if any branch has fewer candidates than its quota) are filled
        by the globally highest-scoring remaining candidates.

        If ``top_k`` is not a multiple of 3, the quota per branch is
        ``top_k // 3`` and the remainder slots are filled globally.
        """
        quota_per_branch = max(1, top_k // len(_RETRIEVAL_BRANCHES))
        selected: List[RetrievalCandidate] = []
        used_ids: Set[str] = set()

        # First pass: fill per-branch quota (candidates already sorted desc).
        branch_filled: Dict[str, int] = {b: 0 for b in _RETRIEVAL_BRANCHES}
        for c in candidates:
            if c.branch in branch_filled and branch_filled[c.branch] < quota_per_branch:
                selected.append(c)
                used_ids.add(c.skill_id)
                branch_filled[c.branch] += 1

        # Second pass: fill remaining slots with globally best unused candidates.
        remaining = top_k - len(selected)
        if remaining > 0:
            for c in candidates:
                if remaining <= 0:
                    break
                if c.skill_id not in used_ids:
                    selected.append(c)
                    used_ids.add(c.skill_id)
                    remaining -= 1

        # Re-sort the final selection by score descending for consistent output.
        selected.sort(key=lambda item: item.score, reverse=True)
        return selected

    @staticmethod
    def _skill_embedding_text(skill: SkillRecord) -> str:
        """Build the text used for embedding-based similarity.

        Only the ``description`` field is used (the concise "when to apply"
        sentence).  ``required_tools`` and other metadata are intentionally
        excluded to keep the embedding focused on semantic intent.
        """
        return skill.description.strip()

    def _utility_for_skill(
        self,
        skill: SkillRecord,
        task_category: List[str],
        utility_index: dict,
    ) -> float:
        row = utility_index.get(skill.id, {})
        if skill.branch == "general":
            conditional = row.get("conditional_utility", {})
            for category in task_category:
                if category in conditional:
                    return float(conditional[category])
            return float(row.get("global_utility", 0.5))
        return float(row.get("utility_score", 0.5))
