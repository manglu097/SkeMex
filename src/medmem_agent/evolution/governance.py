from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from medmem_agent.config import LLMConfig
from medmem_agent.llm import build_llm

from .config import EvoConfig
from .embeddings import EmbeddingManager
from .prompts import build_merge_prompt
from .storage import SkillStore
from .types import SkillRecord


@dataclass
class SkillGovernance:
    store: SkillStore
    embeddings: EmbeddingManager
    evo_config: EvoConfig
    provider: str = "openai_compatible"
    api_base: str = "https://api.openai.com/v1"
    api_key: str | None = None

    def apply_encode_result(self, encode_result) -> Optional[SkillRecord]:
        if self.store.read_only:
            return None
        created: Optional[SkillRecord] = None
        if encode_result.output_action == "CREATE" and encode_result.skill_draft:
            from .encode import TrajectoryEncoder
            encoder = TrajectoryEncoder(
                store=self.store,
                evo_config=self.evo_config,
                provider=self.provider,
                api_base=self.api_base,
                api_key=self.api_key,
            )
            reviewed = encoder.review_and_materialize_draft(encode_result)
            if reviewed:
                branch = encode_result.skill_draft.get("branch", "general")
                relative_dir = None
                if branch == "task_level":
                    cats = encode_result.skill_draft.get("task_category", [])
                    relative_dir = (cats[0] if cats else "uncategorized").lower()
                elif branch == "action_level":
                    tools = encode_result.skill_draft.get("required_tools", [])
                    relative_dir = (tools[0] if tools else "generic").lower()
                created = self.store.create_skill_from_markdown(reviewed, branch=branch, relative_dir=relative_dir)
                self.store.register_skill_utility(created, initial_utility=self.evo_config.encode.draft_initial_utility)
        elif encode_result.output_action == "PATCH" and encode_result.patch_target_id and encode_result.patch_section and encode_result.patch_content:
            self.store.patch_skill_section(
                encode_result.patch_target_id,
                encode_result.patch_section,
                encode_result.patch_content,
            )
        return created

    def manage(self) -> Dict[str, List[str]]:
        if self.store.read_only:
            return {"merged": [], "deprecated": [], "matured": [], "archived": []}
        merged = self._merge_similar_drafts()
        deprecated = self._deprecate_low_utility_skills()
        matured = self._promote_mature_skills()
        archived = self._capacity_control()
        return {
            "merged": merged,
            "deprecated": deprecated,
            "matured": matured,
            "archived": archived,
        }

    def _merge_similar_drafts(self) -> List[str]:
        drafts = self.store.list_skills(statuses=["draft", "active", "mature"])
        merged_ids: List[str] = []
        for idx, left in enumerate(drafts):
            for right in drafts[idx + 1:]:
                if left.branch != right.branch or left.id == right.id:
                    continue
                sim = self.embeddings.similarity(self._merge_text(left), self._merge_text(right))
                if sim < self.evo_config.governance.merge_similarity_threshold:
                    continue
                dominant, secondary = self._pick_dominant(left, right)
                merged_body = self._merge_bodies(dominant.body, secondary.body)
                self.store.patch_skill_section(dominant.id, "Core Action", self._extract_section_for_patch(merged_body, "Core Action"))
                self._mark_status(secondary.id, "deprecated")
                merged_ids.append(f"{dominant.id}<={secondary.id}")
        return merged_ids

    def _deprecate_low_utility_skills(self) -> List[str]:
        utility = self.store.load_utility_index()
        deprecated: List[str] = []
        for skill in self.store.list_skills(statuses=["draft", "active", "mature"]):
            row = utility.get(skill.id, {})
            if skill.branch == "meta_memory":
                continue
            value = float(row.get("global_utility", row.get("utility_score", 0.5)))
            if value < self.evo_config.governance.deprecate_utility_threshold:
                self._mark_status(skill.id, "deprecated")
                row["status"] = "deprecated"
                deprecated.append(skill.id)
        if deprecated:
            self.store.save_utility_index(utility)
        return deprecated

    def _promote_mature_skills(self) -> List[str]:
        utility = self.store.load_utility_index()
        matured: List[str] = []
        for skill in self.store.list_skills(statuses=["active", "draft"]):
            row = utility.get(skill.id, {})
            if skill.branch == "meta_memory":
                continue
            value = float(row.get("global_utility", row.get("utility_score", 0.5)))
            usage = int(row.get("usage_count", 0))
            if value >= self.evo_config.governance.mature_utility_threshold and usage >= self.evo_config.governance.mature_usage_threshold:
                self._mark_status(skill.id, "mature")
                row["status"] = "mature"
                matured.append(skill.id)
        if matured:
            self.store.save_utility_index(utility)
        return matured

    def _capacity_control(self) -> List[str]:
        archived: List[str] = []
        archived += self._archive_branch_overflow("general", self.evo_config.governance.capacity_general)
        archived += self._archive_task_level_overflow()
        archived += self._archive_action_level_overflow()
        archived += self._archive_branch_overflow("meta_memory", self.evo_config.governance.capacity_meta_memory)
        return archived

    def _archive_task_level_overflow(self) -> List[str]:
        archived: List[str] = []
        groups: Dict[str, List[SkillRecord]] = {}
        for skill in self.store.list_skills(branch="task_level", statuses=["draft", "active", "mature"]):
            key = (skill.task_category[0] if skill.task_category else "uncategorized")
            groups.setdefault(key, []).append(skill)
        for _, skills in groups.items():
            archived.extend(self._archive_over_capacity(skills, self.evo_config.governance.capacity_task_level_per_category))
        return archived

    def _archive_action_level_overflow(self) -> List[str]:
        archived: List[str] = []
        groups: Dict[str, List[SkillRecord]] = {}
        for skill in self.store.list_skills(branch="action_level", statuses=["draft", "active", "mature"]):
            # Group by the tool subdirectory (mirrors the on-disk layout where
            # each tool gets its own folder under skills/action_level/<tool>/).
            # Falls back to required_tools[0] for skills created before the
            # directory-per-tool convention was introduced.
            if "/action_level/" in skill.path:
                parts = skill.path.split("/action_level/")[-1].split("/")
                tool_key = parts[0] if len(parts) > 1 else (skill.required_tools[0] if skill.required_tools else "generic")
            else:
                tool_key = skill.required_tools[0] if skill.required_tools else "generic"
            groups.setdefault(tool_key, []).append(skill)
        for _, skills in groups.items():
            archived.extend(self._archive_over_capacity(skills, self.evo_config.governance.capacity_action_level_per_tool))
        return archived

    def _archive_branch_overflow(self, branch: str, capacity: int) -> List[str]:
        skills = self.store.list_skills(branch=branch, statuses=["draft", "active", "mature"])
        return self._archive_over_capacity(skills, capacity)

    def _archive_over_capacity(self, skills: List[SkillRecord], capacity: int) -> List[str]:
        if len(skills) <= capacity:
            return []
        utility = self.store.load_utility_index()
        ranked = sorted(
            skills,
            key=lambda skill: (
                1 if skill.status == "mature" else 0,
                float(utility.get(skill.id, {}).get("global_utility", utility.get(skill.id, {}).get("utility_score", 0.5))),
                int(utility.get(skill.id, {}).get("usage_count", 0)),
            ),
            reverse=True,
        )
        archived: List[str] = []
        for skill in ranked[capacity:]:
            self._mark_status(skill.id, "archived")
            row = utility.get(skill.id)
            if row is not None:
                row["status"] = "archived"
            archived.append(skill.id)
        if archived:
            self.store.save_utility_index(utility)
        return archived

    def _mark_status(self, skill_id: str, new_status: str) -> None:
        """Update the status field in the skill's markdown front matter.

        Uses the SkillStore file lock to avoid race conditions and updates
        ``updated_at`` via the store's own path resolution.
        """
        import re
        skill = self.store.read_skill(skill_id)
        path = skill.path
        with self.store.lock():
            text = open(path, "r", encoding="utf-8").read()
            if 'status: "' in text:
                text = re.sub(r'status:\s*"[^"]+"', f'status: "{new_status}"', text, count=1)
            elif "status:" in text:
                text = re.sub(r"status:\s*[^\n]+", f'status: "{new_status}"', text, count=1)
            else:
                # No status field found — insert after the opening '---'
                text = text.replace("---\n", f'---\nstatus: "{new_status}"\n', 1)
            # Also bump updated_at so version tracking stays accurate.
            from .utils import utc_now_iso
            now = utc_now_iso()
            if 'updated_at: "' in text:
                text = re.sub(r'updated_at:\s*"[^"]*"', f'updated_at: "{now}"', text, count=1)
            open(path, "w", encoding="utf-8").write(text)

    def _merge_bodies(self, dominant_body: str, secondary_body: str) -> str:
        llm = build_llm(
            LLMConfig(
                provider=self.provider,
                model=self.evo_config.encode.model,
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=1000,
                timeout_seconds=60,
            )
        )
        return llm.generate(
            system_prompt="Return only the merged markdown body.",
            user_prompt=build_merge_prompt(dominant_body, secondary_body),
        ).content.strip()

    @staticmethod
    def _pick_dominant(left: SkillRecord, right: SkillRecord) -> tuple[SkillRecord, SkillRecord]:
        rank = {"mature": 3, "active": 2, "draft": 1, "deprecated": 0, "archived": -1}
        if rank.get(left.status, 0) > rank.get(right.status, 0):
            return left, right
        if rank.get(right.status, 0) > rank.get(left.status, 0):
            return right, left
        return (left, right) if left.created_at <= right.created_at else (right, left)

    @staticmethod
    def _merge_text(skill: SkillRecord) -> str:
        return f"{skill.description}\n{skill.body}".strip()

    @staticmethod
    def _extract_section_for_patch(body: str, section_name: str) -> str:
        import re
        pattern = rf"^###\s+{re.escape(section_name)}\s*$([\s\S]*?)(?=^###\s+|\Z)"
        match = re.search(pattern, body, flags=re.MULTILINE)
        if not match:
            return body.strip()
        return match.group(1).strip()
