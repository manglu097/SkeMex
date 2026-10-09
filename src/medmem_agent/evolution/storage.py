from __future__ import annotations

import fcntl
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import EvoConfig
from .types import SkillRecord
from .utils import (
    atomic_write_json,
    bump_minor_version,
    parse_front_matter,
    read_json,
    render_front_matter,
    replace_skill_section,
    utc_now_iso,
)

logger = logging.getLogger(__name__)


@dataclass
class SkillStore:
    """Filesystem-backed skill library.

    All paths are derived from the structured directory layout defined in
    ``StorageConfig``.  Flat per-file path overrides are no longer supported.

    Directory layout under ``experiment_root``::

        <experiment_root>/
          <skills_dirname>/
            general/
            task_level/
            action_level/
            meta_memory/
          <index_dirname>/
            utility_index.json
            categories.json
          <embeddings_dirname>/
            embeddings.json
          <traces_dirname>/

    The global query-embedding cache lives at::

        <root_dir>/global_query_embeddings.json
    """

    repo_root: str
    evo_config: EvoConfig
    read_only: bool = False

    def __post_init__(self) -> None:
        self.repo_root = str(Path(self.repo_root).resolve())
        storage = self.evo_config.storage

        self.experiment_root = Path(self.repo_root) / storage.experiment_root
        self.experiment_root.mkdir(parents=True, exist_ok=True)

        # Structured directory paths — derived purely from *_dirname fields.
        self.skills_root = self.experiment_root / storage.skills_dirname
        self.index_dir = self.experiment_root / storage.index_dirname
        self.embeddings_dir = self.experiment_root / storage.embeddings_dirname
        self.traces_dir = self.experiment_root / storage.traces_dirname
        self.meta_memory_dir = self.skills_root / storage.meta_memory_dirname

        # Concrete file paths.
        self.utility_index_path = self.index_dir / "utility_index.json"
        self.categories_path = self.index_dir / "categories.json"
        self.id_counters_path = self.index_dir / "id_counters.json"
        self.embeddings_path = self.embeddings_dir / "embeddings.json"
        self.global_query_cache_path = (
            Path(self.repo_root) / storage.root_dir / "global_query_embeddings.json"
        )
        self.category_cache_path = (
            Path(self.repo_root) / storage.root_dir / "global_category_cache.json"
        )
        self.lock_path = self.experiment_root / ".skillstore.lock"

        # Ensure all directories exist.
        for directory in (
            self.skills_root,
            self.index_dir,
            self.embeddings_dir,
            self.traces_dir,
            self.meta_memory_dir,
            self.lock_path.parent,
            self.global_query_cache_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self):
        self.lock_path.touch(exist_ok=True)
        with open(self.lock_path, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def ensure_initialized(self) -> None:
        assets_root = Path(__file__).resolve().parents[1] / "evolution_assets"
        if not self.categories_path.exists():
            atomic_write_json(
                self.categories_path,
                read_json(assets_root / "categories_seed.json", default={"active": [], "pending": []}),
            )
        if not self.utility_index_path.exists():
            atomic_write_json(
                self.utility_index_path,
                read_json(assets_root / "utility_index_seed.json", default={}),
            )
        if not self.id_counters_path.exists():
            atomic_write_json(self.id_counters_path, {})
        if not self.embeddings_path.exists():
            atomic_write_json(self.embeddings_path, {})
        if not self.global_query_cache_path.exists():
            atomic_write_json(self.global_query_cache_path, {})
        if not self.category_cache_path.exists():
            atomic_write_json(self.category_cache_path, {})
        self._ensure_meta_memory_skills(assets_root / "meta_memory")

    def _ensure_meta_memory_skills(self, assets_dir: Path) -> None:
        target_dir = self.meta_memory_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for src_path in sorted(assets_dir.glob("*.md")):
            dst_path = target_dir / src_path.name
            if not dst_path.exists() and not self.read_only:
                dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        utility_index = self.load_utility_index()
        changed = False
        for meta_skill in self.list_skills(branch="meta_memory"):
            if meta_skill.id not in utility_index:
                utility_index[meta_skill.id] = {"branch": "meta_memory", "status": meta_skill.status}
                changed = True
        if changed and not self.read_only:
            self.save_utility_index(utility_index)

    def list_skills(
        self,
        *,
        branch: Optional[str] = None,
        statuses: Optional[Iterable[str]] = None,
    ) -> List[SkillRecord]:
        skills: List[SkillRecord] = []
        allowed_statuses = set(statuses) if statuses else None
        for path in sorted(self.skills_root.rglob("*.md")):
            record = self.read_skill(path)
            if branch and record.branch != branch:
                continue
            if allowed_statuses and record.status not in allowed_statuses:
                continue
            skills.append(record)
        return skills

    def read_skill(self, path_or_id: str | Path) -> SkillRecord:
        path = self.resolve_skill_path(path_or_id)
        text = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(text)
        return SkillRecord(
            id=meta.get("id", path.stem),
            name=meta.get("name", path.stem),
            description=meta.get("description", ""),
            version=meta.get("version", "1.0"),
            status=meta.get("status", "draft"),
            branch=meta.get("branch", self._infer_branch_from_path(path)),
            task_category=list(meta.get("task_category", [])),
            source_trajectory_ids=list(meta.get("source_trajectory_ids", [])),
            applicable_conditions=str(meta.get("applicable_conditions", "")),
            required_tools=list(meta.get("required_tools", [])),
            risk_level=meta.get("risk_level", "low"),
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
            body=body,
            path=str(path),
        )

    def resolve_skill_path(self, path_or_id: str | Path) -> Path:
        candidate = Path(path_or_id)
        if candidate.exists():
            return candidate
        for path in self.skills_root.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            meta, _ = parse_front_matter(text)
            if meta.get("id") == str(path_or_id):
                return path
        raise FileNotFoundError(f"Skill not found: {path_or_id}")

    def create_skill_from_markdown(
        self,
        markdown_text: str,
        branch: str,
        relative_dir: Optional[str] = None,
    ) -> SkillRecord:
        if self.read_only:
            raise RuntimeError("SkillStore is read-only in frozen mode.")
        with self.lock():
            meta, body = parse_front_matter(markdown_text)
            now = utc_now_iso()
            meta.setdefault("created_at", now)
            meta["updated_at"] = now
            meta["id"] = self.allocate_skill_id(branch)
            target_dir = self.skills_root / branch
            if relative_dir:
                target_dir = target_dir / relative_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = self._build_skill_filename(meta["id"], meta.get("name") or "skill")
            path = target_dir / f"{filename}.md"
            path.write_text(
                render_front_matter(meta) + "\n\n" + body.strip() + "\n",
                encoding="utf-8",
            )
        return self.read_skill(path)

    def patch_skill_section(self, skill_id: str, section_name: str, new_content: str) -> SkillRecord:
        if self.read_only:
            raise RuntimeError("SkillStore is read-only in frozen mode.")
        path = self.resolve_skill_path(skill_id)
        text = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(text)
        meta["version"] = bump_minor_version(str(meta.get("version", "1.0")))
        meta["updated_at"] = utc_now_iso()
        updated_body = replace_skill_section(body, section_name, new_content)
        path.write_text(
            render_front_matter(meta) + "\n\n" + updated_body.strip() + "\n",
            encoding="utf-8",
        )
        return self.read_skill(path)

    def load_utility_index(self) -> Dict[str, dict]:
        return read_json(self.utility_index_path, default={})

    def save_utility_index(self, data: Dict[str, dict]) -> None:
        if self.read_only:
            raise RuntimeError("SkillStore is read-only in frozen mode.")
        atomic_write_json(self.utility_index_path, data)

    def load_id_counters(self) -> Dict[str, int]:
        raw = read_json(self.id_counters_path, default={})
        if not isinstance(raw, dict):
            return {}
        counters: Dict[str, int] = {}
        for key, value in raw.items():
            try:
                counters[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return counters

    def save_id_counters(self, data: Dict[str, int]) -> None:
        if self.read_only:
            raise RuntimeError("SkillStore is read-only in frozen mode.")
        atomic_write_json(self.id_counters_path, data)

    def allocate_skill_id(self, branch: str) -> str:
        prefix = self._branch_prefix(branch)
        counters = self.load_id_counters()
        current = max(counters.get(branch, 0), self._max_existing_counter(branch, prefix))
        next_counter = current + 1
        counters[branch] = next_counter
        self.save_id_counters(counters)
        return f"{prefix}{next_counter:06d}"

    def load_categories(self) -> Dict[str, list]:
        return read_json(self.categories_path, default={"active": [], "pending": []})

    def save_categories(self, data: Dict[str, list]) -> None:
        if self.read_only:
            raise RuntimeError("SkillStore is read-only in frozen mode.")
        atomic_write_json(self.categories_path, data)

    # ------------------------------------------------------------------
    # Category classification cache (classifier_mode="cache")
    # ------------------------------------------------------------------

    def load_category_cache(self) -> Dict[str, list]:
        """Load the query-hash → task_category cache from disk.

        Returns a dict mapping SHA-256 query hashes (hex strings) to
        lists of category label strings.
        """
        return read_json(self.category_cache_path, default={})

    def save_category_cache(self, data: Dict[str, list]) -> None:
        """Persist the category cache to disk (no-op in read-only mode)."""
        if not self.read_only:
            atomic_write_json(self.category_cache_path, data)

    def update_pending_category(self, label: str, description: str = "") -> Dict[str, list]:
        if self.read_only:
            return self.load_categories()
        data = self.load_categories()
        active_labels = {item["label"] for item in data.get("active", [])}
        if label in active_labels:
            return data
        for item in data.get("pending", []):
            if item.get("label") == label:
                item["count"] = int(item.get("count", 0)) + 1
                if item["count"] >= self.evo_config.categories.new_category_promote_threshold:
                    data.setdefault("active", []).append(
                        {"label": label, "description": description or item.get("description", "")}
                    )
                    data["pending"] = [p for p in data.get("pending", []) if p.get("label") != label]
                self.save_categories(data)
                return data
        data.setdefault("pending", []).append(
            {"label": label, "count": 1, "first_seen": utc_now_iso(), "description": description}
        )
        self.save_categories(data)
        return data

    def register_skill_utility(self, skill: SkillRecord, initial_utility: float = 0.5) -> None:
        if self.read_only:
            return
        index = self.load_utility_index()
        if skill.id in index:
            return
        if skill.branch == "general":
            index[skill.id] = {
                "branch": skill.branch,
                "global_utility": initial_utility,
                "conditional_utility": {},
                "usage_count": 0,
                "adoption_count": 0,
                "success_when_adopted": 0,
                "failure_when_adopted": 0,
                "last_used": "",
                # Fix 3: last_success is now a window index (-1 = never succeeded).
                "last_success_window": -1,
                "memory_tau_windows": self.evo_config.utility.memory_tau_base_windows,
                "status": skill.status,
            }
        elif skill.branch == "meta_memory":
            index[skill.id] = {"branch": skill.branch, "status": skill.status}
        else:
            index[skill.id] = {
                "branch": skill.branch,
                "utility_score": initial_utility,
                "usage_count": 0,
                "adoption_count": 0,
                "success_when_adopted": 0,
                "failure_when_adopted": 0,
                "last_used": "",
                # Fix 3: last_success is now a window index (-1 = never succeeded).
                "last_success_window": -1,
                "memory_tau_windows": self.evo_config.utility.memory_tau_base_windows,
                "status": skill.status,
            }
        self.save_utility_index(index)

    @staticmethod
    def _infer_branch_from_path(path: Path) -> str:
        parts = path.parts
        if "meta_memory" in parts:
            return "meta_memory"
        if "general" in parts:
            return "general"
        if "task_level" in parts:
            return "task_level"
        if "action_level" in parts:
            return "action_level"
        return "general"

    def _max_existing_counter(self, branch: str, prefix: str) -> int:
        max_counter = 0
        for skill in self.list_skills(branch=branch):
            skill_id = str(skill.id)
            if not skill_id.startswith(prefix):
                continue
            suffix = skill_id[len(prefix):]
            if suffix.isdigit():
                max_counter = max(max_counter, int(suffix))
        return max_counter

    @staticmethod
    def _branch_prefix(branch: str) -> str:
        mapping = {
            "general": "g",
            "task_level": "t",
            "action_level": "a",
            "meta_memory": "m",
        }
        return mapping.get(branch, "s")

    @classmethod
    def _build_skill_filename(cls, skill_id: str, name: str) -> str:
        slug = cls._safe_filename(name)
        return f"{skill_id}__{slug}" if slug else skill_id

    @staticmethod
    def _safe_filename(name: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
        normalized = "".join(ch.lower() if ch.lower() in allowed else "_" for ch in name)
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized.strip("_") or "skill"
