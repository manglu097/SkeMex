from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SkillRecord:
    id: str
    name: str
    description: str
    version: str
    status: str
    branch: str
    task_category: List[str] = field(default_factory=list)
    source_trajectory_ids: List[str] = field(default_factory=list)
    applicable_conditions: str = ""  # single concise sentence: when to apply this skill
    required_tools: List[str] = field(default_factory=list)
    risk_level: str = "low"
    created_at: str = ""
    updated_at: str = ""
    body: str = ""
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    id: str
    dataset: str
    query: str
    images: List[str] = field(default_factory=list)
    task_category: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    injected_skill_ids: List[str] = field(default_factory=list)
    prediction: Optional[str] = None
    ground_truth: Any = None
    eval_result: Optional[float] = None
    total_duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    history_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalCandidate:
    skill_id: str
    branch: str
    status: str
    similarity: float
    utility: float
    memory_strength: float
    score: float
    name: str = ""
    description: str = ""
    applicable_conditions: str = ""
    task_category: List[str] = field(default_factory=list)
    body: str = ""


@dataclass
class RetrievalResult:
    task_category: List[str] = field(default_factory=list)
    injected_skill_ids: List[str] = field(default_factory=list)
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    retrieved_skills_block: str = ""


@dataclass
class SkillAdoptionRecord:
    skill_id: str
    status: str
    credit_weight: float


@dataclass
class EncodeResult:
    trajectory_id: str
    extracted_pattern: str = ""
    skill_adoption: List[SkillAdoptionRecord] = field(default_factory=list)
    output_action: str = "NONE"
    skill_draft: Optional[Dict[str, Any]] = None
    patch_target_id: Optional[str] = None
    patch_section: Optional[str] = None
    patch_content: Optional[str] = None
    branch_signal: Optional[Dict[str, Any]] = None  # structured branch hint from Analysis Pass
    raw_response: str = ""


@dataclass
class EvaluationRecord:
    trajectory_id: str
    dataset: str
    reward: float
    raw_eval: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowPerformance:
    window_index: int
    sample_count: int
    average_reward: float
    dataset_breakdown: Dict[str, float] = field(default_factory=dict)
