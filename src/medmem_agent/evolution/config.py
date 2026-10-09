from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    if re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", value):
        return os.getenv(value[1:], "")
    return value


@dataclass
class APIConfig:
    provider: str = "openai_compatible"
    api_base: str = ""
    api_key: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIConfig":
        return cls(
            provider=str(data.get("provider", "openai_compatible")),
            api_base=str(_resolve_value(data.get("api_base", "")) or ""),
            api_key=str(_resolve_value(data.get("api_key", "")) or ""),
        )


def _default_llm_model() -> str:
    """Return the model name from configs/eval/default.json, falling back to
    'gpt-4.1-mini' only when the file is absent or unreadable.  This ensures
    that config-file values always take priority over hard-coded literals."""
    _cfg = Path(__file__).resolve().parents[3] / "configs" / "eval" / "default.json"
    if _cfg.exists():
        try:
            return json.loads(_cfg.read_text(encoding="utf-8")).get("model", "gpt-4.1-mini")
        except Exception:
            pass
    return "gpt-4.1-mini"


@dataclass
class PreScreenConfig:
    """Configuration for the LLM Pre-screening Gating step (§4.1).

    The pre-screen gate runs a lightweight LLM pass over skill metadata before
    the three-channel scoring stage.  It is triggered only when at least one
    branch has more than ``skip_threshold`` available skills, reducing the
    candidate set to at most ``top_k_per_branch`` per branch.
    """

    enabled: bool = True
    model: str = field(default_factory=_default_llm_model)
    skip_threshold: int = 5
    top_k_per_branch: int = 5
    timeout_seconds: int = 30
    max_tokens: int = 512
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreScreenConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            model=data.get("model") or _default_llm_model(),
            skip_threshold=int(data.get("skip_threshold", 5)),
            top_k_per_branch=int(data.get("top_k_per_branch", 5)),
            timeout_seconds=int(data.get("timeout_seconds", 30)),
            max_tokens=int(data.get("max_tokens", 512)),
            temperature=float(data.get("temperature", 0.0)),
        )


@dataclass
class RetrievalConfig:
    """Hyperparameters for the two-stage skill retrieval pipeline.

    prefer_mature_bonus_ratio : float
        Relative bonus applied to the three-channel score of *mature* skills.
        The final score is multiplied by ``(1 + ratio)`` and then clipped to
        ``[0, 1]``.  Default: 0.10 (10 %).
    """

    top_k: int = 3
    lambda_similarity: float = 0.5
    lambda_utility: float = 0.35
    lambda_memory: float = 0.15
    min_similarity_threshold: float = 0.2
    prefer_mature_bonus_ratio: float = 0.10
    embedding_model: str = "text-embedding-3-small"
    pre_screen: PreScreenConfig = field(default_factory=PreScreenConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalConfig":
        return cls(
            top_k=int(data.get("top_k", 3)),
            lambda_similarity=float(data.get("lambda_similarity", 0.5)),
            lambda_utility=float(data.get("lambda_utility", 0.35)),
            lambda_memory=float(data.get("lambda_memory", 0.15)),
            min_similarity_threshold=float(data.get("min_similarity_threshold", 0.2)),
            prefer_mature_bonus_ratio=float(data.get("prefer_mature_bonus_ratio", 0.10)),
            embedding_model=data.get("embedding_model", "text-embedding-3-small"),
            pre_screen=PreScreenConfig.from_dict(data.get("pre_screen", {})),
        )


@dataclass
class CategoryConfig:
    classifier_model: str = field(default_factory=_default_llm_model)
    max_selected_categories: int = 2
    new_category_promote_threshold: int = 3
    classifier_mode: str = "always"
    """Controls how task categories are assigned per sample.

    ``"always"``
        Call the LLM classifier for every sample on every epoch (original
        behaviour, no caching).
    ``"cache"``
        Call the LLM classifier only on the first encounter of a query;
        subsequent calls with the same query return the cached result.
        The cache is keyed by the SHA-256 hash of the query string and
        persisted to ``global_category_cache.json`` under ``storage.root_dir``.
    ``"precomputed"``
        Skip the LLM classifier entirely.  The ``task_category`` field must
        already be present in each sample dict (e.g. pre-populated by an
        offline script).  Falls back to an empty list when the field is absent.
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoryConfig":
        return cls(
            classifier_model=data.get("classifier_model") or _default_llm_model(),
            max_selected_categories=int(data.get("max_selected_categories", 2)),
            new_category_promote_threshold=int(data.get("new_category_promote_threshold", 3)),
            classifier_mode=str(data.get("classifier_mode", "always")),
        )


@dataclass
class BufferConfig:
    window_size: int = 50
    capacity: int = 100
    positive_pool_capacity: int = 50
    negative_pool_capacity: int = 50
    # Soft-isolation parameters for the unified pool.
    # ``min_positive_ratio`` and ``min_negative_ratio`` define the minimum
    # fraction of ``capacity`` that must be reserved for positive and negative
    # trajectories respectively.  When the pool is full and the candidate
    # class already owns fewer slots than its floor, the lowest-value item
    # from the *other* class is evicted instead (cross-class eviction).
    # Set both to 0.0 to disable the floor guarantee (pure global competition).
    min_positive_ratio: float = 0.2
    min_negative_ratio: float = 0.2
    value_alpha: float = 1.0
    value_beta: float = 1.0
    simple_success_step_threshold: int = 2
    loop_repeat_threshold: int = 3
    max_repetitive_chars: int = 10000
    drop_max_steps_stop_trajectory: bool = True
    max_steps_stop_text: str = (
        "Agent stopped: max_steps reached without producing a final answer. "
        "Please try rephrasing your question or increasing max_steps."
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BufferConfig":
        return cls(
            window_size=int(data.get("window_size", 50)),
            capacity=int(data.get("capacity", 100)),
            positive_pool_capacity=int(data.get("positive_pool_capacity", 50)),
            negative_pool_capacity=int(data.get("negative_pool_capacity", 50)),
            min_positive_ratio=float(data.get("min_positive_ratio", 0.2)),
            min_negative_ratio=float(data.get("min_negative_ratio", 0.2)),
            value_alpha=float(data.get("value_alpha", 1.0)),
            value_beta=float(data.get("value_beta", 1.0)),
            simple_success_step_threshold=int(data.get("simple_success_step_threshold", 2)),
            loop_repeat_threshold=int(data.get("loop_repeat_threshold", 3)),
            max_repetitive_chars=int(data.get("max_repetitive_chars", 10000)),
            drop_max_steps_stop_trajectory=bool(data.get("drop_max_steps_stop_trajectory", True)),
            max_steps_stop_text=data.get(
                "max_steps_stop_text",
                "Agent stopped: max_steps reached without producing a final answer. Please try rephrasing your question or increasing max_steps.",
            ),
        )


@dataclass
class EncodeConfig:
    model: str = field(default_factory=_default_llm_model)
    draft_initial_utility: float = 0.5
    trigger_on_window_end: bool = True
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncodeConfig":
        return cls(
            model=data.get("model") or _default_llm_model(),
            draft_initial_utility=float(data.get("draft_initial_utility", 0.5)),
            trigger_on_window_end=bool(data.get("trigger_on_window_end", True)),
        )


@dataclass
class UtilityConfig:
    """Hyperparameters for the utility update and memory-strength channels.

    Update formula fields
    ---------------------
    negative_base_penalty : float
        Fixed base penalty subtracted from the utility delta whenever a skill
        receives ``ADOPTED_NEGATIVE`` credit (``weight == -1``).  This ensures
        a negative-credit event always produces a non-positive contribution
        regardless of the reward level.  Default: 0.10.
    negative_harm_scale : float
        Scales the additional harm term ``max(0, -(reward - category_baseline))``
        for negative-credit events.  Larger values penalise skills more heavily
        when the reward falls below the category baseline.  Default: 0.5.
    positive_advantage_scale : float
        Scales the relative-advantage term ``(reward - category_baseline)`` for
        positive-credit events.  Default: 1.0.

    Category EMA baseline fields
    ----------------------------
    category_ema_alpha : float
        Smoothing factor for the per-(category, reward_type) exponential moving
        average baseline.  Equivalent to weighting the most recent window by
        ``alpha`` and the historical EMA by ``(1 - alpha)``.  A value of 0.2
        corresponds roughly to a 5-window effective history.  Default: 0.2.
    category_ema_min_samples : int
        Minimum number of samples required before the EMA baseline is trusted.
        When sample count is below this threshold the neutral value 0.5 is used
        instead.  Default: 5.
    category_ema_default : float
        Neutral baseline used during cold-start (before ``category_ema_min_samples``
        is reached).  Default: 0.5.

    Penalty / bonus fields
    ----------------------
    risk_penalty_high_negative_ratio : float
        Fractional penalty applied to the utility delta when a *high-risk*
        skill receives a negative credit weight (``weight == -1``).  The
        penalty term is ``current_utility * ratio``, clipped to [0, 1].
        Default: 0.10 (10 %).

    Memory-strength fields (window-based Ebbinghaus decay)
    -------------------------------------------------------
    memory_tau_base_windows : float
        Base decay time-constant in **windows**.  After this many windows
        have elapsed since the last successful adoption the memory strength
        decays to ``e^{-1} \u2248 0.37``.  Default: 10 windows.
    memory_tau_max_windows : float
        Upper cap on the reinforced tau (in windows).  Default: 100 windows.

    Learning-rate schedule fields (cosine warmup)
    ---------------------------------------------
    lr_warmup_steps : int
        Number of adoption events during which the learning rate linearly
        ramps from ``lr_base`` up to ``lr_max``.  After warmup the rate
        follows a cosine decay back toward ``lr_base`` over ``lr_decay_steps``
        and then stays flat at ``lr_base`` indefinitely.  Default: 5.
    lr_max : float
        Peak learning rate reached at the end of the warmup phase.
        Default: 0.20.
    lr_base : float
        Stable learning rate used after the cosine decay has completed.
        Default: 0.05.
    lr_decay_steps : int
        Number of adoption events over which the cosine decay runs
        (starting immediately after warmup ends).  Default: 20.
    """

    # Update formula
    negative_base_penalty: float = 0.10
    negative_harm_scale: float = 0.5
    positive_advantage_scale: float = 1.0
    # Category EMA baseline
    category_ema_alpha: float = 0.2
    category_ema_min_samples: int = 5
    category_ema_default: float = 0.5
    # Risk penalty
    risk_penalty_high_negative_ratio: float = 0.10
    # Clip bounds
    utility_clip_min: float = 0.0
    utility_clip_max: float = 1.0
    # Memory-strength decay
    memory_tau_base_windows: float = 10.0
    memory_tau_max_windows: float = 100.0
    # Cosine-warmup learning-rate schedule
    lr_warmup_steps: int = 5
    lr_max: float = 0.20
    lr_base: float = 0.05
    lr_decay_steps: int = 20

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UtilityConfig":
        return cls(
            negative_base_penalty=float(data.get("negative_base_penalty", 0.10)),
            negative_harm_scale=float(data.get("negative_harm_scale", 0.5)),
            positive_advantage_scale=float(data.get("positive_advantage_scale", 1.0)),
            category_ema_alpha=float(data.get("category_ema_alpha", 0.2)),
            category_ema_min_samples=int(data.get("category_ema_min_samples", 5)),
            category_ema_default=float(data.get("category_ema_default", 0.5)),
            risk_penalty_high_negative_ratio=float(data.get("risk_penalty_high_negative_ratio", 0.10)),
            utility_clip_min=float(data.get("utility_clip_min", 0.0)),
            utility_clip_max=float(data.get("utility_clip_max", 1.0)),
            memory_tau_base_windows=float(data.get("memory_tau_base_windows", 10.0)),
            memory_tau_max_windows=float(data.get("memory_tau_max_windows", 100.0)),
            lr_warmup_steps=int(data.get("lr_warmup_steps", 5)),
            lr_max=float(data.get("lr_max", 0.20)),
            lr_base=float(data.get("lr_base", 0.05)),
            lr_decay_steps=int(data.get("lr_decay_steps", 20)),
        )


@dataclass
class GovernanceConfig:
    manage_every_n_windows: int = 5
    merge_similarity_threshold: float = 0.90
    deprecate_utility_threshold: float = 0.3
    mature_utility_threshold: float = 0.8
    mature_usage_threshold: int = 20
    capacity_general: int = 30
    capacity_task_level_per_category: int = 20
    capacity_action_level_per_tool: int = 10
    capacity_meta_memory: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceConfig":
        return cls(
            manage_every_n_windows=int(data.get("manage_every_n_windows", 5)),
            merge_similarity_threshold=float(data.get("merge_similarity_threshold", 0.90)),
            deprecate_utility_threshold=float(data.get("deprecate_utility_threshold", 0.3)),
            mature_utility_threshold=float(data.get("mature_utility_threshold", 0.8)),
            mature_usage_threshold=int(data.get("mature_usage_threshold", 20)),
            capacity_general=int(data.get("capacity_general", 30)),
            capacity_task_level_per_category=int(data.get("capacity_task_level_per_category", 20)),
            capacity_action_level_per_tool=int(data.get("capacity_action_level_per_tool", 10)),
            capacity_meta_memory=int(data.get("capacity_meta_memory", 10)),
        )


@dataclass
class ContextGuardConfig:
    token_budget: int = 16000
    loop_break_repeat_threshold: int = 3
    key_finding_pin_step: int = 5
    context_trim_ratio: float = 0.7
    context_trim_keep_last_n_steps: int = 3
    context_trim_observation_chars: int = 150

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextGuardConfig":
        return cls(
            token_budget=int(data.get("token_budget", 16000)),
            loop_break_repeat_threshold=int(data.get("loop_break_repeat_threshold", 3)),
            key_finding_pin_step=int(data.get("key_finding_pin_step", 5)),
            context_trim_ratio=float(data.get("context_trim_ratio", 0.7)),
            context_trim_keep_last_n_steps=int(data.get("context_trim_keep_last_n_steps", 3)),
            context_trim_observation_chars=int(data.get("context_trim_observation_chars", 150)),
        )


@dataclass
class EvaluationScheduleConfig:
    enabled: bool = True
    granularity: str = "epoch"
    evaluate_every_n_epochs: int = 1
    evaluate_every_n_windows: int = 1
    selected_benches: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationScheduleConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            granularity=data.get("granularity", "epoch"),
            evaluate_every_n_epochs=int(data.get("evaluate_every_n_epochs", 1)),
            evaluate_every_n_windows=int(data.get("evaluate_every_n_windows", 1)),
            selected_benches=list(data.get("selected_benches", []) or []),
        )


@dataclass
class OnlineConfig:
    benchmark_root: str = ""
    selected_benches: List[str] = field(default_factory=list)
    shuffle_seed: int = 42
    epochs: int = 1
    max_samples_per_dataset: int = 0
    evaluation: EvaluationScheduleConfig = field(default_factory=EvaluationScheduleConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnlineConfig":
        return cls(
            benchmark_root=data.get("benchmark_root", data.get("data_root", "")),
            selected_benches=list(data.get("selected_benches", []) or []),
            shuffle_seed=int(data.get("shuffle_seed", 42)),
            epochs=int(data.get("epochs", 1)),
            max_samples_per_dataset=int(data.get("max_samples_per_dataset", 0)),
            evaluation=EvaluationScheduleConfig.from_dict(data.get("evaluation", {})),
        )


@dataclass
class OfflineConfig:
    freeze_skill_library_during_test: bool = True
    train_split_path: str = ""
    test_split_path: str = ""
    train_root: str = ""
    test_root: str = ""
    selected_benches: List[str] = field(default_factory=list)
    shuffle_seed: int = 42
    epochs: int = 1
    max_train_samples_per_dataset: int = 0
    max_test_samples_per_dataset: int = 0
    evaluation: EvaluationScheduleConfig = field(default_factory=EvaluationScheduleConfig)
    # Controls whether to run training, evaluation, or both in one pass.
    # Allowed values:
    #   "train_then_eval" (default) — train one epoch then immediately evaluate
    #   "train_only"                — train only; skip evaluation entirely
    #   "eval_only"                 — evaluate only using an existing skill library;
    #                                 can be launched in parallel per-bench for speed
    run_mode: str = "train_then_eval"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OfflineConfig":
        return cls(
            freeze_skill_library_during_test=bool(data.get("freeze_skill_library_during_test", True)),
            train_split_path=data.get("train_split_path", ""),
            test_split_path=data.get("test_split_path", ""),
            train_root=data.get("train_root", ""),
            test_root=data.get("test_root", ""),
            selected_benches=list(data.get("selected_benches", []) or []),
            shuffle_seed=int(data.get("shuffle_seed", 42)),
            epochs=int(data.get("epochs", 1)),
            max_train_samples_per_dataset=int(data.get("max_train_samples_per_dataset", 0)),
            max_test_samples_per_dataset=int(data.get("max_test_samples_per_dataset", 0)),
            evaluation=EvaluationScheduleConfig.from_dict(data.get("evaluation", {})),
            run_mode=str(data.get("run_mode", "train_then_eval")),
        )


@dataclass
class StorageConfig:
    """Filesystem layout for the skill library.

    All paths are derived from ``experiment_root`` plus the ``*_dirname``
    fields.  Flat per-file path overrides have been removed; use the
    structured directory layout exclusively.

    Layout under ``experiment_root``
    ---------------------------------
    ``<experiment_root>/
      <skills_dirname>/          # skill Markdown files (branch sub-dirs)
        meta_memory/             # meta-memory skills
      <index_dirname>/
        utility_index.json
        categories.json
      <embeddings_dirname>/
        embeddings.json
      <traces_dirname>/          # per-trajectory usage traces
    ``

    The global query-embedding cache lives one level above at
    ``<root_dir>/global_query_embeddings.json``.
    """

    root_dir: str = "skills"
    experiment_root: str = "skills/default_experiment"
    skills_dirname: str = "skills"
    index_dirname: str = "index"
    embeddings_dirname: str = "embeddings"
    traces_dirname: str = "usage_traces"
    meta_memory_dirname: str = "meta_memory"
    file_lock_timeout_seconds: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageConfig":
        experiment_root = data.get("experiment_root") or data.get("root_dir", "skills/default_experiment")
        return cls(
            root_dir=data.get("root_dir", "skills"),
            experiment_root=experiment_root,
            skills_dirname=data.get("skills_dirname", "skills"),
            index_dirname=data.get("index_dirname", "index"),
            embeddings_dirname=data.get("embeddings_dirname", "embeddings"),
            traces_dirname=data.get("traces_dirname", "usage_traces"),
            meta_memory_dirname=data.get("meta_memory_dirname", "meta_memory"),
            file_lock_timeout_seconds=int(data.get("file_lock_timeout_seconds", 10)),
        )


@dataclass
class EvoConfig:
    mode: str = "online"
    enabled: bool = False
    api: APIConfig = field(default_factory=APIConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    categories: CategoryConfig = field(default_factory=CategoryConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    context_guard: ContextGuardConfig = field(default_factory=ContextGuardConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvoConfig":
        return cls(
            mode=data.get("mode", "online"),
            enabled=bool(data.get("enabled", False)),
            api=APIConfig.from_dict(data.get("api", {})),
            retrieval=RetrievalConfig.from_dict(data.get("retrieval", {})),
            categories=CategoryConfig.from_dict(data.get("categories", {})),
            buffer=BufferConfig.from_dict(data.get("buffer", {})),
            encode=EncodeConfig.from_dict(data.get("encode", {})),
            utility=UtilityConfig.from_dict(data.get("utility", {})),
            governance=GovernanceConfig.from_dict(data.get("governance", {})),
            context_guard=ContextGuardConfig.from_dict(data.get("context_guard", {})),
            online=OnlineConfig.from_dict(data.get("online", {})),
            offline=OfflineConfig.from_dict(data.get("offline", {})),
            storage=StorageConfig.from_dict(data.get("storage", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "EvoConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_evo_config(default_path: str | Path, override_path: Optional[str | Path] = None) -> EvoConfig:
    if not override_path:
        return EvoConfig.from_file(default_path)
    base_dict = json.loads(Path(default_path).read_text(encoding="utf-8"))
    override = json.loads(Path(override_path).read_text(encoding="utf-8"))
    merged = _deep_merge(base_dict, override)
    return EvoConfig.from_dict(merged)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
