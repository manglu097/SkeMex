from __future__ import annotations

import json
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from medmem_agent.agent import AgentLoop
from medmem_agent.tools.registry import ToolRegistry

from .buffer import GatedTrajectoryBuffer
from .categories import DynamicTaskCategoryRegistry
from .config import EvoConfig
from .encode import TrajectoryEncoder
from .evaluation import RewardEvaluator
from .governance import SkillGovernance
from .retrieve import SkillRetriever
from .storage import SkillStore
from .trajectory import TrajectoryBuilder
from .types import Trajectory, WindowPerformance
from .utility import UtilityUpdater
from .utils import atomic_write_json, utc_now_iso

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evolution process logger
# ---------------------------------------------------------------------------

class EvolutionLogger:
    """Append-only JSONL logger for the evolution process.

    Each call to ``log()`` writes one JSON object (one line) to
    ``<output_dir>/evolution_log.jsonl``.  The file is created on first write
    and appended to on subsequent writes, so it survives across resume runs.

    Every record contains at minimum:
    - ``timestamp``: ISO-8601 UTC string.
    - ``event``: one of ``"encode"``, ``"utility_update"``, ``"governance"``,
      ``"buffer_insert"``, ``"window_summary"``.
    - Additional event-specific fields.

    The logger is a no-op when ``output_dir`` is ``None`` or empty.
    """

    def __init__(self, output_dir: Optional[str]) -> None:
        self._path: Optional[Path] = None
        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            self._path = p / "evolution_log.jsonl"

    def log(self, event: str, **kwargs: Any) -> None:
        if self._path is None:
            return
        record = {"timestamp": utc_now_iso(), "event": event, **kwargs}
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("EvolutionLogger: failed to write log entry: %s", exc)


# ---------------------------------------------------------------------------
# Online evolution runner
# ---------------------------------------------------------------------------

@dataclass
class OnlineEvolutionRunner:
    agent: AgentLoop
    evo_config: EvoConfig
    store: SkillStore
    category_registry: DynamicTaskCategoryRegistry
    retriever: SkillRetriever
    evaluator: RewardEvaluator
    encoder: TrajectoryEncoder
    utility_updater: UtilityUpdater
    governance: SkillGovernance
    buffer: GatedTrajectoryBuffer
    sample_runner: Callable[[AgentLoop, Dict[str, Any], str, bool], Dict[str, Any]]
    supports_vision: bool = False
    agent_selector: Optional[Callable[[str], AgentLoop]] = None
    window_end_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    epoch_metrics: List[WindowPerformance] = field(default_factory=list)
    total_windows_seen: int = 0
    current_epoch_index: int = 0
    # output_dir is used to derive the evolution log path.  It is set lazily
    # from the first agent's output_dir when not explicitly provided.
    output_dir: Optional[str] = None
    # Per-dataset pre-rendered tool lists; injected into Analysis prompts to
    # enable "exploration gap" detection.
    _available_tools_cache: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _evo_logger: Optional[EvolutionLogger] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Eagerly initialise the logger if output_dir is already known at
        # construction time (e.g. when passed explicitly by the offline script).
        # If output_dir is None the logger will be created lazily in run_epoch.
        if self.output_dir:
            self._evo_logger = EvolutionLogger(self.output_dir)
        else:
            # Provide a no-op logger so that _finalize_window / _evolve_from_trajectory
            # can always call self._evo_logger.log() without a None check.
            self._evo_logger = EvolutionLogger(None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_epoch(
        self,
        *,
        samples: List[Dict[str, Any]],
        dataset_name: str,
        output_path: str,
        benchmark_evaluator: Optional[Any] = None,
        epoch_index: int = 1,
        resume_window: int = 0,
    ) -> List[Dict[str, Any]]:
        self.store.ensure_initialized()
        self.current_epoch_index = epoch_index

        # Initialise the evolution logger on first call (output_dir may have
        # been set after construction, e.g. by the offline script).
        if self._evo_logger is None:
            log_dir = self.output_dir or str(Path(output_path).parent)
            self._evo_logger = EvolutionLogger(log_dir)

        # Restore the window state when resuming an interrupted run.
        if resume_window > 0 and self.total_windows_seen == 0:
            self.total_windows_seen = resume_window

        results: List[Dict[str, Any]] = []
        current_window_results: List[Dict[str, Any]] = []
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        performance_log = output_file.with_suffix(".performance.json")

        # Compute how many samples belong to completed windows.
        samples_to_skip = 0
        if resume_window > 0:
            samples_to_skip = resume_window * self.evo_config.buffer.window_size
            if samples_to_skip > 0:
                logger.info(f"[resume] Skipping {samples_to_skip} samples (resuming from window {resume_window})")

        pbar = tqdm(enumerate(samples, 1), total=len(samples), desc=f"Epoch {epoch_index} (Window {self.total_windows_seen})")

        for index, sample in pbar:
            if index <= samples_to_skip:
                continue

            pbar.set_description(f"Epoch {epoch_index} (Window {self.total_windows_seen})")

            sample_dataset = sample.get("dataset", dataset_name)
            agent = self._get_agent(sample_dataset)
            _classifier_mode = getattr(self.evo_config.categories, "classifier_mode", "always")
            if _classifier_mode == "precomputed":
                task_category = list(sample.get("task_category") or [])
            else:
                task_category = self.category_registry.classify(sample.get("query", ""))
            retrieval = self.retriever.retrieve(sample.get("query", ""), task_category, current_window=self.total_windows_seen)
            if hasattr(agent, "set_evolution_runtime_context"):
                agent.set_evolution_runtime_context(
                    {
                        "task_category": task_category,
                        "retrieved_skills_block": retrieval.retrieved_skills_block,
                        "injected_skill_ids": retrieval.injected_skill_ids,
                    }
                )

            # Log retrieval decision
            self._evo_logger.log(
                "retrieval",
                epoch=epoch_index,
                window=self.total_windows_seen,
                sample_index=index,
                dataset=sample_dataset,
                task_category=task_category,
                injected_skill_ids=retrieval.injected_skill_ids,
                candidates=[
                    {
                        "skill_id": c.skill_id,
                        "similarity": round(c.similarity, 4),
                        "utility": round(c.utility, 4),
                        "memory": round(c.memory_strength, 4),
                        "combined": round(c.score, 4),
                    }
                    for c in retrieval.candidates
                ],
            )

            result = self.sample_runner(agent, sample, sample_dataset, self.supports_vision)
            scorer = self._resolve_benchmark_evaluator(benchmark_evaluator, sample_dataset)
            benchmark_score: Optional[float] = None
            if scorer is not None:
                if hasattr(scorer, "score_with_details"):
                    benchmark_score, rubric_results = scorer.score_with_details(sample, result)
                    if rubric_results is not None:
                        result.setdefault("metadata", {})
                        result["metadata"]["rubric_results"] = rubric_results
                else:
                    benchmark_score = float(scorer(sample, result))
            eval_record = self.evaluator.evaluate(
                trajectory_id=result["id"],
                dataset=sample_dataset,
                prediction=result.get("prediction"),
                ground_truth=result.get("ground_truth"),
                benchmark_score=benchmark_score,
            )
            history_file = self._resolve_history_file(agent)
            trajectory = TrajectoryBuilder.build(
                sample=sample,
                result=result,
                steps_history=list(getattr(agent, "_steps_history", [])),
                task_category=task_category,
                injected_skill_ids=retrieval.injected_skill_ids,
                eval_result=eval_record.reward,
                history_file=history_file,
            )
            trajectory.metadata.setdefault("retrieval_candidates", [c.__dict__ for c in retrieval.candidates])
            trajectory.metadata.setdefault("retrieved_skills_block", retrieval.retrieved_skills_block)
            insert_result = self.buffer.add(trajectory)

            # Log buffer insert decision
            self._evo_logger.log(
                "buffer_insert",
                epoch=epoch_index,
                window=self.total_windows_seen,
                sample_index=index,
                trajectory_id=trajectory.id,
                dataset=sample_dataset,
                eval_result=eval_record.reward,
                kept=insert_result.kept,
                slot=insert_result.slot_name,
                reason=insert_result.reason,
                value_score=round(insert_result.value_score, 4),
                evicted_id=insert_result.evicted_trajectory_id,
            )

            result["dataset"] = sample_dataset
            result["eval_result"] = eval_record.reward
            result["task_category"] = task_category
            result["injected_skill_ids"] = retrieval.injected_skill_ids
            result["buffer_kept"] = insert_result.kept
            result["buffer_reason"] = insert_result.reason
            result["history_file"] = history_file
            results.append(result)
            current_window_results.append(result)

            if insert_result.kept and self.evo_config.encode.trigger_on_window_end is False:
                self._evolve_from_trajectory(trajectory)

            with open(output_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

            if index % self.evo_config.buffer.window_size == 0:
                self._finalize_window(current_window_results)
                current_window_results = []
                atomic_write_json(performance_log, [item.__dict__ for item in self.epoch_metrics])
                pbar.set_description(f"Epoch {epoch_index} (Window {self.total_windows_seen})")

        if current_window_results:
            self._finalize_window(current_window_results)
            atomic_write_json(performance_log, [item.__dict__ for item in self.epoch_metrics])
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finalize_window(self, window_results: List[Dict[str, Any]]) -> None:
        window_trajectories = self.buffer.all_items()
        encode_results: Dict[str, Any] = {}
        next_window_index = self.total_windows_seen + 1
        if self.evo_config.encode.trigger_on_window_end:
            for trajectory in window_trajectories:
                encode_results[trajectory.id] = self._evolve_from_trajectory(trajectory, apply_utility=False)
            if encode_results:
                update_result = self.utility_updater.update_window(
                    window_trajectories, encode_results, current_window=next_window_index
                )
                skill_deltas = update_result.get("skill_deltas", [])
                # Log utility updates for this window
                if skill_deltas:
                    self._evo_logger.log(
                        "utility_update",
                        epoch=self.current_epoch_index,
                        window=next_window_index,
                        skill_deltas=skill_deltas,
                    )

        self.total_windows_seen = next_window_index
        window_index = self.total_windows_seen
        average_reward = sum(float(item.get("eval_result", 0.0)) for item in window_results) / max(len(window_results), 1)
        dataset_breakdown: Dict[str, float] = {}
        if window_results:
            dataset_groups: Dict[str, List[float]] = {}
            for item in window_results:
                dataset_groups.setdefault(item.get("dataset", "unknown"), []).append(float(item.get("eval_result", 0.0)))
            dataset_breakdown = {
                dataset: sum(values) / len(values)
                for dataset, values in dataset_groups.items()
            }
        performance = WindowPerformance(
            window_index=window_index,
            sample_count=len(window_results),
            average_reward=average_reward,
            dataset_breakdown=dataset_breakdown,
        )
        self.epoch_metrics.append(performance)

        # Log window summary
        self._evo_logger.log(
            "window_summary",
            epoch=self.current_epoch_index,
            window=window_index,
            sample_count=performance.sample_count,
            average_reward=round(average_reward, 4),
            dataset_breakdown={k: round(v, 4) for k, v in dataset_breakdown.items()},
            buffer_size=len(window_trajectories),
            encoded_count=len(encode_results),
        )

        if self.window_end_callback is not None:
            self.window_end_callback(
                {
                    "epoch_index": self.current_epoch_index,
                    "window_index": window_index,
                    "window_results": window_results,
                    "window_performance": performance,
                }
            )
        if window_index % self.evo_config.governance.manage_every_n_windows == 0:
            governance_result = self.governance.manage()
            # Log governance actions
            self._evo_logger.log(
                "governance",
                epoch=self.current_epoch_index,
                window=window_index,
                merged=governance_result.get("merged", []),
                deprecated=governance_result.get("deprecated", []),
                matured=governance_result.get("matured", []),
                archived=governance_result.get("archived", []),
            )
        self.buffer.clear()

    def _get_available_tools_list(self, dataset: str) -> str:
        """Lazily render the available tools list for a specific dataset."""
        if dataset in self._available_tools_cache:
            return self._available_tools_cache[dataset]
        agent = self._get_agent(dataset)
        registry: Optional[ToolRegistry] = getattr(agent, "tool_registry", None)
        if registry is None:
            self._available_tools_cache[dataset] = ""
            return ""
        lines = []
        for name, tool in sorted(registry.tools.items()):
            desc = getattr(tool, "description", "") or ""
            short_desc = desc.split(".")[0].strip()
            lines.append(f"- `{name}`: {short_desc}")
        result = "\n".join(lines)
        self._available_tools_cache[dataset] = result
        return result

    def _evolve_from_trajectory(self, trajectory: Trajectory, apply_utility: bool = True):
        available_tools = self._get_available_tools_list(trajectory.dataset)
        encode_result = self.encoder.encode(trajectory, available_tools_list=available_tools)
        created_skill = self.governance.apply_encode_result(encode_result)
        if created_skill is not None:
            trajectory.metadata.setdefault("created_skill_ids", []).append(created_skill.id)

        # Log encode result
        self._evo_logger.log(
            "encode",
            trajectory_id=trajectory.id,
            dataset=trajectory.dataset,
            eval_result=float(trajectory.eval_result or 0.0),
            output_action=encode_result.output_action,
            output_intent=getattr(encode_result, "output_intent", None),
            patch_target_id=encode_result.patch_target_id,
            created_skill_id=created_skill.id if created_skill else None,
            skill_adoption=[
                {
                    "skill_id": a.skill_id,
                    "credit_weight": a.credit_weight,
                    "adoption_type": getattr(a, "adoption_type", None),
                }
                for a in encode_result.skill_adoption
            ],
            extracted_pattern=encode_result.extracted_pattern[:200]
            if encode_result.extracted_pattern
            else None,
        )

        if apply_utility:
            update_result = self.utility_updater.update(trajectory, encode_result)
            # update() now returns the utility_index dict (backward compat wrapper)
            # skill_deltas are not available in single-trajectory mode; that is fine.

        return encode_result

    def _get_agent(self, dataset_name: str) -> AgentLoop:
        if self.agent_selector is not None:
            return self.agent_selector(dataset_name)
        return self.agent

    @staticmethod
    def _resolve_history_file(agent: AgentLoop) -> str:
        output_dir = Path(agent.output_dir)
        if agent.history_save_mode == "append":
            return str(output_dir / agent.history_filename)
        candidates = sorted(output_dir.glob("run_*.jsonl"))
        return str(candidates[-1]) if candidates else ""

    @staticmethod
    def _resolve_benchmark_evaluator(benchmark_evaluator: Any, dataset_name: str):
        if benchmark_evaluator is None:
            return None
        if isinstance(benchmark_evaluator, dict):
            return benchmark_evaluator.get(dataset_name)
        return benchmark_evaluator
