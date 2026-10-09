from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from medmem_agent.agent import AgentLoop

from .config import EvoConfig
from .online import OnlineEvolutionRunner
from .retrieve import SkillRetriever
from .storage import SkillStore


@dataclass
class OfflineEvolutionRunner:
    train_runner: OnlineEvolutionRunner
    test_agent: AgentLoop
    test_retriever: SkillRetriever
    frozen_store: SkillStore
    sample_runner: Callable[[AgentLoop, Dict[str, Any], str, bool], Dict[str, Any]]
    evo_config: EvoConfig
    supports_vision: bool = False
    test_agent_selector: Optional[Callable[[str], AgentLoop]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve_only(
        self,
        *,
        train_samples: List[Dict[str, Any]],
        train_output_path: str,
        benchmark_evaluator: Optional[Any] = None,
        epoch_index: int = 1,
        resume_window: int = 0,
    ) -> Dict[str, Any]:
        """Run one training epoch without any evaluation.

        Returns the train-epoch result dict produced by
        ``OnlineEvolutionRunner.run_epoch()``.
        """
        return self.train_runner.run_epoch(
            samples=train_samples,
            dataset_name="mixed_train",
            output_path=train_output_path,
            benchmark_evaluator=benchmark_evaluator,
            epoch_index=epoch_index,
            resume_window=resume_window,
        )

    def test_only(
        self,
        *,
        test_samples_by_dataset: Dict[str, List[Dict[str, Any]]],
        test_output_paths: Dict[str, str],
        benchmark_evaluators: Optional[Any] = None,
        current_window: int = 0,
        resume: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run evaluation against a frozen skill library.

        ``current_window`` is the global window index used for Ebbinghaus
        memory-strength scoring during retrieval.  In ``eval_only`` mode
        (where no training has been run in this process) pass the value
        inferred from the utility index via
        ``OfflineEvolutionRunner.infer_current_window()``.

        When ``resume=True``, any samples whose ``id`` already appears in the
        output ``.jsonl`` file are skipped; their existing records are loaded
        and included in the returned results so that summaries are computed
        over the full dataset.

        Returns a dict mapping dataset_name -> list of result dicts.
        """
        test_results_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
        self.frozen_store.ensure_initialized()

        for dataset_name, samples in test_samples_by_dataset.items():
            output_path = Path(test_output_paths[dataset_name])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------
            # Resume: load already-finished records and build skip-set.
            # Use sample id first; fall back to query for deduplication.
            # ----------------------------------------------------------
            completed_ids: set = set()
            completed_queries: set = set()
            dataset_results: List[Dict[str, Any]] = []
            if resume and output_path.exists():
                with open(output_path, encoding="utf-8") as _fh:
                    for _line in _fh:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _rec = json.loads(_line)
                            dataset_results.append(_rec)
                            _sid = str(_rec.get("id", "") or "")
                            _q = str(_rec.get("query", "") or "")
                            if _sid:
                                completed_ids.add(_sid)
                            if _q:
                                completed_queries.add(_q)
                        except Exception:
                            pass
                if completed_ids or completed_queries:
                    print(
                        f"[eval_only resume] {dataset_name}: "
                        f"skipping {len(dataset_results)} already-completed samples."
                    )

            scorer = self._resolve_benchmark_evaluator(benchmark_evaluators, dataset_name)

            for sample in samples:
                # Skip already-completed samples when resuming
                if resume and (completed_ids or completed_queries):
                    _sid = str(sample.get("id", "") or "")
                    _q = str(sample.get("query", "") or "")
                    if (_sid and _sid in completed_ids) or (_q and _q in completed_queries):
                        continue

                agent = self._get_test_agent(dataset_name)
                _classifier_mode = getattr(
                    self.train_runner.evo_config.categories, "classifier_mode", "always"
                )
                if _classifier_mode == "precomputed":
                    task_category = list(sample.get("task_category") or [])
                else:
                    task_category = self.train_runner.category_registry.classify(
                        sample.get("query", "")
                    )

                retrieval = self.test_retriever.retrieve(
                    sample.get("query", ""),
                    task_category,
                    current_window=current_window,
                )
                if hasattr(agent, "set_evolution_runtime_context"):
                    agent.set_evolution_runtime_context(
                        {
                            "task_category": task_category,
                            "retrieved_skills_block": retrieval.retrieved_skills_block,
                            "injected_skill_ids": retrieval.injected_skill_ids,
                        }
                    )
                result = self.sample_runner(agent, sample, dataset_name, self.supports_vision)
                if scorer is not None:
                    result["eval_result"] = float(scorer(sample, result))
                result["task_category"] = task_category
                result["injected_skill_ids"] = retrieval.injected_skill_ids
                dataset_results.append(result)
                with open(output_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")

            test_results_by_dataset[dataset_name] = dataset_results

        return test_results_by_dataset

    def evolve_then_test(
        self,
        *,
        train_samples: List[Dict[str, Any]],
        test_samples_by_dataset: Dict[str, List[Dict[str, Any]]],
        train_output_path: str,
        test_output_paths: Dict[str, str],
        benchmark_evaluators: Optional[Any] = None,
        epoch_index: int = 1,
        resume_window: int = 0,
    ) -> Dict[str, Any]:
        """Train one epoch then immediately evaluate (original behaviour).

        Kept for backward compatibility; internally delegates to
        ``evolve_only`` and ``test_only``.
        """
        train_results = self.evolve_only(
            train_samples=train_samples,
            train_output_path=train_output_path,
            benchmark_evaluator=benchmark_evaluators,
            epoch_index=epoch_index,
            resume_window=resume_window,
        )
        test_results_by_dataset = self.test_only(
            test_samples_by_dataset=test_samples_by_dataset,
            test_output_paths=test_output_paths,
            benchmark_evaluators=benchmark_evaluators,
            current_window=self.train_runner.total_windows_seen,
        )
        return {"train": train_results, "test": test_results_by_dataset}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def infer_current_window(store: SkillStore) -> int:
        """Infer the current window index from the utility index on disk.

        Used in ``eval_only`` mode where no training has been performed in
        the current process.  Returns the maximum ``last_success_window``
        seen across all skills, or 0 if the index is empty.
        """
        utility_index = store.load_utility_index()
        if not utility_index:
            return 0
        return max(
            # last_adopted_window is the canonical field name; fall back to
            # last_success_window for backward compatibility with older runs.
            int(row.get("last_adopted_window", row.get("last_success_window", 0)))
            for row in utility_index.values()
        )

    def _get_test_agent(self, dataset_name: str) -> AgentLoop:
        if self.test_agent_selector is not None:
            return self.test_agent_selector(dataset_name)
        return self.test_agent

    @staticmethod
    def _resolve_benchmark_evaluator(benchmark_evaluators: Any, dataset_name: str):
        if benchmark_evaluators is None:
            return None
        if isinstance(benchmark_evaluators, dict):
            return benchmark_evaluators.get(dataset_name)
        return benchmark_evaluators
