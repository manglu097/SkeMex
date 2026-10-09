from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable


_EVAL_DIR = Path(__file__).resolve().parents[3] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import eval_healthbench  # type: ignore
import eval_livemedbbench  # type: ignore
import eval_standard  # type: ignore

from medmem_agent.evolution.prompts import normalize_dataset_name


@dataclass
class BenchmarkScorer:
    dataset: str
    judge_model: str = "gpt-4.1-mini"

    def __call__(self, sample: Dict[str, Any], result: Dict[str, Any]) -> float:
        score, _ = self.score_with_details(sample, result)
        return score

    def score_with_details(
        self, sample: Dict[str, Any], result: Dict[str, Any]
    ) -> tuple:
        """Return (score: float, rubric_results: list | None).

        For rubric datasets (HealthBench, LiveMedBench) the second element is
        the per-criterion grading list so callers can store it in trajectory
        metadata for downstream encode analysis.  For all other datasets it is
        ``None``.
        """
        record = {
            "query": sample.get("query", result.get("query", "")),
            "prediction": result.get("prediction", "") or "",
            "ground_truth": sample.get("ground_truth", result.get("ground_truth")),
            "success": result.get("success", True),
        }
        dataset = normalize_dataset_name(self.dataset)
        if dataset == "healthbench":
            scored = eval_healthbench.score_sample(record, model=self.judge_model, verbose=False)
            return float(scored.get("sample_score", 0.0)), scored.get("rubric_results")
        if dataset == "livemedbench":
            scored = eval_livemedbbench.score_sample(record, model=self.judge_model, verbose=False)
            return float(scored.get("case_total", 0.0)), scored.get("rubric_results")
        judged = eval_standard.judge_sample(
            prediction=record.get("prediction", ""),
            ground_truth=record.get("ground_truth", ""),
            dataset=self.dataset,
            model=self.judge_model,
        )
        return (1.0 if judged.get("correct", False) else 0.0), None


class MultiBenchmarkScorer(dict):
    def score(self, sample: Dict[str, Any], result: Dict[str, Any]) -> float:
        dataset = sample.get("dataset") or result.get("dataset")
        scorer = self.get(dataset)
        if scorer is None:
            raise KeyError(f"No benchmark scorer registered for dataset={dataset}")
        return float(scorer(sample, result))



def build_benchmark_scorer(dataset: str, judge_model: str = "gpt-4.1-mini") -> Callable[[Dict[str, Any], Dict[str, Any]], float]:
    return BenchmarkScorer(dataset=dataset, judge_model=judge_model)


def build_benchmark_scorers(datasets: Iterable[str], judge_model: str = "gpt-4.1-mini") -> MultiBenchmarkScorer:
    return MultiBenchmarkScorer({
        dataset: build_benchmark_scorer(dataset, judge_model=judge_model)
        for dataset in datasets
    })
