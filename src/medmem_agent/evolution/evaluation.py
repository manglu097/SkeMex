from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import EvaluationRecord


@dataclass
class RewardEvaluator:
    """Unified reward interface for evolution.

    This wrapper keeps evolution modules independent from the repository's
    benchmark-specific evaluator layout. Callers may provide a benchmark score
    directly, or rely on simple fallbacks when no benchmark evaluator is wired
    in yet.
    """

    def evaluate(
        self,
        *,
        trajectory_id: str,
        dataset: str,
        prediction: Any,
        ground_truth: Any,
        raw_eval: Optional[Dict[str, Any]] = None,
        benchmark_score: Optional[float] = None,
    ) -> EvaluationRecord:
        if benchmark_score is not None:
            reward = float(benchmark_score)
        else:
            reward = self._fallback_reward(prediction, ground_truth)
        reward = max(0.0, min(1.0, reward))
        return EvaluationRecord(
            trajectory_id=trajectory_id,
            dataset=dataset,
            reward=reward,
            raw_eval=raw_eval or {},
        )

    def _fallback_reward(self, prediction: Any, ground_truth: Any) -> float:
        if prediction is None:
            return 0.0
        if isinstance(ground_truth, list):
            return 1.0 if str(prediction).strip() in {str(item).strip() for item in ground_truth} else 0.0
        return 1.0 if str(prediction).strip() == str(ground_truth).strip() else 0.0
