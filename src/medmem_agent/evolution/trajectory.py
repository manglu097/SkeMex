from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types import Trajectory


class TrajectoryBuilder:
    """Build unified Trajectory objects from sample/result/agent state."""

    @staticmethod
    def build(
        *,
        sample: Dict[str, Any],
        result: Dict[str, Any],
        steps_history: List[Dict[str, Any]],
        task_category: Optional[List[str]] = None,
        injected_skill_ids: Optional[List[str]] = None,
        eval_result: Optional[float] = None,
        history_file: str = "",
    ) -> Trajectory:
        return Trajectory(
            id=result.get("id") or sample.get("id", "unknown"),
            dataset=result.get("dataset") or sample.get("dataset", "unknown"),
            query=result.get("query") or sample.get("query", ""),
            images=result.get("images") or sample.get("images", []),
            task_category=task_category or [],
            steps=list(steps_history or []),
            injected_skill_ids=injected_skill_ids or [],
            prediction=result.get("prediction"),
            ground_truth=result.get("ground_truth", sample.get("ground_truth")),
            eval_result=eval_result,
            total_duration_s=float(result.get("total_duration_s", 0.0) or 0.0),
            metadata=result.get("metadata") or sample.get("metadata", {}),
            history_file=history_file,
        )
