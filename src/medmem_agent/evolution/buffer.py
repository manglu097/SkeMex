from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import BufferConfig
from .types import Trajectory


INFRA_ERROR_KEYWORDS = (
    "TimeoutError",
    "JSONDecodeError",
    "ConnectionError",
    "ReadTimeout",
    "APIConnectionError",
    "ServiceUnavailable",
)


@dataclass
class BufferInsertResult:
    kept: bool
    slot_name: str
    value_score: float = 0.0
    reason: str = ""
    evicted_trajectory_id: Optional[str] = None


@dataclass
class GatedTrajectoryBuffer:
    """Trajectory buffer with soft-isolation between positive and negative samples.

    Design
    ------
    All trajectories share a single unified ``pool`` (replacing the old
    ``positive_pool`` / ``negative_pool`` hard split).  The total capacity is
    ``config.capacity``.

    **Soft-isolation guarantee**: ``config.min_positive_ratio`` and
    ``config.min_negative_ratio`` define the *minimum* fraction of the total
    capacity that must be reserved for positive and negative trajectories
    respectively.  When the pool is full and a new trajectory arrives:

    1. Compute the *floor* for each class:
       ``pos_floor = ceil(capacity * min_positive_ratio)``
       ``neg_floor = ceil(capacity * min_negative_ratio)``

    2. Determine which existing items are *evictable* — an item is protected
       from eviction if removing it would cause its class count to drop below
       the floor.

    3. Among evictable items, evict the one with the lowest ``_value_score``.
       If the new item's value is not higher than the lowest evictable score,
       the new item is dropped instead.

    Setting both ratios to 0.0 degrades to pure global competition with no
    class-level protection.

    Backward compatibility
    ----------------------
    ``positive_pool`` and ``negative_pool`` properties are kept as read-only
    views so that existing code that reads them (e.g. tests) continues to work.
    """

    config: BufferConfig
    pool: List[Trajectory] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Backward-compatible views
    # ------------------------------------------------------------------

    @property
    def positive_pool(self) -> List[Trajectory]:
        return [t for t in self.pool if self._is_positive(t)]

    @property
    def negative_pool(self) -> List[Trajectory]:
        return [t for t in self.pool if not self._is_positive(t)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, trajectory: Trajectory) -> BufferInsertResult:
        keep, reason = self._passes_gating(trajectory)
        if not keep:
            return BufferInsertResult(kept=False, slot_name="discard", reason=reason)

        slot_name = "positive" if self._is_positive(trajectory) else "negative"
        value = self._value_score(trajectory, self.pool)
        capacity = self.config.capacity

        if len(self.pool) < capacity:
            self.pool.append(trajectory)
            return BufferInsertResult(
                kept=True, slot_name=slot_name, value_score=value, reason="inserted"
            )

        # Pool is full — find the lowest-value evictable item.
        evictable_indices = self._evictable_indices(excluding_new=trajectory)
        if not evictable_indices:
            # Every existing item is protected; drop the new one.
            return BufferInsertResult(
                kept=False,
                slot_name=slot_name,
                value_score=value,
                reason="lower_than_existing",
            )

        scored = [
            (self._value_score(self.pool[i], self.pool), i, self.pool[i])
            for i in evictable_indices
        ]
        lowest_value, lowest_idx, lowest_item = min(scored, key=lambda x: x[0])

        if value > lowest_value:
            self.pool[lowest_idx] = trajectory
            return BufferInsertResult(
                kept=True,
                slot_name=slot_name,
                value_score=value,
                reason="replaced_lower_value",
                evicted_trajectory_id=lowest_item.id,
            )

        return BufferInsertResult(
            kept=False,
            slot_name=slot_name,
            value_score=value,
            reason="lower_than_existing",
        )

    def all_items(self) -> List[Trajectory]:
        return list(self.pool)

    def clear(self) -> None:
        self.pool.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_positive(trajectory: Trajectory) -> bool:
        return float(trajectory.eval_result or 0.0) > 0.5

    def _evictable_indices(self, excluding_new: Trajectory) -> List[int]:
        """Return indices of pool items that can be evicted without violating floors.

        An item at index ``i`` is *evictable* if removing it would not cause
        its class count to fall below the class floor.

        Parameters
        ----------
        excluding_new:
            The incoming trajectory (not yet in the pool).  Its class is used
            to determine whether the new item itself would benefit from a floor
            guarantee (but the new item is not yet in the pool, so it doesn't
            affect current counts).
        """
        capacity = self.config.capacity
        pos_floor = math.ceil(capacity * float(self.config.min_positive_ratio))
        neg_floor = math.ceil(capacity * float(self.config.min_negative_ratio))

        pos_count = sum(1 for t in self.pool if self._is_positive(t))
        neg_count = len(self.pool) - pos_count

        evictable: List[int] = []
        for i, item in enumerate(self.pool):
            if self._is_positive(item):
                # Can evict a positive item only if pos_count > pos_floor
                if pos_count > pos_floor:
                    evictable.append(i)
            else:
                # Can evict a negative item only if neg_count > neg_floor
                if neg_count > neg_floor:
                    evictable.append(i)

        return evictable

    def _passes_gating(self, trajectory: Trajectory) -> Tuple[bool, str]:
        final_answer = str(trajectory.prediction or trajectory.metadata.get("final_answer", "")).strip()
        if self.config.drop_max_steps_stop_trajectory and final_answer == self.config.max_steps_stop_text.strip():
            return False, "max_steps_stop_without_answer"

        for step in trajectory.steps:
            error_text = str(step.get("error", ""))
            if error_text and any(keyword in error_text for keyword in INFRA_ERROR_KEYWORDS):
                return False, "infrastructure_error"
            llm_output = str(step.get("llm_output", ""))
            observation = str(step.get("observation", ""))
            if len(llm_output) > self.config.max_repetitive_chars or len(observation) > self.config.max_repetitive_chars:
                return False, "repetitive_verbose_output"

        if len(trajectory.steps) <= self.config.simple_success_step_threshold and float(trajectory.eval_result or 0.0) == 1.0:
            return False, "too_simple_success"

        if self._has_invalid_repeated_actions(trajectory):
            return False, "invalid_repeated_actions"

        if len(trajectory.steps) > 3:
            return True, "valuable_multi_step"
        if trajectory.injected_skill_ids and float(trajectory.eval_result or 0.0) <= 0.5:
            return True, "valuable_skill_failure"
        return True, "default_keep"

    def _has_invalid_repeated_actions(self, trajectory: Trajectory) -> bool:
        threshold = self.config.loop_repeat_threshold
        if len(trajectory.steps) < threshold:
            return False
        action_pairs = []
        for step in trajectory.steps:
            if "action" not in step:
                action_pairs.append(None)
            else:
                action_pairs.append((step.get("action"), step.get("action_input")))
        for idx in range(len(action_pairs) - threshold + 1):
            window = action_pairs[idx:idx + threshold]
            if window[0] is not None and all(item == window[0] for item in window):
                return True
        return False

    def _value_score(self, trajectory: Trajectory, pool: List[Trajectory]) -> float:
        category_counts = Counter()
        for item in pool:
            for category in item.task_category:
                category_counts[category] += 1
        kc = sum(category_counts.get(cat, 0) for cat in trajectory.task_category)
        steps_term = self.config.value_alpha * math.log(1 + max(len(trajectory.steps), 0))
        injected_term = self.config.value_beta if trajectory.injected_skill_ids else 0.0
        return (steps_term + injected_term) / (1 + kc)
