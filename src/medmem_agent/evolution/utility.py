from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Tuple

from .config import EvoConfig
from .prompts import RUBRIC_DATASETS, normalize_dataset_name
from .storage import SkillStore
from .types import EncodeResult, Trajectory
from .utils import utc_now_iso

# Reserved key inside utility_index.json that stores per-(category, reward_type)
# EMA baseline state.  Using a double-underscore prefix avoids collision with
# any skill_id (which are always alphanumeric).
_CATEGORY_BASELINES_KEY = "__category_baselines__"


def _is_rubric_dataset(dataset: str) -> bool:
    return normalize_dataset_name(dataset) in RUBRIC_DATASETS


def _reward_type(dataset: str) -> str:
    """Return 'rubric' for continuous-score datasets, 'binary' otherwise."""
    return "rubric" if _is_rubric_dataset(dataset) else "binary"


def _baseline_key(category: str, dataset: str) -> str:
    """Compose the EMA state key from category and reward type.

    Using reward_type instead of dataset name keeps the key space compact and
    semantically meaningful: all binary datasets share the same category-level
    difficulty signal, as do all rubric datasets.
    """
    rtype = _reward_type(dataset)
    return f"{category}__{rtype}"


def _cosine_warmup_lr(adoption_count: int, warmup_steps: int, lr_max: float,
                      lr_base: float, decay_steps: int) -> float:
    """Compute the per-step learning rate using a cosine-warmup schedule.

    The schedule has three phases:

    1. **Warmup** (``adoption_count`` in ``[1, warmup_steps]``):
       Learning rate rises linearly from ``lr_base`` to ``lr_max``.

    2. **Cosine decay** (``adoption_count`` in ``(warmup_steps, warmup_steps +
       decay_steps]``):
       Learning rate decays from ``lr_max`` back toward ``lr_base`` following a
       cosine curve.

    3. **Stable** (``adoption_count > warmup_steps + decay_steps``):
       Learning rate stays flat at ``lr_base`` indefinitely.

    Parameters
    ----------
    adoption_count:
        Number of times the skill has been adopted so far (1-based at the
        point of the current update).
    warmup_steps, lr_max, lr_base, decay_steps:
        Hyperparameters sourced from ``UtilityConfig``.
    """
    if warmup_steps <= 0:
        # No warmup — go straight to cosine decay or stable.
        phase_step = adoption_count
        if phase_step <= decay_steps:
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * phase_step / max(decay_steps, 1)))
            return lr_base + (lr_max - lr_base) * cosine_factor
        return lr_base

    if adoption_count <= warmup_steps:
        # Linear warmup: at step 1 → lr_base + 1/warmup_steps * (lr_max - lr_base)
        #                at step warmup_steps → lr_max
        return lr_base + (lr_max - lr_base) * (adoption_count / warmup_steps)

    phase_step = adoption_count - warmup_steps
    if phase_step <= decay_steps:
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * phase_step / decay_steps))
        return lr_base + (lr_max - lr_base) * cosine_factor

    return lr_base


@dataclass
class UtilityUpdater:
    store: SkillStore
    evo_config: EvoConfig

    def update(self, trajectory: Trajectory, encode_result: EncodeResult) -> Dict[str, dict]:
        """Backward-compatible single-trajectory wrapper.

        The v7+ design requires window-level, group-baseline updates.  This
        method is kept only for compatibility and delegates to the batch
        implementation.  The ``current_window`` is inferred as 0 when called
        outside a proper window context.
        """
        result = self.update_window(
            [trajectory], {trajectory.id: encode_result}, current_window=0
        )
        return result["utility_index"]

    def update_window(
        self,
        trajectories: Iterable[Trajectory],
        encode_results: Dict[str, EncodeResult],
        current_window: int = 0,
    ) -> Dict[str, object]:
        """Update utility scores for all skills touched in this window.

        Returns a dict with two keys:
        - ``"utility_index"``: the full updated utility index (as before).
        - ``"skill_deltas"``: a list of per-skill update records for logging,
          each containing ``skill_id``, ``old_utility``, ``new_utility``,
          ``delta``, ``alpha_n``, ``adoption_count``, ``reward``, ``baseline``,
          ``advantage``, and ``weight``.

        Design changes vs. the previous implementation
        -----------------------------------------------
        1. **Category EMA baseline** — the per-(category, reward_type) EMA
           replaces the old within-skill window mean.

        2. **State-branch contribution formula** — piecewise function that
           decouples *direction* from *magnitude*.

        3. **success/failure flags use category baseline**.

        4. **Learning-rate uses cosine-warmup schedule** — replaces the old
           ``1 / (1 + sqrt(adoption_count))`` formula.  The schedule is
           parameterised by ``lr_warmup_steps``, ``lr_max``, ``lr_base``, and
           ``lr_decay_steps`` in ``UtilityConfig``.

        5. **Single-event updates are allowed**.
        """
        if self.store.read_only:
            return {"utility_index": self.store.load_utility_index(), "skill_deltas": []}

        trajectories = list(trajectories)
        utility_index = self.store.load_utility_index()
        now = utc_now_iso()
        cfg = self.evo_config.utility

        # ------------------------------------------------------------------
        # Step 0: Update per-(category, reward_type) EMA baselines using ALL
        # trajectories in the window (not just those with encode results).
        # ------------------------------------------------------------------
        category_baselines: Dict[str, dict] = utility_index.setdefault(
            _CATEGORY_BASELINES_KEY, {}
        )
        alpha = float(cfg.category_ema_alpha)
        for trajectory in trajectories:
            reward = float(trajectory.eval_result or 0.0)
            category = (trajectory.task_category or ["uncategorized"])[0]
            key = _baseline_key(category, trajectory.dataset)
            state = category_baselines.setdefault(
                key, {"ema_reward": float(cfg.category_ema_default), "sample_count": 0}
            )
            state["ema_reward"] = (1.0 - alpha) * float(state["ema_reward"]) + alpha * reward
            state["sample_count"] = int(state.get("sample_count", 0)) + 1

        # ------------------------------------------------------------------
        # Step 1: Usage statistics for every injected skill.
        # ------------------------------------------------------------------
        for trajectory in trajectories:
            for skill_id in trajectory.injected_skill_ids:
                row = utility_index.get(skill_id)
                if not row:
                    continue
                row["usage_count"] = int(row.get("usage_count", 0)) + 1
                row["last_used"] = now
                row.setdefault("status", "active")

        # ------------------------------------------------------------------
        # Step 2: Gather credit events for window-level utility updates.
        # ------------------------------------------------------------------
        grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

        for trajectory in trajectories:
            encode_result = encode_results.get(trajectory.id)
            if encode_result is None:
                continue
            reward = float(trajectory.eval_result or 0.0)
            category = (trajectory.task_category or ["uncategorized"])[0]
            bkey = _baseline_key(category, trajectory.dataset)
            state = category_baselines.get(bkey, {})
            min_samples = int(cfg.category_ema_min_samples)
            if int(state.get("sample_count", 0)) >= min_samples:
                baseline = float(state["ema_reward"])
            else:
                baseline = float(cfg.category_ema_default)

            for adoption in encode_result.skill_adoption:
                skill_id = adoption.skill_id
                row = utility_index.get(skill_id)
                if not row:
                    continue
                weight = float(adoption.credit_weight)
                if weight == 0.0:
                    continue

                branch = row.get("branch", "general")
                row.setdefault("status", "active")

                if abs(weight) == 1.0:
                    row["adoption_count"] = int(row.get("adoption_count", 0)) + 1

                success_flag = (weight == 1.0) and (reward > baseline)
                failure_flag = (weight == -1.0) or (weight == 1.0 and reward <= baseline)

                if success_flag:
                    row["success_when_adopted"] = int(row.get("success_when_adopted", 0)) + 1
                    row["last_adopted_window"] = current_window
                    row["memory_tau_windows"] = self._reinforced_tau_windows(row)
                elif failure_flag:
                    row["failure_when_adopted"] = int(row.get("failure_when_adopted", 0)) + 1
                    row["last_adopted_window"] = current_window
                    row.setdefault(
                        "memory_tau_windows",
                        self.evo_config.utility.memory_tau_base_windows,
                    )

                # IGNORED (weight == 0.2) events are excluded from the main
                # utility update to reduce noise.
                if weight == 0.2:
                    continue

                risk_pen = self._risk_penalty(skill_id, weight, row)

                event = {
                    "reward": reward,
                    "weight": weight,
                    "baseline": baseline,
                    "risk_penalty": risk_pen,
                }

                if branch == "general":
                    grouped[(skill_id, category)].append(event)
                elif branch != "meta_memory":
                    grouped[(skill_id, "__global__")].append(event)

        # ------------------------------------------------------------------
        # Step 3: Apply state-branch delta for each (skill, scope) group.
        # ------------------------------------------------------------------
        pos_scale = float(cfg.positive_advantage_scale)
        neg_base = float(cfg.negative_base_penalty)
        neg_harm = float(cfg.negative_harm_scale)

        # LR schedule parameters
        warmup_steps = int(cfg.lr_warmup_steps)
        lr_max = float(cfg.lr_max)
        lr_base_val = float(cfg.lr_base)
        decay_steps = int(cfg.lr_decay_steps)

        skill_deltas: List[dict] = []

        for (skill_id, scope), events in grouped.items():
            row = utility_index.get(skill_id)
            if not row:
                continue

            # Compute per-event contributions using the piecewise formula.
            total_contrib = 0.0
            for item in events:
                r = item["reward"]
                w = item["weight"]
                b = item["baseline"]
                advantage = r - b

                if w == 1.0:
                    contrib = pos_scale * advantage
                elif w == -1.0:
                    harm_term = neg_harm * max(0.0, -advantage)
                    contrib = -(neg_base + harm_term) - item["risk_penalty"]
                else:
                    contrib = 0.0

                total_contrib += contrib

            # Cosine-warmup learning rate (based on adoption_count).
            adoption_count = max(int(row.get("adoption_count", 0)), 1)
            alpha_n = _cosine_warmup_lr(
                adoption_count,
                warmup_steps=warmup_steps,
                lr_max=lr_max,
                lr_base=lr_base_val,
                decay_steps=decay_steps,
            )
            mean_contrib = total_contrib / len(events)
            delta = alpha_n * mean_contrib

            branch = row.get("branch", "general")
            if branch == "general" and scope != "__global__":
                conditional = row.setdefault("conditional_utility", {})
                default_value = float(
                    row.get("global_utility", self.evo_config.encode.draft_initial_utility)
                )
                old_value = float(conditional.get(scope, default_value))
                new_value = self._clip(old_value + delta)
                conditional[scope] = new_value
                values = list(conditional.values())
                row["global_utility"] = (
                    self._clip(sum(values) / len(values)) if values else self._clip(default_value)
                )
                # Record delta for logging (use global_utility as the reported value)
                skill_deltas.append({
                    "skill_id": skill_id,
                    "scope": scope,
                    "old_utility": old_value,
                    "new_utility": new_value,
                    "delta": delta,
                    "alpha_n": alpha_n,
                    "adoption_count": adoption_count,
                    "mean_contrib": mean_contrib,
                    "reward": events[0]["reward"] if events else None,
                    "baseline": events[0]["baseline"] if events else None,
                    "advantage": (events[0]["reward"] - events[0]["baseline"]) if events else None,
                    "weight": events[0]["weight"] if events else None,
                })
            elif branch != "meta_memory":
                old_value = float(
                    row.get("utility_score", self.evo_config.encode.draft_initial_utility)
                )
                new_value = self._clip(old_value + delta)
                row["utility_score"] = new_value
                skill_deltas.append({
                    "skill_id": skill_id,
                    "scope": scope,
                    "old_utility": old_value,
                    "new_utility": new_value,
                    "delta": delta,
                    "alpha_n": alpha_n,
                    "adoption_count": adoption_count,
                    "mean_contrib": mean_contrib,
                    "reward": events[0]["reward"] if events else None,
                    "baseline": events[0]["baseline"] if events else None,
                    "advantage": (events[0]["reward"] - events[0]["baseline"]) if events else None,
                    "weight": events[0]["weight"] if events else None,
                })

        self.store.save_utility_index(utility_index)
        return {"utility_index": utility_index, "skill_deltas": skill_deltas}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _risk_penalty(self, skill_id: str, weight: float, row: dict) -> float:
        """Compute the risk penalty for a high-risk skill with negative credit."""
        if weight != -1.0:
            return 0.0
        try:
            skill = self.store.read_skill(skill_id)
        except Exception:
            return 0.0
        if getattr(skill, "risk_level", "low") != "high":
            return 0.0
        ratio = float(self.evo_config.utility.risk_penalty_high_negative_ratio)
        branch = row.get("branch", "general")
        if branch == "general":
            current_utility = float(
                row.get("global_utility", self.evo_config.encode.draft_initial_utility)
            )
        else:
            current_utility = float(
                row.get("utility_score", self.evo_config.encode.draft_initial_utility)
            )
        return self._clip(current_utility * ratio)

    def _reinforced_tau_windows(self, row: dict) -> float:
        """Compute the reinforced memory tau (in windows) after a success event."""
        base = float(self.evo_config.utility.memory_tau_base_windows)
        success_count = int(row.get("success_when_adopted", 0))
        tau = base * (1.0 + sqrt(max(success_count, 0)))
        return min(tau, float(self.evo_config.utility.memory_tau_max_windows))

    def _clip(self, value: float) -> float:
        return max(
            float(self.evo_config.utility.utility_clip_min),
            min(float(self.evo_config.utility.utility_clip_max), value),
        )
