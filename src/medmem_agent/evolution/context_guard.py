from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from medmem_agent.config import LLMConfig
from medmem_agent.llm import build_llm

from .config import EvoConfig
from .prompts import build_key_finding_pin_prompt
from .types import Trajectory
from .utils import estimate_text_tokens


@dataclass
class ContextGuardState:
    loop_break_notice: str = ""
    loop_break_armed: bool = False
    pinned_findings: str = ""
    trim_applied: bool = False


@dataclass
class ContextGuard:
    evo_config: EvoConfig
    provider: str = "openai_compatible"
    api_base: str = "https://api.openai.com/v1"
    api_key: str | None = None
    state: ContextGuardState = field(default_factory=ContextGuardState)

    def before_step(self, steps_history: List[Dict], next_step_index: int) -> Dict[str, str]:
        self._update_loop_break(steps_history)
        if not self.state.pinned_findings and next_step_index >= self.evo_config.context_guard.key_finding_pin_step:
            self.state.pinned_findings = self._generate_pinned_findings(steps_history)
        return {
            "loop_break_notice": self.state.loop_break_notice,
            "pinned_findings": self.state.pinned_findings,
        }

    def after_step(self) -> None:
        if self.state.loop_break_armed:
            self.state.loop_break_notice = ""
            self.state.loop_break_armed = False

    def maybe_trim_history_for_prompt(self, steps_history: List[Dict]) -> Tuple[List[Dict], bool]:
        budget = self.evo_config.context_guard.token_budget
        raw_text = self._render_steps_for_budget(steps_history)
        if estimate_text_tokens(raw_text) <= int(budget * self.evo_config.context_guard.context_trim_ratio):
            self.state.trim_applied = False
            return steps_history, False

        keep_last = self.evo_config.context_guard.context_trim_keep_last_n_steps
        obs_limit = self.evo_config.context_guard.context_trim_observation_chars
        trimmed: List[Dict] = []
        split_idx = max(0, len(steps_history) - keep_last)
        for idx, step in enumerate(steps_history):
            new_step = dict(step)
            if idx < split_idx and new_step.get("observation"):
                obs = str(new_step["observation"])
                if len(obs) > obs_limit:
                    new_step["observation"] = obs[:obs_limit] + "...[truncated]"
            trimmed.append(new_step)
        self.state.trim_applied = True
        return trimmed, True

    def reset(self) -> None:
        self.state = ContextGuardState()

    def _update_loop_break(self, steps_history: List[Dict]) -> None:
        threshold = self.evo_config.context_guard.loop_break_repeat_threshold
        if len(steps_history) < threshold:
            return
        recent = steps_history[-threshold:]
        key0 = self._action_key(recent[0])
        if key0 is None:
            return
        if all(self._action_key(item) == key0 for item in recent):
            self.state.loop_break_notice = (
                "[Loop-Break Notice]\n"
                "You have repeated the same action multiple times without making progress. "
                "Do not repeat it again immediately. Either change strategy, use a different tool, "
                "or produce a final answer if enough evidence is already available."
            )
            self.state.loop_break_armed = True

    def _generate_pinned_findings(self, steps_history: List[Dict]) -> str:
        if not steps_history:
            return ""
        excerpt_lines: List[str] = []
        for idx, step in enumerate(steps_history[: self.evo_config.context_guard.key_finding_pin_step], 1):
            obs = str(step.get("observation", "")).strip()
            if obs:
                excerpt_lines.append(f"Step {idx} Observation: {obs}")
        if not excerpt_lines:
            return ""
        llm = build_llm(
            LLMConfig(
                provider=self.provider,
                model=self.evo_config.categories.classifier_model,
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=300,
                timeout_seconds=30,
            )
        )
        return llm.generate(
            system_prompt="You extract concise confirmed findings only.",
            user_prompt=build_key_finding_pin_prompt("\n".join(excerpt_lines)),
        ).content.strip()

    @staticmethod
    def _action_key(step: Dict) -> Tuple[str, str] | None:
        if not step.get("action"):
            return None
        return str(step.get("action")), str(step.get("action_input"))

    @staticmethod
    def _render_steps_for_budget(steps_history: List[Dict]) -> str:
        return "\n".join(
            f"step={item.get('step')} obs={item.get('observation','')} err={item.get('error','')} ans={item.get('answer','')}"
            for item in steps_history
        )
