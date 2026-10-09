"""Reasoning and verification tools.

Two meta-cognitive tools that operate on the agent's own reasoning process:

  1. ReflectionTool  — Step-level self-audit that interrupts the generation
                       flow and forces the agent to switch from "solver" to
                       "critic" role, breaking confirmation bias.  Reuses the
                       agent's main LLM by default (no extra config needed).
  2. VerifierTool    — Role-asymmetric independent audit of the agent's
                       proposed final answer.  Exploits the cognitive gap
                       between "generating" and "judging": even the same model
                       can catch its own errors when evaluating from a fresh,
                       uninvolved perspective.

Design notes
------------
- Prompts are concise but not skeletal: every sentence earns its place by
  guiding model behaviour in a specific, testable way.
- Output formats are compact structured text, not verbose essays.  This keeps
  tool observation tokens low so the agent can iterate quickly.
- ReflectionTool deliberately asks the model to argue *against* the current
  hypothesis (Devil's Advocate), not merely "check for errors".
- VerifierTool emphasises role separation: "you had no part in generating
  this answer" is the single most important sentence in its prompt.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import DEFAULT_API_BASE, ToolParam, ToolResult, call_tool_llm

logger = logging.getLogger(__name__)

# Backward-compat alias used by medrag_search.py
_call_llm = call_tool_llm


# ===========================================================================
# ReflectionTool
# ===========================================================================
_REFLECTION_SYSTEM_PROMPT = """\
You are a Devil's Advocate reviewing the agent's reasoning mid-process.
Your job is to argue AGAINST the current hypothesis and find what is wrong \
or missing.

Check for:
- Logical gaps or unsupported jumps
- Overlooked differential diagnoses or contraindications
- Assumptions stated as facts without evidence
- Whether a tool call or more information is needed before concluding

Output in this format:
Verdict: SOUND | NEEDS_REVISION | UNCERTAIN
Issues: <semicolon-separated list, or "None">
Action: <one concrete next step>\
"""


@dataclass
class ReflectionTool:
    """Step-level self-audit that breaks confirmation bias.

    This tool forces the agent to interrupt its generation flow and switch
    from "solver" role to "critic" role.  The system prompt instructs the
    model to act as a Devil's Advocate — actively arguing against the
    current hypothesis rather than passively checking for errors.

    It can be called at ANY point during reasoning (not just at the end),
    making it effective for auditing intermediate hypotheses, partial plans,
    or individual diagnostic steps.

    Model priority:
      1. ``model`` non-empty → external LLM call.
      2. ``model`` empty (default) → reuse agent's main LLM via ``set_main_llm``.
      3. Neither available → lightweight local heuristic fallback.
    """

    name: str = "reflection"
    description: str = (
        "Self-audit your reasoning at any step. Switches you from solver to "
        "critic role to catch confirmation bias, logical gaps, or missed "
        "differentials. Returns verdict, issues found, and suggested next action."
    )
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model: str = ""
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_seconds: int = 60

    _main_llm: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.model and not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def set_main_llm(self, llm: Any) -> None:
        """Inject the agent's main LLM so reflection can reuse it.

        Called automatically by :class:`~medmem_agent.agent.AgentLoop` after
        the tool registry is built.
        """
        self._main_llm = llm

    # ----- core -----

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        reasoning = str(payload.get("reasoning", "")).strip()
        if not reasoning:
            raise ValueError("'reasoning' is required.")

        reflection_text = self._call(reasoning)

        verdict = self._extract(reflection_text, "Verdict", "UNCERTAIN")
        action = self._extract(reflection_text, "Action", "Proceed carefully.")

        return ToolResult(
            name=self.name,
            payload={
                "verdict": verdict,
                "suggested_action": action,
                "reflection": reflection_text,
            },
            render_fields=["reflection"],
        )

    def _call(self, reasoning: str) -> str:
        """Route the reflection call through the configured model chain."""
        if self.model and self.api_key:
            try:
                return call_tool_llm(
                    api_key=self.api_key,
                    api_base=self.api_base,
                    model=self.model,
                    system_prompt=_REFLECTION_SYSTEM_PROMPT,
                    user_prompt=reasoning,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    timeout=self.timeout_seconds,
                )
            except Exception as e:
                logger.warning(
                    f"External reflection model failed: {e}. "
                    "Falling back to main LLM."
                )

        if self._main_llm is not None:
            try:
                resp = self._main_llm.generate(
                    system_prompt=_REFLECTION_SYSTEM_PROMPT,
                    user_prompt=reasoning,
                )
                return resp.content
            except Exception as e:
                logger.warning(
                    f"Main LLM reflection failed: {e}. "
                    "Using local heuristic fallback."
                )

        logger.warning(
            "ReflectionTool: no LLM available. Using local heuristic. "
            "Call set_main_llm() or configure 'model' in tools config."
        )
        return self._local_heuristic(reasoning)

    # ----- helpers -----

    @staticmethod
    def _extract(text: str, field_name: str, default: str) -> str:
        """Extract a structured field value from reflection/verification text."""
        m = re.search(
            rf"(?:^|\n)\s*\*?\*?{re.escape(field_name)}\*?\*?\s*[:：]\s*(.+)",
            text,
            re.IGNORECASE,
        )
        return m.group(1).strip().rstrip(".") if m else default

    @staticmethod
    def _local_heuristic(reasoning: str) -> str:
        """Lightweight rule-based fallback when no LLM is available."""
        issues: List[str] = []
        low = reasoning.lower()

        if len(reasoning.split()) < 20:
            issues.append("Reasoning too brief for reliable conclusions")
        if "assume" in low or "probably" in low:
            issues.append("Contains unverified assumptions")
        if any(k in low for k in ("drug", "medication", "dose", "prescri")):
            if "contraindic" not in low and "side effect" not in low:
                issues.append("Drug reasoning without contraindication check")

        if issues:
            return (
                f"Verdict: NEEDS_REVISION\n"
                f"Issues: {'; '.join(issues)}\n"
                f"Action: Address the issues before answering."
            )
        return (
            "Verdict: SOUND\n"
            "Issues: None\n"
            "Action: Proceed to formulate the answer."
        )

    # ----- schema -----

    def get_params_schema(self) -> list:
        return [
            ToolParam(
                name="reasoning",
                type="string",
                required=True,
                description=(
                    "A concise summary of the clinical context and your reasoning so far. Can be the full chain "
                    "or just the current step you want to audit. Do NOT paste "
                    "raw tool outputs verbatim — summarise the key logic, "
                    "assumptions, and uncertainties."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"reasoning": "Patient has periorbital edema and proteinuria. '
            'I suspect nephrotic syndrome, but I have not ruled out cardiac '
            'or hepatic causes of edema."}'
        )


# ===========================================================================
# VerifierTool
# ===========================================================================
_VERIFIER_SYSTEM_PROMPT = """\
You are an independent medical answer auditor. You had NO part in generating \
the answer below — you are seeing it for the first time.

Your sole task is to check whether the answer is factually correct and \
clinically safe. Do not rewrite the answer or provide an alternative.

Evaluate:
1. Factual accuracy — are the medical claims correct?
2. Clinical safety — could acting on this answer cause patient harm?
3. Completeness — does it address the core question?

Output in this format:
Correctness: CORRECT | PARTIALLY_CORRECT | INCORRECT
Safety: SAFE | CAUTION | UNSAFE
Issues: <semicolon-separated list, or "None">
Suggestion: <one actionable improvement, or "None">\
"""


@dataclass
class VerifierTool:
    """Role-asymmetric independent audit of a proposed final answer.

    Exploits the cognitive asymmetry between generation and judgement:
    generating an answer requires juggling fluency, logic, and evidence
    simultaneously (high cognitive load), while auditing an answer is a
    focused checklist task (lower cognitive load).  Even the same model
    can reliably catch errors it made during generation when it evaluates
    from a fresh, uninvolved perspective.

    The system prompt's most critical sentence is: "You had NO part in
    generating the answer" — this primes the model into a detached
    evaluator role rather than a defensive author role.
    """

    name: str = "verifier"
    description: str = (
        "Independent audit of your final answer before delivery. "
        "An impartial judge checks factual correctness and clinical safety "
        "from a fresh perspective."
    )
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model: str = "gpt-4o"
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        question = str(payload.get("question", "")).strip()
        proposed_answer = str(payload.get("proposed_answer", "")).strip()
        if not question or not proposed_answer:
            raise ValueError(
                "'question' and 'proposed_answer' are both required."
            )

        if not self.api_key:
            return ToolResult(
                name=self.name,
                payload={
                    "correctness": "UNKNOWN",
                    "safety": "UNKNOWN",
                    "evaluation": "Verifier unavailable: no API key configured.",
                },
                render_fields=["evaluation"],
            )

        user_prompt = f"Question: {question}\nProposed Answer: {proposed_answer}"

        evaluation = call_tool_llm(
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.model,
            system_prompt=_VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout_seconds,
        )

        correctness = ReflectionTool._extract(evaluation, "Correctness", "UNKNOWN")
        safety = ReflectionTool._extract(evaluation, "Safety", "UNKNOWN")

        return ToolResult(
            name=self.name,
            payload={
                "correctness": correctness,
                "safety": safety,
                "evaluation": evaluation,
            },
            render_fields=["evaluation"],
        )

    def get_params_schema(self) -> list:
        return [
            ToolParam(
                name="question",
                type="string",
                required=True,
                description="The original question being answered.",
            ),
            ToolParam(
                name="proposed_answer",
                type="string",
                required=True,
                description=(
                    "Your ready-to-deliver final answer. Include the key "
                    "conclusion and critical details, but keep it focused."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"question": "First-line treatment for type 2 diabetes with CKD?", '
            '"proposed_answer": "Metformin with dose adjustment for eGFR 30-45; '
            'consider SGLT2 inhibitor for renoprotection."}'
        )
