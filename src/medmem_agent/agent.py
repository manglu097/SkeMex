"""
MedMem Agent — ReAct-style agent loop.

Tag schema (v2):
  <planning>  — Optional. High-level plan for multi-step problems.
  <reasoning> — Required every step. Step-level reasoning.
  <tool>      — Tool call block.
  <response>  — Final answer to the user.

Backward compatibility:
  The parser also recognises legacy tags <think>/<plan>/<thinking> as
  aliases for <reasoning>, and <answer> as an alias for <response>.
  This allows old interaction logs and prompts to keep working.
"""

from __future__ import annotations

import json
import logging
import re
import os
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

from .llm import LLMInterface
from .memory import ConversationMemory
from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    MEDICAL_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    FORCED_CONVERGENCE_BLOCK,
    _FEW_SHOT_EXAMPLES,
    _MEDICAL_FEW_SHOT_EXAMPLES,
    _COMMON_MISTAKES,
)
from .tools.registry import ToolRegistry
from .tools.reasoning_tools import ReflectionTool

try:
    from .evolution.context_guard import ContextGuard
except Exception:  # pragma: no cover - optional integration import
    ContextGuard = Any  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phrases that indicate the model already has enough information to answer.
# Used for "orphan reasoning" recovery (model outputs <reasoning> but forgets
# to append <response> or <tool>).
# ---------------------------------------------------------------------------
_SELF_SUFFICIENT_PHRASES = (
    "i have sufficient information",
    "i can answer",
    "i already know",
    "no tool",
    "does not require a tool",
    "don't need a tool",
    "do not need a tool",
    "without a tool",
    "from my knowledge",
    "from general knowledge",
    "from training",
    "i know the answer",
    "sufficient information to answer",
    "enough information to answer",
    "can answer directly",
    "answer directly",
    "based on my knowledge",
    "based on general knowledge",
    "no need to search",
    "no need for a tool",
    "no search needed",
    "must output <response>",
    "must output `<response>`",
    # Chinese
    "不需要工具",
    "不需要搜索",
    "已有足够信息",
    "可以直接回答",
    "根据已有知识",
    "无需搜索",
)

# Tags recognised as "reasoning" (canonical + legacy aliases)
_REASONING_TAGS = ("reasoning", "think", "thinking", "plan")

# Tags recognised as "final response" (canonical + legacy alias)
_RESPONSE_TAGS = ("response", "answer")


def _build_image_path_note(images: List[str]) -> str:
    """Return a text block listing available image paths.

    This note is always appended to ``user_input`` when images are present,
    regardless of whether the LLM supports vision natively.  For vision-capable
    models it serves as an explicit reference so the model can cite the images
    in its reasoning; for text-only models it is the sole mechanism by which
    the agent learns the paths needed to call image-analysis tools.
    """
    lines = ["", "Image files provided:"]
    for i, path in enumerate(images, 1):
        lines.append(f"  [{i}] {path}")
    return "\n".join(lines)


@dataclass
class AgentLoop:
    llm: LLMInterface
    tool_registry: ToolRegistry
    memory: ConversationMemory
    max_steps: int = 6
    system_prompt_override: Optional[str] = None
    output_dir: str = "output"
    history_filename: str = "interaction_history.jsonl"
    history_save_mode: str = "append"
    _steps_history: List[Dict[str, Any]] = field(default_factory=list)
    _evolution_runtime_context: Dict[str, Any] = field(default_factory=dict)
    _context_guard: Optional["ContextGuard"] = None

    def __post_init__(self) -> None:
        """Inject the main LLM into ReflectionTool instances in the registry.

        ReflectionTool performs self-reflection using the agent's own main LLM
        by default (when ``model`` is empty in the config).  We inject
        ``self.llm`` here so the tool can call it without needing its own
        API-key configuration.
        """
        for tool in self.tool_registry.tools.values():
            if isinstance(tool, ReflectionTool):
                tool.set_main_llm(self.llm)
                logger.debug(
                    f"Injected main LLM into ReflectionTool '{tool.name}'."
                )

    def set_evolution_runtime_context(self, context: Optional[Dict[str, Any]]) -> None:
        """Set per-run evolution context such as retrieved skills and categories."""
        self._evolution_runtime_context = dict(context or {})

    def set_context_guard(self, context_guard: Optional["ContextGuard"]) -> None:
        """Attach an optional context-guard controller for meta-memory behaviors."""
        self._context_guard = context_guard

    def run(
        self,
        user_input: str,
        tool_runtime_context: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Run the ReAct agent loop.

        Args:
            user_input: The user's question or task description.
            tool_runtime_context: Optional per-sample context forwarded to
                tools that implement ``set_runtime_context`` (e.g. AgentClinic
                case data).
            images: Optional list of absolute local image paths associated with
                this task.  Two things happen when images are provided:

                1. Their paths are appended to ``user_input`` as a numbered
                   list so the agent always knows which files are available,
                   regardless of whether the underlying LLM supports vision.

                2. On the **first** LLM call only, the image list is forwarded
                   to ``llm.generate``.  If the LLM is vision-capable
                   (``supports_vision=True`` in its config), it will receive
                   the images as base64-encoded content blocks; otherwise the
                   argument is ignored and only the text path note is used.
                   Subsequent steps receive no images to avoid redundant token
                   usage — the model's first-step response already captures the
                   visual information in the conversation history.
        """
        images = images or []
        if self._context_guard is not None:
            self._context_guard.reset()

        # Always append image path note to user_input when images are present.
        # This gives the agent a concrete reference it can copy into tool calls.
        effective_user_input = user_input
        if images:
            effective_user_input = user_input + _build_image_path_note(images)
        # NOTE: evolution context (skills, task_category) is NOT written into memory.
        # It is rendered fresh as {evolution_block} in every step's user prompt,
        # so it never duplicates inside the conversation history.

        self.memory.add("user", effective_user_input)
        self._steps_history = []
        self.tool_registry.set_runtime_context(tool_runtime_context)

        # ---- Build system prompt ----------------------------------------
        base_system_prompt = self.system_prompt_override or DEFAULT_SYSTEM_PROMPT
        skill_list = self.tool_registry.render_skill_list()
        # Choose the appropriate few-shot examples based on prompt type
        is_medical = (base_system_prompt is MEDICAL_SYSTEM_PROMPT
                      or "medical assistant" in base_system_prompt.lower()[:200])
        few_shot = _MEDICAL_FEW_SHOT_EXAMPLES if is_medical else _FEW_SHOT_EXAMPLES

        try:
            fmt_kwargs = {"skill_list": skill_list, "tool_list": skill_list}
            if "{few_shot_examples}" in base_system_prompt:
                fmt_kwargs["few_shot_examples"] = few_shot
            if "{common_mistakes}" in base_system_prompt:
                fmt_kwargs["common_mistakes"] = _COMMON_MISTAKES
            system_prompt = base_system_prompt.format(**fmt_kwargs)
        except KeyError as e:
            logger.warning(
                f"System prompt template has unexpected placeholder {e}. "
                "Falling back to raw substitution."
            )
            system_prompt = base_system_prompt.replace(
                "{skill_list}", skill_list
            ).replace("{few_shot_examples}", few_shot
            ).replace("{common_mistakes}", _COMMON_MISTAKES)

        # ---- Timing: record wall-clock start of the whole run -----------
        run_start = time.monotonic()

        # ---- Main loop --------------------------------------------------
        for step in range(self.max_steps):
            logger.info(f"Agent Step {step + 1}/{self.max_steps}")

            # Images are passed to llm.generate on the first step only.
            # From step 2 onward the model has already processed the images
            # (their content is reflected in the conversation history), so
            # re-sending them would waste tokens without adding information.
            images_for_this_step = images if step == 0 else []

            step_start = time.monotonic()

            try:
                continuation_text = ""
                if step > 0:
                    continuation_text = "\n\nPlease continue the conversation based on the history."

                rendered_history = self.memory.render() or "<empty>"
                if self._context_guard is not None:
                    meta_context = self._context_guard.before_step(self._steps_history, step + 1)
                    trimmed_steps, trimmed = self._context_guard.maybe_trim_history_for_prompt(self._steps_history)
                    if trimmed:
                        rendered_history = self._render_trimmed_history(effective_user_input, trimmed_steps)
                    # Append runtime meta-info (pinned findings, loop-break, context-trim notice)
                    # to the END of history, not to user_input, so it reads as system-level
                    # state information rather than part of the user's request.
                    runtime_suffix = self._render_meta_memory_block(meta_context, trimmed)
                    if runtime_suffix:
                        rendered_history = rendered_history + runtime_suffix

                # Render evolution block fresh every step (not stored in memory to avoid duplication).
                evolution_block = self._render_evolution_block()

                # On the last step, inject a forced-convergence block that
                # instructs the model to output <response> and not call any
                # further tools.  All other steps receive an empty string so
                # the template renders identically to before this change.
                is_last_step = (step == self.max_steps - 1)
                forced_convergence = FORCED_CONVERGENCE_BLOCK if is_last_step else ""
                if is_last_step:
                    logger.info(
                        f"Step {step + 1}/{self.max_steps}: last step — "
                        "injecting forced-convergence block into user prompt."
                    )

                llm_response = self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=USER_PROMPT_TEMPLATE.format(
                        history=rendered_history,
                        user_input=effective_user_input,
                        evolution_block=evolution_block,
                        forced_convergence=forced_convergence,
                        continuation=continuation_text,
                    ),
                    images=images_for_this_step,
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return f"Error: LLM failed to respond. {str(e)}"

            content = llm_response.content.strip()
            logger.debug(f"LLM Output:\n{content}")
            self.memory.add("assistant", content)

            # -- Extract optional <planning> block
            planning_content = self._extract_tag(content, "planning")
            if planning_content:
                logger.info(f"Agent Planning: {planning_content[:300]}")

            # -- Extract <reasoning> (+ legacy aliases)
            reasoning_content = self._extract_reasoning(content)
            if reasoning_content:
                logger.info(f"Agent Reasoning: {reasoning_content[:300]}")
            else:
                logger.warning("No <reasoning> tag found in LLM output.")

            step_record: Dict[str, Any] = {
                "step": step + 1,
                "planning": planning_content,
                "reasoning": reasoning_content,
                "llm_output": content,
            }

            # ==============================================================
            # Priority 1: <tool> call
            # ==============================================================
            tool_block = self._extract_tag(content, "tool")
            if tool_block is not None:
                try:
                    action, action_input = self._parse_action(tool_block)
                    logger.info(f"Executing tool: {action} | input: {action_input}")

                    tool_result = self.tool_registry.run(action, action_input)
                    observation = tool_result.render()

                    logger.info(f"Tool observation (first 300 chars): {observation[:300]}")
                    self.memory.add("observation", observation)

                    step_record.update({
                        "action": action,
                        "action_input": action_input,
                        "observation": observation,
                    })
                    step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                    self._steps_history.append(step_record)

                    # describe_skill is free — don't count toward max_steps
                    if action == "describe_skill":
                        logger.info("describe_skill called (free, not counted toward max_steps)")
                        step -= 1  # Rewind step counter

                    continue

                except json.JSONDecodeError as e:
                    error_msg = (
                        f"Tool call parse error: Action Input is not valid JSON. "
                        f"Details: {e}. Please fix the JSON and try again."
                    )
                    logger.warning(error_msg)
                    self.memory.add("observation", error_msg)
                    step_record["error"] = error_msg
                    step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                    self._steps_history.append(step_record)
                    continue

                except KeyError as e:
                    error_msg = (
                        f"Unknown tool: {e}. "
                        "Please use only the tools listed in Available Tools."
                    )
                    logger.warning(error_msg)
                    self.memory.add("observation", error_msg)
                    step_record["error"] = error_msg
                    step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                    self._steps_history.append(step_record)
                    continue

                except Exception as e:
                    error_msg = f"Tool execution error: {str(e)}"
                    logger.warning(error_msg)
                    self.memory.add("observation", error_msg)
                    step_record["error"] = error_msg
                    step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                    self._steps_history.append(step_record)
                    continue

            # ==============================================================
            # Priority 2: <response> (+ legacy <answer>)
            # ==============================================================
            final_answer = self._extract_response(content)
            if final_answer is not None:
                step_record["answer"] = final_answer
                step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                self._steps_history.append(step_record)
                total_duration = round(time.monotonic() - run_start, 3)
                self._save_history(effective_user_input, final_answer, total_duration)
                return final_answer

            # ==============================================================
            # Priority 3: Legacy "Final Answer:" prefix
            # ==============================================================
            if "Final Answer:" in content:
                final_answer = content.split("Final Answer:")[-1].strip()
                step_record["answer"] = final_answer
                step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                self._steps_history.append(step_record)
                total_duration = round(time.monotonic() - run_start, 3)
                self._save_history(effective_user_input, final_answer, total_duration)
                return final_answer

            # ==============================================================
            # Priority 4: Orphan reasoning recovery
            #
            # Model output <reasoning> but forgot to append <response>/<tool>.
            # Three sub-cases:
            #   (a) reasoning signals self-sufficiency → targeted completion
            #   (b) reasoning followed by trailing plain text → use as answer
            #   (c) no clear signal → nudge with format reminder
            # ==============================================================
            if reasoning_content:
                reasoning_lower = reasoning_content.lower()
                is_self_sufficient = any(
                    phrase in reasoning_lower for phrase in _SELF_SUFFICIENT_PHRASES
                )

                remainder = self._strip_all_structural_tags(content).strip()

                if is_self_sufficient:
                    logger.info(
                        f"Step {step + 1}: Orphan reasoning (self-sufficient). "
                        "Requesting completion."
                    )
                    completion_nudge = (
                        "Your previous response contained a <reasoning> block indicating "
                        "you already have sufficient information to answer, but you forgot "
                        "to output the <response> tag.\n\n"
                        "Please output ONLY the following (no other text):\n"
                        "<response>\n"
                        "[your complete answer here]\n"
                        "</response>"
                    )
                    self.memory.add("observation", completion_nudge)
                    step_record["error"] = "orphan_reasoning_self_sufficient"
                    step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                    self._steps_history.append(step_record)
                    continue

                elif remainder:
                    # Trailing plain text after reasoning — treat as answer
                    logger.info(
                        f"Step {step + 1}: Orphan reasoning with trailing text. "
                        "Using trailing text as response."
                    )
                    step_record["answer"] = remainder
                    step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                    self._steps_history.append(step_record)
                    total_duration = round(time.monotonic() - run_start, 3)
                    self._save_history(effective_user_input, remainder, total_duration)
                    return remainder

                else:
                    nudge = (
                        "Your response only contained a <reasoning> block with no "
                        "<response> or <tool> after it.\n"
                        "You MUST follow <reasoning> with either:\n"
                        "  <response>[your answer]</response>\n"
                        "OR\n"
                        "  <tool>\n  Action: <tool_name>\n  Action Input: <json>\n  </tool>\n"
                        "Do not output any text outside these tags."
                    )
                    logger.warning(f"Step {step + 1}: Orphan reasoning (no signal). Nudging.")
                    self.memory.add("observation", nudge)
                    step_record["error"] = "orphan_reasoning_no_signal"
                    step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                    self._steps_history.append(step_record)
                    continue

            # ==============================================================
            # Priority 5: Last step fallback
            # ==============================================================
            if step == self.max_steps - 1:
                final_answer = self._strip_all_structural_tags(content).strip() or content
                step_record["answer"] = final_answer
                step_record["duration_s"] = round(time.monotonic() - step_start, 3)
                self._steps_history.append(step_record)
                total_duration = round(time.monotonic() - run_start, 3)
                self._save_history(effective_user_input, final_answer, total_duration)
                return final_answer

            # ==============================================================
            # Priority 6: Completely unrecognised output — nudge
            # ==============================================================
            nudge = (
                "Error: Your response did not contain a <reasoning> block. "
                "Every response MUST start with <reasoning>. "
                "Then follow with <response> or <tool>.\n"
                "Example:\n"
                "<reasoning>\n[your reasoning]\nAfter this reasoning block I must output <response>.\n</reasoning>\n"
                "<response>\n[your answer]\n</response>"
            )
            logger.warning(f"Step {step + 1}: No valid tag found. Nudging model.")
            self.memory.add("observation", nudge)
            step_record["error"] = "no_tag_found"
            step_record["duration_s"] = round(time.monotonic() - step_start, 3)
            self._steps_history.append(step_record)

        result = (
            "Agent stopped: max_steps reached without producing a final answer. "
            "Please try rephrasing your question or increasing max_steps."
        )
        total_duration = round(time.monotonic() - run_start, 3)
        self._save_history(effective_user_input, result, total_duration)
        return result

    def _render_evolution_block(self) -> str:
        """Render the evolution augmentation block for the current step.

        Returns an empty string when no evolution context is set (e.g. run_batch.py
        without a skill library), so USER_PROMPT_TEMPLATE degrades gracefully.
        The block is rendered fresh on every step but never written into memory,
        preventing duplication inside the conversation history.
        """
        context = self._evolution_runtime_context or {}
        sections: List[str] = []
        retrieved_skills_block = str(context.get("retrieved_skills_block", "")).strip()
        if retrieved_skills_block:
            sections.append(retrieved_skills_block)
        task_category = context.get("task_category") or []
        if task_category:
            sections.append("## Task Category\n" + ", ".join(str(item) for item in task_category))
        if not sections:
            return ""
        return "\n" + "\n\n".join(sections) + "\n"

    def _render_meta_memory_block(self, meta_context: Dict[str, str], trimmed: bool) -> str:
        sections: List[str] = []
        loop_break_notice = str(meta_context.get("loop_break_notice", "")).strip()
        pinned_findings = str(meta_context.get("pinned_findings", "")).strip()
        if loop_break_notice:
            sections.append(loop_break_notice)
        if pinned_findings:
            if pinned_findings.startswith("[Confirmed Findings]"):
                sections.append(pinned_findings)
            else:
                sections.append("[Confirmed Findings]\n" + pinned_findings)
        if trimmed:
            sections.append(
                "[Context Trim]\nEarlier observations were compressed to stay within the context budget. "
                "Rely on the pinned findings and the most recent steps for decision making."
            )
        return ("\n\n" + "\n\n".join(sections)) if sections else ""

    def _render_trimmed_history(self, effective_user_input: str, trimmed_steps: List[Dict[str, Any]]) -> str:
        lines: List[str] = [f"User: {effective_user_input}"]
        for step in trimmed_steps:
            reasoning = str(step.get("reasoning", "")).strip()
            action = step.get("action")
            action_input = step.get("action_input")
            observation = str(step.get("observation", "")).strip()
            error = str(step.get("error", "")).strip()
            answer = str(step.get("answer", "")).strip()
            if reasoning:
                lines.append(f"Assistant: <reasoning>{reasoning}</reasoning>")
            if action:
                lines.append(f"Assistant: <tool>Action: {action}\nAction Input: {json.dumps(action_input, ensure_ascii=False)}</tool>")
            if observation:
                lines.append(f"Observation: {observation}")
            if error:
                lines.append(f"Observation: {error}")
            if answer:
                lines.append(f"Assistant: <response>{answer}</response>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _save_history(self, user_input: str, final_answer: str, total_duration_s: float = 0.0) -> None:
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            if self.history_save_mode == "append":
                filename = self.history_filename
                mode = "a"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"run_{timestamp}.jsonl"
                mode = "w"

            filepath = os.path.join(self.output_dir, filename)

            record = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "steps": self._steps_history,
                "final_answer": final_answer,
                "total_duration_s": total_duration_s,
            }

            with open(filepath, mode, encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(
                f"Interaction history saved to {filepath} "
                f"(mode: {self.history_save_mode}, total_duration={total_duration_s}s)"
            )
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(content: str) -> Tuple[str, Dict[str, Any]]:
        """Parse Action and Action Input from a <tool> block."""
        action_match = re.search(r"Action:\s*(.+)", content)
        if not action_match:
            raise ValueError(
                f"Missing 'Action:' line in tool block. Content:\n{content}"
            )
        action = action_match.group(1).strip()

        input_match = re.search(
            r"Action Input:\s*(?:```(?:json)?\s*)?([\s\S]*?)(?:```|$)",
            content,
            re.IGNORECASE,
        )
        if not input_match:
            raise ValueError(
                f"Missing 'Action Input:' line in tool block. Content:\n{content}"
            )

        raw_payload = input_match.group(1).strip()

        try:
            return action, json.loads(raw_payload)
        except json.JSONDecodeError:
            collapsed = " ".join(raw_payload.split())
            return action, json.loads(collapsed)

    @staticmethod
    def _extract_tag(content: str, tag: str) -> Optional[str]:
        """Extract inner text of a <tag>...</tag> block.

        Algorithm: find the LAST opening <tag>, then match it with the FIRST
        </tag> that appears after it.  This correctly handles the common case
        where the tag name appears as plain text inside an earlier block, e.g.

            <reasoning>
            I must output <response>.   ← literal text, NOT a real open-tag
            </reasoning>
            <response>actual answer</response>

        Using the LAST opening tag ensures we always find the real structural
        block rather than a false match from embedded tag-name text.
        """
        escaped = re.escape(tag)
        open_pat = re.compile(f"<{escaped}>", re.IGNORECASE)
        close_pat = re.compile(f"</{escaped}>", re.IGNORECASE)

        opens = [m.end() for m in open_pat.finditer(content)]
        if not opens:
            return None

        # Use the last opening tag position
        start = opens[-1]
        close_match = close_pat.search(content, start)
        if not close_match:
            return None

        return content[start : close_match.start()].strip()

    @staticmethod
    def _extract_reasoning(content: str) -> Optional[str]:
        """Extract reasoning content from <reasoning> or legacy alias tags."""
        for tag in _REASONING_TAGS:
            result = AgentLoop._extract_tag(content, tag)
            if result:
                return result
        return None

    @staticmethod
    def _extract_response(content: str) -> Optional[str]:
        """Extract final answer from <response> or legacy <answer> tag."""
        for tag in _RESPONSE_TAGS:
            result = AgentLoop._extract_tag(content, tag)
            if result is not None:
                return result
        return None

    @staticmethod
    def _strip_all_structural_tags(content: str) -> str:
        """Remove all structural tag blocks from content, leaving plain text."""
        all_tags = list(_REASONING_TAGS) + list(_RESPONSE_TAGS) + ["planning", "tool"]
        for tag in all_tags:
            pattern = f"<{tag}>.*?</{tag}>"
            content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)
        return content.strip()

    # ---- Backward-compat aliases ----------------------------------------

    @staticmethod
    def _extract_think(content: str) -> Optional[str]:
        """Backward-compatible alias for _extract_reasoning."""
        return AgentLoop._extract_reasoning(content)

    @staticmethod
    def _strip_tag(content: str, tag: str) -> str:
        pattern = f"<{re.escape(tag)}>.*?</{re.escape(tag)}>"
        return re.sub(pattern, "", content, flags=re.DOTALL).strip()
