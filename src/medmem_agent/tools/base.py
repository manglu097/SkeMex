from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Default OpenAI-compatible API base URL.
# The OpenAI SDK appends ``/chat/completions`` automatically, so this should
# end with ``/v1`` — NOT ``/v1/chat/completions``.
DEFAULT_API_BASE = "https://api.openai.com/v1"


# ===========================================================================
# ToolParam — structured parameter specification
# ===========================================================================

@dataclass
class ToolParam:
    """Describes a single input parameter of a tool.

    Attributes:
        name:        Parameter name as it appears in the JSON payload.
        type:        Data type string shown to the model.
                     Use one of: "string", "number", "integer", "boolean",
                     "list[string]", "list[dict]", "one_of[...]".
        required:    Whether this parameter must be present in every call.
        description: Human-readable explanation of what the parameter does,
                     including valid values, constraints, and examples.
        default:     Default value shown to the model when required=False.
                     Pass None to omit from the rendered output.
    """

    name: str
    type: str
    required: bool
    description: str
    default: Optional[Any] = None


# ===========================================================================
# ToolResult — structured result returned by a tool
# ===========================================================================

@dataclass
class ToolResult:
    """Structured result returned by a tool execution.

    Attributes:
        name:         The name of the tool that produced this result.
        payload:      The full result data as a dictionary (used for logging/saving).
        render_fields: Optional list of payload keys to include when rendering for
                      the agent's conversation history.  When None (default) all
                      keys are rendered, preserving the original behaviour.
    """

    name: str
    payload: Dict[str, Any]
    render_fields: Optional[List[str]] = None

    def render(self) -> str:
        """Render the tool result as a human-readable string for the agent's memory.

        Only the keys listed in ``render_fields`` are included (when set).
        Top-level scalar fields are shown as ``key: value`` pairs; nested
        structures (lists, dicts) are JSON-serialized.
        """
        lines = [f"[Tool: {self.name}]"]
        items = (
            {k: self.payload[k] for k in self.render_fields if k in self.payload}.items()
            if self.render_fields is not None
            else self.payload.items()
        )
        for key, value in items:
            if isinstance(value, (dict, list)):
                lines.append(f"{key}:\n{json.dumps(value, ensure_ascii=False, indent=2)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


# ===========================================================================
# Tool Protocol
# ===========================================================================

@runtime_checkable
class Tool(Protocol):
    """Protocol that all tool implementations must satisfy."""

    name: str
    description: str

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        ...

    def get_params_schema(self) -> List[ToolParam]:
        """Return the structured parameter schema for this tool.

        Each element is a ToolParam describing one accepted input field.
        Tools that implement this method will have their parameters rendered
        in a structured table inside the system prompt, enabling the model
        to call the tool correctly.

        Returns an empty list by default; tools should override this.
        """
        return []


# ===========================================================================
# Shared LLM call helper
# ===========================================================================

def call_tool_llm(
    *,
    api_key: str,
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 1.0,
    timeout: int = 60,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint via the OpenAI SDK.

    This is the **single** LLM call helper for all tools that need to invoke
    a language model (reasoning, verification, patient simulation, RAG
    synthesis, etc.).  Using the SDK instead of raw ``urllib`` avoids HTTP-
    level geo-blocking (403 unsupported_country_region_territory).

    Args:
        api_key: Bearer token / API key.
        api_base: Base URL of the OpenAI-compatible endpoint (e.g.
            ``https://api.openai.com/v1``).  Trailing ``/chat/completions``
            is stripped automatically if present.
        model: Model identifier (e.g. ``"gpt-4o"``).
        system_prompt: System-role message content.
        user_prompt: User-role message content.
        max_tokens: Maximum tokens in the model response.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.

    Returns:
        The assistant's text response.

    Raises:
        RuntimeError: If the API call fails.
    """
    try:
        from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from exc

    # Normalise base_url: SDK expects the base URL without a path suffix.
    base_url = api_base.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(timeout),
        max_retries=2,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        try:
            return (response.choices[0].message.content or "").strip()
        except Exception:
            tb = traceback.format_exc()
            logger.error("call_tool_llm: failed to extract content from response.\n%s", tb)
            raise RuntimeError(
                f"call_tool_llm: failed to extract content from response.\n{tb}"
            )
    except APIStatusError as e:
        raise RuntimeError(
            f"LLM API error {e.status_code}: {e.message}"
        ) from e
    except APIConnectionError as e:
        raise RuntimeError(f"LLM API connection error: {e}") from e
    except APITimeoutError as e:
        raise RuntimeError(f"LLM API timeout after {timeout}s: {e}") from e


# ===========================================================================
# DescribeSkillTool — meta-tool for lazy-loading full tool specs
# ===========================================================================

class DescribeSkillTool:
    """Meta-tool that returns the full parameter specification for a skill.

    This tool is free (does not count toward max_steps) and allows the agent
    to fetch detailed tool specs on demand, enabling token-efficient prompts.
    """

    def __init__(self, name: str = "describe_skill"):
        self.name = name
        self.description = "Get detailed parameter specification for any skill"
        self._registry = None

    def set_registry(self, registry) -> None:
        """Bind the tool registry so we can look up tools."""
        self._registry = registry

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        skill_name = payload.get("skill_name", "").strip()

        if not skill_name:
            return ToolResult(
                name=self.name,
                payload={"error": "Missing required parameter: skill_name"}
            )

        if self._registry is None or skill_name not in self._registry.tools:
            available = sorted(self._registry.tools.keys()) if self._registry else []
            return ToolResult(
                name=self.name,
                payload={
                    "error": f"Unknown skill: {skill_name}",
                    "available_skills": available
                }
            )

        tool = self._registry.tools[skill_name]

        # Build detailed spec
        params = []
        if hasattr(tool, "get_params_schema"):
            try:
                params = tool.get_params_schema()
            except Exception:
                params = []

        param_lines = []
        if params:
            param_lines.append("| Parameter | Type | Required | Description |")
            param_lines.append("| :--- | :--- | :---: | :--- |")
            for p in params:
                req = "Yes" if p.required else "No"
                default = f" (default: `{p.default}`)" if not p.required and p.default is not None else ""
                param_lines.append(f"| `{p.name}` | {p.type} | {req} | {p.description}{default} |")

        usage = ""
        if hasattr(tool, "get_usage_example"):
            usage = tool.get_usage_example()

        spec = f"**{skill_name}**\n{tool.description}\n\n"
        if param_lines:
            spec += "\n".join(param_lines) + "\n\n"
        if usage:
            spec += f"**Example:** `{usage}`"

        return ToolResult(
            name=self.name,
            payload={"skill_name": skill_name, "specification": spec}
        )

    def get_params_schema(self) -> List[ToolParam]:
        return [
            ToolParam(
                name="skill_name",
                type="string",
                required=True,
                description="Name of the skill to describe"
            )
        ]

    def get_usage_example(self) -> str:
        return '{"skill_name": "tavily_search"}'
