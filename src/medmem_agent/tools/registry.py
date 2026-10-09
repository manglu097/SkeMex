from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import Tool, ToolParam, ToolResult, DescribeSkillTool
from .tavily_search import TavilySearchTool
from .medrag_search import MedRAGSearchTool
from .medical_knowledge import (
    DrugInfoLookupTool,
    DrugInteractionCheckTool,
    UMLSConceptLookupTool,
)
from .medical_calculators import (
    PythonCalculatorTool,
    MedicalUnitConverterTool,
    ClinicalScoreCalculatorTool,
)
from .multimodal_tools import MedicalImageAnalysisTool, OCRChartReaderTool
from .reasoning_tools import ReflectionTool, VerifierTool
from .agentclinic_tools import (
    AgentClinicPatientReplyTool,
    AgentClinicRequestExamTool,
    AgentClinicRequestImageTool,
)
from .mediq_tools import MediQPatientReplyTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing and running tools for a specific dataset.

    Supports dynamic loading from a JSON configuration file.  All tool types
    are registered in ``_TOOL_TYPES``; new tools can be added at runtime via
    ``register_tool_type()``.

    Global hyperparameters (API keys, model names, URLs, timeouts) are
    centrally managed in ``configs/global_config.json`` under the ``tools``
    section.  When a :class:`~medmem_agent.config.ToolsGlobalConfig` is
    passed to :meth:`from_config`, those values are automatically merged into
    each tool's constructor kwargs (global tool config takes precedence).
    """

    # Map of tool type names (as used in JSON config) → implementation class
    _TOOL_TYPES: Dict[str, Type] = {
        # Category 1: Retrieval
        "tavily_search": TavilySearchTool,
        "medrag_search": MedRAGSearchTool,
        # Category 2: Medical knowledge bases
        "drug_info_lookup": DrugInfoLookupTool,
        "drug_interaction_check": DrugInteractionCheckTool,
        "umls_concept_lookup": UMLSConceptLookupTool,
        # Category 3: Calculators
        "python_calculator": PythonCalculatorTool,
        "unit_converter": MedicalUnitConverterTool,
        "clinical_score_calculator": ClinicalScoreCalculatorTool,
        # Category 4: Multimodal
        "analyze_medical_image": MedicalImageAnalysisTool,
        "ocr_chart_reader": OCRChartReaderTool,
        # Category 5: Reasoning / verification
        "reflection": ReflectionTool,
        "verifier": VerifierTool,
        # Category 6: AgentClinic benchmark tools
        "agentclinic_patient_reply": AgentClinicPatientReplyTool,
        "agentclinic_request_exam": AgentClinicRequestExamTool,
        "agentclinic_request_image": AgentClinicRequestImageTool,
        # Category 7: MediQ benchmark tools
        "mediq_patient_reply": MediQPatientReplyTool,
    }

    def __init__(self, *, dataset_name: str, tools: Dict[str, Any]) -> None:
        self.dataset_name = dataset_name
        self.tools = tools
        self.runtime_context: Dict[str, Any] = {}

        # Auto-inject describe_skill meta-tool
        describe_skill = DescribeSkillTool()
        describe_skill.set_registry(self)
        self.tools["describe_skill"] = describe_skill

    @classmethod
    def register_tool_type(cls, type_name: str, tool_class: Type) -> None:
        """Register a new tool type for dynamic loading."""
        cls._TOOL_TYPES[type_name] = tool_class

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        global_tools_config: Optional[Any] = None,  # ToolsGlobalConfig
    ) -> "ToolRegistry":
        """Load a ToolRegistry from a JSON configuration file.

        The JSON file must have the following structure::

            {
              "dataset_name": "medical",
              "tools": [
                {
                  "name": "tavily_search",
                  "type": "tavily_search"
                }
              ]
            }

        Per-tool hyperparameters (api_key, model, timeout, etc.) can be
        specified directly in the per-tool entry **or** centrally in
        ``configs/global_config.json`` under the ``tools`` section.

        Merge priority (highest → lowest):
        1. Global tool config (``global_tools_config.get_tool_kwargs(type)``)
        2. Per-tool entry fields in the dataset tools JSON

        Environment variable placeholders ``${VAR_NAME}`` and ``$VAR_NAME``
        in string values are automatically resolved at load time.

        Args:
            config_path: Path to the dataset-specific tools JSON file.
            global_tools_config: Optional :class:`ToolsGlobalConfig` loaded
                from ``global_config.json``.  When provided, shared
                credentials and default hyperparameters are injected
                automatically.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Tool config not found: {config_path}")

        from medmem_agent.config import _resolve_dict
        data = _resolve_dict(json.loads(path.read_text(encoding="utf-8")))

        dataset_name = data.get("dataset_name", "unknown")
        tools: Dict[str, Any] = {}

        for entry in data.get("tools", []):
            tool_type = entry.get("type")
            tool_name = entry.get("name")

            if not tool_type or not tool_name:
                logger.warning(f"Skipping invalid tool entry (missing type/name): {entry}")
                continue

            if tool_type not in cls._TOOL_TYPES:
                logger.error(
                    f"Unsupported tool type '{tool_type}' for tool '{tool_name}'. "
                    f"Available types: {sorted(cls._TOOL_TYPES.keys())}"
                )
                continue

            tool_class = cls._TOOL_TYPES[tool_type]
            try:
                # Start with per-tool entry fields from dataset config
                per_entry = {k: v for k, v in entry.items() if k not in ("type", "name")}
                kwargs: Dict[str, Any] = dict(per_entry)

                # Global config overrides dataset config
                if global_tools_config is not None:
                    kwargs.update(global_tools_config.get_tool_kwargs(tool_type))

                tools[tool_name] = tool_class(name=tool_name, **kwargs)
                logger.debug(f"Loaded tool '{tool_name}' (type: {tool_type})")
            except Exception as e:
                logger.error(
                    f"Failed to initialise tool '{tool_name}' of type '{tool_type}': {e}"
                )

        logger.info(
            f"ToolRegistry loaded for dataset '{dataset_name}': "
            f"{len(tools)} tool(s) — {sorted(tools.keys())}"
        )
        return cls(dataset_name=dataset_name, tools=tools)

    def set_runtime_context(self, context: Optional[Dict[str, Any]]) -> None:
        """Bind benchmark-specific runtime context to all tools that support it.

        This is intended for per-sample case data such as the current AgentClinic
        record. The agent model should only emit clinically meaningful arguments;
        sample-specific JSON objects are injected here by the benchmark runner.
        """
        self.runtime_context = dict(context or {})
        for tool in self.tools.values():
            if hasattr(tool, "set_runtime_context"):
                try:
                    tool.set_runtime_context(self.runtime_context)
                except Exception as e:
                    logger.warning(
                        f"Failed to set runtime context for tool '{getattr(tool, 'name', type(tool).__name__)}': {e}"
                    )

    def run(self, tool_name: str, payload: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given payload.

        Args:
            tool_name: Name of the tool to run (must be in ``self.tools``).
            payload: Input parameters for the tool.

        Returns:
            ToolResult from the tool execution.

        Raises:
            KeyError: If the tool name is not registered.
        """
        if tool_name not in self.tools:
            allowed = sorted(self.tools.keys())
            raise KeyError(
                f"Unknown tool '{tool_name}' for dataset '{self.dataset_name}'. "
                f"Available tools: {allowed}"
            )
        return self.tools[tool_name].run(payload)

    def render_tool_instructions(self) -> str:
        """Render a formatted list of all tools for injection into the system prompt.

        For each tool, outputs:
        1. Tool name and description.
        2. A structured parameter table (name | type | required | description).
        3. A concrete JSON usage example.

        This richer format significantly improves the model's ability to call
        tools with the correct parameter names and types.
        """
        if not self.tools:
            return "<none>"

        sections: List[str] = []
        for tool_name in sorted(self.tools):
            tool = self.tools[tool_name]
            section_lines: List[str] = []

            # --- Header: name + description ---
            section_lines.append(f"### {tool.name}")
            section_lines.append(tool.description)
            section_lines.append("")

            # --- Parameter table ---
            params: List[ToolParam] = []
            if hasattr(tool, "get_params_schema"):
                try:
                    params = tool.get_params_schema()
                except Exception:
                    params = []

            if params:
                section_lines.append("**Parameters:**")
                section_lines.append(
                    "| Parameter | Type | Required | Description |"
                )
                section_lines.append(
                    "| :--- | :--- | :---: | :--- |"
                )
                for p in params:
                    req_mark = "Yes" if p.required else "No"
                    default_note = (
                        f" (default: `{p.default}`)" if not p.required and p.default is not None
                        else ""
                    )
                    section_lines.append(
                        f"| `{p.name}` | {p.type} | {req_mark} | {p.description}{default_note} |"
                    )
                section_lines.append("")

            # --- Usage example ---
            usage = self._get_tool_usage(tool)
            if usage:
                section_lines.append(f"**Example call:** `{usage}`")

            sections.append("\n".join(section_lines))

        return "\n\n---\n\n".join(sections)

    def render_skill_list(self) -> str:
        """Render a compact skill list for token-efficient prompts.

        Each skill shows only: name + one-line description + minimal JSON example.
        Agent can call describe_skill to get full parameter specs on demand.
        """
        if not self.tools:
            return "<none>"

        categories = {
            "检索类": ["tavily_search", "medrag_search"],
            "医学知识库": ["drug_info_lookup", "drug_interaction_check", "umls_concept_lookup"],
            "计算类": ["python_calculator", "unit_converter", "clinical_score_calculator"],
            "多模态": ["analyze_medical_image", "ocr_chart_reader"],
            "推理增强": ["reflection", "verifier"],
            "AgentClinic": ["agentclinic_patient_reply", "agentclinic_request_exam", "agentclinic_request_image"],
            "MediQ": ["mediq_patient_reply"],
        }

        lines = []
        for category, tool_types in categories.items():
            category_tools = [name for name in tool_types if name in self.tools]
            if not category_tools:
                continue

            lines.append(f"**{category}**")
            for tool_name in category_tools:
                tool = self.tools[tool_name]
                usage = self._get_tool_usage(tool)
                lines.append(f"- `{tool_name}`: {tool.description}")
                if usage:
                    lines.append(f"  Example: `{usage}`")
            lines.append("")

        lines.append("**Meta-tool (free, does not count toward step limit):**")
        lines.append("- `describe_skill`: Get detailed parameter spec for any skill")
        lines.append('  Example: `{"skill_name": "tavily_search"}`')

        return "\n".join(lines)

    @staticmethod
    def _get_tool_usage(tool: Any) -> str:
        """Return a usage example string for a tool."""
        if hasattr(tool, "get_usage_example"):
            return tool.get_usage_example()
        return ""
