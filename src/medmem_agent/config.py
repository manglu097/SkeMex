from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_value(value: Any) -> Any:
    """Resolve a whole-string environment placeholder; keep literals unchanged."""
    if not isinstance(value, str):
        return value
    # Legacy: ${VAR} syntax
    if value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.getenv(env_var, "")
    # Legacy: $VAR syntax (whole string is a single env-var reference)
    if re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", value):
        env_var = value[1:]
        return os.getenv(env_var, "")
    # Plain value — return directly
    return value


def _resolve_dict(value: Any) -> Any:
    """Resolve placeholders recursively after JSON parsing (safe for quotes)."""
    if isinstance(value, dict):
        return {key: _resolve_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_dict(item) for item in value]
    return _resolve_value(value)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: int = 30
    headers: Optional[Dict[str, str]] = None
    supports_vision: bool = False
    extra_body: Optional[Dict[str, Any]] = None
    """Additional top-level fields to merge into the request payload.

    Use this to pass provider-specific parameters that are not part of the
    standard OpenAI chat-completions schema.  For example, Qwen3 models
    support ``enable_thinking`` to disable the chain-of-thought scratchpad::

        {"enable_thinking": false}

    These fields are merged **directly** into the JSON payload body, so they
    appear at the same level as ``model``, ``messages``, ``temperature``, etc.
    This mirrors the ``extra_body`` parameter of the official OpenAI Python SDK.
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=data["provider"],
            model=_resolve_value(data["model"]),
            api_base=_resolve_value(data.get("api_base")),
            api_key=_resolve_value(data.get("api_key")),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 512)),
            timeout_seconds=int(data.get("timeout_seconds", 30)),
            headers=_resolve_dict(data.get("headers")),
            supports_vision=bool(data.get("supports_vision", False)),
            extra_body=data.get("extra_body"),
        )

    @classmethod
    def from_file(cls, config_path: str | Path) -> "LLMConfig":
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class DatasetConfig:
    dataset_name: str
    tool_config_path: str
    system_prompt_override: Optional[str] = None
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> "DatasetConfig":
        tool_path = data["tool_config_path"]
        if base_dir and not Path(tool_path).is_absolute():
            tool_path = str((base_dir / tool_path).resolve())

        return cls(
            dataset_name=data["dataset_name"],
            tool_config_path=tool_path,
            system_prompt_override=data.get("system_prompt_override"),
            description=data.get("description", ""),
        )

    @classmethod
    def from_file(cls, config_path: str | Path) -> "DatasetConfig":
        path = Path(config_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, base_dir=path.parent)


@dataclass
class ToolsGlobalConfig:
    """Centralised hyperparameter store for all tools.

    Use environment placeholders for credentials in
    ``configs/global_config.json`` under the ``tools.shared`` block::

        {
          "tools": {
            "shared": {
              "openai_api_key": "${OPENAI_API_KEY}",
              "openai_api_base": "https://api.openai.com/v1/chat/completions",
              "tavily_api_key": "${TAVILY_API_KEY}",
              "umls_api_key":   "...",
              "ncbi_api_key":   "..."
            },
            "tavily_search": { "search_depth": "basic", ... },
            "analyze_medical_image": { "model": "gpt-4o", ... },
            ...
          }
        }

    Call :meth:`get_tool_kwargs` to obtain the merged kwargs dict for a
    specific tool type, ready to be passed to the tool constructor.
    """

    shared: Dict[str, Any] = field(default_factory=dict)
    tool_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Mapping: shared key name → constructor kwarg name for each tool type.
    # This allows the shared block to use canonical names while individual
    # tool classes may use different field names.
    _SHARED_KEY_MAP: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "tavily_search":          {"tavily_api_key": "api_key", "tavily_api_keys": "api_keys"},
        "medrag_search":          {"openai_api_key": "api_key", "openai_api_base": "api_base",
                                   "semantic_scholar_api_key": "semantic_scholar_api_key"},
        "drug_info_lookup":       {},
        "drug_interaction_check": {},
        "umls_concept_lookup":    {"umls_api_key": "umls_api_key"},
        "analyze_medical_image":  {"openai_api_key": "api_key", "openai_api_base": "api_base"},
        "ocr_chart_reader":       {"openai_api_key": "api_key", "openai_api_base": "api_base"},
        # reflection: shared credentials are NOT injected by default because
        # ReflectionTool reuses the main LLM when model is empty.  Only inject
        # them when the per-tool block explicitly sets a non-empty model,
        # which is handled by the per-tool override logic in get_tool_kwargs.
        "reflection":             {},
        "verifier":               {"openai_api_key": "api_key", "openai_api_base": "api_base"},
        "agentclinic_patient_reply": {"openai_api_key": "api_key", "openai_api_base": "api_base"},
        "agentclinic_request_exam":   {"openai_api_key": "api_key", "openai_api_base": "api_base"},
        "agentclinic_request_image": {},
        "mediq_patient_reply": {"openai_api_key": "api_key", "openai_api_base": "api_base"},
    })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolsGlobalConfig":
        """Parse the ``tools`` section of global_config.json.

        Values are read as plain strings.  Legacy ``$VAR`` / ``${VAR}``
        placeholders are still resolved for backward compatibility.
        """
        resolved = _resolve_dict(data)

        # Extract shared block, strip comment keys
        shared = {
            k: v for k, v in resolved.get("shared", {}).items()
            if not k.startswith("_")
        }

        # Extract per-tool param blocks
        tool_params: Dict[str, Dict[str, Any]] = {}
        for key, val in resolved.items():
            if key in ("shared",) or key.startswith("_"):
                continue
            if isinstance(val, dict):
                tool_params[key] = {k: v for k, v in val.items() if not k.startswith("_")}

        return cls(shared=shared, tool_params=tool_params)

    def get_tool_kwargs(self, tool_type: str) -> Dict[str, Any]:
        """Return merged constructor kwargs for *tool_type*.

        Merge priority (highest → lowest):
        1. Per-tool params from ``tools.<tool_type>`` section
           - ``api_base`` and ``api_key`` in the per-tool block override the
             shared ``openai_api_base`` / ``openai_api_key`` values, allowing
             each tool to point to a different endpoint (e.g. a locally
             deployed vLLM model such as hulu-med-32B).
        2. Shared credentials mapped via ``_SHARED_KEY_MAP``

        Empty-string values from the *shared* block are omitted so that the
        tool can fall back to its own ``os.environ`` lookup.  Empty-string
        values from the *per-tool* block are also omitted (they indicate
        "use the shared value").
        """
        kwargs: Dict[str, Any] = {}

        # Inject shared credentials via key mapping
        key_map = self._SHARED_KEY_MAP.get(tool_type, {})
        for shared_key, kwarg_name in key_map.items():
            value = self.shared.get(shared_key, "")
            if value:  # skip empty strings
                kwargs[kwarg_name] = value

        # Per-tool params override shared values.
        # api_base and api_key in the per-tool block take precedence over the
        # shared openai_api_base / openai_api_key values injected above.
        per_tool = self.tool_params.get(tool_type, {})
        for k, v in per_tool.items():
            if isinstance(v, str) and v == "":
                # Empty string in per-tool block means "use shared value" —
                # do not overwrite a non-empty shared value already in kwargs.
                continue
            kwargs[k] = v

        return kwargs

    @classmethod
    def empty(cls) -> "ToolsGlobalConfig":
        return cls()


@dataclass
class GlobalConfig:
    max_steps: int = 6
    output_dir: str = "output"
    history_filename: str = "interaction_history.jsonl"
    history_save_mode: str = "append"  # "append" or "overwrite"
    logging_level: str = "INFO"
    tools: ToolsGlobalConfig = field(default_factory=ToolsGlobalConfig.empty)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "GlobalConfig":
        path = Path(config_path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        tools_cfg = ToolsGlobalConfig.from_dict(data.get("tools", {}))
        return cls(
            max_steps=int(data.get("max_steps", 6)),
            output_dir=data.get("output_dir", "output"),
            history_filename=data.get("history_filename", "interaction_history.jsonl"),
            history_save_mode=data.get("history_save_mode", "append"),
            logging_level=data.get("logging_level", "INFO"),
            tools=tools_cfg,
        )


@dataclass
class AgentRuntimeConfig:
    llm: LLMConfig
    dataset: DatasetConfig
    global_config: GlobalConfig = field(default_factory=GlobalConfig)

    @classmethod
    def load(
        cls,
        *,
        llm_config_path: str | Path,
        dataset_config_path: str | Path,
        global_config_path: Optional[str | Path] = None,
    ) -> "AgentRuntimeConfig":
        global_cfg = GlobalConfig.from_file(global_config_path) if global_config_path else GlobalConfig()
        return cls(
            llm=LLMConfig.from_file(llm_config_path),
            dataset=DatasetConfig.from_file(dataset_config_path),
            global_config=global_cfg,
        )
