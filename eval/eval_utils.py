"""Shared utilities for all evaluation scripts."""
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from medmem_agent.config import _resolve_dict

# ---------------------------------------------------------------------------
# Eval config
# ---------------------------------------------------------------------------

class EvalConfig:
    def __init__(self, path: str):
        with open(path, "r") as f:
            cfg = _resolve_dict(json.load(f))
        self.model: str = cfg.get("model", "gpt-4.1-mini")
        self.api_base: Optional[str] = cfg.get("api_base")
        self.api_key: Optional[str] = cfg.get("api_key")
        self.temperature: float = cfg.get("temperature", 0.0)
        self.max_tokens: int = cfg.get("max_tokens", 512)
        self.max_retries: int = cfg.get("max_retries", 3)
        self.retry_delay: float = cfg.get("retry_delay", 2.0)
        self.mc_mode: str = cfg.get("mc_mode", "exact")  # "exact" | "llm"

    @classmethod
    def default(cls) -> "EvalConfig":
        default_path = Path(__file__).parent.parent / "configs/eval/default.json"
        return cls(str(default_path))


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
_client: Optional[OpenAI] = None
_eval_config: Optional[EvalConfig] = None


def init_client(config: EvalConfig) -> None:
    global _client, _eval_config
    _eval_config = config
    kwargs: Dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["base_url"] = config.api_base
    _client = OpenAI(**kwargs)


def get_client() -> OpenAI:
    """Return the shared OpenAI client.

    If init_client() has not been called explicitly, auto-initialise from the
    canonical eval config file (configs/eval/default.json) so that callers
    never have to wire up the config themselves.  Falls back to a bare
    OpenAI() call (which reads OPENAI_API_KEY from the environment) only when
    the config file is absent or unreadable.
    """
    global _client
    if _client is None:
        _default_cfg = Path(__file__).resolve().parent.parent / "configs" / "eval" / "default.json"
        if _default_cfg.exists():
            try:
                init_client(EvalConfig(str(_default_cfg)))
            except Exception:
                pass  # fall through to bare OpenAI() below
        if _client is None:
            # Last-resort: rely on OPENAI_API_KEY env var.
            _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def call_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
) -> str:
    """Call the LLM. All params default to values from EvalConfig if init_client() was called."""
    # Ensure _eval_config is populated (auto-load from default.json if needed)
    if _eval_config is None:
        get_client()  # triggers auto-load of configs/eval/default.json
    cfg = _eval_config
    if cfg is not None:
        model = model or cfg.model
        temperature = temperature if temperature is not None else cfg.temperature
        max_tokens = max_tokens or cfg.max_tokens
        max_retries = max_retries or cfg.max_retries
        retry_delay = retry_delay or cfg.retry_delay
    else:
        model = model or "gpt-4.1-mini"
        temperature = temperature if temperature is not None else 0.0
        max_tokens = max_tokens or 512
        max_retries = max_retries or 3
        retry_delay = retry_delay or 2.0

    client = get_client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise exc
    return ""


# ---------------------------------------------------------------------------
# MC exact match
# ---------------------------------------------------------------------------
def mc_exact_match(prediction: str, ground_truth: str) -> bool:
    """Return True if the ground-truth option letter appears in the prediction."""
    gt = ground_truth.strip().upper()
    pred = prediction.strip().upper()
    # Match standalone letter: "A", "(A)", "A.", "A:" etc.
    return bool(re.search(rf"\b{re.escape(gt)}\b|[\(\[]{re.escape(gt)}[\)\]]", pred)) or gt in pred

# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Any:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Pretty summary printer
# ---------------------------------------------------------------------------
def print_summary(dataset: str, metrics: Dict[str, Any]) -> None:
    sep = "=" * 60
    print(sep)
    print(f"Dataset : {dataset}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
    print(sep)
